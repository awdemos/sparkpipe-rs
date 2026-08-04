//! Prompt pipeline — port of `text/prompt_pipeline.c` (API declared in
//! `include/sparkpipe/spark_prompt_pipeline.h`) and `text/prompt.c` (API
//! declared in `model-families/glm52/include/sparkpipe/spark_glm52_text_prompt.h`).
//!
//! The pipeline drives the schedule → stage → dispatch loop: it pulls
//! dispatches from the request API, stages prefill token ids and KV block
//! tables into caller-provided host buffers, invokes the caller's
//! prefill/decode kernels, and completes or cancels each dispatch.
//! `submit_text_prompt` (port of `SparkGlm52RequestApiSubmitTextPrompt`)
//! tokenizes prompt text and submits it to the request API.
//!
//! Rust-port deviations from the C surface (behavior is otherwise
//! signature-faithful):
//!   - ABI plumbing (`abi_version`, `descriptor_bytes`, reserved words) is
//!     dropped; the C validation checks against those fields have no Rust
//!     counterpart.
//!   - Two boundaries the C expressed as concrete subsystem pointers are
//!     traits here, because those subsystems are ported in other crates:
//!       - [`RequestApi`] — the request/scheduler API
//!         (`include/sparkpipe/spark_request_api.h`; `api/request.c`). The
//!         pipeline only reads `accepted`/`kind` from a dispatch, so
//!         [`RequestDispatch`] is the minimal surface; the real request-API
//!         port will widen it behind the same trait.
//!       - [`TextPromptTokenizer`] — the tokenizer
//!         (`include/sparkpipe/spark_tokenizer.h`; `text/tokenizer.c`,
//!         ported in `src/tokenizer.rs` concurrently). The C's two encode
//!         variants (with/without `SparkTokenizerWorkspace`) collapse into
//!         one method: the workspace belongs to the tokenizer
//!         implementation, not the caller.
//!   - Host staging buffers are borrowed mutable slices whose lengths are
//!     the capacities; the C's null checks become length checks.
//!   - The C prefill/decode callbacks were function pointers + a `void *`
//!     context; [`run`] takes `FnMut` closures instead.
//!   - Where the C wrote partial results into out-params on error paths
//!     (run stats, submit counts), the Rust port still delivers run stats
//!     via the `&mut RunStats` out-param exactly like C, but
//!     [`submit_text_prompt`] returns `Err` without the partial counts.
//!   - `SparkPromptPipelineRun` returning `SPARK_STATUS_OK` after the
//!     stop-after-first-decode flag fires maps to `Ok(())`; every other C
//!     status maps to the matching [`PromptPipelineError`] variant.

/// Run flag: stop the loop after the first decode (or speculative-verify)
/// dispatch completes (SPARK_PROMPT_PIPELINE_RUN_FLAG_STOP_AFTER_FIRST_DECODE_DISPATCH).
pub const RUN_FLAG_STOP_AFTER_FIRST_DECODE_DISPATCH: u32 = 0x0000_0001;
/// All recognized run flags (SPARK_PROMPT_PIPELINE_RUN_KNOWN_FLAGS).
pub const RUN_KNOWN_FLAGS: u32 = RUN_FLAG_STOP_AFTER_FIRST_DECODE_DISPATCH;
/// Default dispatch-step bound when `max_dispatch_steps == 0`
/// (SPARK_PROMPT_PIPELINE_DEFAULT_MAX_DISPATCH_STEPS).
pub const DEFAULT_MAX_DISPATCH_STEPS: u32 = 4096;

/// Invalid request handle sentinel (SPARK_REQUEST_API_INVALID_HANDLE).
pub const INVALID_REQUEST_HANDLE: u64 = 0;

// Tokenizer encode flags (include/sparkpipe/spark_tokenizer.h), re-declared
// here so `submit_text_prompt` can validate `tokenizer_encode_flags` against
// the known mask without depending on the in-flight tokenizer port.
/// SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH.
pub const TOKENIZER_ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH: u32 = 0x0000_0001;
/// SPARK_TOKENIZER_ENCODE_FLAG_ADD_PREFIX_SPACE.
pub const TOKENIZER_ENCODE_FLAG_ADD_PREFIX_SPACE: u32 = 0x0000_0002;
/// SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_REGEX_PRETOKENIZATION.
pub const TOKENIZER_ENCODE_FLAG_DISABLE_REGEX_PRETOKENIZATION: u32 = 0x0000_0004;
/// SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_PIECE_CACHE.
pub const TOKENIZER_ENCODE_FLAG_DISABLE_PIECE_CACHE: u32 = 0x0000_0008;
/// SPARK_TOKENIZER_ENCODE_KNOWN_FLAGS.
pub const TOKENIZER_ENCODE_KNOWN_FLAGS: u32 = TOKENIZER_ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH
    | TOKENIZER_ENCODE_FLAG_ADD_PREFIX_SPACE
    | TOKENIZER_ENCODE_FLAG_DISABLE_REGEX_PRETOKENIZATION
    | TOKENIZER_ENCODE_FLAG_DISABLE_PIECE_CACHE;

/// Errors mirroring the `SparkStatus` codes the C entry points (and the
/// request-API calls they funnel through) return.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PromptPipelineError {
    /// SPARK_STATUS_INVALID_ARGUMENT.
    #[error("invalid argument")]
    InvalidArgument,
    /// SPARK_STATUS_CAPACITY_EXCEEDED.
    #[error("capacity exceeded")]
    CapacityExceeded,
    /// SPARK_STATUS_NOT_FOUND.
    #[error("not found")]
    NotFound,
    /// SPARK_STATUS_IO_ERROR.
    #[error("io error")]
    IoError,
    /// SPARK_STATUS_PARSE_ERROR.
    #[error("parse error")]
    ParseError,
    /// SPARK_STATUS_BUSY.
    #[error("busy")]
    Busy,
    /// SPARK_STATUS_INTERNAL_ERROR.
    #[error("internal error")]
    InternalError,
    /// SPARK_STATUS_PENDING.
    #[error("pending")]
    Pending,
    /// SPARK_STATUS_UNSUPPORTED.
    #[error("unsupported")]
    Unsupported,
}

/// Dispatch kind (SPARK_REQUEST_API_DISPATCH_KIND_*).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DispatchKind {
    /// SPARK_REQUEST_API_DISPATCH_KIND_NONE.
    #[default]
    None,
    /// SPARK_REQUEST_API_DISPATCH_KIND_PREFILL.
    Prefill,
    /// SPARK_REQUEST_API_DISPATCH_KIND_DECODE_BATCH.
    DecodeBatch,
    /// SPARK_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH.
    PrefillBatch,
    /// SPARK_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH.
    SpeculativeVerifyBatch,
}

/// Scheduled dispatch (minimal port of `SparkRequestApiDispatch`).
///
/// The C dispatch struct is a large scheduler-decision carrier; the prompt
/// pipeline itself only reads `accepted` and `kind`, so this is the surface
/// the [`RequestApi`] boundary exchanges. The real request-API port widens
/// it (scheduler decisions, speculative arrays, ...) behind the same trait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestDispatch {
    /// Non-zero when the scheduler accepted the dispatch (`accepted` in C).
    pub accepted: bool,
    /// Dispatch kind (`kind` in C).
    pub kind: DispatchKind,
}

/// Prefill dispatch view (port of `SparkRequestApiPrefillDispatchView`,
/// minus ABI words and the lane array — lane views belong to the
/// request-API port).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillDispatchView {
    /// Dispatch kind the view was built from (`kind` in C).
    pub kind: DispatchKind,
    /// Active sequences in this dispatch.
    pub active_sequence_count: u32,
    /// First prompt token covered by this step.
    pub prompt_token_offset: u32,
    /// Prompt tokens covered by this step.
    pub prompt_token_count: u32,
    /// Per-lane prompt token stride used by the dispatch.
    pub prompt_token_stride: u32,
    /// Lanes in this dispatch.
    pub lane_count: u32,
}

/// KV block table view handed to the prefill callback
/// (port of `SparkKvBlockTableView`).
#[derive(Debug, Clone, Copy)]
pub struct KvBlockTableView<'a> {
    /// Tokens per KV block.
    pub block_token_count: u32,
    /// Lanes covered by the table.
    pub lane_count: u32,
    /// Per-lane stride of the block-index arrays.
    pub lane_stride: u32,
    /// Per-lane capacity of the block-index arrays.
    pub lane_capacity: u32,
    /// Execution-side block indices (`physical_block_indices` in C; absent
    /// when the configuration carries no execution-side array).
    pub physical_block_indices: Option<&'a [u32]>,
    /// Execution-side per-lane block counts (`lane_physical_block_counts` in
    /// C; shares the host counts in this port's boundary).
    pub lane_physical_block_counts: Option<&'a [u32]>,
    /// Host-side block indices.
    pub host_physical_block_indices: &'a [u32],
    /// Host-side per-lane block counts.
    pub host_lane_physical_block_counts: &'a [u32],
}

/// Prefill dispatch handed to the prefill callback
/// (port of `SparkPromptPipelinePrefillDispatch`).
#[derive(Debug, Clone, Copy)]
pub struct PrefillDispatch<'a> {
    /// Loop step index that produced this dispatch.
    pub step_index: u32,
    /// Dispatch kind ([`DispatchKind::Prefill`] or
    /// [`DispatchKind::PrefillBatch`]).
    pub dispatch_kind: DispatchKind,
    /// Active sequences in this dispatch.
    pub active_sequence_count: u32,
    /// Lanes in this dispatch.
    pub lane_count: u32,
    /// First prompt token covered by this step.
    pub prompt_token_offset: u32,
    /// Prompt tokens covered by this step.
    pub prompt_token_count: u32,
    /// Per-lane prompt token stride used by the dispatch.
    pub prompt_token_stride: u32,
    /// Per-lane stride of `host_token_ids`.
    pub host_token_stride: u32,
    /// Staged prompt token ids (`lane_capacity * host_token_stride` words;
    /// per-lane padding is zero-filled by the request API, as in C).
    pub host_token_ids: &'a [u32],
    /// KV block table for this dispatch.
    pub kv_block_table_view: &'a KvBlockTableView<'a>,
}

/// Submit request passed to [`RequestApi::submit`]
/// (port of `SparkRequestApiSubmitRequest`).
#[derive(Debug, Clone, Copy)]
pub struct SubmitRequest<'a> {
    /// SPARK_REQUEST_API_REQUEST_FLAG_* bits.
    pub flags: u32,
    /// Scheduling priority.
    pub priority: u32,
    /// Prompt tokens (required count, including any encoding overflow).
    pub prompt_token_count: u32,
    /// Thinking-phase token budget.
    pub thinking_token_budget: u32,
    /// Output token budget.
    pub output_token_budget: u32,
    /// Maximum prefill tokens per step (0 = scheduler default).
    pub max_prefill_tokens_per_step: u32,
    /// Caller request id.
    pub request_id: u64,
    /// Caller sequence id.
    pub sequence_id: u64,
    /// Prompt token ids. `prompt_token_count` is the *required* count; on
    /// the encoding-overflow path it exceeds this slice's length (which
    /// holds only the tokens that fit), exactly as the C submits the
    /// required count alongside the caller's storage.
    pub prompt_token_ids: &'a [u32],
}

/// Request/scheduler API boundary (port of the `SparkRequestApi*` calls the
/// prompt pipeline makes). Implemented by the real request-API port and by
/// test fakes.
pub trait RequestApi {
    /// Port of `SparkRequestApiScheduleNext`.
    fn schedule_next(&mut self) -> Result<RequestDispatch, PromptPipelineError>;
    /// Port of `SparkRequestApiDescribePrefillDispatch`.
    fn describe_prefill_dispatch(
        &self,
        dispatch: &RequestDispatch,
    ) -> Result<PrefillDispatchView, PromptPipelineError>;
    /// Port of `SparkRequestApiCopyPrefillDispatchTokenIds`.
    ///
    /// `destination_token_ids` is the whole staging buffer
    /// (`destination_lane_capacity * destination_token_stride` words); the
    /// implementation writes each lane's tokens and zero-fills the per-lane
    /// padding, exactly as the C does.
    fn copy_prefill_dispatch_token_ids(
        &self,
        dispatch: &RequestDispatch,
        destination_token_ids: &mut [u32],
        destination_token_stride: u32,
        destination_lane_capacity: u32,
    ) -> Result<(), PromptPipelineError>;
    /// Port of `SparkRequestApiBuildDispatchKvBlockTableView`.
    ///
    /// The C also took `lane_count_capacity`; here the slice lengths bound
    /// the writable lanes instead.
    fn build_dispatch_kv_block_table_view<'a>(
        &self,
        dispatch: &RequestDispatch,
        host_physical_block_indices: &'a mut [u32],
        execution_physical_block_indices: Option<&'a [u32]>,
        lane_stride: u32,
        lane_capacity: u32,
        lane_physical_block_counts: &'a mut [u32],
    ) -> Result<KvBlockTableView<'a>, PromptPipelineError>;
    /// Port of `SparkRequestApiCompleteDispatch`.
    fn complete_dispatch(&mut self, dispatch: &RequestDispatch) -> Result<(), PromptPipelineError>;
    /// Port of `SparkRequestApiCancelDispatch`.
    fn cancel_dispatch(&mut self, dispatch: &RequestDispatch) -> Result<(), PromptPipelineError>;
    /// Port of `SparkRequestApiSubmit`. Returns the request handle.
    fn submit(&mut self, request: &SubmitRequest) -> Result<u64, PromptPipelineError>;
}

/// Pipeline configuration (port of `SparkPromptPipelineConfiguration`).
///
/// The C's function pointers and callback context are replaced by the
/// closure arguments of [`run`].
pub struct PromptPipelineConfiguration<'a> {
    /// SPARK_PROMPT_PIPELINE_RUN_FLAG_* bits.
    pub run_flags: u32,
    /// Dispatch-step bound; 0 selects [`DEFAULT_MAX_DISPATCH_STEPS`].
    pub max_dispatch_steps: u32,
    /// Prefill token staging (`host_prefill_token_stride *
    /// host_prefill_lane_capacity` words).
    pub host_prefill_token_ids: &'a mut [u32],
    /// Per-lane stride of `host_prefill_token_ids`.
    pub host_prefill_token_stride: u32,
    /// Lane capacity of `host_prefill_token_ids`.
    pub host_prefill_lane_capacity: u32,
    /// Host KV block-index staging (`kv_block_lane_stride *
    /// lane_count_capacity` words).
    pub host_physical_block_indices: &'a mut [u32],
    /// Optional execution-side block-index array mirrored into the KV block
    /// table view (`execution_physical_block_indices` in C).
    pub execution_physical_block_indices: Option<&'a [u32]>,
    /// Per-lane stride of the block-index arrays.
    pub kv_block_lane_stride: u32,
    /// Per-lane capacity of the block-index arrays (`<= kv_block_lane_stride`).
    pub kv_block_lane_capacity: u32,
    /// Per-lane block-count staging (`lane_count_capacity` words).
    pub lane_physical_block_counts: &'a mut [u32],
    /// Lane capacity of the block-count staging.
    pub lane_count_capacity: u32,
}

/// Run statistics (port of `SparkPromptPipelineRunStats`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunStats {
    /// Dispatches scheduled and completed.
    pub completed_dispatch_count: u32,
    /// Prefill dispatches executed.
    pub prefill_dispatch_count: u32,
    /// Decode dispatches executed.
    pub decode_dispatch_count: u32,
    /// Speculative-verify dispatches executed.
    pub speculative_verify_dispatch_count: u32,
    /// Total prefilled tokens across all steps.
    pub prefill_token_count: u32,
    /// Largest single-step prefill token count.
    pub maximum_prefill_token_count: u32,
    /// Largest single-step lane count.
    pub maximum_prefill_lane_count: u32,
    /// Kind of the last completed dispatch.
    pub last_dispatch_kind: DispatchKind,
    /// Set once a decode (or speculative-verify) dispatch completed.
    pub reached_decode_dispatch: bool,
}

/// Port of `SparkGlm52PromptPipelineValidateConfiguration`.
fn validate_configuration(
    configuration: &PromptPipelineConfiguration,
) -> Result<(), PromptPipelineError> {
    let token_words = (configuration.host_prefill_token_stride as usize)
        .checked_mul(configuration.host_prefill_lane_capacity as usize);
    let block_words = (configuration.kv_block_lane_stride as usize)
        .checked_mul(configuration.lane_count_capacity as usize);
    if (configuration.run_flags & !RUN_KNOWN_FLAGS) != 0
        || configuration.host_prefill_token_stride == 0
        || configuration.host_prefill_lane_capacity == 0
        || configuration.kv_block_lane_stride == 0
        || configuration.kv_block_lane_capacity == 0
        || configuration.kv_block_lane_stride < configuration.kv_block_lane_capacity
        || configuration.lane_count_capacity == 0
        || token_words.is_none_or(|words| configuration.host_prefill_token_ids.len() < words)
        || block_words.is_none_or(|words| configuration.host_physical_block_indices.len() < words)
        || configuration.lane_physical_block_counts.len()
            < configuration.lane_count_capacity as usize
    {
        return Err(PromptPipelineError::InvalidArgument);
    }
    Ok(())
}

/// Port of `SparkGlm52PromptPipelineInvokePrefill`.
fn invoke_prefill<A: RequestApi>(
    api: &A,
    configuration: &mut PromptPipelineConfiguration,
    dispatch: &RequestDispatch,
    step_index: u32,
    stats: &mut RunStats,
    prefill_function: &mut dyn FnMut(&PrefillDispatch) -> Result<(), PromptPipelineError>,
) -> Result<(), PromptPipelineError> {
    let prefill_view = api.describe_prefill_dispatch(dispatch)?;
    if prefill_view.lane_count > configuration.host_prefill_lane_capacity
        || prefill_view.lane_count > configuration.lane_count_capacity
        || prefill_view.prompt_token_stride > configuration.host_prefill_token_stride
    {
        return Err(PromptPipelineError::CapacityExceeded);
    }

    let host_token_stride = configuration.host_prefill_token_stride;
    api.copy_prefill_dispatch_token_ids(
        dispatch,
        configuration.host_prefill_token_ids,
        host_token_stride,
        configuration.host_prefill_lane_capacity,
    )?;

    let block_table_view = api.build_dispatch_kv_block_table_view(
        dispatch,
        configuration.host_physical_block_indices,
        configuration.execution_physical_block_indices,
        configuration.kv_block_lane_stride,
        configuration.kv_block_lane_capacity,
        configuration.lane_physical_block_counts,
    )?;

    let prefill_dispatch = PrefillDispatch {
        step_index,
        dispatch_kind: dispatch.kind,
        active_sequence_count: prefill_view.active_sequence_count,
        lane_count: prefill_view.lane_count,
        prompt_token_offset: prefill_view.prompt_token_offset,
        prompt_token_count: prefill_view.prompt_token_count,
        prompt_token_stride: prefill_view.prompt_token_stride,
        host_token_stride,
        host_token_ids: configuration.host_prefill_token_ids,
        kv_block_table_view: &block_table_view,
    };

    prefill_function(&prefill_dispatch)?;

    stats.prefill_dispatch_count += 1;
    stats.prefill_token_count += prefill_view.prompt_token_count;
    stats.maximum_prefill_token_count =
        stats.maximum_prefill_token_count.max(prefill_view.prompt_token_count);
    stats.maximum_prefill_lane_count =
        stats.maximum_prefill_lane_count.max(prefill_view.lane_count);
    Ok(())
}

/// Port of `SparkGlm52PromptPipelineInvokeDecode`.
fn invoke_decode(
    dispatch: &RequestDispatch,
    stats: &mut RunStats,
    decode_function: &mut dyn FnMut(&RequestDispatch) -> Result<(), PromptPipelineError>,
) -> Result<(), PromptPipelineError> {
    decode_function(dispatch)?;

    if dispatch.kind == DispatchKind::SpeculativeVerifyBatch {
        stats.speculative_verify_dispatch_count += 1;
    } else {
        stats.decode_dispatch_count += 1;
    }
    stats.reached_decode_dispatch = true;
    Ok(())
}

/// Port of `SparkPromptPipelineRun`.
///
/// As in C, `stats` receives the accumulated statistics on every exit path
/// (success and error alike).
pub fn run<A: RequestApi>(
    api: &mut A,
    configuration: &mut PromptPipelineConfiguration,
    prefill_function: &mut dyn FnMut(&PrefillDispatch) -> Result<(), PromptPipelineError>,
    decode_function: &mut dyn FnMut(&RequestDispatch) -> Result<(), PromptPipelineError>,
    stats: &mut RunStats,
) -> Result<(), PromptPipelineError> {
    validate_configuration(configuration)?;

    let mut local_stats = RunStats::default();
    let max_dispatch_steps = if configuration.max_dispatch_steps != 0 {
        configuration.max_dispatch_steps
    } else {
        DEFAULT_MAX_DISPATCH_STEPS
    };

    for step_index in 0..max_dispatch_steps {
        let dispatch = match api.schedule_next() {
            Ok(dispatch) => dispatch,
            Err(status) => {
                *stats = local_stats;
                return Err(status);
            }
        };
        if !dispatch.accepted || dispatch.kind == DispatchKind::None {
            *stats = local_stats;
            return Err(PromptPipelineError::Busy);
        }

        local_stats.completed_dispatch_count += 1;
        local_stats.last_dispatch_kind = dispatch.kind;

        let status = match dispatch.kind {
            DispatchKind::Prefill | DispatchKind::PrefillBatch => invoke_prefill(
                api,
                configuration,
                &dispatch,
                step_index,
                &mut local_stats,
                prefill_function,
            ),
            DispatchKind::DecodeBatch | DispatchKind::SpeculativeVerifyBatch => {
                invoke_decode(&dispatch, &mut local_stats, decode_function)
            }
            DispatchKind::None => Err(PromptPipelineError::InvalidArgument),
        };

        let status = match status {
            Ok(()) => api.complete_dispatch(&dispatch),
            Err(status) => {
                let _ = api.cancel_dispatch(&dispatch);
                Err(status)
            }
        };
        if let Err(status) = status {
            *stats = local_stats;
            return Err(status);
        }

        if local_stats.reached_decode_dispatch
            && (configuration.run_flags & RUN_FLAG_STOP_AFTER_FIRST_DECODE_DISPATCH) != 0
        {
            *stats = local_stats;
            return Ok(());
        }
    }

    *stats = local_stats;
    Err(PromptPipelineError::CapacityExceeded)
}

// ---------------------------------------------------------------------------
// text/prompt.c — submit a text prompt (tokenize + request-API submit).
// ---------------------------------------------------------------------------

/// Tokenizer boundary for [`submit_text_prompt`].
///
/// Port of the `SparkTokenizerEncodeUtf8[WithWorkspace]` calls in
/// `text/prompt.c`. The C's optional caller-provided
/// `SparkTokenizerWorkspace` is an implementation concern here: a tokenizer
/// that needs one owns it. The in-flight `src/tokenizer.rs` port is
/// expected to implement this trait (or be wrapped by an adapter).
pub trait TextPromptTokenizer {
    /// Port of `SparkTokenizerEncodeUtf8`: encode `text` into `token_ids`
    /// (capacity = slice length), reporting how many tokens were written and
    /// how many would not fit (`overflow_token_count` in C).
    fn encode_utf8(
        &self,
        text: &str,
        flags: u32,
        token_ids: &mut [u32],
    ) -> Result<PromptEncoding, PromptPipelineError>;
}

/// Encoding result counts (port of the `SparkTokenizerEncoding` fields
/// `text/prompt.c` reads).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PromptEncoding {
    /// Tokens written into the destination buffer.
    pub token_count: u32,
    /// Tokens that did not fit in the destination buffer.
    pub overflow_token_count: u32,
}

/// Text-prompt submit request (port of `SparkGlm52TextPromptSubmitRequest`).
///
/// [`TextPromptSubmitRequest::default`] is the port of
/// `SparkGlm52TextPromptGetDefaultSubmitRequest` (all fields zero).
#[derive(Debug, Clone, Default)]
pub struct TextPromptSubmitRequest {
    /// SPARK_REQUEST_API_REQUEST_FLAG_* bits forwarded to the submit.
    pub request_flags: u32,
    /// Scheduling priority.
    pub priority: u32,
    /// Thinking-phase token budget.
    pub thinking_token_budget: u32,
    /// Output token budget.
    pub output_token_budget: u32,
    /// Maximum prefill tokens per step (0 = scheduler default).
    pub max_prefill_tokens_per_step: u32,
    /// Caller request id.
    pub request_id: u64,
    /// Caller sequence id.
    pub sequence_id: u64,
    /// Prompt text (the C's `prompt_text`/`prompt_text_bytes`; empty is
    /// accepted by validation but rejected after encoding, as in C).
    pub prompt_text: String,
    /// SPARK_TOKENIZER_ENCODE_FLAG_* bits.
    pub tokenizer_encode_flags: u32,
    /// Token capacity of the encode buffer
    /// (`prompt_token_storage_capacity` in C).
    pub prompt_token_capacity: u32,
}

/// Text-prompt submit result (port of `SparkGlm52TextPromptSubmitResult`).
#[derive(Debug, Clone)]
pub struct TextPromptSubmitResult {
    /// Tokens the encoder wrote (`prompt_token_count` in C).
    pub prompt_token_count: u32,
    /// Tokens the prompt actually needs (`required_prompt_token_count` in
    /// C: written + overflow).
    pub required_prompt_token_count: u32,
    /// Encoded prompt token ids (`prompt_token_count` entries). Owned here;
    /// the C wrote into caller-provided storage.
    pub prompt_token_ids: Vec<u32>,
    /// Submitted request handle.
    pub request_handle: u64,
}

/// Port of `SparkGlm52TextPromptValidateSubmitRequest`.
fn validate_submit_request(request: &TextPromptSubmitRequest) -> Result<(), PromptPipelineError> {
    if request.prompt_token_capacity == 0
        || (request.tokenizer_encode_flags & !TOKENIZER_ENCODE_KNOWN_FLAGS) != 0
    {
        return Err(PromptPipelineError::InvalidArgument);
    }
    Ok(())
}

/// Port of `SparkGlm52RequestApiSubmitTextPrompt`.
pub fn submit_text_prompt<A: RequestApi, T: TextPromptTokenizer>(
    api: &mut A,
    tokenizer: &T,
    request: &TextPromptSubmitRequest,
) -> Result<TextPromptSubmitResult, PromptPipelineError> {
    validate_submit_request(request)?;

    let mut token_ids = vec![0u32; request.prompt_token_capacity as usize];
    let encoding = tokenizer.encode_utf8(
        &request.prompt_text,
        request.tokenizer_encode_flags,
        &mut token_ids,
    )?;
    let prompt_token_count = encoding.token_count;
    // Mirrors the C's plain u32 addition of count + overflow.
    let required_prompt_token_count =
        encoding.token_count.wrapping_add(encoding.overflow_token_count);
    if required_prompt_token_count == 0 {
        return Err(PromptPipelineError::InvalidArgument);
    }
    token_ids.truncate(prompt_token_count as usize);

    let submit_request = SubmitRequest {
        flags: request.request_flags,
        priority: request.priority,
        prompt_token_count: required_prompt_token_count,
        thinking_token_budget: request.thinking_token_budget,
        output_token_budget: request.output_token_budget,
        max_prefill_tokens_per_step: request.max_prefill_tokens_per_step,
        request_id: request.request_id,
        sequence_id: request.sequence_id,
        prompt_token_ids: &token_ids,
    };

    // The C reset the out-handle to SPARK_REQUEST_API_INVALID_HANDLE on
    // submit failure; the Rust `Err` return carries the failure instead.
    let request_handle = api.submit(&submit_request)?;
    Ok(TextPromptSubmitResult {
        prompt_token_count,
        required_prompt_token_count,
        prompt_token_ids: token_ids,
        request_handle,
    })
}
