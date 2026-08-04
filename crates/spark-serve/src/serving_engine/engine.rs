//! Serving engine orchestration — port of `api/serving_engine.c`.
//!
//! The engine owns the request-record table, the event ring, and the host
//! staging buffers; per pump step it pulls a [`Dispatch`] from the
//! [`ServingRequestApi`] boundary, drives the caller's prefill/decode
//! callbacks, publishes events, and resolves completions.
//!
//! Rust-port deviations from the C surface (behavior otherwise faithful):
//! - Caller-provided buffer out-params (`request_records`, `event_ring`,
//!   staging arrays) become engine-owned `Vec`s sized from the
//!   configuration capacities.
//! - ABI/descriptor words and per-call revalidation of the engine pointer
//!   disappear; construction-time validation remains, bit for bit.
//! - The tokenizer workspace is created lazily on the first text submit
//!   instead of eagerly at engine init (`SparkServingEngineDestroy` is
//!   `Drop`).

use spark_text::prompt_pipeline::DispatchKind;
use spark_text::tokenizer::{Encoding, Tokenizer, Workspace};

use super::bridge::{
    ApiSubmitRequest, DecodeDispatchView, Dispatch, PrefillDispatchView, ServingRequestApi,
    CONFIGURATION_FLAG_DECODE_BATCHING, CONFIGURATION_FLAG_JIT_KV_PREFETCH,
    CONFIGURATION_FLAG_PREFILL_BATCHING, DISPATCH_FLAG_MTP_COMMIT,
    DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY, INVALID_REQUEST_HANDLE, MAX_DECODE_TOKENS_PER_LANE,
    MAX_DISPATCH_REQUEST_COUNT, MTP_MAX_DRAFT_TOKEN_COUNT,
};
use super::status::ServingStatus;

/// `SPARK_SERVING_ENGINE_ABI_VERSION`.
pub const SERVING_ENGINE_ABI_VERSION: u32 = 4;
/// `SPARK_SERVING_MAX_STOP_TOKEN_IDS`.
pub const MAX_STOP_TOKEN_IDS: usize = 8;
/// `SPARK_SERVING_DEFAULT_OUTPUT_TOKEN_BUDGET`.
pub const DEFAULT_OUTPUT_TOKEN_BUDGET: u32 = 1024;
/// `SPARK_SERVING_DEFAULT_MAX_CONTEXT_TOKENS`
/// (`SPARK_SCHEDULER_MAX_CONTEXT_TOKENS`, the GLM-5.2 context limit).
pub const DEFAULT_MAX_CONTEXT_TOKENS: u32 = 1_048_576;
/// `SPARK_SERVING_DEFAULT_MAX_PUMP_STEPS`.
pub const DEFAULT_MAX_PUMP_STEPS: u32 = 256;
/// `SPARK_SERVING_DEFAULT_REQUEST_ID_BASE`.
pub const DEFAULT_REQUEST_ID_BASE: u64 = 1_000_000_000;
/// `SPARK_SERVING_RECORD_HASH_SLOTS`.
pub const RECORD_HASH_SLOTS: usize = 4096;
/// `SPARK_SERVING_NO_RECORD_SLOT`.
pub const NO_RECORD_SLOT: u32 = u32::MAX;

/// `SPARK_SERVING_ENGINE_FLAG_REQUIRE_PRODUCTION_RUNTIME_CONTRACT`.
pub const ENGINE_FLAG_REQUIRE_PRODUCTION_RUNTIME_CONTRACT: u32 = 0x0000_0001;
/// `SPARK_SERVING_ENGINE_FLAG_AUTO_RELEASE_COMPLETED_REQUESTS`.
pub const ENGINE_FLAG_AUTO_RELEASE_COMPLETED_REQUESTS: u32 = 0x0000_0002;
/// `SPARK_SERVING_ENGINE_FLAG_CLAMP_BUDGET_TO_CONTEXT`.
pub const ENGINE_FLAG_CLAMP_BUDGET_TO_CONTEXT: u32 = 0x0000_0004;
/// `SPARK_SERVING_ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE`.
pub const ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE: u32 = 0x0000_0008;
/// `SPARK_SERVING_ENGINE_DEFAULT_FLAGS`.
pub const ENGINE_DEFAULT_FLAGS: u32 = ENGINE_FLAG_REQUIRE_PRODUCTION_RUNTIME_CONTRACT
    | ENGINE_FLAG_AUTO_RELEASE_COMPLETED_REQUESTS
    | ENGINE_FLAG_CLAMP_BUDGET_TO_CONTEXT;
/// `SPARK_SERVING_ENGINE_KNOWN_FLAGS`.
pub const ENGINE_KNOWN_FLAGS: u32 =
    ENGINE_DEFAULT_FLAGS | ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE;

/// `SPARK_SERVING_SUBMIT_FLAG_REALTIME`.
pub const SUBMIT_FLAG_REALTIME: u32 = super::bridge::REQUEST_FLAG_REALTIME;
/// `SPARK_SERVING_SUBMIT_FLAG_DISABLE_SPECULATION`.
pub const SUBMIT_FLAG_DISABLE_SPECULATION: u32 = super::bridge::REQUEST_FLAG_DISABLE_SPECULATION;
/// `SPARK_SERVING_SUBMIT_KNOWN_FLAGS`.
pub const SUBMIT_KNOWN_FLAGS: u32 = SUBMIT_FLAG_REALTIME | SUBMIT_FLAG_DISABLE_SPECULATION;

/// `SPARK_SERVING_RUNTIME_CONTRACT_FLAG_PREFILL_ACCEPTS_BULK_TOKEN_WINDOWS`.
pub const RUNTIME_CONTRACT_FLAG_PREFILL_ACCEPTS_BULK_TOKEN_WINDOWS: u32 = 0x0000_0001;
/// `SPARK_SERVING_RUNTIME_CONTRACT_FLAG_PREFILL_WRITES_RESIDENT_KV`.
pub const RUNTIME_CONTRACT_FLAG_PREFILL_WRITES_RESIDENT_KV: u32 = 0x0000_0002;
/// `SPARK_SERVING_RUNTIME_CONTRACT_FLAG_DECODE_CONSUMES_RESIDENT_KV`.
pub const RUNTIME_CONTRACT_FLAG_DECODE_CONSUMES_RESIDENT_KV: u32 = 0x0000_0004;
/// `SPARK_SERVING_RUNTIME_CONTRACT_FLAG_DECODE_RETURNS_TOKEN_IDS`.
pub const RUNTIME_CONTRACT_FLAG_DECODE_RETURNS_TOKEN_IDS: u32 = 0x0000_0008;
/// `SPARK_SERVING_RUNTIME_CONTRACT_FLAG_USES_REQUEST_KV_BLOCK_TABLES`.
pub const RUNTIME_CONTRACT_FLAG_USES_REQUEST_KV_BLOCK_TABLES: u32 = 0x0000_0010;
/// `SPARK_SERVING_RUNTIME_CONTRACT_FLAG_INTERNAL_BATCHING_ENABLED`.
pub const RUNTIME_CONTRACT_FLAG_INTERNAL_BATCHING_ENABLED: u32 = 0x0000_0020;
/// `SPARK_SERVING_RUNTIME_CONTRACT_FLAG_JIT_KV_PREFETCH_CONNECTED`.
pub const RUNTIME_CONTRACT_FLAG_JIT_KV_PREFETCH_CONNECTED: u32 = 0x0000_0040;
/// `SPARK_SERVING_RUNTIME_CONTRACT_FLAG_OVERLAPPED_STAGING_READY`.
pub const RUNTIME_CONTRACT_FLAG_OVERLAPPED_STAGING_READY: u32 = 0x0000_0080;
/// `SPARK_SERVING_RUNTIME_CONTRACT_FLAG_OFFICIAL_DSA_INDEXSHARE`.
pub const RUNTIME_CONTRACT_FLAG_OFFICIAL_DSA_INDEXSHARE: u32 = 0x0000_0100;
/// `SPARK_SERVING_RUNTIME_CONTRACT_FLAG_BOUNDED_LONG_CONTEXT_ATTENTION`.
pub const RUNTIME_CONTRACT_FLAG_BOUNDED_LONG_CONTEXT_ATTENTION: u32 = 0x0000_0200;
/// `SPARK_SERVING_RUNTIME_CONTRACT_FLAG_INDEXSHARE_STAGE_BOUNDARY_STATE`.
pub const RUNTIME_CONTRACT_FLAG_INDEXSHARE_STAGE_BOUNDARY_STATE: u32 = 0x0000_0400;
/// `SPARK_SERVING_RUNTIME_CONTRACT_FLAG_MLA_COMPRESSED_KV_CACHE`.
pub const RUNTIME_CONTRACT_FLAG_MLA_COMPRESSED_KV_CACHE: u32 = 0x0000_0800;
/// `SPARK_SERVING_RUNTIME_CONTRACT_FLAG_TAIL_WINDOW_VALIDATION_ONLY`.
pub const RUNTIME_CONTRACT_FLAG_TAIL_WINDOW_VALIDATION_ONLY: u32 = 0x8000_0000;
/// `SPARK_SERVING_RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS`.
pub const RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS: u32 =
    RUNTIME_CONTRACT_FLAG_PREFILL_ACCEPTS_BULK_TOKEN_WINDOWS
        | RUNTIME_CONTRACT_FLAG_PREFILL_WRITES_RESIDENT_KV
        | RUNTIME_CONTRACT_FLAG_DECODE_CONSUMES_RESIDENT_KV
        | RUNTIME_CONTRACT_FLAG_DECODE_RETURNS_TOKEN_IDS
        | RUNTIME_CONTRACT_FLAG_USES_REQUEST_KV_BLOCK_TABLES
        | RUNTIME_CONTRACT_FLAG_INTERNAL_BATCHING_ENABLED
        | RUNTIME_CONTRACT_FLAG_OFFICIAL_DSA_INDEXSHARE
        | RUNTIME_CONTRACT_FLAG_BOUNDED_LONG_CONTEXT_ATTENTION
        | RUNTIME_CONTRACT_FLAG_INDEXSHARE_STAGE_BOUNDARY_STATE
        | RUNTIME_CONTRACT_FLAG_MLA_COMPRESSED_KV_CACHE;
/// `SPARK_SERVING_RUNTIME_CONTRACT_CURRENT_IMPLEMENTED_FLAGS`.
pub const RUNTIME_CONTRACT_CURRENT_IMPLEMENTED_FLAGS: u32 =
    RUNTIME_CONTRACT_FLAG_PREFILL_WRITES_RESIDENT_KV
        | RUNTIME_CONTRACT_FLAG_DECODE_CONSUMES_RESIDENT_KV
        | RUNTIME_CONTRACT_FLAG_DECODE_RETURNS_TOKEN_IDS
        | RUNTIME_CONTRACT_FLAG_USES_REQUEST_KV_BLOCK_TABLES;

/// `SPARK_SERVING_DECODE_RESULT_FLAG_FINISH_REQUEST`.
pub const DECODE_RESULT_FLAG_FINISH_REQUEST: u32 = 0x0000_0001;
/// `SPARK_SERVING_DECODE_RESULT_FLAG_TOKEN_STREAM_SUPPRESSED`.
pub const DECODE_RESULT_FLAG_TOKEN_STREAM_SUPPRESSED: u32 = 0x0000_0002;

/// `SPARK_SERVING_PUMP_FLAG_STOP_AFTER_ONE_DISPATCH`.
pub const PUMP_FLAG_STOP_AFTER_ONE_DISPATCH: u32 = 0x0000_0001;
/// `SPARK_SERVING_PUMP_KNOWN_FLAGS`.
pub const PUMP_KNOWN_FLAGS: u32 = PUMP_FLAG_STOP_AFTER_ONE_DISPATCH;

/// Request handle (`SparkServingRequestHandle`).
pub type ServingRequestHandle = u64;

/// Request record state (`SPARK_SERVING_REQUEST_RECORD_STATE_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RequestRecordState {
    /// `SPARK_SERVING_REQUEST_RECORD_STATE_FREE`.
    #[default]
    Free,
    /// `SPARK_SERVING_REQUEST_RECORD_STATE_SUBMITTED`.
    Submitted,
    /// `SPARK_SERVING_REQUEST_RECORD_STATE_COMPLETED`.
    Completed,
    /// `SPARK_SERVING_REQUEST_RECORD_STATE_CANCELLED`.
    Cancelled,
}

/// Serving event kind (`SPARK_SERVING_EVENT_KIND_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ServingEventKind {
    /// `SPARK_SERVING_EVENT_KIND_NONE`.
    #[default]
    None,
    /// `SPARK_SERVING_EVENT_KIND_REQUEST_ACCEPTED`.
    RequestAccepted,
    /// `SPARK_SERVING_EVENT_KIND_PREFILL_PROGRESS`.
    PrefillProgress,
    /// `SPARK_SERVING_EVENT_KIND_TOKEN`.
    Token,
    /// `SPARK_SERVING_EVENT_KIND_REQUEST_COMPLETED`.
    RequestCompleted,
    /// `SPARK_SERVING_EVENT_KIND_REQUEST_CANCELLED`.
    RequestCancelled,
    /// `SPARK_SERVING_EVENT_KIND_ERROR`.
    Error,
    /// `SPARK_SERVING_EVENT_KIND_BACKPRESSURE`.
    Backpressure,
}

/// One request slot in the engine's record table
/// (`SparkServingRequestRecord`). Token storage is per-record: a fixed
/// `stride`-word row in static mode, a self-growing buffer in dynamic mode
/// (the C's caller arena / `realloc`, respectively).
#[derive(Debug, Clone, Default)]
struct RequestRecord {
    state: RequestRecordState,
    flags: u32,
    prompt_token_count: u32,
    token_capacity: u32,
    thinking_token_budget: u32,
    output_token_budget: u32,
    streamed_decode_token_count: u32,
    request_id: u64,
    sequence_id: u64,
    request_handle: ServingRequestHandle,
    handle_hash_next: u32,
    free_record_next: u32,
    token_ids: Vec<u32>,
}

impl RequestRecord {
    /// `SparkServingInitializeRequestRecord`.
    fn reset(&mut self, token_capacity: u32, keep_storage: bool) {
        let token_ids = if keep_storage { std::mem::take(&mut self.token_ids) } else { Vec::new() };
        *self = RequestRecord {
            state: RequestRecordState::Free,
            handle_hash_next: NO_RECORD_SLOT,
            free_record_next: NO_RECORD_SLOT,
            token_capacity,
            token_ids,
            ..RequestRecord::default()
        };
    }
}

/// Serving event (`SparkServingEvent`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ServingEvent {
    /// Event kind.
    pub kind: ServingEventKind,
    /// Event flags (currently unused upstream, carried as in C).
    pub flags: u32,
    /// Status attached to the event (terminal/error events).
    pub status: ServingStatus,
    /// Token id for `Token` events.
    pub token_id: u32,
    /// Absolute token index within the request's sequence.
    pub token_index: u32,
    /// Tokens in the decode step that produced this event.
    pub token_count: u32,
    /// Prefill progress: first prompt token covered.
    pub prompt_token_offset: u32,
    /// Prompt tokens on the request (or covered by a prefill step).
    pub prompt_token_count: u32,
    /// Dispatch kind that produced this event.
    pub dispatch_kind: DispatchKind,
    /// Dispatch flags that produced this event.
    pub dispatch_flags: u32,
    /// Caller request id.
    pub request_id: u64,
    /// Assigned sequence id.
    pub sequence_id: u64,
    /// Request handle.
    pub request_handle: ServingRequestHandle,
}

/// Submit-token-ids request (`SparkServingSubmitTokenIdsRequest`).
#[derive(Debug, Clone, Copy)]
pub struct SubmitTokenIdsRequest<'a> {
    /// `SPARK_SERVING_SUBMIT_FLAG_*` bits.
    pub flags: u32,
    /// Scheduling priority.
    pub priority: u32,
    /// Thinking-phase token budget (0 = engine default).
    pub thinking_token_budget: u32,
    /// Output token budget (0 = engine default).
    pub output_token_budget: u32,
    /// Maximum prefill tokens per step (0 = engine default).
    pub max_prefill_tokens_per_step: u32,
    /// Caller request id (0 = engine generates one).
    pub request_id: u64,
    /// Caller sequence id (0 = request API assigns one).
    pub sequence_id: u64,
    /// Prompt token ids (`token_count` in C is this slice's length).
    pub token_ids: &'a [u32],
}

/// Submit-text request (`SparkServingSubmitTextRequest`).
#[derive(Debug, Clone, Copy)]
pub struct SubmitTextRequest<'a> {
    /// `SPARK_SERVING_SUBMIT_FLAG_*` bits.
    pub flags: u32,
    /// Scheduling priority.
    pub priority: u32,
    /// Thinking-phase token budget (0 = engine default).
    pub thinking_token_budget: u32,
    /// Output token budget (0 = engine default).
    pub output_token_budget: u32,
    /// Maximum prefill tokens per step (0 = engine default).
    pub max_prefill_tokens_per_step: u32,
    /// `SPARK_TOKENIZER_ENCODE_FLAG_*` bits.
    pub tokenizer_encode_flags: u32,
    /// Caller request id (0 = engine generates one).
    pub request_id: u64,
    /// Caller sequence id (0 = request API assigns one).
    pub sequence_id: u64,
    /// UTF-8 prompt text (must be non-empty; the C's `text_bytes == 0`
    /// strlen path collapses to "the slice is the text").
    pub text: &'a [u8],
}

/// Submit result (`SparkServingSubmitResult`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SubmitResult {
    /// Prompt tokens accepted (or that fit, on encode overflow).
    pub prompt_token_count: u32,
    /// Token capacity the request requires.
    pub required_token_capacity: u32,
    /// Thinking budget after clamping.
    pub thinking_token_budget: u32,
    /// Output budget after clamping.
    pub output_token_budget: u32,
    /// Caller request id.
    pub request_id: u64,
    /// Assigned sequence id.
    pub sequence_id: u64,
    /// Request handle.
    pub request_handle: ServingRequestHandle,
}

/// Decode result (`SparkServingDecodeResult`). Fixed C arrays become `Vec`s
/// at full [`MAX_DISPATCH_REQUEST_COUNT`] capacity; indexing uses the
/// declared C strides (`MAX_DECODE_TOKENS_PER_LANE` /
/// `MTP_MAX_DRAFT_TOKEN_COUNT`), matching `token_ids[lane][token]`.
#[derive(Debug, Clone)]
pub struct ServingDecodeResult {
    /// Lanes the decode callback produced results for.
    pub lane_count: u32,
    /// Maximum tokens per lane the callback may report.
    pub token_stride: u32,
    /// Tokens produced per lane.
    pub token_counts: Vec<u32>,
    /// `SPARK_SERVING_DECODE_RESULT_FLAG_*` bits per lane.
    pub lane_flags: Vec<u32>,
    /// Produced token ids (flat, lane stride [`MAX_DECODE_TOKENS_PER_LANE`]).
    pub token_ids: Vec<u32>,
    /// MTP draft tokens produced per lane.
    pub draft_token_counts: Vec<u32>,
    /// MTP draft token ids (flat, lane stride [`MTP_MAX_DRAFT_TOKEN_COUNT`]).
    pub draft_token_ids: Vec<u32>,
}

impl ServingDecodeResult {
    /// `SparkServingInitializeDecodeResult`.
    pub fn new(lane_count: u32, token_stride: u32) -> Self {
        ServingDecodeResult {
            lane_count,
            token_stride,
            token_counts: vec![0; MAX_DISPATCH_REQUEST_COUNT],
            lane_flags: vec![0; MAX_DISPATCH_REQUEST_COUNT],
            token_ids: vec![0; MAX_DISPATCH_REQUEST_COUNT * MAX_DECODE_TOKENS_PER_LANE],
            draft_token_counts: vec![0; MAX_DISPATCH_REQUEST_COUNT],
            draft_token_ids: vec![0; MAX_DISPATCH_REQUEST_COUNT * MTP_MAX_DRAFT_TOKEN_COUNT],
        }
    }

    /// `token_ids[lane][token]` in C.
    pub fn lane_token_ids(&self, lane: usize) -> &[u32] {
        &self.token_ids[lane * MAX_DECODE_TOKENS_PER_LANE..(lane + 1) * MAX_DECODE_TOKENS_PER_LANE]
    }
}

/// Prefill dispatch handed to the prefill callback (port of
/// `SparkPromptPipelinePrefillDispatch` plus the per-lane view the serving
/// engine publishes progress events from).
pub struct ServingPrefillDispatch<'a> {
    /// Loop step index that produced this dispatch.
    pub step_index: u32,
    /// Dispatch kind (`Prefill` or `PrefillBatch`).
    pub dispatch_kind: DispatchKind,
    /// Active sequences in this dispatch.
    pub active_sequence_count: u32,
    /// Lanes in this dispatch.
    pub lane_count: u32,
    /// First prompt token covered by this step.
    pub prompt_token_offset: u32,
    /// Prompt tokens covered by this step (per lane).
    pub prompt_token_count: u32,
    /// Per-lane prompt token stride used by the dispatch.
    pub prompt_token_stride: u32,
    /// Per-lane stride of `host_token_ids`.
    pub host_token_stride: u32,
    /// Staged prompt token ids (per-lane padding zero-filled).
    pub host_token_ids: &'a [u32],
    /// Per-lane views (`lane_count` entries).
    pub lanes: &'a [super::bridge::PrefillDispatchLaneView],
    /// KV block table for this dispatch.
    pub kv_block_table_view: &'a super::bridge::KvBlockTableView<'a>,
}

/// Decode dispatch handed to the decode callback (port of
/// `SparkServingDecodeDispatch`).
pub struct ServingDecodeDispatch<'a> {
    /// Dispatch kind (`DecodeBatch` or `SpeculativeVerifyBatch`).
    pub dispatch_kind: DispatchKind,
    /// Lanes in this dispatch.
    pub request_count: u32,
    /// Active sequences covered by the KV block table.
    pub active_sequence_count: u32,
    /// Reserved in C (`speculative_token_index`); carried as zero.
    pub speculative_token_index: u32,
    /// The scheduled dispatch this decode serves.
    pub request_dispatch: &'a Dispatch,
    /// KV block table for this dispatch.
    pub kv_block_table_view: &'a super::bridge::KvBlockTableView<'a>,
    /// Per-lane decode view.
    pub decode_view: &'a DecodeDispatchView,
    /// Per-lane input (last committed) token ids.
    pub input_token_ids: &'a [u32],
    /// Draft tokens per lane for speculative-verify dispatches.
    pub speculative_token_count: u32,
    /// Per-lane draft token ids (flat, lane stride
    /// [`super::bridge::MAX_SPECULATIVE_TOKENS`]).
    pub speculative_draft_token_ids: &'a [u32],
}

/// Engine stats (`SparkServingStats`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ServingStats {
    /// Records not in `Free` state.
    pub live_request_count: u32,
    /// Requests queued in the request API.
    pub queued_request_count: u32,
    /// Requests completed in the request API.
    pub completed_request_count: u32,
    /// Requests cancelled in the request API.
    pub cancelled_request_count: u32,
    /// Events currently in the ring.
    pub event_count: u32,
    /// Event ring capacity.
    pub event_capacity: u32,
    /// Events dropped because the ring was full.
    pub dropped_event_count: u32,
    /// Status of the last pump-level operation.
    pub last_status: ServingStatus,
    /// High-water mark of prefill active sequences.
    pub maximum_prefill_active_sequence_count: u32,
    /// High-water mark of prefill lanes.
    pub maximum_prefill_lane_count: u32,
    /// High-water mark of decode active sequences.
    pub maximum_decode_active_sequence_count: u32,
    /// High-water mark of decode lanes.
    pub maximum_decode_lane_count: u32,
    /// Submissions attempted.
    pub submitted_request_count: u64,
    /// Submissions accepted.
    pub accepted_request_count: u64,
    /// Prefill dispatches invoked.
    pub prefill_dispatch_count: u64,
    /// Prefill-batch dispatches invoked.
    pub prefill_batch_dispatch_count: u64,
    /// Prompt tokens prefilled (summed over lanes).
    pub prefill_token_count: u64,
    /// Decode dispatches completed.
    pub decode_dispatch_count: u64,
    /// Tokens streamed to clients.
    pub decoded_token_count: u64,
    /// MTP draft tokens armed for verification.
    pub mtp_draft_token_count: u64,
    /// MTP speculative-verify dispatches.
    pub mtp_verify_dispatch_count: u64,
    /// MTP draft-ready dispatches (request API).
    pub mtp_draft_ready_count: u64,
    /// MTP accepted draft tokens (request API).
    pub mtp_accepted_draft_token_count: u64,
    /// MTP committed tokens (request API).
    pub mtp_committed_token_count: u64,
    /// MTP rejected tokens (request API).
    pub mtp_rejected_token_count: u64,
    /// Request streams completed.
    pub completed_stream_count: u64,
    /// JIT prefetch dispatches (request API).
    pub jit_prefetch_dispatch_count: u64,
    /// JIT prefetched blocks (request API).
    pub jit_prefetch_block_count: u64,
    /// Async JIT prefetch starts (request API).
    pub async_jit_prefetch_start_count: u64,
    /// Async JIT prefetch polls (request API).
    pub async_jit_prefetch_poll_count: u64,
    /// Async JIT prefetch completions (request API).
    pub async_jit_prefetch_completion_count: u64,
    /// Prefix-family dispatches (request API).
    pub prefix_family_dispatch_count: u64,
    /// Prefix-family members (request API).
    pub prefix_family_member_count: u64,
    /// Prefix-family saved prompt tokens (request API).
    pub prefix_family_saved_prompt_token_count: u64,
}

/// Prefill callback (`SparkServingPrefillFunction`).
///
/// `Ok(())` = dispatch consumed; `Err(Pending)` = dispatch accepted but
/// completing asynchronously (complete it later via
/// [`ServingEngine::complete_prefill_dispatch`]); any other `Err` fails the
/// dispatch's requests.
pub type PrefillFunction = Box<dyn FnMut(&ServingPrefillDispatch) -> Result<(), ServingStatus>>;

/// Decode callback (`SparkServingDecodeFunction`).
///
/// Same status contract as [`PrefillFunction`]; asynchronous completions
/// finish through [`ServingEngine::complete_decode_dispatch`]. `Err(Busy)`
/// before accepting re-queues the dispatch through the request API.
pub type DecodeFunction =
    Box<dyn FnMut(&ServingDecodeDispatch, &mut ServingDecodeResult) -> Result<(), ServingStatus>>;

/// Release-sequence callback (`SparkServingReleaseSequenceFunction`):
/// `(request_id, request_generation, sequence_id, token_count)`.
pub type ReleaseSequenceFunction = Box<dyn FnMut(u64, u64, u64, u32) -> Result<(), ServingStatus>>;

/// Engine configuration (`SparkServingEngineConfiguration`).
///
/// Capacities replace the C's caller-provided buffers; the engine allocates
/// them. `request_token_stride == 0` with
/// [`ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE`] selects dynamic per-record
/// token storage; otherwise static `stride`-word rows are required.
pub struct ServingEngineConfiguration {
    /// `SPARK_SERVING_ENGINE_FLAG_*` bits (0 = defaults).
    pub flags: u32,
    /// `SPARK_SERVING_RUNTIME_CONTRACT_FLAG_*` bits.
    pub runtime_contract_flags: u32,
    /// Default thinking budget for submissions that pass 0.
    pub default_thinking_token_budget: u32,
    /// Default output budget (0 = [`DEFAULT_OUTPUT_TOKEN_BUDGET`]).
    pub default_output_token_budget: u32,
    /// Default per-step prefill cap for submissions that pass 0.
    pub default_max_prefill_tokens_per_step: u32,
    /// Context limit (0 = [`DEFAULT_MAX_CONTEXT_TOKENS`]).
    pub max_context_tokens: u32,
    /// Generated request-id base (0 = [`DEFAULT_REQUEST_ID_BASE`]).
    pub request_id_base: u64,
    /// Request-API seam.
    pub request_api: Box<dyn ServingRequestApi>,
    /// Tokenizer for text submissions (optional, as in C).
    pub tokenizer: Option<Tokenizer>,
    /// Request record slots.
    pub request_record_capacity: u32,
    /// Static per-record token row size (0 only in dynamic-storage mode).
    pub request_token_stride: u32,
    /// Event ring slots (must exceed the decode-step safety margin).
    pub event_ring_capacity: u32,
    /// Per-lane stride of the prefill staging buffer.
    pub host_prefill_token_stride: u32,
    /// Lane capacity of the prefill staging buffer.
    pub host_prefill_lane_capacity: u32,
    /// Optional execution-side KV block-index array mirrored into the KV
    /// block table view (`kv_block_lane_stride * lane_count_capacity`
    /// words).
    pub execution_physical_block_indices: Option<Vec<u32>>,
    /// Per-lane stride of the KV block-index arrays.
    pub kv_block_lane_stride: u32,
    /// Per-lane capacity of the KV block-index arrays.
    pub kv_block_lane_capacity: u32,
    /// Lane capacity of the block-count staging.
    pub lane_count_capacity: u32,
    /// Prefill callback.
    pub prefill_function: PrefillFunction,
    /// Decode callback.
    pub decode_function: DecodeFunction,
    /// Optional release-sequence callback.
    pub release_sequence_function: Option<ReleaseSequenceFunction>,
    /// Stop token ids (at most [`MAX_STOP_TOKEN_IDS`]).
    pub stop_token_ids: Vec<u32>,
}

/// `SparkServingNormalizeFlags`.
fn normalize_flags(flags: u32) -> u32 {
    if flags == 0 {
        ENGINE_DEFAULT_FLAGS
    } else {
        flags
    }
}

/// `SparkServingNormalizeOutputTokenBudget`.
fn normalize_output_token_budget(output_token_budget: u32) -> u32 {
    if output_token_budget == 0 {
        DEFAULT_OUTPUT_TOKEN_BUDGET
    } else {
        output_token_budget
    }
}

/// `SparkServingEventRingSafetyMargin`.
fn event_ring_safety_margin() -> u32 {
    (MAX_DISPATCH_REQUEST_COUNT * MAX_DECODE_TOKENS_PER_LANE + MAX_DISPATCH_REQUEST_COUNT + 8)
        as u32
}

/// `SparkServingRuntimeContractIsProduction`.
fn runtime_contract_is_production(runtime_contract_flags: u32) -> bool {
    if (runtime_contract_flags & RUNTIME_CONTRACT_FLAG_TAIL_WINDOW_VALIDATION_ONLY) != 0 {
        return false;
    }
    (runtime_contract_flags & RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS)
        == RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS
}

/// `SparkServingRuntimeContractMatchesRequestApi`.
fn runtime_contract_matches_request_api(
    configuration_flags: u32,
    runtime_contract_flags: u32,
) -> bool {
    if (runtime_contract_flags & RUNTIME_CONTRACT_FLAG_INTERNAL_BATCHING_ENABLED) != 0
        && ((configuration_flags & CONFIGURATION_FLAG_PREFILL_BATCHING) == 0
            || (configuration_flags & CONFIGURATION_FLAG_DECODE_BATCHING) == 0)
    {
        return false;
    }
    if (runtime_contract_flags & RUNTIME_CONTRACT_FLAG_JIT_KV_PREFETCH_CONNECTED) != 0
        && (configuration_flags & CONFIGURATION_FLAG_JIT_KV_PREFETCH) == 0
    {
        return false;
    }
    true
}

/// `SparkServingHashHandle` (murmur3 fmix64 finalizer, as in C).
fn hash_handle(request_handle: u64) -> usize {
    let mut hash = request_handle;
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    (hash % RECORD_HASH_SLOTS as u64) as usize
}

/// `SparkServingFindRecordByHandle` over the intrusive chain table.
fn find_record_by_handle(
    records: &[RequestRecord],
    hash_heads: &[u32],
    request_handle: ServingRequestHandle,
) -> Option<usize> {
    if request_handle == INVALID_REQUEST_HANDLE {
        return None;
    }
    let mut record_index = hash_heads[hash_handle(request_handle)];
    while record_index != NO_RECORD_SLOT && (record_index as usize) < records.len() {
        let record = &records[record_index as usize];
        if record.state != RequestRecordState::Free && record.request_handle == request_handle {
            return Some(record_index as usize);
        }
        record_index = record.handle_hash_next;
    }
    None
}

/// `SparkServingInsertRecordHash`.
fn insert_record_hash(records: &mut [RequestRecord], hash_heads: &mut [u32], record_index: usize) {
    let handle = records[record_index].request_handle;
    if handle == INVALID_REQUEST_HANDLE {
        return;
    }
    let hash_slot = hash_handle(handle);
    records[record_index].handle_hash_next = hash_heads[hash_slot];
    hash_heads[hash_slot] = record_index as u32;
}

/// `SparkServingRemoveRecordHash`.
fn remove_record_hash(records: &mut [RequestRecord], hash_heads: &mut [u32], record_index: usize) {
    let handle = records[record_index].request_handle;
    if handle == INVALID_REQUEST_HANDLE {
        return;
    }
    let hash_slot = hash_handle(handle);
    let mut current_index = hash_heads[hash_slot];
    let mut previous_index = NO_RECORD_SLOT;
    while current_index != NO_RECORD_SLOT {
        if current_index as usize == record_index {
            if previous_index == NO_RECORD_SLOT {
                hash_heads[hash_slot] = records[current_index as usize].handle_hash_next;
            } else {
                records[previous_index as usize].handle_hash_next =
                    records[current_index as usize].handle_hash_next;
            }
            records[current_index as usize].handle_hash_next = NO_RECORD_SLOT;
            return;
        }
        previous_index = current_index;
        current_index = records[current_index as usize].handle_hash_next;
    }
}

/// `SparkServingFindRecordByRequestId` (linear scan, as in C).
fn find_record_by_request_id(records: &[RequestRecord], request_id: u64) -> Option<usize> {
    records.iter().position(|record| {
        record.state != RequestRecordState::Free && record.request_id == request_id
    })
}

/// Serving engine (`SparkServingEngine`).
pub struct ServingEngine {
    flags: u32,
    runtime_contract_flags: u32,
    default_thinking_token_budget: u32,
    default_output_token_budget: u32,
    default_max_prefill_tokens_per_step: u32,
    max_context_tokens: u32,
    next_generated_request_id: u64,
    request_api: Box<dyn ServingRequestApi>,
    tokenizer: Option<Tokenizer>,
    tokenizer_workspace: Option<Workspace>,
    records: Vec<RequestRecord>,
    free_record_head: u32,
    hash_heads: Vec<u32>,
    events: Vec<ServingEvent>,
    event_read_index: usize,
    event_write_index: usize,
    event_count: u32,
    dropped_event_count: u32,
    host_prefill_token_ids: Vec<u32>,
    host_prefill_token_stride: u32,
    host_prefill_lane_capacity: u32,
    host_physical_block_indices: Vec<u32>,
    execution_physical_block_indices: Option<Vec<u32>>,
    kv_block_lane_stride: u32,
    kv_block_lane_capacity: u32,
    lane_physical_block_counts: Vec<u32>,
    lane_count_capacity: u32,
    prefill_function: PrefillFunction,
    decode_function: DecodeFunction,
    release_sequence_function: Option<ReleaseSequenceFunction>,
    stop_token_ids: Vec<u32>,
    stats: ServingStats,
}

impl ServingEngine {
    /// `SparkServingEngineInitialize` (validation included; the engine
    /// value is only produced on success).
    pub fn new(configuration: ServingEngineConfiguration) -> Result<Self, ServingStatus> {
        let flags = normalize_flags(configuration.flags);
        // `SparkServingValidateConfiguration`.
        if (flags & !ENGINE_KNOWN_FLAGS) != 0 {
            return Err(ServingStatus::InvalidArgument);
        }
        if (flags & ENGINE_FLAG_REQUIRE_PRODUCTION_RUNTIME_CONTRACT) != 0
            && !runtime_contract_is_production(configuration.runtime_contract_flags)
        {
            return Err(ServingStatus::InvalidArgument);
        }
        if configuration.request_record_capacity == 0
            || configuration.event_ring_capacity <= event_ring_safety_margin()
            || configuration.host_prefill_token_stride == 0
            || configuration.host_prefill_lane_capacity == 0
            || configuration.kv_block_lane_stride == 0
            || configuration.kv_block_lane_capacity == 0
            || configuration.kv_block_lane_stride < configuration.kv_block_lane_capacity
            || configuration.lane_count_capacity == 0
        {
            return Err(ServingStatus::InvalidArgument);
        }
        let dynamic_storage = (flags & ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE) != 0;
        if dynamic_storage {
            if configuration.request_token_stride != 0 {
                return Err(ServingStatus::InvalidArgument);
            }
        } else if configuration.request_token_stride == 0 {
            return Err(ServingStatus::InvalidArgument);
        }
        if !runtime_contract_matches_request_api(
            configuration.request_api.configuration_flags(),
            configuration.runtime_contract_flags,
        ) {
            return Err(ServingStatus::InvalidArgument);
        }
        if configuration.stop_token_ids.len() > MAX_STOP_TOKEN_IDS {
            return Err(ServingStatus::InvalidArgument);
        }

        let record_capacity = configuration.request_record_capacity as usize;
        let static_stride = if dynamic_storage { 0 } else { configuration.request_token_stride };
        let mut records = Vec::with_capacity(record_capacity);
        for record_index in 0..record_capacity {
            let mut record = RequestRecord {
                token_ids: vec![0; static_stride as usize],
                ..RequestRecord::default()
            };
            record.reset(static_stride, true);
            record.free_record_next = if record_index + 1 < record_capacity {
                record_index as u32 + 1
            } else {
                NO_RECORD_SLOT
            };
            records.push(record);
        }

        let lane_count_capacity = configuration.lane_count_capacity as usize;
        let block_words = configuration.kv_block_lane_stride as usize * lane_count_capacity;
        let prefill_words = configuration.host_prefill_token_stride as usize
            * configuration.host_prefill_lane_capacity as usize;

        let mut engine = ServingEngine {
            flags,
            runtime_contract_flags: configuration.runtime_contract_flags,
            default_thinking_token_budget: configuration.default_thinking_token_budget,
            default_output_token_budget: normalize_output_token_budget(
                configuration.default_output_token_budget,
            ),
            default_max_prefill_tokens_per_step: configuration.default_max_prefill_tokens_per_step,
            max_context_tokens: if configuration.max_context_tokens == 0 {
                DEFAULT_MAX_CONTEXT_TOKENS
            } else {
                configuration.max_context_tokens
            },
            next_generated_request_id: if configuration.request_id_base == 0 {
                DEFAULT_REQUEST_ID_BASE
            } else {
                configuration.request_id_base
            },
            request_api: configuration.request_api,
            tokenizer: configuration.tokenizer,
            tokenizer_workspace: None,
            records,
            free_record_head: 0,
            hash_heads: vec![NO_RECORD_SLOT; RECORD_HASH_SLOTS],
            events: vec![ServingEvent::default(); configuration.event_ring_capacity as usize],
            event_read_index: 0,
            event_write_index: 0,
            event_count: 0,
            dropped_event_count: 0,
            host_prefill_token_ids: vec![0; prefill_words],
            host_prefill_token_stride: configuration.host_prefill_token_stride,
            host_prefill_lane_capacity: configuration.host_prefill_lane_capacity,
            host_physical_block_indices: vec![0; block_words],
            execution_physical_block_indices: configuration.execution_physical_block_indices,
            kv_block_lane_stride: configuration.kv_block_lane_stride,
            kv_block_lane_capacity: configuration.kv_block_lane_capacity,
            lane_physical_block_counts: vec![0; lane_count_capacity],
            lane_count_capacity: configuration.lane_count_capacity,
            prefill_function: configuration.prefill_function,
            decode_function: configuration.decode_function,
            release_sequence_function: configuration.release_sequence_function,
            stop_token_ids: configuration.stop_token_ids,
            stats: ServingStats::default(),
        };
        engine.refresh_stats();
        Ok(engine)
    }

    /// Runtime-contract flags the engine was built with (diagnostics).
    pub fn runtime_contract_flags(&self) -> u32 {
        self.runtime_contract_flags
    }

    /// Current free-list head (`free_record_head` in C; test introspection).
    pub fn free_record_head(&self) -> u32 {
        self.free_record_head
    }

    /// Per-record token capacity by record index (test introspection).
    pub fn record_token_capacity(&self, record_index: usize) -> Option<u32> {
        self.records.get(record_index).map(|record| record.token_capacity)
    }

    /// `SparkServingEventRingFreeCount`.
    fn event_ring_free_count(&self) -> u32 {
        let capacity = self.events.len() as u32;
        capacity.saturating_sub(self.event_count)
    }

    /// `SparkServingRefreshStats`.
    fn refresh_stats(&mut self) {
        let counters = self.request_api.counters();
        self.stats.live_request_count =
            self.records.iter().filter(|record| record.state != RequestRecordState::Free).count()
                as u32;
        self.stats.queued_request_count = counters.queued_request_count;
        self.stats.completed_request_count = counters.completed_request_count;
        self.stats.cancelled_request_count = counters.cancelled_request_count;
        self.stats.event_count = self.event_count;
        self.stats.event_capacity = self.events.len() as u32;
        self.stats.dropped_event_count = self.dropped_event_count;
        self.stats.jit_prefetch_dispatch_count = counters.jit_prefetch_dispatch_count;
        self.stats.jit_prefetch_block_count = counters.jit_prefetch_block_count;
        self.stats.async_jit_prefetch_start_count = counters.async_jit_prefetch_start_count;
        self.stats.async_jit_prefetch_poll_count = counters.async_jit_prefetch_poll_count;
        self.stats.async_jit_prefetch_completion_count =
            counters.async_jit_prefetch_completion_count;
        self.stats.prefix_family_dispatch_count = counters.prefix_family_dispatch_count;
        self.stats.prefix_family_member_count = counters.prefix_family_member_count;
        self.stats.prefix_family_saved_prompt_token_count =
            counters.prefix_family_saved_prompt_token_count;
        self.stats.mtp_draft_ready_count = counters.mtp_draft_ready_count;
        self.stats.mtp_accepted_draft_token_count = counters.mtp_accepted_draft_token_count;
        self.stats.mtp_committed_token_count = counters.mtp_committed_token_count;
        self.stats.mtp_rejected_token_count = counters.mtp_rejected_token_count;
    }

    /// `SparkServingPushEvent`.
    fn push_event(&mut self, event: &ServingEvent) -> Result<(), ServingStatus> {
        if self.events.is_empty() {
            return Err(ServingStatus::InvalidArgument);
        }
        if self.event_count >= self.events.len() as u32 {
            self.dropped_event_count += 1;
            return Err(ServingStatus::CapacityExceeded);
        }
        self.events[self.event_write_index] = *event;
        self.event_write_index += 1;
        if self.event_write_index == self.events.len() {
            self.event_write_index = 0;
        }
        self.event_count += 1;
        Ok(())
    }

    /// `SparkServingPushSimpleEvent`.
    fn push_simple_event(
        &mut self,
        event_kind: ServingEventKind,
        status: ServingStatus,
        record_index: Option<usize>,
    ) -> Result<(), ServingStatus> {
        let mut event = ServingEvent { kind: event_kind, status, ..ServingEvent::default() };
        if let Some(record_index) = record_index {
            let record = &self.records[record_index];
            event.request_id = record.request_id;
            event.sequence_id = record.sequence_id;
            event.request_handle = record.request_handle;
            event.prompt_token_count = record.prompt_token_count;
        }
        self.push_event(&event)
    }

    /// `SparkServingTokenIsStopToken`.
    fn token_is_stop_token(&self, token_id: u32) -> bool {
        self.stop_token_ids.contains(&token_id)
    }

    /// `SparkServingFindFreeRecord`.
    fn find_free_record(&self) -> Option<usize> {
        if self.free_record_head == NO_RECORD_SLOT
            || self.free_record_head as usize >= self.records.len()
        {
            return None;
        }
        if self.records[self.free_record_head as usize].state != RequestRecordState::Free {
            return None;
        }
        Some(self.free_record_head as usize)
    }

    /// `SparkServingEnsureRecordTokenCapacity`.
    fn ensure_record_token_capacity(
        &mut self,
        record_index: usize,
        required_token_capacity: u32,
    ) -> Result<(), ServingStatus> {
        if required_token_capacity == 0 || required_token_capacity > self.max_context_tokens {
            return Err(ServingStatus::CapacityExceeded);
        }
        if required_token_capacity <= self.records[record_index].token_capacity {
            return Ok(());
        }
        if (self.flags & ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE) == 0 {
            return Err(ServingStatus::CapacityExceeded);
        }

        let mut grown_capacity = match self.records[record_index].token_capacity {
            0 => 64,
            capacity => capacity,
        };
        while grown_capacity < required_token_capacity
            && grown_capacity <= self.max_context_tokens / 2
        {
            grown_capacity *= 2;
        }
        if grown_capacity < required_token_capacity {
            grown_capacity = required_token_capacity;
        }
        if grown_capacity > self.max_context_tokens {
            grown_capacity = self.max_context_tokens;
        }
        self.records[record_index].token_ids.resize(grown_capacity as usize, 0);
        self.records[record_index].token_capacity = grown_capacity;
        Ok(())
    }

    /// `SparkServingApplyContextBudget`.
    fn apply_context_budget(
        &self,
        prompt_token_count: u32,
        requested_thinking_token_budget: u32,
        requested_output_token_budget: u32,
    ) -> Result<(u32, u32), ServingStatus> {
        if prompt_token_count == 0 || prompt_token_count > self.max_context_tokens {
            return Err(ServingStatus::CapacityExceeded);
        }

        let mut thinking_token_budget = if requested_thinking_token_budget != 0 {
            requested_thinking_token_budget
        } else {
            self.default_thinking_token_budget
        };
        let mut output_token_budget = if requested_output_token_budget != 0 {
            requested_output_token_budget
        } else {
            self.default_output_token_budget
        };
        if (self.flags & ENGINE_FLAG_CLAMP_BUDGET_TO_CONTEXT) == 0 {
            if prompt_token_count as u64 + thinking_token_budget as u64 + output_token_budget as u64
                > self.max_context_tokens as u64
            {
                return Err(ServingStatus::CapacityExceeded);
            }
            return Ok((thinking_token_budget, output_token_budget));
        }

        let mut available_generation_tokens = self.max_context_tokens - prompt_token_count;
        if thinking_token_budget > available_generation_tokens {
            thinking_token_budget = available_generation_tokens;
        }
        available_generation_tokens -= thinking_token_budget;
        if output_token_budget > available_generation_tokens {
            output_token_budget = available_generation_tokens;
        }
        if thinking_token_budget == 0 && output_token_budget == 0 {
            output_token_budget = 1;
            if prompt_token_count as u64 + output_token_budget as u64
                > self.max_context_tokens as u64
            {
                return Err(ServingStatus::CapacityExceeded);
            }
        }
        Ok((thinking_token_budget, output_token_budget))
    }

    /// `SparkServingResolveSubmittedSequence`.
    fn resolve_submitted_sequence(
        &self,
        request_handle: u64,
        request_id: u64,
    ) -> Result<u64, ServingStatus> {
        if request_handle == 0 || request_id == 0 {
            return Err(ServingStatus::InvalidArgument);
        }
        let cache_state = self.request_api.request_cache_state(request_handle)?;
        if cache_state.request_id != request_id || cache_state.sequence_id == 0 {
            return Err(ServingStatus::InternalError);
        }
        Ok(cache_state.sequence_id)
    }

    /// `SparkServingSubmitPreparedRecord`.
    #[allow(clippy::too_many_arguments)]
    fn submit_prepared_record(
        &mut self,
        record_index: usize,
        request_flags: u32,
        priority: u32,
        thinking_token_budget: u32,
        output_token_budget: u32,
        max_prefill_tokens_per_step: u32,
        request_id: u64,
        sequence_id: u64,
    ) -> Result<SubmitResult, ServingStatus> {
        let record = &self.records[record_index];
        let api_request = ApiSubmitRequest {
            flags: request_flags,
            priority,
            prompt_token_count: record.prompt_token_count,
            thinking_token_budget,
            output_token_budget,
            max_prefill_tokens_per_step: if max_prefill_tokens_per_step != 0 {
                max_prefill_tokens_per_step
            } else {
                self.default_max_prefill_tokens_per_step
            },
            request_id,
            sequence_id,
            prompt_token_ids: &record.token_ids[..record.prompt_token_count as usize],
        };

        let request_handle = self.request_api.submit(&api_request)?;
        let sequence_id = match self.resolve_submitted_sequence(request_handle, request_id) {
            Ok(sequence_id) => sequence_id,
            Err(status) => {
                let _ = self.request_api.cancel_request(request_handle);
                let _ = self.request_api.release_completed_request(request_handle);
                return Err(status);
            }
        };

        if record_index as u32 != self.free_record_head
            || self.records[record_index].state != RequestRecordState::Free
        {
            let _ = self.request_api.cancel_request(request_handle);
            let _ = self.request_api.release_completed_request(request_handle);
            return Err(ServingStatus::InternalError);
        }
        self.free_record_head = self.records[record_index].free_record_next;
        self.records[record_index].free_record_next = NO_RECORD_SLOT;

        {
            let record = &mut self.records[record_index];
            record.state = RequestRecordState::Submitted;
            record.flags = request_flags;
            record.thinking_token_budget = thinking_token_budget;
            record.output_token_budget = output_token_budget;
            record.request_id = request_id;
            record.sequence_id = sequence_id;
            record.request_handle = request_handle;
        }
        insert_record_hash(&mut self.records, &mut self.hash_heads, record_index);
        self.stats.submitted_request_count += 1;
        self.stats.accepted_request_count += 1;

        let record = &self.records[record_index];
        let result = SubmitResult {
            prompt_token_count: record.prompt_token_count,
            required_token_capacity: record.prompt_token_count,
            thinking_token_budget,
            output_token_budget,
            request_id,
            sequence_id,
            request_handle,
        };

        let status = self.push_simple_event(
            ServingEventKind::RequestAccepted,
            ServingStatus::Ok,
            Some(record_index),
        );
        self.refresh_stats();
        status?;
        Ok(result)
    }

    /// `SparkServingEngineSubmitTokenIds`.
    pub fn submit_token_ids(
        &mut self,
        request: &SubmitTokenIdsRequest,
    ) -> Result<SubmitResult, ServingStatus> {
        if (request.flags & !SUBMIT_KNOWN_FLAGS) != 0 || request.token_ids.is_empty() {
            return Err(ServingStatus::InvalidArgument);
        }

        let record_index = self.find_free_record().ok_or(ServingStatus::CapacityExceeded)?;
        let (thinking_token_budget, output_token_budget) = self.apply_context_budget(
            request.token_ids.len() as u32,
            request.thinking_token_budget,
            request.output_token_budget,
        )?;
        let required_token_capacity = request.token_ids.len() as u64
            + thinking_token_budget as u64
            + output_token_budget as u64;
        if required_token_capacity > u32::MAX as u64 {
            return Err(ServingStatus::CapacityExceeded);
        }
        self.ensure_record_token_capacity(record_index, required_token_capacity as u32)?;

        {
            let record = &mut self.records[record_index];
            record.token_ids[..request.token_ids.len()].copy_from_slice(request.token_ids);
            record.prompt_token_count = request.token_ids.len() as u32;
        }
        let request_id = if request.request_id != 0 {
            request.request_id
        } else {
            let request_id = self.next_generated_request_id;
            self.next_generated_request_id += 1;
            request_id
        };
        self.submit_prepared_record(
            record_index,
            request.flags,
            request.priority,
            thinking_token_budget,
            output_token_budget,
            request.max_prefill_tokens_per_step,
            request_id,
            request.sequence_id,
        )
    }

    /// `SparkServingEngineSubmitText`.
    ///
    /// Encode failures return [`SubmitFailure`], which carries the partial
    /// result (`prompt_token_count`/`required_token_capacity`) the C writes
    /// into its out-param on this path.
    pub fn submit_text(
        &mut self,
        request: &SubmitTextRequest,
    ) -> Result<SubmitResult, SubmitFailure> {
        if (request.flags & !SUBMIT_KNOWN_FLAGS) != 0 || request.text.is_empty() {
            return Err(SubmitFailure::from_status(ServingStatus::InvalidArgument));
        }
        if self.tokenizer.is_none() {
            return Err(SubmitFailure::from_status(ServingStatus::InvalidArgument));
        }

        let record_index = self
            .find_free_record()
            .ok_or_else(|| SubmitFailure::from_status(ServingStatus::CapacityExceeded))?;

        if self.tokenizer_workspace.is_none() {
            self.tokenizer_workspace = Some(
                Workspace::new(request.text.len() + 1)
                    .map_err(|error| SubmitFailure::from_status(ServingStatus::from(error)))?,
            );
        }
        let encoding = self.encode_into_record(record_index, request)?;
        let prompt_token_count = encoding.token_count() as u32;

        let (thinking_token_budget, output_token_budget) = self
            .apply_context_budget(
                prompt_token_count,
                request.thinking_token_budget,
                request.output_token_budget,
            )
            .map_err(SubmitFailure::from_status)?;
        let required_token_capacity =
            prompt_token_count as u64 + thinking_token_budget as u64 + output_token_budget as u64;
        if required_token_capacity > u32::MAX as u64 {
            return Err(SubmitFailure::from_status(ServingStatus::CapacityExceeded));
        }
        self.ensure_record_token_capacity(record_index, required_token_capacity as u32)
            .map_err(SubmitFailure::from_status)?;

        {
            let record = &mut self.records[record_index];
            record.token_ids[..encoding.token_count()].copy_from_slice(encoding.token_ids());
            record.prompt_token_count = prompt_token_count;
        }
        let request_id = if request.request_id != 0 {
            request.request_id
        } else {
            let request_id = self.next_generated_request_id;
            self.next_generated_request_id += 1;
            request_id
        };
        self.submit_prepared_record(
            record_index,
            request.flags,
            request.priority,
            thinking_token_budget,
            output_token_budget,
            request.max_prefill_tokens_per_step,
            request_id,
            request.sequence_id,
        )
        .map_err(SubmitFailure::from_status)
    }

    /// The tokenizer-encode step of `SparkServingEngineSubmitText`,
    /// including the dynamic-storage grow-and-retry path.
    fn encode_into_record(
        &mut self,
        record_index: usize,
        request: &SubmitTextRequest,
    ) -> Result<Encoding, SubmitFailure> {
        let mut encoding =
            Encoding::with_capacity(self.records[record_index].token_capacity as usize);
        let status = {
            let tokenizer = self.tokenizer.as_ref().expect("checked above");
            let workspace = self.tokenizer_workspace.as_mut().expect("created above");
            tokenizer.encode_with_workspace(
                request.text,
                request.tokenizer_encode_flags,
                workspace,
                &mut encoding,
            )
        };
        match status {
            Ok(()) => Ok(encoding),
            Err(error) => {
                if matches!(error, spark_text::tokenizer::TokenizerError::CapacityExceeded)
                    && (self.flags & ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE) != 0
                {
                    let required_prompt_token_capacity =
                        (encoding.token_count() + encoding.overflow_token_count()) as u32;
                    self.ensure_record_token_capacity(record_index, required_prompt_token_capacity)
                        .map_err(SubmitFailure::from_status)?;
                    let mut retry_encoding =
                        Encoding::with_capacity(self.records[record_index].token_capacity as usize);
                    let retry_status = {
                        let tokenizer = self.tokenizer.as_ref().expect("checked above");
                        let workspace = self.tokenizer_workspace.as_mut().expect("created above");
                        tokenizer.encode_with_workspace(
                            request.text,
                            request.tokenizer_encode_flags,
                            workspace,
                            &mut retry_encoding,
                        )
                    };
                    return match retry_status {
                        Ok(()) => Ok(retry_encoding),
                        Err(retry_error) => {
                            Err(SubmitFailure::from_encode_error(retry_error, &retry_encoding))
                        }
                    };
                }
                Err(SubmitFailure::from_encode_error(error, &encoding))
            }
        }
    }
}

/// Failure of [`ServingEngine::submit_text`]. Mirrors the C contract where
/// the status is returned and the submit-result out-param still receives
/// the encode progress (`prompt_token_count` / `required_token_capacity`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubmitFailure {
    status: ServingStatus,
    partial_result: Option<SubmitResult>,
}

impl SubmitFailure {
    fn from_status(status: ServingStatus) -> Self {
        SubmitFailure { status, partial_result: None }
    }

    fn from_encode_error(
        error: spark_text::tokenizer::TokenizerError,
        encoding: &Encoding,
    ) -> Self {
        SubmitFailure {
            status: ServingStatus::from(error),
            partial_result: Some(SubmitResult {
                prompt_token_count: encoding.token_count() as u32,
                required_token_capacity: (encoding.token_count() + encoding.overflow_token_count())
                    as u32,
                ..SubmitResult::default()
            }),
        }
    }

    /// The `SparkStatus` the C entry point returns.
    pub fn status(&self) -> ServingStatus {
        self.status
    }

    /// The partial submit result (present on encode-path failures only).
    pub fn partial_result(&self) -> Option<&SubmitResult> {
        self.partial_result.as_ref()
    }
}

impl std::fmt::Display for SubmitFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.status.fmt(formatter)
    }
}

impl std::error::Error for SubmitFailure {}

impl From<SubmitFailure> for ServingStatus {
    fn from(failure: SubmitFailure) -> Self {
        failure.status
    }
}

// ---------------------------------------------------------------------------
// Dispatch invocation and completion.
// ---------------------------------------------------------------------------

impl ServingEngine {
    /// `SparkServingFailDispatchRequests`.
    fn fail_dispatch_requests(&mut self, dispatch: &Dispatch, failure_status: ServingStatus) {
        for lane_index in 0..dispatch.request_count.min(MAX_DISPATCH_REQUEST_COUNT as u32) {
            let handle = dispatch.request_handles[lane_index as usize];
            let Some(record_index) = find_record_by_handle(&self.records, &self.hash_heads, handle)
            else {
                continue;
            };
            let record = &mut self.records[record_index];
            if record.state == RequestRecordState::Completed
                || record.state == RequestRecordState::Cancelled
            {
                continue;
            }
            record.state = RequestRecordState::Cancelled;
            let _ = self.push_simple_event(
                ServingEventKind::RequestCancelled,
                failure_status,
                Some(record_index),
            );
        }
    }

    /// `SparkServingPublishPrefillEvents`.
    fn publish_prefill_events(
        &mut self,
        dispatch: &Dispatch,
        prefill_view: &PrefillDispatchView,
    ) -> Result<(), ServingStatus> {
        for lane in prefill_view.lanes.iter().take(prefill_view.lane_count as usize) {
            let event = ServingEvent {
                kind: ServingEventKind::PrefillProgress,
                status: ServingStatus::Ok,
                prompt_token_offset: lane.prompt_token_offset,
                prompt_token_count: lane.prompt_token_count,
                dispatch_kind: prefill_view.kind,
                dispatch_flags: dispatch.flags,
                request_id: lane.request_id,
                sequence_id: lane.sequence_id,
                request_handle: lane.request_handle,
                ..ServingEvent::default()
            };
            if self.push_event(&event).is_err() {
                return Err(ServingStatus::CapacityExceeded);
            }
        }
        Ok(())
    }

    /// `SparkServingInvokePrefill` (build + callback + events + stats).
    fn invoke_prefill(
        &mut self,
        dispatch: &Dispatch,
        step_index: u32,
    ) -> Result<(), ServingStatus> {
        let prefill_view = self.request_api.describe_prefill_dispatch(dispatch)?;
        if prefill_view.lane_count > self.host_prefill_lane_capacity
            || prefill_view.lane_count > self.lane_count_capacity
            || prefill_view.prompt_token_stride > self.host_prefill_token_stride
        {
            return Err(ServingStatus::CapacityExceeded);
        }
        self.request_api.copy_prefill_dispatch_token_ids(
            dispatch,
            &mut self.host_prefill_token_ids,
            self.host_prefill_token_stride,
            self.host_prefill_lane_capacity,
        )?;

        let callback_status = {
            let block_table_view = self.request_api.build_dispatch_kv_block_table_view(
                dispatch,
                &mut self.host_physical_block_indices,
                self.execution_physical_block_indices.as_deref(),
                self.kv_block_lane_stride,
                self.kv_block_lane_capacity,
                &mut self.lane_physical_block_counts,
            )?;
            let prefill_dispatch = ServingPrefillDispatch {
                step_index,
                dispatch_kind: dispatch.kind,
                active_sequence_count: prefill_view.active_sequence_count,
                lane_count: prefill_view.lane_count,
                prompt_token_offset: prefill_view.prompt_token_offset,
                prompt_token_count: prefill_view.prompt_token_count,
                prompt_token_stride: prefill_view.prompt_token_stride,
                host_token_stride: self.host_prefill_token_stride,
                host_token_ids: &self.host_prefill_token_ids,
                lanes: &prefill_view.lanes,
                kv_block_table_view: &block_table_view,
            };
            (self.prefill_function)(&prefill_dispatch)
        };
        match callback_status {
            Ok(()) | Err(ServingStatus::Pending) => {}
            Err(status) => return Err(status),
        }

        if self.publish_prefill_events(dispatch, &prefill_view).is_err() {
            return Err(ServingStatus::IoError);
        }

        self.stats.prefill_dispatch_count += 1;
        if dispatch.kind == DispatchKind::PrefillBatch {
            self.stats.prefill_batch_dispatch_count += 1;
        }
        self.stats.prefill_token_count +=
            prefill_view.prompt_token_count as u64 * prefill_view.lane_count as u64;
        if prefill_view.active_sequence_count > self.stats.maximum_prefill_active_sequence_count {
            self.stats.maximum_prefill_active_sequence_count = prefill_view.active_sequence_count;
        }
        if prefill_view.lane_count > self.stats.maximum_prefill_lane_count {
            self.stats.maximum_prefill_lane_count = prefill_view.lane_count;
        }
        callback_status
    }

    /// `SparkServingBuildDecodeDispatch`'s per-lane fill (static so it can
    /// run while the KV staging buffers are mutably borrowed for the view).
    fn fill_decode_inputs(
        records: &[RequestRecord],
        hash_heads: &[u32],
        dispatch: &Dispatch,
        decode_view: &DecodeDispatchView,
        input_token_ids: &mut [u32],
        speculative_draft_token_ids: &mut [u32],
    ) -> Result<(), ServingStatus> {
        for lane_index in 0..decode_view.lane_count as usize {
            let handle = dispatch.request_handles[lane_index];
            let record_index = find_record_by_handle(records, hash_heads, handle)
                .ok_or(ServingStatus::InvalidArgument)?;
            let record = &records[record_index];
            if record.prompt_token_count == 0
                || record.prompt_token_count + record.streamed_decode_token_count
                    > record.token_capacity
            {
                return Err(ServingStatus::InvalidArgument);
            }
            let input_token_index =
                (record.prompt_token_count + record.streamed_decode_token_count - 1) as usize;
            input_token_ids[lane_index] = record.token_ids[input_token_index];
            if dispatch.kind == DispatchKind::SpeculativeVerifyBatch {
                for draft_index in 0..dispatch.speculative_token_count as usize {
                    speculative_draft_token_ids
                        [lane_index * super::bridge::MAX_SPECULATIVE_TOKENS + draft_index] =
                        dispatch.speculative_draft_token_ids
                            [lane_index * super::bridge::MAX_SPECULATIVE_TOKENS + draft_index];
                }
            }
        }
        Ok(())
    }

    /// `SparkServingInvokeDecode`.
    fn invoke_decode(&mut self, dispatch: &mut Dispatch) -> Result<(), ServingStatus> {
        let mut decode_result = {
            let block_table_view = self.request_api.build_dispatch_kv_block_table_view(
                dispatch,
                &mut self.host_physical_block_indices,
                self.execution_physical_block_indices.as_deref(),
                self.kv_block_lane_stride,
                self.kv_block_lane_capacity,
                &mut self.lane_physical_block_counts,
            )?;
            let decode_view = self.request_api.describe_decode_dispatch(dispatch)?;

            let mut input_token_ids = vec![0u32; MAX_DISPATCH_REQUEST_COUNT];
            let mut speculative_draft_token_ids =
                vec![0u32; MAX_DISPATCH_REQUEST_COUNT * super::bridge::MAX_SPECULATIVE_TOKENS];
            Self::fill_decode_inputs(
                &self.records,
                &self.hash_heads,
                dispatch,
                &decode_view,
                &mut input_token_ids,
                &mut speculative_draft_token_ids,
            )?;

            let decode_dispatch = ServingDecodeDispatch {
                dispatch_kind: dispatch.kind,
                request_count: dispatch.request_count,
                active_sequence_count: block_table_view.lane_count,
                speculative_token_index: 0,
                request_dispatch: dispatch,
                kv_block_table_view: &block_table_view,
                decode_view: &decode_view,
                input_token_ids: &input_token_ids,
                speculative_token_count: if dispatch.kind == DispatchKind::SpeculativeVerifyBatch {
                    dispatch.speculative_token_count
                } else {
                    0
                },
                speculative_draft_token_ids: &speculative_draft_token_ids,
            };
            let mut decode_result =
                ServingDecodeResult::new(dispatch.request_count, MAX_DECODE_TOKENS_PER_LANE as u32);
            (self.decode_function)(&decode_dispatch, &mut decode_result)?;
            decode_result
        };

        self.complete_decode_dispatch(dispatch, &mut decode_result)
    }

    /// `SparkServingValidateDecodeResult`.
    fn validate_decode_result(
        dispatch: &Dispatch,
        decode_result: &ServingDecodeResult,
    ) -> Result<(), ServingStatus> {
        if decode_result.lane_count != dispatch.request_count
            || decode_result.lane_count == 0
            || decode_result.lane_count > MAX_DISPATCH_REQUEST_COUNT as u32
            || decode_result.token_stride == 0
            || decode_result.token_stride > MAX_DECODE_TOKENS_PER_LANE as u32
        {
            return Err(ServingStatus::InvalidArgument);
        }

        let maximum_token_count = if dispatch.kind == DispatchKind::SpeculativeVerifyBatch {
            dispatch.speculative_verifier_token_count
        } else if (dispatch.flags & DISPATCH_FLAG_MTP_COMMIT) != 0 {
            dispatch.mtp_draft_token_budget + 1
        } else {
            1
        };
        if maximum_token_count == 0 || maximum_token_count > MAX_DECODE_TOKENS_PER_LANE as u32 {
            return Err(ServingStatus::InvalidArgument);
        }

        for lane_index in 0..decode_result.lane_count as usize {
            if decode_result.token_counts[lane_index] == 0
                || decode_result.token_counts[lane_index] > maximum_token_count
                || decode_result.token_counts[lane_index] > decode_result.token_stride
            {
                return Err(ServingStatus::InvalidArgument);
            }
            if decode_result.draft_token_counts[lane_index] > MTP_MAX_DRAFT_TOKEN_COUNT as u32 {
                return Err(ServingStatus::InvalidArgument);
            }
            if decode_result.draft_token_counts[lane_index] != 0
                && (dispatch.kind != DispatchKind::SpeculativeVerifyBatch
                    || (dispatch.flags & DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) == 0)
                && (dispatch.kind != DispatchKind::DecodeBatch
                    || (dispatch.flags & DISPATCH_FLAG_MTP_COMMIT) == 0)
            {
                return Err(ServingStatus::InvalidArgument);
            }
            if dispatch.kind != DispatchKind::SpeculativeVerifyBatch
                && (dispatch.flags & DISPATCH_FLAG_MTP_COMMIT) == 0
                && decode_result.token_counts[lane_index] != 1
            {
                return Err(ServingStatus::InvalidArgument);
            }
        }
        Ok(())
    }

    /// `SparkServingResolveMtpDecode`.
    fn resolve_mtp_decode(
        dispatch: &mut Dispatch,
        decode_result: &ServingDecodeResult,
    ) -> Result<(), ServingStatus> {
        if dispatch.kind != DispatchKind::DecodeBatch
            || (dispatch.flags & DISPATCH_FLAG_MTP_COMMIT) == 0
        {
            return Ok(());
        }
        for lane_index in 0..decode_result.lane_count as usize {
            if decode_result.token_counts[lane_index] == 0 {
                return Err(ServingStatus::InvalidArgument);
            }
            dispatch.decode_committed_token_counts[lane_index] = 1;
        }
        Ok(())
    }

    /// `SparkServingCaptureMtpDraftTokens`.
    fn capture_mtp_draft_tokens(
        dispatch: &Dispatch,
        decode_result: &mut ServingDecodeResult,
        draft_token_ids: &mut [u32],
        draft_lane_stride: usize,
    ) -> Result<u32, ServingStatus> {
        if decode_result.lane_count != 0 && decode_result.draft_token_counts[0] != 0 {
            let draft_token_count = decode_result.draft_token_counts[0];
            if ((dispatch.kind != DispatchKind::SpeculativeVerifyBatch
                || (dispatch.flags & DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) == 0)
                && (dispatch.kind != DispatchKind::DecodeBatch
                    || (dispatch.flags & DISPATCH_FLAG_MTP_COMMIT) == 0))
                || draft_lane_stride < draft_token_count as usize
            {
                return Err(ServingStatus::InvalidArgument);
            }
            for lane_index in 0..decode_result.lane_count as usize {
                if decode_result.draft_token_counts[lane_index] != draft_token_count {
                    return Err(ServingStatus::InvalidArgument);
                }
                for draft_index in 0..draft_token_count as usize {
                    draft_token_ids[lane_index * draft_lane_stride + draft_index] = decode_result
                        .draft_token_ids[lane_index * MTP_MAX_DRAFT_TOKEN_COUNT + draft_index];
                }
            }
            return Ok(draft_token_count);
        }
        if dispatch.kind != DispatchKind::DecodeBatch
            || (dispatch.flags & DISPATCH_FLAG_MTP_COMMIT) == 0
        {
            return Ok(0);
        }
        if draft_lane_stride < dispatch.mtp_draft_token_budget as usize {
            return Err(ServingStatus::InvalidArgument);
        }

        let mut draft_token_count = 0u32;
        for lane_index in 0..decode_result.lane_count as usize {
            if decode_result.token_counts[lane_index] == 0 {
                return Err(ServingStatus::InvalidArgument);
            }
            let lane_draft_count = decode_result.token_counts[lane_index] - 1;
            if lane_draft_count > dispatch.mtp_draft_token_budget {
                return Err(ServingStatus::InvalidArgument);
            }
            if lane_index == 0 {
                draft_token_count = lane_draft_count;
            } else if lane_draft_count != draft_token_count {
                return Err(ServingStatus::InvalidArgument);
            }
            for draft_index in 0..lane_draft_count as usize {
                draft_token_ids[lane_index * draft_lane_stride + draft_index] = decode_result
                    .token_ids[lane_index * MAX_DECODE_TOKENS_PER_LANE + draft_index + 1];
            }
            decode_result.token_counts[lane_index] = 1;
        }
        Ok(draft_token_count)
    }

    /// `SparkServingClampSpeculativeVerifyDecodeResult`.
    fn clamp_speculative_verify_decode_result(
        dispatch: &Dispatch,
        decode_result: &mut ServingDecodeResult,
    ) -> Result<(), ServingStatus> {
        if dispatch.kind != DispatchKind::SpeculativeVerifyBatch {
            return Ok(());
        }
        for lane_index in 0..decode_result.lane_count as usize {
            let accepted_token_count = dispatch.speculative_accepted_token_counts[lane_index];
            let committed_token_count = dispatch.speculative_committed_token_counts[lane_index];
            if committed_token_count == 0
                || committed_token_count != accepted_token_count + 1
                || committed_token_count > decode_result.token_counts[lane_index]
            {
                return Err(ServingStatus::InvalidArgument);
            }
            decode_result.token_ids
                [lane_index * MAX_DECODE_TOKENS_PER_LANE + accepted_token_count as usize] =
                dispatch.speculative_fallback_token_ids[lane_index];
            decode_result.token_counts[lane_index] = committed_token_count;
        }
        Ok(())
    }

    /// `SparkServingResolveSpeculativeDecode`.
    fn resolve_speculative_decode(
        &mut self,
        dispatch: &mut Dispatch,
        decode_result: &ServingDecodeResult,
    ) -> Result<(), ServingStatus> {
        if dispatch.kind != DispatchKind::SpeculativeVerifyBatch {
            return Ok(());
        }

        let verifier_token_count = decode_result.token_counts[0];
        for lane_index in 1..decode_result.lane_count as usize {
            if decode_result.token_counts[lane_index] != verifier_token_count {
                return Err(ServingStatus::InvalidArgument);
            }
        }

        let mut verifier_token_ids =
            vec![0u32; MAX_DISPATCH_REQUEST_COUNT * MAX_DECODE_TOKENS_PER_LANE];
        for lane_index in 0..decode_result.lane_count as usize {
            for token_index in 0..verifier_token_count as usize {
                verifier_token_ids[lane_index * MAX_DECODE_TOKENS_PER_LANE + token_index] =
                    decode_result.token_ids[lane_index * MAX_DECODE_TOKENS_PER_LANE + token_index];
            }
        }

        self.request_api.resolve_speculative_verify_dispatch(
            dispatch,
            &verifier_token_ids,
            MAX_DECODE_TOKENS_PER_LANE as u32,
            verifier_token_count,
        )
    }

    /// `SparkServingPublishDecodeEvents`.
    fn publish_decode_events(
        &mut self,
        dispatch: &Dispatch,
        decode_result: &ServingDecodeResult,
        finish_handles: &mut [ServingRequestHandle],
    ) -> Result<u32, ServingStatus> {
        let mut finish_handle_count = 0u32;
        for lane_index in 0..decode_result.lane_count as usize {
            let handle = dispatch.request_handles[lane_index];
            let record_index = find_record_by_handle(&self.records, &self.hash_heads, handle)
                .ok_or(ServingStatus::InvalidArgument)?;
            let mut lane_finish =
                (decode_result.lane_flags[lane_index] & DECODE_RESULT_FLAG_FINISH_REQUEST) != 0;
            let mut lane_token_count = decode_result.token_counts[lane_index];
            for token_index in 0..lane_token_count as usize {
                if self.token_is_stop_token(
                    decode_result.token_ids[lane_index * MAX_DECODE_TOKENS_PER_LANE + token_index],
                ) {
                    lane_token_count = token_index as u32 + 1;
                    lane_finish = true;
                    break;
                }
            }
            for token_index in 0..lane_token_count as usize {
                let token_id =
                    decode_result.token_ids[lane_index * MAX_DECODE_TOKENS_PER_LANE + token_index];
                let record = &mut self.records[record_index];
                let absolute_index = record.prompt_token_count
                    + record.streamed_decode_token_count
                    + token_index as u32;
                if absolute_index >= record.token_capacity {
                    return Err(ServingStatus::CapacityExceeded);
                }
                record.token_ids[absolute_index as usize] = token_id;
                if (decode_result.lane_flags[lane_index]
                    & DECODE_RESULT_FLAG_TOKEN_STREAM_SUPPRESSED)
                    != 0
                {
                    continue;
                }
                let event = ServingEvent {
                    kind: ServingEventKind::Token,
                    status: ServingStatus::Ok,
                    token_id,
                    token_index: absolute_index,
                    token_count: lane_token_count,
                    dispatch_kind: dispatch.kind,
                    dispatch_flags: dispatch.flags,
                    request_id: record.request_id,
                    sequence_id: record.sequence_id,
                    request_handle: record.request_handle,
                    ..ServingEvent::default()
                };
                if self.push_event(&event).is_err() {
                    return Err(ServingStatus::CapacityExceeded);
                }
                self.stats.decoded_token_count += 1;
            }
            self.records[record_index].streamed_decode_token_count += lane_token_count;
            if lane_finish && (finish_handle_count as usize) < finish_handles.len() {
                finish_handles[finish_handle_count as usize] = handle;
                finish_handle_count += 1;
            }
        }
        Ok(finish_handle_count)
    }

    /// `SparkServingCompleteFinishedHandles`.
    fn complete_finished_handles(
        &mut self,
        finish_handles: &[ServingRequestHandle],
    ) -> Result<(), ServingStatus> {
        for &request_handle in finish_handles {
            let Some(record_index) =
                find_record_by_handle(&self.records, &self.hash_heads, request_handle)
            else {
                continue;
            };
            let status = self.request_api.finish_request_generation(request_handle);
            if status != Ok(()) && status != Err(ServingStatus::NotFound) {
                return status;
            }
            self.records[record_index].state = RequestRecordState::Completed;
            self.push_simple_event(
                ServingEventKind::RequestCompleted,
                ServingStatus::Ok,
                Some(record_index),
            )?;
            self.stats.completed_stream_count += 1;
            if (self.flags & ENGINE_FLAG_AUTO_RELEASE_COMPLETED_REQUESTS) != 0 {
                let _ = self.release_completed_request(request_handle);
            }
        }
        Ok(())
    }

    /// `SparkServingCompleteBudgetFinishedRequests`.
    fn complete_budget_finished_requests(
        &mut self,
        dispatch: &Dispatch,
    ) -> Result<(), ServingStatus> {
        for request_index in 0..dispatch.request_count as usize {
            let handle = dispatch.request_handles[request_index];
            let Some(record_index) = find_record_by_handle(&self.records, &self.hash_heads, handle)
            else {
                continue;
            };
            let cache_state = match self.request_api.request_cache_state(handle) {
                Ok(cache_state) => cache_state,
                Err(ServingStatus::NotFound) => continue,
                Err(status) => return Err(status),
            };
            if cache_state.state == super::bridge::RequestApiState::Completed {
                self.records[record_index].state = RequestRecordState::Completed;
                self.push_simple_event(
                    ServingEventKind::RequestCompleted,
                    ServingStatus::Ok,
                    Some(record_index),
                )?;
                self.stats.completed_stream_count += 1;
                if (self.flags & ENGINE_FLAG_AUTO_RELEASE_COMPLETED_REQUESTS) != 0 {
                    let _ = self.release_completed_request(handle);
                }
            }
        }
        Ok(())
    }

    /// `SparkServingEngineCompleteDecodeDispatch`.
    pub fn complete_decode_dispatch(
        &mut self,
        dispatch: &mut Dispatch,
        decode_result: &mut ServingDecodeResult,
    ) -> Result<(), ServingStatus> {
        Self::validate_decode_result(dispatch, decode_result)?;
        self.resolve_speculative_decode(dispatch, decode_result)?;
        Self::clamp_speculative_verify_decode_result(dispatch, decode_result)?;
        let mut mtp_draft_token_ids =
            vec![0u32; MAX_DISPATCH_REQUEST_COUNT * MTP_MAX_DRAFT_TOKEN_COUNT];
        let mtp_draft_token_count = Self::capture_mtp_draft_tokens(
            dispatch,
            decode_result,
            &mut mtp_draft_token_ids,
            MTP_MAX_DRAFT_TOKEN_COUNT,
        )?;
        Self::resolve_mtp_decode(dispatch, decode_result)?;
        let mut finish_handles = vec![INVALID_REQUEST_HANDLE; MAX_DISPATCH_REQUEST_COUNT];
        let finish_handle_count =
            self.publish_decode_events(dispatch, decode_result, &mut finish_handles)?;
        self.stats.decode_dispatch_count += 1;
        if decode_result.lane_count > self.stats.maximum_decode_active_sequence_count {
            self.stats.maximum_decode_active_sequence_count = decode_result.lane_count;
        }
        if decode_result.lane_count > self.stats.maximum_decode_lane_count {
            self.stats.maximum_decode_lane_count = decode_result.lane_count;
        }
        self.request_api.complete_dispatch(dispatch)?;
        if mtp_draft_token_count != 0 {
            let status = self.request_api.arm_mtp_verify_dispatch(
                dispatch,
                &mtp_draft_token_ids,
                MTP_MAX_DRAFT_TOKEN_COUNT as u32,
                mtp_draft_token_count,
            );
            if status != Ok(()) && status != Err(ServingStatus::NotFound) {
                return status;
            }
            if status.is_ok() {
                self.stats.mtp_draft_token_count +=
                    mtp_draft_token_count as u64 * dispatch.request_count as u64;
            }
        }
        if (dispatch.flags & DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) != 0 {
            self.stats.mtp_verify_dispatch_count += 1;
        }
        self.complete_finished_handles(&finish_handles[..finish_handle_count as usize])?;
        self.complete_budget_finished_requests(dispatch)
    }

    /// `SparkServingEngineCompletePrefillDispatch`.
    pub fn complete_prefill_dispatch(
        &mut self,
        dispatch: &mut Dispatch,
    ) -> Result<(), ServingStatus> {
        if dispatch.kind != DispatchKind::Prefill && dispatch.kind != DispatchKind::PrefillBatch {
            return Err(ServingStatus::InvalidArgument);
        }
        self.request_api.complete_dispatch(dispatch)
    }
}

// ---------------------------------------------------------------------------
// Pump, events, stats, cancellation, release.
// ---------------------------------------------------------------------------

impl ServingEngine {
    /// Shared pump-exit epilogue: record `last_status`, refresh stats, copy
    /// them out, and map the status onto `Result`.
    fn pump_exit(
        &mut self,
        status: ServingStatus,
        stats_out: Option<&mut ServingStats>,
    ) -> Result<(), ServingStatus> {
        self.stats.last_status = status;
        self.refresh_stats();
        if let Some(stats_out) = stats_out {
            *stats_out = self.stats.clone();
        }
        if status.is_ok() {
            Ok(())
        } else {
            Err(status)
        }
    }

    /// `SparkServingEnginePump`.
    pub fn pump(
        &mut self,
        pump_flags: u32,
        max_dispatch_steps: u32,
        stats_out: Option<&mut ServingStats>,
    ) -> Result<(), ServingStatus> {
        let mut stats_out = stats_out;
        if (pump_flags & !PUMP_KNOWN_FLAGS) != 0 {
            return Err(ServingStatus::InvalidArgument);
        }

        let step_limit =
            if max_dispatch_steps != 0 { max_dispatch_steps } else { DEFAULT_MAX_PUMP_STEPS };
        let mut accepted_pending_dispatch_count = 0u32;
        for step_index in 0..step_limit {
            if self.event_ring_free_count() < event_ring_safety_margin() {
                return self.pump_exit(ServingStatus::Busy, stats_out.as_deref_mut());
            }

            let mut dispatch = match self.request_api.schedule_next() {
                Ok(dispatch) => dispatch,
                Err(status @ (ServingStatus::NotFound | ServingStatus::Busy)) => {
                    let status = if accepted_pending_dispatch_count != 0 {
                        ServingStatus::Pending
                    } else {
                        status
                    };
                    return self.pump_exit(status, stats_out.as_deref_mut());
                }
                Err(status) => {
                    return self.pump_exit(status, stats_out.as_deref_mut());
                }
            };
            if !dispatch.accepted || dispatch.kind == DispatchKind::None {
                return self.pump_exit(ServingStatus::Busy, stats_out.as_deref_mut());
            }

            let mut dispatch_was_completed_by_decode_path = false;
            let mut dispatch_was_retried = false;
            let status = if dispatch.kind == DispatchKind::Prefill
                || dispatch.kind == DispatchKind::PrefillBatch
            {
                let status = self.invoke_prefill(&dispatch, step_index);
                if status == Ok(()) {
                    self.complete_prefill_dispatch(&mut dispatch)
                } else {
                    status
                }
            } else if dispatch.kind == DispatchKind::DecodeBatch
                || dispatch.kind == DispatchKind::SpeculativeVerifyBatch
            {
                let status = self.invoke_decode(&mut dispatch);
                if status == Ok(()) {
                    dispatch_was_completed_by_decode_path = true;
                    status
                } else if status == Err(ServingStatus::Busy) {
                    let retry_status = self.request_api.retry_decode_dispatch(&mut dispatch);
                    if retry_status == Ok(()) {
                        dispatch_was_retried = true;
                        Err(ServingStatus::Busy)
                    } else {
                        retry_status
                    }
                } else {
                    status
                }
            } else {
                Err(ServingStatus::InvalidArgument)
            };

            if let Err(status) = status {
                if status == ServingStatus::Pending
                    && (dispatch.kind == DispatchKind::Prefill
                        || dispatch.kind == DispatchKind::PrefillBatch
                        || dispatch.kind == DispatchKind::DecodeBatch
                        || dispatch.kind == DispatchKind::SpeculativeVerifyBatch)
                {
                    accepted_pending_dispatch_count += 1;
                    if (pump_flags & PUMP_FLAG_STOP_AFTER_ONE_DISPATCH) != 0 {
                        return self.pump_exit(ServingStatus::Pending, stats_out.as_deref_mut());
                    }
                    continue;
                }
                if status == ServingStatus::Busy
                    && (dispatch.kind == DispatchKind::Prefill
                        || dispatch.kind == DispatchKind::PrefillBatch
                        || dispatch_was_retried)
                {
                    return self.pump_exit(status, stats_out.as_deref_mut());
                }
                if !dispatch_was_completed_by_decode_path && !dispatch_was_retried {
                    let _ = self.request_api.cancel_dispatch(&mut dispatch);
                    self.fail_dispatch_requests(&dispatch, status);
                }
                return self.pump_exit(status, stats_out.as_deref_mut());
            }

            if (pump_flags & PUMP_FLAG_STOP_AFTER_ONE_DISPATCH) != 0 {
                return self.pump_exit(ServingStatus::Ok, stats_out.as_deref_mut());
            }
        }

        let status = if accepted_pending_dispatch_count != 0 {
            ServingStatus::Pending
        } else {
            ServingStatus::Ok
        };
        self.pump_exit(status, stats_out)
    }

    /// `SparkServingEnginePopEvent`.
    pub fn pop_event(&mut self) -> Result<ServingEvent, ServingStatus> {
        if self.event_count == 0 {
            return Err(ServingStatus::NotFound);
        }
        let event = self.events[self.event_read_index];
        self.events[self.event_read_index] = ServingEvent::default();
        self.event_read_index += 1;
        if self.event_read_index == self.events.len() {
            self.event_read_index = 0;
        }
        self.event_count -= 1;
        self.refresh_stats();
        Ok(event)
    }

    /// `SparkServingEngineGetStats`.
    pub fn get_stats(&mut self) -> ServingStats {
        self.refresh_stats();
        self.stats.clone()
    }

    /// `SparkServingEngineCancelRequest`.
    pub fn cancel_request(
        &mut self,
        request_handle: ServingRequestHandle,
    ) -> Result<(), ServingStatus> {
        let record_index = find_record_by_handle(&self.records, &self.hash_heads, request_handle)
            .ok_or(ServingStatus::NotFound)?;
        self.request_api.cancel_request(request_handle)?;
        self.records[record_index].state = RequestRecordState::Cancelled;
        let status = self.push_simple_event(
            ServingEventKind::RequestCancelled,
            ServingStatus::Ok,
            Some(record_index),
        );
        self.refresh_stats();
        status
    }

    /// `SparkServingEngineFailRequestByRequestId`.
    pub fn fail_request_by_request_id(
        &mut self,
        request_id: u64,
        failure_status: ServingStatus,
    ) -> Result<(), ServingStatus> {
        if request_id == 0 {
            return Err(ServingStatus::InvalidArgument);
        }
        let record_index =
            find_record_by_request_id(&self.records, request_id).ok_or(ServingStatus::NotFound)?;
        let record = &self.records[record_index];
        if record.state == RequestRecordState::Completed
            || record.state == RequestRecordState::Cancelled
        {
            return Ok(());
        }
        let request_handle = record.request_handle;
        let _ = self.request_api.cancel_request(request_handle);
        self.records[record_index].state = RequestRecordState::Cancelled;
        let status = self.push_simple_event(
            ServingEventKind::RequestCancelled,
            failure_status,
            Some(record_index),
        );
        self.refresh_stats();
        status
    }

    /// `SparkServingEngineReleaseCompletedRequest`.
    pub fn release_completed_request(
        &mut self,
        request_handle: ServingRequestHandle,
    ) -> Result<(), ServingStatus> {
        let record_index = find_record_by_handle(&self.records, &self.hash_heads, request_handle)
            .ok_or(ServingStatus::NotFound)?;
        if let Some(release_sequence_function) = self.release_sequence_function.as_mut() {
            let record = &self.records[record_index];
            let token_count =
                record.prompt_token_count as u64 + record.streamed_decode_token_count as u64;
            if token_count == 0 || token_count > u32::MAX as u64 {
                return Err(ServingStatus::CapacityExceeded);
            }
            release_sequence_function(
                record.request_id,
                record.request_handle,
                record.sequence_id,
                token_count as u32,
            )?;
        }
        let status = self.request_api.release_completed_request(request_handle);
        if status != Ok(()) && status != Err(ServingStatus::NotFound) {
            return status;
        }
        let dynamic_storage = (self.flags & ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE) != 0;
        let keep_capacity =
            if dynamic_storage { 0 } else { self.records[record_index].token_capacity };
        remove_record_hash(&mut self.records, &mut self.hash_heads, record_index);
        let free_record_head = self.free_record_head;
        {
            let record = &mut self.records[record_index];
            record.reset(keep_capacity, !dynamic_storage);
            record.free_record_next = free_record_head;
        }
        self.free_record_head = record_index as u32;
        self.refresh_stats();
        Ok(())
    }
}
