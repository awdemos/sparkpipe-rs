//! Request-API boundary for the serving engine.
//!
//! `api/serving_engine.c` calls directly into the concrete `SparkRequestApi`
//! (api/request.c). That coupling is captured here as the
//! [`ServingRequestApi`] trait plus plain-data DTOs, so the engine is driven
//! by any implementation: the in-flight `request_api` port (via an adapter)
//! or the scripted fakes in the integration tests.
//!
//! The DTOs carry exactly the fields the serving engine reads or writes of
//! `SparkRequestApiDispatch`, `SparkRequestApiSubmitRequest`,
//! `SparkRequestApiCacheState`, `SparkRequestApiPrefillDispatchView`, and
//! `SparkRequestApiDecodeDispatchView` — no more. Scheduler-decision
//! internals of the C dispatch struct never cross this boundary.
//!
//! Trait methods map 1:1 onto `SparkRequestApi*` entry points; each doc
//! comment names its C counterpart.

pub use spark_text::prompt_pipeline::DispatchKind;

use super::status::ServingStatus;

/// `SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT`
/// (= `SPARK_SCHEDULER_MAX_PACKED_REQUEST_COUNT`
/// = `SPARK_STAGE_PLAN_MAX_BATCH_BUCKET`).
pub const MAX_DISPATCH_REQUEST_COUNT: usize = spark_sched::stage_plan::MAX_BATCH_BUCKET as usize;
/// `SPARK_REQUEST_MODEL_MAX_SPECULATIVE_TOKENS`.
pub const MAX_SPECULATIVE_TOKENS: usize = 7;
/// `SPARK_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT`.
pub const MTP_MAX_DRAFT_TOKEN_COUNT: usize = 8;
/// `SPARK_SERVING_MAX_DECODE_TOKENS_PER_LANE`.
pub const MAX_DECODE_TOKENS_PER_LANE: usize = MAX_SPECULATIVE_TOKENS + 1;
/// `SPARK_REQUEST_API_INVALID_HANDLE`.
pub const INVALID_REQUEST_HANDLE: u64 = 0;

/// `SPARK_REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH`.
pub const CONFIGURATION_FLAG_JIT_KV_PREFETCH: u32 = 0x0000_0001;
/// `SPARK_REQUEST_API_CONFIGURATION_FLAG_DECODE_BATCHING`.
pub const CONFIGURATION_FLAG_DECODE_BATCHING: u32 = 0x0000_0002;
/// `SPARK_REQUEST_API_CONFIGURATION_FLAG_PREFILL_BATCHING`.
pub const CONFIGURATION_FLAG_PREFILL_BATCHING: u32 = 0x0000_0008;
/// `SPARK_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT`.
pub const CONFIGURATION_FLAG_MTP_COMMIT: u32 = 0x0000_0080;

/// `SPARK_REQUEST_API_REQUEST_FLAG_REALTIME`.
pub const REQUEST_FLAG_REALTIME: u32 = 0x0000_0001;
/// `SPARK_REQUEST_API_REQUEST_FLAG_DISABLE_SPECULATION`.
pub const REQUEST_FLAG_DISABLE_SPECULATION: u32 = 0x0000_0002;

/// `SPARK_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT`.
pub const DISPATCH_FLAG_MTP_COMMIT: u32 = 0x0000_0200;
/// `SPARK_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY`.
pub const DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY: u32 = 0x0000_0400;
/// `SPARK_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY`.
pub const DISPATCH_FLAG_MTP_TREE_VERIFY: u32 = 0x0000_0800;

/// Request slot state (`SPARK_REQUEST_API_STATE_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RequestApiState {
    /// `SPARK_REQUEST_API_STATE_FREE`.
    #[default]
    Free = 0,
    /// `SPARK_REQUEST_API_STATE_QUEUED_PREFILL`.
    QueuedPrefill = 1,
    /// `SPARK_REQUEST_API_STATE_RUNNING_PREFILL`.
    RunningPrefill = 2,
    /// `SPARK_REQUEST_API_STATE_READY_DECODE`.
    ReadyDecode = 3,
    /// `SPARK_REQUEST_API_STATE_RUNNING_DECODE`.
    RunningDecode = 4,
    /// `SPARK_REQUEST_API_STATE_COMPLETED`.
    Completed = 5,
    /// `SPARK_REQUEST_API_STATE_CANCELLED`.
    Cancelled = 6,
    /// `SPARK_REQUEST_API_STATE_WAITING_PREFIX_COHORT`.
    WaitingPrefixCohort = 7,
    /// `SPARK_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY`.
    ReadySpeculativeVerify = 8,
    /// `SPARK_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY`.
    RunningSpeculativeVerify = 9,
}

/// Submit request passed to [`ServingRequestApi::submit`]
/// (port of `SparkRequestApiSubmitRequest`).
#[derive(Debug, Clone, Copy)]
pub struct ApiSubmitRequest<'a> {
    /// `SPARK_REQUEST_API_REQUEST_FLAG_*` bits.
    pub flags: u32,
    /// Scheduling priority.
    pub priority: u32,
    /// Prompt tokens in `prompt_token_ids`.
    pub prompt_token_count: u32,
    /// Thinking-phase token budget.
    pub thinking_token_budget: u32,
    /// Output token budget.
    pub output_token_budget: u32,
    /// Maximum prefill tokens per step (0 = engine/scheduler default).
    pub max_prefill_tokens_per_step: u32,
    /// Caller request id.
    pub request_id: u64,
    /// Caller sequence id (0 = request API assigns one).
    pub sequence_id: u64,
    /// Prompt token ids (`prompt_token_count` words).
    pub prompt_token_ids: &'a [u32],
}

/// Per-request cache state snapshot (port of the fields of
/// `SparkRequestApiCacheState` the serving engine reads).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestCacheState {
    /// Slot state.
    pub state: RequestApiState,
    /// Caller request id recorded at submit.
    pub request_id: u64,
    /// Assigned sequence id (non-zero for a live request).
    pub sequence_id: u64,
}

/// Scheduled dispatch (port of the fields of `SparkRequestApiDispatch` the
/// serving engine touches). Fixed C arrays become `Vec`s sized to
/// [`MAX_DISPATCH_REQUEST_COUNT`] (draft arrays are
/// `MAX_DISPATCH_REQUEST_COUNT * MAX_SPECULATIVE_TOKENS`).
#[derive(Debug, Clone)]
pub struct Dispatch {
    /// Non-zero when the scheduler accepted the dispatch.
    pub accepted: bool,
    /// Dispatch kind.
    pub kind: DispatchKind,
    /// `SPARK_REQUEST_API_DISPATCH_FLAG_*` bits.
    pub flags: u32,
    /// Lanes in this dispatch.
    pub request_count: u32,
    /// Per-lane request handles.
    pub request_handles: Vec<u64>,
    /// Per-lane caller request ids (diagnostics only, as in C).
    pub request_ids: Vec<u64>,
    /// Draft tokens per lane for speculative-verify dispatches.
    pub speculative_token_count: u32,
    /// Verifier tokens per lane for speculative-verify dispatches.
    pub speculative_verifier_token_count: u32,
    /// Out: per-lane committed token counts resolved by verification.
    pub speculative_committed_token_counts: Vec<u32>,
    /// Out: per-lane accepted draft token counts resolved by verification.
    pub speculative_accepted_token_counts: Vec<u32>,
    /// Out: per-lane fallback token ids resolved by verification.
    pub speculative_fallback_token_ids: Vec<u32>,
    /// Per-lane draft token ids (flat, lane stride
    /// [`MAX_SPECULATIVE_TOKENS`]).
    pub speculative_draft_token_ids: Vec<u32>,
    /// MTP draft token budget carried by MTP-commit decode dispatches.
    pub mtp_draft_token_budget: u32,
    /// Out: per-lane committed token counts for MTP-commit decodes.
    pub decode_committed_token_counts: Vec<u32>,
}

impl Dispatch {
    /// A zeroed dispatch (`memset` in C), with all arrays at full capacity.
    pub fn new() -> Self {
        Dispatch {
            accepted: false,
            kind: DispatchKind::None,
            flags: 0,
            request_count: 0,
            request_handles: vec![0; MAX_DISPATCH_REQUEST_COUNT],
            request_ids: vec![0; MAX_DISPATCH_REQUEST_COUNT],
            speculative_token_count: 0,
            speculative_verifier_token_count: 0,
            speculative_committed_token_counts: vec![0; MAX_DISPATCH_REQUEST_COUNT],
            speculative_accepted_token_counts: vec![0; MAX_DISPATCH_REQUEST_COUNT],
            speculative_fallback_token_ids: vec![0; MAX_DISPATCH_REQUEST_COUNT],
            speculative_draft_token_ids: vec![
                0;
                MAX_DISPATCH_REQUEST_COUNT * MAX_SPECULATIVE_TOKENS
            ],
            mtp_draft_token_budget: 0,
            decode_committed_token_counts: vec![0; MAX_DISPATCH_REQUEST_COUNT],
        }
    }
}

impl Default for Dispatch {
    fn default() -> Self {
        Dispatch::new()
    }
}

/// Per-lane prefill view (port of `SparkRequestApiPrefillDispatchLaneView`,
/// minus the redundant `prompt_token_ids` pointer — staged tokens travel in
/// the host buffer instead).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillDispatchLaneView {
    /// Lane index within the dispatch.
    pub request_index: u32,
    /// First prompt token this lane covers.
    pub prompt_token_offset: u32,
    /// Prompt tokens this lane covers.
    pub prompt_token_count: u32,
    /// Request slot index in the request API.
    pub request_slot_index: u32,
    /// Caller request id.
    pub request_id: u64,
    /// Assigned sequence id.
    pub sequence_id: u64,
    /// Request handle.
    pub request_handle: u64,
}

/// Prefill dispatch view (port of `SparkRequestApiPrefillDispatchView`).
#[derive(Debug, Clone)]
pub struct PrefillDispatchView {
    /// Dispatch kind the view was built from.
    pub kind: DispatchKind,
    /// Active sequences in this dispatch.
    pub active_sequence_count: u32,
    /// First prompt token covered by this step.
    pub prompt_token_offset: u32,
    /// Prompt tokens covered by this step (per lane).
    pub prompt_token_count: u32,
    /// Per-lane prompt token stride used by the dispatch.
    pub prompt_token_stride: u32,
    /// Lanes in this dispatch.
    pub lane_count: u32,
    /// Per-lane views (`lane_count` entries are meaningful).
    pub lanes: Vec<PrefillDispatchLaneView>,
}

/// Per-lane decode view (port of `SparkRequestApiDecodeDispatchLaneView`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeDispatchLaneView {
    /// Lane index within the dispatch.
    pub request_index: u32,
    /// Position of the lane's input token within its sequence.
    pub sequence_position: u32,
    /// Tokens in the lane's context (input position + 1).
    pub context_token_count: u32,
    /// Request slot index in the request API.
    pub request_slot_index: u32,
    /// Caller request id.
    pub request_id: u64,
    /// Assigned sequence id.
    pub sequence_id: u64,
    /// Request handle.
    pub request_handle: u64,
    /// MTP resolution bookkeeping (carried verbatim, as in C).
    pub mtp_resolution_base_position: u64,
    /// MTP resolution bookkeeping.
    pub mtp_resolution_proposed_token_count: u32,
    /// MTP resolution bookkeeping.
    pub mtp_resolution_accepted_token_count: u32,
    /// MTP resolution bookkeeping.
    pub mtp_resolution_committed_token_count: u32,
    /// MTP resolution bookkeeping.
    pub mtp_resolution_path_id: u32,
}

/// Decode dispatch view (port of `SparkRequestApiDecodeDispatchView`).
#[derive(Debug, Clone)]
pub struct DecodeDispatchView {
    /// Dispatch kind the view was built from.
    pub kind: DispatchKind,
    /// Active sequences in this dispatch.
    pub active_sequence_count: u32,
    /// Lanes in this dispatch.
    pub lane_count: u32,
    /// Draft tokens per lane for speculative-verify dispatches.
    pub speculative_token_count: u32,
    /// Per-lane views (`lane_count` entries are meaningful).
    pub lanes: Vec<DecodeDispatchLaneView>,
}

/// Counters the serving engine mirrors into its own stats (the
/// `engine->request_api->*` reads in `SparkServingRefreshStats`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RequestApiCounters {
    /// `queued_request_count`.
    pub queued_request_count: u32,
    /// `completed_request_count`.
    pub completed_request_count: u32,
    /// `cancelled_request_count`.
    pub cancelled_request_count: u32,
    /// `jit_prefetch_dispatch_count`.
    pub jit_prefetch_dispatch_count: u64,
    /// `jit_prefetch_block_count`.
    pub jit_prefetch_block_count: u64,
    /// `async_jit_prefetch_start_count`.
    pub async_jit_prefetch_start_count: u64,
    /// `async_jit_prefetch_poll_count`.
    pub async_jit_prefetch_poll_count: u64,
    /// `async_jit_prefetch_completion_count`.
    pub async_jit_prefetch_completion_count: u64,
    /// `prefix_family_dispatch_count`.
    pub prefix_family_dispatch_count: u64,
    /// `prefix_family_member_count`.
    pub prefix_family_member_count: u64,
    /// `prefix_family_saved_prompt_token_count`.
    pub prefix_family_saved_prompt_token_count: u64,
    /// `mtp_draft_ready_count`.
    pub mtp_draft_ready_count: u64,
    /// `mtp_accepted_draft_token_count`.
    pub mtp_accepted_draft_token_count: u64,
    /// `mtp_committed_token_count`.
    pub mtp_committed_token_count: u64,
    /// `mtp_rejected_token_count`.
    pub mtp_rejected_token_count: u64,
}

/// KV block table view handed to the prefill/decode callbacks (the serving
/// engine passes `SparkKvBlockTableView` through unchanged).
pub type KvBlockTableView<'a> = spark_text::prompt_pipeline::KvBlockTableView<'a>;

/// The request-API seam used by the serving engine (port of the
/// `SparkRequestApi*` calls `api/serving_engine.c` makes).
///
/// The in-flight `request_api` port should be adapted onto this trait rather
/// than the engine depending on its concrete types; see the integration
/// tests for a reference implementation (`FakeRequestApi`).
pub trait ServingRequestApi {
    /// `request_api->configuration_flags` (used by the runtime-contract
    /// cross-check in configuration validation).
    fn configuration_flags(&self) -> u32;
    /// `SparkServingRefreshStats`'s reads of the request-API counters.
    fn counters(&self) -> RequestApiCounters;
    /// `SparkRequestApiSubmit`. Returns the request handle.
    fn submit(&mut self, request: &ApiSubmitRequest) -> Result<u64, ServingStatus>;
    /// `SparkRequestApiGetRequestCacheState`.
    fn request_cache_state(&self, handle: u64) -> Result<RequestCacheState, ServingStatus>;
    /// `SparkRequestApiCancelRequest`.
    fn cancel_request(&mut self, handle: u64) -> Result<(), ServingStatus>;
    /// `SparkRequestApiReleaseCompletedRequest`.
    fn release_completed_request(&mut self, handle: u64) -> Result<(), ServingStatus>;
    /// `SparkRequestApiFinishRequestGeneration`.
    fn finish_request_generation(&mut self, handle: u64) -> Result<(), ServingStatus>;
    /// `SparkRequestApiScheduleNext`. `Err(NotFound)`/`Err(Busy)` mean "no
    /// dispatch right now", exactly as in C.
    fn schedule_next(&mut self) -> Result<Dispatch, ServingStatus>;
    /// `SparkRequestApiDescribePrefillDispatch`.
    fn describe_prefill_dispatch(
        &self,
        dispatch: &Dispatch,
    ) -> Result<PrefillDispatchView, ServingStatus>;
    /// `SparkRequestApiCopyPrefillDispatchTokenIds`.
    ///
    /// `destination_token_ids` is the whole staging buffer
    /// (`destination_lane_capacity * destination_token_stride` words); the
    /// implementation writes each lane's tokens and zero-fills the per-lane
    /// padding, exactly as the C does.
    fn copy_prefill_dispatch_token_ids(
        &self,
        dispatch: &Dispatch,
        destination_token_ids: &mut [u32],
        destination_token_stride: u32,
        destination_lane_capacity: u32,
    ) -> Result<(), ServingStatus>;
    /// `SparkRequestApiBuildDispatchKvBlockTableView`.
    ///
    /// The C also took `lane_count_capacity`; here the slice lengths bound
    /// the writable lanes instead.
    fn build_dispatch_kv_block_table_view<'a>(
        &self,
        dispatch: &Dispatch,
        host_physical_block_indices: &'a mut [u32],
        execution_physical_block_indices: Option<&'a [u32]>,
        lane_stride: u32,
        lane_capacity: u32,
        lane_physical_block_counts: &'a mut [u32],
    ) -> Result<KvBlockTableView<'a>, ServingStatus>;
    /// `SparkRequestApiDescribeDecodeDispatch`.
    fn describe_decode_dispatch(
        &self,
        dispatch: &Dispatch,
    ) -> Result<DecodeDispatchView, ServingStatus>;
    /// `SparkRequestApiResolveSpeculativeVerifyDispatch`.
    ///
    /// `verifier_token_ids` is flat with per-lane stride
    /// `verifier_lane_stride`; resolves acceptance into the dispatch's
    /// `speculative_*` out-arrays.
    fn resolve_speculative_verify_dispatch(
        &mut self,
        dispatch: &mut Dispatch,
        verifier_token_ids: &[u32],
        verifier_lane_stride: u32,
        verifier_token_count: u32,
    ) -> Result<(), ServingStatus>;
    /// `SparkRequestApiCompleteDispatch`.
    fn complete_dispatch(&mut self, dispatch: &mut Dispatch) -> Result<(), ServingStatus>;
    /// `SparkRequestApiArmMtpVerifyDispatch`.
    ///
    /// `draft_token_ids` is flat with per-lane stride `draft_lane_stride`;
    /// `Err(NotFound)` means "no lane could be armed" and is tolerated by
    /// the engine, as in C.
    fn arm_mtp_verify_dispatch(
        &mut self,
        dispatch: &mut Dispatch,
        draft_token_ids: &[u32],
        draft_lane_stride: u32,
        draft_token_count: u32,
    ) -> Result<(), ServingStatus>;
    /// `SparkRequestApiRetryDecodeDispatch` (re-queue a decode the decode
    /// callback bounced with `Busy` before accepting).
    fn retry_decode_dispatch(&mut self, dispatch: &mut Dispatch) -> Result<(), ServingStatus>;
    /// `SparkRequestApiCancelDispatch`.
    fn cancel_dispatch(&mut self, dispatch: &mut Dispatch) -> Result<(), ServingStatus>;
}
