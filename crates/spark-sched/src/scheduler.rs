//! Request scheduler — port of `scheduler/scheduler.c`
//! (API declared in `include/sparkpipe/spark_scheduler.h`).
//!
//! Admission pricing against a measured/uniform stage plan, chunked prefill
//! bounded by `max_prefill_tokens_per_step`, prefix-cache
//! probe/reserve/commit/cancel choreography, CUDA-graph padding buckets with
//! measured decode bucket selection, prefill/decode interleave with a
//! reserved decode slot, decode bypass, and adaptive decode/prefill packing.
//!
//! Rust-port deviations from the C surface (behavior is otherwise
//! signature-faithful):
//!   - No GLM52 constants are baked in. The C aliased
//!     `SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH` (256) into
//!     `SPARK_SCHEDULER_DEFAULT_MAX_PREFILL_TOKENS_PER_STEP` and
//!     `SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS` (1M) into
//!     `SPARK_SCHEDULER_MAX_CONTEXT_TOKENS` /
//!     `SPARK_SCHEDULER_KV_BLOCK_TABLE_CAPACITY`; here they are explicit
//!     [`SchedulerConfig`] fields (`default_max_prefill_tokens_per_step`,
//!     `max_context_tokens`). `spark_count` and `max_batch_bucket` are also
//!     config fields replacing `SPARK_SCHEDULER_MAX_SPARK_COUNT` /
//!     `SPARK_SCHEDULER_MAX_PACKED_REQUEST_COUNT`; they are validated to
//!     equal [`stage_plan::CURRENT_SPARK_COUNT`] / [`stage_plan::MAX_BATCH_BUCKET`]
//!     because stage planning still targets the current production ring.
//!   - ABI plumbing (`abi_version`, `descriptor_bytes`, reserved words) is
//!     dropped; the C validation checks against those fields have no Rust
//!     counterpart. `quantization_mode` is the [`QuantizationMode`] enum, so
//!     the C "unknown mode" rejection (e.g. mode 99) is unrepresentable.
//!   - The prefix cache is owned: [`SchedulerConfig::prefix_cache`] takes
//!     `Option<PrefixCache>` by value (required to be `Some` when
//!     [`CONFIGURATION_FLAG_PREFIX_CACHE`] is set; the Rust
//!     [`PrefixCache`](spark_core::prefix_cache::PrefixCache) always owns a
//!     KV arena, so [`CONFIGURATION_FLAG_KV_CACHE_REQUIRED`] adds no check
//!     beyond presence). Callers drive the cache between scheduler calls
//!     through [`Scheduler::prefix_cache`] / [`Scheduler::prefix_cache_mut`].
//!   - Decisions own `Vec`s instead of the C's embedded fixed arrays
//!     (`kv_physical_block_indices[SPARK_SCHEDULER_KV_BLOCK_TABLE_CAPACITY]`
//!     = 64K entries, `dispatch_stages[SPARK_SCHEDULER_MAX_SPARK_COUNT]`,
//!     `packed_requests`/`lanes[SPARK_SCHEDULER_MAX_PACKED_REQUEST_COUNT]`)
//!     and copy the prompt token ids the C only borrows.
//!   - Rejected admissions mirror the C's `SPARK_STATUS_OK` returns as
//!     `Ok(decision)` with `accepted == false` and `rejected_status` set;
//!     `Err` mirrors the C's non-OK returns, where the C wrote the decision
//!     but the caller is not expected to consume it (batch-entry validation
//!     failures still produce a rejected decision in C; here only the error
//!     is observable).
//!   - `rejected_status: Option<SchedulerError>` replaces the C
//!     `rejected_status` status word (`None` == `SPARK_STATUS_OK`).

use spark_core::prefix_cache::{
    hash_prompt_tokens, PrefixCache, PrefixCacheError, Reservation, EMPTY_PARENT_HASH,
};

use crate::stage_plan::{
    self, QuantizationMode, StagePlan, StagePlanError, StagePlanGeometry, StagePlanStage,
};

/// Scheduler ABI version (`SPARK_SCHEDULER_ABI_VERSION`).
pub const ABI_VERSION: u32 = 1;
/// Default per-spark in-flight queue depth
/// (`SPARK_SCHEDULER_DEFAULT_QUEUE_DEPTH_PER_SPARK`).
pub const DEFAULT_QUEUE_DEPTH_PER_SPARK: u32 = 1;
/// Tokens per prefill/KV block (`SPARK_SCHEDULER_PREFILL_BLOCK_TOKENS`).
pub const PREFILL_BLOCK_TOKENS: u32 = 16;
/// Batch-request capacity factor over the packed-request count
/// (`SPARK_SCHEDULER_BATCH_REQUEST_CAPACITY_FACTOR`).
pub const BATCH_REQUEST_CAPACITY_FACTOR: u32 = 2;
/// Request field value meaning "no per-request prompt scheduling limit"
/// (`SPARK_SCHEDULER_NO_PROMPT_LIMIT`).
pub const NO_PROMPT_LIMIT: u32 = 0;

/// Configuration flag: chunked prefill
/// (`SPARK_SCHEDULER_CONFIGURATION_FLAG_CHUNKED_PREFILL`).
pub const CONFIGURATION_FLAG_CHUNKED_PREFILL: u32 = 0x0000_0001;
/// Configuration flag: prefix cache integration
/// (`SPARK_SCHEDULER_CONFIGURATION_FLAG_PREFIX_CACHE`).
pub const CONFIGURATION_FLAG_PREFIX_CACHE: u32 = 0x0000_0002;
/// Configuration flag: CUDA-graph padding
/// (`SPARK_SCHEDULER_CONFIGURATION_FLAG_CUDAGRAPH_PADDING`).
pub const CONFIGURATION_FLAG_CUDAGRAPH_PADDING: u32 = 0x0000_0004;
/// Configuration flag: prefill/decode interleave
/// (`SPARK_SCHEDULER_CONFIGURATION_FLAG_PREFILL_DECODE_INTERLEAVE`).
pub const CONFIGURATION_FLAG_PREFILL_DECODE_INTERLEAVE: u32 = 0x0000_0008;
/// Configuration flag: KV cache arena required
/// (`SPARK_SCHEDULER_CONFIGURATION_FLAG_KV_CACHE_REQUIRED`).
pub const CONFIGURATION_FLAG_KV_CACHE_REQUIRED: u32 = 0x0000_0010;
/// Configuration flag: measured decode bucket selection
/// (`SPARK_SCHEDULER_CONFIGURATION_FLAG_MEASURED_DECODE_BUCKET_SELECTION`).
pub const CONFIGURATION_FLAG_MEASURED_DECODE_BUCKET_SELECTION: u32 = 0x0000_0020;
/// Configuration flag: cross-sequence prefix reuse
/// (`SPARK_SCHEDULER_CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE`).
pub const CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE: u32 = 0x0000_0040;
/// Default configuration flags (`SPARK_SCHEDULER_CONFIGURATION_DEFAULT_FLAGS`).
pub const CONFIGURATION_DEFAULT_FLAGS: u32 = CONFIGURATION_FLAG_CHUNKED_PREFILL
    | CONFIGURATION_FLAG_PREFIX_CACHE
    | CONFIGURATION_FLAG_CUDAGRAPH_PADDING
    | CONFIGURATION_FLAG_PREFILL_DECODE_INTERLEAVE
    | CONFIGURATION_FLAG_KV_CACHE_REQUIRED
    | CONFIGURATION_FLAG_MEASURED_DECODE_BUCKET_SELECTION
    | CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE;
/// All recognized configuration flags
/// (`SPARK_SCHEDULER_CONFIGURATION_KNOWN_FLAGS`).
pub const CONFIGURATION_KNOWN_FLAGS: u32 = CONFIGURATION_DEFAULT_FLAGS;

/// Request flag: decode step (`SPARK_SCHEDULER_REQUEST_FLAG_DECODE`).
pub const REQUEST_FLAG_DECODE: u32 = 0x0000_0001;
/// Request flag: prefill step (`SPARK_SCHEDULER_REQUEST_FLAG_PREFILL`).
pub const REQUEST_FLAG_PREFILL: u32 = 0x0000_0002;
/// All recognized request flags (`SPARK_SCHEDULER_REQUEST_KNOWN_FLAGS`).
pub const REQUEST_KNOWN_FLAGS: u32 = REQUEST_FLAG_DECODE | REQUEST_FLAG_PREFILL;

/// Decision flag: this step is a prefill chunk
/// (`SPARK_SCHEDULER_DECISION_FLAG_PREFILL_CHUNK`).
pub const DECISION_FLAG_PREFILL_CHUNK: u32 = 0x0000_0001;
/// Decision flag: this step is the final prefill chunk
/// (`SPARK_SCHEDULER_DECISION_FLAG_PREFILL_FINAL_CHUNK`).
pub const DECISION_FLAG_PREFILL_FINAL_CHUNK: u32 = 0x0000_0002;
/// Decision flag: the prefix cache supplied tokens
/// (`SPARK_SCHEDULER_DECISION_FLAG_PREFIX_CACHE_USED`).
pub const DECISION_FLAG_PREFIX_CACHE_USED: u32 = 0x0000_0004;
/// Decision flag: the batch was padded to a CUDA-graph bucket
/// (`SPARK_SCHEDULER_DECISION_FLAG_CUDAGRAPH_PADDING`).
pub const DECISION_FLAG_CUDAGRAPH_PADDING: u32 = 0x0000_0008;
/// Decision flag: decode step (`SPARK_SCHEDULER_DECISION_FLAG_DECODE_STEP`).
pub const DECISION_FLAG_DECODE_STEP: u32 = 0x0000_0010;
/// Decision flag: prefill step (`SPARK_SCHEDULER_DECISION_FLAG_PREFILL_STEP`).
pub const DECISION_FLAG_PREFILL_STEP: u32 = 0x0000_0020;
/// Decision flag: prefill admitted against a reserved decode slot
/// (`SPARK_SCHEDULER_DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT`).
pub const DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT: u32 = 0x0000_0040;
/// Decision flag: decode bypassed a queued prefill
/// (`SPARK_SCHEDULER_DECISION_FLAG_DECODE_BYPASS_PREFILL`).
pub const DECISION_FLAG_DECODE_BYPASS_PREFILL: u32 = 0x0000_0080;
/// Decision flag: adaptive decode pack
/// (`SPARK_SCHEDULER_DECISION_FLAG_ADAPTIVE_DECODE_PACK`).
pub const DECISION_FLAG_ADAPTIVE_DECODE_PACK: u32 = 0x0000_0100;
/// Decision flag: adaptive prefill pack
/// (`SPARK_SCHEDULER_DECISION_FLAG_ADAPTIVE_PREFILL_PACK`).
pub const DECISION_FLAG_ADAPTIVE_PREFILL_PACK: u32 = 0x0000_0200;
/// Decision flag: bucket chosen by measured decode bucket selection
/// (`SPARK_SCHEDULER_DECISION_FLAG_MEASURED_DECODE_BUCKET`).
pub const DECISION_FLAG_MEASURED_DECODE_BUCKET: u32 = 0x0000_0400;

/// Dispatch-stage flag: decode (`SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_DECODE`).
pub const DISPATCH_STAGE_FLAG_DECODE: u32 = 0x0000_0001;
/// Dispatch-stage flag: prefill (`SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL`).
pub const DISPATCH_STAGE_FLAG_PREFILL: u32 = 0x0000_0002;
/// Dispatch-stage flag: prefill chunk
/// (`SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL_CHUNK`).
pub const DISPATCH_STAGE_FLAG_PREFILL_CHUNK: u32 = 0x0000_0004;
/// Dispatch-stage flag: final prefill chunk
/// (`SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL_FINAL_CHUNK`).
pub const DISPATCH_STAGE_FLAG_PREFILL_FINAL_CHUNK: u32 = 0x0000_0008;
/// Dispatch-stage flag: CUDA-graph padding
/// (`SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_CUDAGRAPH_PADDING`).
pub const DISPATCH_STAGE_FLAG_CUDAGRAPH_PADDING: u32 = 0x0000_0010;
/// Dispatch-stage flag: prefill reserved decode slot
/// (`SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL_RESERVED_DECODE_SLOT`).
pub const DISPATCH_STAGE_FLAG_PREFILL_RESERVED_DECODE_SLOT: u32 = 0x0000_0020;
/// Dispatch-stage flag: decode bypass prefill
/// (`SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_DECODE_BYPASS_PREFILL`).
pub const DISPATCH_STAGE_FLAG_DECODE_BYPASS_PREFILL: u32 = 0x0000_0040;
/// Dispatch-stage flag: adaptive decode pack
/// (`SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_ADAPTIVE_DECODE_PACK`).
pub const DISPATCH_STAGE_FLAG_ADAPTIVE_DECODE_PACK: u32 = 0x0000_0080;
/// Dispatch-stage flag: adaptive prefill pack
/// (`SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_ADAPTIVE_PREFILL_PACK`).
pub const DISPATCH_STAGE_FLAG_ADAPTIVE_PREFILL_PACK: u32 = 0x0000_0100;
/// Dispatch-stage flag: measured decode bucket
/// (`SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_MEASURED_DECODE_BUCKET`).
pub const DISPATCH_STAGE_FLAG_MEASURED_DECODE_BUCKET: u32 = 0x0000_0200;

/// Errors mirroring the `SparkStatus` codes returned by the C entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SchedulerError {
    /// `SPARK_STATUS_INVALID_ARGUMENT`.
    #[error("invalid argument")]
    InvalidArgument,
    /// `SPARK_STATUS_CAPACITY_EXCEEDED`.
    #[error("capacity exceeded")]
    CapacityExceeded,
    /// `SPARK_STATUS_NOT_FOUND`.
    #[error("not found")]
    NotFound,
    /// `SPARK_STATUS_BUSY`.
    #[error("busy")]
    Busy,
    /// `SPARK_STATUS_DUPLICATE`.
    #[error("duplicate")]
    Duplicate,
    /// `SPARK_STATUS_MODULE_NOT_VALIDATED`.
    #[error("module not validated")]
    ModuleNotValidated,
    /// `SPARK_STATUS_INTERNAL_ERROR`.
    #[error("internal error")]
    InternalError,
}

impl From<PrefixCacheError> for SchedulerError {
    fn from(error: PrefixCacheError) -> Self {
        match error {
            PrefixCacheError::InvalidArgument => SchedulerError::InvalidArgument,
            PrefixCacheError::CapacityExceeded => SchedulerError::CapacityExceeded,
            PrefixCacheError::NotFound => SchedulerError::NotFound,
            PrefixCacheError::Busy => SchedulerError::Busy,
            PrefixCacheError::Duplicate => SchedulerError::Duplicate,
            PrefixCacheError::ModuleNotValidated => SchedulerError::ModuleNotValidated,
            PrefixCacheError::InternalError => SchedulerError::InternalError,
        }
    }
}

impl From<StagePlanError> for SchedulerError {
    fn from(error: StagePlanError) -> Self {
        match error {
            StagePlanError::InvalidArgument(_) => SchedulerError::InvalidArgument,
            StagePlanError::CapacityExceeded(_) => SchedulerError::CapacityExceeded,
            // The C scheduler never surfaces SPARK_STATUS_ABI_MISMATCH; the
            // stage-plan builders it calls cannot produce it either.
            StagePlanError::AbiMismatch(_) | StagePlanError::InternalError(_) => {
                SchedulerError::InternalError
            }
        }
    }
}

/// Scheduler configuration (`SparkSchedulerConfiguration`). See module docs
/// for the config-fields-instead-of-GLM52-constants deviation.
#[derive(Debug)]
pub struct SchedulerConfig {
    /// Spark count of the ring; must equal [`stage_plan::CURRENT_SPARK_COUNT`]
    /// (the C requires `SPARK_SCHEDULER_MAX_SPARK_COUNT`).
    pub spark_count: u32,
    /// In-flight requests per spark; 0 normalizes to
    /// [`DEFAULT_QUEUE_DEPTH_PER_SPARK`].
    pub queue_depth_per_spark: u32,
    /// Measured cost profile id (`stage_plan::MEASURED_PROFILE_*`,
    /// [`stage_plan::PROFILE_UNIFORM_ESTIMATED`]).
    pub measured_profile_id: u32,
    /// The model's layer geometry.
    pub stage_geometry: StagePlanGeometry,
    /// Uniform per-layer estimate, used only with
    /// [`stage_plan::PROFILE_UNIFORM_ESTIMATED`].
    pub estimated_layer_cost_ns: u64,
    /// Uniform final-stage extra estimate, used only with
    /// [`stage_plan::PROFILE_UNIFORM_ESTIMATED`].
    pub estimated_final_stage_extra_cost_ns: u64,
    /// Quantization mode; `Auto` normalizes to `Nvfp4_4Bit` at initialize.
    pub quantization_mode: QuantizationMode,
    /// Per-step prefill token cap; 0 normalizes to
    /// `default_max_prefill_tokens_per_step`.
    pub max_prefill_tokens_per_step: u32,
    /// Fallback for `max_prefill_tokens_per_step == 0`; replaces the C's
    /// baked-in `SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH` (256).
    pub default_max_prefill_tokens_per_step: u32,
    /// Maximum context in tokens; replaces the C's baked-in
    /// `SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS` (1M). In C this only sized
    /// the embedded KV block table (`max_context_tokens /
    /// PREFILL_BLOCK_TOKENS` entries); the Rust decision owns a `Vec`, so
    /// this is advisory and validated nonzero only.
    pub max_context_tokens: u32,
    /// Largest batch bucket / packed-request capacity; must equal
    /// [`stage_plan::MAX_BATCH_BUCKET`] (the C derives
    /// `SPARK_SCHEDULER_MAX_PACKED_REQUEST_COUNT` from it).
    pub max_batch_bucket: u32,
    /// Tokens per prefix-cache block; must match the cache's block size.
    pub prefix_cache_block_tokens: u32,
    /// `CONFIGURATION_FLAG_*` mask; must be nonzero and known.
    pub configuration_flags: u32,
    /// Owned prefix cache; required to be `Some` when
    /// [`CONFIGURATION_FLAG_PREFIX_CACHE`] is set.
    pub prefix_cache: Option<PrefixCache>,
}

/// One scheduling request (`SparkSchedulerRequest`). Borrows the prompt token
/// ids; decisions copy them.
#[derive(Debug, Clone)]
pub struct SchedulerRequest<'a> {
    pub active_sequence_count: u32,
    pub prompt_token_count: u32,
    /// `REQUEST_FLAG_*` mask; exactly one of decode/prefill.
    pub flags: u32,
    pub computed_prompt_token_count: u32,
    /// Must be 0 on input (the scheduler probes the cache itself).
    pub cached_prefix_token_count: u32,
    /// Per-request prefill token cap; [`NO_PROMPT_LIMIT`] means use the
    /// scheduler cap.
    pub max_scheduled_prompt_token_count: u32,
    pub sequence_id: u64,
    /// Prompt token ids; required for prefill when the prefix cache is
    /// enabled, must be `None` for decode.
    pub prompt_token_ids: Option<&'a [u32]>,
}

impl<'a> SchedulerRequest<'a> {
    /// Decode request for `active_sequence_count` sequences
    /// (`SparkTestInitializeDecodeRequest`-shaped convenience).
    pub fn decode(active_sequence_count: u32) -> Self {
        SchedulerRequest {
            active_sequence_count,
            prompt_token_count: 0,
            flags: REQUEST_FLAG_DECODE,
            computed_prompt_token_count: 0,
            cached_prefix_token_count: 0,
            max_scheduled_prompt_token_count: NO_PROMPT_LIMIT,
            sequence_id: 0,
            prompt_token_ids: None,
        }
    }

    /// Prefill request for one prompt.
    pub fn prefill(
        active_sequence_count: u32,
        prompt_token_count: u32,
        sequence_id: u64,
        prompt_token_ids: &'a [u32],
    ) -> Self {
        SchedulerRequest {
            active_sequence_count,
            prompt_token_count,
            flags: REQUEST_FLAG_PREFILL,
            computed_prompt_token_count: 0,
            cached_prefix_token_count: 0,
            max_scheduled_prompt_token_count: NO_PROMPT_LIMIT,
            sequence_id,
            prompt_token_ids: Some(prompt_token_ids),
        }
    }

    fn is_prefill(&self) -> bool {
        self.flags & REQUEST_FLAG_PREFILL != 0
    }

    fn is_decode(&self) -> bool {
        self.flags & REQUEST_FLAG_DECODE != 0
    }
}

/// One packed request inside a decode batch decision
/// (`SparkSchedulerPackedRequest`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedRequest {
    pub request_index: u32,
    pub active_sequence_offset: u32,
    pub active_sequence_count: u32,
    pub request_flags: u32,
    pub scheduled_token_count: u32,
    pub total_scheduled_token_count: u64,
}

/// Per-stage dispatch descriptor (`SparkSchedulerDispatchStage`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchStage {
    pub spark_index: u32,
    pub batch_bucket: u32,
    pub first_layer_index: u32,
    pub layer_count: u32,
    pub stage_flags: u32,
    pub dispatch_flags: u32,
    pub active_sequence_count: u32,
    pub graph_sequence_capacity: u32,
    pub graph_sequence_padding_count: u32,
    pub scheduled_prompt_token_offset: u32,
    pub scheduled_prompt_token_count: u32,
    pub cached_prefix_token_count: u32,
    pub estimated_service_time_ns: u64,
}

/// Admission decision (`SparkSchedulerDecision`). Owns the KV block table,
/// stage plan, dispatch stages, and a copy of the prompt token ids.
#[derive(Debug, Clone)]
pub struct SchedulerDecision {
    pub accepted: bool,
    pub batch_bucket: u32,
    pub quantization_mode: QuantizationMode,
    pub spark_count: u32,
    pub stage_count: u32,
    /// `None` == `SPARK_STATUS_OK`.
    pub rejected_status: Option<SchedulerError>,
    pub decision_flags: u32,
    pub active_sequence_count: u32,
    pub graph_sequence_capacity: u32,
    pub graph_sequence_padding_count: u32,
    pub prompt_token_count: u32,
    pub computed_prompt_token_count: u32,
    pub cached_prefix_token_count: u32,
    pub prefix_cache_block_count: u32,
    pub scheduled_prompt_token_offset: u32,
    pub scheduled_prompt_token_count: u32,
    pub remaining_prompt_token_count_after_step: u32,
    pub prefill_block_count: u32,
    pub cache_commit_token_count_after_step: u32,
    pub kv_block_token_count: u32,
    pub kv_physical_block_count: u32,
    pub kv_cached_physical_block_count: u32,
    pub kv_pending_physical_block_count: u32,
    pub kv_block_table_token_count: u32,
    pub prefix_cache_reservation_epoch: u64,
    pub prefix_cache_parent_hash: u64,
    pub prefix_cache_result_hash: u64,
    pub sequence_id: u64,
    pub prompt_token_ids: Vec<u32>,
    pub total_scheduled_token_count: u64,
    pub estimated_critical_path_ns: u64,
    pub kv_physical_block_indices: Vec<u32>,
    pub stage_plan: StagePlan,
    pub dispatch_stages: Vec<DispatchStage>,
}

impl SchedulerDecision {
    /// Fresh zeroed decision with the scheduler's quantization mode and spark
    /// count set (the C `memset` + field stamps at the top of
    /// `SparkSchedulerAdmit`).
    fn empty(quantization_mode: QuantizationMode, spark_count: u32) -> Self {
        SchedulerDecision {
            accepted: false,
            batch_bucket: 0,
            quantization_mode,
            spark_count,
            stage_count: 0,
            rejected_status: None,
            decision_flags: 0,
            active_sequence_count: 0,
            graph_sequence_capacity: 0,
            graph_sequence_padding_count: 0,
            prompt_token_count: 0,
            computed_prompt_token_count: 0,
            cached_prefix_token_count: 0,
            prefix_cache_block_count: 0,
            scheduled_prompt_token_offset: 0,
            scheduled_prompt_token_count: 0,
            remaining_prompt_token_count_after_step: 0,
            prefill_block_count: 0,
            cache_commit_token_count_after_step: 0,
            kv_block_token_count: 0,
            kv_physical_block_count: 0,
            kv_cached_physical_block_count: 0,
            kv_pending_physical_block_count: 0,
            kv_block_table_token_count: 0,
            prefix_cache_reservation_epoch: 0,
            prefix_cache_parent_hash: 0,
            prefix_cache_result_hash: 0,
            sequence_id: 0,
            prompt_token_ids: Vec::new(),
            total_scheduled_token_count: 0,
            estimated_critical_path_ns: 0,
            kv_physical_block_indices: Vec::new(),
            stage_plan: StagePlan {
                abi_version: 0,
                descriptor_bytes: 0,
                stage_count: 0,
                reserved: 0,
                stages: Vec::new(),
            },
            dispatch_stages: Vec::new(),
        }
    }
}

/// One prefill lane inside a prefill batch decision
/// (`SparkSchedulerPrefillBatchLane`).
#[derive(Debug, Clone)]
pub struct PrefillBatchLane {
    pub request_index: u32,
    pub active_sequence_offset: u32,
    pub active_sequence_count: u32,
    pub prompt_token_count: u32,
    pub computed_prompt_token_count: u32,
    pub cached_prefix_token_count: u32,
    pub scheduled_prompt_token_offset: u32,
    pub scheduled_prompt_token_count: u32,
    pub remaining_prompt_token_count_after_step: u32,
    pub cache_commit_token_count_after_step: u32,
    pub prefix_cache_block_count: u32,
    pub kv_block_token_count: u32,
    pub kv_physical_block_count: u32,
    pub kv_cached_physical_block_count: u32,
    pub kv_pending_physical_block_count: u32,
    pub kv_block_table_token_count: u32,
    pub prefix_cache_reservation_epoch: u64,
    pub prefix_cache_parent_hash: u64,
    pub prefix_cache_result_hash: u64,
    pub sequence_id: u64,
    pub prompt_token_ids: Vec<u32>,
}

/// Decode batch admission decision (`SparkSchedulerBatchDecision`).
#[derive(Debug, Clone)]
pub struct BatchDecision {
    pub accepted: bool,
    /// `None` == `SPARK_STATUS_OK`.
    pub rejected_status: Option<SchedulerError>,
    pub source_request_count: u32,
    pub packed_request_count: u32,
    pub batch_bucket: u32,
    pub active_sequence_count: u32,
    pub graph_sequence_capacity: u32,
    pub graph_sequence_padding_count: u32,
    pub decision_flags: u32,
    pub total_scheduled_token_count: u64,
    pub estimated_critical_path_ns: u64,
    pub stage_decision: SchedulerDecision,
    pub packed_requests: Vec<PackedRequest>,
}

/// Prefill batch admission decision (`SparkSchedulerPrefillBatchDecision`).
#[derive(Debug, Clone)]
pub struct PrefillBatchDecision {
    pub accepted: bool,
    /// `None` == `SPARK_STATUS_OK`.
    pub rejected_status: Option<SchedulerError>,
    pub source_request_count: u32,
    pub packed_request_count: u32,
    pub batch_bucket: u32,
    pub active_sequence_count: u32,
    pub graph_sequence_capacity: u32,
    pub graph_sequence_padding_count: u32,
    pub decision_flags: u32,
    pub maximum_scheduled_prompt_token_count: u32,
    pub total_scheduled_token_count: u64,
    pub estimated_critical_path_ns: u64,
    pub stage_decision: SchedulerDecision,
    pub lanes: Vec<PrefillBatchLane>,
}

/// `SparkGlm52SchedulerRoundDownToMultiple`.
fn round_down_to_multiple(value: u32, multiple: u32) -> u32 {
    if multiple == 0 {
        return value;
    }
    value - (value % multiple)
}

/// `SparkSchedulerStageCostNs`.
fn stage_cost_ns(
    layer_cost_ns: &[u64],
    final_stage_extra_cost_ns: u64,
    stage: &StagePlanStage,
) -> u64 {
    let layer_end = (stage.first_layer_index + stage.layer_count) as usize;
    let mut stage_cost_ns: u64 =
        layer_cost_ns[stage.first_layer_index as usize..layer_end].iter().sum();
    if stage.flags & stage_plan::STAGE_FLAG_FINAL_TOKEN != 0 {
        stage_cost_ns += final_stage_extra_cost_ns;
    }
    stage_cost_ns
}

/// `SparkSchedulerPlanCriticalPathNs`.
fn plan_critical_path_ns(
    stage_plan: &StagePlan,
    layer_cost_ns: &[u64],
    final_stage_extra_cost_ns: u64,
    prefill_block_count: u32,
) -> u64 {
    let prefill_block_count = prefill_block_count.max(1);
    let mut critical_path_ns = 0u64;
    for stage in &stage_plan.stages[..stage_plan.stage_count as usize] {
        let stage_service_time_ns = stage_cost_ns(layer_cost_ns, final_stage_extra_cost_ns, stage)
            * u64::from(prefill_block_count);
        if stage_service_time_ns > critical_path_ns {
            critical_path_ns = stage_service_time_ns;
        }
    }
    critical_path_ns
}

/// `SparkSchedulerEffectiveComputedPromptTokenCount`.
fn effective_computed_prompt_token_count(
    request: &SchedulerRequest,
    cached_prefix_token_count: u32,
) -> u32 {
    request.computed_prompt_token_count.max(cached_prefix_token_count)
}

/// `SparkSchedulerBuildDispatchFlags`.
fn build_dispatch_flags(decision_flags: u32) -> u32 {
    let mut dispatch_flags = 0u32;
    if decision_flags & DECISION_FLAG_DECODE_STEP != 0 {
        dispatch_flags |= DISPATCH_STAGE_FLAG_DECODE;
    }
    if decision_flags & DECISION_FLAG_PREFILL_STEP != 0 {
        dispatch_flags |= DISPATCH_STAGE_FLAG_PREFILL;
    }
    if decision_flags & DECISION_FLAG_PREFILL_CHUNK != 0 {
        dispatch_flags |= DISPATCH_STAGE_FLAG_PREFILL_CHUNK;
    }
    if decision_flags & DECISION_FLAG_PREFILL_FINAL_CHUNK != 0 {
        dispatch_flags |= DISPATCH_STAGE_FLAG_PREFILL_FINAL_CHUNK;
    }
    if decision_flags & DECISION_FLAG_CUDAGRAPH_PADDING != 0 {
        dispatch_flags |= DISPATCH_STAGE_FLAG_CUDAGRAPH_PADDING;
    }
    if decision_flags & DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT != 0 {
        dispatch_flags |= DISPATCH_STAGE_FLAG_PREFILL_RESERVED_DECODE_SLOT;
    }
    if decision_flags & DECISION_FLAG_DECODE_BYPASS_PREFILL != 0 {
        dispatch_flags |= DISPATCH_STAGE_FLAG_DECODE_BYPASS_PREFILL;
    }
    if decision_flags & DECISION_FLAG_MEASURED_DECODE_BUCKET != 0 {
        dispatch_flags |= DISPATCH_STAGE_FLAG_MEASURED_DECODE_BUCKET;
    }
    dispatch_flags
}

/// How a retired prefill batch settles each lane's prefix-cache reservation
/// (the C `SparkGlm52SchedulerReservationFunction` parameter of
/// `SparkSchedulerRetirePrefillBatch`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReservationSettlement {
    Commit,
    Cancel,
}

/// Request scheduler (`SparkScheduler`). Statistics counters are public
/// fields, mirroring the C struct.
#[derive(Debug)]
pub struct Scheduler {
    stage_geometry: StagePlanGeometry,
    estimated_layer_cost_ns: u64,
    estimated_final_stage_extra_cost_ns: u64,
    spark_count: u32,
    queue_depth_per_spark: u32,
    measured_profile_id: u32,
    quantization_mode: QuantizationMode,
    max_prefill_tokens_per_step: u32,
    max_context_tokens: u32,
    max_batch_bucket: u32,
    prefix_cache_block_tokens: u32,
    configuration_flags: u32,
    prefix_cache: Option<PrefixCache>,
    spark_inflight_counts: Vec<u32>,
    prefill_demand: u32,
    // Statistics counters below mirror the C struct fields one-for-one.
    pub admitted_count: u64,
    pub rejected_count: u64,
    pub completed_count: u64,
    pub scheduled_decode_token_count: u64,
    pub scheduled_prefill_token_count: u64,
    pub prefix_cache_hit_token_count: u64,
    pub chunked_prefill_count: u64,
    pub interleaved_prefill_admission_count: u64,
    pub decode_bypass_admission_count: u64,
    pub adaptive_decode_pack_admission_count: u64,
    pub adaptive_decode_pack_request_count: u64,
    pub adaptive_decode_pack_padding_token_count: u64,
    pub adaptive_prefill_pack_admission_count: u64,
    pub adaptive_prefill_pack_request_count: u64,
    pub adaptive_prefill_pack_padding_token_count: u64,
    pub measured_decode_bucket_selection_count: u64,
    pub measured_decode_bucket_padding_token_count: u64,
    pub kv_block_reservation_count: u64,
    pub kv_block_reservation_token_count: u64,
    pub kv_block_cancel_count: u64,
}

impl Scheduler {
    /// `SparkSchedulerInitialize` (validation included).
    pub fn new(configuration: SchedulerConfig) -> Result<Self, SchedulerError> {
        // SparkSchedulerValidateConfiguration.
        if configuration.max_batch_bucket != stage_plan::MAX_BATCH_BUCKET
            || configuration.configuration_flags == 0
            || configuration.configuration_flags & !CONFIGURATION_KNOWN_FLAGS != 0
            || configuration.prefix_cache_block_tokens == 0
            || configuration.default_max_prefill_tokens_per_step == 0
            || configuration.max_context_tokens == 0
        {
            return Err(SchedulerError::InvalidArgument);
        }
        if configuration.configuration_flags & CONFIGURATION_FLAG_PREFIX_CACHE != 0 {
            let prefix_cache =
                configuration.prefix_cache.as_ref().ok_or(SchedulerError::InvalidArgument)?;
            if prefix_cache.block_token_count() != configuration.prefix_cache_block_tokens {
                return Err(SchedulerError::InvalidArgument);
            }
            // CONFIGURATION_FLAG_KV_CACHE_REQUIRED: the Rust PrefixCache
            // always owns its KV arena, so presence of the cache covers the
            // C's kv_cache_arena != NULL check.
        }
        if configuration.measured_profile_id == stage_plan::PROFILE_UNIFORM_ESTIMATED
            && configuration.estimated_layer_cost_ns == 0
        {
            return Err(SchedulerError::InvalidArgument);
        }
        if configuration.stage_geometry.layer_count == 0
            || configuration.stage_geometry.layer_count > stage_plan::MAX_LAYER_COUNT
            || configuration.stage_geometry.first_routed_layer
                > configuration.stage_geometry.layer_count
        {
            return Err(SchedulerError::InvalidArgument);
        }

        let queue_depth_per_spark = if configuration.queue_depth_per_spark == 0 {
            DEFAULT_QUEUE_DEPTH_PER_SPARK
        } else {
            configuration.queue_depth_per_spark
        };
        let quantization_mode = match configuration.quantization_mode {
            QuantizationMode::Auto => QuantizationMode::Nvfp4_4Bit,
            mode => mode,
        };
        let mut max_prefill_tokens_per_step = if configuration.max_prefill_tokens_per_step == 0 {
            configuration.default_max_prefill_tokens_per_step
        } else {
            configuration.max_prefill_tokens_per_step
        };
        if max_prefill_tokens_per_step < configuration.prefix_cache_block_tokens {
            max_prefill_tokens_per_step = configuration.prefix_cache_block_tokens;
        }

        Ok(Scheduler {
            stage_geometry: configuration.stage_geometry,
            estimated_layer_cost_ns: configuration.estimated_layer_cost_ns,
            estimated_final_stage_extra_cost_ns: configuration.estimated_final_stage_extra_cost_ns,
            spark_count: configuration.spark_count,
            queue_depth_per_spark,
            measured_profile_id: configuration.measured_profile_id,
            quantization_mode,
            max_prefill_tokens_per_step,
            max_context_tokens: configuration.max_context_tokens,
            max_batch_bucket: configuration.max_batch_bucket,
            prefix_cache_block_tokens: configuration.prefix_cache_block_tokens,
            configuration_flags: configuration.configuration_flags,
            prefix_cache: configuration.prefix_cache,
            spark_inflight_counts: vec![0; configuration.spark_count as usize],
            prefill_demand: 0,
            admitted_count: 0,
            rejected_count: 0,
            completed_count: 0,
            scheduled_decode_token_count: 0,
            scheduled_prefill_token_count: 0,
            prefix_cache_hit_token_count: 0,
            chunked_prefill_count: 0,
            interleaved_prefill_admission_count: 0,
            decode_bypass_admission_count: 0,
            adaptive_decode_pack_admission_count: 0,
            adaptive_decode_pack_request_count: 0,
            adaptive_decode_pack_padding_token_count: 0,
            adaptive_prefill_pack_admission_count: 0,
            adaptive_prefill_pack_request_count: 0,
            adaptive_prefill_pack_padding_token_count: 0,
            measured_decode_bucket_selection_count: 0,
            measured_decode_bucket_padding_token_count: 0,
            kv_block_reservation_count: 0,
            kv_block_reservation_token_count: 0,
            kv_block_cancel_count: 0,
        })
    }

    /// Read-only access to the owned prefix cache.
    pub fn prefix_cache(&self) -> Option<&PrefixCache> {
        self.prefix_cache.as_ref()
    }

    /// Mutable access to the owned prefix cache (callers drive probe/commit
    /// between scheduler calls, as in the C design).
    pub fn prefix_cache_mut(&mut self) -> Option<&mut PrefixCache> {
        self.prefix_cache.as_mut()
    }

    /// Normalized quantization mode (`SparkSchedulerNormalizeQuantizationMode`
    /// applied at initialize).
    pub fn quantization_mode(&self) -> QuantizationMode {
        self.quantization_mode
    }

    /// Normalized per-step prefill token cap.
    pub fn max_prefill_tokens_per_step(&self) -> u32 {
        self.max_prefill_tokens_per_step
    }

    /// Configuration flags as initialized.
    pub fn configuration_flags(&self) -> u32 {
        self.configuration_flags
    }

    /// Tokens per prefix-cache block.
    pub fn prefix_cache_block_tokens(&self) -> u32 {
        self.prefix_cache_block_tokens
    }

    /// Spark count of the ring.
    pub fn spark_count(&self) -> u32 {
        self.spark_count
    }

    /// Current per-spark in-flight counts.
    pub fn spark_inflight_counts(&self) -> &[u32] {
        &self.spark_inflight_counts
    }

    /// Maximum requests packed into one batch decision
    /// (`SPARK_SCHEDULER_MAX_PACKED_REQUEST_COUNT`).
    pub fn max_packed_request_count(&self) -> u32 {
        self.max_batch_bucket
    }

    /// Maximum requests accepted in one batch request
    /// (`SPARK_SCHEDULER_MAX_BATCH_REQUEST_COUNT`).
    pub fn max_batch_request_count(&self) -> u32 {
        self.max_batch_bucket * BATCH_REQUEST_CAPACITY_FACTOR
    }

    /// KV block table capacity implied by `max_context_tokens`
    /// (`SPARK_SCHEDULER_KV_BLOCK_TABLE_CAPACITY`). Advisory in Rust (the
    /// decision owns a `Vec`); kept for parity with the C layout.
    pub fn kv_block_table_capacity(&self) -> u32 {
        self.max_context_tokens / PREFILL_BLOCK_TOKENS
    }

    fn prompt_cache_is_enabled(&self) -> bool {
        self.configuration_flags & CONFIGURATION_FLAG_PREFIX_CACHE != 0
    }

    fn cross_sequence_prefix_reuse_is_enabled(&self) -> bool {
        self.configuration_flags & CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE != 0
    }

    fn chunked_prefill_is_enabled(&self) -> bool {
        self.configuration_flags & CONFIGURATION_FLAG_CHUNKED_PREFILL != 0
    }

    fn cuda_graph_padding_is_enabled(&self) -> bool {
        self.configuration_flags & CONFIGURATION_FLAG_CUDAGRAPH_PADDING != 0
    }

    fn measured_decode_bucket_selection_is_enabled(&self) -> bool {
        self.configuration_flags & CONFIGURATION_FLAG_MEASURED_DECODE_BUCKET_SELECTION != 0
    }

    fn prefill_decode_interleave_is_enabled(&self) -> bool {
        self.configuration_flags & CONFIGURATION_FLAG_PREFILL_DECODE_INTERLEAVE != 0
    }

    /// `SparkSchedulerBuildMeasuredPlanAndCosts`.
    fn build_measured_plan_and_costs(
        &self,
        batch_bucket: u32,
    ) -> Result<(StagePlan, Vec<u64>, u64), SchedulerError> {
        if self.measured_profile_id == stage_plan::PROFILE_UNIFORM_ESTIMATED {
            let profile = stage_plan::load_uniform_cost_profile(
                &self.stage_geometry,
                self.estimated_layer_cost_ns,
                self.estimated_final_stage_extra_cost_ns,
            )?;
            let plan = stage_plan::build_balanced_with_final_cost(
                &self.stage_geometry,
                &profile.layer_cost_ns,
                profile.final_stage_extra_cost_ns,
                self.spark_count,
            )?;
            return Ok((plan, profile.layer_cost_ns, profile.final_stage_extra_cost_ns));
        }
        let plan = stage_plan::build_measured_balanced_for_quantization(
            &self.stage_geometry,
            self.measured_profile_id,
            batch_bucket,
            self.quantization_mode,
            self.spark_count,
        )?;
        let profile = stage_plan::load_measured_cost_profile_for_quantization(
            &self.stage_geometry,
            self.measured_profile_id,
            batch_bucket,
            self.quantization_mode,
        )?;
        Ok((plan, profile.layer_cost_ns, profile.final_stage_extra_cost_ns))
    }

    /// `SparkSchedulerSelectDecodeBatchBucket` — returns
    /// `(batch_bucket, minimal_batch_bucket)`.
    fn select_decode_batch_bucket(
        &self,
        active_sequence_count: u32,
    ) -> Result<(u32, u32), SchedulerError> {
        let minimal_bucket = stage_plan::select_batch_bucket(active_sequence_count)?;
        if !self.measured_decode_bucket_selection_is_enabled()
            || active_sequence_count <= stage_plan::BUCKET_B16
        {
            return Ok((minimal_bucket, minimal_bucket));
        }

        let mut best_bucket = minimal_bucket;
        let mut best_critical_path_ns = u64::MAX;
        for &candidate_bucket in stage_plan::BATCH_BUCKETS.iter() {
            if candidate_bucket < active_sequence_count {
                continue;
            }
            let Ok((candidate_stage_plan, candidate_layer_cost_ns, candidate_extra_ns)) =
                self.build_measured_plan_and_costs(candidate_bucket)
            else {
                continue;
            };
            let candidate_critical_path_ns = plan_critical_path_ns(
                &candidate_stage_plan,
                &candidate_layer_cost_ns,
                candidate_extra_ns,
                1,
            );
            if candidate_critical_path_ns < best_critical_path_ns
                || (candidate_critical_path_ns == best_critical_path_ns
                    && candidate_bucket < best_bucket)
            {
                best_bucket = candidate_bucket;
                best_critical_path_ns = candidate_critical_path_ns;
            }
        }
        Ok((best_bucket, minimal_bucket))
    }

    /// `SparkSchedulerSelectRequestBatchBucket` — returns
    /// `(batch_bucket, minimal_batch_bucket)`.
    fn select_request_batch_bucket(
        &self,
        request: &SchedulerRequest,
    ) -> Result<(u32, u32), SchedulerError> {
        if request.is_decode() {
            return self.select_decode_batch_bucket(request.active_sequence_count);
        }
        match stage_plan::select_batch_bucket(request.active_sequence_count) {
            Ok(batch_bucket) => Ok((batch_bucket, batch_bucket)),
            // The C maps any failure here to SPARK_STATUS_CAPACITY_EXCEEDED.
            Err(_) => Err(SchedulerError::CapacityExceeded),
        }
    }

    /// `SparkSchedulerStageHasCapacity`.
    fn stage_has_capacity(&self, spark_index: u32, request_is_prefill: bool) -> bool {
        if spark_index >= self.spark_count
            || self.spark_inflight_counts[spark_index as usize] >= self.queue_depth_per_spark
        {
            return false;
        }
        let reserved_slot = u32::from(
            self.prefill_decode_interleave_is_enabled()
                && self.queue_depth_per_spark > 1
                && (request_is_prefill || self.prefill_demand != 0),
        );
        if reserved_slot != 0
            && self.spark_inflight_counts[spark_index as usize]
                >= self.queue_depth_per_spark - reserved_slot
        {
            return false;
        }
        true
    }

    /// `SparkSchedulerDecodeBypassIsActive`.
    fn decode_bypass_is_active(&self) -> bool {
        if !self.prefill_decode_interleave_is_enabled() {
            return false;
        }
        self.spark_inflight_counts[..self.spark_count as usize].iter().any(|&count| count != 0)
    }

    /// `SparkSchedulerSetPrefillDemand`.
    pub fn set_prefill_demand(&mut self, prefill_demand: u32) {
        self.prefill_demand = u32::from(prefill_demand != 0);
    }

    /// `SparkSchedulerSelectPipelineBatchWidth`.
    pub fn select_pipeline_batch_width(
        &self,
        active_request_count: u32,
        batch_capacity: u32,
    ) -> u32 {
        if self.spark_count == 0 || active_request_count == 0 || batch_capacity == 0 {
            return 0;
        }
        let mut batch_width = active_request_count / self.spark_count;
        if active_request_count % self.spark_count != 0 {
            batch_width += 1;
        }
        batch_width.min(batch_capacity)
    }

    /// `SparkSchedulerLookupCachedPrefixTokenCount`.
    fn lookup_cached_prefix_token_count(
        &mut self,
        request: &SchedulerRequest,
    ) -> Result<u32, SchedulerError> {
        if !request.is_prefill()
            || !self.prompt_cache_is_enabled()
            || !self.cross_sequence_prefix_reuse_is_enabled()
        {
            return Ok(0);
        }
        let token_ids = request.prompt_token_ids.ok_or(SchedulerError::InvalidArgument)?;
        let token_ids = token_ids
            .get(..request.prompt_token_count as usize)
            .ok_or(SchedulerError::InvalidArgument)?;
        let prefix_cache = self.prefix_cache.as_mut().ok_or(SchedulerError::InvalidArgument)?;
        let lookup = prefix_cache.probe_prompt(request.sequence_id, token_ids)?;
        Ok(lookup.matched_token_count)
    }

    /// `SparkSchedulerReservePrompt`.
    fn reserve_prompt(
        &mut self,
        request: &SchedulerRequest,
        token_count: u32,
    ) -> Result<Reservation, SchedulerError> {
        let token_ids = request.prompt_token_ids.ok_or(SchedulerError::InvalidArgument)?;
        let token_ids =
            token_ids.get(..token_count as usize).ok_or(SchedulerError::InvalidArgument)?;
        let cross_sequence_reuse = self.cross_sequence_prefix_reuse_is_enabled();
        let prefix_cache = self.prefix_cache.as_mut().ok_or(SchedulerError::InvalidArgument)?;
        if cross_sequence_reuse {
            Ok(prefix_cache.reserve_prompt(request.sequence_id, token_ids)?)
        } else {
            Ok(prefix_cache.reserve_sequence_prompt(request.sequence_id, token_ids)?)
        }
    }

    /// `SparkSchedulerRequestMaxPrefillTokensPerStep`.
    fn request_max_prefill_tokens_per_step(&self, request: &SchedulerRequest) -> u32 {
        let mut max_prefill_tokens_per_step = request.max_scheduled_prompt_token_count;
        if max_prefill_tokens_per_step == 0
            || max_prefill_tokens_per_step > self.max_prefill_tokens_per_step
        {
            max_prefill_tokens_per_step = self.max_prefill_tokens_per_step;
        }
        if max_prefill_tokens_per_step < self.prefix_cache_block_tokens {
            max_prefill_tokens_per_step = self.prefix_cache_block_tokens;
        }
        max_prefill_tokens_per_step
    }

    /// `SparkSchedulerScheduledPrefillTokenCount`.
    fn scheduled_prefill_token_count(
        &self,
        request: &SchedulerRequest,
        computed_prompt_token_count: u32,
    ) -> u32 {
        let remaining_prompt_token_count = request.prompt_token_count - computed_prompt_token_count;
        if !self.chunked_prefill_is_enabled() {
            return remaining_prompt_token_count;
        }
        let max_prefill_tokens_per_step = self.request_max_prefill_tokens_per_step(request);
        if remaining_prompt_token_count <= max_prefill_tokens_per_step {
            return remaining_prompt_token_count;
        }
        let scheduled_prompt_token_count =
            round_down_to_multiple(max_prefill_tokens_per_step, self.prefix_cache_block_tokens);
        if scheduled_prompt_token_count == 0 {
            return remaining_prompt_token_count.min(self.prefix_cache_block_tokens);
        }
        scheduled_prompt_token_count
    }

    /// `SparkSchedulerPrefillBlockCount`.
    fn prefill_block_count(&self, prompt_token_count: u32) -> u32 {
        if prompt_token_count == 0 {
            return 1;
        }
        prompt_token_count.div_ceil(self.prefix_cache_block_tokens)
    }

    /// `SparkSchedulerBuildDecisionFlags`.
    fn build_decision_flags(
        &self,
        request: &SchedulerRequest,
        batch_bucket: u32,
        cached_prefix_token_count: u32,
        scheduled_prompt_token_count: u32,
        remaining_prompt_token_count_after_step: u32,
        decode_bypass_active: bool,
    ) -> u32 {
        let mut decision_flags = 0u32;
        if request.is_decode() {
            decision_flags |= DECISION_FLAG_DECODE_STEP;
            if decode_bypass_active {
                decision_flags |= DECISION_FLAG_DECODE_BYPASS_PREFILL;
            }
        }
        if request.is_prefill() {
            decision_flags |= DECISION_FLAG_PREFILL_STEP;
            if self.prefill_decode_interleave_is_enabled() && self.queue_depth_per_spark > 1 {
                decision_flags |= DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT;
            }
            if remaining_prompt_token_count_after_step == 0 {
                decision_flags |= DECISION_FLAG_PREFILL_FINAL_CHUNK;
            }
            if scheduled_prompt_token_count != 0 && remaining_prompt_token_count_after_step != 0 {
                decision_flags |= DECISION_FLAG_PREFILL_CHUNK;
            }
            if cached_prefix_token_count != 0 {
                decision_flags |= DECISION_FLAG_PREFIX_CACHE_USED;
            }
        }
        if self.cuda_graph_padding_is_enabled() && request.active_sequence_count < batch_bucket {
            decision_flags |= DECISION_FLAG_CUDAGRAPH_PADDING;
        }
        decision_flags
    }

    /// `SparkSchedulerReject`: stamp the decision rejected, bump the counter,
    /// and hand it back (the C returns `SPARK_STATUS_OK` here).
    fn reject(
        &mut self,
        mut decision: SchedulerDecision,
        rejected_status: SchedulerError,
    ) -> SchedulerDecision {
        decision.accepted = false;
        decision.rejected_status = Some(rejected_status);
        self.rejected_count += 1;
        decision
    }

    /// `SparkSchedulerValidateRequest`.
    fn validate_request(&self, request: &SchedulerRequest) -> Result<(), SchedulerError> {
        let is_decode = request.is_decode();
        let is_prefill = request.is_prefill();
        let prompt_token_ids_len = request.prompt_token_ids.map(<[u32]>::len);
        if request.flags & !REQUEST_KNOWN_FLAGS != 0
            || request.active_sequence_count == 0
            || is_decode == is_prefill
            || (is_decode && request.prompt_token_count != 0)
            || (is_decode && request.computed_prompt_token_count != 0)
            || (is_decode && request.cached_prefix_token_count != 0)
            || (is_decode && request.max_scheduled_prompt_token_count != 0)
            || (is_decode && request.sequence_id != 0)
            || (is_decode && request.prompt_token_ids.is_some())
            || (is_prefill && request.prompt_token_count == 0)
            || (is_prefill && request.cached_prefix_token_count != 0)
            || (is_prefill
                && self.prompt_cache_is_enabled()
                && (request.sequence_id == 0
                    || prompt_token_ids_len
                        .is_none_or(|len| len < request.prompt_token_count as usize)))
            || (is_prefill
                && self.prompt_cache_is_enabled()
                && self.cross_sequence_prefix_reuse_is_enabled()
                && request.computed_prompt_token_count != 0)
            || (is_prefill && request.computed_prompt_token_count >= request.prompt_token_count)
        {
            return Err(SchedulerError::InvalidArgument);
        }
        Ok(())
    }

    /// `SparkSchedulerEstimateDecodeChunkNs`.
    fn estimate_decode_chunk_ns(&self, execution_row_count: u32) -> Result<u64, SchedulerError> {
        let (batch_bucket, _minimal_batch_bucket) =
            self.select_decode_batch_bucket(execution_row_count)?;
        let (stage_plan, layer_cost_ns, final_stage_extra_cost_ns) =
            self.build_measured_plan_and_costs(batch_bucket)?;
        let estimated_work_ns =
            plan_critical_path_ns(&stage_plan, &layer_cost_ns, final_stage_extra_cost_ns, 1);
        if estimated_work_ns == 0 {
            return Err(SchedulerError::ModuleNotValidated);
        }
        Ok(estimated_work_ns)
    }

    /// `SparkSchedulerEstimateDecodeWorkNs`.
    pub fn estimate_decode_work_ns(
        &self,
        logical_sequence_count: u32,
        rows_per_sequence: u32,
        execution_row_capacity: u32,
    ) -> Result<u64, SchedulerError> {
        let chunk_shape = stage_plan::execution_chunk_shape(
            logical_sequence_count,
            rows_per_sequence,
            execution_row_capacity,
        )?;
        // SparkSchedulerSumDecodeChunkWorkNs.
        let mut total_work_ns = 0u64;
        let mut remaining_sequence_count = logical_sequence_count;
        for _chunk_index in 0..chunk_shape.chunk_count {
            let chunk_sequence_count =
                remaining_sequence_count.min(chunk_shape.maximum_sequences_per_chunk);
            let chunk_work_ns =
                self.estimate_decode_chunk_ns(chunk_sequence_count * rows_per_sequence)?;
            total_work_ns =
                total_work_ns.checked_add(chunk_work_ns).ok_or(SchedulerError::CapacityExceeded)?;
            remaining_sequence_count -= chunk_sequence_count;
        }
        if remaining_sequence_count != 0 {
            return Err(SchedulerError::InternalError);
        }
        Ok(total_work_ns)
    }

    /// `SparkSchedulerAdmit`.
    pub fn admit(
        &mut self,
        request: &SchedulerRequest,
    ) -> Result<SchedulerDecision, SchedulerError> {
        self.validate_request(request)?;

        let mut decision = SchedulerDecision::empty(self.quantization_mode, self.spark_count);

        let (batch_bucket, minimal_batch_bucket) = match self.select_request_batch_bucket(request) {
            Ok(buckets) => buckets,
            Err(status) => return Ok(self.reject(decision, status)),
        };
        let measured_decode_bucket_selected =
            request.is_decode() && batch_bucket != minimal_batch_bucket;

        let (stage_plan_value, layer_cost_ns, final_stage_extra_cost_ns) =
            match self.build_measured_plan_and_costs(batch_bucket) {
                Ok(plan_and_costs) => plan_and_costs,
                Err(status) => return Ok(self.reject(decision, status)),
            };
        decision.stage_plan = stage_plan_value;

        for stage_index in 0..decision.stage_plan.stage_count {
            if !self.stage_has_capacity(stage_index, request.is_prefill()) {
                return Ok(self.reject(decision, SchedulerError::Busy));
            }
        }

        let mut cached_prefix_token_count = 0u32;
        let mut computed_prompt_token_count = 0u32;
        let mut scheduled_prompt_token_count = 0u32;
        let mut remaining_prompt_token_count_after_step = 0u32;
        let mut prefill_block_count = 1u32;
        let mut prefix_cache_reservation: Option<Reservation> = None;
        if request.is_prefill() {
            cached_prefix_token_count = match self.lookup_cached_prefix_token_count(request) {
                Ok(token_count) => token_count,
                Err(status) => return Ok(self.reject(decision, status)),
            };
            computed_prompt_token_count =
                effective_computed_prompt_token_count(request, cached_prefix_token_count);
            scheduled_prompt_token_count =
                self.scheduled_prefill_token_count(request, computed_prompt_token_count);
            remaining_prompt_token_count_after_step = request.prompt_token_count
                - computed_prompt_token_count
                - scheduled_prompt_token_count;
            prefill_block_count = self.prefill_block_count(scheduled_prompt_token_count);
            if self.prompt_cache_is_enabled() {
                prefix_cache_reservation = match self.reserve_prompt(
                    request,
                    computed_prompt_token_count + scheduled_prompt_token_count,
                ) {
                    Ok(reservation) => Some(reservation),
                    Err(status) => return Ok(self.reject(decision, status)),
                };
            }
        }

        let graph_sequence_padding_count = batch_bucket - request.active_sequence_count;
        let decode_bypass_active = request.is_decode() && self.decode_bypass_is_active();
        let mut decision_flags = self.build_decision_flags(
            request,
            batch_bucket,
            cached_prefix_token_count,
            scheduled_prompt_token_count,
            remaining_prompt_token_count_after_step,
            decode_bypass_active,
        );
        if measured_decode_bucket_selected {
            decision_flags |= DECISION_FLAG_MEASURED_DECODE_BUCKET;
        }
        let dispatch_flags = build_dispatch_flags(decision_flags);

        decision.accepted = true;
        decision.batch_bucket = batch_bucket;
        decision.stage_count = decision.stage_plan.stage_count;
        decision.rejected_status = None;
        decision.decision_flags = decision_flags;
        decision.active_sequence_count = request.active_sequence_count;
        decision.graph_sequence_capacity = batch_bucket;
        decision.graph_sequence_padding_count = graph_sequence_padding_count;
        decision.prompt_token_count = request.prompt_token_count;
        decision.computed_prompt_token_count = computed_prompt_token_count;
        decision.cached_prefix_token_count = cached_prefix_token_count;
        decision.prefix_cache_block_count =
            cached_prefix_token_count / self.prefix_cache_block_tokens;
        decision.scheduled_prompt_token_offset = computed_prompt_token_count;
        decision.scheduled_prompt_token_count = scheduled_prompt_token_count;
        decision.remaining_prompt_token_count_after_step = remaining_prompt_token_count_after_step;
        decision.prefill_block_count = prefill_block_count;
        decision.cache_commit_token_count_after_step =
            computed_prompt_token_count + scheduled_prompt_token_count;
        if request.is_prefill() && self.prompt_cache_is_enabled() {
            let Some(reservation) = prefix_cache_reservation.as_ref() else {
                return Err(SchedulerError::InternalError);
            };
            decision.kv_block_token_count = self.prefix_cache_block_tokens;
            decision.kv_physical_block_count = reservation.physical_block_count;
            decision.kv_cached_physical_block_count = reservation.cached_physical_block_count;
            decision.kv_pending_physical_block_count = reservation.pending_physical_block_count;
            decision.kv_block_table_token_count = reservation.reserved_token_count;
            decision.prefix_cache_reservation_epoch = reservation.reservation_epoch;
            decision.prefix_cache_result_hash = reservation.last_block_hash;
            decision.kv_physical_block_indices = reservation.physical_block_indices.clone();
            if computed_prompt_token_count == 0 {
                decision.prefix_cache_parent_hash = EMPTY_PARENT_HASH;
            } else {
                let token_ids = request.prompt_token_ids.ok_or(SchedulerError::InvalidArgument)?;
                let hash_prefix = token_ids
                    .get(..computed_prompt_token_count as usize)
                    .ok_or(SchedulerError::InvalidArgument)?;
                let hash_result = hash_prompt_tokens(
                    self.prefix_cache_block_tokens,
                    EMPTY_PARENT_HASH,
                    hash_prefix,
                );
                match hash_result {
                    Ok(prompt_hash) => {
                        decision.prefix_cache_parent_hash = prompt_hash.prompt_hash;
                    }
                    Err(status) => {
                        // Roll back exactly like the C: cancel the
                        // reservation, then reject.
                        if let Some(prefix_cache) = self.prefix_cache.as_mut() {
                            let _ = prefix_cache.cancel_reservation(
                                request.sequence_id,
                                reservation.reservation_epoch,
                            );
                        }
                        return Ok(self.reject(decision, status.into()));
                    }
                }
            }
        }
        decision.sequence_id = request.sequence_id;
        if let Some(token_ids) = request.prompt_token_ids {
            decision.prompt_token_ids = token_ids.to_vec();
        }
        decision.total_scheduled_token_count = if request.is_prefill() {
            u64::from(request.active_sequence_count) * u64::from(scheduled_prompt_token_count)
        } else {
            u64::from(request.active_sequence_count)
        };
        decision.estimated_critical_path_ns = 0;

        for stage_index in 0..decision.stage_plan.stage_count as usize {
            let stage = &decision.stage_plan.stages[stage_index];
            let stage_service_time_ns =
                stage_cost_ns(&layer_cost_ns, final_stage_extra_cost_ns, stage)
                    * u64::from(prefill_block_count);
            decision.dispatch_stages.push(DispatchStage {
                spark_index: stage_index as u32,
                batch_bucket,
                first_layer_index: stage.first_layer_index,
                layer_count: stage.layer_count,
                stage_flags: stage.flags,
                dispatch_flags,
                active_sequence_count: request.active_sequence_count,
                graph_sequence_capacity: batch_bucket,
                graph_sequence_padding_count,
                scheduled_prompt_token_offset: computed_prompt_token_count,
                scheduled_prompt_token_count,
                cached_prefix_token_count,
                estimated_service_time_ns: stage_service_time_ns,
            });
            if stage_service_time_ns > decision.estimated_critical_path_ns {
                decision.estimated_critical_path_ns = stage_service_time_ns;
            }
            self.spark_inflight_counts[stage_index] += 1;
        }

        self.admitted_count += 1;
        if request.is_prefill() {
            self.scheduled_prefill_token_count += decision.total_scheduled_token_count;
            self.prefix_cache_hit_token_count +=
                u64::from(request.active_sequence_count) * u64::from(cached_prefix_token_count);
            if decision_flags & DECISION_FLAG_PREFILL_CHUNK != 0 {
                self.chunked_prefill_count += 1;
            }
            if decision_flags & DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT != 0 {
                self.interleaved_prefill_admission_count += 1;
            }
            if self.prompt_cache_is_enabled() {
                self.kv_block_reservation_count +=
                    u64::from(decision.kv_pending_physical_block_count);
                self.kv_block_reservation_token_count +=
                    u64::from(decision.kv_block_table_token_count);
            }
        } else {
            self.scheduled_decode_token_count += decision.total_scheduled_token_count;
            if decision_flags & DECISION_FLAG_DECODE_BYPASS_PREFILL != 0 {
                self.decode_bypass_admission_count += 1;
            }
            if decision_flags & DECISION_FLAG_MEASURED_DECODE_BUCKET != 0 {
                self.measured_decode_bucket_selection_count += 1;
                self.measured_decode_bucket_padding_token_count +=
                    u64::from(decision.graph_sequence_padding_count);
            }
        }
        Ok(decision)
    }

    /// `SparkSchedulerValidateAcceptedDecision`.
    fn validate_accepted_decision(
        &self,
        decision: &SchedulerDecision,
    ) -> Result<(), SchedulerError> {
        if !decision.accepted
            || decision.stage_count == 0
            || decision.stage_count > self.spark_count
            || decision.dispatch_stages.len() < decision.stage_count as usize
        {
            return Err(SchedulerError::InvalidArgument);
        }
        Ok(())
    }

    /// `SparkSchedulerReleaseDecisionInflight`.
    fn release_decision_inflight(
        &mut self,
        decision: &SchedulerDecision,
    ) -> Result<(), SchedulerError> {
        for stage_index in 0..decision.stage_count as usize {
            let spark_index = decision.dispatch_stages[stage_index].spark_index;
            if spark_index >= self.spark_count
                || self.spark_inflight_counts[spark_index as usize] == 0
            {
                return Err(SchedulerError::InvalidArgument);
            }
        }
        for stage_index in 0..decision.stage_count as usize {
            let spark_index = decision.dispatch_stages[stage_index].spark_index;
            self.spark_inflight_counts[spark_index as usize] -= 1;
        }
        Ok(())
    }

    /// `SparkSchedulerComplete`.
    pub fn complete(&mut self, decision: &SchedulerDecision) -> Result<(), SchedulerError> {
        self.validate_accepted_decision(decision)?;
        self.release_decision_inflight(decision)?;
        if decision.decision_flags & DECISION_FLAG_PREFILL_STEP != 0
            && self.prompt_cache_is_enabled()
            && decision.cache_commit_token_count_after_step != 0
        {
            let prefix_cache = self.prefix_cache.as_mut().ok_or(SchedulerError::InvalidArgument)?;
            prefix_cache.commit_reservation(
                decision.sequence_id,
                decision.prefix_cache_reservation_epoch,
            )?;
        }
        self.completed_count += 1;
        Ok(())
    }

    /// `SparkSchedulerCancel`.
    pub fn cancel(&mut self, decision: &SchedulerDecision) -> Result<(), SchedulerError> {
        self.validate_accepted_decision(decision)?;
        self.release_decision_inflight(decision)?;
        if decision.decision_flags & DECISION_FLAG_PREFILL_STEP != 0
            && self.prompt_cache_is_enabled()
            && decision.prefix_cache_reservation_epoch != 0
        {
            let prefix_cache = self.prefix_cache.as_mut().ok_or(SchedulerError::InvalidArgument)?;
            prefix_cache.cancel_reservation(
                decision.sequence_id,
                decision.prefix_cache_reservation_epoch,
            )?;
            self.kv_block_cancel_count += 1;
        }
        Ok(())
    }

    /// `SparkSchedulerValidateDecodeBatchRequest`.
    fn validate_decode_batch_request(
        &self,
        requests: &[SchedulerRequest],
    ) -> Result<(), SchedulerError> {
        if requests.is_empty() || requests.len() as u32 > self.max_batch_request_count() {
            return Err(SchedulerError::InvalidArgument);
        }
        for request in requests {
            self.validate_request(request)?;
            if !request.is_decode() {
                return Err(SchedulerError::InvalidArgument);
            }
        }
        Ok(())
    }

    /// Rejected decode-batch decision (`SparkSchedulerRejectDecodeBatch`).
    fn reject_decode_batch(
        source_request_count: u32,
        rejected_status: SchedulerError,
    ) -> BatchDecision {
        BatchDecision {
            accepted: false,
            rejected_status: Some(rejected_status),
            source_request_count,
            packed_request_count: 0,
            batch_bucket: 0,
            active_sequence_count: 0,
            graph_sequence_capacity: 0,
            graph_sequence_padding_count: 0,
            decision_flags: 0,
            total_scheduled_token_count: 0,
            estimated_critical_path_ns: 0,
            stage_decision: SchedulerDecision::empty(QuantizationMode::Nvfp4_4Bit, 0),
            packed_requests: Vec::new(),
        }
    }

    /// `SparkSchedulerAdmitDecodeBatch`.
    pub fn admit_decode_batch(
        &mut self,
        requests: &[SchedulerRequest],
    ) -> Result<BatchDecision, SchedulerError> {
        let source_request_count = requests.len() as u32;
        // A validation failure writes a rejected batch decision AND returns
        // the status in C; only the error is observable here (see module
        // docs).
        self.validate_decode_batch_request(requests)?;

        let mut packed_requests = Vec::new();
        let mut active_sequence_offset = 0u32;
        let mut active_sequence_count = 0u32;
        for (request_index, request) in requests.iter().enumerate() {
            if packed_requests.len() as u32 >= self.max_packed_request_count() {
                break;
            }
            if request.active_sequence_count > self.max_packed_request_count() {
                return Ok(Self::reject_decode_batch(
                    source_request_count,
                    SchedulerError::CapacityExceeded,
                ));
            }
            if active_sequence_count + request.active_sequence_count
                > self.max_packed_request_count()
            {
                break;
            }
            // SparkSchedulerInitializePackedRequest.
            packed_requests.push(PackedRequest {
                request_index: request_index as u32,
                active_sequence_offset,
                active_sequence_count: request.active_sequence_count,
                request_flags: request.flags,
                scheduled_token_count: request.active_sequence_count,
                total_scheduled_token_count: u64::from(request.active_sequence_count),
            });
            active_sequence_offset += request.active_sequence_count;
            active_sequence_count += request.active_sequence_count;
        }
        let packed_request_count = packed_requests.len() as u32;
        if packed_request_count == 0 || active_sequence_count == 0 {
            return Ok(Self::reject_decode_batch(
                source_request_count,
                SchedulerError::CapacityExceeded,
            ));
        }

        let aggregate_request = SchedulerRequest::decode(active_sequence_count);
        let mut stage_decision = self.admit(&aggregate_request)?;
        if !stage_decision.accepted {
            let rejected_status =
                stage_decision.rejected_status.unwrap_or(SchedulerError::InternalError);
            return Ok(Self::reject_decode_batch(source_request_count, rejected_status));
        }

        stage_decision.decision_flags |= DECISION_FLAG_ADAPTIVE_DECODE_PACK;
        for dispatch_stage in
            &mut stage_decision.dispatch_stages[..stage_decision.stage_count as usize]
        {
            dispatch_stage.dispatch_flags |= DISPATCH_STAGE_FLAG_ADAPTIVE_DECODE_PACK;
        }

        let batch_decision = BatchDecision {
            accepted: true,
            rejected_status: None,
            source_request_count,
            packed_request_count,
            batch_bucket: stage_decision.batch_bucket,
            active_sequence_count,
            graph_sequence_capacity: stage_decision.graph_sequence_capacity,
            graph_sequence_padding_count: stage_decision.graph_sequence_padding_count,
            decision_flags: stage_decision.decision_flags,
            total_scheduled_token_count: stage_decision.total_scheduled_token_count,
            estimated_critical_path_ns: stage_decision.estimated_critical_path_ns,
            stage_decision,
            packed_requests,
        };

        self.adaptive_decode_pack_admission_count += 1;
        self.adaptive_decode_pack_request_count += u64::from(packed_request_count);
        self.adaptive_decode_pack_padding_token_count +=
            u64::from(batch_decision.graph_sequence_padding_count);
        Ok(batch_decision)
    }

    /// Validate an accepted decode-batch decision (the shared preamble of
    /// `SparkSchedulerCompleteDecodeBatch`/`SparkSchedulerCancelDecodeBatch`).
    fn validate_accepted_decode_batch(
        &self,
        batch_decision: &BatchDecision,
    ) -> Result<(), SchedulerError> {
        if !batch_decision.accepted
            || batch_decision.packed_request_count == 0
            || batch_decision.decision_flags & DECISION_FLAG_ADAPTIVE_DECODE_PACK == 0
        {
            return Err(SchedulerError::InvalidArgument);
        }
        Ok(())
    }

    /// `SparkSchedulerCompleteDecodeBatch`.
    pub fn complete_decode_batch(
        &mut self,
        batch_decision: &BatchDecision,
    ) -> Result<(), SchedulerError> {
        self.validate_accepted_decode_batch(batch_decision)?;
        self.complete(&batch_decision.stage_decision)
    }

    /// `SparkSchedulerCancelDecodeBatch`.
    pub fn cancel_decode_batch(
        &mut self,
        batch_decision: &BatchDecision,
    ) -> Result<(), SchedulerError> {
        self.validate_accepted_decode_batch(batch_decision)?;
        self.release_decision_inflight(&batch_decision.stage_decision)
    }

    /// `SparkSchedulerValidatePrefillBatchRequest`.
    fn validate_prefill_batch_request(
        &self,
        requests: &[SchedulerRequest],
    ) -> Result<(), SchedulerError> {
        if requests.is_empty() || requests.len() as u32 > self.max_batch_request_count() {
            return Err(SchedulerError::InvalidArgument);
        }
        for request in requests {
            self.validate_request(request)?;
            if !request.is_prefill() || request.active_sequence_count != 1 {
                return Err(SchedulerError::InvalidArgument);
            }
        }
        Ok(())
    }

    /// Rejected prefill-batch decision (`SparkSchedulerRejectPrefillBatch`).
    fn reject_prefill_batch(
        source_request_count: u32,
        rejected_status: SchedulerError,
    ) -> PrefillBatchDecision {
        PrefillBatchDecision {
            accepted: false,
            rejected_status: Some(rejected_status),
            source_request_count,
            packed_request_count: 0,
            batch_bucket: 0,
            active_sequence_count: 0,
            graph_sequence_capacity: 0,
            graph_sequence_padding_count: 0,
            decision_flags: 0,
            maximum_scheduled_prompt_token_count: 0,
            total_scheduled_token_count: 0,
            estimated_critical_path_ns: 0,
            stage_decision: SchedulerDecision::empty(QuantizationMode::Nvfp4_4Bit, 0),
            lanes: Vec::new(),
        }
    }

    /// `SparkSchedulerCancelAcceptedPrefillBatchReservations`.
    fn cancel_accepted_prefill_batch_reservations(
        &mut self,
        lanes: &[PrefillBatchLane],
    ) -> Result<(), SchedulerError> {
        if !self.prompt_cache_is_enabled() {
            return Ok(());
        }
        for lane in lanes {
            if lane.prefix_cache_reservation_epoch == 0 {
                continue;
            }
            let prefix_cache = self.prefix_cache.as_mut().ok_or(SchedulerError::InvalidArgument)?;
            prefix_cache
                .cancel_reservation(lane.sequence_id, lane.prefix_cache_reservation_epoch)?;
        }
        Ok(())
    }

    /// `SparkSchedulerAdmitPrefillBatch`.
    pub fn admit_prefill_batch(
        &mut self,
        requests: &[SchedulerRequest],
    ) -> Result<PrefillBatchDecision, SchedulerError> {
        let source_request_count = requests.len() as u32;
        self.validate_prefill_batch_request(requests)?;

        let mut active_sequence_count = 0u32;
        let mut packed_request_count = 0u32;
        for request in requests {
            if packed_request_count >= self.max_packed_request_count()
                || active_sequence_count + request.active_sequence_count
                    > self.max_packed_request_count()
            {
                break;
            }
            active_sequence_count += request.active_sequence_count;
            packed_request_count += 1;
        }
        if packed_request_count == 0 || active_sequence_count == 0 {
            return Ok(Self::reject_prefill_batch(
                source_request_count,
                SchedulerError::CapacityExceeded,
            ));
        }

        let batch_bucket = match stage_plan::select_batch_bucket(active_sequence_count) {
            Ok(batch_bucket) => batch_bucket,
            Err(status) => {
                return Ok(Self::reject_prefill_batch(source_request_count, status.into()))
            }
        };

        // The C prefill-batch path uses the measured builders directly (no
        // uniform-estimated fallback, unlike single-request admission).
        let stage_plan_value = match stage_plan::build_measured_balanced_for_quantization(
            &self.stage_geometry,
            self.measured_profile_id,
            batch_bucket,
            self.quantization_mode,
            self.spark_count,
        ) {
            Ok(stage_plan) => stage_plan,
            Err(status) => {
                return Ok(Self::reject_prefill_batch(source_request_count, status.into()))
            }
        };
        let profile = match stage_plan::load_measured_cost_profile_for_quantization(
            &self.stage_geometry,
            self.measured_profile_id,
            batch_bucket,
            self.quantization_mode,
        ) {
            Ok(profile) => profile,
            Err(status) => {
                return Ok(Self::reject_prefill_batch(source_request_count, status.into()))
            }
        };

        for stage_index in 0..stage_plan_value.stage_count {
            if !self.stage_has_capacity(stage_index, true) {
                return Ok(Self::reject_prefill_batch(source_request_count, SchedulerError::Busy));
            }
        }

        let mut lanes: Vec<PrefillBatchLane> = Vec::new();
        let mut maximum_scheduled_prompt_token_count = 0u32;
        let mut maximum_prefill_block_count = 1u32;
        let mut any_remaining_prompt_tokens = false;
        let mut total_cached_prefix_token_count = 0u32;
        let mut total_pending_physical_block_count = 0u32;
        let mut total_scheduled_token_count = 0u64;

        for (request_index, request) in requests[..packed_request_count as usize].iter().enumerate()
        {
            let cached_prefix_token_count = match self.lookup_cached_prefix_token_count(request) {
                Ok(token_count) => token_count,
                Err(status) => {
                    let _ = self.cancel_accepted_prefill_batch_reservations(&lanes);
                    return Err(status);
                }
            };
            let computed_prompt_token_count =
                effective_computed_prompt_token_count(request, cached_prefix_token_count);
            let scheduled_prompt_token_count =
                self.scheduled_prefill_token_count(request, computed_prompt_token_count);
            if scheduled_prompt_token_count == 0 {
                let _ = self.cancel_accepted_prefill_batch_reservations(&lanes);
                return Err(SchedulerError::InvalidArgument);
            }

            let reservation = match self
                .reserve_prompt(request, computed_prompt_token_count + scheduled_prompt_token_count)
            {
                Ok(reservation) => reservation,
                Err(status) => {
                    let _ = self.cancel_accepted_prefill_batch_reservations(&lanes);
                    return Err(status);
                }
            };

            let parent_hash = if computed_prompt_token_count == 0 {
                EMPTY_PARENT_HASH
            } else {
                let token_ids = request.prompt_token_ids.ok_or(SchedulerError::InvalidArgument)?;
                let hash_prefix = token_ids
                    .get(..computed_prompt_token_count as usize)
                    .ok_or(SchedulerError::InvalidArgument)?;
                match hash_prompt_tokens(
                    self.prefix_cache_block_tokens,
                    EMPTY_PARENT_HASH,
                    hash_prefix,
                ) {
                    Ok(prompt_hash) => prompt_hash.prompt_hash,
                    Err(status) => {
                        // Roll back exactly like the C: cancel this lane's
                        // reservation, then every accepted lane's.
                        if let Some(prefix_cache) = self.prefix_cache.as_mut() {
                            let _ = prefix_cache.cancel_reservation(
                                request.sequence_id,
                                reservation.reservation_epoch,
                            );
                        }
                        let _ = self.cancel_accepted_prefill_batch_reservations(&lanes);
                        return Err(status.into());
                    }
                }
            };

            // SparkSchedulerInitializePrefillBatchLane.
            let lane = PrefillBatchLane {
                request_index: request_index as u32,
                active_sequence_offset: request_index as u32,
                active_sequence_count: request.active_sequence_count,
                prompt_token_count: request.prompt_token_count,
                computed_prompt_token_count,
                cached_prefix_token_count,
                scheduled_prompt_token_offset: computed_prompt_token_count,
                scheduled_prompt_token_count,
                remaining_prompt_token_count_after_step: request.prompt_token_count
                    - computed_prompt_token_count
                    - scheduled_prompt_token_count,
                cache_commit_token_count_after_step: computed_prompt_token_count
                    + scheduled_prompt_token_count,
                prefix_cache_block_count: cached_prefix_token_count
                    / self.prefix_cache_block_tokens,
                kv_block_token_count: self.prefix_cache_block_tokens,
                kv_physical_block_count: reservation.physical_block_count,
                kv_cached_physical_block_count: reservation.cached_physical_block_count,
                kv_pending_physical_block_count: reservation.pending_physical_block_count,
                kv_block_table_token_count: reservation.reserved_token_count,
                prefix_cache_reservation_epoch: reservation.reservation_epoch,
                prefix_cache_parent_hash: parent_hash,
                prefix_cache_result_hash: reservation.last_block_hash,
                sequence_id: request.sequence_id,
                prompt_token_ids: request.prompt_token_ids.map(<[u32]>::to_vec).unwrap_or_default(),
            };
            if lane.remaining_prompt_token_count_after_step != 0 {
                any_remaining_prompt_tokens = true;
            }
            lanes.push(lane);
            total_cached_prefix_token_count += cached_prefix_token_count;
            total_pending_physical_block_count += reservation.pending_physical_block_count;
            total_scheduled_token_count += u64::from(scheduled_prompt_token_count);
            if scheduled_prompt_token_count > maximum_scheduled_prompt_token_count {
                maximum_scheduled_prompt_token_count = scheduled_prompt_token_count;
            }
            let prefill_block_count = self.prefill_block_count(scheduled_prompt_token_count);
            if prefill_block_count > maximum_prefill_block_count {
                maximum_prefill_block_count = prefill_block_count;
            }
        }

        let graph_sequence_padding_count = batch_bucket - active_sequence_count;
        let mut decision_flags = DECISION_FLAG_PREFILL_STEP | DECISION_FLAG_ADAPTIVE_PREFILL_PACK;
        if any_remaining_prompt_tokens {
            decision_flags |= DECISION_FLAG_PREFILL_CHUNK;
        } else {
            decision_flags |= DECISION_FLAG_PREFILL_FINAL_CHUNK;
        }
        if total_cached_prefix_token_count != 0 {
            decision_flags |= DECISION_FLAG_PREFIX_CACHE_USED;
        }
        if graph_sequence_padding_count != 0 && self.cuda_graph_padding_is_enabled() {
            decision_flags |= DECISION_FLAG_CUDAGRAPH_PADDING;
        }
        if self.prefill_decode_interleave_is_enabled() && self.queue_depth_per_spark > 1 {
            decision_flags |= DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT;
        }
        let dispatch_flags =
            build_dispatch_flags(decision_flags) | DISPATCH_STAGE_FLAG_ADAPTIVE_PREFILL_PACK;

        let mut stage_decision = SchedulerDecision::empty(self.quantization_mode, self.spark_count);
        stage_decision.stage_plan = stage_plan_value;
        stage_decision.accepted = true;
        stage_decision.batch_bucket = batch_bucket;
        stage_decision.stage_count = stage_decision.stage_plan.stage_count;
        stage_decision.rejected_status = None;
        stage_decision.decision_flags = decision_flags;
        stage_decision.active_sequence_count = active_sequence_count;
        stage_decision.graph_sequence_capacity = batch_bucket;
        stage_decision.graph_sequence_padding_count = graph_sequence_padding_count;
        stage_decision.scheduled_prompt_token_count = maximum_scheduled_prompt_token_count;
        stage_decision.prefill_block_count = maximum_prefill_block_count;
        stage_decision.total_scheduled_token_count = total_scheduled_token_count;

        let mut estimated_critical_path_ns = 0u64;
        for stage_index in 0..stage_decision.stage_plan.stage_count as usize {
            let stage = &stage_decision.stage_plan.stages[stage_index];
            let stage_service_time_ns =
                stage_cost_ns(&profile.layer_cost_ns, profile.final_stage_extra_cost_ns, stage)
                    * u64::from(maximum_prefill_block_count);
            stage_decision.dispatch_stages.push(DispatchStage {
                spark_index: stage_index as u32,
                batch_bucket,
                first_layer_index: stage.first_layer_index,
                layer_count: stage.layer_count,
                stage_flags: stage.flags,
                dispatch_flags,
                active_sequence_count,
                graph_sequence_capacity: batch_bucket,
                graph_sequence_padding_count,
                scheduled_prompt_token_offset: 0,
                scheduled_prompt_token_count: maximum_scheduled_prompt_token_count,
                cached_prefix_token_count: 0,
                estimated_service_time_ns: stage_service_time_ns,
            });
            if stage_service_time_ns > estimated_critical_path_ns {
                estimated_critical_path_ns = stage_service_time_ns;
            }
            self.spark_inflight_counts[stage_index] += 1;
        }
        stage_decision.estimated_critical_path_ns = estimated_critical_path_ns;

        self.admitted_count += 1;
        self.scheduled_prefill_token_count += total_scheduled_token_count;
        self.prefix_cache_hit_token_count += u64::from(total_cached_prefix_token_count);
        if decision_flags & DECISION_FLAG_PREFILL_CHUNK != 0 {
            self.chunked_prefill_count += 1;
        }
        if decision_flags & DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT != 0 {
            self.interleaved_prefill_admission_count += 1;
        }
        self.kv_block_reservation_count += u64::from(total_pending_physical_block_count);
        self.kv_block_reservation_token_count += total_scheduled_token_count;
        self.adaptive_prefill_pack_admission_count += 1;
        self.adaptive_prefill_pack_request_count += u64::from(packed_request_count);
        self.adaptive_prefill_pack_padding_token_count += u64::from(graph_sequence_padding_count);

        Ok(PrefillBatchDecision {
            accepted: true,
            rejected_status: None,
            source_request_count,
            packed_request_count,
            batch_bucket,
            active_sequence_count,
            graph_sequence_capacity: batch_bucket,
            graph_sequence_padding_count,
            decision_flags,
            maximum_scheduled_prompt_token_count,
            total_scheduled_token_count,
            estimated_critical_path_ns,
            stage_decision,
            lanes,
        })
    }

    /// `SparkSchedulerValidateAcceptedPrefillBatchDecision`.
    fn validate_accepted_prefill_batch_decision(
        &self,
        batch_decision: &PrefillBatchDecision,
    ) -> Result<(), SchedulerError> {
        if !batch_decision.accepted
            || batch_decision.packed_request_count == 0
            || batch_decision.packed_request_count > self.max_packed_request_count()
            || !batch_decision.stage_decision.accepted
            || batch_decision.decision_flags & DECISION_FLAG_ADAPTIVE_PREFILL_PACK == 0
            || batch_decision.lanes.len() < batch_decision.packed_request_count as usize
        {
            return Err(SchedulerError::InvalidArgument);
        }
        Ok(())
    }

    /// `SparkSchedulerRetirePrefillBatch`: validate, release the in-flight
    /// reservation, then settle every lane's prefix-cache reservation. The
    /// retirement counter differs per settlement (the C's `retire_counter`
    /// parameter): commit -> `completed_count`, cancel ->
    /// `kv_block_cancel_count`.
    fn retire_prefill_batch(
        &mut self,
        batch_decision: &PrefillBatchDecision,
        settlement: ReservationSettlement,
    ) -> Result<(), SchedulerError> {
        self.validate_accepted_prefill_batch_decision(batch_decision)?;
        self.release_decision_inflight(&batch_decision.stage_decision)?;
        for lane in &batch_decision.lanes[..batch_decision.packed_request_count as usize] {
            if lane.prefix_cache_reservation_epoch == 0 {
                continue;
            }
            let prefix_cache = self.prefix_cache.as_mut().ok_or(SchedulerError::InvalidArgument)?;
            match settlement {
                ReservationSettlement::Commit => prefix_cache
                    .commit_reservation(lane.sequence_id, lane.prefix_cache_reservation_epoch)?,
                ReservationSettlement::Cancel => prefix_cache
                    .cancel_reservation(lane.sequence_id, lane.prefix_cache_reservation_epoch)?,
            }
        }
        match settlement {
            ReservationSettlement::Commit => self.completed_count += 1,
            ReservationSettlement::Cancel => self.kv_block_cancel_count += 1,
        }
        Ok(())
    }

    /// `SparkSchedulerCompletePrefillBatch`.
    pub fn complete_prefill_batch(
        &mut self,
        batch_decision: &PrefillBatchDecision,
    ) -> Result<(), SchedulerError> {
        self.retire_prefill_batch(batch_decision, ReservationSettlement::Commit)
    }

    /// `SparkSchedulerCancelPrefillBatch`.
    pub fn cancel_prefill_batch(
        &mut self,
        batch_decision: &PrefillBatchDecision,
    ) -> Result<(), SchedulerError> {
        self.retire_prefill_batch(batch_decision, ReservationSettlement::Cancel)
    }

    /// `SparkSchedulerBuildKvBlockTable`: the physical block indices backing
    /// the decision's KV block table, in block order.
    pub fn build_kv_block_table(
        &mut self,
        decision: &SchedulerDecision,
    ) -> Result<Vec<u32>, SchedulerError> {
        self.validate_accepted_decision(decision)?;
        if !self.prompt_cache_is_enabled()
            || decision.decision_flags & DECISION_FLAG_PREFILL_STEP == 0
        {
            return Err(SchedulerError::InvalidArgument);
        }
        let prefix_cache = self.prefix_cache.as_mut().ok_or(SchedulerError::InvalidArgument)?;
        Ok(prefix_cache.build_physical_block_table(
            decision.sequence_id,
            decision.kv_block_table_token_count,
        )?)
    }

    /// `SparkSchedulerBuildPrefillBatchKvBlockTables`: one physical block
    /// table per packed lane. `lane_capacity` mirrors the C out-buffer row
    /// capacity: it must be nonzero and each lane's table must fit it
    /// ([`SchedulerError::CapacityExceeded`] otherwise, matching the C's
    /// refusal to truncate).
    pub fn build_prefill_batch_kv_block_tables(
        &mut self,
        batch_decision: &PrefillBatchDecision,
        lane_capacity: u32,
    ) -> Result<Vec<Vec<u32>>, SchedulerError> {
        self.validate_accepted_prefill_batch_decision(batch_decision)?;
        if !self.prompt_cache_is_enabled() || lane_capacity == 0 {
            return Err(SchedulerError::InvalidArgument);
        }
        let mut lane_tables = Vec::with_capacity(batch_decision.packed_request_count as usize);
        for lane in &batch_decision.lanes[..batch_decision.packed_request_count as usize] {
            if lane.sequence_id == 0 || lane.kv_block_table_token_count == 0 {
                return Err(SchedulerError::InvalidArgument);
            }
            let prefix_cache = self.prefix_cache.as_mut().ok_or(SchedulerError::InvalidArgument)?;
            let table = prefix_cache
                .build_physical_block_table(lane.sequence_id, lane.kv_block_table_token_count)?;
            if table.len() as u32 > lane_capacity {
                return Err(SchedulerError::CapacityExceeded);
            }
            lane_tables.push(table);
        }
        Ok(lane_tables)
    }

    /// `SparkSchedulerReleaseSequence`.
    pub fn release_sequence(&mut self, sequence_id: u64) -> Result<(), SchedulerError> {
        if sequence_id == 0 {
            return Err(SchedulerError::InvalidArgument);
        }
        if !self.prompt_cache_is_enabled() {
            return Ok(());
        }
        let prefix_cache = self.prefix_cache.as_mut().ok_or(SchedulerError::InvalidArgument)?;
        prefix_cache.release_sequence(sequence_id)?;
        Ok(())
    }
}
