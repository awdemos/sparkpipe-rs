//! Request/session lifecycle engine — port of `api/request.c` (the C tree's
//! largest file): session and slot lifecycle, stop conditions, validation,
//! dispatch retry/cancel choreography, KV block table views, JIT KV prefetch
//! planning, and the speculative (MTP) dispatch policy with milli-fixed-point
//! commit EMA.
//!
//! Orchestrates the already-ported subsystems the C shares raw struct
//! pointers with: the [`Scheduler`] (owning the [`PrefixCache`], which owns
//! the [`KvArena`]) lives inside one [`RequestApi`] and borrows are split
//! internally; the C's slot pointers become slot indices.
//!
//! Rust-port deviations from the C surface (behavior is otherwise
//! signature-faithful):
//!   - ABI plumbing (`abi_version`, `descriptor_bytes`, reserved words) is
//!     dropped; the C validation checks against those fields have no Rust
//!     counterpart.
//!   - The C's three prefetch function pointers + `void *` context collapse
//!     into the [`KvPrefetchBackend`] trait seam ([`prefetch::KvPrefetchBackend`]).
//!     The async prefetch *backend* itself (the device-copy engine of
//!     `cache/kv_cache.c`) is NOT ported; [`Configuration::use_async_kv_cache_prefetch_backend`]
//!     keeps the config surface: it sets the JIT+ASYNC flags and installs a
//!     caller-provided backend implementation.
//!   - The C's opaque `SparkRequestModelSpeculator` pointer and the
//!     twelve-entry `spark_request_model.h` linker seam collapse into the
//!     [`speculation::RequestModelSeam`] trait; [`speculation::NullSpeculator`]
//!     is the port of `serving/spark_request_model_null.c`. The GLM52 dspark
//!     provider itself is a model module and is not ported here.
//!   - `SparkMtpTreeResolve` validates token ids against the GLM52 output
//!     vocab count in C; here the bound is a [`Configuration`] field
//!     (`output_vocab_count`).
//!   - The slot owns a copy of the prompt token ids (the C borrows the
//!     caller's array). On the tokenizer-overflow path the C's
//!     `prompt_token_count` exceeds the storage it points at; the Rust slot
//!     zero-pads to `prompt_token_count` so every read stays in bounds.
//!   - `SparkKvCacheArenaBuildPrefetchPlan[FromSourceBlocks]` and
//!     `SparkKvCacheArenaMarkPrefetchPlanResident[WithProtectedBlocks]` are
//!     not part of `spark-core`'s `KvArena`; they are ported here
//!     ([`prefetch`]) over the arena's public API. The arena's resident
//!     block capacity has no public getter; it is recovered exactly at
//!     initialization via side-effect-free trim probes
//!     ([`prefetch::probe_resident_block_capacity`]).
//!   - Statistics counters are public fields, mirroring the C struct, and
//!     tests may mutate configuration words (`configuration_flags`,
//!     `decode_batch_target`, `next_prefetch_id`) exactly as the C tests do.

pub mod dispatch;
pub mod prefetch;
pub mod slot;
pub mod speculation;

use std::cell::RefCell;

use spark_core::prefix_cache::PrefixCacheError;
use spark_core::row_allocator::{row_allocator_assign, SlotInput};
use spark_sched::scheduler::{Scheduler, SchedulerError};
use spark_sched::work_control::mtp_tree;
use spark_text::prompt_pipeline::PromptPipelineError;

pub use dispatch::{
    copy_prefill_dispatch_token_ids, describe_prefill_dispatch, DecodeDispatchLaneView,
    DecodeDispatchView, Dispatch, DispatchKind, PrefillDispatchLaneView, PrefillDispatchView,
    DISPATCH_FLAG_DSPARK_CONFIDENCE_TRUNCATED, DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY,
    DISPATCH_FLAG_DSPARK_TAP_CAPTURE, DISPATCH_FLAG_JIT_PREFETCHED_KV,
    DISPATCH_FLAG_JIT_PREFETCH_PENDING, DISPATCH_FLAG_MTP_COMMIT,
    DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY, DISPATCH_FLAG_MTP_TREE_VERIFY,
    DISPATCH_FLAG_PREFILL_BATCH, DISPATCH_FLAG_PREFIX_COHORT, DISPATCH_FLAG_PREFIX_FAMILY_SELECTED,
    DISPATCH_FLAG_PRIORITY_PREEMPTED_QUEUE,
};
pub use prefetch::{
    KvPrefetchBackend, PrefetchBlock, PrefetchPlan, PREFETCH_BLOCK_CAPACITY,
    PREFETCH_BLOCK_DEFAULT_FLAGS, PREFETCH_BLOCK_FLAG_KEY, PREFETCH_BLOCK_FLAG_VALUE,
};
pub use slot::{
    CacheState, RequestApiHandle, Slot, SubmitRequest, DEFAULT_PRIORITY, REALTIME_PRIORITY,
    REQUEST_FLAG_DISABLE_SPECULATION, REQUEST_FLAG_REALTIME, REQUEST_KNOWN_FLAGS, STATE_CANCELLED,
    STATE_COMPLETED, STATE_FREE, STATE_QUEUED_PREFILL, STATE_READY_DECODE,
    STATE_READY_SPECULATIVE_VERIFY, STATE_RUNNING_DECODE, STATE_RUNNING_PREFILL,
    STATE_RUNNING_SPECULATIVE_VERIFY, STATE_WAITING_PREFIX_COHORT,
};
pub use speculation::{
    mtp_tree_resolve, DraftRequest, DraftResult, MtpTreeResolution, NullSpeculator,
    RequestModelSeam, VerifyResult, SPECULATIVE_SOURCE_DRAFTER, SPECULATIVE_SOURCE_MTP,
    VERIFY_RESULT_FLAG_ACCEPTED_ALL, VERIFY_RESULT_FLAG_REJECTED,
};

/// ABI version marker (`SPARK_REQUEST_API_ABI_VERSION`).
pub const ABI_VERSION: u32 = 7;
/// Invalid request handle sentinel (`SPARK_REQUEST_API_INVALID_HANDLE`).
pub const INVALID_HANDLE: u64 = 0;
/// Default prefetch lookahead (`SPARK_REQUEST_API_DEFAULT_PREFETCH_LOOKAHEAD_REQUEST_COUNT`).
pub const DEFAULT_PREFETCH_LOOKAHEAD_REQUEST_COUNT: u32 = 64;
/// Maximum prefetch lane count (`SPARK_KV_CACHE_MAX_PREFETCH_LANE_COUNT`).
pub const MAX_PREFETCH_LANE_COUNT: u32 = 13;
/// Default prefetch lane count (`SPARK_REQUEST_API_DEFAULT_PREFETCH_LANE_COUNT`).
pub const DEFAULT_PREFETCH_LANE_COUNT: u32 = MAX_PREFETCH_LANE_COUNT;
/// Maximum requests per dispatch
/// (`SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT` =
/// `SPARK_SCHEDULER_MAX_PACKED_REQUEST_COUNT`).
pub const MAX_DISPATCH_REQUEST_COUNT: u32 = spark_sched::stage_plan::MAX_BATCH_BUCKET;
/// Maximum prefetch source blocks
/// (`SPARK_REQUEST_API_MAX_PREFETCH_SOURCE_BLOCK_COUNT`).
pub const MAX_PREFETCH_SOURCE_BLOCK_COUNT: u32 = PREFETCH_BLOCK_CAPACITY;

/// Configuration flag: JIT KV prefetch
/// (`SPARK_REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH`).
pub const CONFIGURATION_FLAG_JIT_KV_PREFETCH: u32 = 0x0000_0001;
/// Configuration flag: decode batching
/// (`SPARK_REQUEST_API_CONFIGURATION_FLAG_DECODE_BATCHING`).
pub const CONFIGURATION_FLAG_DECODE_BATCHING: u32 = 0x0000_0002;
/// Configuration flag: prefix cohorting
/// (`SPARK_REQUEST_API_CONFIGURATION_FLAG_PREFIX_COHORTING`).
pub const CONFIGURATION_FLAG_PREFIX_COHORTING: u32 = 0x0000_0004;
/// Configuration flag: prefill batching
/// (`SPARK_REQUEST_API_CONFIGURATION_FLAG_PREFILL_BATCHING`).
pub const CONFIGURATION_FLAG_PREFILL_BATCHING: u32 = 0x0000_0008;
/// Configuration flag: async JIT KV prefetch
/// (`SPARK_REQUEST_API_CONFIGURATION_FLAG_ASYNC_JIT_KV_PREFETCH`).
pub const CONFIGURATION_FLAG_ASYNC_JIT_KV_PREFETCH: u32 = 0x0000_0010;
/// Configuration flag: queue-aware prefix-cache eviction
/// (`SPARK_REQUEST_API_CONFIGURATION_FLAG_QUEUE_AWARE_PREFIX_CACHE_EVICTION`).
pub const CONFIGURATION_FLAG_QUEUE_AWARE_PREFIX_CACHE_EVICTION: u32 = 0x0000_0020;
/// Configuration flag: dspark speculative decode
/// (`SPARK_REQUEST_API_CONFIGURATION_FLAG_DSPARK_SPECULATIVE_DECODE`).
pub const CONFIGURATION_FLAG_DSPARK_SPECULATIVE_DECODE: u32 = 0x0000_0040;
/// Configuration flag: MTP commit
/// (`SPARK_REQUEST_API_CONFIGURATION_FLAG_MTP_COMMIT`).
pub const CONFIGURATION_FLAG_MTP_COMMIT: u32 = 0x0000_0080;
/// Configuration flag: adaptive pipeline batching
/// (`SPARK_REQUEST_API_CONFIGURATION_FLAG_ADAPTIVE_PIPELINE_BATCHING`).
pub const CONFIGURATION_FLAG_ADAPTIVE_PIPELINE_BATCHING: u32 = 0x0000_0100;
/// Configuration flag: MTP force enable
/// (`SPARK_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE`).
pub const CONFIGURATION_FLAG_MTP_FORCE_ENABLE: u32 = 0x0000_0200;
/// Configuration flag: prefer dspark speculation
/// (`SPARK_REQUEST_API_CONFIGURATION_FLAG_PREFER_DSPARK_SPECULATION`).
pub const CONFIGURATION_FLAG_PREFER_DSPARK_SPECULATION: u32 = 0x0000_0800;
/// Default configuration flags (`SPARK_REQUEST_API_CONFIGURATION_DEFAULT_FLAGS`).
pub const CONFIGURATION_DEFAULT_FLAGS: u32 = CONFIGURATION_FLAG_JIT_KV_PREFETCH
    | CONFIGURATION_FLAG_DECODE_BATCHING
    | CONFIGURATION_FLAG_PREFIX_COHORTING
    | CONFIGURATION_FLAG_PREFILL_BATCHING
    | CONFIGURATION_FLAG_QUEUE_AWARE_PREFIX_CACHE_EVICTION;
/// All recognized configuration flags (`SPARK_REQUEST_API_CONFIGURATION_KNOWN_FLAGS`).
pub const CONFIGURATION_KNOWN_FLAGS: u32 = CONFIGURATION_DEFAULT_FLAGS
    | CONFIGURATION_FLAG_ASYNC_JIT_KV_PREFETCH
    | CONFIGURATION_FLAG_DSPARK_SPECULATIVE_DECODE
    | CONFIGURATION_FLAG_MTP_COMMIT
    | CONFIGURATION_FLAG_MTP_FORCE_ENABLE
    | CONFIGURATION_FLAG_ADAPTIVE_PIPELINE_BATCHING
    | CONFIGURATION_FLAG_PREFER_DSPARK_SPECULATION;

/// Maximum MTP draft tokens (`SPARK_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT`).
pub const MTP_MAX_DRAFT_TOKEN_COUNT: u32 = 8;
/// Initial per-slot MTP draft token budget
/// (`SPARK_REQUEST_API_MTP_INITIAL_DRAFT_TOKEN_COUNT`).
pub const MTP_INITIAL_DRAFT_TOKEN_COUNT: u32 = mtp_tree::CANDIDATE_COUNT;
/// Commit-EMA divisor (`SPARK_REQUEST_API_MTP_COMMIT_EMA_DIVISOR`).
pub const MTP_COMMIT_EMA_DIVISOR: i32 = 4;
/// Initial commit-EMA (`SPARK_REQUEST_API_MTP_COMMIT_EMA_INITIAL_MILLI`).
pub const MTP_COMMIT_EMA_INITIAL_MILLI: u32 = 2900;
/// Suppression threshold (`SPARK_REQUEST_API_MTP_SUPPRESS_THRESHOLD_MILLI`).
pub const MTP_SUPPRESS_THRESHOLD_MILLI: u32 = 1150;
/// Reprobe interval in committed tokens (`SPARK_REQUEST_API_MTP_REPROBE_INTERVAL`).
pub const MTP_REPROBE_INTERVAL: u32 = 16;
/// In-flight prefill wave limit (`SPARK_REQUEST_API_PREFILL_INFLIGHT_WAVE_LIMIT`).
pub const PREFILL_INFLIGHT_WAVE_LIMIT: u32 = 12;
/// Pending async prefetch capacity (`SPARK_REQUEST_API_PENDING_PREFETCH_CAPACITY`).
pub const PENDING_PREFETCH_CAPACITY: usize = 8;
/// Handle-hash slot count (`SPARK_REQUEST_API_SLOT_HASH_SLOTS`).
pub const SLOT_HASH_SLOTS: usize = 4096;
/// No-slot sentinel (`SPARK_REQUEST_API_NO_SLOT`).
pub const NO_SLOT: u32 = u32::MAX;
/// Prefix-family group capacity (`SPARK_REQUEST_API_PREFIX_FAMILY_GROUP_CAPACITY`).
pub(crate) const PREFIX_FAMILY_GROUP_CAPACITY: usize = 256;
/// Maximum speculative tokens (`SPARK_REQUEST_MODEL_MAX_SPECULATIVE_TOKENS`).
pub const MAX_SPECULATIVE_TOKENS: u32 = 7;

/// Errors mirroring the `SparkStatus` codes the C entry points return.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RequestApiError {
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
    /// `SPARK_STATUS_INTERNAL_ERROR`.
    #[error("internal error")]
    InternalError,
    /// `SPARK_STATUS_MODULE_NOT_VALIDATED`.
    #[error("module not validated")]
    ModuleNotValidated,
    /// `SPARK_STATUS_HASH_MISMATCH`.
    #[error("hash mismatch")]
    HashMismatch,
}

impl From<SchedulerError> for RequestApiError {
    fn from(error: SchedulerError) -> Self {
        match error {
            SchedulerError::InvalidArgument => RequestApiError::InvalidArgument,
            SchedulerError::CapacityExceeded => RequestApiError::CapacityExceeded,
            SchedulerError::NotFound => RequestApiError::NotFound,
            SchedulerError::Busy => RequestApiError::Busy,
            // The C scheduler has no DUPLICATE/MODULE_NOT_VALIDATED path the
            // request API observes; map conservatively.
            SchedulerError::Duplicate => RequestApiError::InvalidArgument,
            SchedulerError::ModuleNotValidated => RequestApiError::ModuleNotValidated,
            SchedulerError::InternalError => RequestApiError::InternalError,
        }
    }
}

impl From<PrefixCacheError> for RequestApiError {
    fn from(error: PrefixCacheError) -> Self {
        match error {
            PrefixCacheError::InvalidArgument => RequestApiError::InvalidArgument,
            PrefixCacheError::CapacityExceeded => RequestApiError::CapacityExceeded,
            PrefixCacheError::NotFound => RequestApiError::NotFound,
            PrefixCacheError::Busy => RequestApiError::Busy,
            PrefixCacheError::Duplicate => RequestApiError::InvalidArgument,
            PrefixCacheError::ModuleNotValidated => RequestApiError::ModuleNotValidated,
            PrefixCacheError::InternalError => RequestApiError::InternalError,
        }
    }
}

impl From<RequestApiError> for PromptPipelineError {
    fn from(error: RequestApiError) -> Self {
        match error {
            RequestApiError::InvalidArgument => PromptPipelineError::InvalidArgument,
            RequestApiError::CapacityExceeded => PromptPipelineError::CapacityExceeded,
            RequestApiError::NotFound => PromptPipelineError::NotFound,
            RequestApiError::Busy => PromptPipelineError::Busy,
            RequestApiError::InternalError | RequestApiError::HashMismatch => {
                PromptPipelineError::InternalError
            }
            RequestApiError::ModuleNotValidated => PromptPipelineError::Unsupported,
        }
    }
}

/// Configuration (`SparkRequestApiConfiguration`, minus ABI words and the
/// caller-owned slot array — the Rust API owns its slots).
pub struct Configuration {
    /// `CONFIGURATION_FLAG_*` mask; 0 normalizes to [`CONFIGURATION_DEFAULT_FLAGS`].
    pub configuration_flags: u32,
    /// Slot capacity.
    pub request_capacity: u32,
    /// Prefetch lookahead; 0 normalizes to [`DEFAULT_PREFETCH_LOOKAHEAD_REQUEST_COUNT`]
    /// clamped to `request_capacity`.
    pub prefetch_lookahead_request_count: u32,
    /// Prefetch lane count; 0 normalizes to [`DEFAULT_PREFETCH_LANE_COUNT`].
    pub prefetch_lane_count: u32,
    /// Decode batch target; 0 normalizes to [`MAX_DISPATCH_REQUEST_COUNT`].
    pub decode_batch_target: u32,
    /// Resident KV block budget; 0 (or >= the arena's block count) disables
    /// the residency policy.
    pub max_resident_kv_block_count: u32,
    /// Decode execution row capacity; 0 normalizes to
    /// `decode_batch_target * (max speculative tokens + 1)`.
    pub decode_execution_row_capacity: u32,
    /// Scheduler (owns the prefix cache, which owns the KV arena).
    pub scheduler: Scheduler,
    /// JIT KV prefetch dispatch seam; required when
    /// [`CONFIGURATION_FLAG_JIT_KV_PREFETCH`] is set.
    pub kv_prefetch_backend: Option<Box<dyn KvPrefetchBackend>>,
    /// Opaque model speculator (the drafter seam); required when
    /// [`CONFIGURATION_FLAG_DSPARK_SPECULATIVE_DECODE`] is set.
    pub model_speculator: Option<Box<dyn RequestModelSeam>>,
    /// Output vocabulary bound used by the MTP tree resolver's token-id
    /// validation (`SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT` in C).
    pub output_vocab_count: u32,
}

impl Configuration {
    /// `SparkRequestApiConfigurationUseAsyncKvCachePrefetchBackend`: set the
    /// JIT + ASYNC flags and install the prefetch backend. The async backend
    /// device-copy engine itself is not ported; any
    /// [`KvPrefetchBackend`] implementation may be installed here.
    pub fn use_async_kv_cache_prefetch_backend(
        &mut self,
        backend: Box<dyn KvPrefetchBackend>,
    ) -> Result<(), RequestApiError> {
        let mut configuration_flags = normalize_configuration_flags(self.configuration_flags);
        configuration_flags |=
            CONFIGURATION_FLAG_JIT_KV_PREFETCH | CONFIGURATION_FLAG_ASYNC_JIT_KV_PREFETCH;
        self.configuration_flags = configuration_flags;
        self.kv_prefetch_backend = Some(backend);
        Ok(())
    }
}

/// `SparkRequestApiNormalizeConfigurationFlags`.
pub(crate) fn normalize_configuration_flags(configuration_flags: u32) -> u32 {
    if configuration_flags == 0 {
        CONFIGURATION_DEFAULT_FLAGS
    } else {
        configuration_flags
    }
}

/// `SparkRequestApiNormalizePrefetchLookaheadRequestCount`.
fn normalize_prefetch_lookahead_request_count(lookahead: u32, request_capacity: u32) -> u32 {
    let lookahead =
        if lookahead == 0 { DEFAULT_PREFETCH_LOOKAHEAD_REQUEST_COUNT } else { lookahead };
    lookahead.min(request_capacity)
}

/// `SparkRequestApiNormalizePrefetchLaneCount`.
fn normalize_prefetch_lane_count(prefetch_lane_count: u32) -> u32 {
    if prefetch_lane_count == 0 {
        DEFAULT_PREFETCH_LANE_COUNT
    } else {
        prefetch_lane_count
    }
}

/// `SparkRequestApiNormalizeDecodeBatchTarget`.
fn normalize_decode_batch_target(decode_batch_target: u32) -> u32 {
    if decode_batch_target == 0 {
        MAX_DISPATCH_REQUEST_COUNT
    } else {
        decode_batch_target.min(MAX_DISPATCH_REQUEST_COUNT)
    }
}

/// `SparkRequestApiNormalizeDecodeExecutionRowCapacity`.
fn normalize_decode_execution_row_capacity(configuration: &Configuration) -> u32 {
    if configuration.decode_execution_row_capacity != 0 {
        return configuration.decode_execution_row_capacity;
    }
    let decode_batch_target = normalize_decode_batch_target(configuration.decode_batch_target);
    let maximum_speculative_token_count = MAX_SPECULATIVE_TOKENS.max(MTP_MAX_DRAFT_TOKEN_COUNT);
    decode_batch_target * (maximum_speculative_token_count + 1)
}

/// `SparkRequestApiNormalizeMaxResidentKvBlockCount`.
fn normalize_max_resident_kv_block_count(configuration: &Configuration) -> u32 {
    let Some(prefix_cache) = configuration.scheduler.prefix_cache() else {
        return 0;
    };
    let physical_block_count = prefix_cache.arena().physical_block_count();
    if configuration.max_resident_kv_block_count == 0
        || configuration.max_resident_kv_block_count >= physical_block_count
    {
        return 0;
    }
    configuration.max_resident_kv_block_count
}

/// `SparkRequestApiHashHandle` (exact C mixing).
fn hash_handle(handle: RequestApiHandle) -> usize {
    let mut hash = handle;
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    (hash % SLOT_HASH_SLOTS as u64) as usize
}

/// Request/session API (`SparkRequestApi`). Statistics counters and the
/// mutable configuration words are public fields, mirroring the C struct.
pub struct RequestApi {
    pub configuration_flags: u32,
    pub request_capacity: u32,
    pub queued_request_count: u32,
    pub running_request_count: u32,
    pub completed_request_count: u32,
    pub cancelled_request_count: u32,
    pub prefetch_lookahead_request_count: u32,
    pub prefetch_lane_count: u32,
    pub decode_batch_target: u32,
    pub max_resident_kv_block_count: u32,
    pub decode_execution_row_capacity: u32,
    pub(crate) free_slot_head: u32,
    pub(crate) next_handle: u64,
    pub(crate) next_sequence_id: u64,
    pub(crate) submission_counter: u64,
    pub submitted_request_count: u64,
    pub scheduled_prefill_dispatch_count: u64,
    pub stale_prefill_completion_count: u64,
    pub scheduled_decode_dispatch_count: u64,
    pub jit_prefetch_dispatch_count: u64,
    pub jit_prefetch_block_count: u64,
    pub jit_residency_eviction_count: u64,
    pub jit_residency_protected_block_count: u64,
    pub async_jit_prefetch_start_count: u64,
    pub async_jit_prefetch_poll_count: u64,
    pub async_jit_prefetch_completion_count: u64,
    pub next_prefetch_id: u64,
    pub lookahead_protection_sweep_count: u64,
    pub lookahead_protected_block_count: u64,
    pub prefix_family_dispatch_count: u64,
    pub prefix_family_saved_prompt_token_count: u64,
    pub prefix_family_member_count: u64,
    pub dspark_tap_capture_dispatch_count: u64,
    pub dspark_draft_ready_count: u64,
    pub dspark_verify_dispatch_count: u64,
    pub dspark_accepted_draft_token_count: u64,
    pub dspark_committed_token_count: u64,
    pub dspark_rejected_token_count: u64,
    pub mtp_draft_ready_count: u64,
    pub mtp_verify_dispatch_count: u64,
    pub mtp_accepted_draft_token_count: u64,
    pub mtp_committed_token_count: u64,
    pub mtp_rejected_token_count: u64,
    /// Output vocab bound for the MTP tree resolver.
    pub output_vocab_count: u32,
    /// The scheduler; in a `RefCell` so the `prompt_pipeline::RequestApi`
    /// `&self` methods (KV block table view) can drive it. Inherent `&mut
    /// self` methods use `get_mut` and pay nothing.
    scheduler: RefCell<Scheduler>,
    pub(crate) slots: Vec<Slot>,
    /// JIT KV prefetch dispatch seam.
    pub kv_prefetch_backend: Option<Box<dyn KvPrefetchBackend>>,
    /// Opaque model speculator (drafter seam).
    pub model_speculator: Option<Box<dyn RequestModelSeam>>,
    /// Arena resident block capacity, recovered at init (the arena has no
    /// public getter — see module docs).
    pub(crate) arena_resident_block_capacity: u32,
    pub(crate) slot_handle_hash_heads: Vec<u32>,
    pub(crate) pending_prefetches: Vec<prefetch::PendingPrefetch>,
    /// Dispatch staged by the `prompt_pipeline::RequestApi` trait's
    /// `schedule_next` for the follow-up trait calls (the trait's minimized
    /// dispatch surface carries only `accepted`/`kind`).
    pub(crate) staged_dispatch: RefCell<Option<Dispatch>>,
}

impl RequestApi {
    /// `SparkRequestApiInitialize` (configuration validation included).
    pub fn new(configuration: Configuration) -> Result<Self, RequestApiError> {
        // SparkRequestApiValidateConfiguration.
        if configuration.request_capacity == 0 {
            return Err(RequestApiError::InvalidArgument);
        }
        let configuration_flags = normalize_configuration_flags(configuration.configuration_flags);
        let prefetch_lane_count = normalize_prefetch_lane_count(configuration.prefetch_lane_count);
        let decode_batch_target = normalize_decode_batch_target(configuration.decode_batch_target);
        let decode_execution_row_capacity = normalize_decode_execution_row_capacity(&configuration);
        if configuration_flags & !CONFIGURATION_KNOWN_FLAGS != 0
            || prefetch_lane_count == 0
            || prefetch_lane_count > MAX_PREFETCH_LANE_COUNT
            || decode_execution_row_capacity < decode_batch_target
        {
            return Err(RequestApiError::InvalidArgument);
        }
        if configuration_flags & CONFIGURATION_FLAG_ASYNC_JIT_KV_PREFETCH != 0
            && configuration_flags & CONFIGURATION_FLAG_JIT_KV_PREFETCH == 0
        {
            return Err(RequestApiError::InvalidArgument);
        }
        if configuration_flags & CONFIGURATION_FLAG_QUEUE_AWARE_PREFIX_CACHE_EVICTION != 0
            && configuration.scheduler.prefix_cache().is_none()
        {
            return Err(RequestApiError::InvalidArgument);
        }
        if configuration_flags & CONFIGURATION_FLAG_DSPARK_SPECULATIVE_DECODE != 0 {
            match &configuration.model_speculator {
                Some(speculator) if speculator.is_valid() => {}
                _ => return Err(RequestApiError::InvalidArgument),
            }
        }
        if configuration_flags & CONFIGURATION_FLAG_JIT_KV_PREFETCH != 0
            && (configuration.scheduler.prefix_cache().is_none()
                || configuration.kv_prefetch_backend.is_none())
        {
            return Err(RequestApiError::InvalidArgument);
        }

        let max_resident_kv_block_count = normalize_max_resident_kv_block_count(&configuration);
        let prefetch_lookahead_request_count = configuration.prefetch_lookahead_request_count;
        let request_capacity = configuration.request_capacity;
        let mut slots = Vec::with_capacity(request_capacity as usize);
        for slot_index in 0..request_capacity {
            let mut slot = Slot::free();
            slot.free_slot_next =
                if slot_index + 1 < request_capacity { slot_index + 1 } else { NO_SLOT };
            slots.push(slot);
        }
        let Configuration {
            mut scheduler,
            kv_prefetch_backend,
            model_speculator,
            output_vocab_count,
            ..
        } = configuration;
        // The C reads the arena's resident_block_capacity directly; the Rust
        // arena keeps it private, so recover it exactly (see module docs).
        let arena_resident_block_capacity = match scheduler.prefix_cache_mut() {
            Some(prefix_cache) => prefetch::probe_resident_block_capacity(prefix_cache.arena_mut()),
            None => 0,
        };
        Ok(RequestApi {
            configuration_flags,
            request_capacity,
            queued_request_count: 0,
            running_request_count: 0,
            completed_request_count: 0,
            cancelled_request_count: 0,
            prefetch_lookahead_request_count: normalize_prefetch_lookahead_request_count(
                prefetch_lookahead_request_count,
                request_capacity,
            ),
            prefetch_lane_count,
            decode_batch_target,
            max_resident_kv_block_count,
            decode_execution_row_capacity,
            free_slot_head: 0,
            next_handle: 1,
            next_sequence_id: 1,
            submission_counter: 0,
            submitted_request_count: 0,
            scheduled_prefill_dispatch_count: 0,
            stale_prefill_completion_count: 0,
            scheduled_decode_dispatch_count: 0,
            jit_prefetch_dispatch_count: 0,
            jit_prefetch_block_count: 0,
            jit_residency_eviction_count: 0,
            jit_residency_protected_block_count: 0,
            async_jit_prefetch_start_count: 0,
            async_jit_prefetch_poll_count: 0,
            async_jit_prefetch_completion_count: 0,
            next_prefetch_id: 1,
            lookahead_protection_sweep_count: 0,
            lookahead_protected_block_count: 0,
            prefix_family_dispatch_count: 0,
            prefix_family_saved_prompt_token_count: 0,
            prefix_family_member_count: 0,
            dspark_tap_capture_dispatch_count: 0,
            dspark_draft_ready_count: 0,
            dspark_verify_dispatch_count: 0,
            dspark_accepted_draft_token_count: 0,
            dspark_committed_token_count: 0,
            dspark_rejected_token_count: 0,
            mtp_draft_ready_count: 0,
            mtp_verify_dispatch_count: 0,
            mtp_accepted_draft_token_count: 0,
            mtp_committed_token_count: 0,
            mtp_rejected_token_count: 0,
            output_vocab_count,
            scheduler: RefCell::new(scheduler),
            slots,
            kv_prefetch_backend,
            model_speculator,
            arena_resident_block_capacity,
            slot_handle_hash_heads: vec![NO_SLOT; SLOT_HASH_SLOTS],
            pending_prefetches: vec![
                prefetch::PendingPrefetch::default();
                PENDING_PREFETCH_CAPACITY
            ],
            staged_dispatch: RefCell::new(None),
        })
    }

    /// Scheduler access (the `RefCell` lets the `prompt_pipeline::RequestApi`
    /// `&self` methods drive the KV block table build).
    pub fn scheduler(&self) -> &RefCell<Scheduler> {
        &self.scheduler
    }

    /// Read-only slot access (the C struct fields are directly readable).
    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    /// Mutable slot access (the C's caller-owned `request_slots` array;
    /// tests poke scheduling fields directly exactly as the C tests do).
    pub fn slots_mut(&mut self) -> &mut [Slot] {
        &mut self.slots
    }

    /// `SparkRequestApiValidate`.
    pub(crate) fn validate(&self) -> Result<(), RequestApiError> {
        if self.request_capacity == 0
            || self.decode_batch_target == 0
            || self.decode_execution_row_capacity < self.decode_batch_target
            || self.slots.is_empty()
        {
            return Err(RequestApiError::InvalidArgument);
        }
        Ok(())
    }

    // -- configuration flag predicates (the C `SparkRequestApi*IsEnabled`) --

    pub(crate) fn jit_prefetch_is_enabled(&self) -> bool {
        self.configuration_flags & CONFIGURATION_FLAG_JIT_KV_PREFETCH != 0
    }

    pub(crate) fn async_jit_prefetch_is_enabled(&self) -> bool {
        self.configuration_flags & CONFIGURATION_FLAG_ASYNC_JIT_KV_PREFETCH != 0
    }

    pub(crate) fn queue_aware_prefix_cache_eviction_is_enabled(&self) -> bool {
        self.configuration_flags & CONFIGURATION_FLAG_QUEUE_AWARE_PREFIX_CACHE_EVICTION != 0
    }

    pub(crate) fn mtp_commit_is_enabled(&self) -> bool {
        self.configuration_flags & CONFIGURATION_FLAG_MTP_COMMIT != 0
    }

    pub(crate) fn decode_batching_is_enabled(&self) -> bool {
        self.configuration_flags & CONFIGURATION_FLAG_DECODE_BATCHING != 0
    }

    pub(crate) fn adaptive_pipeline_batching_is_enabled(&self) -> bool {
        self.configuration_flags & CONFIGURATION_FLAG_ADAPTIVE_PIPELINE_BATCHING != 0
    }

    pub(crate) fn prefix_cohorting_is_enabled(&self) -> bool {
        self.configuration_flags & CONFIGURATION_FLAG_PREFIX_COHORTING != 0
    }

    pub(crate) fn prefill_batching_is_enabled(&self) -> bool {
        self.configuration_flags & CONFIGURATION_FLAG_PREFILL_BATCHING != 0
    }

    /// `SparkRequestApiCrossSequencePrefixReuseIsEnabled`.
    pub(crate) fn cross_sequence_prefix_reuse_is_enabled(&self) -> bool {
        self.scheduler.borrow().configuration_flags()
            & spark_sched::scheduler::CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE
            != 0
    }

    /// `SparkRequestApiConfigurationHasChunkedPrefill`.
    pub(crate) fn configuration_has_chunked_prefill(&self) -> bool {
        self.scheduler.borrow().configuration_flags()
            & spark_sched::scheduler::CONFIGURATION_FLAG_CHUNKED_PREFILL
            != 0
    }

    /// `SparkRequestModelSpeculationIsEnabled`: the DSPARK flag with a
    /// speculator installed.
    pub(crate) fn speculation_is_enabled(&self) -> bool {
        self.configuration_flags & CONFIGURATION_FLAG_DSPARK_SPECULATIVE_DECODE != 0
            && self.model_speculator.is_some()
    }

    /// `SparkRequestModelSlotCanSpeculate`: speculation enabled && the slot
    /// hasn't disabled speculation && the provider policy allows it.
    pub(crate) fn slot_can_speculate(&self, slot_index: u32) -> bool {
        if !self.speculation_is_enabled() {
            return false;
        }
        let slot = &self.slots[slot_index as usize];
        if slot.flags & slot::REQUEST_FLAG_DISABLE_SPECULATION != 0 {
            return false;
        }
        self.model_speculator.as_ref().is_some_and(|speculator| speculator.slot_can_speculate(slot))
    }

    // -- slot table (the C's intrusive free list + handle hash) --------------

    /// `SparkRequestApiFindFreeSlot`: pop a free slot index.
    pub(crate) fn find_free_slot(&mut self) -> Option<u32> {
        if self.free_slot_head == NO_SLOT || self.free_slot_head >= self.request_capacity {
            return None;
        }
        let slot_index = self.free_slot_head;
        let slot = &mut self.slots[slot_index as usize];
        if slot.state != STATE_FREE {
            return None;
        }
        self.free_slot_head = slot.free_slot_next;
        slot.free_slot_next = NO_SLOT;
        Some(slot_index)
    }

    /// `SparkRequestApiFindSlotByHandle`.
    pub(crate) fn find_slot_by_handle(&self, handle: RequestApiHandle) -> Option<u32> {
        if handle == INVALID_HANDLE {
            return None;
        }
        let mut slot_index = self.slot_handle_hash_heads[hash_handle(handle)];
        while slot_index != NO_SLOT && slot_index < self.request_capacity {
            let slot = &self.slots[slot_index as usize];
            if slot.state != STATE_FREE && slot.handle == handle {
                return Some(slot_index);
            }
            slot_index = slot.handle_hash_next;
        }
        None
    }

    /// `SparkRequestApiInsertSlotHash`.
    pub(crate) fn insert_slot_hash(&mut self, slot_index: u32) {
        if slot_index >= self.request_capacity
            || self.slots[slot_index as usize].handle == INVALID_HANDLE
        {
            return;
        }
        let hash_slot = hash_handle(self.slots[slot_index as usize].handle);
        self.slots[slot_index as usize].handle_hash_next = self.slot_handle_hash_heads[hash_slot];
        self.slot_handle_hash_heads[hash_slot] = slot_index;
    }

    /// `SparkRequestApiRemoveSlotHash`.
    pub(crate) fn remove_slot_hash(&mut self, slot_index: u32) {
        if slot_index >= self.request_capacity {
            return;
        }
        let handle = self.slots[slot_index as usize].handle;
        if handle == INVALID_HANDLE {
            return;
        }
        let hash_slot = hash_handle(handle);
        let mut current_slot = self.slot_handle_hash_heads[hash_slot];
        let mut previous_slot = NO_SLOT;
        while current_slot != NO_SLOT {
            if current_slot == slot_index {
                if previous_slot == NO_SLOT {
                    self.slot_handle_hash_heads[hash_slot] =
                        self.slots[current_slot as usize].handle_hash_next;
                } else {
                    self.slots[previous_slot as usize].handle_hash_next =
                        self.slots[current_slot as usize].handle_hash_next;
                }
                self.slots[current_slot as usize].handle_hash_next = NO_SLOT;
                return;
            }
            previous_slot = current_slot;
            current_slot = self.slots[current_slot as usize].handle_hash_next;
        }
    }

    // -- entry points ---------------------------------------------------------

    /// `SparkRequestApiCurrentPipelineBatchWidth`.
    pub fn current_pipeline_batch_width(&mut self) -> u32 {
        if self.decode_batch_target == 0 {
            return 0;
        }
        if !self.adaptive_pipeline_batching_is_enabled() {
            return self.decode_batch_target;
        }
        let mut highest_ready_priority = 0u32;
        for slot in &self.slots {
            if slot.is_ready_for_dispatch() && slot.priority > highest_ready_priority {
                highest_ready_priority = slot.priority;
            }
        }
        if highest_ready_priority == 0 {
            return 0;
        }
        let mut ready_request_count = 0u32;
        for slot in &self.slots {
            if slot.is_active() && slot.priority == highest_ready_priority {
                ready_request_count += 1;
            }
        }
        self.scheduler
            .get_mut()
            .select_pipeline_batch_width(ready_request_count, self.decode_batch_target)
    }

    /// `SparkRequestApiSubmit`. Returns the request handle.
    pub fn submit(&mut self, request: &SubmitRequest) -> Result<RequestApiHandle, RequestApiError> {
        self.validate()?;
        // SparkRequestApiValidateSubmitRequest.
        if request.flags & !REQUEST_KNOWN_FLAGS != 0 || request.prompt_token_count == 0 {
            return Err(RequestApiError::InvalidArgument);
        }
        let slot_index = self.find_free_slot().ok_or(RequestApiError::CapacityExceeded)?;

        let priority = if request.flags & REQUEST_FLAG_REALTIME != 0 {
            REALTIME_PRIORITY
        } else if request.priority == 0 {
            DEFAULT_PRIORITY
        } else {
            request.priority
        };
        let sequence_id = if request.sequence_id != 0 {
            request.sequence_id
        } else {
            let sequence_id = self.next_sequence_id;
            self.next_sequence_id += 1;
            sequence_id
        };
        let handle = self.next_handle;
        self.next_handle += 1;
        let submission_order = self.submission_counter;
        self.submission_counter += 1;

        let slot = &mut self.slots[slot_index as usize];
        *slot = Slot::free();
        slot.state = STATE_QUEUED_PREFILL;
        slot.flags = request.flags;
        slot.priority = priority;
        slot.prompt_token_count = request.prompt_token_count;
        slot.thinking_token_budget = request.thinking_token_budget;
        slot.output_token_budget = request.output_token_budget;
        slot.remaining_thinking_token_budget = request.thinking_token_budget;
        slot.remaining_output_token_budget = request.output_token_budget;
        slot.max_prefill_tokens_per_step = request.max_prefill_tokens_per_step;
        slot.mtp_next_draft_token_budget = MTP_INITIAL_DRAFT_TOKEN_COUNT;
        slot.request_id = request.request_id;
        slot.sequence_id = sequence_id;
        slot.handle = handle;
        slot.submission_order = submission_order;
        // Copy the prompt; zero-pad to prompt_token_count on the
        // tokenizer-overflow path (see module docs).
        slot.prompt_token_ids = request.prompt_token_ids.clone();
        slot.prompt_token_ids.resize(request.prompt_token_count as usize, 0);
        self.insert_slot_hash(slot_index);

        self.queued_request_count += 1;
        self.submitted_request_count += 1;
        Ok(handle)
    }

    /// `SparkRequestApiGetRequestCacheState`.
    pub fn get_request_cache_state(
        &mut self,
        handle: RequestApiHandle,
    ) -> Result<CacheState, RequestApiError> {
        self.validate()?;
        let slot_index = self.find_slot_by_handle(handle).ok_or(RequestApiError::NotFound)?;
        let slot = &self.slots[slot_index as usize];
        let mut cache_state = CacheState {
            state: slot.state,
            computed_prompt_token_count: slot.computed_prompt_token_count,
            last_committed_prefix_token_count: slot.last_committed_prefix_token_count,
            physical_block_count: 0,
            request_id: slot.request_id,
            sequence_id: slot.sequence_id,
            last_committed_prefix_hash: slot.last_committed_prefix_hash,
        };
        if slot.computed_prompt_token_count != 0 {
            let sequence_id = slot.sequence_id;
            let computed = slot.computed_prompt_token_count;
            let scheduler = self.scheduler.get_mut();
            if let Some(prefix_cache) = scheduler.prefix_cache_mut() {
                let probe = prefix_cache.probe_sequence_residency(sequence_id, computed)?;
                cache_state.physical_block_count = probe.physical_block_count;
            }
        }
        Ok(cache_state)
    }

    /// `SparkRequestModelCancelSequence` through the seam, when enabled
    /// (the head of `SparkRequestApiReleaseSlotSequence`).
    pub(crate) fn cancel_speculator_sequence(
        &mut self,
        sequence_id: u64,
    ) -> Result<(), RequestApiError> {
        if self.speculation_is_enabled() {
            if let Some(speculator) = &mut self.model_speculator {
                let status = speculator.cancel_sequence(sequence_id);
                if status != Err(RequestApiError::NotFound) {
                    status?;
                }
            }
        }
        Ok(())
    }

    /// `SparkRequestApiReleaseSlotSequence`.
    pub(crate) fn release_slot_sequence(&mut self, slot_index: u32) -> Result<(), RequestApiError> {
        let sequence_id = self.slots[slot_index as usize].sequence_id;
        if sequence_id == 0 {
            return Ok(());
        }
        self.cancel_speculator_sequence(sequence_id)?;
        self.scheduler.get_mut().release_sequence(sequence_id)?;
        self.slots[slot_index as usize].sequence_id = 0;
        Ok(())
    }

    /// `SparkRequestApiFinishRequestGeneration`.
    pub fn finish_request_generation(
        &mut self,
        handle: RequestApiHandle,
    ) -> Result<(), RequestApiError> {
        self.validate()?;
        let slot_index = self.find_slot_by_handle(handle).ok_or(RequestApiError::NotFound)?;
        let slot = &self.slots[slot_index as usize];
        let state = slot.state;
        if state == STATE_COMPLETED {
            return Ok(());
        }
        if state == STATE_CANCELLED {
            return Err(RequestApiError::NotFound);
        }
        if state == STATE_RUNNING_PREFILL
            || state == STATE_RUNNING_DECODE
            || state == STATE_RUNNING_SPECULATIVE_VERIFY
            || state == STATE_WAITING_PREFIX_COHORT
        {
            return Err(RequestApiError::Busy);
        }
        if state == STATE_QUEUED_PREFILL {
            if self.queued_request_count == 0 {
                return Err(RequestApiError::InvalidArgument);
            }
            self.queued_request_count -= 1;
        } else if state != STATE_READY_DECODE && state != STATE_READY_SPECULATIVE_VERIFY {
            return Err(RequestApiError::InvalidArgument);
        }
        if state == STATE_READY_SPECULATIVE_VERIFY && self.speculation_is_enabled() {
            let sequence_id = self.slots[slot_index as usize].sequence_id;
            self.cancel_speculator_sequence(sequence_id)?;
        }
        let slot = &mut self.slots[slot_index as usize];
        slot.remaining_thinking_token_budget = 0;
        slot.remaining_output_token_budget = 0;
        slot.state = STATE_COMPLETED;
        self.completed_request_count += 1;
        Ok(())
    }

    /// `SparkRequestApiCancelRequest`.
    pub fn cancel_request(&mut self, handle: RequestApiHandle) -> Result<(), RequestApiError> {
        self.validate()?;
        let slot_index = self.find_slot_by_handle(handle).ok_or(RequestApiError::NotFound)?;
        let state = self.slots[slot_index as usize].state;
        if state == STATE_CANCELLED {
            return Ok(());
        }
        if state == STATE_RUNNING_PREFILL || state == STATE_RUNNING_DECODE {
            return Err(RequestApiError::Busy);
        }
        if state == STATE_QUEUED_PREFILL {
            self.queued_request_count -= 1;
        }
        if state == STATE_COMPLETED {
            self.completed_request_count -= 1;
        }
        self.slots[slot_index as usize].state = STATE_CANCELLED;
        self.cancelled_request_count += 1;
        self.release_slot_sequence(slot_index)
    }

    /// `SparkRequestApiReleaseCompletedRequest`.
    pub fn release_completed_request(
        &mut self,
        handle: RequestApiHandle,
    ) -> Result<(), RequestApiError> {
        self.validate()?;
        let slot_index = self.find_slot_by_handle(handle).ok_or(RequestApiError::NotFound)?;
        let state = self.slots[slot_index as usize].state;
        if state != STATE_COMPLETED && state != STATE_CANCELLED {
            return Err(RequestApiError::Busy);
        }
        self.release_slot_sequence(slot_index)?;
        self.remove_slot_hash(slot_index);
        self.slots[slot_index as usize] = Slot::free();
        self.slots[slot_index as usize].free_slot_next = self.free_slot_head;
        self.free_slot_head = slot_index;
        Ok(())
    }

    /// `SparkRequestApiSlotRemainingDecodeBudget` (free-function form in C).
    pub fn slot_remaining_decode_budget(&self, slot_index: u32) -> u32 {
        self.slots.get(slot_index as usize).map_or(0, Slot::remaining_decode_budget)
    }

    /// `SparkRequestApiSlotHasRealtimePriority` (free-function form in C).
    pub fn slot_has_realtime_priority(&self, slot_index: u32) -> bool {
        self.slots.get(slot_index as usize).is_some_and(Slot::has_realtime_priority)
    }

    /// `SparkRequestApiAssignDraftBudgets`: wave-level draft-budget
    /// assignment via the row allocator. Returns the total rows assigned
    /// including one base row per eligible slot, clamped to the cap.
    pub fn assign_draft_budgets(&mut self, firing_row_cap: u32) -> u32 {
        if firing_row_cap == 0 {
            return 0;
        }
        let eligible = |slot: &Slot| {
            (slot.state == STATE_READY_DECODE || slot.state == STATE_READY_SPECULATIVE_VERIFY)
                && (slot.remaining_thinking_token_budget != 0
                    || slot.remaining_output_token_budget != 0)
        };
        let mut inputs: Vec<SlotInput> = Vec::new();
        for slot in &self.slots {
            if !eligible(slot) {
                continue;
            }
            inputs.push(SlotInput {
                commit_ema_milli: slot.mtp_commit_ema_milli,
                maximum_draft_depth: if slot.mtp_next_draft_token_budget == 0
                    && slot.mtp_probe_countdown != 0
                {
                    0
                } else {
                    mtp_tree::CANDIDATE_COUNT
                },
                probe: 0,
            });
        }
        let mut budgets = vec![0u32; inputs.len()];
        let total = row_allocator_assign(&inputs, firing_row_cap, 1000, &mut budgets);
        let mut apply_index = 0usize;
        for slot in &mut self.slots {
            if apply_index >= budgets.len() {
                break;
            }
            if !eligible(slot) {
                continue;
            }
            slot.mtp_next_draft_token_budget = budgets[apply_index];
            apply_index += 1;
        }
        total
    }
}
