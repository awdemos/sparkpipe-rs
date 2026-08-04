//! Slot record, lifecycle states, and submit/cache-state descriptors — port
//! of the `SparkRequestApiSlot` / `SparkRequestApiSubmitRequest` /
//! `SparkRequestApiCacheState` structs and the pure slot predicates in
//! `api/request.c`.

use super::{MTP_COMMIT_EMA_INITIAL_MILLI, NO_SLOT};

/// Request handle (`SparkRequestApiHandle`).
pub type RequestApiHandle = u64;

/// Slot state: free (`SPARK_REQUEST_API_STATE_FREE`).
pub const STATE_FREE: u32 = 0;
/// Slot state: queued for prefill (`SPARK_REQUEST_API_STATE_QUEUED_PREFILL`).
pub const STATE_QUEUED_PREFILL: u32 = 1;
/// Slot state: prefill dispatch in flight (`SPARK_REQUEST_API_STATE_RUNNING_PREFILL`).
pub const STATE_RUNNING_PREFILL: u32 = 2;
/// Slot state: ready to decode (`SPARK_REQUEST_API_STATE_READY_DECODE`).
pub const STATE_READY_DECODE: u32 = 3;
/// Slot state: decode dispatch in flight (`SPARK_REQUEST_API_STATE_RUNNING_DECODE`).
pub const STATE_RUNNING_DECODE: u32 = 4;
/// Slot state: completed (`SPARK_REQUEST_API_STATE_COMPLETED`).
pub const STATE_COMPLETED: u32 = 5;
/// Slot state: cancelled (`SPARK_REQUEST_API_STATE_CANCELLED`).
pub const STATE_CANCELLED: u32 = 6;
/// Slot state: waiting on a prefix cohort leader
/// (`SPARK_REQUEST_API_STATE_WAITING_PREFIX_COHORT`).
pub const STATE_WAITING_PREFIX_COHORT: u32 = 7;
/// Slot state: ready for speculative verify
/// (`SPARK_REQUEST_API_STATE_READY_SPECULATIVE_VERIFY`).
pub const STATE_READY_SPECULATIVE_VERIFY: u32 = 8;
/// Slot state: speculative verify dispatch in flight
/// (`SPARK_REQUEST_API_STATE_RUNNING_SPECULATIVE_VERIFY`).
pub const STATE_RUNNING_SPECULATIVE_VERIFY: u32 = 9;

/// Request flag: realtime priority (`SPARK_REQUEST_API_REQUEST_FLAG_REALTIME`).
pub const REQUEST_FLAG_REALTIME: u32 = 0x0000_0001;
/// Request flag: disable speculation for this request
/// (`SPARK_REQUEST_API_REQUEST_FLAG_DISABLE_SPECULATION`).
pub const REQUEST_FLAG_DISABLE_SPECULATION: u32 = 0x0000_0002;
/// All recognized request flags (`SPARK_REQUEST_API_REQUEST_FLAG_KNOWN_FLAGS`).
pub const REQUEST_KNOWN_FLAGS: u32 = REQUEST_FLAG_REALTIME | REQUEST_FLAG_DISABLE_SPECULATION;

/// Default scheduling priority (`SPARK_REQUEST_API_DEFAULT_PRIORITY`).
pub const DEFAULT_PRIORITY: u32 = 1_000_000;
/// Realtime scheduling priority (`SPARK_REQUEST_API_REALTIME_PRIORITY`).
pub const REALTIME_PRIORITY: u32 = 4_000_000_000;

/// Submit request (`SparkRequestApiSubmitRequest`, minus ABI words). The
/// prompt token ids are copied into the slot on submit; the C borrows the
/// caller's array for the request's lifetime.
#[derive(Debug, Clone)]
pub struct SubmitRequest {
    /// `REQUEST_FLAG_*` mask.
    pub flags: u32,
    /// Scheduling priority; 0 normalizes to [`DEFAULT_PRIORITY`] and
    /// [`REQUEST_FLAG_REALTIME`] forces [`REALTIME_PRIORITY`].
    pub priority: u32,
    /// Prompt token count (required count, may exceed
    /// `prompt_token_ids.len()` on the tokenizer-overflow path; the slot is
    /// zero-padded in that case — see module docs).
    pub prompt_token_count: u32,
    /// Thinking-phase token budget.
    pub thinking_token_budget: u32,
    /// Output token budget.
    pub output_token_budget: u32,
    /// Per-request prefill step cap (0 = scheduler default).
    pub max_prefill_tokens_per_step: u32,
    /// Caller request id.
    pub request_id: u64,
    /// Caller sequence id; 0 assigns the next internal sequence id.
    pub sequence_id: u64,
    /// Prompt token ids (copied).
    pub prompt_token_ids: Vec<u32>,
}

/// Per-request slot (`SparkRequestApiSlot`, minus ABI words). Intrusive
/// free-list / handle-hash links are indices into the API's slot array, as in
/// C.
#[derive(Debug, Clone)]
pub struct Slot {
    pub state: u32,
    pub flags: u32,
    pub priority: u32,
    pub prompt_token_count: u32,
    pub computed_prompt_token_count: u32,
    pub dispatched_prompt_token_count: u32,
    pub inflight_prefill_dispatch_count: u32,
    pub scheduled_prefill_step_count: u32,
    pub completed_prefill_step_count: u32,
    pub max_prefill_tokens_per_step: u32,
    pub thinking_token_budget: u32,
    pub output_token_budget: u32,
    pub remaining_thinking_token_budget: u32,
    pub remaining_output_token_budget: u32,
    pub scheduled_decode_token_count: u32,
    pub completed_decode_token_count: u32,
    pub last_committed_prefix_token_count: u32,
    pub prefix_scan_hashed_token_count: u32,
    pub prefix_scan_hash: u64,
    pub last_committed_prefix_hash: u64,
    pub request_id: u64,
    pub sequence_id: u64,
    pub committed_prefix_hash: u64,
    pub handle: RequestApiHandle,
    pub submission_order: u64,
    /// Intrusive handle-hash chain link (`handle_hash_next` in C).
    pub handle_hash_next: u32,
    /// Intrusive free-list link while free (`free_slot_next` in C).
    pub free_slot_next: u32,
    /// Owned copy of the prompt token ids (the C borrows the caller's array).
    pub prompt_token_ids: Vec<u32>,
    pub mtp_draft_token_count: u32,
    pub mtp_next_draft_token_budget: u32,
    /// Commit-EMA in milli fixed point (times 1000).
    pub mtp_commit_ema_milli: u32,
    pub mtp_probe_countdown: u32,
    pub mtp_resolution_base_position: u64,
    pub mtp_resolution_proposed_token_count: u32,
    pub mtp_resolution_accepted_token_count: u32,
    pub mtp_resolution_committed_token_count: u32,
    pub mtp_resolution_path_id: u32,
    pub mtp_draft_token_ids: [u32; super::MTP_MAX_DRAFT_TOKEN_COUNT as usize],
}

impl Slot {
    /// `SparkRequestApiInitializeSlot`: fresh free slot.
    pub fn free() -> Self {
        Slot {
            state: STATE_FREE,
            flags: 0,
            priority: 0,
            prompt_token_count: 0,
            computed_prompt_token_count: 0,
            dispatched_prompt_token_count: 0,
            inflight_prefill_dispatch_count: 0,
            scheduled_prefill_step_count: 0,
            completed_prefill_step_count: 0,
            max_prefill_tokens_per_step: 0,
            thinking_token_budget: 0,
            output_token_budget: 0,
            remaining_thinking_token_budget: 0,
            remaining_output_token_budget: 0,
            scheduled_decode_token_count: 0,
            completed_decode_token_count: 0,
            last_committed_prefix_token_count: 0,
            prefix_scan_hashed_token_count: 0,
            prefix_scan_hash: 0,
            last_committed_prefix_hash: 0,
            request_id: 0,
            sequence_id: 0,
            committed_prefix_hash: 0,
            handle: super::INVALID_HANDLE,
            submission_order: 0,
            handle_hash_next: NO_SLOT,
            free_slot_next: NO_SLOT,
            prompt_token_ids: Vec::new(),
            mtp_draft_token_count: 0,
            mtp_next_draft_token_budget: 0,
            mtp_commit_ema_milli: MTP_COMMIT_EMA_INITIAL_MILLI,
            mtp_probe_countdown: 0,
            mtp_resolution_base_position: 0,
            mtp_resolution_proposed_token_count: 0,
            mtp_resolution_accepted_token_count: 0,
            mtp_resolution_committed_token_count: 0,
            mtp_resolution_path_id: 0,
            mtp_draft_token_ids: [0; super::MTP_MAX_DRAFT_TOKEN_COUNT as usize],
        }
    }

    /// `SparkRequestApiSlotIsReadyForDispatch`.
    pub fn is_ready_for_dispatch(&self) -> bool {
        if self.state == STATE_QUEUED_PREFILL {
            return true;
        }
        (self.state == STATE_READY_DECODE || self.state == STATE_READY_SPECULATIVE_VERIFY)
            && (self.remaining_thinking_token_budget != 0
                || self.remaining_output_token_budget != 0)
    }

    /// `SparkRequestApiSlotIsActive`.
    pub fn is_active(&self) -> bool {
        self.state != STATE_FREE && self.state != STATE_COMPLETED && self.state != STATE_CANCELLED
    }

    /// `SparkRequestApiSlotIsSchedulablePrefill`.
    pub fn is_schedulable_prefill(&self) -> bool {
        if self.state == STATE_QUEUED_PREFILL {
            return true;
        }
        self.state == STATE_RUNNING_PREFILL
            && self.dispatched_prompt_token_count < self.prompt_token_count
            && self.inflight_prefill_dispatch_count < super::PREFILL_INFLIGHT_WAVE_LIMIT
    }

    /// `SparkRequestApiSlotIsSchedulableDecode`.
    pub fn is_schedulable_decode(&self) -> bool {
        (self.state == STATE_READY_DECODE
            || (self.state == STATE_READY_SPECULATIVE_VERIFY && self.mtp_draft_token_count != 0))
            && (self.remaining_thinking_token_budget != 0
                || self.remaining_output_token_budget != 0)
    }

    /// `SparkRequestApiSlotIsSchedulableSpeculativeVerify`.
    pub fn is_schedulable_speculative_verify(&self) -> bool {
        self.state == STATE_READY_SPECULATIVE_VERIFY
            && (self.remaining_thinking_token_budget != 0
                || self.remaining_output_token_budget != 0)
    }

    /// `SparkRequestApiSlotHasRealtimePriority`.
    pub fn has_realtime_priority(&self) -> bool {
        self.flags & REQUEST_FLAG_REALTIME != 0 || self.priority >= REALTIME_PRIORITY
    }

    /// `SparkRequestApiSlotHasHigherSchedulingPriority`.
    pub fn has_higher_scheduling_priority_than(&self, current: Option<&Slot>) -> bool {
        let Some(current) = current else {
            return true;
        };
        if self.priority != current.priority {
            return self.priority > current.priority;
        }
        self.submission_order < current.submission_order
    }

    /// `SparkRequestApiSlotsHaveSameSchedulingPriority`.
    pub fn has_same_scheduling_priority(&self, other: &Slot) -> bool {
        self.priority == other.priority
    }

    /// `SparkRequestApiSlotRemainingDecodeBudget`.
    pub fn remaining_decode_budget(&self) -> u32 {
        self.remaining_thinking_token_budget + self.remaining_output_token_budget
    }
}

/// Per-request cache state snapshot (`SparkRequestApiCacheState`, minus ABI
/// words).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheState {
    pub state: u32,
    pub computed_prompt_token_count: u32,
    pub last_committed_prefix_token_count: u32,
    pub physical_block_count: u32,
    pub request_id: u64,
    pub sequence_id: u64,
    pub last_committed_prefix_hash: u64,
}
