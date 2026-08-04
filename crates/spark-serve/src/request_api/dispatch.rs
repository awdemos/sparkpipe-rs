//! Dispatch records, views, and the dispatch lifecycle entry points — port
//! of the dispatch half of `api/request.c`: scheduling (prefill, prefill
//! batch, decode batch, speculative verify batch, `ScheduleNext`),
//! completion / cancel / retry, the describe/copy token-window entry points,
//! and the KV block table views. The `spark_text::prompt_pipeline::RequestApi`
//! trait implementation lives here too.

use spark_sched::scheduler::{
    BatchDecision, PrefillBatchDecision, SchedulerDecision, SchedulerRequest, NO_PROMPT_LIMIT,
    REQUEST_FLAG_DECODE, REQUEST_FLAG_PREFILL,
};
use spark_sched::stage_plan;
use spark_sched::work_control::mtp_tree;
use spark_text::prompt_pipeline::{self, PromptPipelineError};

use super::prefetch;
use super::slot::{
    Slot, STATE_CANCELLED, STATE_COMPLETED, STATE_QUEUED_PREFILL, STATE_READY_DECODE,
    STATE_READY_SPECULATIVE_VERIFY, STATE_RUNNING_DECODE, STATE_RUNNING_PREFILL,
    STATE_RUNNING_SPECULATIVE_VERIFY, STATE_WAITING_PREFIX_COHORT,
};
use super::speculation::{SPECULATIVE_SOURCE_DRAFTER, SPECULATIVE_SOURCE_MTP};
use super::{
    RequestApi, RequestApiError, CONFIGURATION_FLAG_PREFER_DSPARK_SPECULATION,
    MAX_DISPATCH_REQUEST_COUNT, MAX_SPECULATIVE_TOKENS, MTP_MAX_DRAFT_TOKEN_COUNT, NO_SLOT,
    PREFIX_FAMILY_GROUP_CAPACITY,
};

/// Dispatch kind: none (`SPARK_REQUEST_API_DISPATCH_KIND_NONE`).
pub const DISPATCH_KIND_NONE: u32 = 0;
/// Dispatch kind: prefill (`SPARK_REQUEST_API_DISPATCH_KIND_PREFILL`).
pub const DISPATCH_KIND_PREFILL: u32 = 1;
/// Dispatch kind: decode batch (`SPARK_REQUEST_API_DISPATCH_KIND_DECODE_BATCH`).
pub const DISPATCH_KIND_DECODE_BATCH: u32 = 2;
/// Dispatch kind: prefill batch (`SPARK_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH`).
pub const DISPATCH_KIND_PREFILL_BATCH: u32 = 3;
/// Dispatch kind: speculative verify batch
/// (`SPARK_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH`).
pub const DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH: u32 = 4;

/// Dispatch kind (mirrors the C `kind` field's value set; the numeric
/// constants above stay available for the trait bridge).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DispatchKind {
    /// `DISPATCH_KIND_NONE`.
    #[default]
    None = DISPATCH_KIND_NONE as isize,
    /// `DISPATCH_KIND_PREFILL`.
    Prefill = DISPATCH_KIND_PREFILL as isize,
    /// `DISPATCH_KIND_DECODE_BATCH`.
    DecodeBatch = DISPATCH_KIND_DECODE_BATCH as isize,
    /// `DISPATCH_KIND_PREFILL_BATCH`.
    PrefillBatch = DISPATCH_KIND_PREFILL_BATCH as isize,
    /// `DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH`.
    SpeculativeVerifyBatch = DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH as isize,
}

/// Dispatch flag: JIT-prefetched KV (`SPARK_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCHED_KV`).
pub const DISPATCH_FLAG_JIT_PREFETCHED_KV: u32 = 0x0000_0001;
/// Dispatch flag: priority preempted the queue
/// (`SPARK_REQUEST_API_DISPATCH_FLAG_PRIORITY_PREEMPTED_QUEUE`).
pub const DISPATCH_FLAG_PRIORITY_PREEMPTED_QUEUE: u32 = 0x0000_0002;
/// Dispatch flag: prefix cohort (`SPARK_REQUEST_API_DISPATCH_FLAG_PREFIX_COHORT`).
pub const DISPATCH_FLAG_PREFIX_COHORT: u32 = 0x0000_0004;
/// Dispatch flag: JIT prefetch pending (`SPARK_REQUEST_API_DISPATCH_FLAG_JIT_PREFETCH_PENDING`).
pub const DISPATCH_FLAG_JIT_PREFETCH_PENDING: u32 = 0x0000_0008;
/// Dispatch flag: prefill batch (`SPARK_REQUEST_API_DISPATCH_FLAG_PREFILL_BATCH`).
pub const DISPATCH_FLAG_PREFILL_BATCH: u32 = 0x0000_0010;
/// Dispatch flag: prefix family selected
/// (`SPARK_REQUEST_API_DISPATCH_FLAG_PREFIX_FAMILY_SELECTED`).
pub const DISPATCH_FLAG_PREFIX_FAMILY_SELECTED: u32 = 0x0000_0020;
/// Dispatch flag: dspark tap capture (`SPARK_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE`).
pub const DISPATCH_FLAG_DSPARK_TAP_CAPTURE: u32 = 0x0000_0040;
/// Dispatch flag: dspark speculative verify
/// (`SPARK_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY`).
pub const DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY: u32 = 0x0000_0080;
/// Dispatch flag: dspark confidence truncated
/// (`SPARK_REQUEST_API_DISPATCH_FLAG_DSPARK_CONFIDENCE_TRUNCATED`).
pub const DISPATCH_FLAG_DSPARK_CONFIDENCE_TRUNCATED: u32 = 0x0000_0100;
/// Dispatch flag: MTP commit (`SPARK_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT`).
pub const DISPATCH_FLAG_MTP_COMMIT: u32 = 0x0000_0200;
/// Dispatch flag: MTP speculative verify
/// (`SPARK_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY`).
pub const DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY: u32 = 0x0000_0400;
/// Dispatch flag: MTP tree verify (`SPARK_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY`).
pub const DISPATCH_FLAG_MTP_TREE_VERIFY: u32 = 0x0000_0800;

/// Dispatch record (`SparkRequestApiDispatch`, minus ABI words). The
/// scheduler decisions are owned `Option`s (the C embeds them by value).
#[derive(Debug, Clone)]
pub struct Dispatch {
    pub accepted: bool,
    pub kind: DispatchKind,
    pub flags: u32,
    pub request_count: u32,
    pub highest_priority: u32,
    pub shared_prefix_token_count: u32,
    pub shared_prefix_block_count: u32,
    pub prefix_cache_parent_hash: u64,
    pub prefix_cache_result_hash: u64,
    pub request_handles: [u64; MAX_DISPATCH_REQUEST_COUNT as usize],
    pub request_slot_indices: [u32; MAX_DISPATCH_REQUEST_COUNT as usize],
    pub request_ids: [u64; MAX_DISPATCH_REQUEST_COUNT as usize],
    pub sequence_ids: [u64; MAX_DISPATCH_REQUEST_COUNT as usize],
    pub committed_prefix_hash: u64,
    pub prefill_decision: Option<SchedulerDecision>,
    pub prefill_batch_decision: Option<PrefillBatchDecision>,
    pub decode_batch_decision: Option<BatchDecision>,
    pub kv_prefetch_plan: prefetch::PrefetchPlan,
    pub speculative_token_count: u32,
    pub speculative_verifier_token_count: u32,
    pub speculative_max_committed_token_count: u32,
    pub speculative_committed_token_counts: [u32; MAX_DISPATCH_REQUEST_COUNT as usize],
    pub speculative_accepted_token_counts: [u32; MAX_DISPATCH_REQUEST_COUNT as usize],
    pub speculative_fallback_token_ids: [u32; MAX_DISPATCH_REQUEST_COUNT as usize],
    pub speculative_resolution_path_ids: [u32; MAX_DISPATCH_REQUEST_COUNT as usize],
    pub speculative_draft_token_ids:
        [[u32; MAX_SPECULATIVE_TOKENS as usize]; MAX_DISPATCH_REQUEST_COUNT as usize],
    pub speculative_confidence_milli:
        [[u32; MAX_SPECULATIVE_TOKENS as usize]; MAX_DISPATCH_REQUEST_COUNT as usize],
    pub mtp_draft_token_budget: u32,
    pub decode_committed_token_counts: [u32; MAX_DISPATCH_REQUEST_COUNT as usize],
}

impl Default for Dispatch {
    fn default() -> Self {
        Dispatch {
            accepted: false,
            kind: DispatchKind::None,
            flags: 0,
            request_count: 0,
            highest_priority: 0,
            shared_prefix_token_count: 0,
            shared_prefix_block_count: 0,
            prefix_cache_parent_hash: 0,
            prefix_cache_result_hash: 0,
            request_handles: [0; MAX_DISPATCH_REQUEST_COUNT as usize],
            request_slot_indices: [NO_SLOT; MAX_DISPATCH_REQUEST_COUNT as usize],
            request_ids: [0; MAX_DISPATCH_REQUEST_COUNT as usize],
            sequence_ids: [0; MAX_DISPATCH_REQUEST_COUNT as usize],
            committed_prefix_hash: 0,
            prefill_decision: None,
            prefill_batch_decision: None,
            decode_batch_decision: None,
            kv_prefetch_plan: prefetch::PrefetchPlan::default(),
            speculative_token_count: 0,
            speculative_verifier_token_count: 0,
            speculative_max_committed_token_count: 0,
            speculative_committed_token_counts: [0; MAX_DISPATCH_REQUEST_COUNT as usize],
            speculative_accepted_token_counts: [0; MAX_DISPATCH_REQUEST_COUNT as usize],
            speculative_fallback_token_ids: [0; MAX_DISPATCH_REQUEST_COUNT as usize],
            speculative_resolution_path_ids: [0; MAX_DISPATCH_REQUEST_COUNT as usize],
            speculative_draft_token_ids: [[0; MAX_SPECULATIVE_TOKENS as usize];
                MAX_DISPATCH_REQUEST_COUNT as usize],
            speculative_confidence_milli: [[0; MAX_SPECULATIVE_TOKENS as usize];
                MAX_DISPATCH_REQUEST_COUNT as usize],
            mtp_draft_token_budget: 0,
            decode_committed_token_counts: [0; MAX_DISPATCH_REQUEST_COUNT as usize],
        }
    }
}

/// `SparkRequestApiInitializeDispatch`.
pub(crate) fn initialize_dispatch(dispatch: &mut Dispatch) {
    *dispatch = Dispatch::default();
}

/// Prefill dispatch lane view (`SparkRequestApiPrefillDispatchLaneView`,
/// minus ABI words; the prompt token ids are owned — the C borrows the
/// scheduler decision's array).
#[derive(Debug, Clone, Default)]
pub struct PrefillDispatchLaneView {
    pub request_index: u32,
    pub prompt_token_offset: u32,
    pub prompt_token_count: u32,
    pub request_slot_index: u32,
    pub request_id: u64,
    pub sequence_id: u64,
    pub request_handle: u64,
    pub prompt_token_ids: Vec<u32>,
}

/// Prefill dispatch view (`SparkRequestApiPrefillDispatchView`).
#[derive(Debug, Clone, Default)]
pub struct PrefillDispatchView {
    pub kind: DispatchKind,
    pub active_sequence_count: u32,
    pub lane_count: u32,
    pub prompt_token_offset: u32,
    pub prompt_token_count: u32,
    pub prompt_token_stride: u32,
    pub lanes: Vec<PrefillDispatchLaneView>,
}

/// Decode dispatch lane view (`SparkRequestApiDecodeDispatchLaneView`).
#[derive(Debug, Clone, Default)]
pub struct DecodeDispatchLaneView {
    pub request_index: u32,
    pub sequence_position: u32,
    pub context_token_count: u32,
    pub request_slot_index: u32,
    pub request_id: u64,
    pub sequence_id: u64,
    pub request_handle: u64,
    pub mtp_resolution_base_position: u64,
    pub mtp_resolution_proposed_token_count: u32,
    pub mtp_resolution_accepted_token_count: u32,
    pub mtp_resolution_committed_token_count: u32,
    pub mtp_resolution_path_id: u32,
}

/// Decode dispatch view (`SparkRequestApiDecodeDispatchView`).
#[derive(Debug, Clone, Default)]
pub struct DecodeDispatchView {
    pub kind: DispatchKind,
    pub active_sequence_count: u32,
    pub lane_count: u32,
    pub speculative_token_count: u32,
    pub lanes: Vec<DecodeDispatchLaneView>,
}

/// `SparkRequestApiPrefixFamilyChoice` (the leader the family scan picked).
#[derive(Debug, Clone, Copy, Default)]
struct PrefixFamilyChoice {
    leader_slot: u32,
    shared_prefix_token_count: u32,
    realtime_priority: bool,
    saved_prompt_token_count: u64,
    valid: bool,
}

/// `SparkRequestApiPrefixFamilyGroup`.
#[derive(Debug, Clone, Copy)]
struct PrefixFamilyGroup {
    valid: bool,
    prefix_hash: u64,
    shared_prefix_token_count: u32,
    request_count: u32,
    capped_request_count: u32,
    realtime_priority: bool,
    highest_priority: u32,
    earliest_submission_order: u64,
    leader_slot: u32,
}

impl Default for PrefixFamilyGroup {
    fn default() -> Self {
        PrefixFamilyGroup {
            valid: false,
            prefix_hash: 0,
            shared_prefix_token_count: 0,
            request_count: 0,
            capped_request_count: 0,
            realtime_priority: false,
            highest_priority: 0,
            earliest_submission_order: u64::MAX,
            leader_slot: NO_SLOT,
        }
    }
}

/// `SparkRequestApiPrefillBatchShape`.
#[derive(Debug, Clone, Copy, Default)]
struct PrefillBatchShape {
    slot: u32,
    scheduled_prompt_token_count: u32,
    resident_cached_blocks: bool,
    realtime_priority: bool,
    compatible_request_count: u32,
    graph_padding_count: u32,
}

// ---------------------------------------------------------------------------
// Low-level helpers (the C statics, in dependency order).
// ---------------------------------------------------------------------------

impl RequestApi {
    /// `SparkRequestApiPrefixCacheBlockTokenCount` (reads through the
    /// scheduler as the C does).
    pub(crate) fn prefix_cache_block_token_count(&self) -> u32 {
        self.scheduler.borrow().prefix_cache_block_tokens()
    }

    /// `SparkRequestApiSchedulerMaxPrefillTokensPerStep`.
    fn scheduler_max_prefill_tokens_per_step(&self, slot_index: u32) -> u32 {
        let scheduler = self.scheduler.borrow();
        let scheduler_max = scheduler.max_prefill_tokens_per_step();
        let slot_max = self.slots[slot_index as usize].max_prefill_tokens_per_step;
        let mut max_prefill_tokens_per_step = slot_max;
        if max_prefill_tokens_per_step == 0 || max_prefill_tokens_per_step > scheduler_max {
            max_prefill_tokens_per_step = scheduler_max;
        }
        let block_tokens = scheduler.prefix_cache_block_tokens();
        if max_prefill_tokens_per_step < block_tokens {
            max_prefill_tokens_per_step = block_tokens;
        }
        max_prefill_tokens_per_step
    }

    /// `SparkRequestApiRoundDownSchedulerBlock`.
    fn round_down_scheduler_block(&self, token_count: u32) -> u32 {
        let block_token_count = self.prefix_cache_block_token_count();
        if block_token_count == 0 {
            return token_count;
        }
        token_count - (token_count % block_token_count)
    }

    /// `SparkRequestApiProbeReusablePrefixTokenCount` (pure — the C does not
    /// mutate the slot here).
    pub(crate) fn probe_reusable_prefix_token_count(&mut self, slot_index: u32) -> u32 {
        let (sequence_id, prompt, computed) = {
            let slot = &self.slots[slot_index as usize];
            (slot.sequence_id, slot.prompt_token_ids.clone(), slot.computed_prompt_token_count)
        };
        if prompt.is_empty() || self.scheduler.get_mut().prefix_cache().is_none() {
            return computed;
        }
        let scheduler = self.scheduler.get_mut();
        let Some(prefix_cache) = scheduler.prefix_cache_mut() else {
            return computed;
        };
        match prefix_cache.probe_prompt(sequence_id, &prompt) {
            Ok(lookup) => computed.max(lookup.matched_token_count),
            Err(_) => computed,
        }
    }

    /// `SparkRequestApiNextPrefillStepTokenCount`; returns
    /// `(scheduled_token_count, computed_prompt_token_count)`.
    pub(crate) fn next_prefill_step_token_count(&mut self, slot_index: u32) -> (u32, u32) {
        let prompt_token_count = self.slots[slot_index as usize].prompt_token_count;
        let cached_prefix_token_count = self.probe_reusable_prefix_token_count(slot_index);
        let computed_prompt_token_count = self.slots[slot_index as usize]
            .computed_prompt_token_count
            .max(cached_prefix_token_count);
        if computed_prompt_token_count >= prompt_token_count {
            return (0, computed_prompt_token_count);
        }
        let remaining_prompt_token_count = prompt_token_count - computed_prompt_token_count;
        if !self.configuration_has_chunked_prefill() {
            return (remaining_prompt_token_count, computed_prompt_token_count);
        }
        let max_prefill_tokens_per_step = self.scheduler_max_prefill_tokens_per_step(slot_index);
        if remaining_prompt_token_count <= max_prefill_tokens_per_step {
            return (remaining_prompt_token_count, computed_prompt_token_count);
        }
        let mut scheduled_prompt_token_count =
            self.round_down_scheduler_block(max_prefill_tokens_per_step);
        if scheduled_prompt_token_count == 0 {
            scheduled_prompt_token_count =
                remaining_prompt_token_count.min(self.prefix_cache_block_token_count());
        }
        (scheduled_prompt_token_count, computed_prompt_token_count)
    }

    /// `SparkRequestApiCountCommonPrefixTokens`.
    fn count_common_prefix_tokens(&self, left_index: u32, right_index: u32) -> u32 {
        let left = &self.slots[left_index as usize];
        let right = &self.slots[right_index as usize];
        let shared_token_count = left.prompt_token_count.min(right.prompt_token_count);
        for token_index in 0..shared_token_count as usize {
            if left.prompt_token_ids.get(token_index) != right.prompt_token_ids.get(token_index) {
                return token_index as u32;
            }
        }
        shared_token_count
    }

    /// `SparkRequestApiSharedCachePrefixTokenCount`.
    pub(crate) fn shared_cache_prefix_token_count(&self, left_index: u32, right_index: u32) -> u32 {
        if !self.cross_sequence_prefix_reuse_is_enabled()
            || self.scheduler.borrow().prefix_cache().is_none()
        {
            return 0;
        }
        let block_token_count = self.prefix_cache_block_token_count();
        let mut common_prefix_token_count =
            self.count_common_prefix_tokens(left_index, right_index);
        if block_token_count != 0 {
            common_prefix_token_count -= common_prefix_token_count % block_token_count;
        }
        let left = &self.slots[left_index as usize];
        let right = &self.slots[right_index as usize];
        if common_prefix_token_count
            <= left.computed_prompt_token_count.max(right.computed_prompt_token_count)
        {
            return 0;
        }
        common_prefix_token_count
    }

    /// `SparkRequestApiFindBestSharedPrefixTokenCount`.
    pub(crate) fn find_best_shared_prefix_token_count(&self, leader_slot_index: u32) -> u32 {
        if !self.prefix_cohorting_is_enabled() {
            return 0;
        }
        let mut best_shared_prefix_token_count = 0u32;
        for candidate_index in 0..self.request_capacity {
            if candidate_index == leader_slot_index {
                continue;
            }
            let candidate = &self.slots[candidate_index as usize];
            let leader = &self.slots[leader_slot_index as usize];
            if candidate.state != STATE_QUEUED_PREFILL
                || !candidate.has_same_scheduling_priority(leader)
            {
                continue;
            }
            let candidate_shared_prefix_token_count =
                self.shared_cache_prefix_token_count(leader_slot_index, candidate_index);
            if candidate_shared_prefix_token_count > best_shared_prefix_token_count {
                best_shared_prefix_token_count = candidate_shared_prefix_token_count;
            }
        }
        best_shared_prefix_token_count
    }

    /// `SparkRequestApiPrefillCachedBlocksAreResident`.
    pub(crate) fn prefill_cached_blocks_are_resident(
        &mut self,
        slot_index: u32,
        prompt_token_count: u32,
    ) -> bool {
        if !self.jit_prefetch_is_enabled()
            || self.scheduler.get_mut().prefix_cache().is_none()
            || prompt_token_count == 0
        {
            return true;
        }
        let prompt = self.slots[slot_index as usize].prompt_token_ids[..prompt_token_count
            .min(self.slots[slot_index as usize].prompt_token_ids.len() as u32)
            as usize]
            .to_vec();
        let scheduler = self.scheduler.get_mut();
        let Some(prefix_cache) = scheduler.prefix_cache_mut() else {
            return true;
        };
        match prefix_cache.probe_reusable_prefix_residency(&prompt) {
            Ok(probe) => probe.nonresident_block_count == 0,
            Err(_) => false,
        }
    }

    /// `SparkRequestApiRequiredDecodeKvTokenCount`.
    pub(crate) fn required_decode_kv_token_count(
        &self,
        slot_index: u32,
        speculative_token_count: u32,
    ) -> Result<u32, RequestApiError> {
        let slot = &self.slots[slot_index as usize];
        if slot.computed_prompt_token_count == 0 {
            return Err(RequestApiError::InvalidArgument);
        }
        let required_token_count = u64::from(slot.computed_prompt_token_count)
            + u64::from(slot.completed_decode_token_count)
            + 1
            + u64::from(speculative_token_count);
        // The C also bounds this by SPARK_SCHEDULER_MAX_CONTEXT_TOKENS; the
        // Rust scheduler keeps its configured context bound private, so the
        // capacity check is delegated to the prefix cache's
        // ensure_sequence_token_capacity (see module docs).
        if required_token_count > u64::from(u32::MAX) {
            return Err(RequestApiError::CapacityExceeded);
        }
        Ok(required_token_count as u32)
    }

    /// `SparkRequestApiEnsureDecodeSlotKvCapacity`. Takes `&self` through
    /// the scheduler `RefCell` so the trait's `&self` KV-table entry point
    /// can drive it.
    pub(crate) fn ensure_decode_slot_kv_capacity(
        &self,
        slot_index: u32,
        speculative_token_count: u32,
    ) -> Result<u32, RequestApiError> {
        if self.scheduler.borrow().prefix_cache().is_none() {
            return Err(RequestApiError::InvalidArgument);
        }
        let required_token_count =
            self.required_decode_kv_token_count(slot_index, speculative_token_count)?;
        let sequence_id = self.slots[slot_index as usize].sequence_id;
        let mut scheduler = self.scheduler.borrow_mut();
        let prefix_cache = scheduler.prefix_cache_mut().ok_or(RequestApiError::InvalidArgument)?;
        prefix_cache.ensure_sequence_token_capacity(sequence_id, required_token_count)?;
        Ok(required_token_count)
    }

    /// `SparkRequestApiEnsurePendingDecodeSlotKvCapacity`.
    pub(crate) fn ensure_pending_decode_slot_kv_capacity(
        &mut self,
        slot_index: u32,
    ) -> Result<u32, RequestApiError> {
        let speculative_token_count = self.pending_speculative_token_count(slot_index)?;
        self.ensure_decode_slot_kv_capacity(slot_index, speculative_token_count)
    }

    /// `SparkRequestApiDecodeBlocksAreResident`.
    pub(crate) fn decode_blocks_are_resident(&mut self, slot_index: u32) -> bool {
        if self.scheduler.get_mut().prefix_cache().is_none()
            || self.slots[slot_index as usize].computed_prompt_token_count == 0
        {
            return true;
        }
        let Ok(required_token_count) = self.ensure_pending_decode_slot_kv_capacity(slot_index)
        else {
            return false;
        };
        if !self.jit_prefetch_is_enabled() {
            return true;
        }
        let sequence_id = self.slots[slot_index as usize].sequence_id;
        let scheduler = self.scheduler.get_mut();
        let Some(prefix_cache) = scheduler.prefix_cache_mut() else {
            return false;
        };
        match prefix_cache.probe_sequence_residency(sequence_id, required_token_count) {
            Ok(probe) => probe.nonresident_block_count == 0,
            Err(_) => false,
        }
    }

    /// `SparkRequestApiOlderLowerPrioritySchedulableSlotExists`.
    pub(crate) fn older_lower_priority_schedulable_slot_exists(
        &self,
        chosen_slot_index: u32,
    ) -> bool {
        let chosen = &self.slots[chosen_slot_index as usize];
        self.slots.iter().any(|slot| {
            (slot.is_schedulable_prefill() || slot.is_schedulable_decode())
                && slot.submission_order < chosen.submission_order
                && chosen.priority > slot.priority
        })
    }

    /// `SparkRequestApiSlotResidentKvBlockCount`.
    fn slot_resident_kv_block_count(&self, slot_index: u32) -> u32 {
        let slot = &self.slots[slot_index as usize];
        if slot.state == super::STATE_FREE
            || slot.state == STATE_COMPLETED
            || slot.state == STATE_CANCELLED
            || (slot.computed_prompt_token_count == 0
                && (slot.state == STATE_QUEUED_PREFILL
                    || slot.state == STATE_WAITING_PREFIX_COHORT))
        {
            return 0;
        }
        let block_token_count = self.prefix_cache_block_token_count();
        if block_token_count == 0 {
            return 0;
        }
        let mut token_count =
            u64::from(slot.prompt_token_count) + u64::from(slot.scheduled_decode_token_count);
        if slot.state == STATE_RUNNING_SPECULATIVE_VERIFY {
            token_count += 1;
        }
        if token_count > u64::from(u32::MAX) {
            return u32::MAX;
        }
        token_count.div_ceil(u64::from(block_token_count)) as u32
    }

    /// `SparkRequestApiResidentKvBlockCount`.
    pub(crate) fn resident_kv_block_count(&self) -> u64 {
        let mut block_count = 0u64;
        for slot_index in 0..self.request_capacity {
            block_count += u64::from(self.slot_resident_kv_block_count(slot_index));
        }
        block_count
    }

    /// `SparkRequestApiPrefillBlockCountForScheduledTokens`.
    pub(crate) fn prefill_block_count_for_scheduled_tokens(
        &self,
        scheduled_prompt_token_count: u32,
    ) -> u32 {
        if scheduled_prompt_token_count == 0 {
            return 0;
        }
        let block_token_count = self.prefix_cache_block_token_count();
        if block_token_count == 0 {
            return 1;
        }
        scheduled_prompt_token_count.div_ceil(block_token_count)
    }

    /// `SparkRequestApiReservePrefillResidentKvBlocks`.
    pub(crate) fn reserve_prefill_resident_kv_blocks(
        &self,
        slot_index: u32,
        reserved_block_count: &mut u64,
    ) -> bool {
        if self.max_resident_kv_block_count == 0 || self.jit_prefetch_is_enabled() {
            return true;
        }
        let current_block_count = u64::from(self.slot_resident_kv_block_count(slot_index));
        let required_block_count = u64::from(self.prefill_block_count_for_scheduled_tokens(
            self.slots[slot_index as usize].prompt_token_count,
        ));
        let additional_block_count = required_block_count.saturating_sub(current_block_count);
        if *reserved_block_count > u64::from(self.max_resident_kv_block_count)
            || additional_block_count
                > u64::from(self.max_resident_kv_block_count) - *reserved_block_count
        {
            return false;
        }
        *reserved_block_count += additional_block_count;
        true
    }

    /// `SparkRequestApiPrefillHasResidentKvHeadroom`.
    fn prefill_has_resident_kv_headroom(&self, prefill_slot_index: u32) -> bool {
        let mut reserved_block_count = self.resident_kv_block_count();
        self.reserve_prefill_resident_kv_blocks(prefill_slot_index, &mut reserved_block_count)
    }

    /// `SparkRequestApiApplyActiveKvBlockBudget`: compact `selected_slots`
    /// in place so the cumulative block requirement fits the residency
    /// budget.
    pub(crate) fn apply_active_kv_block_budget(
        &self,
        selected_slots: &mut Vec<u32>,
        additional_token_count: u32,
    ) -> Result<(), RequestApiError> {
        if selected_slots.is_empty() {
            return Err(RequestApiError::InvalidArgument);
        }
        if self.max_resident_kv_block_count == 0 {
            return Ok(());
        }
        if self.scheduler.borrow().prefix_cache().is_none() {
            return Err(RequestApiError::InvalidArgument);
        }
        let block_token_count = self.prefix_cache_block_token_count();
        if block_token_count == 0 {
            return Err(RequestApiError::InvalidArgument);
        }
        let mut selected_block_count = 0u64;
        let mut output: Vec<u32> = Vec::with_capacity(selected_slots.len());
        for (input_index, &slot_index) in selected_slots.iter().enumerate() {
            let required_token_count =
                self.required_decode_kv_token_count(slot_index, additional_token_count)?;
            let required_block_count =
                u64::from(required_token_count).div_ceil(u64::from(block_token_count));
            if required_block_count > u64::from(self.max_resident_kv_block_count) {
                if input_index == 0 {
                    return Err(RequestApiError::CapacityExceeded);
                }
                continue;
            }
            if selected_block_count
                > u64::from(self.max_resident_kv_block_count) - required_block_count
            {
                continue;
            }
            output.push(slot_index);
            selected_block_count += required_block_count;
        }
        if output.is_empty() {
            return Err(RequestApiError::CapacityExceeded);
        }
        *selected_slots = output;
        Ok(())
    }

    /// `SparkRequestApiInsertBatchMemberByPriority` (sorted insert with tail
    /// drop at capacity; position 0 is reserved for the leader).
    fn insert_batch_member_by_priority(
        &self,
        selected_slots: &mut Vec<u32>,
        selected_capacity: u32,
        slot_index: u32,
    ) {
        if selected_capacity <= 1 {
            return;
        }
        let slot = &self.slots[slot_index as usize];
        if selected_slots.len() as u32 >= selected_capacity {
            let tail = &self.slots[selected_slots[(selected_capacity - 1) as usize] as usize];
            if !slot.has_higher_scheduling_priority_than(Some(tail)) {
                return;
            }
            selected_slots.truncate((selected_capacity - 1) as usize);
        }
        let mut insert_index = selected_slots.len();
        while insert_index > 1
            && slot.has_higher_scheduling_priority_than(Some(
                &self.slots[selected_slots[insert_index - 1] as usize],
            ))
        {
            insert_index -= 1;
        }
        selected_slots.insert(insert_index, slot_index);
    }

    /// `SparkRequestApiCollectDecodeBatchMembers`.
    pub(crate) fn collect_decode_batch_members(
        &mut self,
        leader_slot_index: u32,
        require_resident_kv: bool,
        selected_capacity: u32,
    ) -> Vec<u32> {
        let mut selected_slots = vec![leader_slot_index];
        for slot_index in 0..self.request_capacity {
            if slot_index == leader_slot_index {
                continue;
            }
            let slot = &self.slots[slot_index as usize];
            if !slot.is_schedulable_decode()
                || self.slot_can_speculate(slot_index) != self.slot_can_speculate(leader_slot_index)
            {
                continue;
            }
            if (require_resident_kv
                || slot.priority < self.slots[leader_slot_index as usize].priority)
                && !self.decode_blocks_are_resident(slot_index)
            {
                continue;
            }
            self.insert_batch_member_by_priority(
                &mut selected_slots,
                selected_capacity,
                slot_index,
            );
        }
        selected_slots
    }

    /// `SparkRequestApiCollectSpeculativeVerifyBatchMembers`.
    fn collect_speculative_verify_batch_members(
        &mut self,
        leader_slot_index: u32,
        leader_token_count: u32,
        leader_source: u32,
        require_resident_kv: bool,
        selected_capacity: u32,
    ) -> Vec<u32> {
        let mut selected_slots = vec![leader_slot_index];
        for slot_index in 0..self.request_capacity {
            if slot_index == leader_slot_index {
                continue;
            }
            let slot = &self.slots[slot_index as usize];
            if !slot.is_schedulable_speculative_verify()
                || self.slot_can_speculate(slot_index) != self.slot_can_speculate(leader_slot_index)
                || ((require_resident_kv
                    || slot.priority < self.slots[leader_slot_index as usize].priority)
                    && !self.decode_blocks_are_resident(slot_index))
            {
                continue;
            }
            match self.get_slot_speculative_draft(slot_index, leader_source) {
                Ok((draft, source))
                    if source == leader_source && draft.token_count == leader_token_count => {}
                _ => continue,
            }
            self.insert_batch_member_by_priority(
                &mut selected_slots,
                selected_capacity,
                slot_index,
            );
        }
        selected_slots
    }

    /// `SparkRequestApiFindBestSchedulableSlot` (the decode / speculative
    /// verify search; the predicate is the parameter as in C).
    pub(crate) fn find_best_schedulable_slot(
        &mut self,
        is_schedulable: fn(&Slot) -> bool,
        excluded_handles: &[u64],
        require_resident_kv: bool,
    ) -> Option<u32> {
        let mut best_slot: Option<u32> = None;
        for slot_index in 0..self.request_capacity {
            let (schedulable, handle) = {
                let slot = &self.slots[slot_index as usize];
                (is_schedulable(slot), slot.handle)
            };
            if !schedulable
                || (require_resident_kv && !self.decode_blocks_are_resident(slot_index))
                || excluded_handles.contains(&handle)
            {
                continue;
            }
            let is_better = match best_slot {
                None => true,
                Some(best_index) => self.slots[slot_index as usize]
                    .has_higher_scheduling_priority_than(Some(&self.slots[best_index as usize])),
            };
            if is_better {
                best_slot = Some(slot_index);
            }
        }
        best_slot
    }
}

// ---------------------------------------------------------------------------
// Prefix family scan + prefill batch shapes (the C statics).
// ---------------------------------------------------------------------------

/// `SparkRequestApiPrefixFamilySavedTokenCount`.
fn prefix_family_saved_token_count(shared_prefix_token_count: u32, request_count: u32) -> u64 {
    if shared_prefix_token_count == 0 || request_count < 2 {
        return 0;
    }
    u64::from(shared_prefix_token_count) * u64::from(request_count - 1)
}

/// `SparkRequestApiPrefixFamilyGroupIsBetter`.
fn prefix_family_group_is_better(
    candidate: &PrefixFamilyGroup,
    current: Option<&PrefixFamilyGroup>,
) -> bool {
    if !candidate.valid || candidate.capped_request_count < 2 {
        return false;
    }
    let Some(current) = current else {
        return true;
    };
    if !current.valid || current.capped_request_count < 2 {
        return true;
    }
    if candidate.highest_priority != current.highest_priority {
        return candidate.highest_priority > current.highest_priority;
    }
    if candidate.realtime_priority != current.realtime_priority {
        return candidate.realtime_priority;
    }
    let candidate_saved_token_count = prefix_family_saved_token_count(
        candidate.shared_prefix_token_count,
        candidate.capped_request_count,
    );
    let current_saved_token_count = prefix_family_saved_token_count(
        current.shared_prefix_token_count,
        current.capped_request_count,
    );
    if candidate_saved_token_count != current_saved_token_count {
        return candidate_saved_token_count > current_saved_token_count;
    }
    if candidate.capped_request_count != current.capped_request_count {
        return candidate.capped_request_count > current.capped_request_count;
    }
    if candidate.shared_prefix_token_count != current.shared_prefix_token_count {
        return candidate.shared_prefix_token_count > current.shared_prefix_token_count;
    }
    candidate.earliest_submission_order < current.earliest_submission_order
}

/// `SparkRequestApiPrefixFamilyLeaderIsBetter`.
fn prefix_family_leader_is_better(candidate: &Slot, current: Option<&Slot>) -> bool {
    let Some(current) = current else {
        return true;
    };
    if candidate.priority != current.priority {
        return candidate.priority > current.priority;
    }
    candidate.submission_order < current.submission_order
}

impl RequestApi {
    /// `SparkRequestApiFindPrefixFamilyGroup`.
    fn find_prefix_family_group(
        groups: &[PrefixFamilyGroup],
        prefix_hash: u64,
        shared_prefix_token_count: u32,
        priority: u32,
    ) -> Option<usize> {
        groups.iter().position(|group| {
            group.valid
                && group.prefix_hash == prefix_hash
                && group.shared_prefix_token_count == shared_prefix_token_count
                && group.highest_priority == priority
        })
    }

    /// `SparkRequestApiAcquirePrefixFamilyGroup`.
    fn acquire_prefix_family_group(
        groups: &mut Vec<PrefixFamilyGroup>,
        prefix_hash: u64,
        shared_prefix_token_count: u32,
        priority: u32,
    ) -> Option<usize> {
        if let Some(index) =
            Self::find_prefix_family_group(groups, prefix_hash, shared_prefix_token_count, priority)
        {
            return Some(index);
        }
        if groups.len() >= PREFIX_FAMILY_GROUP_CAPACITY {
            return None;
        }
        groups.push(PrefixFamilyGroup {
            valid: true,
            prefix_hash,
            shared_prefix_token_count,
            highest_priority: priority,
            ..PrefixFamilyGroup::default()
        });
        Some(groups.len() - 1)
    }

    /// `SparkRequestApiAddSlotToPrefixFamilyGroup`.
    fn add_slot_to_prefix_family_group(&self, group: &mut PrefixFamilyGroup, slot_index: u32) {
        let slot = &self.slots[slot_index as usize];
        group.request_count += 1;
        if group.capped_request_count < MAX_DISPATCH_REQUEST_COUNT {
            group.capped_request_count += 1;
        }
        if slot.priority > group.highest_priority {
            group.highest_priority = slot.priority;
        }
        if slot.has_realtime_priority() {
            group.realtime_priority = true;
        }
        if slot.submission_order < group.earliest_submission_order {
            group.earliest_submission_order = slot.submission_order;
        }
        let current_leader = if group.leader_slot == NO_SLOT {
            None
        } else {
            Some(&self.slots[group.leader_slot as usize])
        };
        if prefix_family_leader_is_better(slot, current_leader) {
            group.leader_slot = slot_index;
        }
    }

    /// `SparkRequestApiExtendPrefixScanHash`.
    fn extend_prefix_scan_hash(
        &mut self,
        slot_index: u32,
        block_token_count: u32,
        needed_token_count: u32,
    ) -> Result<u64, RequestApiError> {
        let mut hashed_token_count = self.slots[slot_index as usize].prefix_scan_hashed_token_count;
        let mut hash_value = self.slots[slot_index as usize].prefix_scan_hash;
        if hashed_token_count == 0
            || hashed_token_count > needed_token_count
            || (hashed_token_count % block_token_count) != 0
        {
            hashed_token_count = 0;
            hash_value = spark_core::prefix_cache::EMPTY_PARENT_HASH;
        }
        while hashed_token_count < needed_token_count {
            let window = {
                let slot = &self.slots[slot_index as usize];
                let start = hashed_token_count as usize;
                let end = (start + block_token_count as usize).min(slot.prompt_token_ids.len());
                slot.prompt_token_ids[start..end].to_vec()
            };
            let block_hash = spark_core::prefix_cache::hash_prompt_tokens(
                block_token_count,
                hash_value,
                &window,
            )?;
            hash_value = block_hash.prompt_hash;
            hashed_token_count += block_token_count;
        }
        let slot = &mut self.slots[slot_index as usize];
        slot.prefix_scan_hashed_token_count = hashed_token_count;
        slot.prefix_scan_hash = hash_value;
        Ok(hash_value)
    }

    /// `SparkGlm52RequestApiBuildBestPrefixFamilyChoice`.
    fn build_best_prefix_family_choice(&mut self) -> Option<PrefixFamilyChoice> {
        if !self.prefix_cohorting_is_enabled() || self.scheduler.borrow().prefix_cache().is_none() {
            return None;
        }
        let block_token_count = self.prefix_cache_block_token_count();
        if block_token_count == 0 {
            return None;
        }
        let mut groups: Vec<PrefixFamilyGroup> = Vec::new();
        for slot_index in 0..self.request_capacity {
            let slot = &self.slots[slot_index as usize];
            if !slot.is_schedulable_prefill() || slot.prompt_token_ids.is_empty() {
                continue;
            }
            let (scheduled_prompt_token_count, reusable_prefix_token_count) =
                self.next_prefill_step_token_count(slot_index);
            if scheduled_prompt_token_count == 0 {
                continue;
            }
            let prompt_token_count = self.slots[slot_index as usize].prompt_token_count;
            let maximum_family_prefix_token_count = {
                let maximum = (reusable_prefix_token_count + scheduled_prompt_token_count)
                    .min(prompt_token_count);
                maximum - (maximum % block_token_count)
            };
            let mut prefix_token_count = {
                let start = reusable_prefix_token_count + block_token_count;
                start - (start % block_token_count)
            };
            if prefix_token_count <= reusable_prefix_token_count {
                prefix_token_count += block_token_count;
            }
            while prefix_token_count <= maximum_family_prefix_token_count {
                let Ok(prefix_hash_value) =
                    self.extend_prefix_scan_hash(slot_index, block_token_count, prefix_token_count)
                else {
                    break;
                };
                let priority = self.slots[slot_index as usize].priority;
                let Some(group_index) = Self::acquire_prefix_family_group(
                    &mut groups,
                    prefix_hash_value,
                    prefix_token_count,
                    priority,
                ) else {
                    prefix_token_count += block_token_count;
                    continue;
                };
                // Borrow split: take the group out, update, put it back.
                let mut group = groups[group_index];
                self.add_slot_to_prefix_family_group(&mut group, slot_index);
                groups[group_index] = group;
                prefix_token_count += block_token_count;
            }
        }
        let mut best_group: Option<&PrefixFamilyGroup> = None;
        for group in &groups {
            if prefix_family_group_is_better(group, best_group) {
                best_group = Some(group);
            }
        }
        let best_group = best_group?;
        if best_group.capped_request_count < 2 {
            return None;
        }
        let saved_prompt_token_count = prefix_family_saved_token_count(
            best_group.shared_prefix_token_count,
            best_group.capped_request_count,
        );
        if best_group.leader_slot == NO_SLOT || saved_prompt_token_count == 0 {
            return None;
        }
        Some(PrefixFamilyChoice {
            leader_slot: best_group.leader_slot,
            shared_prefix_token_count: best_group.shared_prefix_token_count,
            realtime_priority: best_group.realtime_priority,
            saved_prompt_token_count,
            valid: true,
        })
    }

    /// `SparkRequestApiPrefixFamilyChoiceBeatsPrefillSlot`.
    fn prefix_family_choice_beats_prefill_slot(
        &self,
        choice: &PrefixFamilyChoice,
        slot_index: Option<u32>,
    ) -> bool {
        if !choice.valid || choice.leader_slot == NO_SLOT || choice.saved_prompt_token_count == 0 {
            return false;
        }
        let Some(slot_index) = slot_index else {
            return true;
        };
        let slot = &self.slots[slot_index as usize];
        let leader = &self.slots[choice.leader_slot as usize];
        if leader.priority != slot.priority {
            return leader.priority > slot.priority;
        }
        if choice.realtime_priority != slot.has_realtime_priority() {
            return choice.realtime_priority;
        }
        choice.saved_prompt_token_count != 0
    }

    /// `SparkRequestApiBatchBucketCapacityForSequenceCount` (reimplemented
    /// inline: the stage plan's value helper is private — first bucket >=
    /// count).
    pub(crate) fn batch_bucket_capacity_for_sequence_count(active_sequence_count: u32) -> u32 {
        for &bucket in &stage_plan::BATCH_BUCKETS {
            if active_sequence_count <= bucket {
                return bucket;
            }
        }
        0
    }

    /// `SparkRequestApiSlotIsCompatiblePrefillBatchMember`; returns the
    /// candidate's scheduled token count when compatible.
    fn slot_is_compatible_prefill_batch_member(
        &mut self,
        leader_slot_index: u32,
        candidate_slot_index: u32,
        leader_prefill_block_count: u32,
        require_resident_cached_blocks: bool,
    ) -> Option<u32> {
        if candidate_slot_index == leader_slot_index {
            return None;
        }
        let candidate = &self.slots[candidate_slot_index as usize];
        if !candidate.is_schedulable_prefill()
            || self.slot_can_speculate(candidate_slot_index)
                != self.slot_can_speculate(leader_slot_index)
        {
            return None;
        }
        if (require_resident_cached_blocks
            || candidate.priority < self.slots[leader_slot_index as usize].priority)
            && !self.prefill_cached_blocks_are_resident(
                candidate_slot_index,
                candidate.prompt_token_count,
            )
        {
            return None;
        }
        let (candidate_scheduled_prompt_token_count, _) =
            self.next_prefill_step_token_count(candidate_slot_index);
        let candidate_prefill_block_count =
            self.prefill_block_count_for_scheduled_tokens(candidate_scheduled_prompt_token_count);
        if candidate_prefill_block_count == 0
            || candidate_prefill_block_count != leader_prefill_block_count
        {
            return None;
        }
        Some(candidate_scheduled_prompt_token_count)
    }

    /// `SparkRequestApiEvaluatePrefillBatchShape`.
    fn evaluate_prefill_batch_shape(
        &mut self,
        slot_index: u32,
        require_resident_cached_blocks: bool,
    ) -> Option<PrefillBatchShape> {
        let slot = &self.slots[slot_index as usize];
        if !slot.is_schedulable_prefill() {
            return None;
        }
        let prompt_token_count = slot.prompt_token_count;
        if require_resident_cached_blocks
            && !self.prefill_cached_blocks_are_resident(slot_index, prompt_token_count)
        {
            return None;
        }
        let (scheduled_prompt_token_count, _) = self.next_prefill_step_token_count(slot_index);
        if scheduled_prompt_token_count == 0 {
            return None;
        }
        let resident_cached_blocks =
            self.prefill_cached_blocks_are_resident(slot_index, prompt_token_count);
        let leader_prefill_block_count =
            self.prefill_block_count_for_scheduled_tokens(scheduled_prompt_token_count);
        if leader_prefill_block_count == 0 {
            return None;
        }
        let realtime_priority = self.slots[slot_index as usize].has_realtime_priority();
        let mut compatible_request_count = 1u32;
        if self.prefill_batching_is_enabled() {
            let mut batch_target = self.current_pipeline_batch_width();
            if batch_target > MAX_DISPATCH_REQUEST_COUNT {
                batch_target = MAX_DISPATCH_REQUEST_COUNT;
            }
            for candidate_index in 0..self.request_capacity {
                if compatible_request_count >= batch_target {
                    break;
                }
                if self
                    .slot_is_compatible_prefill_batch_member(
                        slot_index,
                        candidate_index,
                        leader_prefill_block_count,
                        true,
                    )
                    .is_some()
                {
                    compatible_request_count += 1;
                }
            }
        }
        let bucket_capacity =
            Self::batch_bucket_capacity_for_sequence_count(compatible_request_count);
        if bucket_capacity == 0 {
            return None;
        }
        Some(PrefillBatchShape {
            slot: slot_index,
            scheduled_prompt_token_count,
            resident_cached_blocks,
            realtime_priority,
            compatible_request_count,
            graph_padding_count: bucket_capacity - compatible_request_count,
        })
    }

    /// `SparkRequestApiPrefillShapeIsBetter`.
    fn prefill_shape_is_better(
        &self,
        candidate: &PrefillBatchShape,
        current: Option<&PrefillBatchShape>,
    ) -> bool {
        let Some(current) = current else {
            return true;
        };
        let candidate_slot = &self.slots[candidate.slot as usize];
        let current_slot = &self.slots[current.slot as usize];
        if candidate_slot.priority != current_slot.priority {
            return candidate_slot.priority > current_slot.priority;
        }
        if candidate.realtime_priority != current.realtime_priority {
            return candidate.realtime_priority;
        }
        if candidate.resident_cached_blocks != current.resident_cached_blocks {
            return candidate.resident_cached_blocks;
        }
        if candidate.compatible_request_count != current.compatible_request_count {
            return candidate.compatible_request_count > current.compatible_request_count;
        }
        if candidate.graph_padding_count != current.graph_padding_count {
            return candidate.graph_padding_count < current.graph_padding_count;
        }
        if candidate.scheduled_prompt_token_count != current.scheduled_prompt_token_count {
            return candidate.scheduled_prompt_token_count > current.scheduled_prompt_token_count;
        }
        candidate_slot.has_higher_scheduling_priority_than(Some(current_slot))
    }

    /// `SparkRequestApiFindBestPrefillSlot`.
    pub(crate) fn find_best_prefill_slot(
        &mut self,
        require_resident_cached_blocks: bool,
    ) -> Option<u32> {
        let mut best_shape: Option<PrefillBatchShape> = None;
        for slot_index in 0..self.request_capacity {
            let Some(candidate_shape) =
                self.evaluate_prefill_batch_shape(slot_index, require_resident_cached_blocks)
            else {
                continue;
            };
            if self.prefill_shape_is_better(&candidate_shape, best_shape.as_ref()) {
                best_shape = Some(candidate_shape);
            }
        }
        best_shape.map(|shape| shape.slot)
    }

    /// `SparkRequestApiShouldFillDecodeBatch`.
    fn should_fill_decode_batch(
        &mut self,
        prefill_slot_index: u32,
        decode_slot_index: u32,
    ) -> bool {
        if !self.decode_batching_is_enabled() {
            return false;
        }
        let prefill = &self.slots[prefill_slot_index as usize];
        let decode = &self.slots[decode_slot_index as usize];
        if prefill.priority < decode.priority {
            return false;
        }
        let decode_priority = decode.priority;
        let batch_target = self.current_pipeline_batch_width();
        let mut ready_decode_count = 0u32;
        for slot in &self.slots {
            if ready_decode_count >= batch_target {
                break;
            }
            if slot.is_schedulable_decode() && slot.priority == decode_priority {
                ready_decode_count += 1;
            }
        }
        ready_decode_count < batch_target
    }

    /// `SparkRequestApiMtpVerifyOutranksDecode` (see the C comment: a
    /// pending draft is sunk cost, so the verify always outranks plain
    /// decode at equal priority).
    fn mtp_verify_outranks_decode(&self, speculative_verify_slot_index: u32) -> bool {
        self.slots[speculative_verify_slot_index as usize].state == STATE_READY_SPECULATIVE_VERIFY
    }

    /// `SparkRequestApiChooseReadySlot`; returns `(chosen_slot,
    /// chosen_is_prefill)`.
    pub(crate) fn choose_ready_slot(
        &mut self,
        prefill_slot: Option<u32>,
        decode_slot: Option<u32>,
        speculative_verify_slot: Option<u32>,
    ) -> (Option<u32>, bool) {
        self.scheduler.get_mut().set_prefill_demand(if prefill_slot.is_some() { 1 } else { 0 });
        let mut chosen_is_prefill = false;
        let mut chosen_slot = decode_slot;
        let mut chosen_is_decode = true;
        if let Some(prefill_index) = prefill_slot {
            let prefill_wins = match chosen_slot {
                None => true,
                Some(chosen_index) => {
                    self.slots[prefill_index as usize].has_higher_scheduling_priority_than(Some(
                        &self.slots[chosen_index as usize],
                    )) || (chosen_is_decode
                        && decode_slot == Some(chosen_index)
                        && ((self.slots[prefill_index as usize]
                            .has_same_scheduling_priority(&self.slots[chosen_index as usize])
                            && self.prefill_has_resident_kv_headroom(prefill_index))
                            || self.should_fill_decode_batch(prefill_index, chosen_index)))
                }
            };
            if prefill_wins {
                chosen_is_prefill = true;
                chosen_is_decode = false;
                chosen_slot = Some(prefill_index);
            }
        }
        let _ = chosen_is_decode;
        if let Some(verify_index) = speculative_verify_slot {
            let verify_wins = match chosen_slot {
                None => true,
                Some(chosen_index) => {
                    let verify_slot = &self.slots[verify_index as usize];
                    let chosen = &self.slots[chosen_index as usize];
                    verify_slot.priority > chosen.priority
                        || (chosen_is_prefill && verify_slot.has_same_scheduling_priority(chosen))
                        || (decode_slot == Some(chosen_index)
                            && !chosen_is_prefill
                            && verify_slot.has_same_scheduling_priority(chosen)
                            && self.mtp_verify_outranks_decode(verify_index))
                }
            };
            if verify_wins {
                chosen_is_prefill = false;
                chosen_slot = Some(verify_index);
            }
        }
        (chosen_slot, chosen_is_prefill)
    }

    /// `SparkRequestApiFillPrefillSchedulerRequest`. The C shares the
    /// slot's prompt array; the Rust borrow splitter clones it into
    /// `prompt_storage` first (see module docs) and the request borrows the
    /// clone.
    fn fill_prefill_scheduler_request<'a>(
        &self,
        slot_index: u32,
        prompt_token_count: u32,
        max_scheduled_prompt_token_count: u32,
        prompt_storage: &'a [u32],
    ) -> SchedulerRequest<'a> {
        let slot = &self.slots[slot_index as usize];
        let computed_prompt_token_count = if self.cross_sequence_prefix_reuse_is_enabled() {
            0
        } else {
            slot.computed_prompt_token_count.max(slot.dispatched_prompt_token_count)
        };
        let max_scheduled = if max_scheduled_prompt_token_count != 0 {
            max_scheduled_prompt_token_count
        } else {
            slot.max_prefill_tokens_per_step
        };
        SchedulerRequest {
            active_sequence_count: 1,
            prompt_token_count,
            flags: REQUEST_FLAG_PREFILL,
            computed_prompt_token_count,
            cached_prefix_token_count: 0,
            max_scheduled_prompt_token_count: max_scheduled,
            sequence_id: slot.sequence_id,
            prompt_token_ids: Some(prompt_storage),
        }
    }

    /// `SparkRequestApiFillDecodeSchedulerRequest` /
    /// `SparkRequestApiFillSpeculativeVerifySchedulerRequest` (identical in
    /// C).
    fn fill_decode_scheduler_request() -> SchedulerRequest<'static> {
        SchedulerRequest {
            active_sequence_count: 1,
            prompt_token_count: 0,
            flags: REQUEST_FLAG_DECODE,
            computed_prompt_token_count: 0,
            cached_prefix_token_count: 0,
            max_scheduled_prompt_token_count: NO_PROMPT_LIMIT,
            sequence_id: 0,
            prompt_token_ids: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Scheduling entry points (the C statics + `SparkRequestApiScheduleNext`).
// ---------------------------------------------------------------------------

impl RequestApi {
    /// `SparkRequestApiSchedulePrefill`.
    fn schedule_prefill(
        &mut self,
        slot_index: u32,
        selected_shared_prefix_token_count: u32,
        dispatch: &mut Dispatch,
    ) -> Result<(), RequestApiError> {
        let mut reserved_block_count = self.resident_kv_block_count();
        if !self.reserve_prefill_resident_kv_blocks(slot_index, &mut reserved_block_count) {
            return Err(RequestApiError::Busy);
        }

        let mut shared_prefix_token_count = selected_shared_prefix_token_count;
        if shared_prefix_token_count == 0 {
            shared_prefix_token_count = self.find_best_shared_prefix_token_count(slot_index);
        }
        let reusable_prefix_token_count = self.probe_reusable_prefix_token_count(slot_index);
        let scheduler_prompt_token_count = self.slots[slot_index as usize].prompt_token_count;
        let mut scheduler_step_token_limit = 0u32;
        if shared_prefix_token_count > reusable_prefix_token_count {
            scheduler_step_token_limit = shared_prefix_token_count - reusable_prefix_token_count;
        }
        if !self.prefill_cached_blocks_are_resident(slot_index, scheduler_prompt_token_count) {
            return Err(RequestApiError::Busy);
        }

        let prompt_token_ids = self.slots[slot_index as usize].prompt_token_ids.clone();
        let scheduler_request = self.fill_prefill_scheduler_request(
            slot_index,
            scheduler_prompt_token_count,
            scheduler_step_token_limit,
            &prompt_token_ids,
        );
        let prefill_decision = self.scheduler.get_mut().admit(&scheduler_request)?;
        if !prefill_decision.accepted {
            dispatch.prefill_decision = Some(prefill_decision);
            return Ok(());
        }

        let mut committed_prefix_token_count = prefill_decision.cache_commit_token_count_after_step;
        {
            let slot = &mut self.slots[slot_index as usize];
            if committed_prefix_token_count > slot.prompt_token_count {
                committed_prefix_token_count = slot.prompt_token_count;
            }
            if slot.state == STATE_QUEUED_PREFILL {
                slot.state = STATE_RUNNING_PREFILL;
                self.queued_request_count -= 1;
            }
            if slot.inflight_prefill_dispatch_count == 0
                && slot.dispatched_prompt_token_count < slot.computed_prompt_token_count
            {
                slot.dispatched_prompt_token_count = slot.computed_prompt_token_count;
            }
            if committed_prefix_token_count > slot.dispatched_prompt_token_count {
                slot.dispatched_prompt_token_count = committed_prefix_token_count;
            }
            slot.inflight_prefill_dispatch_count += 1;
            slot.scheduled_prefill_step_count += 1;
        }
        self.running_request_count += 1;
        self.scheduled_prefill_dispatch_count += 1;

        dispatch.accepted = true;
        dispatch.kind = DispatchKind::Prefill;
        dispatch.request_count = 1;
        dispatch.highest_priority = self.slots[slot_index as usize].priority;
        dispatch.shared_prefix_token_count =
            shared_prefix_token_count.min(committed_prefix_token_count);
        let block_tokens = self.prefix_cache_block_token_count();
        if let Some(shared_prefix_block_count) =
            dispatch.shared_prefix_token_count.checked_div(block_tokens)
        {
            dispatch.shared_prefix_block_count = shared_prefix_block_count;
        }
        dispatch.prefix_cache_parent_hash = prefill_decision.prefix_cache_parent_hash;
        dispatch.prefix_cache_result_hash = prefill_decision.prefix_cache_result_hash;
        {
            let slot = &self.slots[slot_index as usize];
            dispatch.request_handles[0] = slot.handle;
            dispatch.request_slot_indices[0] = slot_index;
            dispatch.request_ids[0] = slot.request_id;
            dispatch.sequence_ids[0] = slot.sequence_id;
        }
        dispatch.prefill_decision = Some(prefill_decision);
        if self.slot_can_speculate(slot_index) {
            dispatch.flags |= DISPATCH_FLAG_DSPARK_TAP_CAPTURE;
            self.dspark_tap_capture_dispatch_count += 1;
        }

        if dispatch.shared_prefix_token_count == 0 {
            return Ok(());
        }

        let shared_prefix_token_count = dispatch.shared_prefix_token_count;
        let leader_can_speculate = self.slot_can_speculate(slot_index);
        for candidate_index in 0..self.request_capacity {
            if dispatch.request_count >= MAX_DISPATCH_REQUEST_COUNT {
                break;
            }
            if candidate_index == slot_index {
                continue;
            }
            let candidate = &self.slots[candidate_index as usize];
            if candidate.state != STATE_QUEUED_PREFILL
                || self.slot_can_speculate(candidate_index) != leader_can_speculate
                || self.shared_cache_prefix_token_count(slot_index, candidate_index)
                    < shared_prefix_token_count
            {
                continue;
            }
            {
                let candidate = &mut self.slots[candidate_index as usize];
                candidate.state = STATE_WAITING_PREFIX_COHORT;
                candidate.scheduled_prefill_step_count += 1;
            }
            self.queued_request_count -= 1;
            self.running_request_count += 1;
            let request_count = dispatch.request_count as usize;
            let candidate = &self.slots[candidate_index as usize];
            dispatch.request_handles[request_count] = candidate.handle;
            dispatch.request_slot_indices[request_count] = candidate_index;
            dispatch.request_ids[request_count] = candidate.request_id;
            dispatch.sequence_ids[request_count] = candidate.sequence_id;
            dispatch.request_count += 1;
        }
        if dispatch.request_count > 1 {
            dispatch.flags |= DISPATCH_FLAG_PREFIX_COHORT;
            if selected_shared_prefix_token_count != 0 {
                dispatch.flags |= DISPATCH_FLAG_PREFIX_FAMILY_SELECTED;
            }
            self.prefix_family_dispatch_count += 1;
            self.prefix_family_member_count += u64::from(dispatch.request_count);
            self.prefix_family_saved_prompt_token_count += prefix_family_saved_token_count(
                dispatch.shared_prefix_token_count,
                dispatch.request_count,
            );
        }
        Ok(())
    }

    /// `SparkRequestApiFindBestPrefillBatchMember`; returns `(slot_index,
    /// scheduled_token_count)`.
    fn find_best_prefill_batch_member(
        &mut self,
        leader_slot_index: u32,
        leader_prefill_block_count: u32,
        selected_handles: &[u64],
        require_resident_cached_blocks: bool,
    ) -> Option<(u32, u32)> {
        let mut best: Option<(u32, u32)> = None;
        for slot_index in 0..self.request_capacity {
            let Some(candidate_scheduled_prompt_token_count) = self
                .slot_is_compatible_prefill_batch_member(
                    leader_slot_index,
                    slot_index,
                    leader_prefill_block_count,
                    require_resident_cached_blocks,
                )
            else {
                continue;
            };
            let handle = self.slots[slot_index as usize].handle;
            if selected_handles.contains(&handle) {
                continue;
            }
            // SparkRequestApiPrefillBatchCandidateIsBetter.
            let is_better = match best {
                None => true,
                Some((best_index, best_scheduled)) => {
                    let candidate = &self.slots[slot_index as usize];
                    let best_slot = &self.slots[best_index as usize];
                    if candidate.priority != best_slot.priority {
                        candidate.priority > best_slot.priority
                    } else if candidate_scheduled_prompt_token_count != best_scheduled {
                        candidate_scheduled_prompt_token_count > best_scheduled
                    } else {
                        candidate.submission_order < best_slot.submission_order
                    }
                }
            };
            if is_better {
                best = Some((slot_index, candidate_scheduled_prompt_token_count));
            }
        }
        best
    }

    /// `SparkRequestApiSchedulePrefillBatch`.
    fn schedule_prefill_batch(
        &mut self,
        first_slot_index: u32,
        dispatch: &mut Dispatch,
    ) -> Result<(), RequestApiError> {
        if !self.prefill_batching_is_enabled() {
            return Err(RequestApiError::NotFound);
        }
        if self.slots[first_slot_index as usize].state != STATE_QUEUED_PREFILL {
            return Err(RequestApiError::NotFound);
        }
        let first_prompt_token_count = self.slots[first_slot_index as usize].prompt_token_count;
        if !self.prefill_cached_blocks_are_resident(first_slot_index, first_prompt_token_count) {
            return Err(RequestApiError::Busy);
        }
        let (leader_scheduled_prompt_token_count, _) =
            self.next_prefill_step_token_count(first_slot_index);
        let leader_prefill_block_count =
            self.prefill_block_count_for_scheduled_tokens(leader_scheduled_prompt_token_count);
        if leader_prefill_block_count == 0 {
            return Err(RequestApiError::NotFound);
        }

        let mut batch_target = self.current_pipeline_batch_width();
        if batch_target > MAX_DISPATCH_REQUEST_COUNT {
            batch_target = MAX_DISPATCH_REQUEST_COUNT;
        }
        let require_resident_batch_members = self.slots[first_slot_index as usize]
            .has_realtime_priority()
            || self.async_jit_prefetch_is_enabled();
        let mut reserved_block_count = self.resident_kv_block_count();
        let mut selected_slots: Vec<u32> = Vec::new();
        let mut selected_handles: Vec<u64> = Vec::new();
        let mut selected_scheduled_prompt_token_counts: Vec<u32> = Vec::new();
        let mut slot = Some(first_slot_index);
        while let Some(slot_index) = slot {
            if selected_slots.len() as u32 >= batch_target {
                break;
            }
            if !self.reserve_prefill_resident_kv_blocks(slot_index, &mut reserved_block_count) {
                if selected_slots.is_empty() {
                    return Err(RequestApiError::Busy);
                }
                break;
            }
            if selected_slots.is_empty() {
                selected_scheduled_prompt_token_counts.push(leader_scheduled_prompt_token_count);
            }
            selected_slots.push(slot_index);
            selected_handles.push(self.slots[slot_index as usize].handle);
            slot = match self.find_best_prefill_batch_member(
                first_slot_index,
                leader_prefill_block_count,
                &selected_handles,
                require_resident_batch_members,
            ) {
                Some((next_slot_index, next_scheduled)) => {
                    selected_scheduled_prompt_token_counts.push(next_scheduled);
                    Some(next_slot_index)
                }
                None => None,
            };
        }
        let request_count = selected_slots.len() as u32;
        if request_count < 2 {
            return Err(RequestApiError::NotFound);
        }

        self.run_slot_array_critical_jit_kv_prefetch(&selected_slots, dispatch)?;
        for &slot_index in &selected_slots {
            let prompt_token_count = self.slots[slot_index as usize].prompt_token_count;
            if !self.prefill_cached_blocks_are_resident(slot_index, prompt_token_count) {
                dispatch.flags |= DISPATCH_FLAG_JIT_PREFETCH_PENDING;
                return Err(RequestApiError::Busy);
            }
        }

        // Build the scheduler requests; the prompts are cloned so the
        // scheduler call does not alias the slot table (see module docs).
        let scheduler_prompts: Vec<Vec<u32>> = selected_slots
            .iter()
            .map(|&slot_index| self.slots[slot_index as usize].prompt_token_ids.clone())
            .collect();
        let mut scheduler_requests: Vec<SchedulerRequest> = Vec::new();
        for (request_index, &slot_index) in selected_slots.iter().enumerate() {
            let prompt_token_count = self.slots[slot_index as usize].prompt_token_count;
            scheduler_requests.push(self.fill_prefill_scheduler_request(
                slot_index,
                prompt_token_count,
                selected_scheduled_prompt_token_counts[request_index],
                &scheduler_prompts[request_index],
            ));
        }
        let prefill_batch_decision =
            self.scheduler.get_mut().admit_prefill_batch(&scheduler_requests)?;
        if !prefill_batch_decision.accepted {
            dispatch.prefill_batch_decision = Some(prefill_batch_decision);
            return Ok(());
        }

        dispatch.accepted = true;
        dispatch.kind = DispatchKind::PrefillBatch;
        dispatch.flags |= DISPATCH_FLAG_PREFILL_BATCH;
        dispatch.request_count = prefill_batch_decision.packed_request_count;
        dispatch.highest_priority = self.slots[first_slot_index as usize].priority;
        for (request_index, &selected_slot_index) in
            selected_slots.iter().enumerate().take(dispatch.request_count as usize)
        {
            let lane_committed =
                prefill_batch_decision.lanes[request_index].cache_commit_token_count_after_step;
            {
                let slot = &mut self.slots[selected_slot_index as usize];
                slot.state = STATE_RUNNING_PREFILL;
                slot.scheduled_prefill_step_count += 1;
                if slot.inflight_prefill_dispatch_count == 0
                    && slot.dispatched_prompt_token_count < slot.computed_prompt_token_count
                {
                    slot.dispatched_prompt_token_count = slot.computed_prompt_token_count;
                }
                let mut lane_committed = lane_committed;
                if lane_committed > slot.prompt_token_count {
                    lane_committed = slot.prompt_token_count;
                }
                if lane_committed > slot.dispatched_prompt_token_count {
                    slot.dispatched_prompt_token_count = lane_committed;
                }
                slot.inflight_prefill_dispatch_count += 1;
            }
            self.queued_request_count -= 1;
            self.running_request_count += 1;
            let slot = &self.slots[selected_slot_index as usize];
            dispatch.request_handles[request_index] = slot.handle;
            dispatch.request_slot_indices[request_index] = selected_slot_index;
            dispatch.request_ids[request_index] = slot.request_id;
            dispatch.sequence_ids[request_index] = slot.sequence_id;
        }
        if self.slot_can_speculate(first_slot_index) {
            dispatch.flags |= DISPATCH_FLAG_DSPARK_TAP_CAPTURE;
            self.dspark_tap_capture_dispatch_count += 1;
        }
        dispatch.prefill_batch_decision = Some(prefill_batch_decision);
        self.scheduled_prefill_dispatch_count += 1;
        Ok(())
    }

    /// `SparkRequestApiAdmitDecodeBatchMembers` (shared by the decode and
    /// speculative-verify batch schedulers). `fill_prefill` selects the C's
    /// `fill_scheduler_request` callback — both C fills are identical
    /// decode requests, so no callback is needed here.
    fn admit_decode_batch_members(
        &mut self,
        selected_slots: &mut Vec<u32>,
        context_extension: u32,
        dispatch: &mut Dispatch,
    ) -> Result<(), RequestApiError> {
        self.apply_active_kv_block_budget(selected_slots, context_extension)?;
        if selected_slots.is_empty() {
            return Err(RequestApiError::NotFound);
        }
        for &slot_index in selected_slots.iter() {
            self.ensure_decode_slot_kv_capacity(slot_index, context_extension)?;
        }
        self.run_slot_array_critical_jit_kv_prefetch(selected_slots, dispatch)?;
        for &slot_index in selected_slots.iter() {
            if !self.decode_blocks_are_resident(slot_index) {
                dispatch.flags |= DISPATCH_FLAG_JIT_PREFETCH_PENDING;
                return Err(RequestApiError::Busy);
            }
        }
        let scheduler_requests: Vec<SchedulerRequest> =
            selected_slots.iter().map(|_| Self::fill_decode_scheduler_request()).collect();
        let decode_batch_decision =
            self.scheduler.get_mut().admit_decode_batch(&scheduler_requests)?;
        let accepted = decode_batch_decision.accepted;
        dispatch.decode_batch_decision = Some(decode_batch_decision);
        if !accepted {
            return Ok(());
        }
        Ok(())
    }

    /// `SparkRequestApiScheduleSpeculativeVerifyBatch`.
    fn schedule_speculative_verify_batch(
        &mut self,
        first_slot_index: u32,
        dispatch: &mut Dispatch,
    ) -> Result<(), RequestApiError> {
        let preferred_source =
            if self.configuration_flags & CONFIGURATION_FLAG_PREFER_DSPARK_SPECULATION != 0
                && self.speculation_is_enabled()
            {
                SPECULATIVE_SOURCE_DRAFTER
            } else {
                0
            };
        let (leader_draft, leader_source) =
            self.get_slot_speculative_draft(first_slot_index, preferred_source)?;
        if leader_draft.token_count == 0 {
            return Err(RequestApiError::NotFound);
        }

        let mut batch_target =
            if self.decode_batching_is_enabled() { self.current_pipeline_batch_width() } else { 1 };
        if batch_target > MAX_DISPATCH_REQUEST_COUNT {
            batch_target = MAX_DISPATCH_REQUEST_COUNT;
        }
        {
            let verifier_row_count = if leader_source == SPECULATIVE_SOURCE_MTP {
                mtp_tree::VERIFIER_ROW_COUNT
            } else {
                leader_draft.token_count + 1
            };
            let row_limited_batch_target = self.decode_execution_row_capacity / verifier_row_count;
            if row_limited_batch_target == 0 {
                return Err(RequestApiError::CapacityExceeded);
            }
            if batch_target > row_limited_batch_target {
                batch_target = row_limited_batch_target;
            }
        }
        let require_resident_batch_members = self.slots[first_slot_index as usize]
            .has_realtime_priority()
            || !self.jit_prefetch_is_enabled()
            || self.async_jit_prefetch_is_enabled();
        let mut selected_slots = self.collect_speculative_verify_batch_members(
            first_slot_index,
            leader_draft.token_count,
            leader_source,
            require_resident_batch_members,
            batch_target,
        );
        let speculative_context_extension = if leader_source == SPECULATIVE_SOURCE_MTP {
            mtp_tree::CONTEXT_EXTENSION
        } else {
            leader_draft.token_count
        };
        self.admit_decode_batch_members(
            &mut selected_slots,
            speculative_context_extension,
            dispatch,
        )?;
        let decode_batch_decision =
            dispatch.decode_batch_decision.as_ref().ok_or(RequestApiError::InternalError)?;
        if !decode_batch_decision.accepted {
            return Ok(());
        }

        dispatch.accepted = true;
        dispatch.kind = DispatchKind::SpeculativeVerifyBatch;
        if leader_source == SPECULATIVE_SOURCE_MTP {
            if leader_draft.token_count != mtp_tree::CANDIDATE_COUNT {
                return Err(RequestApiError::ModuleNotValidated);
            }
            dispatch.flags |= DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY | DISPATCH_FLAG_MTP_TREE_VERIFY;
            dispatch.mtp_draft_token_budget = mtp_tree::CANDIDATE_COUNT;
            dispatch.speculative_verifier_token_count = mtp_tree::VERIFIER_ROW_COUNT;
            dispatch.speculative_max_committed_token_count = mtp_tree::MAX_COMMITTED_TOKEN_COUNT;
        } else {
            dispatch.flags |=
                DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY | DISPATCH_FLAG_DSPARK_TAP_CAPTURE;
            dispatch.speculative_verifier_token_count = leader_draft.token_count + 1;
            dispatch.speculative_max_committed_token_count = leader_draft.token_count + 1;
        }
        if leader_source == SPECULATIVE_SOURCE_MTP && self.slot_can_speculate(first_slot_index) {
            dispatch.flags |= DISPATCH_FLAG_DSPARK_TAP_CAPTURE;
        }
        if dispatch.flags & DISPATCH_FLAG_DSPARK_TAP_CAPTURE != 0 {
            self.dspark_tap_capture_dispatch_count += 1;
        }
        dispatch.request_count = decode_batch_decision.packed_request_count;
        dispatch.highest_priority = self.slots[first_slot_index as usize].priority;
        dispatch.speculative_token_count = leader_draft.token_count;

        for (request_index, &selected_slot_index) in
            selected_slots.iter().enumerate().take(dispatch.request_count as usize)
        {
            let (draft_result, draft_source) =
                self.get_slot_speculative_draft(selected_slot_index, leader_source)?;
            if draft_source != leader_source || draft_result.token_count != leader_draft.token_count
            {
                return Err(RequestApiError::InvalidArgument);
            }
            {
                let slot = &mut self.slots[selected_slot_index as usize];
                slot.state = STATE_RUNNING_SPECULATIVE_VERIFY;
                slot.scheduled_decode_token_count += draft_result.token_count;
            }
            self.running_request_count += 1;
            {
                let slot = &self.slots[selected_slot_index as usize];
                dispatch.request_handles[request_index] = slot.handle;
                dispatch.request_slot_indices[request_index] = selected_slot_index;
                dispatch.request_ids[request_index] = slot.request_id;
                dispatch.sequence_ids[request_index] = slot.sequence_id;
            }
            for token_index in 0..draft_result.token_count as usize {
                dispatch.speculative_draft_token_ids[request_index][token_index] =
                    draft_result.token_ids[token_index];
                dispatch.speculative_confidence_milli[request_index][token_index] =
                    draft_result.confidence_milli[token_index];
            }
        }
        self.scheduled_decode_dispatch_count += 1;
        if leader_source == SPECULATIVE_SOURCE_MTP {
            self.mtp_verify_dispatch_count += 1;
        } else {
            self.dspark_verify_dispatch_count += 1;
        }
        Ok(())
    }

    /// `SparkRequestApiScheduleDecodeBatch`.
    fn schedule_decode_batch(
        &mut self,
        first_slot_index: u32,
        dispatch: &mut Dispatch,
    ) -> Result<(), RequestApiError> {
        let mut batch_disables_speculation = false;
        let batch_target =
            if self.decode_batching_is_enabled() { self.current_pipeline_batch_width() } else { 1 };
        let require_resident_batch_members = self.slots[first_slot_index as usize]
            .has_realtime_priority()
            || !self.jit_prefetch_is_enabled()
            || self.async_jit_prefetch_is_enabled();
        let mut selected_slots = self.collect_decode_batch_members(
            first_slot_index,
            require_resident_batch_members,
            batch_target,
        );
        let mtp_draft_token_budget = self.decode_batch_mtp_budget(&selected_slots);
        self.admit_decode_batch_members(&mut selected_slots, mtp_draft_token_budget, dispatch)?;
        let decode_batch_decision =
            dispatch.decode_batch_decision.as_ref().ok_or(RequestApiError::InternalError)?;
        if !decode_batch_decision.accepted {
            return Ok(());
        }

        dispatch.accepted = true;
        dispatch.kind = DispatchKind::DecodeBatch;
        dispatch.request_count = decode_batch_decision.packed_request_count;
        dispatch.highest_priority = self.slots[first_slot_index as usize].priority;
        for (request_index, &selected_slot_index) in
            selected_slots.iter().enumerate().take(dispatch.request_count as usize)
        {
            self.discard_mtp_draft(selected_slot_index);
            {
                let slot = &mut self.slots[selected_slot_index as usize];
                slot.state = STATE_RUNNING_DECODE;
                slot.scheduled_decode_token_count += 1;
            }
            self.running_request_count += 1;
            {
                let slot = &self.slots[selected_slot_index as usize];
                dispatch.request_handles[request_index] = slot.handle;
                dispatch.request_slot_indices[request_index] = selected_slot_index;
                dispatch.request_ids[request_index] = slot.request_id;
                dispatch.sequence_ids[request_index] = slot.sequence_id;
            }
            dispatch.decode_committed_token_counts[request_index] = 1;
            if self.slots[selected_slot_index as usize].flags
                & super::REQUEST_FLAG_DISABLE_SPECULATION
                != 0
            {
                batch_disables_speculation = true;
            }
        }
        if self.slot_can_speculate(first_slot_index) {
            dispatch.flags |= DISPATCH_FLAG_DSPARK_TAP_CAPTURE;
        }
        if mtp_draft_token_budget != 0 && !batch_disables_speculation {
            dispatch.flags |= DISPATCH_FLAG_MTP_COMMIT;
            dispatch.mtp_draft_token_budget = mtp_draft_token_budget;
        }
        if dispatch.flags & DISPATCH_FLAG_DSPARK_TAP_CAPTURE != 0 {
            self.dspark_tap_capture_dispatch_count += 1;
        }
        self.scheduled_decode_dispatch_count += 1;
        Ok(())
    }

    /// `SparkRequestApiScheduleNext`.
    pub fn schedule_next(&mut self, dispatch: &mut Dispatch) -> Result<(), RequestApiError> {
        self.validate()?;

        initialize_dispatch(dispatch);
        if self.async_jit_prefetch_is_enabled() {
            let completed_prefetch_count_before_poll = self.async_jit_prefetch_completion_count;
            self.poll_pending_jit_kv_prefetches()?;
            if self.async_jit_prefetch_completion_count > completed_prefetch_count_before_poll {
                dispatch.flags |= DISPATCH_FLAG_JIT_PREFETCHED_KV;
            }
        }
        self.refresh_lookahead_prefix_protections()?;

        let mut selected_shared_prefix_token_count = 0u32;
        let mut overlaps_pending_prefetch = false;
        let mut prefill_slot = self.find_best_prefill_slot(false);
        if let Some(prefix_family_choice) = self.build_best_prefix_family_choice() {
            if self.prefix_family_choice_beats_prefill_slot(&prefix_family_choice, prefill_slot) {
                prefill_slot = Some(prefix_family_choice.leader_slot);
                selected_shared_prefix_token_count = prefix_family_choice.shared_prefix_token_count;
            }
        }
        let resident_required = !self.jit_prefetch_is_enabled();
        let mut speculative_verify_slot = self.find_best_schedulable_slot(
            Slot::is_schedulable_speculative_verify,
            &[],
            resident_required,
        );
        let mut decode_slot =
            self.find_best_schedulable_slot(Slot::is_schedulable_decode, &[], resident_required);
        if prefill_slot.is_none() && decode_slot.is_none() && speculative_verify_slot.is_none() {
            return Err(RequestApiError::NotFound);
        }

        let (mut chosen_slot, mut chosen_is_prefill) =
            self.choose_ready_slot(prefill_slot, decode_slot, speculative_verify_slot);

        let mut status = match chosen_slot {
            Some(chosen_index) => {
                self.run_slot_array_critical_jit_kv_prefetch(&[chosen_index], dispatch)
            }
            None => Err(RequestApiError::NotFound),
        };
        if status == Err(RequestApiError::Busy) && self.async_jit_prefetch_is_enabled() {
            let pending_dispatch_flags = dispatch.flags | DISPATCH_FLAG_JIT_PREFETCH_PENDING;
            overlaps_pending_prefetch = true;
            selected_shared_prefix_token_count = 0;
            prefill_slot = self.find_best_prefill_slot(true);
            speculative_verify_slot =
                self.find_best_schedulable_slot(Slot::is_schedulable_speculative_verify, &[], true);
            decode_slot = self.find_best_schedulable_slot(Slot::is_schedulable_decode, &[], true);
            let (re_chosen_slot, re_chosen_is_prefill) =
                self.choose_ready_slot(prefill_slot, decode_slot, speculative_verify_slot);
            chosen_slot = re_chosen_slot;
            chosen_is_prefill = re_chosen_is_prefill;
            let Some(chosen_index) = chosen_slot else {
                return Err(RequestApiError::Busy);
            };
            initialize_dispatch(dispatch);
            dispatch.flags = pending_dispatch_flags;
            status = self.run_slot_array_critical_jit_kv_prefetch(&[chosen_index], dispatch);
        }
        status?;
        if !overlaps_pending_prefetch {
            self.run_opportunistic_jit_kv_prefetch(chosen_slot)?;
        }

        if chosen_is_prefill {
            let prefill_slot_index = prefill_slot.ok_or(RequestApiError::InternalError)?;
            if (decode_slot.is_some()
                && self.slots[prefill_slot_index as usize].has_higher_scheduling_priority_than(
                    decode_slot.map(|index| &self.slots[index as usize]),
                ))
                || self.older_lower_priority_schedulable_slot_exists(prefill_slot_index)
            {
                dispatch.flags |= DISPATCH_FLAG_PRIORITY_PREEMPTED_QUEUE;
            }
            let shared_prefix = if selected_shared_prefix_token_count != 0 {
                selected_shared_prefix_token_count
            } else {
                self.find_best_shared_prefix_token_count(prefill_slot_index)
            };
            if shared_prefix <= self.probe_reusable_prefix_token_count(prefill_slot_index) {
                match self.schedule_prefill_batch(prefill_slot_index, dispatch) {
                    Ok(()) if dispatch.accepted => return Ok(()),
                    Ok(()) => {}
                    Err(RequestApiError::Busy) | Err(RequestApiError::NotFound) => {}
                    Err(error) => return Err(error),
                }
            }
            let prefill_status = self.schedule_prefill(
                prefill_slot_index,
                selected_shared_prefix_token_count,
                dispatch,
            );
            match prefill_status {
                Ok(()) if dispatch.accepted => return Ok(()),
                Ok(()) => {}
                Err(RequestApiError::Busy) => {}
                Err(error) => return Err(error),
            }
            if decode_slot.is_none() && speculative_verify_slot.is_none() {
                return prefill_status;
            }
            chosen_slot = decode_slot.or(speculative_verify_slot);
            chosen_is_prefill = false;
            let saved_dispatch_flags = dispatch.flags & DISPATCH_FLAG_JIT_PREFETCH_PENDING;
            initialize_dispatch(dispatch);
            dispatch.flags = saved_dispatch_flags;
            let _ = chosen_is_prefill;
        }

        if chosen_slot == speculative_verify_slot && speculative_verify_slot.is_some() {
            let verify_index = speculative_verify_slot.ok_or(RequestApiError::InternalError)?;
            if !self.decode_blocks_are_resident(verify_index) {
                dispatch.flags |= DISPATCH_FLAG_JIT_PREFETCH_PENDING;
                return Err(RequestApiError::Busy);
            }
            if self.older_lower_priority_schedulable_slot_exists(verify_index) {
                dispatch.flags |= DISPATCH_FLAG_PRIORITY_PREEMPTED_QUEUE;
            }
            return self.schedule_speculative_verify_batch(verify_index, dispatch);
        }

        let decode_index = decode_slot.ok_or(RequestApiError::InvalidArgument)?;
        if !self.decode_blocks_are_resident(decode_index) {
            dispatch.flags |= DISPATCH_FLAG_JIT_PREFETCH_PENDING;
            return Err(RequestApiError::Busy);
        }
        if self.older_lower_priority_schedulable_slot_exists(decode_index) {
            dispatch.flags |= DISPATCH_FLAG_PRIORITY_PREEMPTED_QUEUE;
        }
        self.schedule_decode_batch(decode_index, dispatch)
    }
}

// ---------------------------------------------------------------------------
// Completion / cancel / retry (the C's dispatch lifecycle tail).
// ---------------------------------------------------------------------------

impl RequestApi {
    /// `SparkRequestApiFinishSlotAfterPrefill`.
    fn finish_slot_after_prefill(&mut self, slot_index: u32, decision: &SchedulerDecision) {
        let mut committed_prompt_token_count = decision.cache_commit_token_count_after_step;
        let prefix_cache_result_hash = decision.prefix_cache_result_hash;
        {
            let slot = &mut self.slots[slot_index as usize];
            if slot.inflight_prefill_dispatch_count != 0 {
                slot.inflight_prefill_dispatch_count -= 1;
            }
            slot.completed_prefill_step_count += 1;
            if committed_prompt_token_count > slot.prompt_token_count {
                committed_prompt_token_count = slot.prompt_token_count;
            }
            if committed_prompt_token_count > slot.computed_prompt_token_count {
                slot.computed_prompt_token_count = committed_prompt_token_count;
                slot.last_committed_prefix_token_count = committed_prompt_token_count;
                slot.last_committed_prefix_hash = prefix_cache_result_hash;
            }
        }
        self.running_request_count -= 1;
        let slot = &mut self.slots[slot_index as usize];
        if slot.computed_prompt_token_count < slot.prompt_token_count {
            if slot.inflight_prefill_dispatch_count != 0 {
                return;
            }
            if slot.dispatched_prompt_token_count > slot.computed_prompt_token_count {
                slot.dispatched_prompt_token_count = slot.computed_prompt_token_count;
            }
            slot.state = STATE_QUEUED_PREFILL;
            self.queued_request_count += 1;
            return;
        }
        if slot.inflight_prefill_dispatch_count != 0 {
            return;
        }
        slot.state = STATE_READY_DECODE;
        if slot.remaining_thinking_token_budget == 0 && slot.remaining_output_token_budget == 0 {
            slot.state = STATE_COMPLETED;
            self.completed_request_count += 1;
        }
    }

    /// `SparkRequestApiFinishSlotAfterPrefillBatchLane`.
    fn finish_slot_after_prefill_batch_lane(
        &mut self,
        slot_index: u32,
        lane: &spark_sched::scheduler::PrefillBatchLane,
    ) {
        let mut committed_prompt_token_count = lane.cache_commit_token_count_after_step;
        let prefix_cache_result_hash = lane.prefix_cache_result_hash;
        {
            let slot = &mut self.slots[slot_index as usize];
            if slot.inflight_prefill_dispatch_count != 0 {
                slot.inflight_prefill_dispatch_count -= 1;
            }
            if committed_prompt_token_count > slot.prompt_token_count {
                committed_prompt_token_count = slot.prompt_token_count;
            }
            if committed_prompt_token_count > slot.computed_prompt_token_count {
                slot.computed_prompt_token_count = committed_prompt_token_count;
                slot.last_committed_prefix_token_count = committed_prompt_token_count;
                slot.last_committed_prefix_hash = prefix_cache_result_hash;
            }
            slot.completed_prefill_step_count += 1;
        }
        self.running_request_count -= 1;
        let slot = &mut self.slots[slot_index as usize];
        if slot.computed_prompt_token_count < slot.prompt_token_count {
            if slot.dispatched_prompt_token_count > slot.computed_prompt_token_count {
                slot.dispatched_prompt_token_count = slot.computed_prompt_token_count;
            }
            slot.state = STATE_QUEUED_PREFILL;
            self.queued_request_count += 1;
            return;
        }
        slot.state = STATE_READY_DECODE;
        if slot.remaining_thinking_token_budget == 0 && slot.remaining_output_token_budget == 0 {
            slot.state = STATE_COMPLETED;
            self.completed_request_count += 1;
        }
    }

    /// `SparkRequestApiCompleteDispatch`.
    pub fn complete_dispatch(&mut self, dispatch: &Dispatch) -> Result<(), RequestApiError> {
        self.validate()?;
        if !dispatch.accepted || dispatch.request_count == 0 {
            return Err(RequestApiError::InvalidArgument);
        }

        if dispatch.kind == DispatchKind::Prefill {
            let prefill_decision =
                dispatch.prefill_decision.as_ref().ok_or(RequestApiError::InvalidArgument)?;
            self.scheduler.get_mut().complete(prefill_decision)?;
            let cache_commit_token_count_after_step =
                prefill_decision.cache_commit_token_count_after_step;
            for request_index in 0..dispatch.request_count as usize {
                let Some(slot_index) =
                    self.find_slot_by_handle(dispatch.request_handles[request_index])
                else {
                    return Err(RequestApiError::InvalidArgument);
                };
                let state = self.slots[slot_index as usize].state;
                if request_index == 0 && state == STATE_CANCELLED {
                    self.stale_prefill_completion_count += 1;
                    continue;
                }
                if (request_index == 0 && state != STATE_RUNNING_PREFILL)
                    || (request_index != 0 && state != STATE_WAITING_PREFIX_COHORT)
                {
                    return Err(RequestApiError::InvalidArgument);
                }
                if request_index != 0 && cache_commit_token_count_after_step != 0 {
                    let source_sequence_id = dispatch.sequence_ids[0];
                    let destination_sequence_id = self.slots[slot_index as usize].sequence_id;
                    let scheduler = self.scheduler.get_mut();
                    let prefix_cache =
                        scheduler.prefix_cache_mut().ok_or(RequestApiError::InvalidArgument)?;
                    prefix_cache.bind_committed_prefix_from_sequence(
                        source_sequence_id,
                        destination_sequence_id,
                        cache_commit_token_count_after_step,
                    )?;
                }
                self.finish_slot_after_prefill(slot_index, prefill_decision);
            }
            return Ok(());
        }
        if dispatch.kind == DispatchKind::PrefillBatch {
            let prefill_batch_decision =
                dispatch.prefill_batch_decision.as_ref().ok_or(RequestApiError::InvalidArgument)?;
            self.scheduler.get_mut().complete_prefill_batch(prefill_batch_decision)?;
            for request_index in 0..dispatch.request_count as usize {
                let slot_index = self
                    .find_slot_by_handle(dispatch.request_handles[request_index])
                    .ok_or(RequestApiError::InvalidArgument)?;
                if self.slots[slot_index as usize].state != STATE_RUNNING_PREFILL {
                    return Err(RequestApiError::InvalidArgument);
                }
                self.finish_slot_after_prefill_batch_lane(
                    slot_index,
                    &prefill_batch_decision.lanes[request_index],
                );
            }
            return Ok(());
        }
        if dispatch.kind == DispatchKind::DecodeBatch {
            let decode_batch_decision =
                dispatch.decode_batch_decision.as_ref().ok_or(RequestApiError::InvalidArgument)?;
            self.scheduler.get_mut().complete_decode_batch(decode_batch_decision)?;
            for request_index in 0..dispatch.request_count as usize {
                let slot_index = self
                    .find_slot_by_handle(dispatch.request_handles[request_index])
                    .ok_or(RequestApiError::InvalidArgument)?;
                if self.slots[slot_index as usize].state != STATE_RUNNING_DECODE {
                    return Err(RequestApiError::InvalidArgument);
                }
                let mut committed_token_count = 1u32;
                if dispatch.flags & DISPATCH_FLAG_MTP_COMMIT != 0 {
                    committed_token_count = dispatch.decode_committed_token_counts[request_index];
                    if committed_token_count != 1
                        || committed_token_count
                            > self.slots[slot_index as usize].remaining_decode_budget()
                    {
                        return Err(RequestApiError::InvalidArgument);
                    }
                }
                self.finish_slot_after_decode(slot_index, committed_token_count);
            }
            return Ok(());
        }
        if dispatch.kind == DispatchKind::SpeculativeVerifyBatch {
            let decode_batch_decision =
                dispatch.decode_batch_decision.as_ref().ok_or(RequestApiError::InvalidArgument)?;
            self.scheduler.get_mut().complete_decode_batch(decode_batch_decision)?;
            if dispatch.speculative_token_count == 0
                || dispatch.speculative_token_count > MAX_SPECULATIVE_TOKENS
            {
                return Err(RequestApiError::InvalidArgument);
            }
            for request_index in 0..dispatch.request_count as usize {
                let slot_index = self
                    .find_slot_by_handle(dispatch.request_handles[request_index])
                    .ok_or(RequestApiError::InvalidArgument)?;
                if self.slots[slot_index as usize].state != STATE_RUNNING_SPECULATIVE_VERIFY {
                    return Err(RequestApiError::InvalidArgument);
                }
                self.finish_slot_after_speculative_verify(
                    slot_index,
                    dispatch.speculative_token_count,
                    dispatch.speculative_accepted_token_counts[request_index],
                    dispatch.speculative_committed_token_counts[request_index],
                    dispatch.speculative_fallback_token_ids[request_index],
                    dispatch.speculative_resolution_path_ids[request_index],
                    dispatch.flags & DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY != 0,
                )?;
            }
            return Ok(());
        }
        Err(RequestApiError::InvalidArgument)
    }

    /// `SparkRequestApiCancelDispatch`.
    pub fn cancel_dispatch(&mut self, dispatch: &Dispatch) -> Result<(), RequestApiError> {
        self.validate()?;
        if !dispatch.accepted {
            return Err(RequestApiError::InvalidArgument);
        }
        if dispatch.kind == DispatchKind::Prefill {
            let prefill_decision =
                dispatch.prefill_decision.as_ref().ok_or(RequestApiError::InvalidArgument)?;
            self.scheduler.get_mut().cancel(prefill_decision)?;
            for request_index in 0..dispatch.request_count as usize {
                let slot_index = self
                    .find_slot_by_handle(dispatch.request_handles[request_index])
                    .ok_or(RequestApiError::InvalidArgument)?;
                let state = self.slots[slot_index as usize].state;
                if (request_index == 0 && state != STATE_RUNNING_PREFILL)
                    || (request_index != 0 && state != STATE_WAITING_PREFIX_COHORT)
                {
                    return Err(RequestApiError::InvalidArgument);
                }
                self.running_request_count -= 1;
                self.slots[slot_index as usize].state = STATE_CANCELLED;
                self.cancelled_request_count += 1;
                self.release_slot_sequence(slot_index)?;
            }
            return Ok(());
        }
        if dispatch.kind == DispatchKind::PrefillBatch {
            let prefill_batch_decision =
                dispatch.prefill_batch_decision.as_ref().ok_or(RequestApiError::InvalidArgument)?;
            self.scheduler.get_mut().cancel_prefill_batch(prefill_batch_decision)?;
            for request_index in 0..dispatch.request_count as usize {
                let slot_index = self
                    .find_slot_by_handle(dispatch.request_handles[request_index])
                    .ok_or(RequestApiError::InvalidArgument)?;
                if self.slots[slot_index as usize].state != STATE_RUNNING_PREFILL {
                    return Err(RequestApiError::InvalidArgument);
                }
                self.running_request_count -= 1;
                self.slots[slot_index as usize].state = STATE_CANCELLED;
                self.cancelled_request_count += 1;
                self.release_slot_sequence(slot_index)?;
            }
            return Ok(());
        }
        if dispatch.kind == DispatchKind::DecodeBatch {
            let decode_batch_decision =
                dispatch.decode_batch_decision.as_ref().ok_or(RequestApiError::InvalidArgument)?;
            self.scheduler.get_mut().cancel_decode_batch(decode_batch_decision)?;
            for request_index in 0..dispatch.request_count as usize {
                let slot_index = self
                    .find_slot_by_handle(dispatch.request_handles[request_index])
                    .ok_or(RequestApiError::InvalidArgument)?;
                if self.slots[slot_index as usize].state != STATE_RUNNING_DECODE {
                    return Err(RequestApiError::InvalidArgument);
                }
                self.running_request_count -= 1;
                self.slots[slot_index as usize].state = STATE_CANCELLED;
                self.cancelled_request_count += 1;
                self.release_slot_sequence(slot_index)?;
            }
            return Ok(());
        }
        if dispatch.kind == DispatchKind::SpeculativeVerifyBatch {
            let decode_batch_decision =
                dispatch.decode_batch_decision.as_ref().ok_or(RequestApiError::InvalidArgument)?;
            self.scheduler.get_mut().cancel_decode_batch(decode_batch_decision)?;
            for request_index in 0..dispatch.request_count as usize {
                let slot_index = self
                    .find_slot_by_handle(dispatch.request_handles[request_index])
                    .ok_or(RequestApiError::InvalidArgument)?;
                if self.slots[slot_index as usize].state != STATE_RUNNING_SPECULATIVE_VERIFY {
                    return Err(RequestApiError::InvalidArgument);
                }
                if self.speculation_is_enabled() {
                    let sequence_id = self.slots[slot_index as usize].sequence_id;
                    if let Some(speculator) = &mut self.model_speculator {
                        let _ = speculator.cancel_sequence(sequence_id);
                    }
                }
                self.running_request_count -= 1;
                self.slots[slot_index as usize].state = STATE_CANCELLED;
                self.cancelled_request_count += 1;
                self.release_slot_sequence(slot_index)?;
            }
            return Ok(());
        }
        Err(RequestApiError::InvalidArgument)
    }

    /// `SparkRequestApiRetryDecodeTokenCount`.
    fn retry_decode_token_count(dispatch: &Dispatch) -> u32 {
        if dispatch.kind == DispatchKind::SpeculativeVerifyBatch {
            return dispatch.speculative_token_count;
        }
        1
    }

    /// `SparkRequestApiValidateRetryDecodeCounters`.
    fn validate_retry_decode_counters(&self, dispatch: &Dispatch) -> Result<(), RequestApiError> {
        if !dispatch.accepted
            || (dispatch.kind != DispatchKind::DecodeBatch
                && dispatch.kind != DispatchKind::SpeculativeVerifyBatch)
            || dispatch.request_count == 0
            || self.running_request_count < dispatch.request_count
            || self.scheduled_decode_dispatch_count == 0
        {
            return Err(RequestApiError::InvalidArgument);
        }
        if Self::retry_decode_token_count(dispatch) == 0 {
            return Err(RequestApiError::InvalidArgument);
        }
        if dispatch.kind == DispatchKind::SpeculativeVerifyBatch {
            if dispatch.flags & DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY != 0 {
                if self.mtp_verify_dispatch_count == 0 {
                    return Err(RequestApiError::InvalidArgument);
                }
            } else if self.dspark_verify_dispatch_count == 0 {
                return Err(RequestApiError::InvalidArgument);
            }
        }
        Ok(())
    }

    /// `SparkRequestApiValidateRetryDecodeSlots`.
    fn validate_retry_decode_slots(&self, dispatch: &Dispatch) -> Result<(), RequestApiError> {
        let retry_token_count = Self::retry_decode_token_count(dispatch);
        for request_index in 0..dispatch.request_count as usize {
            let slot_index = self
                .find_slot_by_handle(dispatch.request_handles[request_index])
                .ok_or(RequestApiError::InvalidArgument)?;
            let expected_state = if dispatch.kind == DispatchKind::DecodeBatch {
                STATE_RUNNING_DECODE
            } else {
                STATE_RUNNING_SPECULATIVE_VERIFY
            };
            let slot = &self.slots[slot_index as usize];
            if slot.state != expected_state || slot.scheduled_decode_token_count < retry_token_count
            {
                return Err(RequestApiError::InvalidArgument);
            }
        }
        Ok(())
    }

    /// `SparkRequestApiRestoreRetriedDecodeSlots`.
    fn restore_retried_decode_slots(&mut self, dispatch: &Dispatch) {
        let retry_token_count = Self::retry_decode_token_count(dispatch);
        for request_index in 0..dispatch.request_count as usize {
            let Some(slot_index) =
                self.find_slot_by_handle(dispatch.request_handles[request_index])
            else {
                continue;
            };
            let slot = &mut self.slots[slot_index as usize];
            slot.scheduled_decode_token_count -= retry_token_count;
            slot.state = if dispatch.kind == DispatchKind::DecodeBatch {
                STATE_READY_DECODE
            } else {
                STATE_READY_SPECULATIVE_VERIFY
            };
        }
    }

    /// `SparkRequestApiRestoreRetriedDecodeCounters`.
    fn restore_retried_decode_counters(&mut self, dispatch: &Dispatch) {
        self.running_request_count -= dispatch.request_count;
        self.scheduled_decode_dispatch_count -= 1;
        if dispatch.kind != DispatchKind::SpeculativeVerifyBatch {
            return;
        }
        if dispatch.flags & DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY != 0 {
            self.mtp_verify_dispatch_count -= 1;
        } else {
            self.dspark_verify_dispatch_count -= 1;
        }
    }

    /// `SparkRequestApiRetryDecodeDispatch`.
    pub fn retry_decode_dispatch(&mut self, dispatch: &Dispatch) -> Result<(), RequestApiError> {
        self.validate()?;
        self.validate_retry_decode_counters(dispatch)?;
        self.validate_retry_decode_slots(dispatch)?;
        let decode_batch_decision =
            dispatch.decode_batch_decision.as_ref().ok_or(RequestApiError::InvalidArgument)?;
        self.scheduler.get_mut().cancel_decode_batch(decode_batch_decision)?;
        self.restore_retried_decode_slots(dispatch);
        self.restore_retried_decode_counters(dispatch);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Describe / copy views and KV block tables.
// ---------------------------------------------------------------------------

/// `SparkRequestApiDispatchLaneCount`.
fn dispatch_lane_count(dispatch: &Dispatch) -> u32 {
    match dispatch.kind {
        DispatchKind::Prefill => {
            dispatch.prefill_decision.as_ref().map_or(0, |decision| decision.active_sequence_count)
        }
        DispatchKind::PrefillBatch => dispatch
            .prefill_batch_decision
            .as_ref()
            .map_or(0, |decision| decision.active_sequence_count),
        DispatchKind::DecodeBatch | DispatchKind::SpeculativeVerifyBatch => dispatch
            .decode_batch_decision
            .as_ref()
            .map_or(0, |decision| decision.active_sequence_count),
        DispatchKind::None => 0,
    }
}

/// `SparkRequestApiDescribePrefillDispatch` (free function in C).
pub fn describe_prefill_dispatch(
    dispatch: &Dispatch,
) -> Result<PrefillDispatchView, RequestApiError> {
    if !dispatch.accepted {
        return Err(RequestApiError::InvalidArgument);
    }
    if dispatch.kind == DispatchKind::Prefill {
        let decision =
            dispatch.prefill_decision.as_ref().ok_or(RequestApiError::InvalidArgument)?;
        if !decision.accepted
            || decision.active_sequence_count == 0
            || decision.active_sequence_count > 1
            || decision.scheduled_prompt_token_count == 0
            || decision.prompt_token_ids.is_empty()
        {
            return Err(RequestApiError::InvalidArgument);
        }
        return Ok(PrefillDispatchView {
            kind: dispatch.kind,
            active_sequence_count: decision.active_sequence_count,
            lane_count: 1,
            prompt_token_offset: decision.scheduled_prompt_token_offset,
            prompt_token_count: decision.scheduled_prompt_token_count,
            prompt_token_stride: decision.scheduled_prompt_token_count,
            lanes: vec![PrefillDispatchLaneView {
                request_index: 0,
                prompt_token_offset: decision.scheduled_prompt_token_offset,
                prompt_token_count: decision.scheduled_prompt_token_count,
                request_slot_index: dispatch.request_slot_indices[0],
                request_id: dispatch.request_ids[0],
                sequence_id: dispatch.sequence_ids[0],
                request_handle: dispatch.request_handles[0],
                prompt_token_ids: decision.prompt_token_ids.clone(),
            }],
        });
    }
    if dispatch.kind == DispatchKind::PrefillBatch {
        let batch_decision =
            dispatch.prefill_batch_decision.as_ref().ok_or(RequestApiError::InvalidArgument)?;
        let lane_count = batch_decision.active_sequence_count;
        let prompt_token_stride = batch_decision.maximum_scheduled_prompt_token_count;
        if !batch_decision.accepted
            || lane_count == 0
            || lane_count > MAX_DISPATCH_REQUEST_COUNT
            || prompt_token_stride == 0
        {
            return Err(RequestApiError::InvalidArgument);
        }
        let mut prefill_view = PrefillDispatchView {
            kind: dispatch.kind,
            active_sequence_count: lane_count,
            lane_count,
            prompt_token_stride,
            ..PrefillDispatchView::default()
        };
        for lane_index in 0..lane_count as usize {
            let lane = &batch_decision.lanes[lane_index];
            if lane.active_sequence_count == 0
                || lane.scheduled_prompt_token_count == 0
                || lane.scheduled_prompt_token_count > prompt_token_stride
                || lane.prompt_token_ids.is_empty()
            {
                return Err(RequestApiError::InvalidArgument);
            }
            if lane_index == 0 {
                prefill_view.prompt_token_offset = lane.scheduled_prompt_token_offset;
            } else if prefill_view.prompt_token_offset != lane.scheduled_prompt_token_offset {
                prefill_view.prompt_token_offset = 0;
            }
            prefill_view.prompt_token_count =
                prefill_view.prompt_token_count.max(lane.scheduled_prompt_token_count);
            prefill_view.lanes.push(PrefillDispatchLaneView {
                request_index: lane.request_index,
                prompt_token_offset: lane.scheduled_prompt_token_offset,
                prompt_token_count: lane.scheduled_prompt_token_count,
                request_slot_index: dispatch.request_slot_indices[lane_index],
                request_id: dispatch.request_ids[lane_index],
                sequence_id: dispatch.sequence_ids[lane_index],
                request_handle: dispatch.request_handles[lane_index],
                prompt_token_ids: lane.prompt_token_ids.clone(),
            });
        }
        return Ok(prefill_view);
    }
    Err(RequestApiError::InvalidArgument)
}

/// `SparkRequestApiCopyPrefillDispatchTokenIds` (free function in C).
pub fn copy_prefill_dispatch_token_ids(
    dispatch: &Dispatch,
    destination_token_ids: &mut [u32],
    destination_token_stride: u32,
    destination_lane_capacity: u32,
) -> Result<(), RequestApiError> {
    let prefill_view = describe_prefill_dispatch(dispatch)?;
    if destination_lane_capacity < prefill_view.lane_count
        || destination_token_stride < prefill_view.prompt_token_stride
        || (destination_token_ids.len() as u64)
            < u64::from(prefill_view.lane_count - 1) * u64::from(destination_token_stride)
                + u64::from(prefill_view.prompt_token_stride)
    {
        return Err(RequestApiError::InvalidArgument);
    }
    for (lane_index, lane) in prefill_view.lanes.iter().enumerate() {
        let destination_lane = &mut destination_token_ids[lane_index
            * destination_token_stride as usize
            ..lane_index * destination_token_stride as usize + destination_token_stride as usize];
        let copy_count = lane.prompt_token_count as usize;
        let source_offset = lane.prompt_token_offset as usize;
        destination_lane[..copy_count]
            .copy_from_slice(&lane.prompt_token_ids[source_offset..source_offset + copy_count]);
        destination_lane[copy_count..].fill(0);
    }
    Ok(())
}

impl RequestApi {
    /// `SparkRequestApiDescribeDecodeDispatch`.
    pub fn describe_decode_dispatch(
        &self,
        dispatch: &Dispatch,
    ) -> Result<DecodeDispatchView, RequestApiError> {
        self.validate()?;
        if !dispatch.accepted
            || (dispatch.kind != DispatchKind::DecodeBatch
                && dispatch.kind != DispatchKind::SpeculativeVerifyBatch)
            || dispatch.request_count == 0
            || dispatch.request_count > MAX_DISPATCH_REQUEST_COUNT
        {
            return Err(RequestApiError::InvalidArgument);
        }
        let mut decode_view = DecodeDispatchView {
            kind: dispatch.kind,
            active_sequence_count: dispatch.request_count,
            lane_count: dispatch.request_count,
            speculative_token_count: dispatch.speculative_token_count,
            ..DecodeDispatchView::default()
        };
        for lane_index in 0..dispatch.request_count as usize {
            let slot_index = self
                .find_slot_by_handle(dispatch.request_handles[lane_index])
                .ok_or(RequestApiError::InvalidArgument)?;
            let slot = &self.slots[slot_index as usize];
            if dispatch.request_slot_indices[lane_index] != slot_index
                || slot.computed_prompt_token_count == 0
                || (slot.state != STATE_RUNNING_DECODE
                    && slot.state != STATE_RUNNING_SPECULATIVE_VERIFY)
            {
                return Err(RequestApiError::InvalidArgument);
            }
            let sequence_position = u64::from(slot.computed_prompt_token_count)
                + u64::from(slot.completed_decode_token_count)
                - 1;
            if sequence_position > u64::from(u32::MAX) - 1 {
                return Err(RequestApiError::InvalidArgument);
            }
            if slot.mtp_resolution_proposed_token_count == 0 {
                if slot.mtp_resolution_base_position != 0
                    || slot.mtp_resolution_accepted_token_count != 0
                    || slot.mtp_resolution_committed_token_count != 0
                    || slot.mtp_resolution_path_id != mtp_tree::RESOLUTION_NONE
                {
                    return Err(RequestApiError::InternalError);
                }
            } else if slot.mtp_resolution_proposed_token_count > MTP_MAX_DRAFT_TOKEN_COUNT
                || slot.mtp_resolution_committed_token_count
                    != slot.mtp_resolution_accepted_token_count + 1
                || slot.mtp_resolution_base_position
                    > u64::MAX - u64::from(slot.mtp_resolution_committed_token_count)
                || slot.mtp_resolution_base_position
                    + u64::from(slot.mtp_resolution_committed_token_count)
                    != sequence_position
                || !mtp_tree::resolution_is_valid(
                    slot.mtp_resolution_proposed_token_count,
                    slot.mtp_resolution_accepted_token_count,
                    slot.mtp_resolution_path_id,
                )
            {
                return Err(RequestApiError::InternalError);
            }
            decode_view.lanes.push(DecodeDispatchLaneView {
                request_index: lane_index as u32,
                sequence_position: sequence_position as u32,
                context_token_count: (sequence_position + 1) as u32,
                request_slot_index: slot_index,
                request_id: slot.request_id,
                sequence_id: slot.sequence_id,
                request_handle: slot.handle,
                mtp_resolution_base_position: slot.mtp_resolution_base_position,
                mtp_resolution_proposed_token_count: slot.mtp_resolution_proposed_token_count,
                mtp_resolution_accepted_token_count: slot.mtp_resolution_accepted_token_count,
                mtp_resolution_committed_token_count: slot.mtp_resolution_committed_token_count,
                mtp_resolution_path_id: slot.mtp_resolution_path_id,
            });
        }
        Ok(decode_view)
    }

    /// `SparkRequestApiBuildDispatchKvBlockTables`. Takes `&self` through
    /// the scheduler `RefCell` (the trait's KV-table entry point is
    /// `&self`); the slot table is only read here.
    pub fn build_dispatch_kv_block_tables(
        &self,
        dispatch: &Dispatch,
        physical_block_indices: &mut [u32],
        lane_stride: u32,
        lane_capacity: u32,
        lane_physical_block_counts: &mut [u32],
    ) -> Result<(), RequestApiError> {
        self.validate()?;
        if !dispatch.accepted
            || physical_block_indices.is_empty()
            || lane_physical_block_counts.is_empty()
            || lane_capacity == 0
            || lane_stride < lane_capacity
        {
            return Err(RequestApiError::InvalidArgument);
        }
        let lane_count_capacity = lane_physical_block_counts.len() as u32;

        if dispatch.kind == DispatchKind::Prefill {
            if lane_count_capacity < 1 {
                return Err(RequestApiError::InvalidArgument);
            }
            for count in lane_physical_block_counts.iter_mut() {
                *count = 0;
            }
            let prefill_decision =
                dispatch.prefill_decision.as_ref().ok_or(RequestApiError::InvalidArgument)?;
            let table = self.scheduler.borrow_mut().build_kv_block_table(prefill_decision)?;
            if table.len() > lane_capacity as usize || table.len() > physical_block_indices.len() {
                return Err(RequestApiError::CapacityExceeded);
            }
            physical_block_indices[..table.len()].copy_from_slice(&table);
            lane_physical_block_counts[0] = table.len() as u32;
            return Ok(());
        }

        if dispatch.kind == DispatchKind::PrefillBatch {
            let prefill_batch_decision =
                dispatch.prefill_batch_decision.as_ref().ok_or(RequestApiError::InvalidArgument)?;
            let lane_tables = self
                .scheduler
                .borrow_mut()
                .build_prefill_batch_kv_block_tables(prefill_batch_decision, lane_capacity)?;
            if lane_count_capacity < lane_tables.len() as u32 {
                return Err(RequestApiError::InvalidArgument);
            }
            for (lane_index, table) in lane_tables.iter().enumerate() {
                let base = lane_index * lane_stride as usize;
                if base + table.len() > physical_block_indices.len() {
                    return Err(RequestApiError::CapacityExceeded);
                }
                physical_block_indices[base..base + table.len()].copy_from_slice(table);
                lane_physical_block_counts[lane_index] = table.len() as u32;
            }
            return Ok(());
        }

        if dispatch.kind == DispatchKind::DecodeBatch
            || dispatch.kind == DispatchKind::SpeculativeVerifyBatch
        {
            if lane_count_capacity < dispatch.request_count {
                return Err(RequestApiError::InvalidArgument);
            }
            for count in lane_physical_block_counts.iter_mut() {
                *count = 0;
            }
            for (lane_index, lane_physical_block_count) in lane_physical_block_counts
                .iter_mut()
                .enumerate()
                .take(dispatch.request_count as usize)
            {
                let slot_index = self
                    .find_slot_by_handle(dispatch.request_handles[lane_index])
                    .ok_or(RequestApiError::InvalidArgument)?;
                {
                    let slot = &self.slots[slot_index as usize];
                    if (slot.state != STATE_RUNNING_DECODE
                        && slot.state != STATE_RUNNING_SPECULATIVE_VERIFY)
                        || slot.computed_prompt_token_count == 0
                    {
                        return Err(RequestApiError::InvalidArgument);
                    }
                }
                let speculative_token_count =
                    if dispatch.kind == DispatchKind::SpeculativeVerifyBatch {
                        if dispatch.flags & DISPATCH_FLAG_MTP_TREE_VERIFY != 0 {
                            mtp_tree::CONTEXT_EXTENSION
                        } else {
                            dispatch.speculative_token_count
                        }
                    } else if dispatch.flags & DISPATCH_FLAG_MTP_COMMIT != 0 {
                        dispatch.mtp_draft_token_budget
                    } else {
                        0
                    };
                let required_token_count =
                    self.ensure_decode_slot_kv_capacity(slot_index, speculative_token_count)?;
                let sequence_id = self.slots[slot_index as usize].sequence_id;
                let table = {
                    let mut scheduler = self.scheduler.borrow_mut();
                    let prefix_cache =
                        scheduler.prefix_cache_mut().ok_or(RequestApiError::InvalidArgument)?;
                    prefix_cache.build_physical_block_table(sequence_id, required_token_count)?
                };
                let base = lane_index * lane_stride as usize;
                if table.len() > lane_capacity as usize
                    || base + table.len() > physical_block_indices.len()
                {
                    return Err(RequestApiError::CapacityExceeded);
                }
                physical_block_indices[base..base + table.len()].copy_from_slice(&table);
                *lane_physical_block_count = table.len() as u32;
            }
            return Ok(());
        }

        Err(RequestApiError::InvalidArgument)
    }

    /// `SparkRequestApiBuildDispatchKvBlockTableView`.
    pub fn build_dispatch_kv_block_table_view<'a>(
        &mut self,
        dispatch: &Dispatch,
        host_physical_block_indices: &'a mut [u32],
        execution_physical_block_indices: Option<&'a [u32]>,
        lane_stride: u32,
        lane_capacity: u32,
        lane_physical_block_counts: &'a mut [u32],
    ) -> Result<prompt_pipeline::KvBlockTableView<'a>, RequestApiError> {
        self.validate()?;
        let lane_count = dispatch_lane_count(dispatch);
        if !dispatch.accepted
            || lane_count == 0
            || lane_count > lane_physical_block_counts.len() as u32
            || host_physical_block_indices.is_empty()
            || lane_capacity == 0
            || lane_stride < lane_capacity
        {
            return Err(RequestApiError::InvalidArgument);
        }
        self.build_dispatch_kv_block_tables(
            dispatch,
            host_physical_block_indices,
            lane_stride,
            lane_capacity,
            lane_physical_block_counts,
        )?;
        Ok(prompt_pipeline::KvBlockTableView {
            block_token_count: self.scheduler.borrow().prefix_cache_block_tokens(),
            lane_count,
            lane_stride,
            lane_capacity,
            physical_block_indices: execution_physical_block_indices
                .map_or(Some(&host_physical_block_indices[..]), Some),
            lane_physical_block_counts: Some(&lane_physical_block_counts[..]),
            host_physical_block_indices: &host_physical_block_indices[..],
            host_lane_physical_block_counts: &lane_physical_block_counts[..],
        })
    }
}

// ---------------------------------------------------------------------------
// The spark_text::prompt_pipeline::RequestApi seam: the minimized trait
// surface. The full `Dispatch` is staged by `schedule_next` for the
// follow-up calls (see module docs).
// ---------------------------------------------------------------------------

impl RequestApi {
    /// Take the staged dispatch if it matches the trait dispatch's shape.
    fn staged_dispatch_matches(&self, dispatch: &prompt_pipeline::RequestDispatch) -> bool {
        self.staged_dispatch.borrow().as_ref().is_some_and(|staged| {
            staged.accepted == dispatch.accepted && trait_kind(staged.kind) == dispatch.kind
        })
    }
}

/// Map the request-API dispatch kind onto the trait's kind.
fn trait_kind(kind: DispatchKind) -> prompt_pipeline::DispatchKind {
    match kind {
        DispatchKind::None => prompt_pipeline::DispatchKind::None,
        DispatchKind::Prefill => prompt_pipeline::DispatchKind::Prefill,
        DispatchKind::DecodeBatch => prompt_pipeline::DispatchKind::DecodeBatch,
        DispatchKind::PrefillBatch => prompt_pipeline::DispatchKind::PrefillBatch,
        DispatchKind::SpeculativeVerifyBatch => {
            prompt_pipeline::DispatchKind::SpeculativeVerifyBatch
        }
    }
}

impl prompt_pipeline::RequestApi for RequestApi {
    fn schedule_next(&mut self) -> Result<prompt_pipeline::RequestDispatch, PromptPipelineError> {
        let mut dispatch = Dispatch::default();
        self.schedule_next(&mut dispatch).map_err(PromptPipelineError::from)?;
        let request_dispatch = prompt_pipeline::RequestDispatch {
            accepted: dispatch.accepted,
            kind: trait_kind(dispatch.kind),
        };
        *self.staged_dispatch.borrow_mut() = Some(dispatch);
        Ok(request_dispatch)
    }

    fn describe_prefill_dispatch(
        &self,
        dispatch: &prompt_pipeline::RequestDispatch,
    ) -> Result<prompt_pipeline::PrefillDispatchView, PromptPipelineError> {
        if !self.staged_dispatch_matches(dispatch) {
            return Err(PromptPipelineError::InvalidArgument);
        }
        let staged = self.staged_dispatch.borrow();
        let staged = staged.as_ref().ok_or(PromptPipelineError::InvalidArgument)?;
        let view = describe_prefill_dispatch(staged).map_err(PromptPipelineError::from)?;
        Ok(prompt_pipeline::PrefillDispatchView {
            kind: dispatch.kind,
            active_sequence_count: view.active_sequence_count,
            prompt_token_offset: view.prompt_token_offset,
            prompt_token_count: view.prompt_token_count,
            prompt_token_stride: view.prompt_token_stride,
            lane_count: view.lane_count,
        })
    }

    fn copy_prefill_dispatch_token_ids(
        &self,
        dispatch: &prompt_pipeline::RequestDispatch,
        destination_token_ids: &mut [u32],
        destination_token_stride: u32,
        destination_lane_capacity: u32,
    ) -> Result<(), PromptPipelineError> {
        if !self.staged_dispatch_matches(dispatch) {
            return Err(PromptPipelineError::InvalidArgument);
        }
        let staged = self.staged_dispatch.borrow();
        let staged = staged.as_ref().ok_or(PromptPipelineError::InvalidArgument)?;
        copy_prefill_dispatch_token_ids(
            staged,
            destination_token_ids,
            destination_token_stride,
            destination_lane_capacity,
        )
        .map_err(PromptPipelineError::from)
    }

    fn build_dispatch_kv_block_table_view<'a>(
        &self,
        dispatch: &prompt_pipeline::RequestDispatch,
        host_physical_block_indices: &'a mut [u32],
        execution_physical_block_indices: Option<&'a [u32]>,
        lane_stride: u32,
        lane_capacity: u32,
        lane_physical_block_counts: &'a mut [u32],
    ) -> Result<prompt_pipeline::KvBlockTableView<'a>, PromptPipelineError> {
        if !self.staged_dispatch_matches(dispatch) {
            return Err(PromptPipelineError::InvalidArgument);
        }
        let staged = self.staged_dispatch.borrow();
        let staged = staged.as_ref().ok_or(PromptPipelineError::InvalidArgument)?;
        self.build_dispatch_kv_block_tables(
            staged,
            host_physical_block_indices,
            lane_stride,
            lane_capacity,
            lane_physical_block_counts,
        )
        .map_err(PromptPipelineError::from)?;
        let lane_count = dispatch_lane_count(staged);
        Ok(prompt_pipeline::KvBlockTableView {
            block_token_count: self.scheduler.borrow().prefix_cache_block_tokens(),
            lane_count,
            lane_stride,
            lane_capacity,
            physical_block_indices: execution_physical_block_indices
                .map_or(Some(&host_physical_block_indices[..]), Some),
            lane_physical_block_counts: Some(&lane_physical_block_counts[..]),
            host_physical_block_indices: &host_physical_block_indices[..],
            host_lane_physical_block_counts: &lane_physical_block_counts[..],
        })
    }

    fn complete_dispatch(
        &mut self,
        dispatch: &prompt_pipeline::RequestDispatch,
    ) -> Result<(), PromptPipelineError> {
        if !self.staged_dispatch_matches(dispatch) {
            return Err(PromptPipelineError::InvalidArgument);
        }
        let staged =
            self.staged_dispatch.borrow_mut().take().ok_or(PromptPipelineError::InvalidArgument)?;
        self.complete_dispatch(&staged).map_err(PromptPipelineError::from)
    }

    fn cancel_dispatch(
        &mut self,
        dispatch: &prompt_pipeline::RequestDispatch,
    ) -> Result<(), PromptPipelineError> {
        if !self.staged_dispatch_matches(dispatch) {
            return Err(PromptPipelineError::InvalidArgument);
        }
        let staged =
            self.staged_dispatch.borrow_mut().take().ok_or(PromptPipelineError::InvalidArgument)?;
        self.cancel_dispatch(&staged).map_err(PromptPipelineError::from)
    }

    fn submit(
        &mut self,
        request: &prompt_pipeline::SubmitRequest,
    ) -> Result<u64, PromptPipelineError> {
        self.submit(&super::SubmitRequest {
            flags: request.flags,
            priority: request.priority,
            prompt_token_count: request.prompt_token_count,
            thinking_token_budget: request.thinking_token_budget,
            output_token_budget: request.output_token_budget,
            max_prefill_tokens_per_step: request.max_prefill_tokens_per_step,
            request_id: request.request_id,
            sequence_id: request.sequence_id,
            prompt_token_ids: request.prompt_token_ids.to_vec(),
        })
        .map_err(PromptPipelineError::from)
    }
}
