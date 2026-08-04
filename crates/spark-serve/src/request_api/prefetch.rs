//! KV prefetch plans, the prefetch backend seam, and the JIT KV prefetch
//! machinery — port of the prefetch-plan build/mark-resident logic from
//! `cache/kv_cache.c` (`SparkKvCacheArenaBuildPrefetchPlan[FromSourceBlocks]`,
//! `SparkKvCacheArenaMarkPrefetchPlanResident[WithProtectedBlocks]`) plus the
//! request API's JIT prefetch statics (`api/request.c`).
//!
//! The `spark-core` `KvArena` does not expose these entry points, so they are
//! ported here over the arena's public API. The plan keeps the C's flat block
//! array plus its accounting counters; the prefix-cache's `Vec`-backed
//! prefetch-source probes are bridged by collecting into a deduplicated
//! source-block list first (the C's fixed `source_blocks[]` array).
//!
//! Rust-port deviations from the C arena internals:
//!   - The arena's `resident_block_capacity` has no public getter; it is
//!     recovered exactly at request-API initialization via the trim's
//!     `target > capacity` validation probe
//!     ([`probe_resident_block_capacity`]), before any block is resident.
//!   - `SparkKvCacheArenaHasValuePayload` is observable through the arena's
//!     `value_block_stride_bytes` getter.
//!   - The plan's block array is a `Vec` (bounded by
//!     [`PREFETCH_BLOCK_CAPACITY`]) instead of the C's fixed inline array.

use spark_core::kv_arena::{KvArena, KvArenaError};
use spark_core::prefix_cache::PrefetchSourceBlock;

use super::dispatch::{
    Dispatch, DISPATCH_FLAG_JIT_PREFETCHED_KV, DISPATCH_FLAG_JIT_PREFETCH_PENDING,
};
use super::slot::{
    STATE_RUNNING_DECODE, STATE_RUNNING_PREFILL, STATE_RUNNING_SPECULATIVE_VERIFY,
    STATE_WAITING_PREFIX_COHORT,
};
use super::{
    RequestApi, RequestApiError, MAX_PREFETCH_LANE_COUNT, MAX_PREFETCH_SOURCE_BLOCK_COUNT,
};

/// Maximum blocks per prefetch plan (`SPARK_KV_CACHE_PREFETCH_BLOCK_CAPACITY`).
pub const PREFETCH_BLOCK_CAPACITY: u32 = 1024;

/// Prefetch block flag: key plane (`SPARK_KV_CACHE_PREFETCH_BLOCK_FLAG_KEY`).
pub const PREFETCH_BLOCK_FLAG_KEY: u32 = 0x0000_0001;
/// Prefetch block flag: value plane (`SPARK_KV_CACHE_PREFETCH_BLOCK_FLAG_VALUE`).
pub const PREFETCH_BLOCK_FLAG_VALUE: u32 = 0x0000_0002;
/// Default prefetch block flags (`SPARK_KV_CACHE_PREFETCH_BLOCK_DEFAULT_FLAGS`).
pub const PREFETCH_BLOCK_DEFAULT_FLAGS: u32 = PREFETCH_BLOCK_FLAG_KEY | PREFETCH_BLOCK_FLAG_VALUE;

/// Prefetch block record (`SparkKvCachePrefetchBlock`, minus ABI words).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefetchBlock {
    pub lane_index: u32,
    pub physical_block_index: u32,
    pub token_capacity: u32,
    pub first_token_index: u32,
    pub token_count: u32,
    pub flags: u32,
    pub generation: u64,
    pub parent_hash: u64,
    pub block_hash: u64,
    pub content_hash: u64,
    pub key_device_address: usize,
    pub value_device_address: usize,
}

/// Prefetch plan (`SparkKvCachePrefetchPlan`, minus ABI words; the C's
/// `prefetch_block_count` is `blocks.len()`).
#[derive(Debug, Clone, Default)]
pub struct PrefetchPlan {
    pub lane_count: u32,
    pub requested_physical_block_count: u32,
    pub resident_block_count: u32,
    pub duplicate_block_count: u32,
    pub missing_block_count: u32,
    pub lane_block_counts: [u32; MAX_PREFETCH_LANE_COUNT as usize],
    pub blocks: Vec<PrefetchBlock>,
}

impl PrefetchPlan {
    /// `prefetch_plan->prefetch_block_count`.
    pub fn prefetch_block_count(&self) -> u32 {
        self.blocks.len() as u32
    }

    /// `SparkKvCachePrefetchPlanInitialize`.
    fn initialize(lane_count: u32, requested_physical_block_count: u32) -> Self {
        PrefetchPlan { lane_count, requested_physical_block_count, ..PrefetchPlan::default() }
    }

    /// `SparkKvCachePrefetchPlanAlreadyContainsBlock`.
    fn already_contains_block(&self, physical_block_index: u32) -> bool {
        self.blocks.iter().any(|block| block.physical_block_index == physical_block_index)
    }

    /// `SparkKvCachePrefetchPlanContainsBlockBeforeIndex`.
    fn contains_block_before_index(
        &self,
        block_index_limit: usize,
        physical_block_index: u32,
    ) -> bool {
        let limit = block_index_limit.min(self.blocks.len());
        self.blocks[..limit].iter().any(|block| block.physical_block_index == physical_block_index)
    }

    /// `SparkKvCacheValidatePrefetchPlan`.
    fn validate(&self) -> Result<(), RequestApiError> {
        if self.lane_count == 0
            || self.lane_count > MAX_PREFETCH_LANE_COUNT
            || self.prefetch_block_count() > PREFETCH_BLOCK_CAPACITY
        {
            return Err(RequestApiError::InvalidArgument);
        }
        for block in &self.blocks {
            if block.lane_index >= self.lane_count
                || block.flags & PREFETCH_BLOCK_DEFAULT_FLAGS == 0
                || block.flags & !PREFETCH_BLOCK_DEFAULT_FLAGS != 0
            {
                return Err(RequestApiError::InvalidArgument);
            }
        }
        Ok(())
    }
}

/// Pending async prefetch (`SparkRequestApiPendingPrefetch`).
#[derive(Debug, Clone, Default)]
pub struct PendingPrefetch {
    pub active: bool,
    pub poll_count: u32,
    pub prefetch_id: u64,
    pub prefetch_plan: PrefetchPlan,
}

/// The JIT KV prefetch dispatch seam — the C's `kv_prefetch_function` /
/// `kv_prefetch_start_function` / `kv_prefetch_poll_function` trio plus
/// `kv_prefetch_context` collapsed into a trait object. Sync dispatch calls
/// [`KvPrefetchBackend::prefetch`]; async dispatch calls
/// [`KvPrefetchBackend::start_prefetch`] then
/// [`KvPrefetchBackend::poll_prefetch`] until it stops returning
/// [`RequestApiError::Busy`].
pub trait KvPrefetchBackend {
    /// `SparkRequestApiKvPrefetchFunction` (sync JIT prefetch).
    fn prefetch(&mut self, _prefetch_plan: &PrefetchPlan) -> Result<(), RequestApiError> {
        Err(RequestApiError::NotFound)
    }

    /// `SparkRequestApiKvPrefetchStartFunction`.
    fn start_prefetch(
        &mut self,
        _prefetch_id: u64,
        _prefetch_plan: &PrefetchPlan,
    ) -> Result<(), RequestApiError> {
        Err(RequestApiError::NotFound)
    }

    /// `SparkRequestApiKvPrefetchPollFunction`;
    /// [`RequestApiError::Busy`] means still in flight.
    fn poll_prefetch(
        &mut self,
        _prefetch_id: u64,
        _prefetch_plan: &PrefetchPlan,
    ) -> Result<(), RequestApiError> {
        Err(RequestApiError::NotFound)
    }
}

/// `SparkKvCacheArenaHasValuePayload`.
fn arena_has_value_payload(arena: &KvArena) -> bool {
    arena.value_block_stride_bytes() != 0
}

/// `SparkKvCacheValidatePrefetchSourceBlock` (the arena-facing parts; the
/// flags word rides alongside the prefix cache's `PrefetchSourceBlock`,
/// which omits it — see the `spark-core` module docs).
fn validate_prefetch_source_block(
    arena: &KvArena,
    source_block: &PrefetchSourceBlock,
    flags: u32,
) -> Result<(), RequestApiError> {
    if source_block.physical_block_index >= arena.physical_block_count()
        || flags & PREFETCH_BLOCK_DEFAULT_FLAGS == 0
        || flags & !PREFETCH_BLOCK_DEFAULT_FLAGS != 0
    {
        return Err(RequestApiError::InvalidArgument);
    }
    if flags & PREFETCH_BLOCK_FLAG_VALUE != 0 && !arena_has_value_payload(arena) {
        return Err(RequestApiError::InvalidArgument);
    }
    if source_block.token_capacity != 0 && source_block.token_count > source_block.token_capacity {
        return Err(RequestApiError::InvalidArgument);
    }
    Ok(())
}

/// `SparkKvCachePrefetchPlanAddSourceBlock`.
fn prefetch_plan_add_source_block(
    arena: &KvArena,
    prefetch_plan: &mut PrefetchPlan,
    source_block: &PrefetchSourceBlock,
    flags: u32,
) -> Result<(), RequestApiError> {
    let physical_block_index = source_block.physical_block_index;
    let block_view =
        arena.resolve_block(physical_block_index).map_err(|_| RequestApiError::InternalError)?;
    let block_index = prefetch_plan.blocks.len();
    let lane_index = (block_index as u32) % prefetch_plan.lane_count;
    prefetch_plan.blocks.push(PrefetchBlock {
        lane_index,
        physical_block_index,
        token_capacity: block_view.token_capacity,
        first_token_index: source_block.first_token_index,
        token_count: source_block.token_count,
        flags,
        generation: block_view.generation,
        parent_hash: source_block.parent_hash,
        block_hash: source_block.block_hash,
        content_hash: source_block.content_hash,
        key_device_address: block_view.key_device_address,
        value_device_address: block_view.value_device_address,
    });
    prefetch_plan.lane_block_counts[lane_index as usize] += 1;
    Ok(())
}

/// `SparkKvCacheArenaBuildPrefetchPlanFromSourceBlocks`. `source_block_flags`
/// supplies the flags word the prefix-cache source probe omits (KEY|VALUE,
/// matching the C probes).
pub(crate) fn build_prefetch_plan_from_source_blocks(
    arena: &KvArena,
    source_blocks: &[PrefetchSourceBlock],
    source_block_flags: u32,
    lane_count: u32,
) -> Result<PrefetchPlan, RequestApiError> {
    if lane_count == 0 || lane_count > MAX_PREFETCH_LANE_COUNT {
        return Err(RequestApiError::InvalidArgument);
    }
    let mut prefetch_plan = PrefetchPlan::initialize(lane_count, source_blocks.len() as u32);
    for source_block in source_blocks {
        validate_prefetch_source_block(arena, source_block, source_block_flags)?;
        let physical_block_index = source_block.physical_block_index;
        if prefetch_plan.already_contains_block(physical_block_index) {
            prefetch_plan.duplicate_block_count += 1;
            continue;
        }
        if !arena.is_allocated(physical_block_index) {
            prefetch_plan.missing_block_count += 1;
            return Err(RequestApiError::NotFound);
        }
        if source_block.generation != 0
            && source_block.generation != arena.generation(physical_block_index)
        {
            return Err(RequestApiError::HashMismatch);
        }
        if arena.is_resident(physical_block_index) {
            prefetch_plan.resident_block_count += 1;
            continue;
        }
        if prefetch_plan.prefetch_block_count() >= PREFETCH_BLOCK_CAPACITY {
            return Err(RequestApiError::CapacityExceeded);
        }
        prefetch_plan_add_source_block(
            arena,
            &mut prefetch_plan,
            source_block,
            source_block_flags,
        )?;
    }
    Ok(prefetch_plan)
}

/// `SparkKvCacheArenaBuildPrefetchPlan`: plan the given physical blocks.
pub(crate) fn build_prefetch_plan(
    arena: &KvArena,
    physical_block_indices: &[u32],
    lane_count: u32,
) -> Result<PrefetchPlan, RequestApiError> {
    if physical_block_indices.len() as u32 > PREFETCH_BLOCK_CAPACITY {
        return Err(RequestApiError::CapacityExceeded);
    }
    let mut flags = PREFETCH_BLOCK_FLAG_KEY;
    if arena_has_value_payload(arena) {
        flags |= PREFETCH_BLOCK_FLAG_VALUE;
    }
    // SparkKvCachePrefetchSourceInitializeFromPhysicalBlock for each index.
    let source_blocks: Vec<PrefetchSourceBlock> = physical_block_indices
        .iter()
        .map(|&physical_block_index| PrefetchSourceBlock {
            physical_block_index,
            token_capacity: 0,
            first_token_index: 0,
            token_count: 0,
            generation: 0,
            parent_hash: 0,
            block_hash: 0,
            content_hash: 0,
        })
        .collect();
    build_prefetch_plan_from_source_blocks(arena, &source_blocks, flags, lane_count)
}

/// `SparkKvCacheArenaMakeRoomForResidentBlocks`.
fn make_room_for_resident_blocks(
    arena: &mut KvArena,
    resident_block_capacity: u32,
    prefetch_plan: &PrefetchPlan,
    protected_physical_block_indices: &[u32],
    new_resident_block_count: u32,
) -> Result<(), RequestApiError> {
    if new_resident_block_count > resident_block_capacity {
        return Err(RequestApiError::CapacityExceeded);
    }
    if arena.resident_block_count() + u64::from(new_resident_block_count)
        <= u64::from(resident_block_capacity)
    {
        return Ok(());
    }
    let target_resident_block_count = resident_block_capacity - new_resident_block_count;
    // The C victim selection skips both the protected list and the plan's
    // own blocks; the Rust trim takes a single protected list, so
    // concatenate them.
    let mut protected: Vec<u32> =
        Vec::with_capacity(protected_physical_block_indices.len() + prefetch_plan.blocks.len());
    protected.extend_from_slice(protected_physical_block_indices);
    protected.extend(prefetch_plan.blocks.iter().map(|block| block.physical_block_index));
    arena.trim_resident_blocks(&protected, target_resident_block_count).map_err(
        |error| match error {
            KvArenaError::CapacityExceeded => RequestApiError::CapacityExceeded,
            KvArenaError::InvalidArgument => RequestApiError::InvalidArgument,
            _ => RequestApiError::InternalError,
        },
    )?;
    Ok(())
}

/// `SparkKvCacheArenaMarkPrefetchPlanResidentWithProtectedBlocks`.
pub(crate) fn mark_prefetch_plan_resident_with_protected_blocks(
    arena: &mut KvArena,
    resident_block_capacity: u32,
    prefetch_plan: &PrefetchPlan,
    protected_physical_block_indices: &[u32],
) -> Result<(), RequestApiError> {
    prefetch_plan.validate()?;

    let mut new_resident_block_count = 0u32;
    for (block_index, prefetch_block) in prefetch_plan.blocks.iter().enumerate() {
        if prefetch_block.physical_block_index >= arena.physical_block_count() {
            return Err(RequestApiError::InvalidArgument);
        }
        if !arena.is_allocated(prefetch_block.physical_block_index) {
            return Err(RequestApiError::NotFound);
        }
        if arena.generation(prefetch_block.physical_block_index) != prefetch_block.generation {
            return Err(RequestApiError::HashMismatch);
        }
        if !arena.is_resident(prefetch_block.physical_block_index)
            && !prefetch_plan
                .contains_block_before_index(block_index, prefetch_block.physical_block_index)
        {
            new_resident_block_count += 1;
        }
    }

    make_room_for_resident_blocks(
        arena,
        resident_block_capacity,
        prefetch_plan,
        protected_physical_block_indices,
        new_resident_block_count,
    )?;

    for prefetch_block in &prefetch_plan.blocks {
        // The C marks even blocks already resident (idempotent flag set).
        arena
            .mark_block_resident(prefetch_block.physical_block_index)
            .map_err(|_| RequestApiError::InternalError)?;
    }
    Ok(())
}

/// `SparkKvCacheArenaMarkPrefetchPlanResident`.
pub(crate) fn mark_prefetch_plan_resident(
    arena: &mut KvArena,
    resident_block_capacity: u32,
    prefetch_plan: &PrefetchPlan,
) -> Result<(), RequestApiError> {
    mark_prefetch_plan_resident_with_protected_blocks(
        arena,
        resident_block_capacity,
        prefetch_plan,
        &[],
    )
}

/// Recover the arena's resident block capacity, which the `spark-core`
/// `KvArena` does not expose (see module docs). Exploits the trim's
/// `target > capacity ⇒ InvalidArgument` validation: with no resident
/// blocks the trim is otherwise a no-op, so the largest accepted target is
/// exactly the capacity. Must run at initialization, before any block is
/// resident.
pub(crate) fn probe_resident_block_capacity(arena: &mut KvArena) -> u32 {
    let physical_block_count = arena.physical_block_count();
    let mut low = 0u32;
    let mut high = physical_block_count;
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        match arena.trim_resident_blocks(&[], mid) {
            Ok(_) => low = mid,
            Err(_) => high = mid - 1,
        }
    }
    low
}

// ---------------------------------------------------------------------------
// RequestApi JIT prefetch machinery (ports of the C statics).
// ---------------------------------------------------------------------------

/// `SparkRequestApiCollectPhysicalBlockIndex`.
fn collect_physical_block_index(physical_block_indices: &mut Vec<u32>, physical_block_index: u32) {
    if physical_block_indices.contains(&physical_block_index) {
        return;
    }
    if physical_block_indices.len() < MAX_PREFETCH_SOURCE_BLOCK_COUNT as usize {
        physical_block_indices.push(physical_block_index);
    }
}

/// `SparkRequestApiCollectPrefetchSourceBlock`.
fn collect_prefetch_source_block(
    source_blocks: &mut Vec<PrefetchSourceBlock>,
    source_block: &PrefetchSourceBlock,
) {
    let duplicate = source_blocks.iter().any(|existing| {
        existing.physical_block_index == source_block.physical_block_index
            && existing.block_hash == source_block.block_hash
            && existing.content_hash == source_block.content_hash
    });
    if !duplicate && source_blocks.len() < MAX_PREFETCH_SOURCE_BLOCK_COUNT as usize {
        source_blocks.push(*source_block);
    }
}

impl RequestApi {
    /// `SparkRequestApiCollectPrefillSlotBlocks`.
    fn collect_prefill_slot_blocks(
        &mut self,
        slot_index: u32,
        physical_block_indices: &mut Vec<u32>,
    ) -> Result<(), RequestApiError> {
        let prompt = self.slots[slot_index as usize].prompt_token_ids.clone();
        let scheduler = self.scheduler.get_mut();
        let prefix_cache = scheduler.prefix_cache_mut().ok_or(RequestApiError::InvalidArgument)?;
        let probe =
            prefix_cache.probe_physical_block_table(&prompt, MAX_PREFETCH_SOURCE_BLOCK_COUNT)?;
        for &block in &probe.physical_block_indices {
            if physical_block_indices.len() >= MAX_PREFETCH_SOURCE_BLOCK_COUNT as usize {
                break;
            }
            collect_physical_block_index(physical_block_indices, block);
        }
        Ok(())
    }

    /// `SparkRequestApiCollectDecodeSlotBlocks`.
    fn collect_decode_slot_blocks(
        &mut self,
        slot_index: u32,
        physical_block_indices: &mut Vec<u32>,
    ) -> Result<(), RequestApiError> {
        let required_token_count = self.ensure_pending_decode_slot_kv_capacity(slot_index)?;
        let sequence_id = self.slots[slot_index as usize].sequence_id;
        let scheduler = self.scheduler.get_mut();
        let prefix_cache = scheduler.prefix_cache_mut().ok_or(RequestApiError::InvalidArgument)?;
        let table = prefix_cache.build_physical_block_table(sequence_id, required_token_count)?;
        for block in table {
            if physical_block_indices.len() >= MAX_PREFETCH_SOURCE_BLOCK_COUNT as usize {
                break;
            }
            collect_physical_block_index(physical_block_indices, block);
        }
        Ok(())
    }

    /// `SparkRequestApiCollectProtectedSlotBlocks`.
    fn collect_protected_slot_blocks(
        &mut self,
        slot_index: u32,
        physical_block_indices: &mut Vec<u32>,
    ) -> Result<(), RequestApiError> {
        let slot = &self.slots[slot_index as usize];
        if slot.is_schedulable_prefill() {
            return self.collect_prefill_slot_blocks(slot_index, physical_block_indices);
        }
        if (slot.is_schedulable_decode()
            || slot.is_schedulable_speculative_verify()
            || slot.state == STATE_RUNNING_DECODE
            || slot.state == STATE_RUNNING_SPECULATIVE_VERIFY
            || slot.state == STATE_RUNNING_PREFILL
            || slot.state == STATE_WAITING_PREFIX_COHORT)
            && slot.computed_prompt_token_count != 0
        {
            return self.collect_decode_slot_blocks(slot_index, physical_block_indices);
        }
        Ok(())
    }

    /// `SparkRequestApiCollectRunningProtectedBlocks`.
    fn collect_running_protected_blocks(
        &mut self,
        physical_block_indices: &mut Vec<u32>,
    ) -> Result<(), RequestApiError> {
        for slot_index in 0..self.request_capacity {
            if physical_block_indices.len() >= MAX_PREFETCH_SOURCE_BLOCK_COUNT as usize {
                break;
            }
            let state = self.slots[slot_index as usize].state;
            if state != STATE_RUNNING_PREFILL
                && state != STATE_RUNNING_DECODE
                && state != STATE_WAITING_PREFIX_COHORT
            {
                continue;
            }
            match self.collect_protected_slot_blocks(slot_index, physical_block_indices) {
                Ok(()) | Err(RequestApiError::NotFound) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    /// `SparkRequestApiCollectPrefillSlotPrefetchSources`.
    fn collect_prefill_slot_prefetch_sources(
        &mut self,
        slot_index: u32,
        source_blocks: &mut Vec<PrefetchSourceBlock>,
    ) -> Result<(), RequestApiError> {
        let prompt = self.slots[slot_index as usize].prompt_token_ids.clone();
        let scheduler = self.scheduler.get_mut();
        let prefix_cache = scheduler.prefix_cache_mut().ok_or(RequestApiError::InvalidArgument)?;
        let probe = prefix_cache
            .probe_reusable_prefix_prefetch_sources(&prompt, MAX_PREFETCH_SOURCE_BLOCK_COUNT)?;
        for source_block in &probe.source_blocks {
            if source_blocks.len() >= MAX_PREFETCH_SOURCE_BLOCK_COUNT as usize {
                break;
            }
            collect_prefetch_source_block(source_blocks, source_block);
        }
        Ok(())
    }

    /// `SparkRequestApiCollectDecodeSlotPrefetchSources`.
    fn collect_decode_slot_prefetch_sources(
        &mut self,
        slot_index: u32,
        source_blocks: &mut Vec<PrefetchSourceBlock>,
    ) -> Result<(), RequestApiError> {
        let required_token_count = self.ensure_pending_decode_slot_kv_capacity(slot_index)?;
        let sequence_id = self.slots[slot_index as usize].sequence_id;
        let scheduler = self.scheduler.get_mut();
        let prefix_cache = scheduler.prefix_cache_mut().ok_or(RequestApiError::InvalidArgument)?;
        let slot_source_blocks =
            prefix_cache.build_sequence_prefetch_sources(sequence_id, required_token_count)?;
        for source_block in &slot_source_blocks {
            if source_blocks.len() >= MAX_PREFETCH_SOURCE_BLOCK_COUNT as usize {
                break;
            }
            collect_prefetch_source_block(source_blocks, source_block);
        }
        Ok(())
    }

    /// `SparkRequestApiFindBestPrefetchLookaheadSlot`.
    fn find_best_prefetch_lookahead_slot(&self, selected_handles: &[u64]) -> Option<u32> {
        let mut best_slot: Option<u32> = None;
        for slot_index in 0..self.request_capacity {
            let slot = &self.slots[slot_index as usize];
            if !slot.is_schedulable_prefill()
                && !slot.is_schedulable_decode()
                && !slot.is_schedulable_speculative_verify()
            {
                continue;
            }
            if selected_handles.contains(&slot.handle) {
                continue;
            }
            let is_better = match best_slot {
                None => true,
                Some(best_index) => {
                    slot.has_higher_scheduling_priority_than(Some(&self.slots[best_index as usize]))
                }
            };
            if is_better {
                best_slot = Some(slot_index);
            }
        }
        best_slot
    }

    /// `SparkRequestApiRefreshLookaheadPrefixProtections`.
    pub(crate) fn refresh_lookahead_prefix_protections(&mut self) -> Result<(), RequestApiError> {
        if !self.queue_aware_prefix_cache_eviction_is_enabled() {
            return Ok(());
        }
        if self.scheduler.get_mut().prefix_cache().is_none() {
            return Err(RequestApiError::InvalidArgument);
        }
        {
            let scheduler = self.scheduler.get_mut();
            let prefix_cache =
                scheduler.prefix_cache_mut().ok_or(RequestApiError::InvalidArgument)?;
            prefix_cache.reset_lookahead_protection()?;
        }
        let mut selected_handles: Vec<u64> = Vec::new();
        let mut total_protected_block_count = 0u64;
        for _ in 0..self.prefetch_lookahead_request_count {
            if selected_handles.len() >= MAX_PREFETCH_SOURCE_BLOCK_COUNT as usize {
                break;
            }
            let Some(slot_index) = self.find_best_prefetch_lookahead_slot(&selected_handles) else {
                break;
            };
            let (handle, prompt, priority) = {
                let slot = &self.slots[slot_index as usize];
                (slot.handle, slot.prompt_token_ids.clone(), slot.priority)
            };
            selected_handles.push(handle);
            if prompt.is_empty() {
                continue;
            }
            let scheduler = self.scheduler.get_mut();
            let prefix_cache =
                scheduler.prefix_cache_mut().ok_or(RequestApiError::InvalidArgument)?;
            let protection = prefix_cache.protect_prompt_lookahead(&prompt, priority)?;
            total_protected_block_count += u64::from(protection.protected_block_count);
        }
        self.lookahead_protection_sweep_count += 1;
        self.lookahead_protected_block_count += total_protected_block_count;
        Ok(())
    }

    /// `SparkRequestApiJitResidencyPolicyIsEnabled`.
    fn jit_residency_policy_is_enabled(&self) -> bool {
        self.jit_prefetch_is_enabled()
            && self.max_resident_kv_block_count != 0
            && self.scheduler.borrow().prefix_cache().is_some()
    }

    /// `SparkRequestApiApplyJitKvResidencyPolicy`.
    pub(crate) fn apply_jit_kv_residency_policy(
        &mut self,
        protected_prefetch_plan: Option<&PrefetchPlan>,
        additional_protected_physical_block_indices: &[u32],
    ) -> Result<(), RequestApiError> {
        if !self.jit_residency_policy_is_enabled() {
            return Ok(());
        }
        let mut protected: Vec<u32> = Vec::new();
        if let Some(plan) = protected_prefetch_plan {
            for block in &plan.blocks {
                if protected.len() >= MAX_PREFETCH_SOURCE_BLOCK_COUNT as usize {
                    break;
                }
                collect_physical_block_index(&mut protected, block.physical_block_index);
            }
        }
        for &block in additional_protected_physical_block_indices {
            collect_physical_block_index(&mut protected, block);
        }
        for pending in &self.pending_prefetches {
            if !pending.active {
                continue;
            }
            for block in &pending.prefetch_plan.blocks {
                if protected.len() >= MAX_PREFETCH_SOURCE_BLOCK_COUNT as usize {
                    break;
                }
                collect_physical_block_index(&mut protected, block.physical_block_index);
            }
        }
        self.collect_running_protected_blocks(&mut protected)?;

        let max_resident = self.max_resident_kv_block_count;
        let scheduler = self.scheduler.get_mut();
        let prefix_cache = scheduler.prefix_cache_mut().ok_or(RequestApiError::InvalidArgument)?;
        let evicted_block_count =
            prefix_cache.trim_resident_blocks_by_reuse_score(max_resident, &protected)?;
        self.jit_residency_eviction_count += u64::from(evicted_block_count);
        self.jit_residency_protected_block_count +=
            protected.len() as u64 + prefix_cache.lookahead_protected_block_count;
        Ok(())
    }

    /// `SparkRequestApiPrefetchBlockIsResident`.
    pub(crate) fn prefetch_block_is_resident(&self, prefetch_block: &PrefetchBlock) -> bool {
        if !self.cross_sequence_prefix_reuse_is_enabled() {
            return false;
        }
        let scheduler = self.scheduler.borrow();
        let Some(prefix_cache) = scheduler.prefix_cache() else {
            return false;
        };
        let arena = prefix_cache.arena();
        if prefetch_block.physical_block_index >= arena.physical_block_count() {
            return false;
        }
        arena.is_allocated(prefetch_block.physical_block_index)
            && arena.is_resident(prefetch_block.physical_block_index)
            && arena.generation(prefetch_block.physical_block_index) == prefetch_block.generation
    }

    /// `SparkRequestApiPrefetchPlanIsResident`.
    pub(crate) fn prefetch_plan_is_resident(&self, prefetch_plan: &PrefetchPlan) -> bool {
        prefetch_plan.blocks.iter().all(|block| self.prefetch_block_is_resident(block))
    }

    /// `SparkRequestApiPendingPrefetchesCoverPlan`.
    fn pending_prefetches_cover_plan(&self, prefetch_plan: &PrefetchPlan) -> bool {
        prefetch_plan.blocks.iter().all(|block| {
            if self.prefetch_block_is_resident(block) {
                return true;
            }
            self.pending_prefetches.iter().any(|pending| {
                pending.active
                    && pending.prefetch_plan.blocks.iter().any(|pending_block| {
                        pending_block.physical_block_index == block.physical_block_index
                            && pending_block.generation == block.generation
                    })
            })
        })
    }

    /// `SparkRequestApiClearPendingPrefetch`.
    fn clear_pending_prefetch(&mut self, pending_index: usize) {
        if pending_index < self.pending_prefetches.len() {
            self.pending_prefetches[pending_index] = PendingPrefetch::default();
        }
    }

    /// `SparkRequestApiPollOnePendingPrefetch`.
    fn poll_one_pending_prefetch(&mut self, pending_index: usize) -> Result<(), RequestApiError> {
        if pending_index >= self.pending_prefetches.len()
            || !self.pending_prefetches[pending_index].active
        {
            return Err(RequestApiError::InvalidArgument);
        }
        let prefetch_plan = self.pending_prefetches[pending_index].prefetch_plan.clone();
        let prefetch_id = self.pending_prefetches[pending_index].prefetch_id;
        let status = {
            let backend =
                self.kv_prefetch_backend.as_mut().ok_or(RequestApiError::InvalidArgument)?;
            backend.poll_prefetch(prefetch_id, &prefetch_plan)
        };
        self.pending_prefetches[pending_index].poll_count += 1;
        self.async_jit_prefetch_poll_count += 1;
        match status {
            Err(RequestApiError::Busy) => return Err(RequestApiError::Busy),
            Err(error) => {
                self.clear_pending_prefetch(pending_index);
                return Err(error);
            }
            Ok(()) => {}
        }
        let result = (|| -> Result<(), RequestApiError> {
            {
                let resident_block_capacity = self.arena_resident_block_capacity;
                let scheduler = self.scheduler.get_mut();
                let prefix_cache =
                    scheduler.prefix_cache_mut().ok_or(RequestApiError::InvalidArgument)?;
                mark_prefetch_plan_resident(
                    prefix_cache.arena_mut(),
                    resident_block_capacity,
                    &prefetch_plan,
                )?;
            }
            self.apply_jit_kv_residency_policy(Some(&prefetch_plan), &[])
        })();
        if let Err(error) = result {
            self.clear_pending_prefetch(pending_index);
            return Err(error);
        }
        self.jit_prefetch_dispatch_count += 1;
        self.jit_prefetch_block_count += u64::from(prefetch_plan.prefetch_block_count());
        self.async_jit_prefetch_completion_count += 1;
        self.clear_pending_prefetch(pending_index);
        Ok(())
    }

    /// `SparkRequestApiPollPendingJitKvPrefetches`.
    pub(crate) fn poll_pending_jit_kv_prefetches(&mut self) -> Result<(), RequestApiError> {
        for pending_index in 0..self.pending_prefetches.len() {
            if !self.pending_prefetches[pending_index].active {
                continue;
            }
            match self.poll_one_pending_prefetch(pending_index) {
                Ok(()) | Err(RequestApiError::Busy) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    /// `SparkRequestApiStartAsyncJitKvPrefetch`.
    fn start_async_jit_kv_prefetch(
        &mut self,
        prefetch_plan: &PrefetchPlan,
    ) -> Result<(), RequestApiError> {
        let Some(pending_index) =
            self.pending_prefetches.iter().position(|pending| !pending.active)
        else {
            return Err(RequestApiError::Busy);
        };
        let prefetch_id = self.next_prefetch_id;
        self.next_prefetch_id += 1;
        if self.next_prefetch_id == 0 {
            self.next_prefetch_id = 1;
        }
        {
            let backend =
                self.kv_prefetch_backend.as_mut().ok_or(RequestApiError::InvalidArgument)?;
            backend.start_prefetch(prefetch_id, prefetch_plan)?;
        }
        self.pending_prefetches[pending_index] = PendingPrefetch {
            active: true,
            poll_count: 0,
            prefetch_id,
            prefetch_plan: prefetch_plan.clone(),
        };
        self.async_jit_prefetch_start_count += 1;
        self.poll_one_pending_prefetch(pending_index)
    }

    /// `SparkRequestApiDispatchJitKvPrefetchWithProtectedBlocks`.
    pub(crate) fn dispatch_jit_kv_prefetch_with_protected_blocks(
        &mut self,
        prefetch_plan: &PrefetchPlan,
        additional_protected_physical_block_indices: &[u32],
    ) -> Result<(), RequestApiError> {
        self.validate()?;
        if !self.jit_prefetch_is_enabled() {
            return Ok(());
        }
        if self.async_jit_prefetch_is_enabled() {
            if prefetch_plan.prefetch_block_count() == 0
                || self.prefetch_plan_is_resident(prefetch_plan)
            {
                return self.apply_jit_kv_residency_policy(
                    Some(prefetch_plan),
                    additional_protected_physical_block_indices,
                );
            }
            self.poll_pending_jit_kv_prefetches()?;
            if self.prefetch_plan_is_resident(prefetch_plan) {
                return self.apply_jit_kv_residency_policy(
                    Some(prefetch_plan),
                    additional_protected_physical_block_indices,
                );
            }
            if self.pending_prefetches_cover_plan(prefetch_plan) {
                return Err(RequestApiError::Busy);
            }
            return self.start_async_jit_kv_prefetch(prefetch_plan);
        }

        if prefetch_plan.prefetch_block_count() == 0 {
            return self.apply_jit_kv_residency_policy(
                Some(prefetch_plan),
                additional_protected_physical_block_indices,
            );
        }
        {
            let backend =
                self.kv_prefetch_backend.as_mut().ok_or(RequestApiError::InvalidArgument)?;
            backend.prefetch(prefetch_plan)?;
        }
        {
            let resident_block_capacity = self.arena_resident_block_capacity;
            let scheduler = self.scheduler.get_mut();
            let prefix_cache =
                scheduler.prefix_cache_mut().ok_or(RequestApiError::InvalidArgument)?;
            mark_prefetch_plan_resident_with_protected_blocks(
                prefix_cache.arena_mut(),
                resident_block_capacity,
                prefetch_plan,
                additional_protected_physical_block_indices,
            )?;
        }
        self.apply_jit_kv_residency_policy(
            Some(prefetch_plan),
            additional_protected_physical_block_indices,
        )?;
        self.jit_prefetch_dispatch_count += 1;
        self.jit_prefetch_block_count += u64::from(prefetch_plan.prefetch_block_count());
        Ok(())
    }

    /// `SparkRequestApiDispatchJitKvPrefetch`.
    pub fn dispatch_jit_kv_prefetch(
        &mut self,
        prefetch_plan: &PrefetchPlan,
    ) -> Result<(), RequestApiError> {
        self.dispatch_jit_kv_prefetch_with_protected_blocks(prefetch_plan, &[])
    }

    /// `SparkRequestApiBuildSlotArrayJitKvPrefetchPlan`.
    fn build_slot_array_jit_kv_prefetch_plan(
        &mut self,
        slots: &[u32],
    ) -> Result<PrefetchPlan, RequestApiError> {
        if slots.is_empty() {
            return Err(RequestApiError::InvalidArgument);
        }
        if !self.jit_prefetch_is_enabled() {
            return Ok(PrefetchPlan::default());
        }
        let mut source_blocks: Vec<PrefetchSourceBlock> = Vec::new();
        for &slot_index in slots {
            if source_blocks.len() >= MAX_PREFETCH_SOURCE_BLOCK_COUNT as usize {
                break;
            }
            let slot = &self.slots[slot_index as usize];
            if slot.is_schedulable_prefill() {
                self.collect_prefill_slot_prefetch_sources(slot_index, &mut source_blocks)?;
            } else if slot.is_schedulable_decode() || slot.is_schedulable_speculative_verify() {
                self.collect_decode_slot_prefetch_sources(slot_index, &mut source_blocks)?;
            } else {
                return Err(RequestApiError::InvalidArgument);
            }
        }
        let scheduler = self.scheduler.borrow();
        let prefix_cache = scheduler.prefix_cache().ok_or(RequestApiError::InvalidArgument)?;
        let mut flags = PREFETCH_BLOCK_FLAG_KEY;
        if arena_has_value_payload(prefix_cache.arena()) {
            flags |= PREFETCH_BLOCK_FLAG_VALUE;
        }
        build_prefetch_plan_from_source_blocks(
            prefix_cache.arena(),
            &source_blocks,
            flags,
            self.prefetch_lane_count,
        )
    }

    /// `SparkRequestApiRunSlotArrayCriticalJitKvPrefetch`.
    pub(crate) fn run_slot_array_critical_jit_kv_prefetch(
        &mut self,
        slots: &[u32],
        dispatch: &mut Dispatch,
    ) -> Result<(), RequestApiError> {
        if !self.jit_prefetch_is_enabled() {
            return Ok(());
        }
        let mut critical_physical_block_indices: Vec<u32> = Vec::new();
        for &slot_index in slots {
            match self
                .collect_protected_slot_blocks(slot_index, &mut critical_physical_block_indices)
            {
                Ok(()) | Err(RequestApiError::NotFound) => {}
                Err(error) => return Err(error),
            }
        }
        dispatch.kv_prefetch_plan = self.build_slot_array_jit_kv_prefetch_plan(slots)?;
        let prefetch_plan = dispatch.kv_prefetch_plan.clone();
        let status = self.dispatch_jit_kv_prefetch_with_protected_blocks(
            &prefetch_plan,
            &critical_physical_block_indices,
        );
        match status {
            Err(RequestApiError::Busy) => {
                dispatch.flags |= DISPATCH_FLAG_JIT_PREFETCH_PENDING;
                Err(RequestApiError::Busy)
            }
            Err(error) => Err(error),
            Ok(()) => {
                if dispatch.kv_prefetch_plan.prefetch_block_count() != 0 {
                    dispatch.flags |= DISPATCH_FLAG_JIT_PREFETCHED_KV;
                }
                Ok(())
            }
        }
    }

    /// `SparkRequestApiPrefetchPlanFitsResidentLimit`.
    fn prefetch_plan_fits_resident_limit(
        &self,
        prefetch_plan: Option<&PrefetchPlan>,
        protected_physical_block_indices: &[u32],
    ) -> bool {
        if self.max_resident_kv_block_count == 0 {
            return true;
        }
        let mut combined: Vec<u32> = Vec::new();
        for &block in protected_physical_block_indices {
            collect_physical_block_index(&mut combined, block);
        }
        if let Some(plan) = prefetch_plan {
            for block in &plan.blocks {
                collect_physical_block_index(&mut combined, block.physical_block_index);
            }
        }
        combined.len() as u32 <= self.max_resident_kv_block_count
    }

    /// `SparkRequestApiBuildJitKvPrefetchPlan`.
    pub fn build_jit_kv_prefetch_plan(&mut self) -> Result<PrefetchPlan, RequestApiError> {
        self.validate()?;
        if !self.jit_prefetch_is_enabled() {
            let scheduler = self.scheduler.borrow();
            let prefix_cache = scheduler.prefix_cache().ok_or(RequestApiError::InvalidArgument)?;
            return build_prefetch_plan(prefix_cache.arena(), &[], self.prefetch_lane_count);
        }
        let mut selected_handles: Vec<u64> = Vec::new();
        let mut source_blocks: Vec<PrefetchSourceBlock> = Vec::new();
        for _ in 0..self.prefetch_lookahead_request_count {
            if selected_handles.len() >= MAX_PREFETCH_SOURCE_BLOCK_COUNT as usize {
                break;
            }
            let Some(slot_index) = self.find_best_prefetch_lookahead_slot(&selected_handles) else {
                break;
            };
            selected_handles.push(self.slots[slot_index as usize].handle);
            if self.slots[slot_index as usize].is_schedulable_prefill() {
                self.collect_prefill_slot_prefetch_sources(slot_index, &mut source_blocks)?;
            } else {
                self.collect_decode_slot_prefetch_sources(slot_index, &mut source_blocks)?;
            }
            if source_blocks.len() >= MAX_PREFETCH_SOURCE_BLOCK_COUNT as usize {
                break;
            }
        }
        let scheduler = self.scheduler.borrow();
        let prefix_cache = scheduler.prefix_cache().ok_or(RequestApiError::InvalidArgument)?;
        let mut flags = PREFETCH_BLOCK_FLAG_KEY;
        if arena_has_value_payload(prefix_cache.arena()) {
            flags |= PREFETCH_BLOCK_FLAG_VALUE;
        }
        build_prefetch_plan_from_source_blocks(
            prefix_cache.arena(),
            &source_blocks,
            flags,
            self.prefetch_lane_count,
        )
    }

    /// `SparkRequestApiRunOpportunisticJitKvPrefetch`.
    pub(crate) fn run_opportunistic_jit_kv_prefetch(
        &mut self,
        protected_slot: Option<u32>,
    ) -> Result<(), RequestApiError> {
        if !self.jit_prefetch_is_enabled() {
            return Ok(());
        }
        let mut protected_physical_block_indices: Vec<u32> = Vec::new();
        if let Some(slot_index) = protected_slot {
            match self
                .collect_protected_slot_blocks(slot_index, &mut protected_physical_block_indices)
            {
                Ok(()) | Err(RequestApiError::NotFound) => {}
                Err(error) => return Err(error),
            }
        }
        let prefetch_plan = self.build_jit_kv_prefetch_plan()?;
        if !self.prefetch_plan_fits_resident_limit(
            Some(&prefetch_plan),
            &protected_physical_block_indices,
        ) {
            return self.apply_jit_kv_residency_policy(None, &protected_physical_block_indices);
        }
        match self.dispatch_jit_kv_prefetch_with_protected_blocks(
            &prefetch_plan,
            &protected_physical_block_indices,
        ) {
            Err(RequestApiError::Busy) | Err(RequestApiError::CapacityExceeded) => {
                self.apply_jit_kv_residency_policy(None, &protected_physical_block_indices)
            }
            status => status,
        }
    }
}
