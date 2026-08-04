//! Content-addressed prefix cache — port of `cache/prefix_cache.c`
//! (API declared in `include/sparkpipe/spark_prefix_cache.h`).
//!
//! Block-sized spans of token ids are hashed into a parent-chained content
//! hash (FNV-1a mixing, identical constants to the C source) so a second
//! sequence sharing a prompt prefix hits the same entries. Physical KV blocks
//! come from an owned [`KvArena`]; the C version also supports an arena-less
//! mode, but the Rust port always owns an arena (callers drive residency
//! through [`PrefixCache::arena_mut`]).
//!
//! Storage mirrors the C layout: `Vec` backing with intrusive free lists
//! linked through the `reserved` word using [`NO_ENTRY`] sentinels, and all
//! cross-references are `u32` indices. All statistics counters of the C
//! `SparkPrefixCache` struct are preserved as public fields, including the
//! quirk that [`PrefixCache::reset`] does not clear the `reuse_scored_*`
//! counters.

use crate::kv_arena::{KvArena, KvArenaError};

/// Hash of the empty prefix (SPARK_PREFIX_CACHE_EMPTY_PARENT_HASH).
pub const EMPTY_PARENT_HASH: u64 = 1469598103934665603;
/// Sentinel: no physical block (SPARK_PREFIX_CACHE_NO_PHYSICAL_BLOCK).
pub const NO_PHYSICAL_BLOCK: u32 = 0xffff_ffff;
/// Sentinel: no entry/binding slot (SPARK_PREFIX_CACHE_NO_ENTRY).
pub const NO_ENTRY: u32 = 0xffff_ffff;
/// Maximum tokens per cache block (SPARK_PREFIX_CACHE_MAX_BLOCK_TOKENS).
pub const MAX_BLOCK_TOKENS: u32 = 256;

/// Entry flag: slot holds a live entry (SPARK_PREFIX_CACHE_ENTRY_FLAG_VALID).
pub const ENTRY_FLAG_VALID: u32 = 0x0000_0001;
/// Entry flag: committed full block offered for cross-sequence reuse.
pub const ENTRY_FLAG_REUSABLE: u32 = 0x0000_0002;
/// Entry flag: reserved but not yet committed.
pub const ENTRY_FLAG_PENDING: u32 = 0x0000_0004;
/// Entry flag: private to its sequence, never a content-addressed hit.
pub const ENTRY_FLAG_LIVE_ONLY: u32 = 0x0000_0008;

const BINDING_FLAG_VALID: u32 = 0x0000_0001;
const BINDING_FLAG_PENDING: u32 = 0x0000_0002;

const FNV_PRIME: u64 = 1099511628211;
const CONTENT_HASH_SEED: u64 = 7809847782465536322;
const GOLDEN_GAMMA_32: u32 = 0x9e37_79b9;

const REUSE_SCORE_LOOKAHEAD_BASE: u64 = 1_000_000_000_000_000;
const REUSE_SCORE_PRIORITY_WEIGHT: u64 = 1_000_000_000;
const REUSE_SCORE_REQUEST_WEIGHT: u64 = 10_000_000;
const REUSE_SCORE_REFERENCE_WEIGHT: u64 = 1_000_000_000_000;
const REUSE_SCORE_TOKEN_DEPTH_WEIGHT: u64 = 1024;

/// Errors mirroring the `SparkStatus` codes returned by the C entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PrefixCacheError {
    #[error("invalid argument")]
    InvalidArgument,
    #[error("capacity exceeded (entries, bindings, or physical blocks exhausted)")]
    CapacityExceeded,
    #[error("entry or binding not found")]
    NotFound,
    #[error("entry is busy (referenced) or in a conflicting state")]
    Busy,
    #[error("conflicting binding already exists for the target sequence")]
    Duplicate,
    #[error("block growth would corrupt a shared prefix (needs copy-on-write)")]
    ModuleNotValidated,
    #[error("internal error (free-list corruption)")]
    InternalError,
}

impl From<KvArenaError> for PrefixCacheError {
    fn from(error: KvArenaError) -> Self {
        match error {
            KvArenaError::InvalidArgument => PrefixCacheError::InvalidArgument,
            KvArenaError::CapacityExceeded => PrefixCacheError::CapacityExceeded,
            KvArenaError::NotFound => PrefixCacheError::NotFound,
            KvArenaError::Busy => PrefixCacheError::Busy,
            KvArenaError::Internal => PrefixCacheError::InternalError,
        }
    }
}

fn mix_u64(hash_value: u64, value: u64) -> u64 {
    let mut hash_value = hash_value ^ value;
    hash_value = hash_value.wrapping_mul(FNV_PRIME);
    hash_value ^= hash_value >> 32;
    hash_value
}

/// Hash one block of tokens chained onto `parent_hash`
/// (SparkPrefixCacheHashBlock — exact C algorithm and constants).
pub fn hash_block(token_ids: &[u32], parent_hash: u64) -> u64 {
    let mut hash_value = parent_hash ^ FNV_PRIME;
    for (token_index, token_id) in token_ids.iter().enumerate() {
        hash_value =
            mix_u64(hash_value, (*token_id as u64).wrapping_add((token_index as u64) << 32));
    }
    mix_u64(hash_value, token_ids.len() as u64)
}

fn hash_block_content(token_ids: &[u32]) -> u64 {
    let mut hash_value = CONTENT_HASH_SEED;
    for (token_index, token_id) in token_ids.iter().enumerate() {
        hash_value = mix_u64(
            hash_value,
            ((*token_id as u64) << 1) ^ ((token_index as u32).wrapping_add(GOLDEN_GAMMA_32) as u64),
        );
    }
    mix_u64(hash_value, token_ids.len() as u64)
}

/// Result of hashing a whole prompt (SparkPrefixCachePromptHash).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptHash {
    pub token_count: u32,
    pub hashed_token_count: u32,
    pub block_count: u32,
    pub last_block_token_count: u32,
    pub parent_hash: u64,
    pub prompt_hash: u64,
}

/// Hash a prompt block-by-block, chaining each block hash into the next
/// (SparkPrefixCacheHashPromptTokens).
pub fn hash_prompt_tokens(
    block_token_count: u32,
    parent_hash: u64,
    token_ids: &[u32],
) -> Result<PromptHash, PrefixCacheError> {
    if block_token_count == 0 || block_token_count > MAX_BLOCK_TOKENS {
        return Err(PrefixCacheError::InvalidArgument);
    }
    let token_count = token_ids.len() as u32;
    let mut hash_value = parent_hash;
    let mut token_offset = 0usize;
    let mut block_count = 0u32;
    while token_offset < token_ids.len() {
        let current = (token_ids.len() - token_offset).min(block_token_count as usize);
        hash_value = hash_block(&token_ids[token_offset..token_offset + current], hash_value);
        token_offset += current;
        block_count += 1;
    }
    Ok(PromptHash {
        token_count,
        hashed_token_count: token_count,
        block_count,
        last_block_token_count: if token_ids.is_empty() {
            0
        } else {
            (token_count - 1) % block_token_count + 1
        },
        parent_hash,
        prompt_hash: hash_value,
    })
}

/// Cache configuration (SparkPrefixCacheConfiguration, minus the caller-owned
/// arrays and arena pointer — the Rust cache owns its storage and arena).
#[derive(Debug, Clone, Copy)]
pub struct PrefixCacheConfig {
    pub block_token_count: u32,
    pub entry_count: u32,
    pub physical_block_count: u32,
    pub sequence_binding_count: u32,
}

/// Lookup/probe result (SparkPrefixCacheLookup).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lookup {
    pub requested_token_count: u32,
    pub matched_token_count: u32,
    pub matched_block_count: u32,
    pub next_token_index: u32,
    pub physical_block_index: u32,
    pub sequence_id: u64,
    pub last_block_hash: u64,
}

impl Lookup {
    fn new(sequence_id: u64, requested_token_count: u32) -> Self {
        Self {
            requested_token_count,
            matched_token_count: 0,
            matched_block_count: 0,
            next_token_index: 0,
            physical_block_index: NO_PHYSICAL_BLOCK,
            sequence_id,
            last_block_hash: EMPTY_PARENT_HASH,
        }
    }
}

/// Reservation result (SparkPrefixCacheReservation). The C version writes
/// into a caller-provided index array (skipping the capacity check when the
/// array is null); the Rust version always returns an exactly-sized Vec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reservation {
    pub requested_token_count: u32,
    pub reserved_token_count: u32,
    pub reusable_token_count: u32,
    pub physical_block_count: u32,
    pub cached_physical_block_count: u32,
    pub pending_physical_block_count: u32,
    pub sequence_id: u64,
    pub reservation_epoch: u64,
    pub last_block_hash: u64,
    pub physical_block_indices: Vec<u32>,
}

/// Probe result for the reusable-prefix physical block table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalBlockTableProbe {
    pub matched_token_count: u32,
    pub physical_block_indices: Vec<u32>,
}

/// Residency breakdown of a reusable-prefix match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixResidencyProbe {
    pub matched_token_count: u32,
    pub resident_block_count: u32,
    pub nonresident_block_count: u32,
}

/// Residency breakdown of a live sequence's bound blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceResidencyProbe {
    pub physical_block_count: u32,
    pub resident_block_count: u32,
    pub nonresident_block_count: u32,
}

/// Counts returned by lookahead protection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LookaheadProtection {
    pub protected_token_count: u32,
    pub protected_block_count: u32,
}

/// Prefetch source descriptor (SparkKvCachePrefetchSourceBlock). The C struct
/// also carries a `flags` word fixed to KEY|VALUE; it is constant there and
/// omitted here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefetchSourceBlock {
    pub physical_block_index: u32,
    pub token_capacity: u32,
    pub first_token_index: u32,
    pub token_count: u32,
    pub generation: u64,
    pub parent_hash: u64,
    pub block_hash: u64,
    pub content_hash: u64,
}

/// Probe result for reusable-prefix prefetch sources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefetchSourceProbe {
    pub matched_token_count: u32,
    pub source_blocks: Vec<PrefetchSourceBlock>,
}

/// Read-only snapshot of one cache entry (for tests and diagnostics; the C
/// struct fields are directly readable by callers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntryView {
    pub flags: u32,
    pub token_count: u32,
    pub first_token_index: u32,
    pub physical_block_index: u32,
    pub reference_count: u32,
    pub lookahead_priority: u32,
    pub lookahead_request_count: u32,
    pub parent_hash: u64,
    pub block_hash: u64,
    pub content_hash: u64,
    pub last_used_tick: u64,
    pub reservation_epoch: u64,
    pub committed_epoch: u64,
    pub lookahead_protection_epoch: u64,
}

#[derive(Debug, Clone)]
struct Entry {
    flags: u32,
    token_count: u32,
    first_token_index: u32,
    physical_block_index: u32,
    reference_count: u32,
    /// Intrusive free-list link (next free entry index) while unallocated.
    reserved: u32,
    lookahead_priority: u32,
    lookahead_request_count: u32,
    parent_hash: u64,
    block_hash: u64,
    content_hash: u64,
    last_used_tick: u64,
    reservation_epoch: u64,
    committed_epoch: u64,
    lookahead_protection_epoch: u64,
}

impl Entry {
    fn unallocated() -> Self {
        Self {
            flags: 0,
            token_count: 0,
            first_token_index: 0,
            physical_block_index: NO_PHYSICAL_BLOCK,
            reference_count: 0,
            reserved: 0,
            lookahead_priority: 0,
            lookahead_request_count: 0,
            parent_hash: 0,
            block_hash: 0,
            content_hash: 0,
            last_used_tick: 0,
            reservation_epoch: 0,
            committed_epoch: 0,
            lookahead_protection_epoch: 0,
        }
    }

    fn is_evictable(&self) -> bool {
        self.flags & ENTRY_FLAG_VALID != 0
            && self.flags & ENTRY_FLAG_PENDING == 0
            && self.reference_count == 0
    }

    fn view(&self) -> EntryView {
        EntryView {
            flags: self.flags,
            token_count: self.token_count,
            first_token_index: self.first_token_index,
            physical_block_index: self.physical_block_index,
            reference_count: self.reference_count,
            lookahead_priority: self.lookahead_priority,
            lookahead_request_count: self.lookahead_request_count,
            parent_hash: self.parent_hash,
            block_hash: self.block_hash,
            content_hash: self.content_hash,
            last_used_tick: self.last_used_tick,
            reservation_epoch: self.reservation_epoch,
            committed_epoch: self.committed_epoch,
            lookahead_protection_epoch: self.lookahead_protection_epoch,
        }
    }
}

#[derive(Debug, Clone)]
struct Binding {
    flags: u32,
    entry_index: u32,
    first_token_index: u32,
    token_count: u32,
    physical_block_index: u32,
    /// Intrusive free-list link (next free binding index) while unallocated.
    reserved: u32,
    sequence_id: u64,
    parent_hash: u64,
    block_hash: u64,
    acquire_epoch: u64,
}

impl Binding {
    fn unallocated() -> Self {
        Self {
            flags: 0,
            entry_index: NO_ENTRY,
            first_token_index: 0,
            token_count: 0,
            physical_block_index: NO_PHYSICAL_BLOCK,
            reserved: 0,
            sequence_id: 0,
            parent_hash: 0,
            block_hash: 0,
            acquire_epoch: 0,
        }
    }
}

/// A "better victim" is evicted first: lower lookahead priority wins, ties
/// broken by older `last_used_tick`.
fn protected_victim_is_better(candidate: &Entry, current: &Entry) -> bool {
    if candidate.lookahead_priority != current.lookahead_priority {
        return candidate.lookahead_priority < current.lookahead_priority;
    }
    candidate.last_used_tick < current.last_used_tick
}

/// State of one reusable-prefix chain walk (SparkGlm52PrefixCacheWalk).
#[derive(Debug, Default)]
struct ChainWalk {
    parent_hash: u64,
    block_hash: u64,
    token_offset: u32,
    reusable_token_count: u32,
    matched_block_count: u32,
}

/// Eviction candidate for reuse-scored resident trimming.
#[derive(Debug, Clone, Copy)]
struct ResidentEvictionCandidate {
    physical_block_index: u32,
    has_prefix_entry: bool,
    has_lookahead_protection: bool,
    keep_score: u64,
    last_used_epoch: u64,
}

impl ResidentEvictionCandidate {
    fn none() -> Self {
        Self {
            physical_block_index: NO_PHYSICAL_BLOCK,
            has_prefix_entry: false,
            has_lookahead_protection: false,
            keep_score: u64::MAX,
            last_used_epoch: u64::MAX,
        }
    }

    fn is_better(&self, current: &ResidentEvictionCandidate) -> bool {
        if current.physical_block_index == NO_PHYSICAL_BLOCK {
            return true;
        }
        if self.keep_score != current.keep_score {
            return self.keep_score < current.keep_score;
        }
        if self.has_prefix_entry != current.has_prefix_entry {
            return !self.has_prefix_entry;
        }
        if self.last_used_epoch != current.last_used_epoch {
            return self.last_used_epoch < current.last_used_epoch;
        }
        self.physical_block_index < current.physical_block_index
    }
}

/// Content-addressed prefix cache owning its entries, sequence bindings, and
/// KV block arena.
#[derive(Debug)]
pub struct PrefixCache {
    block_token_count: u32,
    physical_block_count: u32,
    entries: Vec<Entry>,
    bindings: Vec<Binding>,
    arena: KvArena,
    free_entry_head: u32,
    free_binding_head: u32,
    // Statistics/tick fields below are public to mirror the C struct.
    pub tick: u64,
    pub operation_epoch: u64,
    pub lookup_count: u64,
    pub hit_count: u64,
    pub miss_count: u64,
    pub inserted_block_count: u64,
    pub evicted_block_count: u64,
    pub acquired_block_count: u64,
    pub released_block_count: u64,
    pub reserved_block_count: u64,
    pub committed_reserved_block_count: u64,
    pub cancelled_reserved_block_count: u64,
    pub live_only_block_count: u64,
    pub lookahead_protection_epoch: u64,
    pub lookahead_protected_block_count: u64,
    pub lookahead_protected_eviction_skip_count: u64,
    pub reuse_scored_resident_eviction_count: u64,
    pub reuse_scored_lookahead_eviction_count: u64,
    pub reuse_scored_untracked_eviction_count: u64,
    pub reuse_scored_capacity_stall_count: u64,
}

impl PrefixCache {
    /// Create a cache (SparkPrefixCacheInitialize), including the C
    /// cross-check of arena geometry against the cache configuration.
    pub fn new(config: &PrefixCacheConfig, arena: KvArena) -> Result<Self, PrefixCacheError> {
        if config.block_token_count == 0
            || config.block_token_count > MAX_BLOCK_TOKENS
            || config.entry_count == 0
            || config.physical_block_count == 0
            || config.sequence_binding_count == 0
        {
            return Err(PrefixCacheError::InvalidArgument);
        }
        if arena.physical_block_count() < config.physical_block_count
            || arena.block_token_count() != config.block_token_count
        {
            return Err(PrefixCacheError::InvalidArgument);
        }
        let mut entries = vec![Entry::unallocated(); config.entry_count as usize];
        let entry_len = entries.len();
        for (index, entry) in entries.iter_mut().enumerate() {
            entry.reserved = if index + 1 < entry_len { index as u32 + 1 } else { NO_ENTRY };
        }
        let mut bindings = vec![Binding::unallocated(); config.sequence_binding_count as usize];
        let binding_len = bindings.len();
        for (index, binding) in bindings.iter_mut().enumerate() {
            binding.reserved = if index + 1 < binding_len { index as u32 + 1 } else { NO_ENTRY };
        }
        Ok(Self {
            block_token_count: config.block_token_count,
            physical_block_count: config.physical_block_count,
            entries,
            bindings,
            arena,
            free_entry_head: 0,
            free_binding_head: 0,
            tick: 0,
            operation_epoch: 0,
            lookup_count: 0,
            hit_count: 0,
            miss_count: 0,
            inserted_block_count: 0,
            evicted_block_count: 0,
            acquired_block_count: 0,
            released_block_count: 0,
            reserved_block_count: 0,
            committed_reserved_block_count: 0,
            cancelled_reserved_block_count: 0,
            live_only_block_count: 0,
            lookahead_protection_epoch: 0,
            lookahead_protected_block_count: 0,
            lookahead_protected_eviction_skip_count: 0,
            reuse_scored_resident_eviction_count: 0,
            reuse_scored_lookahead_eviction_count: 0,
            reuse_scored_untracked_eviction_count: 0,
            reuse_scored_capacity_stall_count: 0,
        })
    }

    /// Read-only access to the owned KV arena.
    pub fn arena(&self) -> &KvArena {
        &self.arena
    }

    /// Mutable access to the owned KV arena (callers drive residency through
    /// the same arena, as in the C design).
    pub fn arena_mut(&mut self) -> &mut KvArena {
        &mut self.arena
    }

    /// Tokens per cache block.
    pub fn block_token_count(&self) -> u32 {
        self.block_token_count
    }

    /// Number of entry slots.
    pub fn entry_count(&self) -> u32 {
        self.entries.len() as u32
    }

    /// Read-only snapshot of an entry slot (None when out of range).
    pub fn entry_view(&self, entry_index: u32) -> Option<EntryView> {
        self.entries.get(entry_index as usize).map(Entry::view)
    }

    fn maximum_reusable_token_count(&self, token_count: u32) -> u32 {
        if token_count <= 1 {
            return 0;
        }
        (token_count - 1) - ((token_count - 1) % self.block_token_count)
    }

    fn full_block_token_count(&self, token_count: u32) -> u32 {
        token_count - (token_count % self.block_token_count)
    }

    fn ceil_block_count(&self, token_count: u32) -> u32 {
        token_count.div_ceil(self.block_token_count)
    }

    fn entry_is_reusable(&self, entry_index: u32) -> bool {
        let entry = &self.entries[entry_index as usize];
        entry.flags & ENTRY_FLAG_VALID != 0
            && entry.flags & ENTRY_FLAG_REUSABLE != 0
            && entry.flags & ENTRY_FLAG_PENDING == 0
            && entry.token_count == self.block_token_count
    }

    fn entry_has_current_lookahead_protection(&self, entry_index: u32) -> bool {
        self.lookahead_protection_epoch != 0
            && self.entries[entry_index as usize].lookahead_protection_epoch
                == self.lookahead_protection_epoch
    }

    fn find_entry(
        &self,
        parent_hash: u64,
        block_hash: u64,
        content_hash: u64,
        first_token_index: u32,
        token_count: u32,
        reusable_only: bool,
    ) -> Option<u32> {
        for (index, entry) in self.entries.iter().enumerate() {
            if entry.flags & ENTRY_FLAG_VALID != 0
                && entry.parent_hash == parent_hash
                && entry.block_hash == block_hash
                && entry.content_hash == content_hash
                && entry.first_token_index == first_token_index
                && entry.token_count == token_count
                && (!reusable_only || self.entry_is_reusable(index as u32))
            {
                return Some(index as u32);
            }
        }
        None
    }

    fn find_binding(&self, sequence_id: u64, entry_index: u32) -> Option<u32> {
        for (index, binding) in self.bindings.iter().enumerate() {
            if binding.flags & BINDING_FLAG_VALID != 0
                && binding.sequence_id == sequence_id
                && binding.entry_index == entry_index
            {
                return Some(index as u32);
            }
        }
        None
    }

    fn find_binding_at_token_offset(
        &self,
        sequence_id: u64,
        first_token_index: u32,
    ) -> Option<u32> {
        for (index, binding) in self.bindings.iter().enumerate() {
            if binding.flags & BINDING_FLAG_VALID != 0
                && binding.sequence_id == sequence_id
                && binding.first_token_index == first_token_index
            {
                return Some(index as u32);
            }
        }
        None
    }

    fn find_free_binding(&self) -> Option<u32> {
        if self.free_binding_head == NO_ENTRY {
            return None;
        }
        let head = self.free_binding_head;
        if head as usize >= self.bindings.len()
            || self.bindings[head as usize].flags & BINDING_FLAG_VALID != 0
        {
            return None;
        }
        Some(head)
    }

    fn select_victim(&mut self) -> Option<u32> {
        if self.free_entry_head != NO_ENTRY {
            let head = self.free_entry_head;
            if head as usize >= self.entries.len()
                || self.entries[head as usize].flags & ENTRY_FLAG_VALID != 0
            {
                return None;
            }
            return Some(head);
        }
        let mut unprotected_victim: Option<u32> = None;
        let mut protected_victim: Option<u32> = None;
        for index in 0..self.entries.len() as u32 {
            let entry = &self.entries[index as usize];
            if entry.flags & ENTRY_FLAG_VALID == 0 {
                return Some(index);
            }
            if !entry.is_evictable() {
                continue;
            }
            if self.entry_has_current_lookahead_protection(index) {
                let better = match protected_victim {
                    None => true,
                    Some(current) => protected_victim_is_better(
                        &self.entries[index as usize],
                        &self.entries[current as usize],
                    ),
                };
                if better {
                    protected_victim = Some(index);
                }
                continue;
            }
            let better = match unprotected_victim {
                None => true,
                Some(current) => {
                    self.entries[index as usize].last_used_tick
                        < self.entries[current as usize].last_used_tick
                }
            };
            if better {
                unprotected_victim = Some(index);
            }
        }
        if unprotected_victim.is_some() {
            return unprotected_victim;
        }
        if protected_victim.is_some() {
            self.lookahead_protected_eviction_skip_count += 1;
        }
        protected_victim
    }

    fn invalidate_entry(&mut self, entry_index: u32) -> Result<(), PrefixCacheError> {
        if entry_index as usize >= self.entries.len() {
            return Err(PrefixCacheError::InvalidArgument);
        }
        let entry = &self.entries[entry_index as usize];
        if entry.flags & ENTRY_FLAG_VALID == 0 {
            return Ok(());
        }
        if entry.reference_count != 0 {
            return Err(PrefixCacheError::Busy);
        }
        let physical_block_index = entry.physical_block_index;
        if physical_block_index != NO_PHYSICAL_BLOCK {
            self.arena.free_block(physical_block_index)?;
        }
        let entry = &mut self.entries[entry_index as usize];
        *entry = Entry::unallocated();
        entry.reserved = self.free_entry_head;
        self.free_entry_head = entry_index;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn install_entry(
        &mut self,
        entry_index: u32,
        entry_flags: u32,
        parent_hash: u64,
        block_hash: u64,
        content_hash: u64,
        first_token_index: u32,
        token_count: u32,
        operation_epoch: u64,
    ) -> Result<(), PrefixCacheError> {
        if entry_index as usize >= self.entries.len() {
            return Err(PrefixCacheError::InvalidArgument);
        }
        let entry_was_valid = self.entries[entry_index as usize].flags & ENTRY_FLAG_VALID != 0;
        let physical_block_index;
        if entry_was_valid {
            if self.entries[entry_index as usize].reference_count != 0 {
                return Err(PrefixCacheError::Busy);
            }
            physical_block_index = self.entries[entry_index as usize].physical_block_index;
            self.arena.recycle_block(physical_block_index)?;
            self.evicted_block_count += 1;
        } else {
            if self.free_entry_head != entry_index {
                return Err(PrefixCacheError::InternalError);
            }
            let next_free_entry_index = self.entries[entry_index as usize].reserved;
            physical_block_index = self.arena.acquire_block()?;
            self.free_entry_head = next_free_entry_index;
        }
        let entry = &mut self.entries[entry_index as usize];
        entry.flags = ENTRY_FLAG_VALID | entry_flags;
        entry.token_count = token_count;
        entry.first_token_index = first_token_index;
        entry.physical_block_index = physical_block_index;
        entry.reference_count = 0;
        entry.reserved = 0;
        entry.parent_hash = parent_hash;
        entry.block_hash = block_hash;
        entry.content_hash = content_hash;
        entry.reservation_epoch = operation_epoch;
        entry.committed_epoch = 0;
        entry.lookahead_priority = 0;
        entry.lookahead_request_count = 0;
        entry.lookahead_protection_epoch = 0;
        self.tick += 1;
        entry.last_used_tick = self.tick;
        self.inserted_block_count += 1;
        if entry_flags & ENTRY_FLAG_LIVE_ONLY != 0 {
            self.live_only_block_count += 1;
        }
        Ok(())
    }

    fn release_binding(&mut self, binding_index: u32) -> Result<(), PrefixCacheError> {
        if binding_index as usize >= self.bindings.len() {
            return Err(PrefixCacheError::InvalidArgument);
        }
        let binding = &self.bindings[binding_index as usize];
        if binding.flags & BINDING_FLAG_VALID == 0 {
            return Ok(());
        }
        let entry_index = binding.entry_index;
        let physical_block_index = binding.physical_block_index;
        if entry_index as usize >= self.entries.len()
            || self.entries[entry_index as usize].reference_count == 0
        {
            return Err(PrefixCacheError::InvalidArgument);
        }
        self.arena.release_block_reference(physical_block_index)?;
        self.entries[entry_index as usize].reference_count -= 1;
        self.released_block_count += 1;
        let binding = &mut self.bindings[binding_index as usize];
        *binding = Binding::unallocated();
        binding.reserved = self.free_binding_head;
        self.free_binding_head = binding_index;
        Ok(())
    }

    fn acquire_entry_for_sequence(
        &mut self,
        sequence_id: u64,
        entry_index: u32,
        operation_epoch: u64,
        binding_is_pending: bool,
    ) -> Result<(), PrefixCacheError> {
        if entry_index as usize >= self.entries.len() {
            return Err(PrefixCacheError::InvalidArgument);
        }
        if let Some(binding_index) = self.find_binding(sequence_id, entry_index) {
            if binding_is_pending {
                self.bindings[binding_index as usize].flags |= BINDING_FLAG_PENDING;
            }
            return Ok(());
        }
        let binding_index = self.find_free_binding().ok_or(PrefixCacheError::CapacityExceeded)?;
        let physical_block_index = self.entries[entry_index as usize].physical_block_index;
        self.arena.retain_block(physical_block_index)?;
        let next_free_binding_index = self.bindings[binding_index as usize].reserved;
        if self.free_binding_head != binding_index {
            let _ = self.arena.release_block_reference(physical_block_index);
            return Err(PrefixCacheError::InternalError);
        }
        self.free_binding_head = next_free_binding_index;
        let entry = &self.entries[entry_index as usize];
        let binding = &mut self.bindings[binding_index as usize];
        binding.flags = BINDING_FLAG_VALID;
        if binding_is_pending {
            binding.flags |= BINDING_FLAG_PENDING;
        }
        binding.entry_index = entry_index;
        binding.first_token_index = entry.first_token_index;
        binding.token_count = entry.token_count;
        binding.physical_block_index = entry.physical_block_index;
        binding.sequence_id = sequence_id;
        binding.parent_hash = entry.parent_hash;
        binding.block_hash = entry.block_hash;
        binding.acquire_epoch = operation_epoch;
        self.entries[entry_index as usize].reference_count += 1;
        self.acquired_block_count += 1;
        Ok(())
    }

    fn rollback_epoch(
        &mut self,
        sequence_id: u64,
        operation_epoch: u64,
    ) -> Result<(), PrefixCacheError> {
        for binding_index in 0..self.bindings.len() as u32 {
            let binding = &self.bindings[binding_index as usize];
            if binding.flags & BINDING_FLAG_VALID != 0
                && binding.sequence_id == sequence_id
                && binding.acquire_epoch == operation_epoch
            {
                self.release_binding(binding_index)?;
            }
        }
        for entry_index in 0..self.entries.len() as u32 {
            let entry = &self.entries[entry_index as usize];
            if entry.flags & ENTRY_FLAG_VALID != 0
                && entry.flags & ENTRY_FLAG_PENDING != 0
                && entry.reservation_epoch == operation_epoch
                && entry.reference_count == 0
            {
                self.invalidate_entry(entry_index)?;
                self.cancelled_reserved_block_count += 1;
            }
        }
        Ok(())
    }

    /// Probe for a cached prefix without acquiring it
    /// (SparkPrefixCacheProbePrompt).
    pub fn probe_prompt(
        &mut self,
        sequence_id: u64,
        token_ids: &[u32],
    ) -> Result<Lookup, PrefixCacheError> {
        if token_ids.is_empty() || sequence_id == 0 {
            return Err(PrefixCacheError::InvalidArgument);
        }
        let token_count = token_ids.len() as u32;
        let mut lookup = Lookup::new(sequence_id, token_count);
        self.lookup_count += 1;
        let reusable_token_count = self.maximum_reusable_token_count(token_count);
        let mut parent_hash = EMPTY_PARENT_HASH;
        let mut token_offset = 0u32;
        while token_offset < reusable_token_count {
            let block_tokens =
                &token_ids[token_offset as usize..(token_offset + self.block_token_count) as usize];
            let block_hash = hash_block(block_tokens, parent_hash);
            let content_hash = hash_block_content(block_tokens);
            let mut entry_index: Option<u32> = None;
            if let Some(own_binding_index) =
                self.find_binding_at_token_offset(sequence_id, token_offset)
            {
                let own_entry_index = self.bindings[own_binding_index as usize].entry_index;
                if (own_entry_index as usize) < self.entries.len() {
                    let own_entry = &self.entries[own_entry_index as usize];
                    if own_entry.flags & ENTRY_FLAG_VALID != 0
                        && own_entry.first_token_index == token_offset
                        && own_entry.token_count == self.block_token_count
                        && own_entry.parent_hash == parent_hash
                        && own_entry.block_hash == block_hash
                        && own_entry.content_hash == content_hash
                    {
                        entry_index = Some(own_entry_index);
                    }
                }
            }
            if entry_index.is_none() {
                entry_index = self.find_entry(
                    parent_hash,
                    block_hash,
                    content_hash,
                    token_offset,
                    self.block_token_count,
                    true,
                );
            }
            let Some(entry_index) = entry_index else {
                break;
            };
            self.tick += 1;
            let entry = &mut self.entries[entry_index as usize];
            entry.last_used_tick = self.tick;
            parent_hash = block_hash;
            lookup.matched_token_count += self.block_token_count;
            lookup.matched_block_count += 1;
            lookup.physical_block_index = entry.physical_block_index;
            lookup.last_block_hash = block_hash;
            token_offset += self.block_token_count;
        }
        lookup.next_token_index = lookup.matched_token_count;
        if lookup.matched_token_count != 0 {
            self.hit_count += 1;
        } else {
            self.miss_count += 1;
        }
        Ok(lookup)
    }

    /// Step the cached prefix chain one block, returning the matched entry
    /// index (SparkPrefixCacheWalkNext, including its exact tick behaviour:
    /// the tick advances on continuation and terminating steps, and a matched
    /// entry's recency is stamped with `tick + 1`).
    fn walk_next(&mut self, token_ids: &[u32], walk: &mut ChainWalk) -> Option<u32> {
        if walk.matched_block_count == 0 && walk.token_offset == 0 {
            walk.parent_hash = EMPTY_PARENT_HASH;
            walk.reusable_token_count = self.maximum_reusable_token_count(token_ids.len() as u32);
        } else {
            self.tick += 1;
            walk.parent_hash = walk.block_hash;
            walk.token_offset += self.block_token_count;
        }
        if walk.token_offset >= walk.reusable_token_count {
            return None;
        }
        let offset = walk.token_offset as usize;
        let block_tokens = &token_ids[offset..offset + self.block_token_count as usize];
        walk.block_hash = hash_block(block_tokens, walk.parent_hash);
        let content_hash = hash_block_content(block_tokens);
        let entry_index = self.find_entry(
            walk.parent_hash,
            walk.block_hash,
            content_hash,
            walk.token_offset,
            self.block_token_count,
            true,
        )?;
        self.entries[entry_index as usize].last_used_tick = self.tick + 1;
        walk.matched_block_count += 1;
        Some(entry_index)
    }

    /// Probe the physical block table of the reusable prefix
    /// (SparkPrefixCacheProbePhysicalBlockTable). Refuses with
    /// [`PrefixCacheError::CapacityExceeded`] rather than truncating when
    /// `physical_block_capacity` is too small.
    pub fn probe_physical_block_table(
        &mut self,
        token_ids: &[u32],
        physical_block_capacity: u32,
    ) -> Result<PhysicalBlockTableProbe, PrefixCacheError> {
        if token_ids.is_empty() {
            return Err(PrefixCacheError::InvalidArgument);
        }
        let mut physical_block_indices = Vec::new();
        let mut walk = ChainWalk::default();
        while let Some(entry_index) = self.walk_next(token_ids, &mut walk) {
            if physical_block_indices.len() as u32 >= physical_block_capacity {
                return Err(PrefixCacheError::CapacityExceeded);
            }
            physical_block_indices.push(self.entries[entry_index as usize].physical_block_index);
        }
        let matched_token_count = physical_block_indices.len() as u32 * self.block_token_count;
        Ok(PhysicalBlockTableProbe { matched_token_count, physical_block_indices })
    }

    /// Probe prefetch sources for the reusable prefix
    /// (SparkPrefixCacheProbeReusablePrefixPrefetchSources).
    pub fn probe_reusable_prefix_prefetch_sources(
        &mut self,
        token_ids: &[u32],
        source_block_capacity: u32,
    ) -> Result<PrefetchSourceProbe, PrefixCacheError> {
        if token_ids.is_empty() {
            return Err(PrefixCacheError::InvalidArgument);
        }
        let mut source_blocks = Vec::new();
        let mut walk = ChainWalk::default();
        while let Some(entry_index) = self.walk_next(token_ids, &mut walk) {
            if source_blocks.len() as u32 >= source_block_capacity {
                return Err(PrefixCacheError::CapacityExceeded);
            }
            source_blocks.push(self.fill_prefetch_source_block(entry_index)?);
        }
        let matched_token_count = source_blocks.len() as u32 * self.block_token_count;
        Ok(PrefetchSourceProbe { matched_token_count, source_blocks })
    }

    /// Probe residency of the reusable prefix
    /// (SparkPrefixCacheProbeReusablePrefixResidency).
    pub fn probe_reusable_prefix_residency(
        &mut self,
        token_ids: &[u32],
    ) -> Result<PrefixResidencyProbe, PrefixCacheError> {
        if token_ids.is_empty() {
            return Err(PrefixCacheError::InvalidArgument);
        }
        let mut resident_block_count = 0u32;
        let mut nonresident_block_count = 0u32;
        let mut walk = ChainWalk::default();
        while let Some(entry_index) = self.walk_next(token_ids, &mut walk) {
            let physical_block_index = self.entries[entry_index as usize].physical_block_index;
            if physical_block_index < self.physical_block_count
                && !self.arena.is_resident(physical_block_index)
            {
                nonresident_block_count += 1;
            } else {
                resident_block_count += 1;
            }
        }
        Ok(PrefixResidencyProbe {
            matched_token_count: (resident_block_count + nonresident_block_count)
                * self.block_token_count,
            resident_block_count,
            nonresident_block_count,
        })
    }

    /// Advance the lookahead protection epoch (SparkPrefixCacheResetLookaheadProtection).
    pub fn reset_lookahead_protection(&mut self) -> Result<(), PrefixCacheError> {
        self.lookahead_protection_epoch = self.lookahead_protection_epoch.wrapping_add(1);
        if self.lookahead_protection_epoch == 0 {
            self.lookahead_protection_epoch = 1;
            for entry in &mut self.entries {
                entry.lookahead_protection_epoch = 0;
                entry.lookahead_priority = 0;
                entry.lookahead_request_count = 0;
            }
        }
        Ok(())
    }

    /// Mark the reusable prefix blocks of a prompt as protected for the
    /// current lookahead epoch (SparkPrefixCacheProtectPromptLookahead).
    pub fn protect_prompt_lookahead(
        &mut self,
        token_ids: &[u32],
        demand_weight: u32,
    ) -> Result<LookaheadProtection, PrefixCacheError> {
        if token_ids.is_empty() {
            return Err(PrefixCacheError::InvalidArgument);
        }
        if self.lookahead_protection_epoch == 0 {
            self.reset_lookahead_protection()?;
        }
        let reusable_token_count = self.maximum_reusable_token_count(token_ids.len() as u32);
        let mut parent_hash = EMPTY_PARENT_HASH;
        let mut protected_block_count = 0u32;
        let mut token_offset = 0u32;
        while token_offset < reusable_token_count {
            let block_tokens =
                &token_ids[token_offset as usize..(token_offset + self.block_token_count) as usize];
            let block_hash = hash_block(block_tokens, parent_hash);
            let content_hash = hash_block_content(block_tokens);
            let Some(entry_index) = self.find_entry(
                parent_hash,
                block_hash,
                content_hash,
                token_offset,
                self.block_token_count,
                true,
            ) else {
                break;
            };
            let entry = &mut self.entries[entry_index as usize];
            if entry.lookahead_protection_epoch != self.lookahead_protection_epoch {
                entry.lookahead_protection_epoch = self.lookahead_protection_epoch;
                entry.lookahead_priority = demand_weight;
                entry.lookahead_request_count = 1;
                self.lookahead_protected_block_count += 1;
            } else {
                if demand_weight > entry.lookahead_priority {
                    entry.lookahead_priority = demand_weight;
                }
                entry.lookahead_request_count = entry.lookahead_request_count.saturating_add(1);
            }
            protected_block_count += 1;
            parent_hash = block_hash;
            token_offset += self.block_token_count;
        }
        Ok(LookaheadProtection {
            protected_token_count: protected_block_count * self.block_token_count,
            protected_block_count,
        })
    }

    /// Most valuable valid entry holding `physical_block_index` (the C code
    /// keeps the entry that is the *worst* eviction victim, i.e. highest
    /// priority / most recent).
    fn find_resident_entry_for_physical_block(&self, physical_block_index: u32) -> Option<u32> {
        let mut best_entry: Option<u32> = None;
        for (index, entry) in self.entries.iter().enumerate() {
            if entry.flags & ENTRY_FLAG_VALID == 0
                || entry.physical_block_index != physical_block_index
            {
                continue;
            }
            let replace = match best_entry {
                None => true,
                Some(best) => protected_victim_is_better(&self.entries[best as usize], entry),
            };
            if replace {
                best_entry = Some(index as u32);
            }
        }
        best_entry
    }

    /// Arena block `last_used_epoch` stamp (C reads `block->last_used_epoch`
    /// directly; the arena exposes it as an accessor).
    fn block_last_used_epoch(&self, physical_block_index: u32) -> u64 {
        self.arena.last_used_epoch(physical_block_index)
    }

    fn compute_resident_block_keep_score(
        &self,
        physical_block_index: u32,
        entry_index: Option<u32>,
    ) -> u64 {
        let mut score = self.block_last_used_epoch(physical_block_index);
        score = score.saturating_add(
            (self.arena.reference_count(physical_block_index) as u64)
                .saturating_mul(REUSE_SCORE_REFERENCE_WEIGHT),
        );
        let Some(entry_index) = entry_index else {
            return score;
        };
        let entry = &self.entries[entry_index as usize];
        score = score.saturating_add(entry.last_used_tick);
        score = score.saturating_add(
            (entry.reference_count as u64).saturating_mul(REUSE_SCORE_REFERENCE_WEIGHT),
        );
        let prefix_token_depth = entry.first_token_index + entry.token_count;
        score = score.saturating_add(
            (prefix_token_depth as u64).saturating_mul(REUSE_SCORE_TOKEN_DEPTH_WEIGHT),
        );
        if self.entry_has_current_lookahead_protection(entry_index) {
            score = score.saturating_add(REUSE_SCORE_LOOKAHEAD_BASE);
            score = score.saturating_add(
                (entry.lookahead_priority as u64).saturating_mul(REUSE_SCORE_PRIORITY_WEIGHT),
            );
            score = score.saturating_add(
                (entry.lookahead_request_count as u64).saturating_mul(REUSE_SCORE_REQUEST_WEIGHT),
            );
        }
        score
    }

    fn select_resident_reuse_score_victim(
        &self,
        hard_protected_physical_block_indices: &[u32],
    ) -> Option<ResidentEvictionCandidate> {
        let mut victim = ResidentEvictionCandidate::none();
        // C iterates the arena's physical_block_count.
        for physical_block_index in 0..self.arena.physical_block_count() {
            if !self.arena.is_allocated(physical_block_index)
                || !self.arena.is_resident(physical_block_index)
                || hard_protected_physical_block_indices.contains(&physical_block_index)
            {
                continue;
            }
            let entry_index = self.find_resident_entry_for_physical_block(physical_block_index);
            let candidate = ResidentEvictionCandidate {
                physical_block_index,
                has_prefix_entry: entry_index.is_some(),
                has_lookahead_protection: entry_index
                    .is_some_and(|index| self.entry_has_current_lookahead_protection(index)),
                keep_score: self
                    .compute_resident_block_keep_score(physical_block_index, entry_index),
                last_used_epoch: self.block_last_used_epoch(physical_block_index),
            };
            if candidate.is_better(&victim) {
                victim = candidate;
            }
        }
        if victim.physical_block_index == NO_PHYSICAL_BLOCK {
            None
        } else {
            Some(victim)
        }
    }

    /// Trim resident blocks down to `max_resident_block_count`, choosing
    /// victims by reuse keep-score (SparkPrefixCacheTrimResidentBlocksByReuseScore).
    /// Returns the number of blocks evicted.
    ///
    /// Remaining deviations from C: the arena-side `resident_evicted_block_count`
    /// / `resident_capacity_stall_count` increments the C code performs
    /// directly on the arena struct are arena-internal here (`KvArena` keeps
    /// them private), and a mid-trim stall returns `Err(CapacityExceeded)`
    /// without the partial eviction count the C out-param would carry.
    pub fn trim_resident_blocks_by_reuse_score(
        &mut self,
        max_resident_block_count: u32,
        hard_protected_physical_block_indices: &[u32],
    ) -> Result<u32, PrefixCacheError> {
        if max_resident_block_count > self.arena.physical_block_count() {
            return Err(PrefixCacheError::InvalidArgument);
        }
        let mut evicted_block_count = 0u32;
        while self.arena.resident_block_count() > max_resident_block_count as u64 {
            let Some(victim) =
                self.select_resident_reuse_score_victim(hard_protected_physical_block_indices)
            else {
                self.reuse_scored_capacity_stall_count += 1;
                return Err(PrefixCacheError::CapacityExceeded);
            };
            self.arena.mark_block_nonresident(victim.physical_block_index)?;
            self.reuse_scored_resident_eviction_count += 1;
            if victim.has_lookahead_protection {
                self.reuse_scored_lookahead_eviction_count += 1;
            }
            if !victim.has_prefix_entry {
                self.reuse_scored_untracked_eviction_count += 1;
            }
            evicted_block_count += 1;
        }
        Ok(evicted_block_count)
    }

    /// Probe and then acquire (bind) the matched reusable prefix for
    /// `sequence_id` (SparkPrefixCacheLookupPrompt).
    pub fn lookup_prompt(
        &mut self,
        sequence_id: u64,
        token_ids: &[u32],
    ) -> Result<Lookup, PrefixCacheError> {
        let probe = self.probe_prompt(sequence_id, token_ids)?;
        self.operation_epoch += 1;
        let operation_epoch = self.operation_epoch;
        let mut parent_hash = EMPTY_PARENT_HASH;
        let mut token_offset = 0u32;
        while token_offset < probe.matched_token_count {
            let block_tokens =
                &token_ids[token_offset as usize..(token_offset + self.block_token_count) as usize];
            let block_hash = hash_block(block_tokens, parent_hash);
            let content_hash = hash_block_content(block_tokens);
            let Some(entry_index) = self.find_entry(
                parent_hash,
                block_hash,
                content_hash,
                token_offset,
                self.block_token_count,
                true,
            ) else {
                let _ = self.rollback_epoch(sequence_id, operation_epoch);
                return Err(PrefixCacheError::InternalError);
            };
            if let Err(status) =
                self.acquire_entry_for_sequence(sequence_id, entry_index, operation_epoch, false)
            {
                let _ = self.rollback_epoch(sequence_id, operation_epoch);
                return Err(status);
            }
            parent_hash = block_hash;
            token_offset += self.block_token_count;
        }
        Ok(probe)
    }

    fn reserve_prompt_internal(
        &mut self,
        sequence_id: u64,
        token_ids: &[u32],
        allow_cross_sequence_reuse: bool,
    ) -> Result<Reservation, PrefixCacheError> {
        if token_ids.is_empty() || sequence_id == 0 {
            return Err(PrefixCacheError::InvalidArgument);
        }
        let token_count = token_ids.len() as u32;
        let block_count = self.ceil_block_count(token_count);
        self.operation_epoch += 1;
        let operation_epoch = self.operation_epoch;
        let mut reservation = Reservation {
            requested_token_count: token_count,
            reserved_token_count: 0,
            reusable_token_count: 0,
            physical_block_count: 0,
            cached_physical_block_count: 0,
            pending_physical_block_count: 0,
            sequence_id,
            reservation_epoch: operation_epoch,
            last_block_hash: EMPTY_PARENT_HASH,
            physical_block_indices: Vec::with_capacity(block_count as usize),
        };
        let reusable_token_count = self.maximum_reusable_token_count(token_count);
        let mut parent_hash = EMPTY_PARENT_HASH;
        let mut token_offset = 0u32;
        for _block_index in 0..block_count {
            let block_token_count = self.block_token_count.min(token_count - token_offset);
            let is_full_block = block_token_count == self.block_token_count;
            let block_tokens =
                &token_ids[token_offset as usize..(token_offset + block_token_count) as usize];
            let block_hash = hash_block(block_tokens, parent_hash);
            let content_hash = hash_block_content(block_tokens);
            let existing_binding = self.find_binding_at_token_offset(sequence_id, token_offset);
            let mut entry_index: Option<u32> = existing_binding.and_then(|binding_index| {
                let index = self.bindings[binding_index as usize].entry_index;
                ((index as usize) < self.entries.len()).then_some(index)
            });
            if entry_index.is_none()
                && allow_cross_sequence_reuse
                && is_full_block
                && token_offset < reusable_token_count
            {
                entry_index = self.find_entry(
                    parent_hash,
                    block_hash,
                    content_hash,
                    token_offset,
                    block_token_count,
                    true,
                );
            }
            let entry_index = match entry_index {
                Some(index) => {
                    if self.entry_is_reusable(index) {
                        reservation.cached_physical_block_count += 1;
                        reservation.reusable_token_count += block_token_count;
                    } else if self.entries[index as usize].flags & ENTRY_FLAG_PENDING != 0 {
                        reservation.pending_physical_block_count += 1;
                    }
                    index
                }
                None => {
                    let Some(victim_index) = self.select_victim() else {
                        let _ = self.rollback_epoch(sequence_id, operation_epoch);
                        return Err(PrefixCacheError::CapacityExceeded);
                    };
                    let mut entry_flags = ENTRY_FLAG_PENDING;
                    if !is_full_block || !allow_cross_sequence_reuse {
                        entry_flags |= ENTRY_FLAG_LIVE_ONLY;
                    }
                    if let Err(status) = self.install_entry(
                        victim_index,
                        entry_flags,
                        parent_hash,
                        block_hash,
                        content_hash,
                        token_offset,
                        block_token_count,
                        operation_epoch,
                    ) {
                        let _ = self.rollback_epoch(sequence_id, operation_epoch);
                        return Err(status);
                    }
                    reservation.pending_physical_block_count += 1;
                    self.reserved_block_count += 1;
                    victim_index
                }
            };
            let binding_is_pending =
                self.entries[entry_index as usize].flags & ENTRY_FLAG_PENDING != 0;
            if let Err(status) = self.acquire_entry_for_sequence(
                sequence_id,
                entry_index,
                operation_epoch,
                binding_is_pending,
            ) {
                let _ = self.rollback_epoch(sequence_id, operation_epoch);
                return Err(status);
            }
            reservation
                .physical_block_indices
                .push(self.entries[entry_index as usize].physical_block_index);
            self.tick += 1;
            self.entries[entry_index as usize].last_used_tick = self.tick;
            parent_hash = block_hash;
            token_offset += block_token_count;
        }
        reservation.reserved_token_count = token_count;
        reservation.physical_block_count = block_count;
        reservation.last_block_hash = parent_hash;
        Ok(reservation)
    }

    /// Reserve blocks for a prompt, reusing committed content-addressed
    /// blocks where possible (SparkPrefixCacheReservePrompt).
    pub fn reserve_prompt(
        &mut self,
        sequence_id: u64,
        token_ids: &[u32],
    ) -> Result<Reservation, PrefixCacheError> {
        self.reserve_prompt_internal(sequence_id, token_ids, true)
    }

    /// Reserve blocks for a prompt without cross-sequence content reuse —
    /// every block is private/live-only (SparkPrefixCacheReserveSequencePrompt).
    pub fn reserve_sequence_prompt(
        &mut self,
        sequence_id: u64,
        token_ids: &[u32],
    ) -> Result<Reservation, PrefixCacheError> {
        self.reserve_prompt_internal(sequence_id, token_ids, false)
    }

    /// Commit a reservation epoch: pending entries become committed (full,
    /// non-live-only blocks become reusable), pending bindings clear
    /// (SparkPrefixCacheCommitReservation).
    pub fn commit_reservation(
        &mut self,
        sequence_id: u64,
        reservation_epoch: u64,
    ) -> Result<(), PrefixCacheError> {
        if sequence_id == 0 || reservation_epoch == 0 {
            return Err(PrefixCacheError::InvalidArgument);
        }
        for entry_index in 0..self.entries.len() as u32 {
            let entry = &self.entries[entry_index as usize];
            if entry.flags & ENTRY_FLAG_VALID != 0
                && entry.flags & ENTRY_FLAG_PENDING != 0
                && entry.reservation_epoch == reservation_epoch
            {
                let physical_block_index = entry.physical_block_index;
                self.arena.mark_block_resident(physical_block_index)?;
                let entry = &mut self.entries[entry_index as usize];
                entry.flags &= !ENTRY_FLAG_PENDING;
                if entry.token_count == self.block_token_count
                    && entry.flags & ENTRY_FLAG_LIVE_ONLY == 0
                {
                    entry.flags |= ENTRY_FLAG_REUSABLE;
                } else {
                    entry.flags |= ENTRY_FLAG_LIVE_ONLY;
                }
                entry.committed_epoch = reservation_epoch;
                self.committed_reserved_block_count += 1;
            }
        }
        for binding in &mut self.bindings {
            if binding.flags & BINDING_FLAG_VALID != 0
                && binding.sequence_id == sequence_id
                && binding.acquire_epoch == reservation_epoch
            {
                binding.flags &= !BINDING_FLAG_PENDING;
            }
        }
        Ok(())
    }

    /// Cancel a reservation epoch, rolling back its bindings and unreferenced
    /// pending entries (SparkPrefixCacheCancelReservation).
    pub fn cancel_reservation(
        &mut self,
        sequence_id: u64,
        reservation_epoch: u64,
    ) -> Result<(), PrefixCacheError> {
        if sequence_id == 0 || reservation_epoch == 0 {
            return Err(PrefixCacheError::InvalidArgument);
        }
        self.rollback_epoch(sequence_id, reservation_epoch)
    }

    /// Reserve and immediately commit a prompt (SparkPrefixCacheCommitPrompt).
    pub fn commit_prompt(
        &mut self,
        sequence_id: u64,
        token_ids: &[u32],
    ) -> Result<Lookup, PrefixCacheError> {
        let token_count = token_ids.len() as u32;
        let mut lookup = Lookup::new(sequence_id, token_count);
        let reservation = self.reserve_prompt(sequence_id, token_ids)?;
        if let Err(status) = self.commit_reservation(sequence_id, reservation.reservation_epoch) {
            let _ = self.cancel_reservation(sequence_id, reservation.reservation_epoch);
            return Err(status);
        }
        let committed_token_count = self.full_block_token_count(token_count);
        lookup.matched_token_count = committed_token_count;
        lookup.matched_block_count = committed_token_count / self.block_token_count;
        lookup.next_token_index = committed_token_count;
        lookup.last_block_hash = reservation.last_block_hash;
        Ok(lookup)
    }

    /// Ensure a live sequence owns enough KV blocks to write `token_count`
    /// positions; blocks added here are live-only and the call is idempotent
    /// for the same or smaller counts (SparkPrefixCacheEnsureSequenceTokenCapacity).
    pub fn ensure_sequence_token_capacity(
        &mut self,
        sequence_id: u64,
        token_count: u32,
    ) -> Result<(), PrefixCacheError> {
        if sequence_id == 0 || token_count == 0 {
            return Err(PrefixCacheError::InvalidArgument);
        }
        let block_count = self.ceil_block_count(token_count);
        let mut short_binding: Option<u32> = None;
        let mut short_entry: Option<u32> = None;
        let mut parent_hash = EMPTY_PARENT_HASH;
        self.operation_epoch += 1;
        let operation_epoch = self.operation_epoch;

        for block_index in 0..block_count {
            let first_token_index = block_index * self.block_token_count;
            let required_block_token_count =
                self.block_token_count.min(token_count - first_token_index);
            if let Some(binding_index) =
                self.find_binding_at_token_offset(sequence_id, first_token_index)
            {
                let binding = &self.bindings[binding_index as usize];
                let entry_index = binding.entry_index;
                if entry_index as usize >= self.entries.len()
                    || binding.physical_block_index >= self.physical_block_count
                    || binding.token_count == 0
                    || binding.token_count > self.block_token_count
                {
                    let _ = self.rollback_epoch(sequence_id, operation_epoch);
                    return Err(PrefixCacheError::InvalidArgument);
                }
                let binding_physical_block_index = binding.physical_block_index;
                let binding_token_count = binding.token_count;
                let binding_block_hash = binding.block_hash;
                let entry = &self.entries[entry_index as usize];
                if entry.flags & ENTRY_FLAG_VALID == 0
                    || entry.flags & ENTRY_FLAG_PENDING != 0
                    || entry.first_token_index != first_token_index
                    || entry.physical_block_index != binding_physical_block_index
                    || entry.token_count != binding_token_count
                {
                    let _ = self.rollback_epoch(sequence_id, operation_epoch);
                    return Err(PrefixCacheError::Busy);
                }
                if binding_token_count < required_block_token_count {
                    // Only the final, private, live-only prompt block may grow
                    // in place. Growing a shared partial block would corrupt a
                    // fork and requires an explicit KV copy-on-write backend.
                    if entry.flags & ENTRY_FLAG_LIVE_ONLY == 0
                        || entry.reference_count != 1
                        || short_binding.is_some()
                    {
                        let _ = self.rollback_epoch(sequence_id, operation_epoch);
                        return Err(PrefixCacheError::ModuleNotValidated);
                    }
                    short_binding = Some(binding_index);
                    short_entry = Some(entry_index);
                }
                parent_hash = binding_block_hash;
                continue;
            }

            let Some(victim_index) = self.select_victim() else {
                let _ = self.rollback_epoch(sequence_id, operation_epoch);
                return Err(PrefixCacheError::CapacityExceeded);
            };
            let block_hash = mix_u64(mix_u64(parent_hash, sequence_id), first_token_index as u64);
            let content_hash = mix_u64(block_hash, operation_epoch);
            if let Err(status) = self.install_entry(
                victim_index,
                ENTRY_FLAG_PENDING | ENTRY_FLAG_LIVE_ONLY,
                parent_hash,
                block_hash,
                content_hash,
                first_token_index,
                self.block_token_count,
                operation_epoch,
            ) {
                let _ = self.rollback_epoch(sequence_id, operation_epoch);
                return Err(status);
            }
            self.reserved_block_count += 1;
            if let Err(status) =
                self.acquire_entry_for_sequence(sequence_id, victim_index, operation_epoch, true)
            {
                let _ = self.rollback_epoch(sequence_id, operation_epoch);
                return Err(status);
            }
            parent_hash = block_hash;
        }

        if let Err(status) = self.commit_reservation(sequence_id, operation_epoch) {
            let _ = self.rollback_epoch(sequence_id, operation_epoch);
            return Err(status);
        }
        if let (Some(binding_index), Some(entry_index)) = (short_binding, short_entry) {
            self.bindings[binding_index as usize].token_count = self.block_token_count;
            let entry = &mut self.entries[entry_index as usize];
            entry.token_count = self.block_token_count;
            entry.flags &= !ENTRY_FLAG_REUSABLE;
            entry.flags |= ENTRY_FLAG_LIVE_ONLY;
        }
        Ok(())
    }

    /// Build the physical block table for a bound sequence
    /// (SparkPrefixCacheBuildPhysicalBlockTable).
    pub fn build_physical_block_table(
        &mut self,
        sequence_id: u64,
        token_count: u32,
    ) -> Result<Vec<u32>, PrefixCacheError> {
        if sequence_id == 0 || token_count == 0 {
            return Err(PrefixCacheError::InvalidArgument);
        }
        let block_count = self.ceil_block_count(token_count);
        let mut physical_block_indices = Vec::with_capacity(block_count as usize);
        for block_index in 0..block_count {
            let token_offset = block_index * self.block_token_count;
            let required_block_token_count = self.block_token_count.min(token_count - token_offset);
            let Some(binding_index) = self.find_binding_at_token_offset(sequence_id, token_offset)
            else {
                return Err(PrefixCacheError::NotFound);
            };
            let binding = &self.bindings[binding_index as usize];
            if binding.token_count < required_block_token_count {
                return Err(PrefixCacheError::NotFound);
            }
            physical_block_indices.push(binding.physical_block_index);
        }
        Ok(physical_block_indices)
    }

    /// Fill one prefetch source descriptor from an entry
    /// (SparkPrefixCacheFillPrefetchSourceBlock; generation via `resolve_block`).
    fn fill_prefetch_source_block(
        &self,
        entry_index: u32,
    ) -> Result<PrefetchSourceBlock, PrefixCacheError> {
        let entry = &self.entries[entry_index as usize];
        if entry.flags & ENTRY_FLAG_VALID == 0
            || entry.physical_block_index == NO_PHYSICAL_BLOCK
            || entry.physical_block_index >= self.physical_block_count
        {
            return Err(PrefixCacheError::InvalidArgument);
        }
        let generation = self
            .arena
            .resolve_block(entry.physical_block_index)
            .map(|view| view.generation)
            .unwrap_or(0);
        Ok(PrefetchSourceBlock {
            physical_block_index: entry.physical_block_index,
            token_capacity: self.block_token_count,
            first_token_index: entry.first_token_index,
            token_count: entry.token_count,
            generation,
            parent_hash: entry.parent_hash,
            block_hash: entry.block_hash,
            content_hash: entry.content_hash,
        })
    }

    /// Build prefetch source descriptors for a bound sequence
    /// (SparkPrefixCacheBuildSequencePrefetchSources).
    pub fn build_sequence_prefetch_sources(
        &mut self,
        sequence_id: u64,
        token_count: u32,
    ) -> Result<Vec<PrefetchSourceBlock>, PrefixCacheError> {
        if sequence_id == 0 || token_count == 0 {
            return Err(PrefixCacheError::InvalidArgument);
        }
        let block_count = self.ceil_block_count(token_count);
        let mut source_blocks = Vec::with_capacity(block_count as usize);
        let mut token_offset = 0u32;
        for _block_index in 0..block_count {
            let Some(binding_index) = self.find_binding_at_token_offset(sequence_id, token_offset)
            else {
                return Err(PrefixCacheError::NotFound);
            };
            let binding = &self.bindings[binding_index as usize];
            let entry_index = binding.entry_index;
            if entry_index as usize >= self.entries.len() {
                return Err(PrefixCacheError::NotFound);
            }
            token_offset += binding.token_count;
            source_blocks.push(self.fill_prefetch_source_block(entry_index)?);
        }
        Ok(source_blocks)
    }

    /// Probe residency of a live sequence's bound blocks
    /// (SparkPrefixCacheProbeSequenceResidency).
    pub fn probe_sequence_residency(
        &mut self,
        sequence_id: u64,
        token_count: u32,
    ) -> Result<SequenceResidencyProbe, PrefixCacheError> {
        if sequence_id == 0 || token_count == 0 {
            return Err(PrefixCacheError::InvalidArgument);
        }
        let block_count = self.ceil_block_count(token_count);
        let mut resident_block_count = 0u32;
        let mut nonresident_block_count = 0u32;
        let mut token_offset = 0u32;
        for _block_index in 0..block_count {
            let Some(binding_index) = self.find_binding_at_token_offset(sequence_id, token_offset)
            else {
                return Err(PrefixCacheError::NotFound);
            };
            let binding = &self.bindings[binding_index as usize];
            let physical_block_index = binding.physical_block_index;
            token_offset += binding.token_count;
            if physical_block_index < self.physical_block_count
                && !self.arena.is_resident(physical_block_index)
            {
                nonresident_block_count += 1;
            } else {
                resident_block_count += 1;
            }
        }
        Ok(SequenceResidencyProbe {
            physical_block_count: block_count,
            resident_block_count,
            nonresident_block_count,
        })
    }

    /// Bind the committed prefix blocks of one sequence to another
    /// (SparkPrefixCacheBindCommittedPrefixFromSequence).
    pub fn bind_committed_prefix_from_sequence(
        &mut self,
        source_sequence_id: u64,
        target_sequence_id: u64,
        token_count: u32,
    ) -> Result<(), PrefixCacheError> {
        if source_sequence_id == 0
            || target_sequence_id == 0
            || source_sequence_id == target_sequence_id
            || token_count == 0
        {
            return Err(PrefixCacheError::InvalidArgument);
        }
        self.operation_epoch += 1;
        let operation_epoch = self.operation_epoch;
        let mut token_offset = 0u32;
        while token_offset < token_count {
            let Some(source_binding_index) =
                self.find_binding_at_token_offset(source_sequence_id, token_offset)
            else {
                let _ = self.rollback_epoch(target_sequence_id, operation_epoch);
                return Err(PrefixCacheError::NotFound);
            };
            let source_binding = &self.bindings[source_binding_index as usize];
            let source_entry_index = source_binding.entry_index;
            let source_binding_token_count = source_binding.token_count;
            if source_entry_index as usize >= self.entries.len()
                || source_binding_token_count == 0
                || token_offset + source_binding_token_count > token_count
            {
                let _ = self.rollback_epoch(target_sequence_id, operation_epoch);
                return Err(PrefixCacheError::NotFound);
            }
            let entry = &self.entries[source_entry_index as usize];
            if entry.flags & ENTRY_FLAG_VALID == 0 || entry.flags & ENTRY_FLAG_PENDING != 0 {
                let _ = self.rollback_epoch(target_sequence_id, operation_epoch);
                return Err(PrefixCacheError::Busy);
            }
            if let Some(target_binding_index) =
                self.find_binding_at_token_offset(target_sequence_id, token_offset)
            {
                if self.bindings[target_binding_index as usize].entry_index != source_entry_index {
                    let _ = self.rollback_epoch(target_sequence_id, operation_epoch);
                    return Err(PrefixCacheError::Duplicate);
                }
            }
            if let Err(status) = self.acquire_entry_for_sequence(
                target_sequence_id,
                source_entry_index,
                operation_epoch,
                false,
            ) {
                let _ = self.rollback_epoch(target_sequence_id, operation_epoch);
                return Err(status);
            }
            token_offset += source_binding_token_count;
        }
        Ok(())
    }

    /// Release all bindings of a sequence and invalidate any unreferenced
    /// non-reusable entries (SparkPrefixCacheReleaseSequence).
    pub fn release_sequence(&mut self, sequence_id: u64) -> Result<(), PrefixCacheError> {
        if sequence_id == 0 {
            return Err(PrefixCacheError::InvalidArgument);
        }
        for binding_index in 0..self.bindings.len() as u32 {
            let binding = &self.bindings[binding_index as usize];
            if binding.flags & BINDING_FLAG_VALID != 0 && binding.sequence_id == sequence_id {
                self.release_binding(binding_index)?;
            }
        }
        for entry_index in 0..self.entries.len() as u32 {
            let entry = &self.entries[entry_index as usize];
            if entry.flags & ENTRY_FLAG_VALID != 0
                && entry.flags & ENTRY_FLAG_REUSABLE == 0
                && entry.reference_count == 0
            {
                self.invalidate_entry(entry_index)?;
            }
        }
        Ok(())
    }

    /// Reset the cache: all entries and bindings freed, arena blocks
    /// recycled, counters cleared. Mirrors the C quirk of NOT clearing the
    /// `reuse_scored_*` counters (SparkPrefixCacheReset).
    pub fn reset(&mut self) -> Result<(), PrefixCacheError> {
        let entry_len = self.entries.len();
        for (index, entry) in self.entries.iter_mut().enumerate() {
            *entry = Entry::unallocated();
            entry.reserved = if index + 1 < entry_len { index as u32 + 1 } else { NO_ENTRY };
        }
        let binding_len = self.bindings.len();
        for (index, binding) in self.bindings.iter_mut().enumerate() {
            *binding = Binding::unallocated();
            binding.reserved = if index + 1 < binding_len { index as u32 + 1 } else { NO_ENTRY };
        }
        // SparkKvCacheArenaReset: reinitializes every block and clears stats.
        self.arena.reset();
        self.tick = 0;
        self.free_entry_head = 0;
        self.free_binding_head = 0;
        self.operation_epoch = 0;
        self.lookup_count = 0;
        self.hit_count = 0;
        self.miss_count = 0;
        self.inserted_block_count = 0;
        self.evicted_block_count = 0;
        self.acquired_block_count = 0;
        self.released_block_count = 0;
        self.reserved_block_count = 0;
        self.committed_reserved_block_count = 0;
        self.cancelled_reserved_block_count = 0;
        self.live_only_block_count = 0;
        self.lookahead_protection_epoch = 0;
        self.lookahead_protected_block_count = 0;
        self.lookahead_protected_eviction_skip_count = 0;
        Ok(())
    }
}
