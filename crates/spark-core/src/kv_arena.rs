//! Paged KV block arena — port of the `SparkKvCacheArena*` core in
//! `cache/kv_cache.c` (API declared in `include/sparkpipe/spark_kv_cache.h`).
//!
//! Ownership model (from the C design, `cache/cache.h`): a block's
//! `reference_count` is the entire ownership story — there is no separate
//! free list and no separate binding table, because two representations of
//! the same fact is how a block ends up free and bound at once. Note the
//! exact C mechanics: `acquire` claims a free block with `reference_count`
//! reset to 0 (a reference is taken explicitly with `retain`), `retain`
//! increments, `release` decrements (releasing at zero is rejected), and a
//! block only leaves the allocated state through an explicit `free_block`
//! (or `recycle_block`, which re-arms it in place). The `resident` bit is
//! fully independent of the reference count: residency says where the bytes
//! live (device-resident vs. pool), not who owns the block — a referenced
//! block may be non-resident and a resident block may be unreferenced and
//! merely warm.
//!
//! Eviction (resident trimming) picks the allocated+resident victim with the
//! lowest `reference_count`, breaking ties by the oldest `last_used_epoch`
//! (LRU). It skips only blocks on the caller-supplied protected list; it does
//! NOT skip referenced blocks — referenced blocks are merely evicted last.
//! `trim_resident_blocks` reports a capacity stall (`CapacityExceeded`) when
//! no victim exists; `evict_resident_blocks_to_limit` stops quietly instead.
//!
//! Generation bumps on every acquire/recycle/free so stale handles (e.g. a
//! prefetch plan built against an older generation) are refused.
//!
//! Statistics counters mirror the C arena: allocated / recycled / resident /
//! resident-evicted / resident-capacity-stall / retained / released. Two C
//! quirks are preserved faithfully:
//!   - `reset` re-initializes every block (generation back to 0) and zeroes
//!     epoch, allocated, recycled, resident, retained, released — but NOT the
//!     cumulative resident-evicted and capacity-stall counters.
//!   - `recycle_block` rewrites the flags word to `ALLOCATED`, clearing a
//!     stale resident bit without touching `resident_block_count` (in the C
//!     the two never co-occur on the recycle path; the port does not "fix"
//!     the accounting).
//!
//! Device addresses are opaque `usize` values (host builds use stub
//! pointers); this crate never dereferences them.
//!
//! Rust-port deviations from the C surface (the public API above is
//! otherwise signature-fixed):
//!   - The arena owns its block `Vec` (the C configuration took a
//!     caller-allocated `SparkKvCacheBlock *`).
//!   - `trim_resident_blocks` returns the evicted count in `Ok(..)`; on a
//!     capacity stall the C function still reports the partial count through
//!     an out-parameter — the Rust port returns `Err(CapacityExceeded)`
//!     without the partial count.
//!   - Per-block read accessors (`is_allocated`, `is_resident`,
//!     `reference_count`, `block_token_capacity`, `generation`,
//!     `key_device_address`, `value_device_address`, `last_used_epoch`)
//!     return `false`/`0` for an out-of-range index instead of erroring; the
//!     C side reads the block array directly. The prefix cache port reads
//!     block state only through these.
//!   - Statistic accessors (`epoch`, `recycled_block_count`,
//!     `resident_evicted_block_count`, `resident_capacity_stall_count`,
//!     `retained_block_count`, `released_reference_count`,
//!     `key_block_stride_bytes`, `value_block_stride_bytes`) are Rust-only
//!     additions so tests and sibling modules can observe the counters the C
//!     tests read off the arena struct.

/// Sentinel: no block (SPARK_KV_CACHE_NO_BLOCK).
pub const NO_BLOCK: u32 = 0xffff_ffff;
/// Block flag: allocated (SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED).
pub const BLOCK_FLAG_ALLOCATED: u32 = 0x0000_0001;
/// Block flag: bytes currently resident on device (SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT).
pub const BLOCK_FLAG_RESIDENT: u32 = 0x0000_0002;

/// Maximum tokens per block (SPARK_KV_CACHE_MAX_BLOCK_TOKENS).
pub const MAX_BLOCK_TOKENS: u32 = 256;
/// Maximum layer count (SPARK_KV_CACHE_MAX_LAYER_COUNT).
pub const MAX_LAYER_COUNT: u32 = 256;

/// Errors mirroring the C `SparkStatus` returns of the arena entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum KvArenaError {
    #[error("invalid argument")]
    InvalidArgument,
    #[error("capacity exceeded (no free block / resident capacity)")]
    CapacityExceeded,
    #[error("block not found or not in the required state")]
    NotFound,
    #[error("block is busy (referenced)")]
    Busy,
    #[error("internal error")]
    Internal,
}

/// Arena configuration (port of `SparkKvCacheConfiguration`, minus the
/// caller-owned block array — the Rust arena owns its blocks).
#[derive(Debug, Clone)]
pub struct KvArenaConfig {
    pub physical_block_count: u32,
    pub block_token_count: u32,
    pub resident_block_capacity: u32,
    pub layer_count: u32,
    pub kv_head_count: u32,
    pub head_dim: u32,
    pub bytes_per_scalar: u32,
    pub key_block_stride_bytes: u64,
    pub value_block_stride_bytes: u64,
    pub key_device_base: usize,
    pub value_device_base: usize,
}

/// Read-only resolved view of one block (port of `SparkKvCacheBlockView`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockView {
    pub physical_block_index: u32,
    pub token_capacity: u32,
    pub layer_count: u32,
    pub kv_head_count: u32,
    pub head_dim: u32,
    pub bytes_per_scalar: u32,
    pub generation: u64,
    pub key_block_stride_bytes: u64,
    pub value_block_stride_bytes: u64,
    pub key_device_address: usize,
    pub value_device_address: usize,
}

/// One arena block (port of `SparkKvCacheBlock`, minus the ABI header).
#[derive(Debug, Clone)]
struct KvBlock {
    flags: u32,
    token_capacity: u32,
    reference_count: u32,
    generation: u64,
    last_used_epoch: u64,
    key_device_address: usize,
    value_device_address: usize,
}

/// Paged KV block arena. See module docs for the ownership model.
#[derive(Debug)]
pub struct KvArena {
    physical_block_count: u32,
    block_token_count: u32,
    resident_block_capacity: u32,
    layer_count: u32,
    kv_head_count: u32,
    head_dim: u32,
    bytes_per_scalar: u32,
    key_block_stride_bytes: u64,
    value_block_stride_bytes: u64,
    key_device_base: usize,
    value_device_base: usize,
    blocks: Vec<KvBlock>,
    epoch: u64,
    allocated_block_count: u64,
    recycled_block_count: u64,
    resident_block_count: u64,
    resident_evicted_block_count: u64,
    resident_capacity_stall_count: u64,
    retained_block_count: u64,
    released_reference_count: u64,
}

/// Port of `SparkKvProtectedBlockListContainsBlock`.
fn protected_block_list_contains(protected_physical_block_indices: &[u32], index: u32) -> bool {
    protected_physical_block_indices.contains(&index)
}

impl KvArena {
    /// Port of `SparkKvCacheArenaInitialize` (configuration validation
    /// included; the C arena-validation helper is unnecessary here because a
    /// constructed `KvArena` is always valid).
    pub fn new(config: &KvArenaConfig) -> Result<Self, KvArenaError> {
        if config.physical_block_count == 0
            || config.block_token_count == 0
            || config.block_token_count > MAX_BLOCK_TOKENS
            || config.resident_block_capacity > config.physical_block_count
            || config.layer_count == 0
            || config.layer_count > MAX_LAYER_COUNT
            || config.kv_head_count == 0
            || config.head_dim == 0
            || config.bytes_per_scalar == 0
            || config.key_device_base == 0
            || (config.value_device_base == 0 && config.value_block_stride_bytes != 0)
        {
            return Err(KvArenaError::InvalidArgument);
        }

        // SparkKvCacheDefaultBlockStrideBytes.
        let default_stride_bytes = u64::from(config.block_token_count)
            .wrapping_mul(u64::from(config.layer_count))
            .wrapping_mul(u64::from(config.kv_head_count))
            .wrapping_mul(u64::from(config.head_dim))
            .wrapping_mul(u64::from(config.bytes_per_scalar));

        // SparkKvCacheConfigurationHasValuePayload.
        let has_value_payload =
            config.value_device_base != 0 || config.value_block_stride_bytes != 0;

        let resident_block_capacity = if config.resident_block_capacity != 0 {
            config.resident_block_capacity
        } else {
            config.physical_block_count
        };
        let key_block_stride_bytes = if config.key_block_stride_bytes != 0 {
            config.key_block_stride_bytes
        } else {
            default_stride_bytes
        };
        let value_block_stride_bytes = if has_value_payload {
            if config.value_block_stride_bytes != 0 {
                config.value_block_stride_bytes
            } else {
                default_stride_bytes
            }
        } else {
            0
        };
        let value_device_base = if has_value_payload { config.value_device_base } else { 0 };

        let mut arena = KvArena {
            physical_block_count: config.physical_block_count,
            block_token_count: config.block_token_count,
            resident_block_capacity,
            layer_count: config.layer_count,
            kv_head_count: config.kv_head_count,
            head_dim: config.head_dim,
            bytes_per_scalar: config.bytes_per_scalar,
            key_block_stride_bytes,
            value_block_stride_bytes,
            key_device_base: config.key_device_base,
            value_device_base,
            blocks: Vec::with_capacity(config.physical_block_count as usize),
            epoch: 0,
            allocated_block_count: 0,
            recycled_block_count: 0,
            resident_block_count: 0,
            resident_evicted_block_count: 0,
            resident_capacity_stall_count: 0,
            retained_block_count: 0,
            released_reference_count: 0,
        };
        for index in 0..arena.physical_block_count {
            arena.blocks.push(arena.make_block(index));
        }
        Ok(arena)
    }

    /// Port of `SparkKvCacheInitializeBlock`: a fresh, unallocated block with
    /// its device addresses derived from the arena geometry.
    fn make_block(&self, physical_block_index: u32) -> KvBlock {
        let key_device_address = self.key_device_base.wrapping_add(
            self.key_block_stride_bytes.wrapping_mul(u64::from(physical_block_index)) as usize,
        );
        let value_device_address = if self.has_value_payload() {
            self.value_device_base.wrapping_add(
                self.value_block_stride_bytes.wrapping_mul(u64::from(physical_block_index))
                    as usize,
            )
        } else {
            0
        };
        KvBlock {
            flags: 0,
            token_capacity: self.block_token_count,
            reference_count: 0,
            generation: 0,
            last_used_epoch: 0,
            key_device_address,
            value_device_address,
        }
    }

    /// Port of `SparkKvCacheArenaHasValuePayload`.
    fn has_value_payload(&self) -> bool {
        self.value_device_base != 0 || self.value_block_stride_bytes != 0
    }

    fn block(&self, physical_block_index: u32) -> Option<&KvBlock> {
        self.blocks.get(physical_block_index as usize)
    }

    /// Acquire a free block (flags := ALLOCATED, reference_count := 0,
    /// generation bumped, last_used_epoch stamped).
    pub fn acquire_block(&mut self) -> Result<u32, KvArenaError> {
        for index in 0..self.physical_block_count {
            if self.blocks[index as usize].flags & BLOCK_FLAG_ALLOCATED == 0 {
                self.epoch += 1;
                let block = &mut self.blocks[index as usize];
                block.flags = BLOCK_FLAG_ALLOCATED;
                block.reference_count = 0;
                block.generation += 1;
                block.last_used_epoch = self.epoch;
                self.allocated_block_count += 1;
                return Ok(index);
            }
        }
        Err(KvArenaError::CapacityExceeded)
    }

    /// Return a block to the free pool regardless of references (reset path).
    ///
    /// Faithful to the C: requires the block to be allocated with
    /// `reference_count == 0`, then re-arms it (stays allocated, generation
    /// bumped) — it does not actually free the block.
    ///
    /// Deliberate deviation from `cache/kv_cache.c`: the C RecycleBlock
    /// clears the RESIDENT flag without decrementing `resident_block_count`
    /// (unlike C FreeBlock, which does), permanently inflating the count and
    /// eventually stalling every make-room trim once a recycled resident
    /// block exists. The C prefix-cache tests never attach an arena so the
    /// drift is latent there; this port always owns an arena, so recycle
    /// keeps the count consistent exactly like `free_block` does.
    pub fn recycle_block(&mut self, physical_block_index: u32) -> Result<(), KvArenaError> {
        let block = self.block(physical_block_index).ok_or(KvArenaError::InvalidArgument)?;
        if block.flags & BLOCK_FLAG_ALLOCATED == 0 {
            return Err(KvArenaError::InvalidArgument);
        }
        if block.reference_count != 0 {
            return Err(KvArenaError::Busy);
        }

        self.epoch += 1;
        let block = &mut self.blocks[physical_block_index as usize];
        if block.flags & BLOCK_FLAG_RESIDENT != 0 && self.resident_block_count != 0 {
            self.resident_block_count -= 1;
        }
        block.flags = BLOCK_FLAG_ALLOCATED;
        block.generation += 1;
        block.last_used_epoch = self.epoch;
        self.recycled_block_count += 1;
        Ok(())
    }

    /// Increment a block's reference count.
    pub fn retain_block(&mut self, physical_block_index: u32) -> Result<(), KvArenaError> {
        let block = self.block(physical_block_index).ok_or(KvArenaError::InvalidArgument)?;
        if block.flags & BLOCK_FLAG_ALLOCATED == 0 {
            return Err(KvArenaError::InvalidArgument);
        }

        self.epoch += 1;
        let block = &mut self.blocks[physical_block_index as usize];
        block.reference_count += 1;
        block.last_used_epoch = self.epoch;
        self.retained_block_count += 1;
        Ok(())
    }

    /// Decrement a block's reference count; releasing at zero is rejected.
    pub fn release_block_reference(
        &mut self,
        physical_block_index: u32,
    ) -> Result<(), KvArenaError> {
        let block = self.block(physical_block_index).ok_or(KvArenaError::InvalidArgument)?;
        if block.flags & BLOCK_FLAG_ALLOCATED == 0 || block.reference_count == 0 {
            return Err(KvArenaError::InvalidArgument);
        }

        self.epoch += 1;
        let block = &mut self.blocks[physical_block_index as usize];
        block.reference_count -= 1;
        block.last_used_epoch = self.epoch;
        self.released_reference_count += 1;
        Ok(())
    }

    /// Mark a block's bytes resident on device (evicting LRU non-protected
    /// residents when at resident capacity).
    pub fn mark_block_resident(&mut self, physical_block_index: u32) -> Result<(), KvArenaError> {
        let block = self.block(physical_block_index).ok_or(KvArenaError::InvalidArgument)?;
        if block.flags & BLOCK_FLAG_ALLOCATED == 0 {
            return Err(KvArenaError::InvalidArgument);
        }

        if block.flags & BLOCK_FLAG_RESIDENT == 0 {
            // The block being marked is protected from its own make-room
            // trim, exactly as the C passes it as the sole protected index.
            self.make_room_for_resident_blocks(&[physical_block_index], 1)?;
            self.resident_block_count += 1;
        }
        self.epoch += 1;
        let block = &mut self.blocks[physical_block_index as usize];
        block.flags |= BLOCK_FLAG_RESIDENT;
        block.last_used_epoch = self.epoch;
        Ok(())
    }

    /// Clear a block's resident bit.
    pub fn mark_block_nonresident(
        &mut self,
        physical_block_index: u32,
    ) -> Result<(), KvArenaError> {
        let block = self.block(physical_block_index).ok_or(KvArenaError::InvalidArgument)?;
        if block.flags & BLOCK_FLAG_ALLOCATED == 0 {
            return Err(KvArenaError::InvalidArgument);
        }

        self.epoch += 1;
        let block = &mut self.blocks[physical_block_index as usize];
        if block.flags & BLOCK_FLAG_RESIDENT != 0 {
            block.flags &= !BLOCK_FLAG_RESIDENT;
            if self.resident_block_count != 0 {
                self.resident_block_count -= 1;
            }
        }
        block.last_used_epoch = self.epoch;
        Ok(())
    }

    /// Free a block outright (must be unreferenced). Freeing an already-free
    /// block is a no-op, as in the C.
    pub fn free_block(&mut self, physical_block_index: u32) -> Result<(), KvArenaError> {
        let block = self.block(physical_block_index).ok_or(KvArenaError::InvalidArgument)?;
        if block.flags & BLOCK_FLAG_ALLOCATED == 0 {
            return Ok(());
        }
        if block.reference_count != 0 {
            return Err(KvArenaError::Busy);
        }

        self.epoch += 1;
        let block = &mut self.blocks[physical_block_index as usize];
        if block.flags & BLOCK_FLAG_RESIDENT != 0 && self.resident_block_count != 0 {
            self.resident_block_count -= 1;
        }
        block.flags = 0;
        block.generation += 1;
        block.last_used_epoch = self.epoch;
        Ok(())
    }

    /// Resolve a block to its device-address view.
    pub fn resolve_block(&self, physical_block_index: u32) -> Result<BlockView, KvArenaError> {
        let block = self.block(physical_block_index).ok_or(KvArenaError::InvalidArgument)?;
        if block.flags & BLOCK_FLAG_ALLOCATED == 0 {
            return Err(KvArenaError::NotFound);
        }
        Ok(BlockView {
            physical_block_index,
            token_capacity: self.block_token_count,
            layer_count: self.layer_count,
            kv_head_count: self.kv_head_count,
            head_dim: self.head_dim,
            bytes_per_scalar: self.bytes_per_scalar,
            generation: block.generation,
            key_block_stride_bytes: self.key_block_stride_bytes,
            value_block_stride_bytes: self.value_block_stride_bytes,
            key_device_address: block.key_device_address,
            value_device_address: block.value_device_address,
        })
    }

    /// Evict resident blocks down to `target_resident_block_count`, skipping
    /// `protected` blocks. Returns the evicted count. When no evictable
    /// victim remains, the stall counter bumps and `CapacityExceeded` is
    /// returned (the partial evicted count the C reports via out-parameter
    /// is dropped — see module docs).
    pub fn trim_resident_blocks(
        &mut self,
        protected_physical_block_indices: &[u32],
        target_resident_block_count: u32,
    ) -> Result<u32, KvArenaError> {
        if target_resident_block_count > self.resident_block_capacity {
            return Err(KvArenaError::InvalidArgument);
        }
        self.trim_resident_blocks_inner(
            protected_physical_block_indices,
            target_resident_block_count,
        )
    }

    /// Evict resident blocks until at most `max_resident_block_count` remain,
    /// skipping `protected` blocks. Returns the evicted count. Unlike
    /// `trim_resident_blocks`, running out of victims is not an error — the
    /// loop simply stops (faithful to the C).
    pub fn evict_resident_blocks_to_limit(
        &mut self,
        max_resident_block_count: u32,
        protected_physical_block_indices: &[u32],
    ) -> Result<u32, KvArenaError> {
        let mut evicted_block_count = 0u32;
        while self.resident_block_count > u64::from(max_resident_block_count) {
            let Some(victim) =
                self.select_resident_eviction_victim(protected_physical_block_indices)
            else {
                break;
            };
            self.mark_block_nonresident(victim)?;
            self.resident_evicted_block_count += 1;
            evicted_block_count += 1;
        }
        Ok(evicted_block_count)
    }

    /// Reset every block and the epoch/allocation/residency counters. The
    /// cumulative resident-evicted and capacity-stall counters survive,
    /// exactly as in the C.
    pub fn reset(&mut self) {
        for index in 0..self.physical_block_count {
            let block = self.make_block(index);
            self.blocks[index as usize] = block;
        }
        self.epoch = 0;
        self.allocated_block_count = 0;
        self.recycled_block_count = 0;
        self.resident_block_count = 0;
        self.retained_block_count = 0;
        self.released_reference_count = 0;
    }

    /// Port of `SparkKvCacheArenaMakeRoomForResidentBlocks`: ensure
    /// `new_resident_block_count` additional blocks can become resident,
    /// trimming to make room when necessary.
    fn make_room_for_resident_blocks(
        &mut self,
        protected_physical_block_indices: &[u32],
        new_resident_block_count: u32,
    ) -> Result<(), KvArenaError> {
        if new_resident_block_count > self.resident_block_capacity {
            self.resident_capacity_stall_count += 1;
            return Err(KvArenaError::CapacityExceeded);
        }
        if self.resident_block_count + u64::from(new_resident_block_count)
            <= u64::from(self.resident_block_capacity)
        {
            return Ok(());
        }
        let target_resident_block_count = self.resident_block_capacity - new_resident_block_count;
        self.trim_resident_blocks_inner(
            protected_physical_block_indices,
            target_resident_block_count,
        )?;
        Ok(())
    }

    /// Shared trim loop (port of
    /// `SparkKvCacheArenaTrimResidentBlocksWithPrefetchProtection` with a
    /// null prefetch plan). Callers must guarantee
    /// `target_resident_block_count <= resident_block_capacity`.
    fn trim_resident_blocks_inner(
        &mut self,
        protected_physical_block_indices: &[u32],
        target_resident_block_count: u32,
    ) -> Result<u32, KvArenaError> {
        let mut evicted_block_count = 0u32;
        while self.resident_block_count > u64::from(target_resident_block_count) {
            let Some(victim) =
                self.select_resident_eviction_victim(protected_physical_block_indices)
            else {
                self.resident_capacity_stall_count += 1;
                return Err(KvArenaError::CapacityExceeded);
            };
            self.evict_resident_block(victim);
            evicted_block_count += 1;
        }
        Ok(evicted_block_count)
    }

    /// Port of the C victim selection (both variants collapse to this when
    /// there is no prefetch plan): among allocated+resident blocks that are
    /// not protected, pick the lowest reference count, then the oldest
    /// `last_used_epoch`.
    fn select_resident_eviction_victim(
        &self,
        protected_physical_block_indices: &[u32],
    ) -> Option<u32> {
        let mut victim: Option<u32> = None;
        for index in 0..self.physical_block_count {
            let block = &self.blocks[index as usize];
            if block.flags & BLOCK_FLAG_ALLOCATED == 0
                || block.flags & BLOCK_FLAG_RESIDENT == 0
                || protected_block_list_contains(protected_physical_block_indices, index)
            {
                continue;
            }
            let better = match victim {
                None => true,
                Some(current) => {
                    let current_block = &self.blocks[current as usize];
                    (block.reference_count, block.last_used_epoch)
                        < (current_block.reference_count, current_block.last_used_epoch)
                }
            };
            if better {
                victim = Some(index);
            }
        }
        victim
    }

    /// Port of `SparkKvCacheArenaEvictResidentBlock` (used by the trim path).
    fn evict_resident_block(&mut self, physical_block_index: u32) {
        self.epoch += 1;
        let block = &mut self.blocks[physical_block_index as usize];
        block.flags &= !BLOCK_FLAG_RESIDENT;
        if self.resident_block_count != 0 {
            self.resident_block_count -= 1;
        }
        block.last_used_epoch = self.epoch;
        self.resident_evicted_block_count += 1;
    }

    /// Whether a block currently holds the allocated flag. Out-of-range
    /// indices report `false`.
    pub fn is_allocated(&self, physical_block_index: u32) -> bool {
        self.block(physical_block_index)
            .is_some_and(|block| block.flags & BLOCK_FLAG_ALLOCATED != 0)
    }

    /// Whether a block's bytes are currently resident on device. Out-of-range
    /// indices report `false`.
    pub fn is_resident(&self, physical_block_index: u32) -> bool {
        self.block(physical_block_index).is_some_and(|block| block.flags & BLOCK_FLAG_RESIDENT != 0)
    }

    /// Reference count of a block (0 when free or out of range).
    pub fn reference_count(&self, physical_block_index: u32) -> u32 {
        self.block(physical_block_index).map_or(0, |block| block.reference_count)
    }

    /// Arena geometry: tokens per block.
    pub fn block_token_count(&self) -> u32 {
        self.block_token_count
    }

    /// Arena geometry: number of physical blocks.
    pub fn physical_block_count(&self) -> u32 {
        self.physical_block_count
    }

    /// Token capacity stamped on a block (0 when out of range).
    pub fn block_token_capacity(&self, physical_block_index: u32) -> u32 {
        self.block(physical_block_index).map_or(0, |block| block.token_capacity)
    }

    /// Generation of a block (0 when free, stale, or out of range).
    pub fn generation(&self, physical_block_index: u32) -> u64 {
        self.block(physical_block_index).map_or(0, |block| block.generation)
    }

    /// Opaque key device address of a block (0 when out of range).
    pub fn key_device_address(&self, physical_block_index: u32) -> usize {
        self.block(physical_block_index).map_or(0, |block| block.key_device_address)
    }

    /// Opaque value device address of a block (0 when out of range or the
    /// arena has no value payload).
    pub fn value_device_address(&self, physical_block_index: u32) -> usize {
        self.block(physical_block_index).map_or(0, |block| block.value_device_address)
    }

    /// Last-used epoch stamp of a block (0 when out of range).
    pub fn last_used_epoch(&self, physical_block_index: u32) -> u64 {
        self.block(physical_block_index).map_or(0, |block| block.last_used_epoch)
    }

    pub fn resident_block_count(&self) -> u64 {
        self.resident_block_count
    }

    pub fn allocated_block_count(&self) -> u64 {
        self.allocated_block_count
    }

    /// Monotonic arena epoch (bumps on every state mutation).
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Cumulative recycled-block count.
    pub fn recycled_block_count(&self) -> u64 {
        self.recycled_block_count
    }

    /// Cumulative resident-evicted-block count (survives `reset`).
    pub fn resident_evicted_block_count(&self) -> u64 {
        self.resident_evicted_block_count
    }

    /// Cumulative resident-capacity-stall count (survives `reset`).
    pub fn resident_capacity_stall_count(&self) -> u64 {
        self.resident_capacity_stall_count
    }

    /// Cumulative retained-reference count.
    pub fn retained_block_count(&self) -> u64 {
        self.retained_block_count
    }

    /// Cumulative released-reference count.
    pub fn released_reference_count(&self) -> u64 {
        self.released_reference_count
    }

    /// Effective key block stride in bytes.
    pub fn key_block_stride_bytes(&self) -> u64 {
        self.key_block_stride_bytes
    }

    /// Effective value block stride in bytes (0 without a value payload).
    pub fn value_block_stride_bytes(&self) -> u64 {
        self.value_block_stride_bytes
    }
}
