//! Tests for `spark_core::kv_arena` — ports of the arena-relevant scenarios
//! from `tests/test_glm52_kv_cache.c` (prefetch-plan, async-backend,
//! capacity-estimator, and JIT-budget scenarios are out of scope for this
//! port). Device addresses are usize sentinels, as in the C host tests.

use spark_core::kv_arena::{
    KvArena, KvArenaConfig, KvArenaError, BLOCK_FLAG_ALLOCATED, BLOCK_FLAG_RESIDENT,
};

const BLOCK_TOKENS: u32 = 16;
const LAYER_COUNT: u32 = 78; // SPARK_GLM52_MODEL_LAYER_COUNT
const KV_HEAD_COUNT: u32 = 8;
const HEAD_DIM: u32 = 128;
const BYTES_PER_SCALAR: u32 = 2;
const KEY_BASE: usize = 0x1_0000_0000;
const VALUE_BASE: usize = 0x2_0000_0000;

fn expected_stride() -> u64 {
    u64::from(BLOCK_TOKENS)
        * u64::from(LAYER_COUNT)
        * u64::from(KV_HEAD_COUNT)
        * u64::from(HEAD_DIM)
        * u64::from(BYTES_PER_SCALAR)
}

/// Port of `SparkTestInitializeKvCacheArena`.
fn test_config(physical_block_count: u32) -> KvArenaConfig {
    KvArenaConfig {
        physical_block_count,
        block_token_count: BLOCK_TOKENS,
        resident_block_capacity: 0,
        layer_count: LAYER_COUNT,
        kv_head_count: KV_HEAD_COUNT,
        head_dim: HEAD_DIM,
        bytes_per_scalar: BYTES_PER_SCALAR,
        key_block_stride_bytes: 0,
        value_block_stride_bytes: 0,
        key_device_base: KEY_BASE,
        value_device_base: VALUE_BASE,
    }
}

/// Port of `SparkTestInitializeKvCacheArenaWithResidentCapacity`.
fn test_config_with_resident_capacity(
    physical_block_count: u32,
    resident_block_capacity: u32,
) -> KvArenaConfig {
    KvArenaConfig { resident_block_capacity, ..test_config(physical_block_count) }
}

fn arena(physical_block_count: u32) -> KvArena {
    KvArena::new(&test_config(physical_block_count)).unwrap()
}

/// Port of `SparkTestKvCacheAllocatesResidentDeviceBlocks`.
#[test]
fn allocates_resident_device_blocks() {
    let mut arena = arena(4);
    let stride = expected_stride();
    assert_eq!(arena.key_block_stride_bytes(), stride);
    assert_eq!(arena.value_block_stride_bytes(), stride);

    let first = arena.acquire_block().unwrap();
    assert_eq!(first, 0);
    // Acquire resets the reference count to 0; retain takes the reference.
    assert_eq!(arena.reference_count(first), 0);
    arena.retain_block(first).unwrap();
    assert_eq!(arena.reference_count(first), 1);
    arena.mark_block_resident(first).unwrap();
    assert!(arena.is_resident(first));

    let view = arena.resolve_block(first).unwrap();
    assert_eq!(view.key_device_address, KEY_BASE);
    assert_eq!(view.value_device_address, VALUE_BASE);
    assert_eq!(view.key_block_stride_bytes, stride);
    assert_eq!(view.value_block_stride_bytes, stride);
    assert_eq!(view.token_capacity, BLOCK_TOKENS);
    assert_eq!(view.layer_count, LAYER_COUNT);
    assert_eq!(view.kv_head_count, KV_HEAD_COUNT);
    assert_eq!(view.head_dim, HEAD_DIM);
    assert_eq!(view.bytes_per_scalar, BYTES_PER_SCALAR);
    assert_eq!(view.generation, 1);

    let second = arena.acquire_block().unwrap();
    assert_eq!(second, 1);
    let view = arena.resolve_block(second).unwrap();
    assert_eq!(view.key_device_address, KEY_BASE + stride as usize);
    assert_eq!(view.value_device_address, VALUE_BASE + stride as usize);

    assert_eq!(arena.allocated_block_count(), 2);
    assert_eq!(arena.retained_block_count(), 1);
}

/// Port of the non-prefetch half of `SparkTestKvCacheCanEvictResidentOwnedBlocks`.
#[test]
fn resident_mark_and_unmark_cycle() {
    let mut arena = arena(2);
    let index = arena.acquire_block().unwrap();
    arena.retain_block(index).unwrap();
    arena.mark_block_resident(index).unwrap();
    assert_eq!(arena.resident_block_count(), 1);

    arena.mark_block_nonresident(index).unwrap();
    // Residency is independent of ownership: the reference survives.
    assert_eq!(arena.reference_count(index), 1);
    assert!(arena.is_allocated(index));
    assert!(!arena.is_resident(index));
    assert_eq!(arena.resident_block_count(), 0);
}

/// Port of `SparkTestKvCacheResidentEvictionRespectsProtectedBlocks`. Note
/// the semantics demonstrated here: referenced blocks (reference_count == 1)
/// ARE evicted from residency — only the protected list is skipped.
#[test]
fn resident_eviction_respects_protected_blocks() {
    let mut arena = arena(3);
    let mut indices = [0u32; 3];
    for (i, slot) in indices.iter_mut().enumerate() {
        *slot = arena.acquire_block().unwrap();
        assert_eq!(*slot, i as u32);
        arena.retain_block(*slot).unwrap();
        arena.mark_block_resident(*slot).unwrap();
    }
    assert_eq!(arena.resident_block_count(), 3);

    let evicted = arena.evict_resident_blocks_to_limit(1, &[indices[0]]).unwrap();
    assert_eq!(evicted, 2);
    assert_eq!(arena.resident_block_count(), 1);
    assert!(arena.is_resident(indices[0]));
    assert!(!arena.is_resident(indices[1]));
    assert!(!arena.is_resident(indices[2]));
    assert_eq!(arena.resident_evicted_block_count(), 2);
}

/// Port of `SparkTestKvCacheEvictsUnprotectedResidentBlocksToLimit`.
#[test]
fn evicts_unprotected_resident_blocks_to_limit() {
    let mut arena = arena(4);
    let first = arena.acquire_block().unwrap();
    let second = arena.acquire_block().unwrap();
    let third = arena.acquire_block().unwrap();
    arena.mark_block_resident(first).unwrap();
    arena.mark_block_resident(second).unwrap();
    arena.mark_block_resident(third).unwrap();
    assert_eq!(arena.resident_block_count(), 3);

    let evicted = arena.evict_resident_blocks_to_limit(1, &[first]).unwrap();
    assert_eq!(evicted, 2);
    assert_eq!(arena.resident_block_count(), 1);
    assert!(arena.is_resident(first));
    assert!(!arena.is_resident(second));
    assert!(!arena.is_resident(third));
}

/// When every resident block is protected, `evict_resident_blocks_to_limit`
/// stops quietly (no error, partial progress) — unlike `trim_resident_blocks`.
#[test]
fn evict_to_limit_with_all_protected_is_not_an_error() {
    let mut arena = arena(2);
    let first = arena.acquire_block().unwrap();
    let second = arena.acquire_block().unwrap();
    arena.mark_block_resident(first).unwrap();
    arena.mark_block_resident(second).unwrap();

    let evicted = arena.evict_resident_blocks_to_limit(0, &[first, second]).unwrap();
    assert_eq!(evicted, 0);
    assert_eq!(arena.resident_block_count(), 2);
}

/// Port of `SparkTestKvCacheRejectsRecyclingRetainedBlocks`.
#[test]
fn rejects_recycling_retained_blocks() {
    let mut arena = arena(2);
    let index = arena.acquire_block().unwrap();
    arena.retain_block(index).unwrap();
    assert_eq!(arena.recycle_block(index), Err(KvArenaError::Busy));
    arena.release_block_reference(index).unwrap();

    let generation = arena.generation(index);
    arena.recycle_block(index).unwrap();
    assert_eq!(arena.generation(index), generation + 1);
    // Recycle re-arms in place: still allocated.
    assert!(arena.is_allocated(index));
    assert_eq!(arena.recycled_block_count(), 1);

    arena.free_block(index).unwrap();
    assert!(!arena.is_allocated(index));
}

/// Double-release (release at reference_count == 0) is rejected, as is
/// releasing an unallocated block.
#[test]
fn double_release_is_rejected() {
    let mut arena = arena(1);
    let index = arena.acquire_block().unwrap();
    // Acquire leaves reference_count at 0: the first release must fail.
    assert_eq!(arena.release_block_reference(index), Err(KvArenaError::InvalidArgument));

    arena.retain_block(index).unwrap();
    arena.retain_block(index).unwrap();
    arena.release_block_reference(index).unwrap();
    arena.release_block_reference(index).unwrap();
    assert_eq!(arena.reference_count(index), 0);
    assert_eq!(arena.release_block_reference(index), Err(KvArenaError::InvalidArgument));
    assert_eq!(arena.released_reference_count(), 2);

    arena.free_block(index).unwrap();
    assert_eq!(arena.release_block_reference(index), Err(KvArenaError::InvalidArgument));
}

/// Acquire hands out every block, then reports capacity; freeing makes the
/// same index available again.
#[test]
fn acquire_exhaustion_and_reuse() {
    let mut arena = arena(2);
    assert_eq!(arena.acquire_block().unwrap(), 0);
    assert_eq!(arena.acquire_block().unwrap(), 1);
    assert_eq!(arena.acquire_block(), Err(KvArenaError::CapacityExceeded));

    arena.free_block(0).unwrap();
    assert_eq!(arena.acquire_block().unwrap(), 0);
    assert_eq!(arena.acquire_block(), Err(KvArenaError::CapacityExceeded));
}

/// Port of the capacity side of
/// `SparkTestKvCachePrefetchPlanResidencyIsAtomicUnderCapacity` and
/// `SparkTestKvCachePrefetchPlanEvictsColdBlocksAsGroup`, driven through the
/// mark-resident path: at resident capacity, marking a new block evicts the
/// coldest unprotected resident.
#[test]
fn mark_resident_at_capacity_evicts_cold_block() {
    let mut arena = KvArena::new(&test_config_with_resident_capacity(3, 2)).unwrap();
    let cold = arena.acquire_block().unwrap();
    let first = arena.acquire_block().unwrap();
    let second = arena.acquire_block().unwrap();
    arena.mark_block_resident(cold).unwrap();
    assert_eq!(arena.resident_block_count(), 1);

    arena.mark_block_resident(first).unwrap();
    assert_eq!(arena.resident_block_count(), 2);
    assert!(arena.is_resident(cold));

    arena.mark_block_resident(second).unwrap();
    assert_eq!(arena.resident_block_count(), 2);
    assert!(!arena.is_resident(cold));
    assert!(arena.is_resident(first));
    assert!(arena.is_resident(second));
    assert_eq!(arena.resident_evicted_block_count(), 1);
}

/// A resident-capacity stall: trim to a target no unprotected victim can
/// reach bumps the stall counter and reports `CapacityExceeded`, leaving the
/// protected residents in place.
#[test]
fn trim_with_all_residents_protected_stalls() {
    let mut arena = KvArena::new(&test_config_with_resident_capacity(2, 2)).unwrap();
    let first = arena.acquire_block().unwrap();
    let second = arena.acquire_block().unwrap();
    arena.mark_block_resident(first).unwrap();
    arena.mark_block_resident(second).unwrap();

    assert_eq!(
        arena.trim_resident_blocks(&[first, second], 0),
        Err(KvArenaError::CapacityExceeded)
    );
    assert_eq!(arena.resident_capacity_stall_count(), 1);
    assert!(arena.is_resident(first));
    assert!(arena.is_resident(second));
    assert_eq!(arena.resident_block_count(), 2);
}

/// Trim-to-target honours LRU (`last_used_epoch`): the oldest residents go
/// first, and a retained block outlives unreferenced ones.
#[test]
fn trim_to_target_evicts_coldest_first() {
    let mut arena = arena(4);
    let mut indices = [0u32; 4];
    for slot in indices.iter_mut() {
        *slot = arena.acquire_block().unwrap();
        arena.mark_block_resident(*slot).unwrap();
    }
    assert_eq!(arena.resident_block_count(), 4);

    // LRU order: 0, 1, 2, 3 (marking stamped increasing epochs).
    let evicted = arena.trim_resident_blocks(&[], 1).unwrap();
    assert_eq!(evicted, 3);
    assert_eq!(arena.resident_block_count(), 1);
    assert!(!arena.is_resident(indices[0]));
    assert!(!arena.is_resident(indices[1]));
    assert!(!arena.is_resident(indices[2]));
    assert!(arena.is_resident(indices[3]));

    // Reference count dominates epoch: re-resident everything, retain
    // indices[0], and trim again — the unreferenced blocks go first.
    for slot in &indices {
        arena.mark_block_resident(*slot).unwrap();
    }
    arena.retain_block(indices[0]).unwrap();
    let evicted = arena.trim_resident_blocks(&[indices[0]], 1).unwrap();
    assert_eq!(evicted, 3);
    assert!(arena.is_resident(indices[0]));
    assert!(!arena.is_resident(indices[1]));
    assert!(!arena.is_resident(indices[2]));
    assert!(!arena.is_resident(indices[3]));
}

/// A trim target above the resident capacity is rejected without mutating.
#[test]
fn trim_target_above_capacity_is_invalid() {
    let mut arena = arena(2);
    let index = arena.acquire_block().unwrap();
    arena.mark_block_resident(index).unwrap();
    assert_eq!(arena.trim_resident_blocks(&[], 3), Err(KvArenaError::InvalidArgument));
    assert_eq!(arena.resident_block_count(), 1);
}

/// `free_block`: freeing a free block is a no-op; freeing a referenced block
/// is `Busy`; freeing a resident unreferenced block drops the resident count.
#[test]
fn free_block_semantics() {
    let mut arena = arena(2);
    // Already free: no-op success, as in the C.
    arena.free_block(0).unwrap();

    let index = arena.acquire_block().unwrap();
    arena.retain_block(index).unwrap();
    assert_eq!(arena.free_block(index), Err(KvArenaError::Busy));

    arena.release_block_reference(index).unwrap();
    arena.mark_block_resident(index).unwrap();
    assert_eq!(arena.resident_block_count(), 1);
    arena.free_block(index).unwrap();
    assert_eq!(arena.resident_block_count(), 0);
    assert!(!arena.is_allocated(index));
    assert_eq!(arena.resolve_block(index), Err(KvArenaError::NotFound));
}

/// Generation bumps on acquire, recycle, and free; reset returns blocks to
/// generation 0.
#[test]
fn generation_bumps_on_acquire_recycle_free() {
    let mut arena = arena(1);
    assert_eq!(arena.generation(0), 0);
    let index = arena.acquire_block().unwrap();
    assert_eq!(arena.generation(index), 1);
    arena.recycle_block(index).unwrap();
    assert_eq!(arena.generation(index), 2);
    arena.free_block(index).unwrap();
    assert_eq!(arena.generation(index), 3);

    arena.reset();
    assert_eq!(arena.generation(0), 0);
    assert_eq!(arena.acquire_block().unwrap(), 0);
    assert_eq!(arena.generation(0), 1);
}

/// `reset` restores the pool and zeroes the epoch/allocation/residency
/// counters — but, faithful to the C, preserves the cumulative evicted and
/// stall counters.
#[test]
fn reset_restores_pool_and_preserves_cumulative_counters() {
    let mut arena = KvArena::new(&test_config_with_resident_capacity(2, 2)).unwrap();
    let first = arena.acquire_block().unwrap();
    let second = arena.acquire_block().unwrap();
    arena.mark_block_resident(first).unwrap();
    arena.mark_block_resident(second).unwrap();
    arena.retain_block(first).unwrap();
    assert_eq!(arena.trim_resident_blocks(&[], 1).unwrap(), 1);
    assert_eq!(arena.trim_resident_blocks(&[first], 0), Err(KvArenaError::CapacityExceeded));
    assert_eq!(arena.resident_evicted_block_count(), 1);
    assert_eq!(arena.resident_capacity_stall_count(), 1);

    arena.reset();
    assert!(!arena.is_allocated(first));
    assert!(!arena.is_allocated(second));
    assert_eq!(arena.epoch(), 0);
    assert_eq!(arena.allocated_block_count(), 0);
    assert_eq!(arena.resident_block_count(), 0);
    assert_eq!(arena.retained_block_count(), 0);
    assert_eq!(arena.released_reference_count(), 0);
    // The C reset leaves these two cumulative counters alone.
    assert_eq!(arena.resident_evicted_block_count(), 1);
    assert_eq!(arena.resident_capacity_stall_count(), 1);
}

/// Port of the arena half of `SparkTestKvCacheSupportsMlaPrimaryOnlyArenaAndPrefetch`:
/// a key-only arena carries no value payload.
#[test]
fn key_only_arena_has_no_value_payload() {
    let config = KvArenaConfig {
        physical_block_count: 2,
        block_token_count: BLOCK_TOKENS,
        resident_block_capacity: 0,
        layer_count: 1,
        kv_head_count: 1,
        head_dim: 32,
        bytes_per_scalar: BYTES_PER_SCALAR,
        key_block_stride_bytes: 64,
        value_block_stride_bytes: 0,
        key_device_base: KEY_BASE,
        value_device_base: 0,
    };
    let mut arena = KvArena::new(&config).unwrap();
    assert_eq!(arena.value_block_stride_bytes(), 0);

    let index = arena.acquire_block().unwrap();
    assert_eq!(arena.value_device_address(index), 0);
    let view = arena.resolve_block(index).unwrap();
    assert_eq!(view.value_device_address, 0);
    assert_eq!(view.value_block_stride_bytes, 0);
    assert_eq!(view.key_device_address, KEY_BASE);
    // Explicit key stride overrides the geometry default.
    assert_eq!(view.key_block_stride_bytes, 64);
}

/// Invalid configurations are rejected, mirroring
/// `SparkKvCacheConfigurationIsValid`.
#[test]
fn invalid_configurations_are_rejected() {
    let cases = [
        KvArenaConfig { physical_block_count: 0, ..test_config(4) },
        KvArenaConfig { block_token_count: 0, ..test_config(4) },
        KvArenaConfig {
            block_token_count: 257, // > MAX_BLOCK_TOKENS
            ..test_config(4)
        },
        KvArenaConfig {
            resident_block_capacity: 5, // > physical_block_count
            ..test_config(4)
        },
        KvArenaConfig { layer_count: 0, ..test_config(4) },
        KvArenaConfig { kv_head_count: 0, ..test_config(4) },
        KvArenaConfig { head_dim: 0, ..test_config(4) },
        KvArenaConfig { bytes_per_scalar: 0, ..test_config(4) },
        KvArenaConfig { key_device_base: 0, ..test_config(4) },
        KvArenaConfig {
            value_device_base: 0,
            value_block_stride_bytes: 64, // stride without a base
            ..test_config(4)
        },
    ];
    for config in &cases {
        assert_eq!(KvArena::new(config).unwrap_err(), KvArenaError::InvalidArgument);
    }
}

/// Out-of-range and wrong-state errors across the entry points, plus the
/// out-of-range accessor contract (false/0, never a panic).
#[test]
fn out_of_range_and_wrong_state_errors() {
    let mut arena = arena(2);
    assert_eq!(arena.resolve_block(0), Err(KvArenaError::NotFound));
    assert_eq!(arena.resolve_block(2), Err(KvArenaError::InvalidArgument));
    assert_eq!(arena.retain_block(0), Err(KvArenaError::InvalidArgument));
    assert_eq!(arena.retain_block(2), Err(KvArenaError::InvalidArgument));
    assert_eq!(arena.mark_block_resident(2), Err(KvArenaError::InvalidArgument));
    assert_eq!(arena.mark_block_nonresident(2), Err(KvArenaError::InvalidArgument));
    assert_eq!(arena.free_block(2), Err(KvArenaError::InvalidArgument));
    assert_eq!(arena.recycle_block(2), Err(KvArenaError::InvalidArgument));

    assert!(!arena.is_allocated(2));
    assert!(!arena.is_resident(2));
    assert_eq!(arena.reference_count(2), 0);
    assert_eq!(arena.block_token_capacity(2), 0);
    assert_eq!(arena.generation(2), 0);
    assert_eq!(arena.key_device_address(2), 0);
    assert_eq!(arena.value_device_address(2), 0);
    assert_eq!(arena.last_used_epoch(2), 0);
}

/// The epoch bumps on every state mutation and `last_used_epoch` tracks it.
#[test]
fn epoch_tracks_mutations() {
    let mut arena = arena(1);
    assert_eq!(arena.epoch(), 0);
    let index = arena.acquire_block().unwrap();
    assert_eq!(arena.epoch(), 1);
    assert_eq!(arena.last_used_epoch(index), 1);
    arena.retain_block(index).unwrap();
    assert_eq!(arena.last_used_epoch(index), 2);
    arena.mark_block_resident(index).unwrap();
    assert_eq!(arena.last_used_epoch(index), 3);
}

/// Flag constants mirror the C bit values.
#[test]
fn flag_constants_match_c() {
    assert_eq!(BLOCK_FLAG_ALLOCATED, 0x1);
    assert_eq!(BLOCK_FLAG_RESIDENT, 0x2);
    assert_eq!(spark_core::kv_arena::NO_BLOCK, u32::MAX);
}
