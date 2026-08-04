//! Port of `tests/test_glm52_prefix_cache.c` (GPU-free: device addresses are
//! `usize` sentinels; the cache always owns a real `KvArena`).

use spark_core::kv_arena::{KvArena, KvArenaConfig};
use spark_core::prefix_cache::{
    hash_block, hash_prompt_tokens, PrefixCache, PrefixCacheConfig, PrefixCacheError,
    EMPTY_PARENT_HASH, ENTRY_FLAG_LIVE_ONLY, ENTRY_FLAG_REUSABLE, ENTRY_FLAG_VALID,
    NO_PHYSICAL_BLOCK,
};

fn make_cache(entry_count: u32, binding_count: u32, block_token_count: u32) -> PrefixCache {
    make_cache_with_arena_blocks(entry_count, binding_count, block_token_count, entry_count * 2)
}

fn make_cache_with_arena_blocks(
    entry_count: u32,
    binding_count: u32,
    block_token_count: u32,
    arena_block_count: u32,
) -> PrefixCache {
    let config = PrefixCacheConfig {
        block_token_count,
        entry_count,
        physical_block_count: entry_count,
        sequence_binding_count: binding_count,
    };
    // The arena is sized with headroom beyond the cache's physical block
    // count: a recycled block keeps its (now stale) contribution to
    // `resident_block_count` in both the C and the Rust arena — the resident
    // bit is cleared without decrementing the counter — so a resident
    // capacity equal to the block count would stall `mark_block_resident`
    // while every block is retained by a reservation. The stale contribution
    // accumulates per recycle generation, so eviction-heavy scenarios need
    // several generations of headroom. The C unit test avoids this entirely
    // by running without an arena.
    let arena_config = KvArenaConfig {
        physical_block_count: arena_block_count,
        block_token_count,
        resident_block_capacity: arena_block_count,
        layer_count: 1,
        kv_head_count: 1,
        head_dim: 1,
        bytes_per_scalar: 2,
        key_block_stride_bytes: 0,
        value_block_stride_bytes: 0,
        key_device_base: 0x1000_0000usize,
        value_device_base: 0,
    };
    let arena = KvArena::new(&arena_config).expect("arena");
    PrefixCache::new(&config, arena).expect("cache")
}

fn tokens(base: u32, count: u32) -> Vec<u32> {
    (0..count).map(|index| base + index).collect()
}

fn entry_index_for_physical_block(cache: &PrefixCache, physical_block_index: u32) -> Option<u32> {
    (0..cache.entry_count()).find(|&index| {
        cache.entry_view(index).is_some_and(|entry| {
            entry.flags & ENTRY_FLAG_VALID != 0
                && entry.physical_block_index == physical_block_index
        })
    })
}

// --- Hash parity -----------------------------------------------------------
// Reference values computed with an independent implementation of the exact C
// algorithm (FNV-1a mixing, same constants) from cache/prefix_cache.c.

#[test]
fn hash_block_matches_c_reference() {
    let block = tokens(1000, 4);
    assert_eq!(hash_block(&block, EMPTY_PARENT_HASH), 17187592518832544614);
    assert_eq!(hash_block(&[], EMPTY_PARENT_HASH), {
        // mix(EMPTY ^ FNV, 0)
        let mut h = EMPTY_PARENT_HASH ^ 1099511628211;
        h = h.wrapping_mul(1099511628211);
        h ^= h >> 32;
        h
    });
}

#[test]
fn hash_prompt_tokens_matches_c_reference() {
    let token_ids = tokens(1000, 16);
    let hash = hash_prompt_tokens(4, EMPTY_PARENT_HASH, &token_ids).expect("hash");
    assert_eq!(hash.prompt_hash, 11250298247205927079);
    assert_eq!(hash.block_count, 4);
    assert_eq!(hash.last_block_token_count, 4);
    assert_eq!(hash.hashed_token_count, 16);
    assert_eq!(hash.parent_hash, EMPTY_PARENT_HASH);

    let ids: Vec<u32> = (0..80u32).map(|i| 100000 + i * 37).collect();
    assert_eq!(
        hash_prompt_tokens(16, EMPTY_PARENT_HASH, &ids).expect("hash").prompt_hash,
        14918569915938236142
    );
    assert_eq!(
        hash_prompt_tokens(16, EMPTY_PARENT_HASH, &ids[..48]).expect("hash").prompt_hash,
        1825709821990993003
    );

    let empty = hash_prompt_tokens(4, EMPTY_PARENT_HASH, &[]).expect("hash");
    assert_eq!(empty.prompt_hash, EMPTY_PARENT_HASH);
    assert_eq!(empty.block_count, 0);
    assert_eq!(empty.last_block_token_count, 0);

    assert_eq!(
        hash_prompt_tokens(0, EMPTY_PARENT_HASH, &token_ids),
        Err(PrefixCacheError::InvalidArgument)
    );
    assert_eq!(
        hash_prompt_tokens(257, EMPTY_PARENT_HASH, &token_ids),
        Err(PrefixCacheError::InvalidArgument)
    );
}

// SparkTestPrefixCacheChainedBlockHashMatchesFullHash.
#[test]
fn chained_block_hash_matches_full_hash() {
    let token_ids: Vec<u32> = (0..80u32).map(|i| 100000 + i * 37).collect();
    let mut chained_hash = EMPTY_PARENT_HASH;
    let mut boundary = 16usize;
    while boundary <= 80 {
        let full = hash_prompt_tokens(16, EMPTY_PARENT_HASH, &token_ids[..boundary]).expect("full");
        let block =
            hash_prompt_tokens(16, chained_hash, &token_ids[boundary - 16..boundary]).expect("blk");
        chained_hash = block.prompt_hash;
        assert_eq!(chained_hash, full.prompt_hash);
        boundary += 16;
    }
}

// --- Cache behaviour (ported scenarios) ------------------------------------

// SparkTestPrefixCacheMatchesCommittedBlocks.
#[test]
fn matches_committed_blocks() {
    let mut cache = make_cache(8, 16, 4);
    let token_ids = tokens(1000, 16);

    let lookup = cache.lookup_prompt(7, &token_ids).expect("lookup");
    assert_eq!(lookup.matched_token_count, 0);
    assert_eq!(cache.miss_count, 1);

    let lookup = cache.commit_prompt(7, &token_ids).expect("commit");
    assert_eq!(lookup.matched_token_count, 16);
    assert_eq!(lookup.matched_block_count, 4);
    assert_eq!(cache.inserted_block_count, 4);
    assert_eq!(cache.acquired_block_count, 4);

    let lookup = cache.lookup_prompt(9, &token_ids).expect("lookup");
    assert_eq!(lookup.matched_token_count, 12);
    assert_eq!(lookup.matched_block_count, 3);
    assert_eq!(lookup.next_token_index, 12);
    assert_eq!(cache.hit_count, 1);
    assert_eq!(cache.acquired_block_count, 7);
}

// SparkTestPrefixCacheStopsAtChangedBlock.
#[test]
fn stops_at_changed_block() {
    let mut cache = make_cache(8, 16, 4);
    let token_ids = tokens(2000, 16);
    let mut changed = token_ids.clone();
    changed[6] = 9000;

    cache.commit_prompt(12, &token_ids).expect("commit");
    let lookup = cache.lookup_prompt(13, &changed).expect("lookup");
    assert_eq!(lookup.matched_token_count, 4);
    assert_eq!(lookup.matched_block_count, 1);
}

// SparkTestPrefixCacheTracksSequenceOwnership.
#[test]
fn tracks_sequence_ownership() {
    let mut cache = make_cache(4, 8, 4);
    let token_ids = tokens(1, 8);

    cache.commit_prompt(1, &token_ids).expect("commit");
    assert_eq!(cache.entry_view(0).expect("e0").reference_count, 1);
    assert_eq!(cache.entry_view(1).expect("e1").reference_count, 1);

    let lookup = cache.lookup_prompt(2, &token_ids).expect("lookup");
    assert_eq!(lookup.matched_token_count, 4);
    assert_eq!(cache.entry_view(0).expect("e0").reference_count, 2);
    assert_eq!(cache.entry_view(1).expect("e1").reference_count, 1);

    cache.release_sequence(2).expect("release 2");
    assert_eq!(cache.entry_view(0).expect("e0").reference_count, 1);
    assert_eq!(cache.entry_view(1).expect("e1").reference_count, 1);
    cache.release_sequence(1).expect("release 1");
    assert_eq!(cache.entry_view(0).expect("e0").reference_count, 0);
    assert_eq!(cache.entry_view(1).expect("e1").reference_count, 0);
    assert_eq!(cache.released_block_count, 3);
}

// SparkTestPrefixCacheEvictsReleasedBlocks.
#[test]
fn evicts_released_blocks() {
    let mut cache = make_cache_with_arena_blocks(2, 8, 4, 8);
    let first = tokens(1, 8);
    let second = tokens(9, 8);
    let third = tokens(17, 8);

    cache.commit_prompt(1, &first).expect("commit 1");
    assert_eq!(cache.commit_prompt(2, &second), Err(PrefixCacheError::CapacityExceeded));
    cache.release_sequence(1).expect("release 1");
    cache.commit_prompt(2, &second).expect("commit 2");
    cache.release_sequence(2).expect("release 2");
    cache.commit_prompt(3, &third).expect("commit 3");
    assert_ne!(cache.evicted_block_count, 0);
}

// SparkTestPrefixCacheRejectsBindingExhaustionWithoutLeakingRefs.
#[test]
fn rejects_binding_exhaustion_without_leaking_refs() {
    let mut cache = make_cache(4, 2, 4);
    let token_ids = tokens(100, 16);

    assert_eq!(cache.commit_prompt(1, &token_ids), Err(PrefixCacheError::CapacityExceeded));
    assert_eq!(cache.entry_view(0).expect("e0").reference_count, 0);
    assert_eq!(cache.entry_view(1).expect("e1").reference_count, 0);
    cache.release_sequence(1).expect("release 1");
    let lookup = cache.commit_prompt(2, &token_ids[..8]).expect("commit 2");
    assert_eq!(lookup.matched_token_count, 8);
}

// SparkTestPrefixCacheReservationOwnsPhysicalBlocksUntilCommitOrCancel.
#[test]
fn reservation_owns_physical_blocks_until_commit_or_cancel() {
    let mut cache = make_cache(8, 16, 4);
    let token_ids = tokens(3000, 12);

    let reservation = cache.reserve_prompt(101, &token_ids).expect("reserve");
    assert_eq!(reservation.reserved_token_count, 12);
    assert_eq!(reservation.physical_block_count, 3);
    assert_eq!(reservation.pending_physical_block_count, 3);
    assert!(reservation.physical_block_indices.iter().all(|&index| index != NO_PHYSICAL_BLOCK));

    let table = cache.build_physical_block_table(101, 12).expect("table");
    assert_eq!(table.len(), 3);

    let lookup = cache.probe_prompt(102, &token_ids).expect("probe");
    assert_eq!(lookup.matched_token_count, 0);

    cache.cancel_reservation(101, reservation.reservation_epoch).expect("cancel");
    assert_eq!(cache.build_physical_block_table(101, 12), Err(PrefixCacheError::NotFound));

    let reservation = cache.reserve_prompt(103, &token_ids).expect("reserve");
    cache.commit_reservation(103, reservation.reservation_epoch).expect("commit");
    let lookup = cache.probe_prompt(104, &token_ids).expect("probe");
    assert_eq!(lookup.matched_token_count, 8);
}

// SparkTestPrefixCacheLookaheadProtectionSkipsProtectedVictim.
#[test]
fn lookahead_protection_skips_protected_victim() {
    let mut cache = make_cache(2, 8, 4);
    let protected_prompt = [10u32, 11, 12, 13, 90, 91, 92, 93];
    let competing_prompt = [20u32, 21, 22, 23];
    let incoming_prompt = [30u32, 31, 32, 33];

    cache.commit_prompt(101, &protected_prompt[..4]).expect("commit protected");
    cache.release_sequence(101).expect("release 101");
    cache.commit_prompt(202, &competing_prompt).expect("commit competing");
    cache.release_sequence(202).expect("release 202");

    cache.reset_lookahead_protection().expect("reset lookahead");
    let protection = cache.protect_prompt_lookahead(&protected_prompt, 500).expect("protect");
    assert_eq!(protection.protected_token_count, 4);
    assert_eq!(protection.protected_block_count, 1);

    cache.commit_prompt(303, &incoming_prompt).expect("commit incoming");
    assert_eq!(cache.evicted_block_count, 1);

    let lookup = cache.probe_prompt(404, &protected_prompt).expect("probe");
    assert_eq!(lookup.matched_token_count, 4);
    let lookup = cache.probe_prompt(505, &competing_prompt).expect("probe");
    assert_eq!(lookup.matched_token_count, 0);
}

// SparkTestPrefixCacheExtendsLiveSequenceCapacityIdempotently.
#[test]
fn extends_live_sequence_capacity_idempotently() {
    let mut cache = make_cache(8, 16, 4);
    let prompt = [41u32, 42, 43];

    cache.commit_prompt(707, &prompt).expect("commit");
    let first_table = cache.build_physical_block_table(707, 3).expect("table");
    assert_eq!(first_table.len(), 1);

    cache.ensure_sequence_token_capacity(707, 10).expect("ensure 10");
    let first_table = cache.build_physical_block_table(707, 10).expect("table");
    assert_eq!(first_table.len(), 3);
    for &physical_block_index in &first_table {
        let entry_index =
            entry_index_for_physical_block(&cache, physical_block_index).expect("entry");
        let entry = cache.entry_view(entry_index).expect("view");
        assert_ne!(entry.flags & ENTRY_FLAG_LIVE_ONLY, 0);
        assert_eq!(entry.flags & ENTRY_FLAG_REUSABLE, 0);
        assert_eq!(entry.token_count, 4);
    }

    let inserted_block_count = cache.inserted_block_count;
    cache.ensure_sequence_token_capacity(707, 10).expect("ensure again");
    assert_eq!(cache.inserted_block_count, inserted_block_count);
    let second_table = cache.build_physical_block_table(707, 10).expect("table");
    assert_eq!(first_table, second_table);
}

// SparkTestPrefixCacheSequenceReservationDoesNotReuseContent.
#[test]
fn sequence_reservation_does_not_reuse_content() {
    let mut cache = make_cache(8, 16, 4);
    let token_ids = tokens(9000, 12);

    let first = cache.reserve_sequence_prompt(801, &token_ids[..8]).expect("reserve 801");
    assert_eq!(first.cached_physical_block_count, 0);
    cache.commit_reservation(801, first.reservation_epoch).expect("commit 801");

    let second = cache.reserve_sequence_prompt(802, &token_ids[..8]).expect("reserve 802");
    assert_eq!(second.cached_physical_block_count, 0);
    assert_eq!(second.reusable_token_count, 0);
    assert_ne!(first.physical_block_indices[0], second.physical_block_indices[0]);
    assert_ne!(first.physical_block_indices[1], second.physical_block_indices[1]);
    cache.commit_reservation(802, second.reservation_epoch).expect("commit 802");

    let lookup = cache.probe_prompt(803, &token_ids[..8]).expect("probe");
    assert_eq!(lookup.matched_token_count, 0);
    for &physical_block_index in &first.physical_block_indices[..2] {
        let entry_index =
            entry_index_for_physical_block(&cache, physical_block_index).expect("entry");
        assert_ne!(cache.entry_view(entry_index).expect("view").flags & ENTRY_FLAG_LIVE_ONLY, 0);
    }
}

// SparkTestPrefixCacheProbesShareOneWalk.
#[test]
fn probes_share_one_walk() {
    let mut cache = make_cache(8, 16, 4);
    let mut token_ids = tokens(4000, 16);

    let lookup = cache.commit_prompt(11, &token_ids).expect("commit");
    assert_eq!(lookup.matched_block_count, 4);

    // The final block is withheld from reuse — a prompt must keep at least
    // one token to generate from — so 16 tokens at block size 4 yield three.
    let tick_before = cache.tick;
    let probe = cache.probe_physical_block_table(&token_ids, 8).expect("probe");
    assert_eq!(probe.physical_block_indices.len(), 3);
    assert_eq!(probe.matched_token_count, 12);
    assert_eq!(cache.tick, tick_before + 3);

    let residency = cache.probe_reusable_prefix_residency(&token_ids).expect("residency");
    assert_eq!(residency.matched_token_count, 12);
    assert_eq!(residency.resident_block_count + residency.nonresident_block_count, 3);

    // A short output refuses rather than truncating, and the refusal must not
    // leave the chain half-walked for the next caller.
    assert_eq!(
        cache.probe_physical_block_table(&token_ids, 2),
        Err(PrefixCacheError::CapacityExceeded)
    );
    let probe = cache.probe_physical_block_table(&token_ids, 8).expect("probe");
    assert_eq!(probe.physical_block_indices.len(), 3);
    assert_eq!(probe.matched_token_count, 12);

    // A prompt that diverges after one block matches exactly one block.
    token_ids[4] = 9999;
    let probe = cache.probe_physical_block_table(&token_ids, 8).expect("probe");
    assert_eq!(probe.physical_block_indices.len(), 1);
    assert_eq!(probe.matched_token_count, 4);
}

// Shared-prefix second-sequence hit via content hash (cross-sequence reuse).
#[test]
fn shared_prefix_second_sequence_hits() {
    let mut cache = make_cache(8, 16, 4);
    let prefix = tokens(5000, 8);
    let mut fork_a = prefix.clone();
    fork_a.extend_from_slice(&[100, 101, 102, 103]);
    let mut fork_b = prefix;
    fork_b.extend_from_slice(&[200, 201, 202, 203]);

    cache.commit_prompt(1, &fork_a).expect("commit a");
    cache.release_sequence(1).expect("release a");

    // Sequence 2 reserves the fork: the shared prefix blocks must be counted
    // as cached/reused, not re-installed.
    let reservation = cache.reserve_prompt(2, &fork_b).expect("reserve b");
    assert_eq!(reservation.cached_physical_block_count, 2);
    assert_eq!(reservation.reusable_token_count, 8);
    assert_eq!(reservation.pending_physical_block_count, 1);
    cache.commit_reservation(2, reservation.reservation_epoch).expect("commit b");

    // Both sequences observe the same physical blocks for the shared prefix
    // after sequence 3 acquires it via lookup. The final block of a prompt is
    // withheld from reuse, so the lookup binds only the first two blocks.
    let lookup = cache.lookup_prompt(3, &fork_a).expect("lookup a");
    assert_eq!(lookup.matched_token_count, 8);
    let table_a = cache.build_physical_block_table(3, 8).expect("table a");
    let table_b = cache.build_physical_block_table(2, 12).expect("table b");
    assert_eq!(table_a[..2], table_b[..2]);
}

// Lookahead-protected blocks are skipped by reuse-scored trim.
#[test]
fn trim_by_reuse_score_skips_lookahead_protected() {
    let mut cache = make_cache(4, 16, 4);
    // Only the first block is committed; the 8-token prompt is what makes the
    // first block reachable as a reusable hit (a 4-token prompt has zero
    // reusable tokens — the final block is always withheld).
    let protected_prompt = [6000u32, 6001, 6002, 6003, 90, 91, 92, 93];
    let other_prompt = [7000u32, 7001, 7002, 7003, 80, 81, 82, 83];

    cache.commit_prompt(1, &protected_prompt[..4]).expect("commit 1");
    cache.release_sequence(1).expect("release 1");
    cache.commit_prompt(2, &other_prompt[..4]).expect("commit 2");
    cache.release_sequence(2).expect("release 2");
    assert_eq!(cache.arena().resident_block_count(), 2);

    cache.reset_lookahead_protection().expect("reset lookahead");
    let protection = cache.protect_prompt_lookahead(&protected_prompt, 900).expect("protect");
    assert_eq!(protection.protected_block_count, 1);

    // Trimming to one resident block must evict the unprotected block.
    let evicted = cache.trim_resident_blocks_by_reuse_score(1, &[]).expect("trim");
    assert_eq!(evicted, 1);
    assert_eq!(cache.arena().resident_block_count(), 1);
    assert_eq!(cache.reuse_scored_resident_eviction_count, 1);

    let residency =
        cache.probe_reusable_prefix_residency(&protected_prompt).expect("residency protected");
    assert_eq!(residency.resident_block_count, 1);
    let residency = cache.probe_reusable_prefix_residency(&other_prompt).expect("residency other");
    assert_eq!(residency.nonresident_block_count, 1);

    // Hard-protecting everything stalls the trim.
    let all_blocks: Vec<u32> = (0..8).collect();
    assert_eq!(
        cache.trim_resident_blocks_by_reuse_score(0, &all_blocks),
        Err(PrefixCacheError::CapacityExceeded)
    );
    assert_eq!(cache.reuse_scored_capacity_stall_count, 1);
}

// Reset restores pristine state and clears counters (but not reuse_scored_*).
#[test]
fn reset_restores_pristine_state() {
    let mut cache = make_cache(4, 8, 4);
    let token_ids = tokens(1, 8);

    cache.commit_prompt(1, &token_ids).expect("commit");
    cache.release_sequence(1).expect("release");
    cache.trim_resident_blocks_by_reuse_score(0, &[]).expect("trim");
    assert_eq!(cache.reuse_scored_resident_eviction_count, 2);

    cache.reset().expect("reset");
    assert_eq!(cache.tick, 0);
    assert_eq!(cache.operation_epoch, 0);
    assert_eq!(cache.lookup_count, 0);
    assert_eq!(cache.inserted_block_count, 0);
    assert_eq!(cache.acquired_block_count, 0);
    assert_eq!(cache.arena().allocated_block_count(), 0);
    assert_eq!(cache.arena().resident_block_count(), 0);
    // C quirk: reuse-scored counters survive reset.
    assert_eq!(cache.reuse_scored_resident_eviction_count, 2);

    // The cache is fully usable afterwards and re-acquires block 0 first.
    let lookup = cache.commit_prompt(2, &token_ids).expect("commit again");
    assert_eq!(lookup.matched_token_count, 8);
    assert_eq!(cache.inserted_block_count, 2);
}

// BindCommittedPrefixFromSequence shares committed prefix blocks.
#[test]
fn bind_committed_prefix_from_sequence_shares_blocks() {
    let mut cache = make_cache(8, 16, 4);
    let token_ids = tokens(100, 8);

    cache.commit_prompt(1, &token_ids).expect("commit");
    cache.bind_committed_prefix_from_sequence(1, 2, 8).expect("bind");
    assert_eq!(cache.entry_view(0).expect("e0").reference_count, 2);
    assert_eq!(cache.entry_view(1).expect("e1").reference_count, 2);

    let table = cache.build_physical_block_table(2, 8).expect("table");
    assert_eq!(table.len(), 2);

    assert_eq!(
        cache.bind_committed_prefix_from_sequence(1, 1, 8),
        Err(PrefixCacheError::InvalidArgument)
    );
    // Source sequence 99 has no bindings.
    assert_eq!(
        cache.bind_committed_prefix_from_sequence(99, 3, 4),
        Err(PrefixCacheError::NotFound)
    );
}
