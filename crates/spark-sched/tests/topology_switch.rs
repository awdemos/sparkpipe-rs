//! Host-verifiable port of `tests/test_topology_switch.c`.
//!
//! Everything the machine decides is schedule arithmetic over three seams —
//! the tier, the swap device, and the checkpoint write path — so the whole
//! protocol is checkable without a GPU: the swap mock completes after a
//! programmed number of polls and remembers which recipes it was asked to
//! move between, the write mock remembers every manifest offset, and the
//! mock tier implements just enough of the NVMe tier contract (capacity,
//! a clock hand that steps over pinned records, reserve/commit) for the
//! scenarios to say "the pinned blocks survived a full tier" and "the
//! drain cost no extra step" rather than hoping.

use std::collections::{HashMap, VecDeque};

use spark_sched::topology_switch::{
    kv_key, ManifestWriter, ResumeClass, SwapDevice, SwitchState, SwitchTier, TierNeed,
    TopologyRecipe, TopologyStrategy, TopologySwitch, TopologySwitchConfig, TopologySwitchError,
    WriteReservation,
};

const BLOCK_BYTES: u64 = 4096;
const MAX_SEQS: u32 = 4;
const MAX_BLOCKS: u32 = 8;
const NAMESPACE: u64 = 0x5150;
const RECIPE_TP_ID: u64 = 11;
const RECIPE_PP_ID: u64 = 22;

// -- the mock tier -----------------------------------------------------------

/// Just enough of the NVMe tier contract for the switch scenarios:
/// committed records with a capacity, a FIFO clock hand that steps over
/// pinned records on eviction, the reserve/commit/abort write protocol,
/// and a lookahead plan that a `pump` turns into device submissions.
struct MockTier {
    capacity: usize,
    records: HashMap<u64, u64>,
    pending: HashMap<u64, u64>,
    pins: HashMap<u64, u64>,
    clock: VecDeque<u64>,
    next_offset: u64,
    planned: Vec<u64>,
    submits: u64,
    evictions: u64,
    pinned_eviction_skips: u64,
}

impl MockTier {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            records: HashMap::new(),
            pending: HashMap::new(),
            pins: HashMap::new(),
            clock: VecDeque::new(),
            next_offset: 1 << 20,
            planned: Vec::new(),
            submits: 0,
            evictions: 0,
            pinned_eviction_skips: 0,
        }
    }

    fn evict_until_room(&mut self) -> Result<(), TopologySwitchError> {
        // One full pass over the clock: if every record is pinned the
        // reserve genuinely has nowhere to go.
        for _ in 0..=self.clock.len() {
            if self.records.len() + self.pending.len() < self.capacity {
                return Ok(());
            }
            let key = match self.clock.pop_front() {
                Some(key) => key,
                None => return Err(TopologySwitchError::CapacityExceeded),
            };
            if self.pins.get(&key).copied().unwrap_or(0) != 0 {
                self.pinned_eviction_skips += 1;
                self.clock.push_back(key);
                continue;
            }
            self.records.remove(&key);
            self.evictions += 1;
        }
        if self.records.len() + self.pending.len() < self.capacity {
            Ok(())
        } else {
            Err(TopologySwitchError::CapacityExceeded)
        }
    }

    /// Publish a block the way serving write-back would: reserve + commit
    /// under the same namespaced key the switch computes.
    fn publish(&mut self, key: u64) {
        let reservation = self.reserve_write(key).expect("reserve");
        self.commit_write(&reservation).expect("commit");
    }

    /// The tier's between-steps pump: planned lookahead fetches become
    /// device read submissions.
    fn pump(&mut self) {
        self.submits += self.planned.len() as u64;
        self.planned.clear();
    }
}

impl SwitchTier for MockTier {
    fn pin(&mut self, key: u64, pinned: bool) -> Result<(), TopologySwitchError> {
        if pinned {
            if !self.records.contains_key(&key) {
                return Err(TopologySwitchError::NotFound);
            }
            *self.pins.entry(key).or_insert(0) += 1;
            return Ok(());
        }
        if let Some(count) = self.pins.get_mut(&key) {
            *count -= 1;
            if *count == 0 {
                self.pins.remove(&key);
            }
            return Ok(());
        }
        if self.records.contains_key(&key) {
            Ok(())
        } else {
            Err(TopologySwitchError::NotFound)
        }
    }

    fn offset_of(&mut self, key: u64) -> Result<u64, TopologySwitchError> {
        self.records.get(&key).copied().ok_or(TopologySwitchError::NotFound)
    }

    fn reserve_write(&mut self, key: u64) -> Result<WriteReservation, TopologySwitchError> {
        if let Some(&offset) = self.records.get(&key) {
            return Ok(WriteReservation { device_offset: offset, already_present: true });
        }
        self.evict_until_room()?;
        let offset = self.next_offset;
        self.next_offset += BLOCK_BYTES;
        self.pending.insert(key, offset);
        Ok(WriteReservation { device_offset: offset, already_present: false })
    }

    fn commit_write(&mut self, reservation: &WriteReservation) -> Result<(), TopologySwitchError> {
        let (&key, &offset) = self
            .pending
            .iter()
            .find(|(_, &offset)| offset == reservation.device_offset)
            .ok_or(TopologySwitchError::NotFound)?;
        self.pending.remove(&key);
        self.records.insert(key, offset);
        self.clock.push_back(key);
        Ok(())
    }

    fn abort_write(&mut self, reservation: &WriteReservation) {
        self.pending.retain(|_, &mut offset| offset != reservation.device_offset);
    }

    fn plan_lookahead(
        &mut self,
        needs: &[TierNeed],
        _step_now: u32,
    ) -> Result<(), TopologySwitchError> {
        for need in needs {
            if self.records.contains_key(&need.key) {
                self.planned.push(need.key);
            }
        }
        Ok(())
    }
}

// -- the mock swap device ----------------------------------------------------

struct MockSwap {
    from_recipe_id: u64,
    target_recipe_id: u64,
    polls_per_swap: u32,
    polls_left: u32,
    begins: u32,
}

impl SwapDevice for MockSwap {
    fn begin_swap(
        &mut self,
        from: &TopologyRecipe,
        target: &TopologyRecipe,
    ) -> Result<(), TopologySwitchError> {
        self.from_recipe_id = from.recipe_id;
        self.target_recipe_id = target.recipe_id;
        self.polls_left = self.polls_per_swap;
        self.begins += 1;
        Ok(())
    }

    fn poll_swap(&mut self) -> Result<(), TopologySwitchError> {
        if self.polls_left != 0 {
            self.polls_left -= 1;
            return Err(TopologySwitchError::Busy);
        }
        Ok(())
    }
}

// -- the mock checkpoint write path ------------------------------------------

#[derive(Default)]
struct MockWrites {
    offsets: Vec<u64>,
    bytes: Vec<u32>,
}

impl ManifestWriter for MockWrites {
    fn write_block(
        &mut self,
        device_offset: u64,
        payload: &[u8],
    ) -> Result<(), TopologySwitchError> {
        self.offsets.push(device_offset);
        self.bytes.push(payload.len() as u32);
        Ok(())
    }
}

// -- the fixture ---------------------------------------------------------------

struct Fixture {
    sw: TopologySwitch<MockTier, MockSwap, MockWrites>,
    config: TopologySwitchConfig,
    recipe_tp: TopologyRecipe,
    recipe_pp: TopologyRecipe,
}

fn fixture_open(tier_records: usize, swap_polls: u32) -> Fixture {
    let recipe_tp = TopologyRecipe {
        recipe_id: RECIPE_TP_ID,
        weight_pack_bytes: 100_000_000_000, // 100 GB
        strategy: TopologyStrategy::TensorParallel,
    };
    let recipe_pp = TopologyRecipe {
        recipe_id: RECIPE_PP_ID,
        weight_pack_bytes: 80_000_000_000, // 80 GB
        strategy: TopologyStrategy::PipelineParallel,
    };
    let config = TopologySwitchConfig {
        kv_namespace: NAMESPACE,
        max_sequences: MAX_SEQS,
        max_blocks_per_sequence: MAX_BLOCKS,
        step_time_microseconds: 20_000, // a 20 ms step
        manifest_block_bytes: BLOCK_BYTES as u32,
        nvme_read_bytes_per_second: 5_000_000_000, // 5 GB/s
        nvme_write_bytes_per_second: 2_000_000_000, // 2 GB/s
        swap_fixed_microseconds: 500_000,          // 0.5 s
        initial_recipe: recipe_tp,
    };
    let tier = MockTier::new(tier_records);
    let swap = MockSwap {
        from_recipe_id: 0,
        target_recipe_id: 0,
        polls_per_swap: swap_polls,
        polls_left: 0,
        begins: 0,
    };
    let sw = TopologySwitch::new(config, tier, swap, MockWrites::default())
        .expect("the machine initialises");
    Fixture { sw, config, recipe_tp, recipe_pp }
}

fn fixture_publish_blocks(fixture: &mut Fixture, content_hashes: &[u64]) {
    for &hash in content_hashes {
        fixture.sw.tier_mut().publish(kv_key(NAMESPACE, hash));
    }
}

fn fixture_track_with_kv(
    fixture: &mut Fixture,
    sequence_id: u64,
    position_tokens: u32,
    content_hashes: &[u64],
) {
    assert_eq!(
        fixture.sw.track_sequence(sequence_id, RECIPE_TP_ID),
        Ok(()),
        "sequence tracks under the running recipe"
    );
    assert_eq!(
        fixture.sw.set_sequence_kv(sequence_id, position_tokens, content_hashes),
        Ok(()),
        "its tier keys register"
    );
}

/// Drive advance until Steady, bounded so a wedged machine fails the test
/// rather than hanging it. Returns the number of advance calls used.
fn fixture_advance_to_steady(fixture: &mut Fixture, first_step: u32) -> u32 {
    for step in first_step..first_step + 64 {
        if fixture.sw.advance(step) == SwitchState::Steady {
            return step - first_step + 1;
        }
    }
    0
}

// -- the scenarios -------------------------------------------------------------

#[test]
fn keying_excludes_the_strategy() {
    let key_a = kv_key(NAMESPACE, 777);
    assert_ne!(key_a, 0, "a key is never zero (zero means unhashed)");
    assert_eq!(kv_key(NAMESPACE, 777), key_a, "keying is deterministic");
    assert_ne!(kv_key(NAMESPACE, 778), key_a, "different content hashes differently");
    assert_ne!(kv_key(0x9999, 777), key_a, "a different model namespace hashes differently");
    // The namespace carries model + neutral geometry and NOT the strategy,
    // so the same function names a block before and after a TP<->PP switch.
    // What would make it differ — a strategy field — is absent from the
    // signature by construction.
}

#[test]
fn initialisation_validates_the_shape() {
    let fixture = fixture_open(32, 0);
    assert!(fixture.sw.admissions_open(), "admissions start open");
    assert_eq!(fixture.sw.state(), SwitchState::Steady, "the machine starts STEADY");
    assert_eq!(
        fixture.sw.current_recipe().recipe_id,
        RECIPE_TP_ID,
        "the running recipe is the initial one"
    );

    let mut broken = fixture.config;
    broken.manifest_block_bytes = 16; // cannot hold one key list
    assert_eq!(
        TopologySwitch::new(
            broken,
            MockTier::new(32),
            MockSwap {
                from_recipe_id: 0,
                target_recipe_id: 0,
                polls_per_swap: 0,
                polls_left: 0,
                begins: 0,
            },
            MockWrites::default(),
        )
        .err(),
        Some(TopologySwitchError::InvalidArgument),
        "a manifest too small for its own key list is refused"
    );

    let mut broken = fixture.config;
    broken.nvme_read_bytes_per_second = 0;
    assert_eq!(
        TopologySwitch::new(
            broken,
            MockTier::new(32),
            MockSwap {
                from_recipe_id: 0,
                target_recipe_id: 0,
                polls_per_swap: 0,
                polls_left: 0,
                begins: 0,
            },
            MockWrites::default(),
        )
        .err(),
        Some(TopologySwitchError::InvalidArgument),
        "a zero bandwidth is refused: the budget would lie"
    );
}

#[test]
fn admission_gate_closes_synchronously_at_begin() {
    let mut fixture = fixture_open(32, 0);
    assert_eq!(
        fixture.sw.begin(&fixture.recipe_tp),
        Err(TopologySwitchError::InvalidArgument),
        "switching to the running recipe is refused, not a fast path"
    );
    assert_eq!(fixture.sw.begin(&fixture.recipe_pp), Ok(()), "a real target starts the switch");
    assert!(!fixture.sw.admissions_open(), "admissions are closed the moment Begin returns");
    assert_eq!(
        fixture.sw.track_sequence(99, RECIPE_TP_ID),
        Err(TopologySwitchError::Busy),
        "the tracking backstop agrees"
    );
    assert_eq!(
        fixture.sw.begin(&fixture.recipe_tp),
        Err(TopologySwitchError::Busy),
        "a second Begin mid-switch is BUSY"
    );
    assert_ne!(fixture_advance_to_steady(&mut fixture, 1), 0);
    assert!(fixture.sw.admissions_open(), "admissions reopen when the new recipe is resident");
    assert_eq!(
        fixture.sw.current_recipe().recipe_id,
        RECIPE_PP_ID,
        "the machine now serves the target recipe"
    );
}

#[test]
fn quiesce_waits_for_the_last_boundary_and_costs_no_extra_step() {
    let mut fixture = fixture_open(32, 0);
    let blocks_a = [1000, 1001];
    let blocks_b = [1002, 1003];
    fixture_publish_blocks(&mut fixture, &blocks_a);
    fixture_publish_blocks(&mut fixture, &blocks_b);
    fixture_track_with_kv(&mut fixture, 1, 128, &blocks_a);
    fixture_track_with_kv(&mut fixture, 2, 96, &blocks_b);
    fixture_track_with_kv(&mut fixture, 3, 64, &blocks_a);
    assert_eq!(
        fixture.sw.begin(&fixture.recipe_pp),
        Ok(()),
        "the switch begins with three in flight"
    );
    fixture.sw.sequence_at_boundary(1).unwrap();
    fixture.sw.sequence_at_boundary(2).unwrap();
    assert_eq!(
        fixture.sw.advance(1),
        SwitchState::Quiesce,
        "one sequence still decoding holds the machine in QUIESCE"
    );
    // Sequence 3 does not reach a boundary; it finishes. Completing
    // mid-quiesce is a drain, and checkpointing finished work is waste.
    assert_eq!(
        fixture.sw.sequence_complete(3),
        Ok(()),
        "a sequence completing mid-quiesce drains it"
    );
    assert_eq!(
        fixture_advance_to_steady(&mut fixture, 2),
        1,
        "drained, the whole switch bar the swap is ONE advance: \
         checkpoint and resume cost no step of their own"
    );
    let statistics = fixture.sw.statistics();
    assert_eq!(
        statistics.sequences_completed_mid_switch, 1,
        "the mid-switch completion is counted"
    );
    assert_eq!(statistics.sequences_checkpointed, 2, "only live sequences were checkpointed");
    assert_eq!(fixture.sw.resume_class_of(1), ResumeClass::Warm, "sequence 1 resumes warm");
    assert_eq!(fixture.sw.resume_class_of(2), ResumeClass::Warm, "sequence 2 resumes warm");
    assert_eq!(
        fixture.sw.writer().offsets.len(),
        2,
        "one manifest write per checkpointed sequence"
    );
    let swap = fixture.sw.swap_device();
    assert!(
        swap.from_recipe_id == RECIPE_TP_ID && swap.target_recipe_id == RECIPE_PP_ID,
        "the swap device saw TP out, PP in"
    );
    assert_eq!(statistics.switches_completed, 1, "the switch completed");
}

#[test]
fn swap_poll_loop_holds_until_the_recipe_is_resident() {
    let mut fixture = fixture_open(32, 3);
    let blocks = [2000, 2001];
    fixture_publish_blocks(&mut fixture, &blocks);
    fixture_track_with_kv(&mut fixture, 7, 128, &blocks);
    fixture.sw.begin(&fixture.recipe_pp).unwrap();
    fixture.sw.sequence_at_boundary(7).unwrap();
    assert_eq!(
        fixture.sw.advance(1),
        SwitchState::Swap,
        "checkpoint and swap issue complete in the drain step"
    );
    assert_eq!(fixture.sw.swap_device().begins, 1, "begin_swap fired exactly once");
    assert_eq!(fixture.sw.advance(2), SwitchState::Swap, "first poll: still streaming");
    assert_eq!(fixture.sw.advance(3), SwitchState::Swap, "second poll: still streaming");
    assert_eq!(
        fixture.sw.advance(4),
        SwitchState::Steady,
        "when the stream lands, resume and STEADY are the same advance"
    );
}

#[test]
fn checkpoint_pins_survive_a_full_tiers_worth_of_churn() {
    let mut fixture = fixture_open(32, 2);
    let blocks = [3000, 3001, 3002, 3003];
    fixture_publish_blocks(&mut fixture, &blocks);
    fixture_track_with_kv(&mut fixture, 8, 256, &blocks);
    fixture.sw.begin(&fixture.recipe_pp).unwrap();
    fixture.sw.sequence_at_boundary(8).unwrap();
    assert_eq!(fixture.sw.advance(1), SwitchState::Swap, "checkpointed, swap in flight");
    // Mid-swap, ordinary serving write-back continues and floods the
    // 32-record tier with fresh records.
    for index in 0..64u64 {
        fixture.sw.tier_mut().publish(9000 + index);
    }
    assert_ne!(fixture.sw.tier().evictions, 0, "the churn really evicted");
    assert_ne!(
        fixture.sw.tier().pinned_eviction_skips,
        0,
        "the clock met the pins and stepped over them"
    );
    let mut alive = 0;
    for &block in &blocks {
        if fixture.sw.tier_mut().offset_of(kv_key(NAMESPACE, block)).is_ok() {
            alive += 1;
        }
    }
    assert_eq!(alive, 4, "every pinned block survived the flood");
    assert_ne!(fixture_advance_to_steady(&mut fixture, 2), 0, "the switch lands");
    assert_eq!(
        fixture.sw.resume_class_of(8),
        ResumeClass::Warm,
        "and the sequence resumes warm because of it"
    );
}

#[test]
fn resume_warm_hits_reload_and_the_never_written_back_recompute() {
    let mut fixture = fixture_open(32, 0);
    let blocks_warm = [4000, 4001, 4002];
    let blocks_cold = [4100, 4101];
    fixture_publish_blocks(&mut fixture, &blocks_warm);
    // blocks_cold are never published: the tier never saw them.
    fixture_track_with_kv(&mut fixture, 11, 192, &blocks_warm);
    fixture_track_with_kv(&mut fixture, 12, 128, &blocks_cold);
    fixture.sw.begin(&fixture.recipe_pp).unwrap();
    fixture.sw.sequence_at_boundary(11).unwrap();
    fixture.sw.sequence_at_boundary(12).unwrap();
    let submits_before = fixture.sw.tier().submits;
    assert_ne!(fixture_advance_to_steady(&mut fixture, 1), 0, "the switch lands");
    assert_eq!(fixture.sw.resume_class_of(11), ResumeClass::Warm, "published blocks resume warm");
    assert_eq!(
        fixture.sw.resume_class_of(12),
        ResumeClass::Recompute,
        "absent blocks go back through prefill"
    );
    let statistics = fixture.sw.statistics();
    assert_eq!(
        statistics.tier_blocks_absent, 2,
        "the checkpoint counted the absent blocks it could not pin"
    );
    assert!(
        statistics.sequences_resumed_warm == 1 && statistics.sequences_resumed_recompute == 1,
        "both classes are counted"
    );
    fixture.sw.tier_mut().pump();
    assert!(
        fixture.sw.tier().submits > submits_before,
        "the warm sequence's blocks went into the tier's lookahead: \
         resume is a planned fetch, not a demand stall"
    );
}

#[test]
fn eviction_between_serving_and_switch_sends_the_sequence_to_prefill() {
    let mut fixture = fixture_open(8, 0); // an 8-record tier, on purpose
    let blocks = [5000, 5001, 5002, 5003];
    fixture_publish_blocks(&mut fixture, &blocks);
    fixture_track_with_kv(&mut fixture, 21, 256, &blocks);
    // Long after the write-back, a busy serving mix floods the tier and
    // the sequence's records age out — before any switch is requested.
    for index in 0..32u64 {
        fixture.sw.tier_mut().publish(8000 + index);
    }
    fixture.sw.begin(&fixture.recipe_pp).unwrap();
    fixture.sw.sequence_at_boundary(21).unwrap();
    assert_ne!(fixture_advance_to_steady(&mut fixture, 1), 0, "the switch lands");
    assert_eq!(
        fixture.sw.resume_class_of(21),
        ResumeClass::Recompute,
        "evicted from the 1TB means recompute, by the protocol's own rule"
    );
}

#[test]
fn budget_arithmetic_is_the_contract_in_numbers() {
    let fixture = fixture_open(32, 0);
    let budget = TopologySwitch::<MockTier, MockSwap, MockWrites>::estimate_budget(
        &fixture.config,
        &fixture.recipe_pp,
        4,
        10_000_000_000,
    )
    .expect("the estimate computes");
    assert_eq!(budget.quiesce_us, 20_000, "quiesce is one decode step");
    // 4 manifests x 4096 B at 2 GB/s = 8 us.
    assert_eq!(budget.checkpoint_us, 8, "checkpoint is manifest writes");
    assert_eq!(budget.swap_fixed_us, 500_000, "the fixed cost passes through");
    // 80 GB at 5 GB/s = 16 s.
    assert_eq!(budget.swap_stream_us, 16_000_000, "the stream is pack bytes over read bandwidth");
    assert_eq!(budget.resume_warm_us, 2_000_000, "the warm reload is priced separately");
    assert_eq!(
        budget.total_us,
        20_000 + 8 + 500_000 + 16_000_000,
        "and the total EXCLUDES it: resume overlaps serving"
    );
}
