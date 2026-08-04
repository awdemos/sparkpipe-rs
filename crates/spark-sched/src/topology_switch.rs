//! Topology switching policy — port of `scheduler/topology_switch.c`
//! (API declared in `include/sparkpipe/spark_topology_switch.h`).
//!
//! TP <-> PP switch with requests in flight: QUIESCE drains in-flight
//! sequences to a token boundary, CHECKPOINT pins every sequence's tier
//! blocks and writes one manifest record per sequence, SWAP streams the
//! target recipe's packs (the whole 20 s budget lives here), and RESUME
//! classifies each sequence WARM (every block still on the tier — skip
//! prefill, plan the blocks into the tier lookahead) or RECOMPUTE (at
//! least one block evicted — back through prefill).
//!
//! The tier key scheme is strategy-neutral by construction: the namespace
//! commits to the model identity and the neutral cache geometry but never
//! to the strategy, so a switch re-shards KV residency instead of
//! invalidating content. [`kv_key`] is the whole story in one function.
//!
//! Deliberate deviations from the C original (mechanics, not semantics):
//!
//! - The caller-provided arena blob (`tables`) is replaced by owned
//!   `Vec`s allocated once in [`TopologySwitch::new`];
//!   [`TopologySwitch::table_bytes`] is kept purely as parity accounting
//!   of what the C arena layout would occupy.
//! - The tier dependency (`SparkNvmeTier *` in the C configuration) is
//!   the [`SwitchTier`] trait, the swap device vtable is [`SwapDevice`],
//!   and the checkpoint write callback is [`ManifestWriter`]. The state
//!   machine remains host-verifiable: tests drive all three with mocks.
//! - `SparkStatus` returns become `Result<_, TopologySwitchError>`, and
//!   the C `last_error` status word becomes `Option<TopologySwitchError>`
//!   (`None` == `SPARK_STATUS_OK`).
//! - The ABI handshake (`abi_version` / `descriptor_bytes`) is dropped;
//!   Rust type-checking replaces it.
//! - Manifest records are serialised little-endian. The C code `memcpy`s
//!   native-endian words; on the reference platform that IS little-endian,
//!   and the explicit choice keeps the bytes platform-deterministic.
//!   The FNV-1a key hash likewise reads the content hash as little-endian
//!   bytes, matching the C byte order exactly.

/// Hash-domain constant folded into every manifest's tier key
/// (`TOPOLOGY_SWITCH_MANIFEST_DOMAIN`). Token-run hashes commit to token
/// ids only, so a fixed magic over a sequence id can never collide with
/// one by construction.
pub const MANIFEST_DOMAIN: u64 = 0x9e37_79b9_7f4a_7c15;

/// Resume plans blocks into the tier lookahead in chunks of this size
/// (`TOPOLOGY_SWITCH_PLAN_CHUNK`).
pub const PLAN_CHUNK: usize = 16;

/// Manifest header: sequence id, recipe id, position, block count.
const MANIFEST_HEADER_BYTES: u64 = 24;

/// `sizeof(TopologySwitchSequence)` in the C source; used only by
/// [`TopologySwitch::table_bytes`] so the parity number matches.
const C_SEQUENCE_RECORD_BYTES: u64 = 32;

const FNV_PRIME: u64 = 1099511628211;

/// Errors mirroring the `SparkStatus` codes the C entry points (and the
/// vtables they call) can produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TopologySwitchError {
    #[error("invalid argument")]
    InvalidArgument,
    #[error("capacity exceeded")]
    CapacityExceeded,
    #[error("not found")]
    NotFound,
    #[error("I/O error")]
    IoError,
    #[error("parse error")]
    ParseError,
    #[error("schema error")]
    SchemaError,
    #[error("hash mismatch")]
    HashMismatch,
    #[error("module not validated")]
    ModuleNotValidated,
    #[error("validation failed")]
    ValidationFailed,
    #[error("ABI mismatch")]
    AbiMismatch,
    #[error("target mismatch")]
    TargetMismatch,
    #[error("compiler error")]
    CompilerError,
    #[error("driver load error")]
    DriverLoadError,
    #[error("route not found")]
    RouteNotFound,
    #[error("busy")]
    Busy,
    #[error("duplicate")]
    Duplicate,
    #[error("internal error")]
    InternalError,
    #[error("pending")]
    Pending,
    #[error("unsupported")]
    Unsupported,
}

/// The strategies the ring switches between (`SparkTopologyStrategy`).
/// Kept as an explicit enum rather than a degree integer because the
/// switch protocol only ever asks "same or different" — the degrees live
/// in the recipes, not here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyStrategy {
    TensorParallel,
    PipelineParallel,
}

/// One (model, strategy, precision) weight instantiation
/// (`SparkTopologyRecipe`). `recipe_id` is the caller's hash over the
/// model identity, the strategy and the pack manifest: two recipes with
/// the same id ARE the same residency, and [`TopologySwitch::begin`] to
/// the running recipe is refused as a misconfiguration, not fast-pathed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopologyRecipe {
    pub recipe_id: u64,
    /// This rank's pack bytes the swap must stream.
    pub weight_pack_bytes: u64,
    pub strategy: TopologyStrategy,
}

/// The switch state machine (`SparkTopologySwitchState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchState {
    Steady,
    Quiesce,
    Checkpoint,
    Swap,
    Resume,
}

/// What resume decided about a sequence (`SparkTopologySwitchResumeClass`).
/// Warm means every block was on the tier and prefill is skipped entirely;
/// Recompute means at least one block was evicted and the sequence goes
/// back through prefill — which still hits the tier for whatever survived,
/// through the ordinary demand path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeClass {
    Pending,
    Warm,
    Recompute,
}

/// The 20-second question, answered in parts (`SparkTopologySwitchBudget`).
/// `resume_warm_us` is reported but NOT added to `total_us`: resume
/// overlaps the first decode steps through the tier's lookahead, so it is
/// traffic, not wall time. Everything else is serial.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SwitchBudget {
    /// One decode step.
    pub quiesce_us: u64,
    /// Manifest writes at write bandwidth.
    pub checkpoint_us: u64,
    /// As configured.
    pub swap_fixed_us: u64,
    /// Target pack bytes at read bandwidth.
    pub swap_stream_us: u64,
    /// Warm KV bytes at read bandwidth, overlapped.
    pub resume_warm_us: u64,
    /// quiesce + checkpoint + fixed + stream.
    pub total_us: u64,
}

/// Switch counters (`SparkTopologySwitchStatistics`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SwitchStatistics {
    pub switches_completed: u64,
    pub sequences_checkpointed: u64,
    pub sequences_resumed_warm: u64,
    pub sequences_resumed_recompute: u64,
    /// Finished during quiesce: free.
    pub sequences_completed_mid_switch: u64,
    pub blocks_pinned: u64,
    pub manifest_writes: u64,
    pub manifest_bytes: u64,
    /// Checkpoint found already on the tier.
    pub tier_blocks_found: u64,
    /// Not yet written back; recompute at resume.
    pub tier_blocks_absent: u64,
    pub swaps_started: u64,
}

/// Configuration (`SparkTopologySwitchConfiguration`). No GLM52 constants
/// anywhere: every capacity and every bandwidth is supplied by the caller.
#[derive(Debug, Clone, Copy)]
pub struct TopologySwitchConfig {
    /// Model + neutral geometry, strategy-free.
    pub kv_namespace: u64,
    /// Checkpoint table capacity.
    pub max_sequences: u32,
    /// Manifest capacity per sequence.
    pub max_blocks_per_sequence: u32,
    /// One decode step: the quiesce bound.
    pub step_time_microseconds: u32,
    /// Tier record size for manifests.
    pub manifest_block_bytes: u32,
    /// Budget arithmetic only.
    pub nvme_read_bytes_per_second: u64,
    /// Budget arithmetic only.
    pub nvme_write_bytes_per_second: u64,
    /// The swap's fixed cost — module unload, allocator reset, bind — with
    /// the streaming priced separately.
    pub swap_fixed_microseconds: u64,
    /// What the rank is running when the machine starts. Begin to this
    /// recipe is refused as a no-op, so it must be the truth.
    pub initial_recipe: TopologyRecipe,
}

/// One block the resume phase plans into the tier lookahead
/// (`SparkNvmeTierNeed`). `key` is the namespaced tier key.
#[derive(Debug, Clone, Copy, Default)]
pub struct TierNeed {
    pub key: u64,
    pub need_by_step: u32,
}

/// A tier write slot acquisition (`SparkNvmeTierWriteReservation`).
#[derive(Debug, Clone, Copy)]
pub struct WriteReservation {
    pub device_offset: u64,
    /// The key already names a committed record; no new slot was taken.
    pub already_present: bool,
}

/// The NVMe tier seam (`SparkNvmeTier` as the switch uses it): pins,
/// offset lookup, the write reservation protocol, and lookahead planning.
/// The real port of `cache/nvme_tier.c` implements this; the tests drive
/// it with a mock, and the state machine cannot tell the difference.
pub trait SwitchTier {
    /// Pin (or unpin) a block against the eviction clock. Pinning a key
    /// the tier does not hold returns [`TopologySwitchError::NotFound`].
    fn pin(&mut self, key: u64, pinned: bool) -> Result<(), TopologySwitchError>;
    /// The device offset of a committed record, `NotFound` if absent.
    fn offset_of(&mut self, key: u64) -> Result<u64, TopologySwitchError>;
    /// Reserve a write slot for `key`. `ReserveWrite` is the tier's only
    /// eviction path, which is why checkpoint pins come before it.
    fn reserve_write(&mut self, key: u64) -> Result<WriteReservation, TopologySwitchError>;
    fn commit_write(&mut self, reservation: &WriteReservation) -> Result<(), TopologySwitchError>;
    /// Abandon a reservation whose payload write failed.
    fn abort_write(&mut self, reservation: &WriteReservation);
    /// Plan blocks into the prefetch lookahead. A plan failure costs
    /// prefetch, never correctness, so the switch deliberately ignores it.
    fn plan_lookahead(
        &mut self,
        needs: &[TierNeed],
        step_now: u32,
    ) -> Result<(), TopologySwitchError>;
}

/// The swap device: the rank's weight loader (`SparkTopologySwitchSwapDevice`).
pub trait SwapDevice {
    /// Start the unload-plus-load from the running recipe to the target.
    /// Must not block: the packs are pre-staged on NVMe and the stream is
    /// asynchronous.
    fn begin_swap(
        &mut self,
        from: &TopologyRecipe,
        target: &TopologyRecipe,
    ) -> Result<(), TopologySwitchError>;
    /// `Ok(())` once the target recipe is resident and bindable,
    /// `Err(Busy)` while the stream is in flight.
    fn poll_swap(&mut self) -> Result<(), TopologySwitchError>;
}

/// The checkpoint's write path (`SparkTopologySwitchWriteBlock`): the one
/// payload a checkpoint must newly write (the per-sequence manifest) goes
/// through here.
pub trait ManifestWriter {
    fn write_block(
        &mut self,
        device_offset: u64,
        payload: &[u8],
    ) -> Result<(), TopologySwitchError>;
}

/// `SparkHashBytes` (spark_status.h): order-sensitive FNV-1a over the raw
/// bytes. Seeding with the namespace makes the namespace commit FIRST and
/// the content second — the same content under two models diverges, the
/// same content under two strategies of one model does not, which is
/// exactly the partition the tier needs. The content hash is read as
/// little-endian bytes, matching the C byte order on the reference target.
fn hash_bytes(mut hash: u64, data: &[u8]) -> u64 {
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// The tier key (`SparkTopologySwitchKvKey`). Never zero; zero means
/// unhashed on the tier.
pub fn kv_key(kv_namespace: u64, content_hash: u64) -> u64 {
    hash_bytes(kv_namespace, &content_hash.to_le_bytes()) | 1
}

/// A manifest's tier key: the sequence id behind the domain magic, inside
/// the same namespace as the blocks it names (`TopologySwitchManifestKey`).
fn manifest_key(kv_namespace: u64, sequence_id: u64) -> u64 {
    let content = hash_bytes(MANIFEST_DOMAIN, &sequence_id.to_le_bytes()) | 1;
    kv_key(kv_namespace, content)
}

/// Slot state (`TOPOLOGY_SWITCH_SEQUENCE_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Free,
    Active,
    AtBoundary,
}

/// One tracked sequence (`TopologySwitchSequence`). Fixed slots, no free
/// list: slot index is identity while a sequence lives.
#[derive(Debug, Clone, Copy)]
struct SequenceSlot {
    sequence_id: u64,
    /// The recipe it was admitted under.
    recipe_id: u64,
    position_tokens: u32,
    block_count: u32,
    state: SlotState,
    resume_class: ResumeClass,
    /// Block pins taken; write reservation may still retry.
    pins_done: bool,
    /// Pins down, manifest committed.
    checkpointed: bool,
}

impl Default for SequenceSlot {
    fn default() -> Self {
        Self {
            sequence_id: 0,
            recipe_id: 0,
            position_tokens: 0,
            block_count: 0,
            state: SlotState::Free,
            resume_class: ResumeClass::Pending,
            pins_done: false,
            checkpointed: false,
        }
    }
}

/// The topology switch state machine (`SparkTopologySwitch`).
///
/// Phases retry instead of failing: every phase's work is bounded
/// bookkeeping plus non-blocking trait calls, and every trait call can say
/// Busy for reasons that resolve themselves. [`TopologySwitch::advance`]
/// carries a per-phase cursor and `last_error` instead of an abort path.
/// There is no CANCEL: cancelling mid-swap would leave residency split
/// between recipes, the one state worse than either recipe alone.
pub struct TopologySwitch<T: SwitchTier, S: SwapDevice, W: ManifestWriter> {
    config: TopologySwitchConfig,
    tier: T,
    swap_device: S,
    writer: W,
    /// One record per sequence the scheduler may have in flight.
    sequences: Vec<SequenceSlot>,
    /// `max_sequences * max_blocks_per_sequence` tier keys, one contiguous
    /// run per sequence slot.
    block_keys: Vec<u64>,
    /// One manifest, staged for the writer; manifests go one at a time.
    manifest_buffer: Vec<u8>,
    current_recipe: TopologyRecipe,
    target_recipe: TopologyRecipe,
    state: SwitchState,
    sequence_count: u32,
    /// Checkpoint/resume progress within a phase.
    phase_cursor: u32,
    /// begin_swap issued, polling for resident.
    swap_started: bool,
    /// Why a phase is retrying, when it is (`None` == SPARK_STATUS_OK).
    last_error: Option<TopologySwitchError>,
    statistics: SwitchStatistics,
}

impl<T: SwitchTier, S: SwapDevice, W: ManifestWriter> TopologySwitch<T, S, W> {
    /// Bookkeeping bytes the C caller provides as `tables`
    /// (`SparkTopologySwitchTableBytes`). The Rust port owns its tables, so
    /// this is parity accounting for capacity planning only.
    pub fn table_bytes(config: &TopologySwitchConfig) -> u64 {
        let align8 = |value: u64| (value + 7) & !7;
        let mut total = align8(u64::from(config.max_sequences) * C_SEQUENCE_RECORD_BYTES);
        total = align8(
            total + u64::from(config.max_sequences) * u64::from(config.max_blocks_per_sequence) * 8,
        );
        align8(total + u64::from(config.manifest_block_bytes))
    }

    /// `SparkTopologySwitchInitialize`. The ABI handshake is dropped (Rust
    /// type-checking replaces it); every other validation the C performs
    /// is preserved.
    pub fn new(
        config: TopologySwitchConfig,
        tier: T,
        swap_device: S,
        writer: W,
    ) -> Result<Self, TopologySwitchError> {
        if config.max_sequences == 0
            || config.max_blocks_per_sequence == 0
            || config.step_time_microseconds == 0
        {
            return Err(TopologySwitchError::InvalidArgument);
        }
        // Zero bandwidth makes the budget silently zero-cost, which is the
        // one estimate this module must never produce: refuse it at init.
        if config.nvme_read_bytes_per_second == 0 || config.nvme_write_bytes_per_second == 0 {
            return Err(TopologySwitchError::InvalidArgument);
        }
        // A manifest that cannot hold its own block list is a checkpoint
        // that truncates silently. Checked here, where the mistake is free.
        let manifest_capacity =
            MANIFEST_HEADER_BYTES + u64::from(config.max_blocks_per_sequence) * 8;
        if manifest_capacity > u64::from(config.manifest_block_bytes) {
            return Err(TopologySwitchError::InvalidArgument);
        }
        let slot_count = config.max_sequences as usize;
        let key_count = slot_count * config.max_blocks_per_sequence as usize;
        Ok(Self {
            config,
            tier,
            swap_device,
            writer,
            sequences: vec![SequenceSlot::default(); slot_count],
            block_keys: vec![0; key_count],
            manifest_buffer: vec![0; config.manifest_block_bytes as usize],
            current_recipe: config.initial_recipe,
            target_recipe: config.initial_recipe,
            state: SwitchState::Steady,
            sequence_count: 0,
            phase_cursor: 0,
            swap_started: false,
            last_error: None,
            statistics: SwitchStatistics::default(),
        })
    }

    /// The scheduler's admission gate (`SparkTopologySwitchAdmissionsOpen`).
    /// Closed from the moment [`TopologySwitch::begin`] lands until the new
    /// recipe is resident and classified.
    pub fn admissions_open(&self) -> bool {
        self.state == SwitchState::Steady
    }

    /// `SparkTopologySwitchStateOf`.
    pub fn state(&self) -> SwitchState {
        self.state
    }

    /// The recipe the machine is serving or switching to
    /// (`SparkTopologySwitchCurrentRecipe`).
    pub fn current_recipe(&self) -> &TopologyRecipe {
        &self.current_recipe
    }

    /// Why a phase is retrying (`SparkTopologySwitchLastError`). `None`
    /// in Steady and whenever the machine is making progress; `Some` names
    /// the failure the next [`TopologySwitch::advance`] will retry.
    pub fn last_error(&self) -> Option<TopologySwitchError> {
        self.last_error
    }

    /// `SparkTopologySwitchGetStatistics`.
    pub fn statistics(&self) -> SwitchStatistics {
        self.statistics
    }

    /// The tier the machine checkpoints into and resumes from. Rust
    /// replacement for the C caller holding its own `SparkNvmeTier *`
    /// alongside the switch.
    pub fn tier(&self) -> &T {
        &self.tier
    }

    /// Mutable tier access, for the serving-time write-back and demand
    /// paths that run alongside the switch machine.
    pub fn tier_mut(&mut self) -> &mut T {
        &mut self.tier
    }

    /// The swap device, for post-switch inspection by the caller.
    pub fn swap_device(&self) -> &S {
        &self.swap_device
    }

    /// The manifest writer, for post-switch inspection by the caller.
    pub fn writer(&self) -> &W {
        &self.writer
    }

    /// Register a sequence admitted under the CURRENT recipe
    /// (`SparkTopologySwitchTrackSequence`). Admission is the scheduler's
    /// gate; the state check here is the backstop — a sequence tracked
    /// mid-switch would bind to the recipe being unloaded.
    pub fn track_sequence(
        &mut self,
        sequence_id: u64,
        recipe_id: u64,
    ) -> Result<(), TopologySwitchError> {
        if sequence_id == 0 {
            return Err(TopologySwitchError::InvalidArgument);
        }
        if self.state != SwitchState::Steady {
            return Err(TopologySwitchError::Busy);
        }
        if recipe_id != self.current_recipe.recipe_id {
            return Err(TopologySwitchError::InvalidArgument);
        }
        if self.find_sequence(sequence_id).is_some() {
            return Err(TopologySwitchError::Duplicate);
        }
        for slot in &mut self.sequences {
            if slot.state != SlotState::Free {
                continue;
            }
            slot.sequence_id = sequence_id;
            slot.recipe_id = recipe_id;
            slot.position_tokens = 0;
            slot.block_count = 0;
            slot.state = SlotState::Active;
            slot.resume_class = ResumeClass::Pending;
            slot.pins_done = false;
            slot.checkpointed = false;
            self.sequence_count += 1;
            return Ok(());
        }
        Err(TopologySwitchError::CapacityExceeded)
    }

    /// Name a sequence's tier block keys and position
    /// (`SparkTopologySwitchSetSequenceKv`). Kept current by the caller,
    /// so a checkpoint never has to ask. After checkpoint the block set is
    /// frozen on the tier; updating it would commit a manifest that names
    /// the wrong keys, hence Busy.
    pub fn set_sequence_kv(
        &mut self,
        sequence_id: u64,
        position_tokens: u32,
        content_hashes: &[u64],
    ) -> Result<(), TopologySwitchError> {
        if content_hashes.len() > self.config.max_blocks_per_sequence as usize {
            return Err(TopologySwitchError::CapacityExceeded);
        }
        let slot = self.find_sequence(sequence_id).ok_or(TopologySwitchError::NotFound)?;
        if self.sequences[slot].checkpointed {
            return Err(TopologySwitchError::Busy);
        }
        let base = slot * self.config.max_blocks_per_sequence as usize;
        for (index, &content_hash) in content_hashes.iter().enumerate() {
            self.block_keys[base + index] = kv_key(self.config.kv_namespace, content_hash);
        }
        self.sequences[slot].block_count = content_hashes.len() as u32;
        self.sequences[slot].position_tokens = position_tokens;
        Ok(())
    }

    /// A quiesce signal: the sequence reached a token boundary
    /// (`SparkTopologySwitchSequenceAtBoundary`).
    pub fn sequence_at_boundary(&mut self, sequence_id: u64) -> Result<(), TopologySwitchError> {
        let slot = self.find_sequence(sequence_id).ok_or(TopologySwitchError::NotFound)?;
        self.sequences[slot].state = SlotState::AtBoundary;
        Ok(())
    }

    /// The other quiesce signal: the sequence finished
    /// (`SparkTopologySwitchSequenceComplete`). A sequence that completes
    /// mid-switch is dropped from the checkpoint set — checkpointing
    /// finished work is the purest waste there is. Pins taken at
    /// checkpoint drop here, because resume skips freed slots.
    pub fn sequence_complete(&mut self, sequence_id: u64) -> Result<(), TopologySwitchError> {
        let slot = self.find_sequence(sequence_id).ok_or(TopologySwitchError::NotFound)?;
        if self.sequences[slot].pins_done {
            let base = slot * self.config.max_blocks_per_sequence as usize;
            let count = self.sequences[slot].block_count as usize;
            let keys = &self.block_keys[base..base + count];
            let tier = &mut self.tier;
            for &key in keys {
                let _ = tier.pin(key, false);
            }
        }
        if self.state != SwitchState::Steady && !self.sequences[slot].checkpointed {
            self.statistics.sequences_completed_mid_switch += 1;
        }
        self.sequences[slot].state = SlotState::Free;
        self.sequences[slot].resume_class = ResumeClass::Pending;
        self.sequences[slot].block_count = 0;
        self.sequence_count -= 1;
        Ok(())
    }

    /// Request the switch (`SparkTopologySwitchBegin`). Steady only;
    /// admissions close synchronously inside this call. Switching to the
    /// recipe already running is an argument error, not a fast path — a
    /// no-op switch that looks like a real one is how an operator script
    /// hides a misconfiguration.
    pub fn begin(&mut self, target: &TopologyRecipe) -> Result<(), TopologySwitchError> {
        if target.recipe_id == 0 {
            return Err(TopologySwitchError::InvalidArgument);
        }
        if self.state != SwitchState::Steady {
            return Err(TopologySwitchError::Busy);
        }
        if target.recipe_id == self.current_recipe.recipe_id {
            return Err(TopologySwitchError::InvalidArgument);
        }
        self.target_recipe = *target;
        self.state = SwitchState::Quiesce;
        self.phase_cursor = 0;
        self.swap_started = false;
        self.last_error = None;
        Ok(())
    }

    /// The between-steps driver (`SparkTopologySwitchAdvance`). Advances
    /// Quiesce -> Checkpoint once every tracked sequence is at a token
    /// boundary or complete, Checkpoint -> Swap once manifests and pins are
    /// down, Swap -> Resume when the swap device reports resident, and
    /// Resume -> Steady when every sequence is classified. Returns the
    /// state it left the machine in, so the caller's loop can log
    /// transitions without a second query.
    ///
    /// The C switch's fall-through is preserved: the drain's last boundary
    /// IS this step, checkpoint is bookkeeping rather than I/O waits, and
    /// issuing the swap is one non-blocking call, so a fully-drained switch
    /// can walk Quiesce -> Steady in a single advance.
    pub fn advance(&mut self, step_now: u32) -> SwitchState {
        if self.state == SwitchState::Quiesce && self.quiesce_drained() {
            self.state = SwitchState::Checkpoint;
            self.phase_cursor = 0;
        }
        if self.state == SwitchState::Checkpoint && self.checkpoint_run() {
            self.state = SwitchState::Swap;
            self.swap_started = false;
        }
        if self.state == SwitchState::Swap {
            let mut issued = true;
            if !self.swap_started {
                match self.swap_device.begin_swap(&self.current_recipe, &self.target_recipe) {
                    Ok(()) => {
                        self.swap_started = true;
                        self.statistics.swaps_started += 1;
                    }
                    Err(error) => {
                        self.last_error = Some(error);
                        issued = false;
                    }
                }
            }
            if issued {
                match self.swap_device.poll_swap() {
                    Ok(()) => {
                        self.state = SwitchState::Resume;
                        self.phase_cursor = 0;
                    }
                    Err(TopologySwitchError::Busy) => {}
                    Err(error) => {
                        self.last_error = Some(error);
                    }
                }
            }
        }
        if self.state == SwitchState::Resume {
            while (self.phase_cursor as usize) < self.sequences.len() {
                let slot = self.phase_cursor as usize;
                self.phase_cursor += 1;
                if self.sequences[slot].state == SlotState::Free {
                    continue;
                }
                self.resume_one(slot, step_now);
            }
            self.current_recipe = self.target_recipe;
            self.state = SwitchState::Steady;
            self.last_error = None;
            self.statistics.switches_completed += 1;
        }
        self.state
    }

    /// What resume decided, per sequence (`SparkTopologySwitchResumeClassOf`).
    /// Valid once the machine is back to Steady (and while Resume runs);
    /// Pending before classification. An unknown sequence id reports
    /// Recompute, matching the C null/not-found fallback.
    pub fn resume_class_of(&self, sequence_id: u64) -> ResumeClass {
        match self.find_sequence(sequence_id) {
            Some(slot) => self.sequences[slot].resume_class,
            None => ResumeClass::Recompute,
        }
    }

    /// The serial-time estimate for switching to `target` with
    /// `warm_kv_bytes` of checkpointed KV to reload
    /// (`SparkTopologySwitchEstimateBudget`). Bandwidths and the fixed swap
    /// cost come from the configuration, so an estimate and the run that
    /// validates it read the same numbers. Multiplications wrap like the C
    /// `uint64_t` arithmetic they mirror.
    pub fn estimate_budget(
        config: &TopologySwitchConfig,
        target: &TopologyRecipe,
        active_sequence_count: u32,
        warm_kv_bytes: u64,
    ) -> Result<SwitchBudget, TopologySwitchError> {
        if config.nvme_read_bytes_per_second == 0 || config.nvme_write_bytes_per_second == 0 {
            return Err(TopologySwitchError::InvalidArgument);
        }
        let manifest_bytes =
            u64::from(active_sequence_count) * u64::from(config.manifest_block_bytes);
        let checkpoint_us =
            manifest_bytes.wrapping_mul(1_000_000) / config.nvme_write_bytes_per_second;
        let swap_stream_us =
            target.weight_pack_bytes.wrapping_mul(1_000_000) / config.nvme_read_bytes_per_second;
        let resume_warm_us =
            warm_kv_bytes.wrapping_mul(1_000_000) / config.nvme_read_bytes_per_second;
        let quiesce_us = u64::from(config.step_time_microseconds);
        let swap_fixed_us = config.swap_fixed_microseconds;
        Ok(SwitchBudget {
            quiesce_us,
            checkpoint_us,
            swap_fixed_us,
            swap_stream_us,
            resume_warm_us,
            total_us: quiesce_us + checkpoint_us + swap_fixed_us + swap_stream_us,
        })
    }

    fn find_sequence(&self, sequence_id: u64) -> Option<usize> {
        self.sequences
            .iter()
            .position(|slot| slot.state != SlotState::Free && slot.sequence_id == sequence_id)
    }

    /// Every tracked sequence must report a token boundary (or complete).
    /// The bound is one decode step: a step is atomic at the boundary by
    /// design, so "stop issuing and let what is in flight land" can never
    /// wait longer.
    fn quiesce_drained(&self) -> bool {
        !self.sequences.iter().any(|slot| slot.state == SlotState::Active)
    }

    /// Checkpoint every live sequence, one per cursor step, retrying the
    /// failed sequence on the next advance (`TopologySwitchCheckpointRun`).
    fn checkpoint_run(&mut self) -> bool {
        while (self.phase_cursor as usize) < self.sequences.len() {
            let slot = self.phase_cursor as usize;
            self.phase_cursor += 1;
            if self.sequences[slot].state == SlotState::Free || self.sequences[slot].checkpointed {
                continue;
            }
            if let Err(error) = self.checkpoint_one(slot) {
                // Retry this sequence next advance; the cursor does not
                // advance past it, so nothing is skipped, and pins_done
                // keeps the retry from double-pinning what the first
                // attempt already pinned.
                self.phase_cursor -= 1;
                self.last_error = Some(error);
                return false;
            }
        }
        true
    }

    /// Pin every block still on the tier (BEFORE any manifest reserve,
    /// because reserve_write is the tier's only eviction path and an
    /// unpinned block is fair game for it), count the absent ones, then
    /// write one manifest (`TopologySwitchCheckpointOne`).
    fn checkpoint_one(&mut self, slot: usize) -> Result<(), TopologySwitchError> {
        let blocks_per_sequence = self.config.max_blocks_per_sequence as usize;
        let base = slot * blocks_per_sequence;
        let count = self.sequences[slot].block_count as usize;
        if !self.sequences[slot].pins_done {
            let keys = &self.block_keys[base..base + count];
            let tier = &mut self.tier;
            let statistics = &mut self.statistics;
            for &key in keys {
                match tier.pin(key, true) {
                    Ok(()) => {
                        statistics.tier_blocks_found += 1;
                        statistics.blocks_pinned += 1;
                    }
                    // Never written back during serving. Counted, not
                    // repaired: the bytes live in device memory and the
                    // checkpoint is not a copy engine — resume will
                    // classify Recompute.
                    Err(TopologySwitchError::NotFound) => {
                        statistics.tier_blocks_absent += 1;
                    }
                    Err(error) => return Err(error),
                }
            }
            self.sequences[slot].pins_done = true;
        }
        // Serialise: id, recipe, position, count, then the key run.
        self.manifest_buffer.fill(0);
        {
            let sequence = &self.sequences[slot];
            let buffer = &mut self.manifest_buffer;
            buffer[0..8].copy_from_slice(&sequence.sequence_id.to_le_bytes());
            buffer[8..16].copy_from_slice(&sequence.recipe_id.to_le_bytes());
            buffer[16..20].copy_from_slice(&sequence.position_tokens.to_le_bytes());
            buffer[20..24].copy_from_slice(&sequence.block_count.to_le_bytes());
            for index in 0..count {
                let key = self.block_keys[base + index];
                let at = MANIFEST_HEADER_BYTES as usize + index * 8;
                buffer[at..at + 8].copy_from_slice(&key.to_le_bytes());
            }
        }
        let key = manifest_key(self.config.kv_namespace, self.sequences[slot].sequence_id);
        let reservation = self.tier.reserve_write(key)?;
        if !reservation.already_present {
            if let Err(error) =
                self.writer.write_block(reservation.device_offset, &self.manifest_buffer)
            {
                self.tier.abort_write(&reservation);
                return Err(error);
            }
            self.tier.commit_write(&reservation)?;
        }
        self.sequences[slot].checkpointed = true;
        self.statistics.sequences_checkpointed += 1;
        self.statistics.manifest_writes += 1;
        self.statistics.manifest_bytes += u64::from(self.config.manifest_block_bytes);
        Ok(())
    }

    /// Classification is per sequence and total: every pinned block still
    /// on the tier is Warm and goes into the lookahead; one absent block
    /// makes the whole sequence Recompute, because resuming decode from
    /// position N requires the whole prefix, not most of it
    /// (`TopologySwitchResumeOne`).
    fn resume_one(&mut self, slot: usize, step_now: u32) {
        let blocks_per_sequence = self.config.max_blocks_per_sequence as usize;
        let base = slot * blocks_per_sequence;
        let count = self.sequences[slot].block_count as usize;
        let keys = &self.block_keys[base..base + count];
        let tier = &mut self.tier;
        let mut warm = true;
        for &key in keys {
            if tier.offset_of(key).is_err() {
                warm = false;
            }
            // The pin drops either way: its job was to protect the block
            // between checkpoint and this moment, and this moment has
            // arrived.
            let _ = tier.pin(key, false);
        }
        if warm && count != 0 {
            let mut needs = [TierNeed::default(); PLAN_CHUNK];
            for chunk in keys.chunks(PLAN_CHUNK) {
                for (fill, &key) in chunk.iter().enumerate() {
                    // The sequence re-issues as soon as admissions open;
                    // its first layer needs block zero immediately. One
                    // deadline for all is the honest version of "now".
                    needs[fill] = TierNeed { key, need_by_step: step_now + 1 };
                }
                // A plan failure costs prefetch, never correctness: the
                // demand path is the fallback, deliberately unchecked.
                let _ = tier.plan_lookahead(&needs[..chunk.len()], step_now);
            }
            self.sequences[slot].resume_class = ResumeClass::Warm;
            self.statistics.sequences_resumed_warm += 1;
        } else {
            self.sequences[slot].resume_class = ResumeClass::Recompute;
            self.statistics.sequences_resumed_recompute += 1;
        }
    }
}
