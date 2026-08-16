//! Rank daemon core (port of `node/rank_daemon.c`).
//!
//! This is the daemon's state machine: the work queue with FNV-32 dedup and
//! lane-sequence dependency ordering, the transaction-ledger handshake
//! (`handle_work`), the pump that forwards/registers/submits work, the
//! inflight completion tracking that turns driver completions into ledger
//! commits and final events, and the CUDA-resident IPC client
//! (split-read message pump, submit/await-result, hello-contract
//! validation). The cross-thread mutexes of the C become plain owned state;
//! the completion callback's wake signal is a [`WakePipe`].
//!
//! Seams (following the crate's trait pattern): [`SubmitEngine`] abstracts
//! `builder.submit_work` vs. the CUDA-resident socket submit,
//! [`WorkForwarder`] abstracts the downstream-rank socket write/ack, and
//! [`Link`] (in [`super::net`]) abstracts the byte streams. The dlopen'd
//! builder/driver/transport modules stay behind `spark-sys` and are not
//! ported; see the port ledger.
//!
//! Deviations from the C layout (all behavior-preserving):
//! - open-addressing hash tables (work dedup, inflight completions) become
//!   `HashMap`s keyed by the same fingerprints/keys;
//! - the 67 MiB static resident payload buffer becomes an on-demand `Vec`
//!   sized by each message header's `payload_bytes`;
//! - `SparkReportError` text side channels are dropped (the status codes
//!   are identical);
//! - stderr diagnostics become `eprintln!`.

use std::collections::{HashMap, VecDeque};

use spark_sched::work_control::work_transaction::Identity;
use spark_sched::work_control::{
    self, get_transaction_identity, packet_fingerprint, validate_packet, WorkControlConfig,
    WorkControlPacket, FLAG_RELEASE_SEQUENCES,
};

use super::net::{io_status, Link};
use super::resident_ipc::{
    self, DriverCompletion, DsparkDraftResult, IpcCompletion, IpcHeader, IpcStats, IpcSubmitResult,
    DRIVER_COMPLETION_FLAG_DRAFT_TOKEN_IDS, DRIVER_COMPLETION_FLAG_TOKEN_IDS,
};
use super::ring_runtime::{
    FinalEvent, QuantizationMode, RankPlan, RingModelGeometry, FINAL_EVENT_DESCRIPTOR_BYTES,
    FINAL_EVENT_FLAG_DSPARK_DRAFT, FINAL_EVENT_MAGIC, RANK_FLAG_FINAL_STAGE, RANK_FLAG_HAS_NEXT,
};
use super::status::{from_work_control, status_from_code, Result, SparkStatus};
use super::wire;
use super::work_ledger::{
    TransactionLedger, STATE_ACCEPTED, STATE_CANCELLED, STATE_COMMITTED, STATE_EXECUTING,
    STATE_FAILED,
};

/// C: `SPARK_RING_DAEMON_WORK_QUEUE_CAPACITY`.
pub const WORK_QUEUE_CAPACITY: usize = 64;
/// C: `SPARK_RING_DAEMON_TRANSACTION_LEDGER_CAPACITY`.
pub const TRANSACTION_LEDGER_CAPACITY: u32 = 4096;
/// C: `SPARK_RING_DAEMON_INFLIGHT_COMPLETION_CAPACITY`.
pub const INFLIGHT_COMPLETION_CAPACITY: usize = 16384;
/// C: `SPARK_RING_DAEMON_INFLIGHT_TRANSACTION_CAPACITY`.
pub const INFLIGHT_TRANSACTION_CAPACITY: usize = 4096;
/// C: `SPARK_RING_DAEMON_DRIVER_COMPLETION_QUEUE_CAPACITY`.
pub const DRIVER_COMPLETION_QUEUE_CAPACITY: usize = 4096;
/// C: `SPARK_RING_DAEMON_FINAL_EVENT_QUEUE_CAPACITY`.
pub const FINAL_EVENT_QUEUE_CAPACITY: usize = 2048;

/// C: `SPARK_RING_DAEMON_CONNECT_RETRY_NS`.
pub const CONNECT_RETRY_NS: u64 = 250_000;
/// C: `SPARK_RING_DAEMON_RUNNER_PROGRESS_NS`.
pub const RUNNER_PROGRESS_NS: u64 = 250_000;
/// C: the resident submit-result await poll timeout (30000 ms).
pub const SUBMIT_RESULT_TIMEOUT_NS: u64 = 30_000_000_000;

/// C: `SPARK_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT`, the pipeline
/// slot bound passed to packet validation.
pub const MAX_PIPELINE_SLOT_COUNT: u32 = 1024;

// ---------------------------------------------------------------------------
// Configuration and argument parsing
// ---------------------------------------------------------------------------

/// C: `SparkRingDaemonConfig`. Path options are `Option`s mirroring the C
/// NULL-or-set pointers.
#[derive(Debug, Clone)]
pub struct RankDaemonConfig {
    pub rank_index: u32,
    pub rank_is_set: bool,
    pub moe_pack_root: Option<String>,
    pub model_quantization_mode: QuantizationMode,
    pub stagepack_root: Option<String>,
    pub transport_shared_object_path: Option<String>,
    pub driver_path: Option<String>,
    pub node_context_builder_shared_object_path: Option<String>,
    pub embedding_pack_path: Option<String>,
    pub cuda_resident_socket_path: Option<String>,
    pub program_name: String,
    pub node_target: String,
    pub max_active_sequence_count: u32,
    pub port_base: u32,
    pub final_event_bind_address: String,
    pub final_event_return_host: String,
    pub own_final_event: bool,
    pub transport_busy_poll: bool,
}

impl Default for RankDaemonConfig {
    /// C defaults (applied by the C configuration setup): program
    /// "glm52.ring.rank.production", bind 0.0.0.0, return host "spark0",
    /// max-active 1024, port base 52100, quantization fp8.
    fn default() -> Self {
        Self {
            rank_index: 0,
            rank_is_set: false,
            moe_pack_root: None,
            model_quantization_mode: QuantizationMode::Fp8E4m3,
            stagepack_root: None,
            transport_shared_object_path: None,
            driver_path: None,
            node_context_builder_shared_object_path: None,
            embedding_pack_path: None,
            cuda_resident_socket_path: None,
            program_name: "glm52.ring.rank.production".to_string(),
            node_target: String::new(),
            max_active_sequence_count: 1024,
            port_base: super::ring_runtime::DEFAULT_PORT_BASE,
            final_event_bind_address: "0.0.0.0".to_string(),
            final_event_return_host: "spark0".to_string(),
            own_final_event: false,
            transport_busy_poll: false,
        }
    }
}

/// C: `SparkRingDaemonParseArguments`. The return codes are the C's:
/// `Err(-1)` unknown/malformed argument, `Err(-2)` missing required option,
/// `Err(-3)` bad quantization mode. `args` excludes the program name.
pub fn parse_arguments(args: &[String]) -> std::result::Result<RankDaemonConfig, i32> {
    let mut configuration = RankDaemonConfig::default();
    let mut index = 0usize;
    let string_arg = |index: usize| -> std::result::Result<String, i32> {
        args.get(index + 1).cloned().ok_or(-1)
    };
    while index < args.len() {
        match args[index].as_str() {
            "--rank" => {
                let value = string_arg(index)?;
                configuration.rank_index = value.parse().map_err(|_| -1)?;
                configuration.rank_is_set = true;
                index += 1;
            }
            "--moe-pack-root" => {
                configuration.moe_pack_root = Some(string_arg(index)?);
                index += 1;
            }
            "--model-quantization" => {
                let value = string_arg(index)?;
                configuration.model_quantization_mode =
                    QuantizationMode::parse(&value).map_err(|_| -3)?;
                index += 1;
            }
            "--stagepack-root" => {
                configuration.stagepack_root = Some(string_arg(index)?);
                index += 1;
            }
            "--transport-so" => {
                configuration.transport_shared_object_path = Some(string_arg(index)?);
                index += 1;
            }
            "--driver-so" => {
                configuration.driver_path = Some(string_arg(index)?);
                index += 1;
            }
            "--node-context-builder-so" => {
                configuration.node_context_builder_shared_object_path = Some(string_arg(index)?);
                index += 1;
            }
            "--embedding-pack" => {
                configuration.embedding_pack_path = Some(string_arg(index)?);
                index += 1;
            }
            "--cuda-resident-socket" => {
                configuration.cuda_resident_socket_path = Some(string_arg(index)?);
                index += 1;
            }
            "--program" => {
                configuration.program_name = string_arg(index)?;
                index += 1;
            }
            "--node-target" => {
                configuration.node_target = string_arg(index)?;
                index += 1;
            }
            "--max-active" => {
                let value = string_arg(index)?;
                configuration.max_active_sequence_count = value.parse().map_err(|_| -1)?;
                index += 1;
            }
            "--port-base" => {
                let value = string_arg(index)?;
                configuration.port_base = value.parse().map_err(|_| -1)?;
                index += 1;
            }
            "--final-event-bind" => {
                configuration.final_event_bind_address = string_arg(index)?;
                index += 1;
            }
            "--final-event-return-host" => {
                configuration.final_event_return_host = string_arg(index)?;
                index += 1;
            }
            "--own-final-event" => configuration.own_final_event = true,
            "--transport-busy-poll" => configuration.transport_busy_poll = true,
            _ => return Err(-1),
        }
        index += 1;
    }
    if !configuration.rank_is_set {
        return Err(-2);
    }
    if configuration.cuda_resident_socket_path.is_some() {
        return Ok(configuration);
    }
    if configuration.moe_pack_root.is_none()
        || configuration.stagepack_root.is_none()
        || configuration.transport_shared_object_path.is_none()
        || configuration.driver_path.is_none()
        || configuration.node_context_builder_shared_object_path.is_none()
        || configuration.embedding_pack_path.is_none()
    {
        return Err(-2);
    }
    Ok(configuration)
}

// ---------------------------------------------------------------------------
// Work queue
// ---------------------------------------------------------------------------

/// C: `SPARK_RING_DAEMON_WORK_STATE_*`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkState {
    Ready,
    WaitingForward,
    WaitingSubmit,
}

/// C: `SPARK_RING_DAEMON_WORK_PHASE_*` (0-based local phase from flags, in
/// the same order as `work_control::transaction_phase`).
pub fn work_packet_phase(packet: &WorkControlPacket) -> u32 {
    // C order: RELEASE, PREFILL, VERIFY, DECODE with values 3,0,2,1
    // (local enum `PREFILL=0, DECODE=1, VERIFY=2, RELEASE=3`).
    work_control::transaction_phase(packet) - 1
}

struct QueuedWork {
    packet: WorkControlPacket,
    wire_bytes: Vec<u8>,
    fingerprint32: u32,
    state: WorkState,
    submitted: bool,
    forwarded: bool,
}

/// C: `SparkRingDaemonWorkPacketMatches` (exact byte equality over
/// `[0, descriptor_bytes)`).
fn packet_matches(left: &WorkControlPacket, left_wire: &[u8], right_wire: &[u8]) -> bool {
    left.descriptor_bytes as usize == right_wire.len() && left_wire == right_wire
}

// ---------------------------------------------------------------------------
// Engine seams
// ---------------------------------------------------------------------------

/// A completion delivered by the engine (C: the completion callback's
/// driver completion plus an optional dSPARK draft result).
#[derive(Debug, Clone)]
pub struct EngineCompletion {
    pub completion: DriverCompletion,
    pub draft: Option<DsparkDraftResult>,
}

/// C: what a successful `submit_work` produced — completions the engine
/// fired synchronously (builder callback) or that arrived interleaved while
/// awaiting the resident submit result.
#[derive(Debug, Default)]
pub struct SubmitOutcome {
    pub completions: Vec<EngineCompletion>,
}

/// Submit seam: the C builder's `submit_work` callback interface and the
/// CUDA-resident socket submit both implement this.
pub trait SubmitEngine {
    /// `Ok` — the engine accepted the work (completions fired synchronously
    /// ride back in the outcome). `Err(Busy)` — engine backpressure; the
    /// daemon rolls back completion ownership and retries later. Any other
    /// `Err` fails the head work.
    fn submit(&mut self, packet: &WorkControlPacket) -> Result<SubmitOutcome>;
}

/// Forward seam: writing the work packet to the next rank and awaiting its
/// acknowledgement. `Err(Busy)` / `Err(RouteNotFound)` park the head in
/// `WAITING_FORWARD`; `Ok(())` marks it forwarded.
pub trait WorkForwarder {
    fn forward(
        &mut self,
        packet_wire: &[u8],
        identity: &Identity,
        packet_fingerprint: u64,
    ) -> Result<()>;
}

/// Forwarder for a rank with no downstream (C: `HAS_NEXT` clear →
/// `ForwardWork` returns OK immediately).
pub struct NoForwarder;

impl WorkForwarder for NoForwarder {
    fn forward(&mut self, _: &[u8], _: &Identity, _: u64) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Inflight tracking
// ---------------------------------------------------------------------------

/// C: the inflight completion key — fingerprint of the three words
/// (`request_id`, `sequence_id`, `sequence_position`) via
/// `SparkWorkTransactionFingerprintBytes` (C
/// `SparkRingDaemonInflightCompletionHash`).
pub fn inflight_completion_hash(request_id: u64, sequence_id: u64, sequence_position: u64) -> u64 {
    let mut words = [0u8; 24];
    words[0..8].copy_from_slice(&request_id.to_le_bytes());
    words[8..16].copy_from_slice(&sequence_id.to_le_bytes());
    words[16..24].copy_from_slice(&sequence_position.to_le_bytes());
    spark_sched::work_control::work_transaction::fingerprint_bytes(&words)
}

/// C: `SparkRingDaemonInflightTransaction`.
#[derive(Debug, Clone)]
struct InflightTransaction {
    identity: Identity,
    packet_hash: u64,
    remaining_completion_count: u32,
    terminal_state: u32,
    terminal_status: u32,
}

/// C: `SparkRingDaemonInflightCompletion`.
#[derive(Debug, Clone, Copy)]
struct InflightCompletion {
    owner: u64,
    request_generation: u64,
}

/// C: `SparkRingDaemonDriverCompletionRecord`.
#[derive(Debug, Clone)]
pub struct DriverCompletionRecord {
    pub completion: DriverCompletion,
    pub draft: Option<DsparkDraftResult>,
}

/// Daemon counters (C: the `work_*` / `cuda_resident_*` fields printed by
/// `PrintReady`).
#[derive(Debug, Default, Clone)]
pub struct DaemonCounters {
    pub work_receive_count: u64,
    pub work_duplicate_count: u64,
    pub work_submit_count: u64,
    pub work_error_count: u64,
    pub work_deferred_count: u64,
    pub work_wake_count: u64,
    pub work_forward_count: u64,
    pub driver_completion_count: u64,
    pub cuda_resident_submit_count: u64,
    pub cuda_resident_completion_count: u64,
    pub cuda_resident_error_count: u64,
    pub timer_wake_count: u64,
}

/// The rank daemon runtime state (C: `SparkRingDaemonRuntime`, minus the
/// sockets/fds — those live behind the seams and the poll-loop wiring,
/// which is ported separately).
pub struct RankDaemonCore {
    pub work_config: WorkControlConfig,
    pub geometry: RingModelGeometry,
    pub rank_plan: RankPlan,
    queue: VecDeque<QueuedWork>,
    pub ledger: TransactionLedger,
    inflight_transactions: HashMap<u64, InflightTransaction>,
    inflight_completions: HashMap<u64, InflightCompletion>,
    driver_completion_queue: VecDeque<DriverCompletionRecord>,
    driver_completion_queue_overflow: bool,
    final_event_queue: VecDeque<FinalEvent>,
    pub counters: DaemonCounters,
    /// C: `driver_inflight_count` (+ open/warned diagnostics).
    pub driver_inflight_count: u32,
    driver_inflight_open_ns: u64,
    driver_inflight_warned: bool,
}

impl RankDaemonCore {
    pub fn new(
        work_config: WorkControlConfig,
        geometry: RingModelGeometry,
        rank_plan: RankPlan,
    ) -> Result<Self> {
        Ok(Self {
            work_config,
            geometry,
            rank_plan,
            queue: VecDeque::new(),
            ledger: TransactionLedger::new(TRANSACTION_LEDGER_CAPACITY)?,
            inflight_transactions: HashMap::new(),
            inflight_completions: HashMap::new(),
            driver_completion_queue: VecDeque::new(),
            driver_completion_queue_overflow: false,
            final_event_queue: VecDeque::new(),
            counters: DaemonCounters::default(),
            driver_inflight_count: 0,
            driver_inflight_open_ns: 0,
            driver_inflight_warned: false,
        })
    }

    pub fn work_queue_count(&self) -> usize {
        self.queue.len()
    }

    pub fn driver_completion_queue_count(&self) -> usize {
        self.driver_completion_queue.len()
    }

    pub fn final_event_queue_count(&self) -> usize {
        self.final_event_queue.len()
    }

    pub fn final_events(&self) -> &VecDeque<FinalEvent> {
        &self.final_event_queue
    }

    /// Test/inspection view of a queued packet (C tests index
    /// `work_queue[]` directly).
    pub fn queued_packet(&self, index: usize) -> Option<&WorkControlPacket> {
        self.queue.get(index).map(|entry| &entry.packet)
    }

    /// Test/inspection view of a queued work state.
    pub fn queued_state(&self, index: usize) -> Option<WorkState> {
        self.queue.get(index).map(|entry| entry.state)
    }

    /// Number of active inflight completion mappings (C test helper
    /// `CountInflightCompletionMappings`).
    pub fn inflight_completion_mapping_count(&self) -> usize {
        self.inflight_completions.len()
    }

    /// C: `SparkRingDaemonQueueWork` (`Ok(true)` — newly queued;
    /// `Ok(false)` — exact-byte duplicate).
    pub fn queue_work(&mut self, packet: &WorkControlPacket) -> Result<bool> {
        let wire_bytes = wire::packet_to_wire(&self.work_config, packet)?;
        let fingerprint32 = wire::packet_dedup_hash32(&wire_bytes);
        self.counters.work_receive_count += 1;
        if let Some(existing) = self.queue.iter_mut().find(|entry| {
            entry.fingerprint32 == fingerprint32
                && packet_matches(&entry.packet, &entry.wire_bytes, &wire_bytes)
        }) {
            if !existing.submitted {
                existing.packet = packet.clone();
                existing.forwarded = false;
                existing.state = WorkState::Ready;
            }
            self.counters.work_duplicate_count += 1;
            return Ok(false);
        }
        if self.queue.len() >= WORK_QUEUE_CAPACITY {
            return Err(SparkStatus::CapacityExceeded);
        }
        self.queue.push_back(QueuedWork {
            packet: packet.clone(),
            wire_bytes,
            fingerprint32,
            state: WorkState::Ready,
            submitted: false,
            forwarded: false,
        });
        Ok(true)
    }

    /// C: `SparkRingDaemonPopWork`.
    fn pop_work(&mut self) {
        self.queue.pop_front();
    }

    /// C: `SparkRingDaemonDeferWork` (rotate head to tail, carrying state).
    fn defer_work(&mut self) {
        if self.queue.len() <= 1 {
            return;
        }
        if let Some(head) = self.queue.pop_front() {
            self.queue.push_back(head);
        }
    }

    /// C: `SparkRingDaemonWakeDeferredWork`.
    pub fn wake_deferred_work(&mut self) {
        for entry in self.queue.iter_mut() {
            if entry.state != WorkState::Ready {
                entry.state = WorkState::Ready;
                self.counters.work_wake_count += 1;
            }
        }
    }

    /// C: `SparkRingDaemonHasWaitingWork`.
    pub fn has_waiting_work(&self) -> bool {
        self.queue.iter().any(|entry| entry.state != WorkState::Ready)
    }

    /// C: `SparkRingDaemonHasQueuedDependency` +
    /// `SparkRingDaemonCandidateIsDependency` (the C's dependency hash table
    /// is a collision index over the packet's lanes; comparing lane pairs
    /// directly is equivalent).
    pub fn has_queued_dependency(&self, packet: &WorkControlPacket) -> bool {
        if self.queue.len() <= 1 {
            return false;
        }
        let packet_phase = work_packet_phase(packet);
        let packet_is_release = packet.flags & FLAG_RELEASE_SEQUENCES != 0;
        self.queue.iter().any(|entry| {
            let candidate = &entry.packet;
            if std::ptr::eq(candidate, packet)
                || candidate.control_generation != packet.control_generation
            {
                return false;
            }
            let candidate_phase = work_packet_phase(candidate);
            let candidate_is_release = candidate.flags & FLAG_RELEASE_SEQUENCES != 0;
            candidate.lanes[..candidate.lane_count as usize].iter().any(|candidate_lane| {
                let Some(packet_lane) = packet.lanes[..packet.lane_count as usize]
                    .iter()
                    .find(|lane| lane.sequence_id == candidate_lane.sequence_id)
                else {
                    return false;
                };
                if packet_is_release && !candidate_is_release {
                    return true;
                }
                candidate_lane.sequence_position < packet_lane.sequence_position
                    || (candidate_lane.sequence_position == packet_lane.sequence_position
                        && candidate_phase < packet_phase)
            })
        })
    }

    /// C: `SparkRingDaemonTransitionPacket`.
    pub fn transition_packet(
        &mut self,
        packet: &WorkControlPacket,
        next_state: u32,
        terminal_status: SparkStatus,
    ) -> Result<()> {
        let identity = get_transaction_identity(packet).map_err(from_work_control)?;
        let packet_hash = packet_fingerprint(packet, &self.work_config);
        if packet_hash == 0 {
            return Err(SparkStatus::InternalError);
        }
        self.ledger.distributed_transition(&identity, packet_hash, next_state, terminal_status)
    }

    /// C: `SparkRingDaemonFailHeadWork`.
    fn fail_head_work(&mut self, status: SparkStatus) {
        let Some(head) = self.queue.front() else {
            return;
        };
        let packet = head.packet.clone();
        let _ = self.transition_packet(&packet, STATE_FAILED, status);
        self.counters.work_error_count += 1;
        eprintln!(
            "rank_work_failed status={} transaction={} request={} sequence={} position={} queued={}",
            status as u32,
            packet.step_generation,
            packet.request_id,
            packet.sequence_id,
            packet.sequence_position,
            self.queue.len()
        );
        self.pop_work();
    }

    /// C: `SparkRingDaemonHandleWork` — returns the carried `SparkStatus`
    /// exactly like the C (`Ok`, `Duplicate`, a terminal status, or an
    /// error), not a Result.
    pub fn handle_work(&mut self, packet: &WorkControlPacket) -> SparkStatus {
        if let Err(error) = validate_packet(
            packet,
            &self.work_config,
            self.rank_plan.execution_row_capacity,
            MAX_PIPELINE_SLOT_COUNT,
        ) {
            return from_work_control(error);
        }
        let identity = match get_transaction_identity(packet) {
            Ok(identity) => identity,
            Err(error) => return from_work_control(error),
        };
        let packet_hash = packet_fingerprint(packet, &self.work_config);
        if packet_hash == 0 {
            return SparkStatus::InternalError;
        }
        if self.queue.len() >= WORK_QUEUE_CAPACITY {
            return SparkStatus::CapacityExceeded;
        }
        let observe = self.ledger.distributed_observe(&identity, packet_hash);
        if observe == Err(SparkStatus::Duplicate) {
            self.counters.work_duplicate_count += 1;
            if let Ok(entry) = self.ledger.find(&identity) {
                if entry.state == STATE_FAILED || entry.state == STATE_CANCELLED {
                    return status_from_code(entry.terminal_status)
                        .unwrap_or(SparkStatus::InternalError);
                }
            }
            return SparkStatus::Duplicate;
        }
        if let Err(status) = observe {
            return status;
        }
        if let Err(status) = self.ledger.distributed_transition(
            &identity,
            packet_hash,
            STATE_ACCEPTED,
            SparkStatus::Pending,
        ) {
            return status;
        }
        match self.queue_work(packet) {
            Ok(_) => SparkStatus::Ok,
            Err(status) => {
                let _ = self.ledger.distributed_transition(
                    &identity,
                    packet_hash,
                    STATE_FAILED,
                    status,
                );
                status
            }
        }
    }

    /// C: `SparkRingDaemonRegisterInflightTransaction`. RELEASE packets
    /// short-circuit (no ownership) exactly like the C.
    pub fn register_inflight_transaction(
        &mut self,
        packet: &WorkControlPacket,
    ) -> Result<Option<u64>> {
        if packet.flags & FLAG_RELEASE_SEQUENCES != 0 {
            return Ok(None);
        }
        if packet.lane_count == 0 || packet.lane_count > self.work_config.max_lane_count {
            return Err(SparkStatus::InvalidArgument);
        }
        let identity = get_transaction_identity(packet).map_err(from_work_control)?;
        let packet_hash = packet_fingerprint(packet, &self.work_config);
        if packet_hash == 0 {
            return Err(SparkStatus::InternalError);
        }
        if self.inflight_transactions.len() >= INFLIGHT_TRANSACTION_CAPACITY {
            return Err(SparkStatus::Busy);
        }
        if self.inflight_completions.len() + packet.lane_count as usize
            > INFLIGHT_COMPLETION_CAPACITY
        {
            return Err(SparkStatus::CapacityExceeded);
        }
        let transaction_key = wire::identity_fingerprint(&identity);
        if self.inflight_transactions.contains_key(&transaction_key) {
            return Err(SparkStatus::ValidationFailed);
        }
        let transaction = InflightTransaction {
            identity,
            packet_hash,
            remaining_completion_count: packet.lane_count,
            terminal_state: 0,
            terminal_status: SparkStatus::Pending as u32,
        };
        // Insert completion mappings; roll back on any failure (C:
        // RemoveTransactionCompletionMappings + ReleaseInflightTransaction).
        let mut inserted = Vec::with_capacity(packet.lane_count as usize);
        for lane in packet.lanes[..packet.lane_count as usize].iter() {
            let key =
                inflight_completion_hash(lane.request_id, lane.sequence_id, lane.sequence_position);
            if self.inflight_completions.contains_key(&key) {
                for key in inserted {
                    self.inflight_completions.remove(&key);
                }
                return Err(SparkStatus::CapacityExceeded);
            }
            self.inflight_completions.insert(
                key,
                InflightCompletion {
                    owner: transaction_key,
                    request_generation: lane.request_generation,
                },
            );
            inserted.push(key);
        }
        if self.driver_inflight_count == 0 {
            self.driver_inflight_open_ns = super::net::monotonic_ns();
            self.driver_inflight_warned = false;
        }
        if packet.lane_count > u32::MAX - self.driver_inflight_count {
            for key in inserted {
                self.inflight_completions.remove(&key);
            }
            return Err(SparkStatus::CapacityExceeded);
        }
        self.driver_inflight_count += packet.lane_count;
        self.inflight_transactions.insert(transaction_key, transaction);
        Ok(Some(transaction_key))
    }

    /// C: `SparkRingDaemonCancelInflightTransaction`.
    pub fn cancel_inflight_transaction(&mut self, transaction_key: u64) -> Result<()> {
        if !self.inflight_transactions.contains_key(&transaction_key) {
            return Err(SparkStatus::NotFound);
        }
        let owner_keys: Vec<u64> = self
            .inflight_completions
            .iter()
            .filter(|(_, entry)| entry.owner == transaction_key)
            .map(|(key, _)| *key)
            .collect();
        let removed_count = owner_keys.len() as u32;
        for key in owner_keys {
            self.inflight_completions.remove(&key);
        }
        self.driver_inflight_count = self.driver_inflight_count.saturating_sub(removed_count);
        self.inflight_transactions.remove(&transaction_key);
        if self.driver_inflight_count == 0 {
            self.driver_inflight_warned = false;
        }
        Ok(())
    }

    /// C: `SparkRingDaemonQueueDriverCompletion` (the completion callback's
    /// cross-thread handoff; single-threaded here, wake handled by the poll
    /// loop wiring).
    pub fn queue_driver_completion(&mut self, record: DriverCompletionRecord) -> Result<()> {
        if let Some(draft) = &record.draft {
            if !draft.has_valid_descriptor() {
                return Err(SparkStatus::ValidationFailed);
            }
        }
        if self.driver_completion_queue.len() >= DRIVER_COMPLETION_QUEUE_CAPACITY {
            self.driver_completion_queue_overflow = true;
            return Err(SparkStatus::CapacityExceeded);
        }
        self.driver_completion_queue.push_back(record);
        Ok(())
    }

    /// C: `SparkRingDaemonValidateDriverCompletionRecord`.
    fn validate_driver_completion_record(record: &DriverCompletionRecord) -> Result<()> {
        let completion = &record.completion;
        if completion.request_id == 0
            || completion.sequence_id == 0
            || completion.token_count > resident_ipc::COMPLETION_TOKEN_CAPACITY as u32
            || completion.draft_token_count > resident_ipc::COMPLETION_DRAFT_TOKEN_CAPACITY as u32
        {
            return Err(SparkStatus::ValidationFailed);
        }
        let has_token_ids = completion.completion_flags & DRIVER_COMPLETION_FLAG_TOKEN_IDS != 0;
        let has_draft_tokens =
            completion.completion_flags & DRIVER_COMPLETION_FLAG_DRAFT_TOKEN_IDS != 0;
        if has_token_ids != (completion.token_count != 0)
            || has_draft_tokens != (completion.draft_token_count != 0)
        {
            return Err(SparkStatus::ValidationFailed);
        }
        if let Some(draft) = &record.draft {
            if !draft.has_valid_descriptor() {
                return Err(SparkStatus::ValidationFailed);
            }
        }
        Ok(())
    }

    /// C: `SparkRingDaemonCompletionNeedsFinalEvent`.
    fn completion_needs_final_event(
        &self,
        record: &DriverCompletionRecord,
        effective_status: SparkStatus,
        transaction_terminal_state: u32,
    ) -> bool {
        if self.rank_plan.flags & RANK_FLAG_FINAL_STAGE == 0 || transaction_terminal_state != 0 {
            return false;
        }
        if effective_status != SparkStatus::Ok {
            return true;
        }
        record.completion.completion_flags & DRIVER_COMPLETION_FLAG_TOKEN_IDS != 0
            && record.completion.token_count != 0
    }

    /// C: `SparkRingDaemonBuildFinalEvent`.
    fn build_final_event(
        transaction: &InflightTransaction,
        inflight_completion: &InflightCompletion,
        record: &DriverCompletionRecord,
        effective_status: SparkStatus,
    ) -> FinalEvent {
        let completion = &record.completion;
        let mut event = FinalEvent {
            magic: FINAL_EVENT_MAGIC,
            descriptor_bytes: FINAL_EVENT_DESCRIPTOR_BYTES,
            status: effective_status as u32,
            program_id: completion.program_id,
            driver_dispatch_slot: completion.driver_dispatch_slot,
            accepted_token_count: completion.accepted_token_count,
            request_id: completion.request_id,
            sequence_id: completion.sequence_id,
            sequence_position: completion.sequence_position,
            service_time_ns: completion.service_time_ns,
            control_generation: transaction.identity.control_generation,
            transaction_id: transaction.identity.transaction_id,
            dispatch_generation: transaction.identity.dispatch_generation,
            request_generation: inflight_completion.request_generation,
            step_generation: transaction.identity.step_generation,
            step_chunk_index: transaction.identity.step_chunk_index,
            step_chunk_count: transaction.identity.step_chunk_count,
            transaction_phase: transaction.identity.transaction_phase,
            ..FinalEvent::default()
        };
        if effective_status != SparkStatus::Ok {
            return event;
        }
        event.completion_flags = completion.completion_flags;
        event.token_count = completion.token_count;
        event.token_ids[..completion.token_count as usize]
            .copy_from_slice(&completion.token_ids[..completion.token_count as usize]);
        event.draft_token_count = completion.draft_token_count;
        event.draft_token_ids[..completion.draft_token_count as usize]
            .copy_from_slice(&completion.draft_token_ids[..completion.draft_token_count as usize]);
        if let Some(draft) = &record.draft {
            event.extension_flags |= FINAL_EVENT_FLAG_DSPARK_DRAFT;
            event.dspark_draft = draft.clone();
        }
        event
    }

    /// C: `SparkRingDaemonQueueFinalEvent`.
    pub fn queue_final_event(&mut self, event: FinalEvent) -> Result<()> {
        if self.final_event_queue.len() >= FINAL_EVENT_QUEUE_CAPACITY {
            return Err(SparkStatus::CapacityExceeded);
        }
        self.final_event_queue.push_back(event);
        Ok(())
    }

    /// C: `SparkRingDaemonPopFinalEvent`.
    pub fn pop_final_event(&mut self) {
        self.final_event_queue.pop_front();
    }

    /// C: `SparkRingDaemonProcessDriverCompletion`.
    pub fn process_driver_completion(&mut self, record: &DriverCompletionRecord) -> Result<()> {
        let completion_key = inflight_completion_hash(
            record.completion.request_id,
            record.completion.sequence_id,
            record.completion.sequence_position,
        );
        let Some(inflight_completion) = self.inflight_completions.get(&completion_key).copied()
        else {
            // C: NOT_FOUND maps to VALIDATION_FAILED.
            return Err(SparkStatus::ValidationFailed);
        };
        let Some(transaction) = self.inflight_transactions.get(&inflight_completion.owner) else {
            return Err(SparkStatus::ValidationFailed);
        };
        if transaction.remaining_completion_count == 0 {
            return Err(SparkStatus::ValidationFailed);
        }
        let validation = Self::validate_driver_completion_record(record);
        let effective_status = match validation {
            Ok(()) => {
                status_from_code(record.completion.status).unwrap_or(SparkStatus::InternalError)
            }
            Err(status) => status,
        };
        let transaction =
            self.inflight_transactions.get(&inflight_completion.owner).expect("checked above");
        let transaction_terminal_state = transaction.terminal_state;
        let emit_final_event =
            self.completion_needs_final_event(record, effective_status, transaction_terminal_state);
        if emit_final_event && self.final_event_queue.len() >= FINAL_EVENT_QUEUE_CAPACITY {
            return Err(SparkStatus::Busy);
        }
        let transaction =
            self.inflight_transactions.get_mut(&inflight_completion.owner).expect("checked above");
        if transaction.terminal_state == 0 && effective_status != SparkStatus::Ok {
            transaction.terminal_state = STATE_FAILED;
            transaction.terminal_status = effective_status as u32;
        }
        let event = emit_final_event.then(|| {
            Self::build_final_event(transaction, &inflight_completion, record, effective_status)
        });
        let identity = transaction.identity;
        let packet_hash = transaction.packet_hash;
        let remaining_completion_count = transaction.remaining_completion_count - 1;
        if remaining_completion_count == 0 {
            let terminal_state = if transaction.terminal_state != 0 {
                transaction.terminal_state
            } else {
                STATE_COMMITTED
            };
            let terminal_status = if terminal_state == STATE_COMMITTED {
                SparkStatus::Ok
            } else {
                status_from_code(transaction.terminal_status).unwrap_or(SparkStatus::InternalError)
            };
            let status = self.ledger.distributed_transition(
                &identity,
                packet_hash,
                terminal_state,
                terminal_status,
            );
            if status != Ok(()) && status != Err(SparkStatus::Duplicate) {
                return status;
            }
        }
        self.inflight_completions.remove(&completion_key);
        let transaction =
            self.inflight_transactions.get_mut(&inflight_completion.owner).expect("checked above");
        transaction.remaining_completion_count = remaining_completion_count;
        if self.driver_inflight_count != 0 {
            self.driver_inflight_count -= 1;
        }
        self.counters.driver_completion_count += 1;
        if self.driver_inflight_count == 0 {
            self.driver_inflight_warned = false;
        } else {
            self.driver_inflight_open_ns = super::net::monotonic_ns();
        }
        if remaining_completion_count == 0 {
            self.inflight_transactions.remove(&inflight_completion.owner);
        }
        if let Some(event) = event {
            self.queue_final_event(event)?;
        }
        Ok(())
    }

    /// C: `SparkRingDaemonPumpDriverCompletions`. Returns progress (0/1).
    pub fn pump_driver_completions(&mut self) -> u32 {
        let mut progress = 0u32;
        if self.driver_completion_queue_overflow {
            self.driver_completion_queue_overflow = false;
            eprintln!("rank_driver_completion_queue_overflow rank={}", self.rank_plan.rank_index);
            self.counters.work_error_count += 1;
            // C: `SparkRingDaemonRunning = 0` — the port surfaces this as a
            // hard condition for the poll loop; see the port ledger.
            return 1;
        }
        loop {
            let Some(record) = self.driver_completion_queue.front().cloned() else {
                return progress;
            };
            match self.process_driver_completion(&record) {
                Err(SparkStatus::Busy) => return progress,
                Err(status) => {
                    self.counters.work_error_count += 1;
                    eprintln!(
                        "rank_driver_completion_rejected rank={} status={} request={} sequence={} position={}",
                        self.rank_plan.rank_index,
                        status as u32,
                        record.completion.request_id,
                        record.completion.sequence_id,
                        record.completion.sequence_position
                    );
                }
                Ok(()) => {}
            }
            self.driver_completion_queue.pop_front();
            progress = 1;
        }
    }

    /// C: `SparkRingDaemonPumpQueuedWork`. Returns progress (0/1).
    pub fn pump_queued_work(
        &mut self,
        forwarder: &mut dyn WorkForwarder,
        engine: &mut dyn SubmitEngine,
    ) -> Result<u32> {
        if self.queue.is_empty() {
            return Ok(0);
        }
        let mut attempts = self.queue.len();
        while attempts != 0 {
            attempts -= 1;
            let head = self.queue.front().expect("attempts bounded by queue len");
            if head.state == WorkState::WaitingSubmit {
                self.defer_work();
                continue;
            }
            let has_next = self.rank_plan.flags & RANK_FLAG_HAS_NEXT != 0;
            let head = self.queue.front().expect("attempts bounded by queue len");
            if head.state != WorkState::WaitingForward && self.has_queued_dependency(&head.packet) {
                self.defer_work();
                continue;
            }
            let mut forward_done = !has_next || head.forwarded;
            if !forward_done {
                let packet = head.packet.clone();
                let status = self.transition_packet(&packet, STATE_ACCEPTED, SparkStatus::Pending);
                if status != Ok(()) && status != Err(SparkStatus::Duplicate) {
                    self.fail_head_work(status.unwrap_err());
                    return Ok(1);
                }
                let identity = get_transaction_identity(&packet).map_err(from_work_control)?;
                let packet_hash = packet_fingerprint(&packet, &self.work_config);
                let wire_bytes =
                    self.queue.front().expect("attempts bounded by queue len").wire_bytes.clone();
                match forwarder.forward(&wire_bytes, &identity, packet_hash) {
                    Ok(()) => {
                        let head = self.queue.front_mut().expect("queue unchanged");
                        head.forwarded = true;
                        head.state = WorkState::Ready;
                        self.counters.work_forward_count += 1;
                        forward_done = true;
                    }
                    Err(SparkStatus::Busy) | Err(SparkStatus::RouteNotFound) => {
                        let head = self.queue.front_mut().expect("queue unchanged");
                        head.state = WorkState::WaitingForward;
                        self.counters.work_deferred_count += 1;
                        return Ok(0);
                    }
                    Err(status) => {
                        self.fail_head_work(status);
                        return Ok(1);
                    }
                }
            }
            let packet = self.queue.front().expect("attempts bounded by queue len").packet.clone();
            let status = self.transition_packet(&packet, STATE_ACCEPTED, SparkStatus::Pending);
            if status != Ok(()) && status != Err(SparkStatus::Duplicate) {
                self.fail_head_work(status.unwrap_err());
                return Ok(1);
            }
            let head_submitted =
                self.queue.front().expect("attempts bounded by queue len").submitted;
            if !head_submitted {
                let transaction_key = match self.register_inflight_transaction(&packet) {
                    Ok(key) => key,
                    Err(SparkStatus::Busy) | Err(SparkStatus::CapacityExceeded) => {
                        let head = self.queue.front_mut().expect("queue unchanged");
                        head.state = WorkState::WaitingSubmit;
                        self.counters.work_deferred_count += 1;
                        self.defer_work();
                        continue;
                    }
                    Err(status) => {
                        self.fail_head_work(status);
                        return Ok(1);
                    }
                };
                match engine.submit(&packet) {
                    Ok(outcome) => {
                        self.counters.work_submit_count += 1;
                        self.queue.front_mut().expect("queue unchanged").submitted = true;
                        for completion in outcome.completions {
                            if self
                                .queue_driver_completion(DriverCompletionRecord {
                                    completion: completion.completion,
                                    draft: completion.draft,
                                })
                                .is_err()
                            {
                                self.counters.cuda_resident_error_count += 1;
                            }
                        }
                        let status =
                            self.transition_packet(&packet, STATE_EXECUTING, SparkStatus::Pending);
                        if status != Ok(()) && status != Err(SparkStatus::Duplicate) {
                            eprintln!(
                                "rank_work_transition_failed status={} request={} sequence={} position={}",
                                status.unwrap_err() as u32,
                                packet.request_id,
                                packet.sequence_id,
                                packet.sequence_position
                            );
                            self.counters.work_error_count += 1;
                            return Err(status.unwrap_err());
                        }
                        if packet.flags & FLAG_RELEASE_SEQUENCES != 0 {
                            let status =
                                self.transition_packet(&packet, STATE_COMMITTED, SparkStatus::Ok);
                            if status != Ok(()) && status != Err(SparkStatus::Duplicate) {
                                self.counters.work_error_count += 1;
                                return Err(status.unwrap_err());
                            }
                        }
                    }
                    Err(SparkStatus::Busy) => {
                        if let Some(key) = transaction_key {
                            let _ = self.cancel_inflight_transaction(key);
                        }
                        let head = self.queue.front_mut().expect("queue unchanged");
                        head.state = WorkState::WaitingSubmit;
                        self.counters.work_deferred_count += 1;
                        self.defer_work();
                        continue;
                    }
                    Err(status) => {
                        if let Some(key) = transaction_key {
                            let _ = self.cancel_inflight_transaction(key);
                        }
                        self.fail_head_work(status);
                        return Ok(1);
                    }
                }
            }
            if forward_done && self.queue.front().expect("attempts bounded by queue len").submitted
            {
                self.pop_work();
                return Ok(1);
            }
        }
        Ok(0)
    }

    /// C: `SparkRingDaemonNextTimerNs` (kept on the core so the poll-loop
    /// wiring can compute its ppoll timeout).
    pub fn next_timer_ns(
        &self,
        work_output_retry_mono_ns: u64,
        final_event_retry_mono_ns: u64,
    ) -> u64 {
        let now_ns = super::net::monotonic_ns();
        let mut next_ns = 0u64;
        if self.driver_inflight_count != 0 {
            next_ns = now_ns + RUNNER_PROGRESS_NS;
        }
        if self.has_waiting_work() {
            next_ns = super::net::min_nonzero_ns(next_ns, now_ns + RUNNER_PROGRESS_NS);
        }
        if !self.queue.is_empty() {
            next_ns = super::net::min_nonzero_ns(next_ns, work_output_retry_mono_ns);
        }
        if !self.final_event_queue.is_empty() {
            next_ns = super::net::min_nonzero_ns(next_ns, final_event_retry_mono_ns);
        }
        next_ns
    }
}

// ---------------------------------------------------------------------------
// CUDA-resident message pump (split reads)
// ---------------------------------------------------------------------------

/// A fully-read resident message (C: header + `cuda_resident_payload`).
#[derive(Debug, Clone)]
pub struct ResidentMessage {
    pub header: IpcHeader,
    pub payload: Vec<u8>,
}

/// C: the `cuda_resident_read_header(_offset)` / `cuda_resident_payload` /
/// `cuda_resident_read_payload_offset` read state and
/// `SparkRingDaemonReadResidentMessage` with `timeout_ms == 0` (the
/// nonblocking form; the blocking form loops with a deadline, see
/// [`ResidentMessagePump::read_message_blocking`]).
///
/// `Ok(None)` is the C's `SPARK_STATUS_BUSY`: the message is incomplete and
/// the tracked offsets are preserved. After a full message the offsets
/// reset (C: `SparkRingDaemonResetResidentRead`).
pub struct ResidentMessagePump {
    header_bytes: [u8; resident_ipc::IPC_HEADER_BYTES as usize],
    header_offset: usize,
    payload: Vec<u8>,
    payload_offset: usize,
    have_header: bool,
    /// C: `SPARK_CUDA_RESIDENT_IPC_MAX_CONTROL_PAYLOAD_BYTES`, supplied by
    /// the caller from its work-control config (no model constants).
    maximum_payload_bytes: u32,
}

impl ResidentMessagePump {
    pub fn new(maximum_payload_bytes: u32) -> Self {
        Self {
            header_bytes: [0; resident_ipc::IPC_HEADER_BYTES as usize],
            header_offset: 0,
            payload: Vec::new(),
            payload_offset: 0,
            have_header: false,
            maximum_payload_bytes,
        }
    }

    /// Current header-read offset (C: `cuda_resident_read_header_offset`).
    pub fn header_offset(&self) -> usize {
        self.header_offset
    }

    /// Current payload-read offset (C: `cuda_resident_read_payload_offset`).
    pub fn payload_offset(&self) -> usize {
        self.payload_offset
    }

    /// C: `SparkRingDaemonResetResidentRead`.
    fn reset(&mut self) {
        self.header_offset = 0;
        self.payload_offset = 0;
        self.payload.clear();
        self.have_header = false;
    }

    /// C: `SparkRingDaemonReadResidentBytes` over the two buffers, then
    /// `SparkRingDaemonReadResidentMessage`. EOF maps to `RouteNotFound`
    /// like the C read path.
    pub fn read_message(&mut self, link: &mut dyn Link) -> Result<Option<ResidentMessage>> {
        // Header phase.
        while self.header_offset < resident_ipc::IPC_HEADER_BYTES as usize {
            match link.read(&mut self.header_bytes[self.header_offset..]) {
                Ok(0) => return Err(SparkStatus::RouteNotFound),
                Ok(got) => self.header_offset += got,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return Ok(None),
                Err(_) => return Err(SparkStatus::RouteNotFound),
            }
        }
        if !self.have_header {
            let header = resident_ipc::decode_header(&self.header_bytes);
            resident_ipc::validate_header(&header, 0, self.maximum_payload_bytes)?;
            self.payload = vec![0u8; header.payload_bytes as usize];
            self.have_header = true;
        }
        // Payload phase.
        while self.payload_offset < self.payload.len() {
            match link.read(&mut self.payload[self.payload_offset..]) {
                Ok(0) => return Err(SparkStatus::RouteNotFound),
                Ok(got) => self.payload_offset += got,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return Ok(None),
                Err(_) => return Err(SparkStatus::RouteNotFound),
            }
        }
        let message = ResidentMessage {
            header: resident_ipc::decode_header(&self.header_bytes),
            payload: std::mem::take(&mut self.payload),
        };
        self.reset();
        Ok(Some(message))
    }

    /// Blocking read with a deadline (C: `ReadResidentBytes` with a poll
    /// timeout; the submit path uses 30000 ms). Sleeps briefly between
    /// nonblocking attempts.
    pub fn read_message_blocking(
        &mut self,
        link: &mut dyn Link,
        timeout_ns: u64,
    ) -> Result<ResidentMessage> {
        let deadline = super::net::monotonic_ns().saturating_add(timeout_ns);
        loop {
            if let Some(message) = self.read_message(link)? {
                return Ok(message);
            }
            if super::net::monotonic_ns() >= deadline {
                return Err(SparkStatus::Busy);
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}

/// C: `SparkRingDaemonHandleCudaResidentMessage` — validates and routes a
/// resident message. Returns `Ok(true)` when the message was consumed.
pub fn handle_resident_message(
    core: &mut RankDaemonCore,
    message: &ResidentMessage,
) -> Result<bool> {
    let header = &message.header;
    match header.kind {
        resident_ipc::IPC_KIND_COMPLETION => {
            let Some(completion_message) = resident_ipc::decode_completion(&message.payload) else {
                core.counters.cuda_resident_error_count += 1;
                return Ok(false);
            };
            if completion_message.descriptor_bytes != resident_ipc::IPC_COMPLETION_BYTES
                || completion_message.flags & !resident_ipc::COMPLETION_KNOWN_FLAGS != 0
            {
                core.counters.cuda_resident_error_count += 1;
                return Ok(false);
            }
            let draft =
                if completion_message.flags & resident_ipc::COMPLETION_FLAG_DSPARK_DRAFT != 0 {
                    if !completion_message.dspark_draft.has_valid_descriptor() {
                        core.counters.cuda_resident_error_count += 1;
                        return Ok(false);
                    }
                    Some(completion_message.dspark_draft.clone())
                } else {
                    None
                };
            core.counters.cuda_resident_completion_count += 1;
            if core
                .queue_driver_completion(DriverCompletionRecord {
                    completion: completion_message.completion.clone(),
                    draft,
                })
                .is_err()
            {
                core.counters.cuda_resident_error_count += 1;
            }
            Ok(true)
        }
        resident_ipc::IPC_KIND_STATS => {
            let Some(stats) = decode_stats_payload(&message.payload) else {
                core.counters.cuda_resident_error_count += 1;
                return Ok(true);
            };
            if stats.state == resident_ipc::STATE_FAILED {
                core.counters.cuda_resident_error_count += 1;
            }
            Ok(true)
        }
        resident_ipc::IPC_KIND_SUBMIT_RESULT => {
            let Some(result) = resident_ipc::decode_submit_result(&message.payload) else {
                core.counters.cuda_resident_error_count += 1;
                return Ok(false);
            };
            if result.status != SparkStatus::Ok as u32 {
                core.counters.cuda_resident_error_count += 1;
            }
            Ok(true)
        }
        _ => {
            core.counters.cuda_resident_error_count += 1;
            Ok(false)
        }
    }
}

fn decode_stats_payload(payload: &[u8]) -> Option<IpcStats> {
    let bytes: &[u8; resident_ipc::IPC_STATS_BYTES as usize] = payload.try_into().ok()?;
    Some(resident_ipc::decode_stats(bytes))
}

/// C: the contract checks in `SparkRingDaemonConnectCudaResident` after the
/// HELLO_ACK (the stderr mismatch line is kept as `eprintln!`).
pub fn validate_resident_contract(
    core: &RankDaemonCore,
    stats: &IpcStats,
    expected_moe_backend_kind: u32,
) -> Result<()> {
    if stats.state != resident_ipc::STATE_READY {
        return Err(SparkStatus::Busy);
    }
    let plan = &core.rank_plan;
    let mismatch = stats.logical_lane_capacity != plan.logical_lane_capacity
        || stats.execution_row_capacity != plan.execution_row_capacity
        || stats.kv_physical_block_capacity == 0
        || stats.kv_logical_block_capacity < stats.kv_physical_block_capacity
        || stats.model_quantization_mode != plan.quantization_mode.code()
        || stats.moe_backend_kind != expected_moe_backend_kind
        || stats.moe_bound_layer_count == 0
        || stats.moe_bound_layer_count != stats.moe_expected_layer_count
        || (stats.kv_nvme_enabled != 0
            && stats.kv_nvme_mode != resident_ipc::NVME_MODE_SYNCHRONOUS_FULL_HISTORY
            && stats.kv_nvme_mode != resident_ipc::NVME_MODE_BATCHED_COHORT_JIT
            && stats.kv_nvme_mode != resident_ipc::NVME_MODE_ASYNC_SELECTED_JIT)
        || (plan.logical_lane_capacity >= core.work_config.max_lane_count
            && (stats.kv_nvme_enabled == 0
                || (stats.kv_nvme_mode != resident_ipc::NVME_MODE_BATCHED_COHORT_JIT
                    && stats.kv_nvme_mode != resident_ipc::NVME_MODE_ASYNC_SELECTED_JIT)
                || stats.kv_resident_bytes_per_token == 0
                || stats.kv_resident_pool_bytes == 0
                || stats.kv_nvme_capacity_bytes == 0
                || stats.kv_nvme_batch_block_capacity == 0));
    if mismatch {
        eprintln!(
            "rank_cuda_resident_contract_mismatch rank={} logical={}/{} execution={}/{} kv_blocks={} logical_blocks={} quantization={}/{} moe_backend={} moe_layers={}/{} nvme={} nvme_mode={} blocker={:?}",
            plan.rank_index,
            stats.logical_lane_capacity,
            plan.logical_lane_capacity,
            stats.execution_row_capacity,
            plan.execution_row_capacity,
            stats.kv_physical_block_capacity,
            stats.kv_logical_block_capacity,
            stats.model_quantization_mode,
            plan.quantization_mode.code(),
            stats.moe_backend_kind,
            stats.moe_bound_layer_count,
            stats.moe_expected_layer_count,
            stats.kv_nvme_enabled,
            stats.kv_nvme_mode,
            stats.blocker_ascii(),
        );
        return Err(SparkStatus::ModuleNotValidated);
    }
    Ok(())
}

trait BlockerAscii {
    fn blocker_ascii(&self) -> String;
}

impl BlockerAscii for IpcStats {
    fn blocker_ascii(&self) -> String {
        let end = self.blocker.iter().position(|&byte| byte == 0).unwrap_or(self.blocker.len());
        String::from_utf8_lossy(&self.blocker[..end]).into_owned()
    }
}

// ---------------------------------------------------------------------------
// CUDA-resident submit engine
// ---------------------------------------------------------------------------

/// C: the resident-socket branch of `SparkRingDaemonSubmitWork` +
/// `SparkRingDaemonAwaitResidentSubmitResult` + the handshake half of
/// `SparkRingDaemonConnectCudaResident`.
pub struct CudaResidentEngine {
    link: Box<dyn Link>,
    pump: ResidentMessagePump,
    rank_index: u32,
    next_sequence_number: u64,
    work_config: WorkControlConfig,
    /// C: `cuda_resident_submit_count` (the core counter is incremented by
    /// the daemon; this mirrors the engine-side diagnostic).
    pub submit_count: u64,
}

impl CudaResidentEngine {
    /// Wraps an already-connected, already-handshaken link (the state after
    /// [`CudaResidentEngine::handshake`]).
    pub fn new(link: Box<dyn Link>, rank_index: u32, work_config: WorkControlConfig) -> Self {
        Self {
            link,
            pump: ResidentMessagePump::new(
                resident_ipc::max_control_payload_bytes(&work_config) as u32
            ),
            rank_index,
            next_sequence_number: 0,
            work_config,
            submit_count: 0,
        }
    }

    /// C: `SparkRingDaemonWriteResidentMessage`.
    fn write_resident_message(&mut self, kind: u32, payload: &[u8]) -> Result<()> {
        let header = resident_ipc::initialize_header(
            kind,
            self.rank_index,
            self.next_sequence_number,
            payload.len() as u32,
        );
        self.next_sequence_number += 1;
        self.link.write_all_status(&resident_ipc::encode_header(&header))?;
        if !payload.is_empty() {
            self.link.write_all_status(payload)?;
        }
        Ok(())
    }

    /// C: the connect + HELLO + HELLO_ACK + contract half of
    /// `SparkRingDaemonConnectCudaResident` (link already connected).
    /// Leaves the link nonblocking on success.
    pub fn handshake(link: Box<dyn Link>, core: &RankDaemonCore, process_id: u64) -> Result<Self> {
        let mut engine = Self::new(link, core.rank_plan.rank_index, core.work_config.clone());
        let hello = resident_ipc::IpcHello {
            descriptor_bytes: resident_ipc::IPC_HELLO_BYTES,
            rank_index: core.rank_plan.rank_index,
            rank_count: core.geometry.stage_count,
            expected_cuda_generation: 0,
            control_generation: 0,
            process_id,
        };
        engine.write_resident_message(
            resident_ipc::IPC_KIND_HELLO,
            &resident_ipc::encode_hello(&hello),
        )?;
        // Blocking read of the HELLO_ACK stats (C: ReadFull on a blocking fd).
        let message =
            engine.pump.read_message_blocking(engine.link.as_mut(), SUBMIT_RESULT_TIMEOUT_NS)?;
        resident_ipc::validate_header(
            &message.header,
            resident_ipc::IPC_KIND_HELLO_ACK,
            resident_ipc::IPC_STATS_BYTES,
        )?;
        if message.header.payload_bytes != resident_ipc::IPC_STATS_BYTES {
            return Err(SparkStatus::AbiMismatch);
        }
        let stats = decode_stats_payload(&message.payload).ok_or(SparkStatus::AbiMismatch)?;
        validate_resident_contract(
            core,
            &stats,
            core.rank_plan.quantization_mode.expected_moe_backend_kind(),
        )?;
        engine.link.set_nonblocking(true).map_err(|_| SparkStatus::InternalError)?;
        Ok(engine)
    }

    /// C: `SparkRingDaemonAwaitResidentSubmitResult` — reads until the
    /// submit result arrives, queueing interleaved completions.
    fn await_submit_result(&mut self) -> Result<SubmitOutcome> {
        let mut outcome = SubmitOutcome::default();
        loop {
            let message =
                match self.pump.read_message_blocking(self.link.as_mut(), SUBMIT_RESULT_TIMEOUT_NS)
                {
                    Ok(message) => message,
                    Err(_) => {
                        // C: teardown + BUSY on any read failure.
                        return Err(SparkStatus::Busy);
                    }
                };
            match message.header.kind {
                resident_ipc::IPC_KIND_SUBMIT_RESULT => {
                    if message.header.payload_bytes != resident_ipc::IPC_SUBMIT_RESULT_BYTES {
                        return Err(SparkStatus::Busy);
                    }
                    let result: IpcSubmitResult =
                        resident_ipc::decode_submit_result(&message.payload)
                            .ok_or(SparkStatus::Busy)?;
                    return status_from_code(result.status)
                        .map(|status| match status {
                            SparkStatus::Ok => Ok(outcome),
                            other => Err(other),
                        })
                        .unwrap_or(Err(SparkStatus::ValidationFailed));
                }
                resident_ipc::IPC_KIND_COMPLETION => {
                    if message.header.payload_bytes != resident_ipc::IPC_COMPLETION_BYTES {
                        return Err(SparkStatus::Busy);
                    }
                    let completion: IpcCompletion =
                        resident_ipc::decode_completion(&message.payload)
                            .ok_or(SparkStatus::Busy)?;
                    let draft =
                        if completion.flags & resident_ipc::COMPLETION_FLAG_DSPARK_DRAFT != 0 {
                            Some(completion.dspark_draft)
                        } else {
                            None
                        };
                    outcome
                        .completions
                        .push(EngineCompletion { completion: completion.completion, draft });
                }
                _ => return Err(SparkStatus::Busy),
            }
        }
    }
}

impl SubmitEngine for CudaResidentEngine {
    fn submit(&mut self, packet: &WorkControlPacket) -> Result<SubmitOutcome> {
        let message = resident_ipc::initialize_submit_work(
            &self.work_config,
            packet,
            resident_ipc::SUBMIT_WORK_FLAG_EXPECT_RESULT,
        )?;
        let payload = resident_ipc::encode_submit_work(&self.work_config, &message)?;
        self.write_resident_message(resident_ipc::IPC_KIND_SUBMIT_WORK, &payload)
            .map_err(|_| SparkStatus::Busy)?;
        self.submit_count += 1;
        self.await_submit_result()
    }
}

// ---------------------------------------------------------------------------
// Work-control socket path
// ---------------------------------------------------------------------------

/// C: `SparkDistributedWorkInitializeAcknowledgement` (the convenience
/// wrapper: ACCEPTED state on OK/DUPLICATE, FAILED otherwise).
pub fn distributed_initialize_acknowledgement(
    identity: &Identity,
    packet_hash: u64,
    status: SparkStatus,
) -> wire::WorkAcknowledgement {
    let state = if status == SparkStatus::Ok || status == SparkStatus::Duplicate {
        wire::WIRE_STATE_ACCEPTED
    } else {
        wire::WIRE_STATE_FAILED
    };
    wire::initialize_acknowledgement(Some(identity), packet_hash, state, status)
}

/// C: `SparkRingDaemonPumpWorkControl` for one already-accepted input
/// connection (the listener/accept and the partial-write acknowledgement
/// flush are poll-loop wiring; the packet read/handle/ack state machine is
/// here). Returns the number of packets handled.
pub fn pump_work_control(core: &mut RankDaemonCore, link: &mut dyn Link) -> Result<u32> {
    let mut progress = 0u32;
    loop {
        // Read the prefix first, then the rest of the packet (C: expected
        // bytes switch from PREFIX to descriptor_bytes once flags are in).
        let prefix_bytes = core.work_config.packet_prefix_bytes() as usize;
        let mut prefix = vec![0u8; prefix_bytes];
        let mut offset = 0usize;
        while offset < prefix_bytes {
            match link.read(&mut prefix[offset..]) {
                Ok(0) => return Ok(progress), // peer closed (C resets the socket)
                Ok(got) => offset += got,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    if offset == 0 {
                        return Ok(progress);
                    }
                    return Err(io_status(&error));
                }
                Err(error) => return Err(io_status(&error)),
            }
        }
        let descriptor_bytes =
            u32::from_le_bytes(prefix[8..12].try_into().expect("prefix layout")) as usize;
        if descriptor_bytes < prefix_bytes
            || descriptor_bytes
                > spark_sched::work_control::calculate_packet_bytes(
                    &core.work_config,
                    core.work_config.max_lane_count,
                ) as usize
        {
            core.counters.work_error_count += 1;
            return Err(SparkStatus::ValidationFailed);
        }
        let mut packet_bytes = prefix;
        packet_bytes.resize(descriptor_bytes, 0);
        while offset < descriptor_bytes {
            match link.read(&mut packet_bytes[offset..]) {
                Ok(0) => return Err(SparkStatus::RouteNotFound),
                Ok(got) => offset += got,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(io_status(&error)),
            }
        }
        let packet = wire::packet_from_wire(&core.work_config, &packet_bytes)?;
        let packet_hash = packet_fingerprint(&packet, &core.work_config);
        let identity = match get_transaction_identity(&packet) {
            Ok(identity) if packet_hash != 0 => identity,
            _ => {
                core.counters.work_error_count += 1;
                return Err(SparkStatus::ValidationFailed);
            }
        };
        let acknowledgement_status = core.handle_work(&packet);
        if acknowledgement_status != SparkStatus::Ok
            && acknowledgement_status != SparkStatus::Duplicate
            && acknowledgement_status != SparkStatus::Busy
            && acknowledgement_status != SparkStatus::CapacityExceeded
        {
            core.counters.work_error_count += 1;
            eprintln!(
                "rank_work_accept_failed status={} transaction={} request={} queued={}",
                acknowledgement_status as u32,
                identity.step_generation,
                packet.request_id,
                core.work_queue_count()
            );
        }
        let acknowledgement =
            distributed_initialize_acknowledgement(&identity, packet_hash, acknowledgement_status);
        link.write_all_status(&wire::acknowledgement_to_wire(&acknowledgement))?;
        progress += 1;
    }
}

// ---------------------------------------------------------------------------
// Downstream (next rank) forward path
// ---------------------------------------------------------------------------

/// C: `SparkRingDaemonForwardWork` +
/// `SparkRingDaemonReadWorkOutputAcknowledgement` over a connected,
/// nonblocking link. `Err(Busy)` means the write/ack is still in flight and
/// a later `forward` call resumes it (the C's
/// `work_output_waiting_for_acknowledgement` state).
pub struct SocketForwarder {
    link: Box<dyn Link>,
    write_offset: usize,
    pending_wire: Vec<u8>,
    waiting_for_acknowledgement: bool,
    acknowledgement_bytes: [u8; wire::ACKNOWLEDGEMENT_WIRE_BYTES],
    acknowledgement_read_offset: usize,
    packet_hash: u64,
}

impl SocketForwarder {
    pub fn new(link: Box<dyn Link>) -> Self {
        Self {
            link,
            write_offset: 0,
            pending_wire: Vec::new(),
            waiting_for_acknowledgement: false,
            acknowledgement_bytes: [0; wire::ACKNOWLEDGEMENT_WIRE_BYTES],
            acknowledgement_read_offset: 0,
            packet_hash: 0,
        }
    }

    /// C: `SparkRingDaemonReadWorkOutputAcknowledgement`.
    fn read_acknowledgement(&mut self, identity: &Identity) -> Result<()> {
        while self.acknowledgement_read_offset < wire::ACKNOWLEDGEMENT_WIRE_BYTES {
            match self
                .link
                .read(&mut self.acknowledgement_bytes[self.acknowledgement_read_offset..])
            {
                Ok(0) => {
                    self.reset();
                    return Err(SparkStatus::RouteNotFound);
                }
                Ok(got) => self.acknowledgement_read_offset += got,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    return Err(SparkStatus::Busy)
                }
                Err(_) => {
                    self.reset();
                    return Err(SparkStatus::RouteNotFound);
                }
            }
        }
        let acknowledgement = wire::acknowledgement_from_wire(&self.acknowledgement_bytes);
        let status = wire::validate_acknowledgement(&acknowledgement, identity, self.packet_hash)
            .map(|carried| status_from_code(carried).unwrap_or(SparkStatus::InternalError));
        self.reset();
        match status {
            Ok(SparkStatus::Busy) | Ok(SparkStatus::CapacityExceeded) => Err(SparkStatus::Busy),
            Ok(SparkStatus::Ok) | Ok(SparkStatus::Duplicate) => Ok(()),
            Ok(other) => Err(other),
            Err(status) => Err(status),
        }
    }

    /// C: `SparkRingDaemonResetWorkOutputSocket` (state half; the socket
    /// itself stays open — reconnect is poll-loop wiring).
    fn reset(&mut self) {
        self.write_offset = 0;
        self.pending_wire.clear();
        self.waiting_for_acknowledgement = false;
        self.acknowledgement_read_offset = 0;
        self.packet_hash = 0;
    }
}

impl WorkForwarder for SocketForwarder {
    fn forward(
        &mut self,
        packet_wire: &[u8],
        identity: &Identity,
        packet_fingerprint: u64,
    ) -> Result<()> {
        if self.waiting_for_acknowledgement {
            return self.read_acknowledgement(identity);
        }
        if self.pending_wire.is_empty() {
            self.pending_wire = packet_wire.to_vec();
        }
        while self.write_offset < self.pending_wire.len() {
            match self.link.write(&self.pending_wire[self.write_offset..]) {
                Ok(0) => {
                    self.reset();
                    return Err(SparkStatus::RouteNotFound);
                }
                Ok(written) => self.write_offset += written,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    return Err(SparkStatus::Busy)
                }
                Err(_) => {
                    self.reset();
                    return Err(SparkStatus::RouteNotFound);
                }
            }
        }
        if packet_fingerprint == 0 {
            return Err(SparkStatus::InternalError);
        }
        self.packet_hash = packet_fingerprint;
        self.waiting_for_acknowledgement = true;
        self.acknowledgement_read_offset = 0;
        Err(SparkStatus::Busy)
    }
}

// ---------------------------------------------------------------------------
// Tests: verbatim ports of `tests/test_glm52_ring_rank_daemon.c`
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use spark_sched::work_control::{
        transaction_phase, ABI_VERSION, FLAG_MTP_DRAFT, FLAG_PREFILL, PACKET_MAGIC,
        STAGE_PLAN_BUCKET_B16,
    };
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;

    use crate::rank_daemon::ring_runtime::{
        self, RingModelGeometry, RingPackLayout, DEFAULT_PORT_BASE,
    };
    use crate::rank_daemon::shape::{ShapeModelInputs, TpModelGeometry};

    // -- C test helpers ------------------------------------------------------

    /// GLM52-reference work-control capacities through config (deviation #1).
    fn glm52_work_config() -> WorkControlConfig {
        WorkControlConfig {
            mtp_draft_token_count: 6,
            dspark_max_speculative_token_count: 7,
            maximum_context_tokens: 1_048_576,
            kv_block_tokens: 64,
            max_prefill_tokens_per_packet: 256,
            output_vocab_count: 154_880,
            max_lane_count: 1024,
            max_active_sequence_count: 1024,
            cohort_capacity: 1023,
        }
    }

    /// GLM52-production geometry through config (see ring_runtime tests).
    fn glm52_geometry() -> RingModelGeometry {
        RingModelGeometry {
            layer_count: 78,
            first_routed_layer: 3,
            weight_layer_count: 79,
            hidden_dimension: 6144,
            hidden_bf16_bytes_per_sequence: 12288,
            maximum_context_tokens: 1_048_576,
            dsa_selected_token_count: 2048,
            dsa_selected_index_bytes_per_sequence: 8192,
            max_speculative_rows_per_lane: 8,
            max_batch_bucket: 1024,
            shape_inputs: ShapeModelInputs::new(78, 6144, 2048, 12288, 512 + 64, 1),
            tp_geometry: TpModelGeometry::new(64, 192 + 64, 192 + 256, 256),
            stage_count: 13,
            default_stage_layer_counts: vec![6; 13],
            host_prefix: "10.10.100.".to_string(),
            host_index_base: 10,
            pack_layout: RingPackLayout::default(),
            max_routed_layers_per_stage: spark_sched::stage_plan::MAX_ROUTED_LAYERS_PER_STAGE,
        }
    }

    /// C: `SparkTestRankDaemonInitializeRuntime` + the per-test
    /// `rank_plan.flags` / `execution_row_capacity` assignments.
    fn test_core(flags: u32, execution_row_capacity: u32) -> RankDaemonCore {
        let geometry = glm52_geometry();
        let mut plan = ring_runtime::build_rank_plan(
            &geometry,
            0,
            64,
            DEFAULT_PORT_BASE,
            QuantizationMode::Fp8E4m3,
        )
        .unwrap();
        plan.flags = flags;
        plan.execution_row_capacity = execution_row_capacity;
        RankDaemonCore::new(glm52_work_config(), geometry, plan).unwrap()
    }

    /// C: `SparkTestRankDaemonPhaseFromFlags` (0-based local enum).
    fn phase_from_flags(flags: u32) -> u32 {
        if flags & FLAG_RELEASE_SEQUENCES != 0 {
            return 3;
        }
        if flags & FLAG_PREFILL != 0 {
            return 0;
        }
        if flags
            & (spark_sched::work_control::FLAG_DSPARK_SPECULATIVE_VERIFY
                | spark_sched::work_control::FLAG_MTP_SPECULATIVE_VERIFY)
            != 0
        {
            return 2;
        }
        1
    }

    /// C: `SparkTestRankDaemonTransactionId`.
    fn test_transaction_id(sequence_id: u64, sequence_position: u64, flags: u32) -> u64 {
        sequence_id * 1_000_000 + sequence_position * 16 + u64::from(phase_from_flags(flags) + 1)
    }

    /// C: `SparkTestRankDaemonBuildPacket`.
    fn build_packet(
        config: &WorkControlConfig,
        sequence_id: u64,
        sequence_position: u64,
        flags: u32,
    ) -> WorkControlPacket {
        let prefill = flags & FLAG_PREFILL != 0;
        let mut packet = WorkControlPacket::zeroed(config);
        packet.magic = PACKET_MAGIC;
        packet.abi_version = ABI_VERSION;
        packet.descriptor_bytes = config.calculate_packet_bytes(1);
        packet.flags = flags;
        packet.control_generation = 7;
        packet.request_id = sequence_id;
        packet.sequence_id = sequence_id;
        packet.sequence_position = sequence_position;
        packet.active_sequence_count = 1;
        packet.set_lane_count(config, 1);
        packet.new_token_count = 1;
        packet.pipeline_slot = 0;
        packet.block_token_count = 16;
        packet.kv_block_table_token_count = sequence_position as u32 + 1;
        packet.max_blocks_per_sequence = 1;
        packet.rows_per_lane = 1;
        packet.execution_row_count = 1;
        packet.execution_batch_bucket = STAGE_PLAN_BUCKET_B16;
        packet.input_token_id = 1;
        packet.prefill_token_ids[0] = 1;
        packet.mtp_draft_token_count = if flags & FLAG_MTP_DRAFT != 0 { 1 } else { 0 };
        packet.lanes[0].request_id = sequence_id;
        packet.lanes[0].request_generation = sequence_id + 1000;
        packet.lanes[0].sequence_id = sequence_id;
        packet.lanes[0].sequence_position = sequence_position;
        packet.lanes[0].request_slot_index = 0;
        packet.lanes[0].context_token_count = sequence_position as u32 + 1;
        packet.lanes[0].input_token_id = 1;
        packet.lanes[0].mtp_draft_token_count = packet.mtp_draft_token_count;
        packet.lanes[0].mtp_resolution_path_id =
            spark_sched::work_control::mtp_tree::RESOLUTION_NONE as u16;
        if !prefill {
            packet.prefill_token_ids[0] = 0;
        }
        packet.step_chunk_index = 0;
        packet.step_chunk_count = 1;
        packet.transaction_phase = transaction_phase(&packet);
        let control_generation = packet.control_generation;
        let transaction_id = test_transaction_id(sequence_id, sequence_position, flags);
        spark_sched::work_control::set_transaction_identity(
            &mut packet,
            config,
            control_generation,
            transaction_id,
            transaction_id,
            sequence_position + 1,
        )
        .unwrap();
        packet
    }

    /// C: `SparkTestRankDaemonBuildTwoLanePacket`.
    fn build_two_lane_packet(
        config: &WorkControlConfig,
        first_sequence_id: u64,
        sequence_position: u64,
        flags: u32,
    ) -> WorkControlPacket {
        let mut packet = build_packet(config, first_sequence_id, sequence_position, flags);
        packet.active_sequence_count = 2;
        packet.set_lane_count(config, 2);
        packet.execution_row_count = 2;
        packet.descriptor_bytes = config.calculate_packet_bytes(2);
        packet.lanes[1] = packet.lanes[0].clone();
        packet.lanes[1].request_id = first_sequence_id + 1;
        packet.lanes[1].request_generation = first_sequence_id + 2001;
        packet.lanes[1].sequence_id = first_sequence_id + 1;
        packet.lanes[1].request_slot_index = 1;
        let control_generation = packet.control_generation;
        let transaction_id = test_transaction_id(first_sequence_id, sequence_position, flags);
        spark_sched::work_control::set_transaction_identity(
            &mut packet,
            config,
            control_generation,
            transaction_id,
            transaction_id,
            sequence_position + 1,
        )
        .unwrap();
        packet
    }

    /// C: `SparkTestRankDaemonBuildCompletionRecord`.
    fn build_completion_record(
        lane: &spark_sched::work_control::WorkControlLane,
        token_id: u32,
    ) -> DriverCompletionRecord {
        let mut completion = DriverCompletion {
            request_id: lane.request_id,
            sequence_id: lane.sequence_id,
            sequence_position: lane.sequence_position,
            program_id: 9,
            driver_dispatch_slot: lane.request_slot_index,
            accepted_token_count: 1,
            completion_flags: DRIVER_COMPLETION_FLAG_TOKEN_IDS,
            token_count: 1,
            ..DriverCompletion::default()
        };
        completion.token_ids[0] = token_id;
        DriverCompletionRecord { completion, draft: None }
    }

    /// C: `SparkTestRankDaemonObservePacket`.
    fn observe_packet(core: &mut RankDaemonCore, packet: &WorkControlPacket) {
        let identity = get_transaction_identity(packet).unwrap();
        let packet_hash = packet_fingerprint(packet, &core.work_config);
        assert!(packet_hash != 0);
        core.ledger.distributed_observe(&identity, packet_hash).unwrap();
    }

    /// C: the `SparkTestRankDaemonSubmitWork` fake (status + completion
    /// count statics).
    struct FakeSubmitEngine {
        status: SparkStatus,
        completions: u32,
    }

    impl SubmitEngine for FakeSubmitEngine {
        fn submit(&mut self, packet: &WorkControlPacket) -> Result<SubmitOutcome> {
            if self.status != SparkStatus::Ok || self.completions == 0 {
                return match self.status {
                    SparkStatus::Ok => Ok(SubmitOutcome::default()),
                    other => Err(other),
                };
            }
            let mut outcome = SubmitOutcome::default();
            for (lane_index, lane) in packet.lanes[..packet.lane_count as usize].iter().enumerate()
            {
                let mut completion = DriverCompletion {
                    request_id: lane.request_id,
                    sequence_id: lane.sequence_id,
                    sequence_position: lane.sequence_position,
                    program_id: 3,
                    driver_dispatch_slot: lane_index as u32,
                    accepted_token_count: 1,
                    completion_flags: DRIVER_COMPLETION_FLAG_TOKEN_IDS,
                    token_count: 1,
                    status: SparkStatus::Ok as u32,
                    ..DriverCompletion::default()
                };
                completion.token_ids[0] = 100 + lane_index as u32;
                outcome.completions.push(EngineCompletion { completion, draft: None });
            }
            Ok(outcome)
        }
    }

    /// Forwarder that never connects (C: `EnsureWorkOutputSocket` failure →
    /// `ROUTE_NOT_FOUND`).
    struct FailingForwarder;

    impl WorkForwarder for FailingForwarder {
        fn forward(&mut self, _: &[u8], _: &Identity, _: u64) -> Result<()> {
            Err(SparkStatus::RouteNotFound)
        }
    }

    fn ledger_state(core: &RankDaemonCore, packet: &WorkControlPacket) -> u32 {
        let identity = get_transaction_identity(packet).unwrap();
        let packet_hash = packet_fingerprint(packet, &core.work_config);
        core.ledger.distributed_find(&identity, packet_hash).unwrap().state
    }

    fn read_full(stream: &mut UnixStream, byte_count: usize) -> Vec<u8> {
        let mut buffer = vec![0u8; byte_count];
        stream.read_exact(&mut buffer).unwrap();
        buffer
    }

    // -- The 14 C tests ------------------------------------------------------

    /// C: `SparkTestRankDaemonReadsSplitResidentMessage`.
    #[test]
    fn reads_split_resident_message() {
        let config = glm52_work_config();
        let (left, mut right) = UnixStream::pair().unwrap();
        left.set_nonblocking(true).unwrap();
        let mut left: Box<dyn Link> = Box::new(left);
        let mut pump =
            ResidentMessagePump::new(resident_ipc::max_control_payload_bytes(&config) as u32);
        let payload = IpcSubmitResult {
            descriptor_bytes: resident_ipc::IPC_SUBMIT_RESULT_BYTES,
            status: SparkStatus::Ok as u32,
            stats: IpcStats::default(),
        };
        let payload_bytes = resident_ipc::encode_submit_result(&payload);
        let header = resident_ipc::initialize_header(
            resident_ipc::IPC_KIND_SUBMIT_RESULT,
            4,
            9,
            resident_ipc::IPC_SUBMIT_RESULT_BYTES,
        );
        let header_bytes = resident_ipc::encode_header(&header);
        let header_split = header_bytes.len() / 2;
        let payload_split = payload_bytes.len() / 2;
        right.write_all(&header_bytes[..header_split]).unwrap();
        assert!(matches!(pump.read_message(left.as_mut()), Ok(None)));
        assert_eq!(pump.header_offset(), header_split);
        right.write_all(&header_bytes[header_split..]).unwrap();
        right.write_all(&payload_bytes[..payload_split]).unwrap();
        assert!(matches!(pump.read_message(left.as_mut()), Ok(None)));
        assert_eq!(pump.header_offset(), resident_ipc::IPC_HEADER_BYTES as usize);
        assert_eq!(pump.payload_offset(), payload_split);
        right.write_all(&payload_bytes[payload_split..]).unwrap();
        let message = pump.read_message(left.as_mut()).unwrap().expect("complete message");
        assert_eq!(message.header.kind, resident_ipc::IPC_KIND_SUBMIT_RESULT);
        assert_eq!(message.payload, payload_bytes.to_vec());
        assert_eq!(pump.header_offset(), 0);
        assert_eq!(pump.payload_offset(), 0);
    }

    /// C: `SparkTestRankDaemonRequestsSubmitResult`.
    #[test]
    fn requests_submit_result() {
        let config = glm52_work_config();
        let packet = build_packet(&config, 31, 2, FLAG_PREFILL);
        let (left, mut right) = UnixStream::pair().unwrap();
        let mut engine = CudaResidentEngine::new(Box::new(left), 1, config.clone());
        let result = IpcSubmitResult {
            descriptor_bytes: resident_ipc::IPC_SUBMIT_RESULT_BYTES,
            status: SparkStatus::Ok as u32,
            stats: IpcStats::default(),
        };
        let header = resident_ipc::initialize_header(
            resident_ipc::IPC_KIND_SUBMIT_RESULT,
            1,
            1,
            resident_ipc::IPC_SUBMIT_RESULT_BYTES,
        );
        right.write_all(&resident_ipc::encode_header(&header)).unwrap();
        right.write_all(&resident_ipc::encode_submit_result(&result)).unwrap();
        assert!(engine.submit(&packet).is_ok());
        let submitted_header_bytes = read_full(&mut right, resident_ipc::IPC_HEADER_BYTES as usize);
        let submitted_header =
            resident_ipc::decode_header(&submitted_header_bytes.try_into().unwrap());
        assert_eq!(submitted_header.kind, resident_ipc::IPC_KIND_SUBMIT_WORK);
        assert_eq!(
            submitted_header.payload_bytes,
            resident_ipc::IPC_SUBMIT_WORK_PREFIX_BYTES + packet.descriptor_bytes
        );
        let submitted_bytes = read_full(&mut right, submitted_header.payload_bytes as usize);
        let mut cursor = wire::Reader::new(&submitted_bytes);
        let submitted_descriptor = cursor.u32().unwrap();
        let submitted_flags = cursor.u32().unwrap();
        assert_eq!(submitted_descriptor, submitted_header.payload_bytes);
        assert_eq!(submitted_flags, resident_ipc::SUBMIT_WORK_FLAG_EXPECT_RESULT);
        let submitted_packet = wire::packet_from_wire(&config, &submitted_bytes[8..]).unwrap();
        let submitted = resident_ipc::IpcSubmitWork {
            descriptor_bytes: submitted_descriptor,
            flags: submitted_flags,
            work_packet: submitted_packet,
        };
        assert_eq!(
            resident_ipc::validate_submit_work(&config, &submitted, submitted_header.payload_bytes),
            Ok(())
        );
    }

    /// C: `SparkTestRankDaemonPacketIdentityIncludesPhase`.
    #[test]
    fn packet_identity_includes_phase() {
        let config = glm52_work_config();
        let mut core = test_core(0, 16);
        let prefill = build_packet(&config, 41, 8, FLAG_PREFILL);
        let mut decode = prefill.clone();
        decode.flags = FLAG_MTP_DRAFT;
        assert_eq!(core.queue_work(&prefill), Ok(true));
        assert_eq!(core.queue_work(&decode), Ok(true));
        assert_eq!(core.work_queue_count(), 2);
        assert_eq!(core.counters.work_duplicate_count, 0);
        assert_eq!(core.queue_work(&decode), Ok(false));
        assert_eq!(core.work_queue_count(), 2);
        assert_eq!(core.counters.work_duplicate_count, 1);
    }

    /// C: `SparkTestRankDaemonFindsLaneDependency`.
    #[test]
    fn finds_lane_dependency() {
        let config = glm52_work_config();
        let mut core = test_core(0, 16);
        let mut earlier = build_packet(&config, 51, 4, FLAG_PREFILL);
        earlier.set_lane_count(&config, 2);
        earlier.active_sequence_count = 2;
        earlier.descriptor_bytes = config.calculate_packet_bytes(2);
        earlier.lanes[1].request_id = 52;
        earlier.lanes[1].sequence_id = 52;
        earlier.lanes[1].sequence_position = 4;
        let current = build_packet(&config, 52, 5, FLAG_PREFILL);
        assert_eq!(core.queue_work(&earlier), Ok(true));
        assert_eq!(core.queue_work(&current), Ok(true));
        assert!(core.has_queued_dependency(&core.queue[1].packet));
        core.pop_work();
        assert!(!core.has_queued_dependency(&core.queue[0].packet));
    }

    /// C: `SparkTestRankDaemonDecodeWaitsForSamePositionPrefill`.
    #[test]
    fn decode_waits_for_same_position_prefill() {
        let config = glm52_work_config();
        let mut core = test_core(0, 16);
        let prefill = build_packet(&config, 61, 8, FLAG_PREFILL);
        let decode = build_packet(&config, 61, 8, FLAG_MTP_DRAFT);
        assert_eq!(core.queue_work(&prefill), Ok(true));
        assert_eq!(core.queue_work(&decode), Ok(true));
        assert!(core.has_queued_dependency(&core.queue[1].packet));
    }

    /// C: `SparkTestRankDaemonForwardWaitPreservesFifo`.
    #[test]
    fn forward_wait_preserves_fifo() {
        let config = glm52_work_config();
        let mut core = test_core(RANK_FLAG_HAS_NEXT, 16);
        let packet = build_packet(&config, 71, 0, FLAG_PREFILL);
        observe_packet(&mut core, &packet);
        assert_eq!(core.queue_work(&packet), Ok(true));
        let mut next = packet.clone();
        next.sequence_position = 1;
        next.lanes[0].sequence_position = 1;
        next.kv_block_table_token_count = 2;
        next.lanes[0].context_token_count = 2;
        let control_generation = next.control_generation;
        let transaction_id =
            test_transaction_id(next.sequence_id, next.sequence_position, next.flags);
        let step_generation = next.sequence_position + 1;
        spark_sched::work_control::set_transaction_identity(
            &mut next,
            &config,
            control_generation,
            transaction_id,
            transaction_id,
            step_generation,
        )
        .unwrap();
        observe_packet(&mut core, &next);
        assert_eq!(core.queue_work(&next), Ok(true));
        core.queue[0].state = WorkState::WaitingForward;
        let mut forwarder = FailingForwarder;
        let mut engine = FakeSubmitEngine { status: SparkStatus::Ok, completions: 0 };
        assert_eq!(core.pump_queued_work(&mut forwarder, &mut engine), Ok(0));
        assert_eq!(core.work_queue_count(), 2);
        assert_eq!(core.queued_packet(0).unwrap().sequence_position, 0);
        assert_eq!(core.queued_state(0), Some(WorkState::WaitingForward));
    }

    /// C: `SparkTestRankDaemonBackpressuresFullWorkQueue`.
    #[test]
    fn backpressures_full_work_queue() {
        let config = glm52_work_config();
        let mut core = test_core(0, 1);
        for index in 0..WORK_QUEUE_CAPACITY {
            let filler = build_packet(&config, 1000 + index as u64, 0, FLAG_PREFILL);
            assert_eq!(core.queue_work(&filler), Ok(true));
        }
        let (left, mut right) = UnixStream::pair().unwrap();
        left.set_nonblocking(true).unwrap();
        let mut left: Box<dyn Link> = Box::new(left);
        let packet = build_packet(&config, 81, 0, FLAG_PREFILL);
        right.write_all(&wire::packet_to_wire(&config, &packet).unwrap()).unwrap();
        assert_eq!(pump_work_control(&mut core, left.as_mut()), Ok(1));
        let acknowledgement_bytes = read_full(&mut right, wire::ACKNOWLEDGEMENT_WIRE_BYTES);
        let acknowledgement =
            wire::acknowledgement_from_wire(&acknowledgement_bytes.try_into().unwrap());
        assert_eq!(acknowledgement.status, SparkStatus::CapacityExceeded as u32);
        assert_eq!(core.work_queue_count(), WORK_QUEUE_CAPACITY);
    }

    /// C: `SparkTestRankDaemonCommittedDuplicateIsAcknowledged`.
    #[test]
    fn committed_duplicate_is_acknowledged() {
        let config = glm52_work_config();
        let mut core = test_core(0, 1);
        let packet = build_packet(&config, 91, 0, FLAG_PREFILL);
        observe_packet(&mut core, &packet);
        core.transition_packet(&packet, STATE_ACCEPTED, SparkStatus::Pending).unwrap();
        core.transition_packet(&packet, STATE_EXECUTING, SparkStatus::Pending).unwrap();
        core.transition_packet(&packet, STATE_COMMITTED, SparkStatus::Ok).unwrap();
        assert_eq!(core.handle_work(&packet), SparkStatus::Duplicate);
        assert_eq!(core.work_queue_count(), 0);
    }

    /// C: `SparkTestRankDaemonRejectsIdentityReuseWithDifferentBytes`.
    #[test]
    fn rejects_identity_reuse_with_different_bytes() {
        let config = glm52_work_config();
        let mut core = test_core(0, 1);
        let packet = build_packet(&config, 101, 0, FLAG_PREFILL);
        assert_eq!(core.handle_work(&packet), SparkStatus::Ok);
        let mut altered_packet = packet.clone();
        altered_packet.pipeline_slot = 1;
        assert_eq!(core.handle_work(&altered_packet), SparkStatus::ValidationFailed);
        assert_eq!(core.work_queue_count(), 1);
    }

    /// C: `SparkTestRankDaemonCommitsAfterEveryLaneCompletion`.
    #[test]
    fn commits_after_every_lane_completion() {
        let config = glm52_work_config();
        let mut core = test_core(RANK_FLAG_FINAL_STAGE, 16);
        let packet = build_two_lane_packet(&config, 111, 4, FLAG_MTP_DRAFT);
        observe_packet(&mut core, &packet);
        core.transition_packet(&packet, STATE_ACCEPTED, SparkStatus::Pending).unwrap();
        let transaction_key = core.register_inflight_transaction(&packet).unwrap();
        assert!(transaction_key.is_some());
        assert_eq!(core.driver_inflight_count, 2);
        assert_eq!(core.inflight_completion_mapping_count(), 2);
        core.transition_packet(&packet, STATE_EXECUTING, SparkStatus::Pending).unwrap();
        let record = build_completion_record(&packet.lanes[0], 701);
        assert_eq!(core.process_driver_completion(&record), Ok(()));
        assert_eq!(core.driver_inflight_count, 1);
        assert_eq!(core.final_event_queue_count(), 1);
        assert_eq!(core.final_events()[0].request_generation, packet.lanes[0].request_generation);
        assert_eq!(core.final_events()[0].transaction_id, packet.transaction_id);
        assert_eq!(ledger_state(&core, &packet), STATE_EXECUTING);
        let record = build_completion_record(&packet.lanes[1], 702);
        assert_eq!(core.process_driver_completion(&record), Ok(()));
        assert_eq!(core.driver_inflight_count, 0);
        assert_eq!(core.inflight_completion_mapping_count(), 0);
        assert_eq!(core.final_event_queue_count(), 2);
        assert_eq!(core.final_events()[1].request_generation, packet.lanes[1].request_generation);
        assert_eq!(ledger_state(&core, &packet), STATE_COMMITTED);
    }

    /// C: `SparkTestRankDaemonRejectsStaleCompletionWithoutLosingOwner`.
    #[test]
    fn rejects_stale_completion_without_losing_owner() {
        let config = glm52_work_config();
        let mut core = test_core(0, 16);
        let packet = build_packet(&config, 121, 7, FLAG_MTP_DRAFT);
        observe_packet(&mut core, &packet);
        core.transition_packet(&packet, STATE_ACCEPTED, SparkStatus::Pending).unwrap();
        let transaction_key = core.register_inflight_transaction(&packet).unwrap().unwrap();
        core.transition_packet(&packet, STATE_EXECUTING, SparkStatus::Pending).unwrap();
        let mut record = build_completion_record(&packet.lanes[0], 703);
        record.completion.sequence_position += 1;
        assert_eq!(core.process_driver_completion(&record), Err(SparkStatus::ValidationFailed));
        assert_eq!(core.driver_inflight_count, 1);
        assert_eq!(core.inflight_completion_mapping_count(), 1);
        assert_eq!(core.cancel_inflight_transaction(transaction_key), Ok(()));
        assert_eq!(core.driver_inflight_count, 0);
        assert_eq!(core.inflight_completion_mapping_count(), 0);
    }

    /// C: `SparkTestRankDaemonSynchronousCompletionWaitsForExecutionState`.
    #[test]
    fn synchronous_completion_waits_for_execution_state() {
        let config = glm52_work_config();
        let mut core = test_core(RANK_FLAG_FINAL_STAGE, 16);
        let packet = build_packet(&config, 131, 9, FLAG_MTP_DRAFT);
        let mut forwarder = NoForwarder;
        let mut engine = FakeSubmitEngine { status: SparkStatus::Ok, completions: 1 };
        assert_eq!(core.handle_work(&packet), SparkStatus::Ok);
        assert_eq!(core.pump_queued_work(&mut forwarder, &mut engine), Ok(1));
        assert_eq!(core.work_queue_count(), 0);
        assert_eq!(core.driver_inflight_count, 1);
        assert_eq!(core.driver_completion_queue_count(), 1);
        assert_eq!(ledger_state(&core, &packet), STATE_EXECUTING);
        assert_eq!(core.pump_driver_completions(), 1);
        assert_eq!(core.driver_inflight_count, 0);
        assert_eq!(core.driver_completion_queue_count(), 0);
        assert_eq!(core.final_event_queue_count(), 1);
        let event = &core.final_events()[0];
        assert_eq!(event.control_generation, packet.control_generation);
        assert_eq!(event.transaction_id, packet.transaction_id);
        assert_eq!(event.dispatch_generation, packet.dispatch_generation);
        assert_eq!(event.request_generation, packet.lanes[0].request_generation);
        assert_eq!(event.step_generation, packet.step_generation);
        assert_eq!(ledger_state(&core, &packet), STATE_COMMITTED);
    }

    /// C: `SparkTestRankDaemonBusySubmitRollsBackCompletionOwnership`.
    #[test]
    fn busy_submit_rolls_back_completion_ownership() {
        let config = glm52_work_config();
        let mut core = test_core(0, 16);
        let packet = build_packet(&config, 141, 3, FLAG_MTP_DRAFT);
        let mut forwarder = NoForwarder;
        let mut engine = FakeSubmitEngine { status: SparkStatus::Busy, completions: 0 };
        assert_eq!(core.handle_work(&packet), SparkStatus::Ok);
        assert_eq!(core.pump_queued_work(&mut forwarder, &mut engine), Ok(0));
        assert_eq!(core.work_queue_count(), 1);
        assert_eq!(core.driver_inflight_count, 0);
        assert_eq!(core.inflight_completion_mapping_count(), 0);
        assert_eq!(ledger_state(&core, &packet), STATE_ACCEPTED);
    }

    /// C: `SparkTestRankDaemonReleaseCommitsWithoutCompletion`.
    #[test]
    fn release_commits_without_completion() {
        let config = glm52_work_config();
        let mut core = test_core(0, 1);
        let mut packet = build_packet(&config, 151, 0, FLAG_RELEASE_SEQUENCES);
        packet.new_token_count = 0;
        packet.rows_per_lane = 0;
        packet.execution_row_count = 0;
        packet.execution_batch_bucket = 0;
        packet.input_token_id = 0;
        packet.prefill_token_ids[0] = 0;
        packet.mtp_draft_token_count = 0;
        packet.lanes[0].input_token_id = 0;
        packet.lanes[0].mtp_draft_token_count = 0;
        packet.lanes[0].sequence_position = 0;
        packet.sequence_position = 0;
        packet.kv_block_table_token_count = 1;
        packet.lanes[0].context_token_count = 1;
        packet.transaction_phase = transaction_phase(&packet);
        let control_generation = packet.control_generation;
        let transaction_id =
            test_transaction_id(packet.sequence_id, packet.sequence_position, packet.flags);
        spark_sched::work_control::set_transaction_identity(
            &mut packet,
            &config,
            control_generation,
            transaction_id,
            transaction_id,
            1,
        )
        .unwrap();
        let mut forwarder = NoForwarder;
        let mut engine = FakeSubmitEngine { status: SparkStatus::Ok, completions: 0 };
        assert_eq!(core.handle_work(&packet), SparkStatus::Ok);
        assert_eq!(core.pump_queued_work(&mut forwarder, &mut engine), Ok(1));
        assert_eq!(core.work_queue_count(), 0);
        assert_eq!(core.driver_inflight_count, 0);
        assert_eq!(core.inflight_completion_mapping_count(), 0);
        assert_eq!(ledger_state(&core, &packet), STATE_COMMITTED);
    }
}
