//! Work-output plane — port of the `node/backend.c` work queue, the
//! nonblocking output socket with its write/acknowledge choreography, and
//! the sequence-release queue.
//!
//! The C retains each queued packet in a single-slot-size arena; here every
//! queue position owns an `Option<QueuedPacket>` — same bounded-count
//! semantics, no allocator on the dispatch path after the initial `Vec`
//! allocation (per-packet bytes are a one-time `Vec` per enqueue, mirroring
//! the arena acquire).

use std::io;
use std::os::unix::io::OwnedFd;

use spark_sched::work_control::work_transaction::Identity;
use spark_sched::work_control::{self, WorkControlConfig, WorkControlPacket};
use spark_serve::serving_engine::ServingStatus;

use super::net;
use super::wire::{self, WorkAcknowledgement, ACKNOWLEDGEMENT_BYTES};

/// `SPARK_RING_SERVICE_BACKEND_PREFILL_RESERVE_CAPACITY`.
pub const PREFILL_RESERVE_CAPACITY: u32 = 1;

/// One queued packet: the parsed packet (for identity/validation) plus its
/// canonical bytes (the wire form and the dedup comparison domain).
#[derive(Debug, Clone)]
struct QueuedPacket {
    packet: WorkControlPacket,
    identity: Identity,
    bytes: Vec<u8>,
}

/// `SparkRingServiceBackendReleaseRecord`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReleaseRecord {
    pub request_id: u64,
    pub request_generation: u64,
    pub sequence_id: u64,
    pub token_count: u32,
}

/// The work-output queue and socket (`state->work_queue*` +
/// `state->work_output_*` in C).
pub struct WorkOutput {
    config: WorkControlConfig,
    session_id_base: u64,
    /// `rank_plan.flags & SPARK_RING_RUNTIME_RANK_FLAG_HAS_NEXT`.
    has_next: bool,
    next_host_name: String,
    next_port: u32,
    /// `state->rank_plan.execution_row_capacity` (packet validation bound).
    execution_row_capacity: u32,
    /// `SPARK_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT` (validation).
    max_pipeline_slot_count: u32,
    queue: Vec<Option<QueuedPacket>>,
    head: usize,
    count: usize,
    write_offset: usize,
    socket: Option<OwnedFd>,
    connecting: bool,
    acknowledgement_bytes: [u8; ACKNOWLEDGEMENT_BYTES],
    acknowledgement_read_offset: usize,
    waiting_for_acknowledgement: bool,
    packet_hash: u64,
    release_queue: Vec<ReleaseRecord>,
    release_head: usize,
    release_count: usize,
}

impl WorkOutput {
    /// The C sizes the queue from the node-context builder's prefill window
    /// (`MAX_PREFILL_TOKENS * 2`) and the release ring from the request
    /// capacity.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: WorkControlConfig,
        session_id_base: u64,
        has_next: bool,
        next_host_name: String,
        next_port: u32,
        execution_row_capacity: u32,
        max_pipeline_slot_count: u32,
        queue_capacity: usize,
        release_capacity: usize,
    ) -> Self {
        WorkOutput {
            config,
            session_id_base,
            has_next,
            next_host_name,
            next_port,
            execution_row_capacity,
            max_pipeline_slot_count,
            queue: (0..queue_capacity).map(|_| None).collect(),
            head: 0,
            count: 0,
            write_offset: 0,
            socket: None,
            connecting: false,
            acknowledgement_bytes: [0u8; ACKNOWLEDGEMENT_BYTES],
            acknowledgement_read_offset: 0,
            waiting_for_acknowledgement: false,
            packet_hash: 0,
            release_queue: vec![ReleaseRecord::default(); release_capacity],
            release_head: 0,
            release_count: 0,
        }
    }

    /// `rank_plan.flags & SPARK_RING_RUNTIME_RANK_FLAG_HAS_NEXT`.
    pub fn has_next(&self) -> bool {
        self.has_next
    }

    /// Queue depth (for tests and poll-descriptor interest).
    pub fn queue_count(&self) -> usize {
        self.count
    }

    pub fn queue_capacity(&self) -> usize {
        self.queue.len()
    }

    pub fn release_count(&self) -> usize {
        self.release_count
    }

    pub fn has_socket(&self) -> bool {
        self.socket.is_some()
    }

    /// Head packet (for tests and ack validation), parsed form.
    pub fn head_packet(&self) -> Option<&WorkControlPacket> {
        if self.count == 0 {
            return None;
        }
        self.queue[self.head].as_ref().map(|queued| &queued.packet)
    }

    /// Release record at a queue offset (for tests).
    pub fn release_record(&self, offset: usize) -> Option<&ReleaseRecord> {
        if offset >= self.release_count {
            return None;
        }
        Some(&self.release_queue[(self.release_head + offset) % self.release_queue.len()])
    }

    /// `SparkRingServiceBackendStampWorkPacketChunk`.
    pub fn stamp_work_packet_chunk(
        &self,
        packet: &mut WorkControlPacket,
        step_chunk_index: u32,
        step_chunk_count: u32,
    ) -> Result<(), ServingStatus> {
        if self.session_id_base == 0
            || packet.descriptor_bytes < self.config.packet_prefix_bytes()
            || packet.descriptor_bytes
                > self.config.calculate_packet_bytes(self.config.max_lane_count)
            || packet.lane_count == 0
            || packet.lane_count > self.config.max_lane_count
        {
            return Err(ServingStatus::InvalidArgument);
        }
        work_control::finalize_transaction(
            packet,
            &self.config,
            self.session_id_base,
            step_chunk_index,
            step_chunk_count,
        )
        .map_err(|_| ServingStatus::InvalidArgument)
    }

    /// `SparkRingServiceBackendStampWorkPacket`.
    pub fn stamp_work_packet(&self, packet: &mut WorkControlPacket) -> Result<(), ServingStatus> {
        let mut step_chunk_count = packet.step_chunk_count;
        let mut step_chunk_index = packet.step_chunk_index;
        if step_chunk_count == 0 || step_chunk_index >= step_chunk_count {
            step_chunk_index = 0;
            step_chunk_count = 1;
        }
        self.stamp_work_packet_chunk(packet, step_chunk_index, step_chunk_count)
    }

    /// `SparkRingServiceBackendEnqueueWorkPacket`.
    pub fn enqueue_work_packet(&mut self, packet: &WorkControlPacket) -> Result<(), ServingStatus> {
        if packet.descriptor_bytes < self.config.packet_prefix_bytes()
            || packet.descriptor_bytes
                > self.config.calculate_packet_bytes(self.config.max_lane_count)
        {
            return Err(ServingStatus::InvalidArgument);
        }
        let packet_identity = work_control::get_transaction_identity(packet)
            .map_err(|_| ServingStatus::InvalidArgument)?;
        if !self.has_next {
            return Ok(());
        }
        let packet_bytes = wire::serialize_packet(&self.config, packet)?;
        for offset in 0..self.count {
            let index = (self.head + offset) % self.queue.len();
            let queued = self.queue[index].as_ref().ok_or(ServingStatus::InternalError)?;
            if wire::identities_match(&queued.identity, &packet_identity) {
                if queued.bytes != packet_bytes {
                    return Err(ServingStatus::ValidationFailed);
                }
                return Ok(());
            }
        }
        if self.count >= self.queue.len() {
            return Err(ServingStatus::CapacityExceeded);
        }
        let tail = (self.head + self.count) % self.queue.len();
        if self.queue[tail].is_some() {
            return Err(ServingStatus::InternalError);
        }
        self.queue[tail] = Some(QueuedPacket {
            packet: packet.clone(),
            identity: packet_identity,
            bytes: packet_bytes,
        });
        self.count += 1;
        Ok(())
    }

    /// `SparkRingWorkControlValidatePacket` against this rank's execution
    /// geometry, as the C call sites do before enqueueing.
    pub fn validate_packet(&self, packet: &WorkControlPacket) -> Result<(), ServingStatus> {
        work_control::validate_packet(
            packet,
            &self.config,
            self.execution_row_capacity,
            self.max_pipeline_slot_count,
        )
        .map_err(|_| ServingStatus::InvalidArgument)
    }

    /// `SparkRingServiceBackendPopWorkPacket` (release failure there is
    /// `abort()` in C; here the slot drop is infallible).
    fn pop_work_packet(&mut self) {
        if self.count == 0 {
            return;
        }
        self.queue[self.head] = None;
        self.head = (self.head + 1) % self.queue.len();
        self.count -= 1;
    }

    /// `SparkRingServiceBackendResetWorkOutputAcknowledgement`.
    fn reset_acknowledgement(&mut self) {
        self.acknowledgement_bytes = [0u8; ACKNOWLEDGEMENT_BYTES];
        self.acknowledgement_read_offset = 0;
        self.waiting_for_acknowledgement = false;
        self.packet_hash = 0;
    }

    /// `SparkRingServiceBackendReadWorkOutputAcknowledgement`.
    fn read_work_output_acknowledgement(
        &mut self,
        packet: &WorkControlPacket,
        identity: &Identity,
    ) -> Result<(), ServingStatus> {
        let fd = self.socket.as_ref().ok_or(ServingStatus::InvalidArgument)?;
        if !self.waiting_for_acknowledgement {
            return Err(ServingStatus::InvalidArgument);
        }
        while self.acknowledgement_read_offset < ACKNOWLEDGEMENT_BYTES {
            let remaining = ACKNOWLEDGEMENT_BYTES - self.acknowledgement_read_offset;
            match net::read_once(
                net::raw_fd(fd),
                &mut self.acknowledgement_bytes[self.acknowledgement_read_offset
                    ..self.acknowledgement_read_offset + remaining],
            ) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    return Err(ServingStatus::Busy)
                }
                Err(_) => return Err(ServingStatus::RouteNotFound),
                Ok(0) => return Err(ServingStatus::RouteNotFound),
                Ok(got) => self.acknowledgement_read_offset += got,
            }
        }
        let raw = self.acknowledgement_bytes;
        let acknowledgement = WorkAcknowledgement::decode(&raw)?;
        let _ = packet;
        Err(acknowledgement.validate(&raw, identity, self.packet_hash))
    }

    /// `SparkRingServiceBackendDropWorkOutputSocket`.
    pub fn drop_socket(&mut self) {
        self.socket = None;
        self.connecting = false;
        self.write_offset = 0;
        self.reset_acknowledgement();
    }

    /// `SparkRingServiceBackendStartWorkOutputSocket`.
    fn start_socket(&mut self) -> Result<(), ServingStatus> {
        if !self.has_next {
            return Ok(());
        }
        if std::env::var_os("SPARKPIPE_STAGE_COMPLETION_DEBUG").is_some() {
            eprintln!("ring_work_connect host={} port={}", self.next_host_name, self.next_port);
        }
        let (fd, connecting) = net::connect_socket(&self.next_host_name, self.next_port)?;
        if std::env::var_os("SPARKPIPE_STAGE_COMPLETION_DEBUG").is_some() {
            eprintln!("ring_work_connect_result connecting={connecting}");
        }
        self.socket = Some(fd);
        self.connecting = connecting;
        if connecting {
            return Err(ServingStatus::Busy);
        }
        Ok(())
    }

    /// `SparkRingServiceBackendCheckWorkOutputConnect`.
    fn check_socket_connect(&mut self) -> Result<(), ServingStatus> {
        let fd = self.socket.as_ref().ok_or(ServingStatus::Busy)?;
        if !self.connecting {
            return Ok(());
        }
        match net::check_connect(net::raw_fd(fd)) {
            Ok(true) => {
                self.connecting = false;
                if std::env::var_os("SPARKPIPE_STAGE_COMPLETION_DEBUG").is_some() {
                    eprintln!("ring_work_connected");
                }
                Ok(())
            }
            Ok(false) => Err(ServingStatus::Busy),
            Err(_) => {
                self.drop_socket();
                self.start_socket()
            }
        }
    }

    /// `SparkRingServiceBackendEnsureWorkOutputSocket`.
    fn ensure_socket(&mut self) -> Result<(), ServingStatus> {
        if !self.has_next {
            return Ok(());
        }
        if self.socket.is_some() {
            return self.check_socket_connect();
        }
        self.start_socket()
    }

    /// `SparkRingServiceBackendPumpWorkOutput`.
    pub fn pump_work_output(&mut self) -> Result<(), ServingStatus> {
        if !self.has_next || self.count == 0 {
            return Ok(());
        }
        self.ensure_socket()?;
        self.flush_work_output(&mut |_packet: &WorkControlPacket, _status: ServingStatus| {})
    }

    /// `SparkRingServiceBackendFlushWorkOutput`. `fail_cohort` mirrors
    /// `SparkRingServiceBackendFailWorkPacketCohort` for the hard-negative
    /// acknowledgement path.
    pub fn flush_work_output(
        &mut self,
        fail_cohort: &mut dyn FnMut(&WorkControlPacket, ServingStatus),
    ) -> Result<(), ServingStatus> {
        if !self.has_next || self.count == 0 {
            return Ok(());
        }
        loop {
            let (identity, packet_bytes_count) = {
                let queued = self.queue[self.head].as_ref().ok_or(ServingStatus::InternalError)?;
                (queued.identity, queued.packet.descriptor_bytes)
            };
            if self.waiting_for_acknowledgement {
                let packet = self.queue[self.head]
                    .as_ref()
                    .ok_or(ServingStatus::InternalError)?
                    .packet
                    .clone();
                let status = self.read_work_output_acknowledgement(&packet, &identity);
                match status {
                    Err(ServingStatus::Busy) => return Err(ServingStatus::Busy),
                    Err(ServingStatus::RouteNotFound) => {
                        self.drop_socket();
                        return Err(ServingStatus::RouteNotFound);
                    }
                    Err(ServingStatus::Ok) | Err(ServingStatus::Duplicate) | Ok(()) => {
                        self.reset_acknowledgement();
                        self.pop_work_packet();
                        if self.count == 0 {
                            return Ok(());
                        }
                        continue;
                    }
                    Err(ServingStatus::CapacityExceeded) => {
                        self.reset_acknowledgement();
                        self.write_offset = 0;
                        return Err(ServingStatus::Busy);
                    }
                    Err(hard_status) => {
                        // Hard-negative acknowledgement: retransmitting
                        // replays byte-identical content which is rejected
                        // identically forever, so fail the owning cohort
                        // deterministically and drop the poisoned packet.
                        eprintln!(
                            "ring_work_failed status={} transaction={} request={} sequence={} position={} queued={}",
                            hard_status.code(),
                            packet.step_generation,
                            packet.request_id,
                            packet.sequence_id,
                            packet.sequence_position,
                            self.count
                        );
                        fail_cohort(&packet, hard_status);
                        self.reset_acknowledgement();
                        self.pop_work_packet();
                        if self.count == 0 {
                            return Ok(());
                        }
                        continue;
                    }
                }
            }
            let written = {
                let queued = self.queue[self.head].as_ref().ok_or(ServingStatus::InternalError)?;
                let fd = self.socket.as_ref().ok_or(ServingStatus::InternalError)?;
                net::write_once(
                    net::raw_fd(fd),
                    &queued.bytes[self.write_offset..packet_bytes_count as usize],
                )
            };
            match written {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    return Err(ServingStatus::Busy)
                }
                Err(_) => {
                    self.drop_socket();
                    return Err(ServingStatus::RouteNotFound);
                }
                Ok(0) => return Err(ServingStatus::Busy),
                Ok(got) => {
                    self.write_offset += got;
                }
            }
            if self.write_offset != packet_bytes_count as usize {
                continue;
            }
            let packet_hash = {
                let queued = self.queue[self.head].as_ref().ok_or(ServingStatus::InternalError)?;
                wire::hash_bytes(&queued.bytes)
            };
            if packet_hash == 0 {
                return Err(ServingStatus::InternalError);
            }
            self.packet_hash = packet_hash;
            self.write_offset = 0;
            self.waiting_for_acknowledgement = true;
            self.acknowledgement_read_offset = 0;
            self.acknowledgement_bytes = [0u8; ACKNOWLEDGEMENT_BYTES];
        }
    }

    /// `SparkRingServiceBackendQueueSequenceRelease`
    /// (`release_sequence_function` callback body).
    pub fn queue_sequence_release(
        &mut self,
        request_id: u64,
        request_generation: u64,
        sequence_id: u64,
        token_count: u32,
        context_tokens: u32,
        max_speculative_token_count: u32,
    ) -> Result<(), ServingStatus> {
        if request_id == 0 || request_generation == 0 || sequence_id == 0 || token_count == 0 {
            return Err(ServingStatus::InvalidArgument);
        }
        let token_count = if token_count <= context_tokens - max_speculative_token_count {
            token_count + max_speculative_token_count
        } else {
            context_tokens
        };
        for offset in 0..self.release_count {
            let index = (self.release_head + offset) % self.release_queue.len();
            let record = &mut self.release_queue[index];
            if record.request_id == request_id
                && record.request_generation == request_generation
                && record.sequence_id == sequence_id
            {
                if token_count > record.token_count {
                    record.token_count = token_count;
                }
                return Ok(());
            }
        }
        if self.release_count >= self.release_queue.len() {
            return Err(ServingStatus::CapacityExceeded);
        }
        let tail = (self.release_head + self.release_count) % self.release_queue.len();
        self.release_queue[tail] =
            ReleaseRecord { request_id, request_generation, sequence_id, token_count };
        self.release_count += 1;
        Ok(())
    }

    /// `SparkRingServiceBackendBuildReleasePacket`.
    pub fn build_release_packet(
        &self,
        lane_count: u32,
        kv_block_tokens: u32,
        max_blocks_per_sequence: u32,
        mtp_resolution_none: u16,
    ) -> Result<WorkControlPacket, ServingStatus> {
        if lane_count == 0
            || lane_count as usize > self.release_count
            || lane_count > self.config.max_lane_count
        {
            return Err(ServingStatus::InvalidArgument);
        }
        let mut packet = WorkControlPacket::zeroed(&self.config);
        packet.set_lane_count(&self.config, lane_count);
        packet.magic = work_control::PACKET_MAGIC;
        packet.abi_version = work_control::ABI_VERSION;
        packet.descriptor_bytes = self.config.calculate_packet_bytes(lane_count);
        packet.flags = work_control::FLAG_RELEASE_SEQUENCES;
        packet.active_sequence_count = lane_count;
        packet.block_token_count = kv_block_tokens;
        packet.max_blocks_per_sequence = max_blocks_per_sequence;
        let mut maximum_token_count = 0u32;
        for lane_index in 0..lane_count as usize {
            let queue_index = (self.release_head + lane_index) % self.release_queue.len();
            let record = &self.release_queue[queue_index];
            if record.request_id == 0
                || record.request_generation == 0
                || record.sequence_id == 0
                || record.token_count == 0
            {
                return Err(ServingStatus::InternalError);
            }
            let lane = &mut packet.lanes[lane_index];
            lane.request_id = record.request_id;
            lane.request_generation = record.request_generation;
            lane.sequence_id = record.sequence_id;
            lane.context_token_count = record.token_count;
            lane.mtp_resolution_path_id = mtp_resolution_none;
            if record.token_count > maximum_token_count {
                maximum_token_count = record.token_count;
            }
        }
        packet.request_id = packet.lanes[0].request_id;
        packet.sequence_id = packet.lanes[0].sequence_id;
        packet.kv_block_table_token_count = maximum_token_count;
        self.stamp_work_packet(&mut packet)?;
        work_control::validate_packet(&packet, &self.config, self.execution_row_capacity, 1)
            .map_err(|_| ServingStatus::InvalidArgument)?;
        Ok(packet)
    }

    /// Drop `lane_count` release records after a successful release pump
    /// (the tail of `SparkRingServiceBackendPumpSequenceReleases`).
    pub fn pop_release_records(&mut self, lane_count: usize) {
        let lane_count = lane_count.min(self.release_count);
        self.release_head = (self.release_head + lane_count) % self.release_queue.len();
        self.release_count -= lane_count;
    }
}
