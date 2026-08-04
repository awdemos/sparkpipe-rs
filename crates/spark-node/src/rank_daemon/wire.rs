//! Wire codecs for the rank daemon's socket protocols.
//!
//! All layouts are little-endian at the C struct offsets, pinned against a
//! compile-and-run probe of the C headers: identity 56, acknowledgement 96,
//! work-control packet prefix 4288 / lane 96 / full packet 102592 (GLM52
//! capacities), final event 272.

use spark_sched::work_control::work_transaction::{
    fingerprint_bytes, Identity, ABI_VERSION, ACK_MAGIC,
};
use spark_sched::work_control::{WorkControlConfig, WorkControlPacket};

use super::resident_ipc::DsparkDraftResult;
use super::ring_runtime::{FinalEvent, FINAL_EVENT_DESCRIPTOR_BYTES, FINAL_EVENT_MAGIC};
use super::status::{status_from_code, Result, SparkStatus};

/// C: `SPARK_WORK_TRANSACTION_IDENTITY_BYTES`.
pub const IDENTITY_WIRE_BYTES: usize = 56;
/// C: `SPARK_WORK_TRANSACTION_ACKNOWLEDGEMENT_BYTES`.
pub const ACKNOWLEDGEMENT_WIRE_BYTES: usize = 96;
/// C: `SPARK_RING_RUNTIME_FINAL_EVENT_DESCRIPTOR_BYTES`.
pub const FINAL_EVENT_WIRE_BYTES: usize = 272;

/// C: `SPARK_WORK_TRANSACTION_STATE_ACCEPTED` (ack states on the wire).
pub const WIRE_STATE_ACCEPTED: u32 = 2;
/// C: `SPARK_WORK_TRANSACTION_STATE_FAILED`.
pub const WIRE_STATE_FAILED: u32 = 6;

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub(crate) struct Reader<'a> {
    pub bytes: &'a [u8],
    pub offset: usize,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    pub fn take(&mut self, count: usize) -> Option<&'a [u8]> {
        let slice = self.bytes.get(self.offset..self.offset + count)?;
        self.offset += count;
        Some(slice)
    }

    pub fn skip(&mut self, count: usize) -> Option<()> {
        self.take(count).map(|_| ())
    }

    pub fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    pub fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
}

/// C-layout identity bytes (the hash domain of the transaction ledger).
pub fn identity_to_wire(identity: &Identity) -> [u8; IDENTITY_WIRE_BYTES] {
    let mut out = Vec::with_capacity(IDENTITY_WIRE_BYTES);
    push_u64(&mut out, identity.control_generation);
    push_u64(&mut out, identity.transaction_id);
    push_u64(&mut out, identity.dispatch_generation);
    push_u64(&mut out, identity.request_generation);
    push_u64(&mut out, identity.step_generation);
    push_u32(&mut out, identity.step_chunk_index);
    push_u32(&mut out, identity.step_chunk_count);
    push_u32(&mut out, identity.transaction_phase);
    push_u32(&mut out, 0); // C: reserved0
    out.try_into().expect("identity layout is exactly 56 bytes")
}

pub fn identity_from_wire(bytes: &[u8; IDENTITY_WIRE_BYTES]) -> Identity {
    let mut cursor = Reader::new(bytes);
    // C: reserved0 (validated as zero by ValidateIdentity) has no Rust field.
    Identity {
        control_generation: cursor.u64().unwrap_or(0),
        transaction_id: cursor.u64().unwrap_or(0),
        dispatch_generation: cursor.u64().unwrap_or(0),
        request_generation: cursor.u64().unwrap_or(0),
        step_generation: cursor.u64().unwrap_or(0),
        step_chunk_index: cursor.u32().unwrap_or(0),
        step_chunk_count: cursor.u32().unwrap_or(0),
        transaction_phase: cursor.u32().unwrap_or(0),
    }
}

/// C: `SparkWorkTransactionFingerprintBytes(identity, IDENTITY_BYTES)`.
pub fn identity_fingerprint(identity: &Identity) -> u64 {
    fingerprint_bytes(&identity_to_wire(identity))
}

/// Serialize the packet's canonical C wire bytes (`[0, descriptor_bytes)`);
/// the exact bytes the C daemon writes to the next rank and submits to the
/// resident engine. Fails when the packet's descriptor is inconsistent.
pub fn packet_to_wire(config: &WorkControlConfig, packet: &WorkControlPacket) -> Result<Vec<u8>> {
    let expected = config.calculate_packet_bytes(packet.active_sequence_count);
    if expected == 0 || packet.descriptor_bytes != expected {
        return Err(SparkStatus::InvalidArgument);
    }
    Ok(packet.canonical_bytes(config))
}

/// Decode a packet from its C wire bytes (inverse of [`packet_to_wire`]).
/// `bytes.len()` must equal the packet's `descriptor_bytes`.
pub fn packet_from_wire(config: &WorkControlConfig, bytes: &[u8]) -> Result<WorkControlPacket> {
    let mut cursor = Reader::new(bytes);
    let mut packet = WorkControlPacket::zeroed(config);
    let read = |cursor: &mut Reader| cursor.u32().ok_or(SparkStatus::AbiMismatch);
    let read64 = |cursor: &mut Reader| cursor.u64().ok_or(SparkStatus::AbiMismatch);
    packet.magic = read(&mut cursor)?;
    packet.abi_version = read(&mut cursor)?;
    packet.descriptor_bytes = read(&mut cursor)?;
    packet.flags = read(&mut cursor)?;
    packet.request_id = read64(&mut cursor)?;
    packet.sequence_id = read64(&mut cursor)?;
    packet.sequence_position = read64(&mut cursor)?;
    packet.deadline_time_ns = read64(&mut cursor)?;
    packet.control_generation = read64(&mut cursor)?;
    packet.transaction_id = read64(&mut cursor)?;
    packet.dispatch_generation = read64(&mut cursor)?;
    packet.request_generation = read64(&mut cursor)?;
    packet.step_generation = read64(&mut cursor)?;
    packet.step_chunk_index = read(&mut cursor)?;
    packet.step_chunk_count = read(&mut cursor)?;
    packet.transaction_phase = read(&mut cursor)?;
    packet.reserved_transaction = read(&mut cursor)?;
    packet.active_sequence_count = read(&mut cursor)?;
    packet.new_token_count = read(&mut cursor)?;
    packet.pipeline_slot = read(&mut cursor)?;
    packet.priority = read(&mut cursor)?;
    packet.block_token_count = read(&mut cursor)?;
    packet.kv_block_table_token_count = read(&mut cursor)?;
    packet.max_blocks_per_sequence = read(&mut cursor)?;
    packet.mtp_draft_token_count = read(&mut cursor)?;
    packet.input_token_id = read(&mut cursor)?;
    packet.speculative_token_count = read(&mut cursor)?;
    packet.speculative_token_index = read(&mut cursor)?;
    for slot in packet.speculative_draft_token_ids.iter_mut() {
        *slot = read(&mut cursor)?;
    }
    packet.lane_count = read(&mut cursor)?;
    packet.rows_per_lane = read(&mut cursor)?;
    packet.execution_row_count = read(&mut cursor)?;
    packet.execution_batch_bucket = read(&mut cursor)?;
    for slot in packet.prefill_token_ids.iter_mut() {
        *slot = read(&mut cursor)?;
    }
    // Padding to the 8-aligned `lanes` offset (C struct layout).
    while cursor.offset % 8 != 0 {
        cursor.skip(1).ok_or(SparkStatus::AbiMismatch)?;
    }
    if cursor.offset as u32 != config.packet_prefix_bytes() {
        return Err(SparkStatus::AbiMismatch);
    }
    let lane_bytes = config.lane_bytes() as usize;
    packet.set_lane_count(config, packet.lane_count);
    for lane_index in 0..packet.lane_count as usize {
        let lane_start = cursor.offset;
        let lane = &mut packet.lanes[lane_index];
        lane.request_id = read64(&mut cursor)?;
        lane.request_generation = read64(&mut cursor)?;
        lane.step_generation = read64(&mut cursor)?;
        lane.sequence_id = read64(&mut cursor)?;
        lane.sequence_position = read64(&mut cursor)?;
        lane.request_slot_index = read(&mut cursor)?;
        lane.context_token_count = read(&mut cursor)?;
        lane.input_token_id = read(&mut cursor)?;
        lane.mtp_draft_token_count = read(&mut cursor)?;
        lane.speculative_token_count = read(&mut cursor)?;
        lane.mtp_resolution_proposed_token_count =
            *cursor.take(1).ok_or(SparkStatus::AbiMismatch)?.first().unwrap_or(&0);
        lane.mtp_resolution_accepted_token_count =
            *cursor.take(1).ok_or(SparkStatus::AbiMismatch)?.first().unwrap_or(&0);
        let path_bytes = cursor.take(2).ok_or(SparkStatus::AbiMismatch)?;
        lane.mtp_resolution_path_id = u16::from_le_bytes([path_bytes[0], path_bytes[1]]);
        for slot in lane.speculative_draft_token_ids.iter_mut() {
            *slot = read(&mut cursor)?;
        }
        // Lane tail padding to the C lane stride.
        let consumed = cursor.offset - lane_start;
        if consumed > lane_bytes {
            return Err(SparkStatus::AbiMismatch);
        }
        cursor.skip(lane_bytes - consumed).ok_or(SparkStatus::AbiMismatch)?;
    }
    if cursor.offset != bytes.len() || bytes.len() != packet.descriptor_bytes as usize {
        return Err(SparkStatus::AbiMismatch);
    }
    Ok(packet)
}

/// C: `SparkWorkTransactionAcknowledgement` (96 bytes on the wire).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WorkAcknowledgement {
    pub magic: u32,
    pub abi_version: u32,
    pub descriptor_bytes: u32,
    /// C: `SparkStatus status` as its code.
    pub status: u32,
    pub transaction_state: u32,
    pub reserved0: u32,
    pub identity: Identity,
    pub packet_fingerprint: u64,
    pub reserved1: u64,
}

/// C: `SparkWorkTransactionInitializeAcknowledgement`.
pub fn initialize_acknowledgement(
    identity: Option<&Identity>,
    packet_fingerprint: u64,
    transaction_state: u32,
    status: SparkStatus,
) -> WorkAcknowledgement {
    WorkAcknowledgement {
        magic: ACK_MAGIC,
        abi_version: ABI_VERSION,
        descriptor_bytes: ACKNOWLEDGEMENT_WIRE_BYTES as u32,
        status: status as u32,
        transaction_state,
        reserved0: 0,
        identity: identity.copied().unwrap_or_default(),
        packet_fingerprint,
        reserved1: 0,
    }
}

/// C: `SparkWorkTransactionStateIsStored`.
fn state_is_stored(state: u32) -> bool {
    (1..=6).contains(&state)
}

/// C: `SparkWorkTransactionValidateAcknowledgement`; on success the ack's
/// carried status code is the `Ok` value.
pub fn validate_acknowledgement(
    acknowledgement: &WorkAcknowledgement,
    expected_identity: &Identity,
    expected_packet_fingerprint: u64,
) -> std::result::Result<u32, SparkStatus> {
    if expected_packet_fingerprint == 0 {
        return Err(SparkStatus::InvalidArgument);
    }
    if acknowledgement.magic != ACK_MAGIC
        || acknowledgement.abi_version != ABI_VERSION
        || acknowledgement.descriptor_bytes != ACKNOWLEDGEMENT_WIRE_BYTES as u32
    {
        return Err(SparkStatus::AbiMismatch);
    }
    if spark_sched::work_control::work_transaction::validate_identity(&acknowledgement.identity)
        .is_err()
        || acknowledgement.identity != *expected_identity
        || acknowledgement.packet_fingerprint != expected_packet_fingerprint
        || status_from_code(acknowledgement.status).is_none()
        || !state_is_stored(acknowledgement.transaction_state)
        || acknowledgement.reserved0 != 0
        || acknowledgement.reserved1 != 0
    {
        return Err(SparkStatus::ValidationFailed);
    }
    Ok(acknowledgement.status)
}

pub fn acknowledgement_to_wire(ack: &WorkAcknowledgement) -> [u8; ACKNOWLEDGEMENT_WIRE_BYTES] {
    let mut out = Vec::with_capacity(ACKNOWLEDGEMENT_WIRE_BYTES);
    push_u32(&mut out, ack.magic);
    push_u32(&mut out, ack.abi_version);
    push_u32(&mut out, ack.descriptor_bytes);
    push_u32(&mut out, ack.status);
    push_u32(&mut out, ack.transaction_state);
    push_u32(&mut out, ack.reserved0);
    out.extend_from_slice(&identity_to_wire(&ack.identity));
    push_u64(&mut out, ack.packet_fingerprint);
    push_u64(&mut out, ack.reserved1);
    out.try_into().expect("ack layout is exactly 96 bytes")
}

pub fn acknowledgement_from_wire(bytes: &[u8; ACKNOWLEDGEMENT_WIRE_BYTES]) -> WorkAcknowledgement {
    let mut cursor = Reader::new(bytes);
    let magic = cursor.u32().unwrap_or(0);
    let abi_version = cursor.u32().unwrap_or(0);
    let descriptor_bytes = cursor.u32().unwrap_or(0);
    let status = cursor.u32().unwrap_or(0);
    let transaction_state = cursor.u32().unwrap_or(0);
    let reserved0 = cursor.u32().unwrap_or(0);
    let identity = identity_from_wire(
        cursor
            .take(IDENTITY_WIRE_BYTES)
            .unwrap_or(&[0; IDENTITY_WIRE_BYTES])
            .try_into()
            .unwrap_or(&[0; IDENTITY_WIRE_BYTES]),
    );
    WorkAcknowledgement {
        magic,
        abi_version,
        descriptor_bytes,
        status,
        transaction_state,
        reserved0,
        identity,
        packet_fingerprint: cursor.u64().unwrap_or(0),
        reserved1: cursor.u64().unwrap_or(0),
    }
}

/// C: `SparkRingRuntimeFinalEvent` wire serialization (272 bytes).
pub fn final_event_to_wire(event: &FinalEvent) -> [u8; FINAL_EVENT_WIRE_BYTES] {
    let mut out = Vec::with_capacity(FINAL_EVENT_WIRE_BYTES);
    push_u32(&mut out, event.magic);
    push_u32(&mut out, event.descriptor_bytes);
    push_u32(&mut out, event.status);
    push_u32(&mut out, event.program_id);
    push_u32(&mut out, event.driver_dispatch_slot);
    push_u32(&mut out, event.accepted_token_count);
    push_u32(&mut out, event.completion_flags);
    push_u32(&mut out, event.token_count);
    for value in event.token_ids {
        push_u32(&mut out, value);
    }
    push_u32(&mut out, event.draft_token_count);
    for value in event.draft_token_ids {
        push_u32(&mut out, value);
    }
    // C struct padding: `request_id` (u64) is 8-aligned at offset 104.
    push_u32(&mut out, 0);
    push_u64(&mut out, event.request_id);
    push_u64(&mut out, event.sequence_id);
    push_u64(&mut out, event.sequence_position);
    push_u64(&mut out, event.service_time_ns);
    push_u64(&mut out, event.control_generation);
    push_u64(&mut out, event.transaction_id);
    push_u64(&mut out, event.dispatch_generation);
    push_u64(&mut out, event.request_generation);
    push_u64(&mut out, event.step_generation);
    push_u32(&mut out, event.step_chunk_index);
    push_u32(&mut out, event.step_chunk_count);
    push_u32(&mut out, event.transaction_phase);
    push_u32(&mut out, event.reserved_transaction);
    push_u32(&mut out, event.extension_flags);
    push_u32(&mut out, event.reserved0);
    // C: `SparkGlm52DsparkDraftResult` (72 bytes) inline.
    push_u32(&mut out, event.dspark_draft.abi_version);
    push_u32(&mut out, event.dspark_draft.descriptor_bytes);
    push_u32(&mut out, event.dspark_draft.flags);
    push_u32(&mut out, event.dspark_draft.token_count);
    for value in event.dspark_draft.confidence_milli {
        push_u32(&mut out, value);
    }
    for value in event.dspark_draft.token_ids {
        push_u32(&mut out, value);
    }
    debug_assert_eq!(out.len(), FINAL_EVENT_WIRE_BYTES);
    out.try_into().expect("final event layout is exactly 272 bytes")
}

pub fn final_event_from_wire(bytes: &[u8; FINAL_EVENT_WIRE_BYTES]) -> FinalEvent {
    let mut event = FinalEvent::default();
    let mut cursor = Reader::new(bytes);
    event.magic = cursor.u32().unwrap_or(0);
    event.descriptor_bytes = cursor.u32().unwrap_or(0);
    event.status = cursor.u32().unwrap_or(0);
    event.program_id = cursor.u32().unwrap_or(0);
    event.driver_dispatch_slot = cursor.u32().unwrap_or(0);
    event.accepted_token_count = cursor.u32().unwrap_or(0);
    event.completion_flags = cursor.u32().unwrap_or(0);
    event.token_count = cursor.u32().unwrap_or(0);
    for slot in event.token_ids.iter_mut() {
        *slot = cursor.u32().unwrap_or(0);
    }
    event.draft_token_count = cursor.u32().unwrap_or(0);
    for slot in event.draft_token_ids.iter_mut() {
        *slot = cursor.u32().unwrap_or(0);
    }
    cursor.skip(4); // C struct padding before the 8-aligned u64 group
    event.request_id = cursor.u64().unwrap_or(0);
    event.sequence_id = cursor.u64().unwrap_or(0);
    event.sequence_position = cursor.u64().unwrap_or(0);
    event.service_time_ns = cursor.u64().unwrap_or(0);
    event.control_generation = cursor.u64().unwrap_or(0);
    event.transaction_id = cursor.u64().unwrap_or(0);
    event.dispatch_generation = cursor.u64().unwrap_or(0);
    event.request_generation = cursor.u64().unwrap_or(0);
    event.step_generation = cursor.u64().unwrap_or(0);
    event.step_chunk_index = cursor.u32().unwrap_or(0);
    event.step_chunk_count = cursor.u32().unwrap_or(0);
    event.transaction_phase = cursor.u32().unwrap_or(0);
    event.reserved_transaction = cursor.u32().unwrap_or(0);
    event.extension_flags = cursor.u32().unwrap_or(0);
    event.reserved0 = cursor.u32().unwrap_or(0);
    event.dspark_draft = DsparkDraftResult {
        abi_version: cursor.u32().unwrap_or(0),
        descriptor_bytes: cursor.u32().unwrap_or(0),
        flags: cursor.u32().unwrap_or(0),
        token_count: cursor.u32().unwrap_or(0),
        confidence_milli: core::array::from_fn(|_| cursor.u32().unwrap_or(0)),
        token_ids: core::array::from_fn(|_| cursor.u32().unwrap_or(0)),
    };
    event
}

/// C: `SparkRingRuntimeValidateFinalEvent` checks performed by the receiver.
pub fn validate_final_event(event: &FinalEvent) -> Result<()> {
    if event.magic != FINAL_EVENT_MAGIC || event.descriptor_bytes != FINAL_EVENT_DESCRIPTOR_BYTES {
        return Err(SparkStatus::AbiMismatch);
    }
    Ok(())
}

/// The C daemon's FNV-32 packet dedup hash over the wire bytes
/// (`HashPacket` in `rank_daemon.c`: offset 2166136261, prime 16777619 over
/// `packet[0..descriptor_bytes)`).
pub fn packet_dedup_hash32(wire_bytes: &[u8]) -> u32 {
    let mut hash: u32 = 2166136261;
    for &byte in wire_bytes {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(16777619);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_sched::work_control::WorkControlConfig;

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

    fn glm52_packet() -> (WorkControlConfig, WorkControlPacket) {
        let config = glm52_work_config();
        let mut packet = WorkControlPacket::zeroed(&config);
        packet.active_sequence_count = 2;
        packet.lane_count = 2;
        packet.descriptor_bytes = config.calculate_packet_bytes(2);
        packet.set_lane_count(&config, 2);
        packet.request_id = 7;
        packet.lanes[0].sequence_id = 100;
        packet.lanes[1].sequence_id = 101;
        (config, packet)
    }

    #[test]
    fn packet_wire_round_trip() {
        let (config, packet) = glm52_packet();
        let bytes = packet_to_wire(&config, &packet).unwrap();
        assert_eq!(bytes.len(), packet.descriptor_bytes as usize);
        let decoded = packet_from_wire(&config, &bytes).unwrap();
        assert_eq!(decoded.request_id, 7);
        assert_eq!(decoded.lane_count, 2);
        assert_eq!(decoded.lanes[1].sequence_id, 101);
        assert_eq!(decoded.descriptor_bytes, packet.descriptor_bytes);
        // Wire bytes are canonical: re-encoding the decoded packet matches.
        assert_eq!(packet_to_wire(&config, &decoded).unwrap(), bytes);
    }

    #[test]
    fn packet_wire_rejects_short_reads() {
        let (config, packet) = glm52_packet();
        let bytes = packet_to_wire(&config, &packet).unwrap();
        assert_eq!(
            packet_from_wire(&config, &bytes[..bytes.len() - 1]).unwrap_err(),
            SparkStatus::AbiMismatch
        );
    }

    #[test]
    fn acknowledgement_round_trip_and_validate() {
        let identity = Identity {
            control_generation: 7,
            transaction_id: 42,
            dispatch_generation: 1,
            request_generation: 1001,
            step_generation: 2,
            step_chunk_index: 0,
            step_chunk_count: 1,
            transaction_phase: 1,
        };
        let ack =
            initialize_acknowledgement(Some(&identity), 99, WIRE_STATE_ACCEPTED, SparkStatus::Ok);
        assert_eq!(ack.magic, 0x4B41_5457);
        let bytes = acknowledgement_to_wire(&ack);
        assert_eq!(&bytes[0..4], &0x4B41_5457u32.to_le_bytes());
        let decoded = acknowledgement_from_wire(&bytes);
        assert_eq!(decoded, ack);
        assert_eq!(validate_acknowledgement(&decoded, &identity, 99), Ok(0));
        assert_eq!(
            validate_acknowledgement(&decoded, &identity, 98),
            Err(SparkStatus::ValidationFailed)
        );
        let mut bad = decoded;
        bad.magic = 0;
        assert_eq!(validate_acknowledgement(&bad, &identity, 99), Err(SparkStatus::AbiMismatch));
    }

    #[test]
    fn final_event_round_trip() {
        let event = FinalEvent {
            magic: FINAL_EVENT_MAGIC,
            descriptor_bytes: FINAL_EVENT_DESCRIPTOR_BYTES,
            status: 0,
            program_id: 3,
            token_count: 1,
            token_ids: [101, 0, 0, 0, 0, 0, 0, 0],
            request_id: 7,
            sequence_id: 8,
            transaction_id: 42,
            control_generation: 7,
            step_chunk_count: 1,
            transaction_phase: 1,
            ..FinalEvent::default()
        };
        let bytes = final_event_to_wire(&event);
        assert_eq!(&bytes[0..4], &0x3545_4650u32.to_le_bytes());
        // C probe offset: request_id at 104.
        assert_eq!(&bytes[104..112], &7u64.to_le_bytes());
        let decoded = final_event_from_wire(&bytes);
        assert_eq!(decoded, event);
        assert_eq!(validate_final_event(&decoded), Ok(()));
        let mut bad = event;
        bad.magic = 0;
        assert_eq!(validate_final_event(&bad), Err(SparkStatus::AbiMismatch));
    }

    #[test]
    fn dedup_hash_matches_c_fnv32() {
        // C: offset 2166136261, prime 16777619 over the wire bytes.
        assert_eq!(packet_dedup_hash32(&[]), 2166136261);
        assert_eq!(packet_dedup_hash32(b"a"), 0xE40C_292C);
        assert_ne!(packet_dedup_hash32(b"ab"), packet_dedup_hash32(b"ac"));
    }
}
