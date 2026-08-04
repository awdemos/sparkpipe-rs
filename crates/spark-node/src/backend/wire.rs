//! Wire codecs for the backend's socket protocols, as explicit little-endian
//! reads/writes at the C struct offsets (the pattern `spark-serve`'s frame
//! codec established). Layouts were probed against the C tree:
//!
//! - `SparkWorkTransactionAcknowledgement` — 96 bytes.
//! - `SparkRingRuntimeFinalEvent` — 272 bytes (embeds
//!   `SparkRequestModelDraftResult` a.k.a. `SparkGlm52DsparkDraftResult`, 72).
//! - `SparkCudaResidentIpc*` — header 32, hello 32, stats 592,
//!   submit-result 600, completion 264 (embeds `SparkModelDriverCompletion`,
//!   184), submit-decode lane 96, decode header 98392.
//! - `SparkRingWorkControlPacket` canonical serialization — the exact byte
//!   domain `spark_sched::work_control` hashes and the C writes to the
//!   socket. Re-implemented here because `spark-sched`'s copy is private and
//!   the backend is the socket endpoint.
//!
//! Also ported here, because `spark-sched`'s `work_transaction` submodule
//! did not need them: the credit ledger (`SparkWorkTransactionCreditLedger`)
//! and the acknowledgement init/validate pair (`spark_distributed_work.h`
//! inline wrappers included).

use spark_sched::work_control::work_transaction::{self, Identity};
use spark_sched::work_control::{WorkControlConfig, WorkControlPacket};
use spark_serve::serving_engine::ServingStatus;

// ---------------------------------------------------------------------------
// Little-endian helpers
// ---------------------------------------------------------------------------

pub fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("codec bounds"))
}

pub fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("codec bounds"))
}

pub fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

pub fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

/// Map a wire status word onto the shared status space
/// (`event->status > SPARK_STATUS_UNSUPPORTED` is the C's validity bound).
pub fn status_from_u32(value: u32) -> Option<ServingStatus> {
    Some(match value {
        0 => ServingStatus::Ok,
        1 => ServingStatus::InvalidArgument,
        2 => ServingStatus::CapacityExceeded,
        3 => ServingStatus::NotFound,
        4 => ServingStatus::IoError,
        5 => ServingStatus::ParseError,
        6 => ServingStatus::SchemaError,
        7 => ServingStatus::HashMismatch,
        8 => ServingStatus::ModuleNotValidated,
        9 => ServingStatus::ValidationFailed,
        10 => ServingStatus::AbiMismatch,
        11 => ServingStatus::TargetMismatch,
        12 => ServingStatus::CompilerError,
        13 => ServingStatus::DriverLoadError,
        14 => ServingStatus::RouteNotFound,
        15 => ServingStatus::Busy,
        16 => ServingStatus::Duplicate,
        17 => ServingStatus::InternalError,
        18 => ServingStatus::Pending,
        19 => ServingStatus::Unsupported,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Credit ledger (`SparkWorkTransactionCreditLedger` + the distributed-work
// inline wrappers; `spark-sched` ported only the identity half)
// ---------------------------------------------------------------------------

/// Little-endian u16 helpers (decode-lane resolution path ids).
pub fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().expect("u16 read bounds"))
}

/// Little-endian u16 store.
pub fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

/// `SPARK_WORK_TRANSACTION_CREDIT_DOMAIN_*`.
pub const CREDIT_DOMAIN_TRANSPORT_WINDOW: usize = 0;
pub const CREDIT_DOMAIN_RESIDENT_RESERVATION: usize = 1;
pub const CREDIT_DOMAIN_EXECUTION: usize = 2;
pub const CREDIT_DOMAIN_COMPLETION_OWNERSHIP: usize = 3;
pub const CREDIT_DOMAIN_COUNT: usize = 4;

/// `SparkWorkTransactionCreditDomain`.
#[derive(Debug, Clone, Copy, Default)]
struct CreditDomain {
    capacity: u32,
    in_use: u32,
}

/// `SparkWorkTransactionCreditLedger` (the abi/descriptor words are constant
/// by construction here, so validation reduces to domain bounds).
#[derive(Debug, Clone, Default)]
pub struct CreditLedger {
    domains: [CreditDomain; CREDIT_DOMAIN_COUNT],
}

impl CreditLedger {
    /// `SparkDistributedWorkInitializeCreditLedger`.
    pub fn initialize(capacities: [u32; CREDIT_DOMAIN_COUNT]) -> Self {
        let mut ledger = CreditLedger::default();
        for (domain, capacity) in ledger.domains.iter_mut().zip(capacities) {
            domain.capacity = capacity;
        }
        ledger
    }

    /// `SparkDistributedWorkAcquireCredits` (CAPACITY_EXCEEDED maps to BUSY
    /// in the distributed-work wrapper).
    pub fn acquire(&mut self, domain: usize, credit_count: u32) -> Result<(), ServingStatus> {
        if domain >= CREDIT_DOMAIN_COUNT || credit_count == 0 {
            return Err(ServingStatus::InvalidArgument);
        }
        let credit_domain = &mut self.domains[domain];
        if credit_domain.in_use > credit_domain.capacity
            || credit_count > credit_domain.capacity - credit_domain.in_use
        {
            return Err(ServingStatus::Busy);
        }
        credit_domain.in_use += credit_count;
        Ok(())
    }

    /// `SparkDistributedWorkReleaseCredits`.
    pub fn release(&mut self, domain: usize, credit_count: u32) -> Result<(), ServingStatus> {
        if domain >= CREDIT_DOMAIN_COUNT || credit_count == 0 {
            return Err(ServingStatus::InvalidArgument);
        }
        let credit_domain = &mut self.domains[domain];
        if credit_count > credit_domain.in_use {
            return Err(ServingStatus::ValidationFailed);
        }
        credit_domain.in_use -= credit_count;
        Ok(())
    }

    /// `SparkDistributedWorkAvailableCredits`.
    pub fn available(&self, domain: usize) -> u32 {
        if domain >= CREDIT_DOMAIN_COUNT {
            return 0;
        }
        let credit_domain = &self.domains[domain];
        if credit_domain.in_use > credit_domain.capacity {
            return 0;
        }
        credit_domain.capacity - credit_domain.in_use
    }

    /// `memset(&ledger, 0, ...)` on teardown.
    pub fn reset(&mut self) {
        *self = CreditLedger::default();
    }
}

/// `SparkWorkTransactionIdentitiesMatch` (the Rust `Identity` has no
/// `reserved0` field; it is zero by construction on both sides).
pub fn identities_match(left: &Identity, right: &Identity) -> bool {
    left.control_generation == right.control_generation
        && left.transaction_id == right.transaction_id
        && left.dispatch_generation == right.dispatch_generation
        && left.request_generation == right.request_generation
        && left.step_generation == right.step_generation
        && left.step_chunk_index == right.step_chunk_index
        && left.step_chunk_count == right.step_chunk_count
        && left.transaction_phase == right.transaction_phase
}

// ---------------------------------------------------------------------------
// Work acknowledgement (`SparkWorkTransactionAcknowledgement`, 96 bytes)
// ---------------------------------------------------------------------------

/// `SPARK_WORK_TRANSACTION_ACK_MAGIC`.
pub const ACK_MAGIC: u32 = work_transaction::ACK_MAGIC;
/// `SPARK_WORK_TRANSACTION_ABI_VERSION`.
pub const WORK_TRANSACTION_ABI_VERSION: u32 = work_transaction::ABI_VERSION;
/// `SPARK_WORK_TRANSACTION_ACKNOWLEDGEMENT_BYTES`.
pub const ACKNOWLEDGEMENT_BYTES: usize = 96;

/// `SPARK_WORK_TRANSACTION_STATE_ACCEPTED` / `..._FAILED`.
pub const TRANSACTION_STATE_ACCEPTED: u32 = 2;
pub const TRANSACTION_STATE_FAILED: u32 = 6;

/// `SparkDistributedWorkAcknowledgement`.
#[derive(Debug, Clone, Copy, Default)]
pub struct WorkAcknowledgement {
    pub status: u32,
    pub transaction_state: u32,
    pub identity: Identity,
    pub packet_fingerprint: u64,
}

impl WorkAcknowledgement {
    /// `SparkDistributedWorkInitializeAcknowledgement`.
    pub fn initialize(identity: &Identity, packet_fingerprint: u64, status: ServingStatus) -> Self {
        WorkAcknowledgement {
            status: status.code(),
            transaction_state: if status == ServingStatus::Ok || status == ServingStatus::Duplicate
            {
                TRANSACTION_STATE_ACCEPTED
            } else {
                TRANSACTION_STATE_FAILED
            },
            identity: *identity,
            packet_fingerprint,
        }
    }

    /// Serialize to the 96-byte wire form.
    pub fn encode(&self) -> [u8; ACKNOWLEDGEMENT_BYTES] {
        let mut bytes = [0u8; ACKNOWLEDGEMENT_BYTES];
        write_u32(&mut bytes, 0, ACK_MAGIC);
        write_u32(&mut bytes, 4, WORK_TRANSACTION_ABI_VERSION);
        write_u32(&mut bytes, 8, ACKNOWLEDGEMENT_BYTES as u32);
        write_u32(&mut bytes, 12, self.status);
        write_u32(&mut bytes, 16, self.transaction_state);
        // 20: reserved0
        write_u64(&mut bytes, 24, self.identity.control_generation);
        write_u64(&mut bytes, 32, self.identity.transaction_id);
        write_u64(&mut bytes, 40, self.identity.dispatch_generation);
        write_u64(&mut bytes, 48, self.identity.request_generation);
        write_u64(&mut bytes, 56, self.identity.step_generation);
        write_u32(&mut bytes, 64, self.identity.step_chunk_index);
        write_u32(&mut bytes, 68, self.identity.step_chunk_count);
        write_u32(&mut bytes, 72, self.identity.transaction_phase);
        // 76: identity.reserved0
        write_u64(&mut bytes, 80, self.packet_fingerprint);
        // 88: reserved1
        bytes
    }

    /// Parse the 96-byte wire form.
    pub fn decode(bytes: &[u8]) -> Result<Self, ServingStatus> {
        if bytes.len() < ACKNOWLEDGEMENT_BYTES {
            return Err(ServingStatus::InvalidArgument);
        }
        Ok(WorkAcknowledgement {
            status: read_u32(bytes, 12),
            transaction_state: read_u32(bytes, 16),
            identity: Identity {
                control_generation: read_u64(bytes, 24),
                transaction_id: read_u64(bytes, 32),
                dispatch_generation: read_u64(bytes, 40),
                request_generation: read_u64(bytes, 48),
                step_generation: read_u64(bytes, 56),
                step_chunk_index: read_u32(bytes, 64),
                step_chunk_count: read_u32(bytes, 68),
                transaction_phase: read_u32(bytes, 72),
            },
            packet_fingerprint: read_u64(bytes, 80),
        })
    }

    /// `SparkDistributedWorkValidateAcknowledgement`. On a well-formed ack
    /// the ack's own status is returned, exactly as the C.
    pub fn validate(
        &self,
        raw: &[u8],
        expected_identity: &Identity,
        expected_packet_fingerprint: u64,
    ) -> ServingStatus {
        if expected_packet_fingerprint == 0 {
            return ServingStatus::InvalidArgument;
        }
        if raw.len() < ACKNOWLEDGEMENT_BYTES
            || read_u32(raw, 0) != ACK_MAGIC
            || read_u32(raw, 4) != WORK_TRANSACTION_ABI_VERSION
            || read_u32(raw, 8) != ACKNOWLEDGEMENT_BYTES as u32
        {
            return ServingStatus::AbiMismatch;
        }
        if work_transaction::validate_identity(&self.identity).is_err()
            || !identities_match(&self.identity, expected_identity)
            || self.packet_fingerprint != expected_packet_fingerprint
            || status_from_u32(self.status).is_none()
            || !(1..=7).contains(&self.transaction_state)
            || read_u32(raw, 20) != 0
            || read_u32(raw, 76) != 0
            || read_u64(raw, 88) != 0
        {
            return ServingStatus::ValidationFailed;
        }
        status_from_u32(self.status).unwrap_or(ServingStatus::ValidationFailed)
    }
}

// ---------------------------------------------------------------------------
// dspark draft result (`SparkRequestModelDraftResult`, 72 bytes)
// ---------------------------------------------------------------------------

/// `SPARK_REQUEST_MODEL_MAX_SPECULATIVE_TOKENS`.
pub const DRAFT_TOKEN_CAPACITY: usize = 7;
/// `SPARK_GLM52_DSPARK_ABI_VERSION` (= `SPARK_REQUEST_MODEL_ABI_VERSION`).
pub const DSPARK_ABI_VERSION: u32 = 2;
/// `SPARK_GLM52_DSPARK_DRAFT_RESULT_DESCRIPTOR_BYTES`.
pub const DSPARK_DRAFT_RESULT_BYTES: usize = 72;

/// `SparkGlm52DsparkDraftResult` (a.k.a. `SparkRequestModelDraftResult`).
#[derive(Debug, Clone, Copy, Default)]
pub struct DsparkDraftResult {
    pub abi_version: u32,
    pub descriptor_bytes: u32,
    pub flags: u32,
    pub token_count: u32,
    pub confidence_milli: [u32; DRAFT_TOKEN_CAPACITY],
    pub token_ids: [u32; DRAFT_TOKEN_CAPACITY],
}

impl DsparkDraftResult {
    pub fn decode(bytes: &[u8]) -> Self {
        let mut draft = DsparkDraftResult {
            abi_version: read_u32(bytes, 0),
            descriptor_bytes: read_u32(bytes, 4),
            flags: read_u32(bytes, 8),
            token_count: read_u32(bytes, 12),
            ..DsparkDraftResult::default()
        };
        for index in 0..DRAFT_TOKEN_CAPACITY {
            draft.confidence_milli[index] = read_u32(bytes, 16 + index * 4);
            draft.token_ids[index] = read_u32(bytes, 44 + index * 4);
        }
        draft
    }

    pub fn encode(&self, bytes: &mut [u8]) {
        write_u32(bytes, 0, self.abi_version);
        write_u32(bytes, 4, self.descriptor_bytes);
        write_u32(bytes, 8, self.flags);
        write_u32(bytes, 12, self.token_count);
        for index in 0..DRAFT_TOKEN_CAPACITY {
            write_u32(bytes, 16 + index * 4, self.confidence_milli[index]);
            write_u32(bytes, 44 + index * 4, self.token_ids[index]);
        }
    }
}

// ---------------------------------------------------------------------------
// Final event (`SparkRingRuntimeFinalEvent`, 272 bytes)
// ---------------------------------------------------------------------------

/// `SPARK_RING_RUNTIME_FINAL_EVENT_MAGIC`.
pub const FINAL_EVENT_MAGIC: u32 = 0x3545_4650;
/// `SPARK_RING_RUNTIME_FINAL_EVENT_DESCRIPTOR_BYTES`.
pub const FINAL_EVENT_BYTES: usize = 272;
/// `SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY`.
pub const COMPLETION_TOKEN_CAPACITY: usize = 8;
/// `SPARK_MODEL_DRIVER_COMPLETION_DRAFT_TOKEN_CAPACITY`.
pub const COMPLETION_DRAFT_TOKEN_CAPACITY: usize = 8;
/// `SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS`.
pub const COMPLETION_FLAG_TOKEN_IDS: u32 = 0x0000_0001;
/// `SPARK_MODEL_DRIVER_COMPLETION_FLAG_DRAFT_TOKEN_IDS`.
pub const COMPLETION_FLAG_DRAFT_TOKEN_IDS: u32 = 0x0000_0002;
/// `SPARK_RING_SERVICE_BACKEND_MODEL_COMPLETION_KNOWN_FLAGS`.
pub const MODEL_COMPLETION_KNOWN_FLAGS: u32 =
    COMPLETION_FLAG_TOKEN_IDS | COMPLETION_FLAG_DRAFT_TOKEN_IDS;
/// `SPARK_RING_RUNTIME_FINAL_EVENT_FLAG_DSPARK_DRAFT`.
pub const FINAL_EVENT_FLAG_DSPARK_DRAFT: u32 = 0x0000_0001;
/// `SPARK_RING_RUNTIME_FINAL_EVENT_KNOWN_FLAGS`.
pub const FINAL_EVENT_KNOWN_FLAGS: u32 = FINAL_EVENT_FLAG_DSPARK_DRAFT;

/// `SparkRingRuntimeFinalEvent`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FinalEvent {
    pub status: u32,
    pub program_id: u32,
    pub driver_dispatch_slot: u32,
    pub accepted_token_count: u32,
    pub completion_flags: u32,
    pub token_count: u32,
    pub token_ids: [u32; COMPLETION_TOKEN_CAPACITY],
    pub draft_token_count: u32,
    pub draft_token_ids: [u32; COMPLETION_DRAFT_TOKEN_CAPACITY],
    pub request_id: u64,
    pub sequence_id: u64,
    pub sequence_position: u64,
    pub service_time_ns: u64,
    pub control_generation: u64,
    pub transaction_id: u64,
    pub dispatch_generation: u64,
    pub request_generation: u64,
    pub step_generation: u64,
    pub step_chunk_index: u32,
    pub step_chunk_count: u32,
    pub transaction_phase: u32,
    pub reserved_transaction: u32,
    pub extension_flags: u32,
    pub dspark_draft: DsparkDraftResult,
}

impl FinalEvent {
    pub fn decode(bytes: &[u8]) -> Result<Self, ServingStatus> {
        if bytes.len() < FINAL_EVENT_BYTES
            || read_u32(bytes, 0) != FINAL_EVENT_MAGIC
            || read_u32(bytes, 4) != FINAL_EVENT_BYTES as u32
        {
            return Err(ServingStatus::AbiMismatch);
        }
        let mut event = FinalEvent {
            status: read_u32(bytes, 8),
            program_id: read_u32(bytes, 12),
            driver_dispatch_slot: read_u32(bytes, 16),
            accepted_token_count: read_u32(bytes, 20),
            completion_flags: read_u32(bytes, 24),
            token_count: read_u32(bytes, 28),
            draft_token_count: read_u32(bytes, 64),
            request_id: read_u64(bytes, 104),
            sequence_id: read_u64(bytes, 112),
            sequence_position: read_u64(bytes, 120),
            service_time_ns: read_u64(bytes, 128),
            control_generation: read_u64(bytes, 136),
            transaction_id: read_u64(bytes, 144),
            dispatch_generation: read_u64(bytes, 152),
            request_generation: read_u64(bytes, 160),
            step_generation: read_u64(bytes, 168),
            step_chunk_index: read_u32(bytes, 176),
            step_chunk_count: read_u32(bytes, 180),
            transaction_phase: read_u32(bytes, 184),
            reserved_transaction: read_u32(bytes, 188),
            extension_flags: read_u32(bytes, 192),
            dspark_draft: DsparkDraftResult::decode(&bytes[200..]),
            ..FinalEvent::default()
        };
        for index in 0..COMPLETION_TOKEN_CAPACITY {
            event.token_ids[index] = read_u32(bytes, 32 + index * 4);
            event.draft_token_ids[index] = read_u32(bytes, 68 + index * 4);
        }
        Ok(event)
    }

    pub fn encode(&self) -> [u8; FINAL_EVENT_BYTES] {
        let mut bytes = [0u8; FINAL_EVENT_BYTES];
        write_u32(&mut bytes, 0, FINAL_EVENT_MAGIC);
        write_u32(&mut bytes, 4, FINAL_EVENT_BYTES as u32);
        write_u32(&mut bytes, 8, self.status);
        write_u32(&mut bytes, 12, self.program_id);
        write_u32(&mut bytes, 16, self.driver_dispatch_slot);
        write_u32(&mut bytes, 20, self.accepted_token_count);
        write_u32(&mut bytes, 24, self.completion_flags);
        write_u32(&mut bytes, 28, self.token_count);
        for index in 0..COMPLETION_TOKEN_CAPACITY {
            write_u32(&mut bytes, 32 + index * 4, self.token_ids[index]);
        }
        write_u32(&mut bytes, 64, self.draft_token_count);
        for index in 0..COMPLETION_DRAFT_TOKEN_CAPACITY {
            write_u32(&mut bytes, 68 + index * 4, self.draft_token_ids[index]);
        }
        write_u64(&mut bytes, 104, self.request_id);
        write_u64(&mut bytes, 112, self.sequence_id);
        write_u64(&mut bytes, 120, self.sequence_position);
        write_u64(&mut bytes, 128, self.service_time_ns);
        write_u64(&mut bytes, 136, self.control_generation);
        write_u64(&mut bytes, 144, self.transaction_id);
        write_u64(&mut bytes, 152, self.dispatch_generation);
        write_u64(&mut bytes, 160, self.request_generation);
        write_u64(&mut bytes, 168, self.step_generation);
        write_u32(&mut bytes, 176, self.step_chunk_index);
        write_u32(&mut bytes, 180, self.step_chunk_count);
        write_u32(&mut bytes, 184, self.transaction_phase);
        write_u32(&mut bytes, 188, self.reserved_transaction);
        write_u32(&mut bytes, 192, self.extension_flags);
        self.dspark_draft.encode(&mut bytes[200..]);
        bytes
    }

    /// `SparkRingServiceBackendValidateFinalEventEnvelope`.
    pub fn validate_envelope(&self) -> Result<(), ServingStatus> {
        if status_from_u32(self.status).is_none()
            || (self.completion_flags & !MODEL_COMPLETION_KNOWN_FLAGS) != 0
            || (self.extension_flags & !FINAL_EVENT_KNOWN_FLAGS) != 0
            || self.request_id == 0
            || self.control_generation == 0
            || self.transaction_id == 0
            || self.dispatch_generation == 0
            || self.request_generation == 0
            || self.sequence_id == 0
            || self.step_generation == 0
            || self.step_chunk_count == 0
            || self.step_chunk_index >= self.step_chunk_count
            || (self.transaction_phase != work_transaction::PHASE_DECODE
                && self.transaction_phase != work_transaction::PHASE_VERIFY)
            || self.reserved_transaction != 0
            || self.token_count as usize > COMPLETION_TOKEN_CAPACITY
            || self.draft_token_count as usize > COMPLETION_DRAFT_TOKEN_CAPACITY
        {
            return Err(ServingStatus::ValidationFailed);
        }
        let has_token_ids = (self.completion_flags & COMPLETION_FLAG_TOKEN_IDS) != 0;
        let has_draft_tokens = (self.completion_flags & COMPLETION_FLAG_DRAFT_TOKEN_IDS) != 0;
        if (has_token_ids != (self.token_count != 0))
            || (has_draft_tokens != (self.draft_token_count != 0))
        {
            return Err(ServingStatus::ValidationFailed);
        }
        if self.status == ServingStatus::Ok.code() && (!has_token_ids || self.token_count == 0) {
            return Err(ServingStatus::ValidationFailed);
        }
        Ok(())
    }

    /// `SparkRingServiceBackendFinalEventIdentityMatches`.
    pub fn identity_matches(&self, other: &FinalEvent) -> bool {
        self.control_generation == other.control_generation
            && self.transaction_id == other.transaction_id
            && self.dispatch_generation == other.dispatch_generation
            && self.request_id == other.request_id
            && self.request_generation == other.request_generation
            && self.sequence_id == other.sequence_id
            && self.sequence_position == other.sequence_position
            && self.step_generation == other.step_generation
            && self.step_chunk_index == other.step_chunk_index
            && self.step_chunk_count == other.step_chunk_count
            && self.transaction_phase == other.transaction_phase
    }

    /// The event fingerprint (`SparkDistributedWorkHashBytes` over the wire
    /// descriptor bytes, as in C).
    pub fn fingerprint(&self) -> u64 {
        work_transaction::fingerprint_bytes(&self.encode())
    }
}

// ---------------------------------------------------------------------------
// CUDA resident IPC codecs (`SparkCudaResidentIpc*`)
// ---------------------------------------------------------------------------

/// `SPARK_CUDA_RESIDENT_IPC_ABI_VERSION`.
pub const IPC_ABI_VERSION: u32 = 24;
/// `SPARK_CUDA_RESIDENT_IPC_MAGIC`.
pub const IPC_MAGIC: u32 = 0x5244_5543;
/// `SPARK_CUDA_RESIDENT_IPC_HEADER_BYTES`.
pub const IPC_HEADER_BYTES: usize = 32;
/// `SPARK_CUDA_RESIDENT_IPC_HELLO_BYTES`.
pub const IPC_HELLO_BYTES: usize = 32;
/// `SPARK_CUDA_RESIDENT_IPC_STATS_BYTES`.
pub const IPC_STATS_BYTES: usize = 592;
/// `SPARK_CUDA_RESIDENT_IPC_SUBMIT_RESULT_BYTES`.
pub const IPC_SUBMIT_RESULT_BYTES: usize = 600;
/// `SPARK_CUDA_RESIDENT_IPC_COMPLETION_BYTES`.
pub const IPC_COMPLETION_BYTES: usize = 264;
/// `SPARK_CUDA_RESIDENT_IPC_ERROR_TEXT_BYTES`.
pub const IPC_ERROR_TEXT_BYTES: usize = 160;
/// `SPARK_CUDA_RESIDENT_IPC_SUBMIT_DECODE_HEADER_BYTES`
/// (`offsetof(SparkCudaResidentIpcSubmitDecode, kv_physical_block_indices)`).
pub const IPC_SUBMIT_DECODE_HEADER_BYTES: usize = 98_392;
/// `SPARK_CUDA_RESIDENT_IPC_MAX_CONTROL_PAYLOAD_BYTES` for the reader cap:
/// the largest payload the backend ever reads (submit-result, 600 bytes).
pub const IPC_MAX_READ_PAYLOAD_BYTES: usize = IPC_SUBMIT_RESULT_BYTES;

/// `SPARK_CUDA_RESIDENT_IPC_KIND_*`.
pub const IPC_KIND_HELLO: u32 = 1;
pub const IPC_KIND_HELLO_ACK: u32 = 2;
pub const IPC_KIND_SUBMIT_WORK: u32 = 3;
pub const IPC_KIND_SUBMIT_RESULT: u32 = 4;
pub const IPC_KIND_COMPLETION: u32 = 5;
pub const IPC_KIND_SUBMIT_PREFILL: u32 = 10;
pub const IPC_KIND_SUBMIT_DECODE: u32 = 11;

/// `SPARK_CUDA_RESIDENT_IPC_STATE_READY`.
pub const IPC_STATE_READY: u32 = 2;

/// `SPARK_CUDA_RESIDENT_IPC_COMPLETION_FLAG_DSPARK_DRAFT` / known flags.
pub const IPC_COMPLETION_FLAG_DSPARK_DRAFT: u32 = 0x0000_0001;
pub const IPC_COMPLETION_KNOWN_FLAGS: u32 = IPC_COMPLETION_FLAG_DSPARK_DRAFT;

/// `SPARK_CUDA_RESIDENT_IPC_SUBMIT_FLAG_INTERNAL_KV_DIRECTORY`.
pub const IPC_SUBMIT_FLAG_INTERNAL_KV_DIRECTORY: u32 = 0x0000_0001;
/// `SPARK_CUDA_RESIDENT_IPC_SUBMIT_KNOWN_FLAGS`.
pub const IPC_SUBMIT_KNOWN_FLAGS: u32 = IPC_SUBMIT_FLAG_INTERNAL_KV_DIRECTORY;
/// `SPARK_CUDA_RESIDENT_IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT`.
pub const IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT: u32 = 0x0000_0001;

/// `SPARK_RING_NODE_CONTEXT_BUILDER_NVME_MODE_*` (validated on handshake).
pub const NVME_MODE_SYNCHRONOUS_FULL_HISTORY: u32 = 1;
pub const NVME_MODE_BATCHED_COHORT_JIT: u32 = 2;
pub const NVME_MODE_ASYNC_SELECTED_JIT: u32 = 3;

/// `SparkCudaResidentIpcHeader`.
#[derive(Debug, Clone, Copy, Default)]
pub struct IpcHeader {
    pub kind: u32,
    pub payload_bytes: u32,
    pub rank_index: u32,
    pub sequence_number: u64,
}

impl IpcHeader {
    /// `SparkCudaResidentIpcInitializeHeader`.
    pub fn initialize(
        kind: u32,
        rank_index: u32,
        sequence_number: u64,
        payload_bytes: u32,
    ) -> Self {
        IpcHeader { kind, payload_bytes, rank_index, sequence_number }
    }

    pub fn encode(&self) -> [u8; IPC_HEADER_BYTES] {
        let mut bytes = [0u8; IPC_HEADER_BYTES];
        write_u32(&mut bytes, 0, IPC_MAGIC);
        write_u32(&mut bytes, 4, IPC_ABI_VERSION);
        write_u32(&mut bytes, 8, IPC_HEADER_BYTES as u32);
        write_u32(&mut bytes, 12, self.kind);
        write_u32(&mut bytes, 16, self.payload_bytes);
        write_u32(&mut bytes, 20, self.rank_index);
        write_u64(&mut bytes, 24, self.sequence_number);
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Self {
        IpcHeader {
            kind: read_u32(bytes, 12),
            payload_bytes: read_u32(bytes, 16),
            rank_index: read_u32(bytes, 20),
            sequence_number: read_u64(bytes, 24),
        }
    }

    /// `SparkCudaResidentIpcValidateHeader` (`expected_kind == 0` = any).
    pub fn validate(
        &self,
        raw: &[u8],
        expected_kind: u32,
        maximum_payload_bytes: u32,
    ) -> Result<(), ServingStatus> {
        if raw.len() < IPC_HEADER_BYTES
            || read_u32(raw, 0) != IPC_MAGIC
            || read_u32(raw, 4) != IPC_ABI_VERSION
            || read_u32(raw, 8) != IPC_HEADER_BYTES as u32
            || (expected_kind != 0 && self.kind != expected_kind)
            || self.payload_bytes > maximum_payload_bytes
        {
            return Err(ServingStatus::AbiMismatch);
        }
        Ok(())
    }
}

/// `SparkCudaResidentIpcHello`.
#[derive(Debug, Clone, Copy, Default)]
pub struct IpcHello {
    pub rank_index: u32,
    pub rank_count: u32,
    pub expected_cuda_generation: u32,
    pub control_generation: u64,
    pub process_id: u64,
}

impl IpcHello {
    pub fn encode(&self) -> [u8; IPC_HELLO_BYTES] {
        let mut bytes = [0u8; IPC_HELLO_BYTES];
        write_u32(&mut bytes, 0, IPC_HELLO_BYTES as u32);
        write_u32(&mut bytes, 4, self.rank_index);
        write_u32(&mut bytes, 8, self.rank_count);
        write_u32(&mut bytes, 12, self.expected_cuda_generation);
        write_u64(&mut bytes, 16, self.control_generation);
        write_u64(&mut bytes, 24, self.process_id);
        bytes
    }
}

/// The fields of `SparkCudaResidentIpcStats` the handshake contract reads
/// (plus the blocker text), at their C offsets.
#[derive(Debug, Clone, Default)]
pub struct IpcStats {
    pub state: u32,
    pub kv_physical_block_capacity: u32,
    pub kv_logical_block_capacity: u32,
    pub kv_nvme_enabled: u32,
    pub kv_nvme_mode: u32,
    pub work_queue_depth: u32,
    pub work_queue_capacity: u32,
    pub logical_lane_capacity: u32,
    pub execution_row_capacity: u32,
    pub moe_backend_kind: u32,
    pub moe_bound_layer_count: u32,
    pub moe_expected_layer_count: u32,
    pub fp8_scaled_gemm_bound_plan_count: u32,
    pub fp8_scaled_gemm_expected_plan_count: u32,
    pub model_quantization_mode: u32,
    pub kv_resident_bytes_per_token: u64,
    pub kv_resident_pool_bytes: u64,
    pub kv_nvme_capacity_bytes: u64,
    pub kv_nvme_batch_block_capacity: u32,
    pub blocker: String,
}

impl IpcStats {
    pub fn decode(bytes: &[u8]) -> Result<Self, ServingStatus> {
        if bytes.len() < IPC_STATS_BYTES || read_u32(bytes, 0) != IPC_STATS_BYTES as u32 {
            return Err(ServingStatus::AbiMismatch);
        }
        let blocker_bytes = &bytes[432..432 + IPC_ERROR_TEXT_BYTES];
        let blocker_end =
            blocker_bytes.iter().position(|&b| b == 0).unwrap_or(IPC_ERROR_TEXT_BYTES);
        Ok(IpcStats {
            state: read_u32(bytes, 4),
            kv_nvme_enabled: read_u32(bytes, 72),
            kv_physical_block_capacity: read_u32(bytes, 76),
            kv_logical_block_capacity: read_u32(bytes, 80),
            kv_nvme_mode: read_u32(bytes, 84),
            kv_resident_bytes_per_token: read_u64(bytes, 176),
            kv_resident_pool_bytes: read_u64(bytes, 184),
            kv_nvme_capacity_bytes: read_u64(bytes, 192),
            kv_nvme_batch_block_capacity: read_u32(bytes, 208),
            work_queue_depth: read_u32(bytes, 224),
            work_queue_capacity: read_u32(bytes, 228),
            logical_lane_capacity: read_u32(bytes, 288),
            execution_row_capacity: read_u32(bytes, 292),
            moe_backend_kind: read_u32(bytes, 308),
            moe_bound_layer_count: read_u32(bytes, 312),
            moe_expected_layer_count: read_u32(bytes, 316),
            fp8_scaled_gemm_bound_plan_count: read_u32(bytes, 320),
            fp8_scaled_gemm_expected_plan_count: read_u32(bytes, 324),
            model_quantization_mode: read_u32(bytes, 328),
            blocker: String::from_utf8_lossy(&blocker_bytes[..blocker_end]).into_owned(),
        })
    }
}

/// `SparkCudaResidentIpcSubmitResult` (descriptor, status, stats).
#[derive(Debug, Clone, Default)]
pub struct IpcSubmitResult {
    pub status: u32,
    pub stats: IpcStats,
}

impl IpcSubmitResult {
    pub fn decode(bytes: &[u8]) -> Result<Self, ServingStatus> {
        if bytes.len() < IPC_SUBMIT_RESULT_BYTES
            || read_u32(bytes, 0) != IPC_SUBMIT_RESULT_BYTES as u32
        {
            return Err(ServingStatus::AbiMismatch);
        }
        Ok(IpcSubmitResult { status: read_u32(bytes, 4), stats: IpcStats::decode(&bytes[8..])? })
    }
}

/// `SparkCudaResidentIpcCompletion` — the fields the backend reads.
#[derive(Debug, Clone, Default)]
pub struct IpcCompletion {
    pub flags: u32,
    /// `completion.request_id`.
    pub request_id: u64,
    /// `completion.sequence_id`.
    pub sequence_id: u64,
    /// `completion.status`.
    pub status: u32,
}

impl IpcCompletion {
    pub fn decode(bytes: &[u8]) -> Result<Self, ServingStatus> {
        if bytes.len() < IPC_COMPLETION_BYTES || read_u32(bytes, 0) != IPC_COMPLETION_BYTES as u32 {
            return Err(ServingStatus::AbiMismatch);
        }
        let flags = read_u32(bytes, 4);
        if (flags & !IPC_COMPLETION_KNOWN_FLAGS) != 0 {
            return Err(ServingStatus::AbiMismatch);
        }
        // SparkModelDriverCompletion at offset 8: request_id +0,
        // sequence_id +8, status +112 (after token/draft arrays).
        Ok(IpcCompletion {
            flags,
            request_id: read_u64(bytes, 8),
            sequence_id: read_u64(bytes, 16),
            status: read_u32(bytes, 120),
        })
    }
}

/// `SparkCudaResidentIpcCalculateSubmitWorkBytes`
/// (8-byte prefix + the packet's descriptor bytes).
pub fn submit_work_bytes(packet_descriptor_bytes: u32) -> u32 {
    8 + packet_descriptor_bytes
}

// ---------------------------------------------------------------------------
// Work-control packet canonical serialization
// (mirrors spark-sched's private `canonical_bytes`; the backend is the wire
// endpoint and needs the bytes themselves)
// ---------------------------------------------------------------------------

/// Serialize a packet to its canonical little-endian form, exactly
/// `descriptor_bytes` long — the same bytes `SparkRingWorkControlPacket`
/// lays out in C memory. Returns `Err(InvalidArgument)` when the packet's
/// `descriptor_bytes` disagrees with the configured layout.
pub fn serialize_packet(
    config: &WorkControlConfig,
    packet: &WorkControlPacket,
) -> Result<Vec<u8>, ServingStatus> {
    if packet.descriptor_bytes != config.calculate_packet_bytes(packet.lane_count)
        || packet.descriptor_bytes < config.packet_prefix_bytes()
    {
        return Err(ServingStatus::InvalidArgument);
    }
    let mut bytes = Vec::with_capacity(packet.descriptor_bytes as usize);
    let push_u32 = |value: u32, bytes: &mut Vec<u8>| bytes.extend_from_slice(&value.to_le_bytes());
    let push_u64 = |value: u64, bytes: &mut Vec<u8>| bytes.extend_from_slice(&value.to_le_bytes());
    push_u32(packet.magic, &mut bytes);
    push_u32(packet.abi_version, &mut bytes);
    push_u32(packet.descriptor_bytes, &mut bytes);
    push_u32(packet.flags, &mut bytes);
    push_u64(packet.request_id, &mut bytes);
    push_u64(packet.sequence_id, &mut bytes);
    push_u64(packet.sequence_position, &mut bytes);
    push_u64(packet.deadline_time_ns, &mut bytes);
    push_u64(packet.control_generation, &mut bytes);
    push_u64(packet.transaction_id, &mut bytes);
    push_u64(packet.dispatch_generation, &mut bytes);
    push_u64(packet.request_generation, &mut bytes);
    push_u64(packet.step_generation, &mut bytes);
    push_u32(packet.step_chunk_index, &mut bytes);
    push_u32(packet.step_chunk_count, &mut bytes);
    push_u32(packet.transaction_phase, &mut bytes);
    push_u32(packet.reserved_transaction, &mut bytes);
    push_u32(packet.active_sequence_count, &mut bytes);
    push_u32(packet.new_token_count, &mut bytes);
    push_u32(packet.pipeline_slot, &mut bytes);
    push_u32(packet.priority, &mut bytes);
    push_u32(packet.block_token_count, &mut bytes);
    push_u32(packet.kv_block_table_token_count, &mut bytes);
    push_u32(packet.max_blocks_per_sequence, &mut bytes);
    push_u32(packet.mtp_draft_token_count, &mut bytes);
    push_u32(packet.input_token_id, &mut bytes);
    push_u32(packet.speculative_token_count, &mut bytes);
    push_u32(packet.speculative_token_index, &mut bytes);
    for &token_id in &packet.speculative_draft_token_ids {
        push_u32(token_id, &mut bytes);
    }
    push_u32(packet.lane_count, &mut bytes);
    push_u32(packet.rows_per_lane, &mut bytes);
    push_u32(packet.execution_row_count, &mut bytes);
    push_u32(packet.execution_batch_bucket, &mut bytes);
    for &token_id in &packet.prefill_token_ids {
        push_u32(token_id, &mut bytes);
    }
    while bytes.len() % 8 != 0 {
        bytes.push(0);
    }
    let lane_bytes = config.lane_bytes() as usize;
    for lane in packet.lanes.iter().take(packet.lane_count as usize) {
        let lane_start = bytes.len();
        push_u64(lane.request_id, &mut bytes);
        push_u64(lane.request_generation, &mut bytes);
        push_u64(lane.step_generation, &mut bytes);
        push_u64(lane.sequence_id, &mut bytes);
        push_u64(lane.sequence_position, &mut bytes);
        push_u32(lane.request_slot_index, &mut bytes);
        push_u32(lane.context_token_count, &mut bytes);
        push_u32(lane.input_token_id, &mut bytes);
        push_u32(lane.mtp_draft_token_count, &mut bytes);
        push_u32(lane.speculative_token_count, &mut bytes);
        bytes.push(lane.mtp_resolution_proposed_token_count);
        bytes.push(lane.mtp_resolution_accepted_token_count);
        bytes.extend_from_slice(&lane.mtp_resolution_path_id.to_le_bytes());
        for &token_id in &lane.speculative_draft_token_ids {
            push_u32(token_id, &mut bytes);
        }
        bytes.resize(lane_start + lane_bytes, 0);
    }
    debug_assert_eq!(bytes.len() as u32, packet.descriptor_bytes);
    Ok(bytes)
}

/// `SparkDistributedWorkHashBytes` over packet bytes.
pub fn hash_bytes(data: &[u8]) -> u64 {
    work_transaction::fingerprint_bytes(data)
}
