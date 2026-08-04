//! CUDA-resident IPC client — port of the `node/backend.c`
//! `SparkRingServiceBackend*CudaResident*` functions.
//!
//! The resident decode stage owns the GPU; the backend talks to it over a
//! Unix stream socket with the `SparkCudaResidentIpc` wire protocol
//! ([`super::wire`]). All connection state lives in [`ResidentConnection`]
//! (embedded in [`super::state::BackendCore`]); the message handlers are
//! free functions because they also touch the pending-decode arena and the
//! serving engine.
//!
//! Status discipline (mirrors the C exactly):
//! * connect: path too long → `CapacityExceeded`, connect failure →
//!   `RouteNotFound`, contract mismatch → `ModuleNotValidated` (with a
//!   stderr dump), daemon not ready → `Busy`.
//! * `ensure` collapses every connect failure to `Busy` and enforces a
//!   250 ms reconnect backoff.
//! * submit: credit exhaustion → `Busy`; a write failure releases the
//!   credit, tears the connection down, and returns `Busy`.
//! * pump: `Busy` from the reader is "no complete message" → `Ok`; any real
//!   error tears the connection down and propagates.
//! * completion with an un-releasable credit = stale completion for a
//!   torn-down connection: log and ignore (do not loop the teardown).

use std::os::unix::io::OwnedFd;

use spark_serve::serving_engine::bridge::Dispatch;
use spark_serve::serving_engine::engine::ServingEngine;
use spark_serve::serving_engine::ServingStatus;

use spark_sched::work_control::WorkControlPacket;

use super::net;
use super::state::BackendCore;
use super::wire::{
    self, CreditLedger, IpcCompletion, IpcHeader, IpcHello, IpcStats, IpcSubmitResult,
    CREDIT_DOMAIN_COMPLETION_OWNERSHIP, CREDIT_DOMAIN_EXECUTION,
    CREDIT_DOMAIN_RESIDENT_RESERVATION, CREDIT_DOMAIN_TRANSPORT_WINDOW, IPC_HEADER_BYTES,
    IPC_KIND_COMPLETION, IPC_KIND_HELLO, IPC_KIND_HELLO_ACK, IPC_KIND_SUBMIT_RESULT,
    IPC_KIND_SUBMIT_WORK, IPC_MAX_READ_PAYLOAD_BYTES, IPC_STATE_READY,
    IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT, NVME_MODE_ASYNC_SELECTED_JIT, NVME_MODE_BATCHED_COHORT_JIT,
    NVME_MODE_SYNCHRONOUS_FULL_HISTORY,
};

/// `SparkCudaResidentIpcReader` — incremental split (header, payload)
/// nonblocking reader state.
#[derive(Debug)]
struct IpcReader {
    header: [u8; IPC_HEADER_BYTES],
    header_offset: usize,
    expected_payload_bytes: usize,
    payload: [u8; IPC_MAX_READ_PAYLOAD_BYTES],
    payload_offset: usize,
}

impl IpcReader {
    fn new() -> Self {
        IpcReader {
            header: [0u8; IPC_HEADER_BYTES],
            header_offset: 0,
            expected_payload_bytes: 0,
            payload: [0u8; IPC_MAX_READ_PAYLOAD_BYTES],
            payload_offset: 0,
        }
    }

    /// `SparkCudaResidentIpcReaderReset`.
    fn reset(&mut self) {
        self.header_offset = 0;
        self.expected_payload_bytes = 0;
        self.payload_offset = 0;
    }
}

/// `SparkRingServiceBackendState`'s CUDA-resident half.
pub struct ResidentConnection {
    /// `state->cuda_resident_fd`.
    pub fd: Option<OwnedFd>,
    /// `state->cuda_resident_socket_path`.
    pub socket_path: String,
    /// `state->cuda_resident_submit_capacity`.
    pub submit_capacity: u32,
    /// `state->cuda_resident_submit_count`.
    pub submit_count: u64,
    /// `state->cuda_resident_completion_count`.
    pub completion_count: u64,
    /// `state->cuda_resident_rejection_count`.
    pub rejection_count: u64,
    /// `state->cuda_resident_retry_after_ns`.
    pub retry_after_ns: u64,
    /// `state->resident_next_sequence_number` (outgoing message sequence).
    pub next_sequence_number: u64,
    /// `state->credit_ledger`.
    pub credits: CreditLedger,
    /// Failure completions received in the decode-callback context, where
    /// the serving engine cannot be re-entered; the pump context drains
    /// them through [`pump_responses`] before reading the socket. The C
    /// re-enters the engine from the callback instead.
    pub deferred_completions: Vec<IpcCompletion>,
    reader: IpcReader,
}

impl ResidentConnection {
    pub fn new(socket_path: &str, session_id_base: u64) -> Self {
        ResidentConnection {
            fd: None,
            socket_path: socket_path.to_string(),
            submit_capacity: 0,
            submit_count: 0,
            completion_count: 0,
            rejection_count: 0,
            retry_after_ns: 0,
            next_sequence_number: session_id_base,
            credits: CreditLedger::initialize([0, 0, 0, 0]),
            deferred_completions: Vec::new(),
            reader: IpcReader::new(),
        }
    }

    pub fn is_connected(&self) -> bool {
        self.fd.is_some()
    }
}

/// `SparkRingServiceBackendTeardownCudaResident`.
pub fn teardown(core: &mut BackendCore, reason: &str) {
    if core.resident.fd.is_some() {
        eprintln!("ring_resident_disconnected reason={reason}");
    }
    core.resident.fd = None;
    core.resident.submit_capacity = 0;
    core.resident.credits.reset();
    core.resident.reader.reset();
}

/// `SparkRingServiceBackendResidentWriteMessage` (header + payload, blocking
/// full write on the connected socket).
fn write_message(core: &mut BackendCore, kind: u32, payload: &[u8]) -> Result<(), ServingStatus> {
    let fd = core.resident.fd.as_ref().ok_or(ServingStatus::InvalidArgument)?;
    let raw = net::raw_fd(fd);
    let sequence_number = core.resident.next_sequence_number;
    core.resident.next_sequence_number += 1;
    let header = IpcHeader::initialize(
        kind,
        core.rank_plan.rank_index,
        sequence_number,
        payload.len() as u32,
    );
    net::write_full(raw, &header.encode())?;
    if !payload.is_empty() {
        net::write_full(raw, payload)?;
    }
    Ok(())
}

/// One reader step: `SparkCudaResidentIpcReadHeader` + `...ReadPayload`.
/// `Err(Busy)` means "no complete message yet" (state kept); on success the
/// reader is reset and the header plus payload are returned.
fn try_read_message(core: &mut BackendCore) -> Result<(IpcHeader, Vec<u8>), ServingStatus> {
    let fd = core.resident.fd.as_ref().ok_or(ServingStatus::InvalidArgument)?;
    let raw = net::raw_fd(fd);
    let reader = &mut core.resident.reader;
    while reader.header_offset < IPC_HEADER_BYTES {
        match net::read_once(raw, &mut reader.header[reader.header_offset..]) {
            Ok(0) => return Err(ServingStatus::RouteNotFound),
            Ok(count) => reader.header_offset += count,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                return Err(ServingStatus::Busy);
            }
            Err(_) => return Err(ServingStatus::IoError),
        }
    }
    let header = IpcHeader::decode(&reader.header);
    header.validate(&reader.header, 0, IPC_MAX_READ_PAYLOAD_BYTES as u32)?;
    let expected = header.payload_bytes as usize;
    reader.expected_payload_bytes = expected;
    while reader.payload_offset < expected {
        match net::read_once(raw, &mut reader.payload[reader.payload_offset..expected]) {
            Ok(0) => return Err(ServingStatus::RouteNotFound),
            Ok(count) => reader.payload_offset += count,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                return Err(ServingStatus::Busy);
            }
            Err(_) => return Err(ServingStatus::IoError),
        }
    }
    let payload = reader.payload[..expected].to_vec();
    reader.reset();
    Ok((header, payload))
}

/// `SparkRingServiceBackendResidentReadMessage` (`timeout_ms == 0` is the
/// pure nonblocking form used by the pump).
fn read_message(
    core: &mut BackendCore,
    timeout_ms: u32,
) -> Result<(IpcHeader, Vec<u8>), ServingStatus> {
    if core.resident.fd.is_none() {
        return Err(ServingStatus::InvalidArgument);
    }
    loop {
        match try_read_message(core) {
            Err(ServingStatus::Busy) if timeout_ms != 0 => {}
            other => return other,
        }
        let raw = net::raw_fd(core.resident.fd.as_ref().expect("checked above"));
        let revents = net::poll_one(raw, net::POLLIN, timeout_ms as i32)?;
        if revents == 0 {
            return Err(ServingStatus::Busy);
        }
        if (revents & (net::POLLERR | net::POLLHUP | net::POLLNVAL)) != 0 {
            return Err(ServingStatus::RouteNotFound);
        }
    }
}

/// `SparkRingServiceBackendConnectCudaResident` — handshake and contract
/// validation. Completions for submissions charged to a torn-down connection
/// can never match on the new one, so in-flight pendings are failed
/// deterministically before the fresh credit ledger replaces the old
/// accounting (the C comment's remaining gap: executed work is discarded,
/// not replayed).
pub fn connect(
    core: &mut BackendCore,
    engine: &mut ServingEngine,
    cancel_dispatch: &mut dyn FnMut(&mut Dispatch) -> Result<(), ServingStatus>,
) -> Result<(), ServingStatus> {
    core.pendings.fail_inflight(ServingStatus::RouteNotFound, engine, cancel_dispatch);
    let socket_path = core.resident.socket_path.clone();
    if socket_path.is_empty() {
        return Err(ServingStatus::InvalidArgument);
    }
    let expected_moe_backend_kind = core.rank_plan.quantization_mode.expected_moe_backend_kind();
    let fd = net::unix_connect(&socket_path)?;
    core.resident.fd = Some(fd);

    let hello = IpcHello {
        rank_index: core.rank_plan.rank_index,
        rank_count: core.config.geometry.stage_count,
        expected_cuda_generation: 0,
        control_generation: core.session_id_base,
        process_id: u64::from(std::process::id()),
    };
    let mut status = write_message(core, IPC_KIND_HELLO, &hello.encode());
    if status.is_ok() {
        status = connect_read_handshake(core, expected_moe_backend_kind);
    }
    if let Err(error) = status {
        core.resident.fd = None;
        return Err(error);
    }
    core.resident.reader.reset();
    Ok(())
}

/// The read half of [`connect`]: HELLO_ACK header, stats payload, READY
/// state, the full contract check, then the credit ledger install.
fn connect_read_handshake(
    core: &mut BackendCore,
    expected_moe_backend_kind: u32,
) -> Result<(), ServingStatus> {
    let raw = net::raw_fd(core.resident.fd.as_ref().expect("connect set fd"));
    let mut header_bytes = [0u8; IPC_HEADER_BYTES];
    net::read_bounded(raw, &mut header_bytes, 5000)?;
    let header = IpcHeader::decode(&header_bytes);
    header.validate(&header_bytes, IPC_KIND_HELLO_ACK, wire::IPC_STATS_BYTES as u32)?;
    if header.payload_bytes != wire::IPC_STATS_BYTES as u32 {
        return Err(ServingStatus::AbiMismatch);
    }
    let mut stats_bytes = [0u8; wire::IPC_STATS_BYTES];
    net::read_bounded(raw, &mut stats_bytes, 5000)?;
    let stats = IpcStats::decode(&stats_bytes)?;
    if stats.state != IPC_STATE_READY {
        return Err(ServingStatus::Busy);
    }
    let max_lane_count = core.work_control.max_lane_count;
    let nvme_mode_valid = stats.kv_nvme_enabled == 0
        || stats.kv_nvme_mode == NVME_MODE_SYNCHRONOUS_FULL_HISTORY
        || stats.kv_nvme_mode == NVME_MODE_BATCHED_COHORT_JIT
        || stats.kv_nvme_mode == NVME_MODE_ASYNC_SELECTED_JIT;
    let wide_ring_nvme_required = core.rank_plan.logical_lane_capacity >= max_lane_count
        && (stats.kv_nvme_enabled == 0
            || (stats.kv_nvme_mode != NVME_MODE_BATCHED_COHORT_JIT
                && stats.kv_nvme_mode != NVME_MODE_ASYNC_SELECTED_JIT)
            || stats.kv_resident_bytes_per_token == 0
            || stats.kv_resident_pool_bytes == 0
            || stats.kv_nvme_capacity_bytes == 0
            || stats.kv_nvme_batch_block_capacity == 0);
    let contract_ok = stats.logical_lane_capacity == core.rank_plan.logical_lane_capacity
        && stats.execution_row_capacity == core.rank_plan.execution_row_capacity
        && stats.kv_physical_block_capacity != 0
        && stats.kv_physical_block_capacity <= stats.kv_logical_block_capacity
        && stats.kv_logical_block_capacity == core.kv_logical_block_capacity
        && stats.work_queue_capacity != 0
        && stats.work_queue_depth <= stats.work_queue_capacity
        && stats.model_quantization_mode == core.rank_plan.quantization_mode.code()
        && stats.moe_backend_kind == expected_moe_backend_kind
        && stats.moe_bound_layer_count != 0
        && stats.moe_bound_layer_count == stats.moe_expected_layer_count
        && core
            .rank_plan
            .quantization_mode
            .validate_fp8_plan_counts(
                stats.fp8_scaled_gemm_bound_plan_count,
                stats.fp8_scaled_gemm_expected_plan_count,
            )
            .is_ok()
        && nvme_mode_valid
        && !wide_ring_nvme_required;
    if !contract_ok {
        eprintln!(
            "ring_resident_contract_mismatch logical={}/{} execution={}/{} \
             kv_blocks={} logical_blocks={}/{} quantization={}/{} moe_backend={} \
             moe_layers={}/{} fp8_scaled_gemm={}/{} nvme={} nvme_mode={} blocker={}",
            stats.logical_lane_capacity,
            core.rank_plan.logical_lane_capacity,
            stats.execution_row_capacity,
            core.rank_plan.execution_row_capacity,
            stats.kv_physical_block_capacity,
            stats.kv_logical_block_capacity,
            core.kv_logical_block_capacity,
            stats.model_quantization_mode,
            core.rank_plan.quantization_mode.code(),
            stats.moe_backend_kind,
            stats.moe_bound_layer_count,
            stats.moe_expected_layer_count,
            stats.fp8_scaled_gemm_bound_plan_count,
            stats.fp8_scaled_gemm_expected_plan_count,
            stats.kv_nvme_enabled,
            stats.kv_nvme_mode,
            stats.blocker,
        );
        return Err(ServingStatus::ModuleNotValidated);
    }
    core.kv_physical_block_capacity = stats.kv_physical_block_capacity;
    core.resident.submit_capacity = stats.work_queue_capacity;
    let pending_capacity = core.pendings.pending_capacity() as u32;
    let mut capacities = [0u32; wire::CREDIT_DOMAIN_COUNT];
    capacities[CREDIT_DOMAIN_TRANSPORT_WINDOW] = 1;
    capacities[CREDIT_DOMAIN_RESIDENT_RESERVATION] = stats.work_queue_capacity;
    capacities[CREDIT_DOMAIN_EXECUTION] = pending_capacity;
    capacities[CREDIT_DOMAIN_COMPLETION_OWNERSHIP] = pending_capacity;
    core.resident.credits = CreditLedger::initialize(capacities);
    if stats.work_queue_depth != 0 {
        core.resident
            .credits
            .acquire(CREDIT_DOMAIN_RESIDENT_RESERVATION, stats.work_queue_depth)?;
    }
    Ok(())
}

/// `SparkRingServiceBackendEnsureCudaResident` — 250 ms reconnect backoff;
/// every connect failure collapses to `Busy`.
pub fn ensure(
    core: &mut BackendCore,
    engine: &mut ServingEngine,
    cancel_dispatch: &mut dyn FnMut(&mut Dispatch) -> Result<(), ServingStatus>,
) -> Result<(), ServingStatus> {
    if core.resident.socket_path.is_empty() {
        return Err(ServingStatus::InvalidArgument);
    }
    if core.resident.fd.is_some() {
        return Ok(());
    }
    let now_ns = net::monotonic_ns();
    if now_ns < core.resident.retry_after_ns {
        return Err(ServingStatus::Busy);
    }
    core.resident.retry_after_ns = now_ns + 250_000_000;
    match connect(core, engine, cancel_dispatch) {
        Ok(()) => {
            eprintln!("ring_resident_connected");
            Ok(())
        }
        Err(status) => {
            eprintln!("ring_resident_connect_retry status={}", status.code());
            Err(ServingStatus::Busy)
        }
    }
}

/// `SparkRingServiceBackendRequireResidentSubmitCredits`.
pub fn require_credits(
    core: &mut BackendCore,
    engine: &mut ServingEngine,
    cancel_dispatch: &mut dyn FnMut(&mut Dispatch) -> Result<(), ServingStatus>,
    required_credit_count: u32,
) -> Result<(), ServingStatus> {
    if required_credit_count == 0 {
        return Err(ServingStatus::InvalidArgument);
    }
    ensure(core, engine, cancel_dispatch)?;
    pump_responses(core, engine, cancel_dispatch)?;
    if core.resident.credits.available(CREDIT_DOMAIN_RESIDENT_RESERVATION) >= required_credit_count
    {
        Ok(())
    } else {
        Err(ServingStatus::Busy)
    }
}

/// `SparkRingServiceBackendSubmitResidentMessage` — acquire a reservation
/// credit (`CapacityExceeded` → `Busy`), write, and on write failure
/// release + teardown + `Busy`.
pub fn submit_message(
    core: &mut BackendCore,
    kind: u32,
    payload: &[u8],
) -> Result<(), ServingStatus> {
    if core.resident.fd.is_none() {
        return Err(ServingStatus::Busy);
    }
    if let Err(status) = core.resident.credits.acquire(CREDIT_DOMAIN_RESIDENT_RESERVATION, 1) {
        return Err(if status == ServingStatus::CapacityExceeded {
            ServingStatus::Busy
        } else {
            status
        });
    }
    if let Err(_status) = write_message(core, kind, payload) {
        let _ = core.resident.credits.release(CREDIT_DOMAIN_RESIDENT_RESERVATION, 1);
        teardown(core, "submit_message_write");
        return Err(ServingStatus::Busy);
    }
    core.resident.submit_count += 1;
    Ok(())
}

/// `SparkCudaResidentIpcInitializeSubmitWork`: the SUBMIT_WORK payload is an
/// 8-byte prefix (`descriptor_bytes`, `submit_flags`) followed by the
/// canonical packet serialization.
pub fn submit_work_payload(
    work_control: &spark_sched::work_control::WorkControlConfig,
    packet: &WorkControlPacket,
    submit_flags: u32,
) -> Result<Vec<u8>, ServingStatus> {
    let packet_bytes = wire::serialize_packet(work_control, packet)?;
    let descriptor_bytes = wire::submit_work_bytes(packet_bytes.len() as u32);
    let mut payload = Vec::with_capacity(descriptor_bytes as usize);
    payload.extend_from_slice(&descriptor_bytes.to_le_bytes());
    payload.extend_from_slice(&submit_flags.to_le_bytes());
    payload.extend_from_slice(&packet_bytes);
    Ok(payload)
}

/// `SparkRingServiceBackendSubmitWorkToResident`.
pub fn submit_work(
    core: &mut BackendCore,
    packet: &WorkControlPacket,
    submit_flags: u32,
) -> Result<(), ServingStatus> {
    let payload = submit_work_payload(&core.work_control, packet, submit_flags)?;
    submit_message(core, IPC_KIND_SUBMIT_WORK, &payload)
}

/// `SparkRingServiceBackendResidentAwaitSubmitResult` (180 s header timeout;
/// completions are handled inline while waiting).
pub fn await_submit_result(
    core: &mut BackendCore,
    engine: &mut ServingEngine,
    cancel_dispatch: &mut dyn FnMut(&mut Dispatch) -> Result<(), ServingStatus>,
    stats_out: Option<&mut IpcStats>,
) -> Result<(), ServingStatus> {
    let mut stats_out = stats_out;
    loop {
        // The reader validates the header (magic/ABI/kind/payload bound) as
        // part of the read, so the C's second `ValidateHeader` here is
        // already covered.
        let (header, payload) = match read_message(core, 180_000) {
            Ok(message) => message,
            Err(status) => {
                eprintln!("ring_resident_await_failed step=header status={}", status.code());
                teardown(core, "await_header");
                return Err(ServingStatus::Busy);
            }
        };
        if header.kind == IPC_KIND_SUBMIT_RESULT {
            let result = handle_submit_result(core, &header, &payload)?;
            if let Some(out) = stats_out.as_deref_mut() {
                *out = result.stats.clone();
            }
            return result.into_status();
        }
        if header.kind == IPC_KIND_COMPLETION {
            handle_completion(core, &header, &payload, engine, cancel_dispatch)?;
            continue;
        }
        teardown(core, "await_unknown_kind");
        return Err(ServingStatus::Busy);
    }
}

/// `SparkRingServiceBackendSubmitReleaseToResident` — fire-and-forget by
/// default (ordering safety is the resident's FIFO: the release entered the
/// queue before any dispatch that could reuse its blocks). The synchronous
/// await remains for a sparkdev bisecting a reuse suspicion
/// (`SPARKPIPE_RELEASE_SYNC_AWAIT`).
pub fn submit_release(
    core: &mut BackendCore,
    engine: &mut ServingEngine,
    cancel_dispatch: &mut dyn FnMut(&mut Dispatch) -> Result<(), ServingStatus>,
    packet: &WorkControlPacket,
) -> Result<(), ServingStatus> {
    require_credits(core, engine, cancel_dispatch, 1)?;
    submit_work(core, packet, IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT)?;
    if std::env::var_os("SPARKPIPE_RELEASE_SYNC_AWAIT").is_none() {
        return Ok(());
    }
    await_submit_result(core, engine, cancel_dispatch, None)
}

/// `SparkRingServiceBackendPumpCudaResidentResponses` (up to 64 messages).
/// Deferred callback-context failure completions are routed first (their
/// credits were already released at capture time).
pub fn pump_responses(
    core: &mut BackendCore,
    engine: &mut ServingEngine,
    cancel_dispatch: &mut dyn FnMut(&mut Dispatch) -> Result<(), ServingStatus>,
) -> Result<(), ServingStatus> {
    if core.resident.fd.is_none() && core.resident.deferred_completions.is_empty() {
        return Err(ServingStatus::Busy);
    }
    if !core.resident.deferred_completions.is_empty() {
        let deferred = std::mem::take(&mut core.resident.deferred_completions);
        for completion in &deferred {
            route_failure_completion(core, completion, engine, cancel_dispatch)?;
        }
    }
    if core.resident.fd.is_none() {
        return Ok(());
    }
    let mut status = Ok(());
    for _message_count in 0..64 {
        let (header, payload) = match read_message(core, 0) {
            Ok(message) => message,
            Err(ServingStatus::Busy) => return Ok(()),
            Err(error) => {
                status = Err(error);
                break;
            }
        };
        status = match header.kind {
            IPC_KIND_COMPLETION => {
                handle_completion(core, &header, &payload, engine, cancel_dispatch)
            }
            IPC_KIND_SUBMIT_RESULT => handle_submit_result(core, &header, &payload)
                .and_then(|result| result.into_status()),
            _ => Err(ServingStatus::AbiMismatch),
        };
        if status.is_err() {
            break;
        }
    }
    if status.is_ok() {
        return Ok(());
    }
    teardown(core, "resident_response");
    status
}

/// `SparkRingServiceBackendHandleResidentCompletion`.
pub fn handle_completion(
    core: &mut BackendCore,
    header: &IpcHeader,
    payload: &[u8],
    engine: &mut ServingEngine,
    cancel_dispatch: &mut dyn FnMut(&mut Dispatch) -> Result<(), ServingStatus>,
) -> Result<(), ServingStatus> {
    if header.payload_bytes != wire::IPC_COMPLETION_BYTES as u32 {
        return Err(ServingStatus::AbiMismatch);
    }
    let completion = IpcCompletion::decode(payload)?;
    core.resident.completion_count += 1;
    if let Err(credit_status) = core.resident.credits.release(CREDIT_DOMAIN_RESIDENT_RESERVATION, 1)
    {
        // Stale completion for work charged to a torn-down connection: its
        // pending was failed on reconnect, so the fresh ledger has no credit
        // to release. Log and ignore rather than tearing the connection
        // down in a loop.
        eprintln!(
            "ring_resident_stale_completion ledger_status={} request={} sequence={}",
            credit_status.code(),
            completion.request_id,
            completion.sequence_id
        );
        return Ok(());
    }
    let Some(completion_status) = wire::status_from_u32(completion.status) else {
        // The C casts the raw u32 to SparkStatus; here an unrepresentable
        // code is a wire-protocol violation.
        return Err(ServingStatus::AbiMismatch);
    };
    if completion_status != ServingStatus::Ok {
        return route_failure_completion(core, &completion, engine, cancel_dispatch);
    }
    Ok(())
}

/// Decode-callback-context credit requirement: the engine is mid-`pump` and
/// cannot be re-entered, so this variant never connects (a missing
/// connection reports `Busy`; the engine then retries the dispatch, and the
/// backend pump reconnects) and defers failure completions to the pump
/// context ([`ResidentConnection::deferred_completions`]). Success-path
/// completions only release a credit and are handled inline, as in C.
pub fn require_credits_for_callback(
    core: &mut BackendCore,
    required_credit_count: u32,
) -> Result<(), ServingStatus> {
    if required_credit_count == 0 {
        return Err(ServingStatus::InvalidArgument);
    }
    if core.resident.fd.is_none() {
        return Err(ServingStatus::Busy);
    }
    pump_responses_for_callback(core)?;
    if core.resident.credits.available(CREDIT_DOMAIN_RESIDENT_RESERVATION) >= required_credit_count
    {
        Ok(())
    } else {
        Err(ServingStatus::Busy)
    }
}

/// The callback-context companion of [`pump_responses`]: submit results are
/// handled fully (they never touch the engine); success completions release
/// their credit; failure completions are deferred.
pub fn pump_responses_for_callback(core: &mut BackendCore) -> Result<(), ServingStatus> {
    if core.resident.fd.is_none() {
        return Err(ServingStatus::Busy);
    }
    let mut status = Ok(());
    for _message_count in 0..64 {
        let (header, payload) = match read_message(core, 0) {
            Ok(message) => message,
            Err(ServingStatus::Busy) => return Ok(()),
            Err(error) => {
                status = Err(error);
                break;
            }
        };
        status = match header.kind {
            IPC_KIND_COMPLETION => handle_completion_for_callback(core, &header, &payload),
            IPC_KIND_SUBMIT_RESULT => handle_submit_result(core, &header, &payload)
                .and_then(|result| result.into_status()),
            _ => Err(ServingStatus::AbiMismatch),
        };
        if status.is_err() {
            break;
        }
    }
    if status.is_ok() {
        return Ok(());
    }
    teardown(core, "resident_response");
    status
}

/// Callback-context completion handling: validate + credit release inline
/// (including the stale-completion log-and-ignore); failure routing is
/// deferred to the pump context.
fn handle_completion_for_callback(
    core: &mut BackendCore,
    header: &IpcHeader,
    payload: &[u8],
) -> Result<(), ServingStatus> {
    if header.payload_bytes != wire::IPC_COMPLETION_BYTES as u32 {
        return Err(ServingStatus::AbiMismatch);
    }
    let completion = IpcCompletion::decode(payload)?;
    core.resident.completion_count += 1;
    if let Err(credit_status) = core.resident.credits.release(CREDIT_DOMAIN_RESIDENT_RESERVATION, 1)
    {
        eprintln!(
            "ring_resident_stale_completion ledger_status={} request={} sequence={}",
            credit_status.code(),
            completion.request_id,
            completion.sequence_id
        );
        return Ok(());
    }
    let Some(completion_status) = wire::status_from_u32(completion.status) else {
        return Err(ServingStatus::AbiMismatch);
    };
    if completion_status != ServingStatus::Ok {
        core.resident.deferred_completions.push(completion);
    }
    Ok(())
}

/// The failure-routing half of [`handle_completion`], shared by the pump
/// handler and the deferred-completion drain.
fn route_failure_completion(
    core: &mut BackendCore,
    completion: &IpcCompletion,
    engine: &mut ServingEngine,
    cancel_dispatch: &mut dyn FnMut(&mut Dispatch) -> Result<(), ServingStatus>,
) -> Result<(), ServingStatus> {
    let Some(failure_status) = wire::status_from_u32(completion.status) else {
        return Err(ServingStatus::AbiMismatch);
    };
    eprintln!(
        "ring_resident_work_failed status={} request={} sequence={}",
        failure_status.code(),
        completion.request_id,
        completion.sequence_id
    );
    if completion.request_id == 0 {
        return Err(failure_status);
    }
    if let Some(index) = core.pendings.find_for_request(completion.request_id) {
        return core.pendings.fail_pending_decode(index, failure_status, engine, cancel_dispatch);
    }
    match engine.fail_request_by_request_id(completion.request_id, failure_status) {
        Ok(()) => Ok(()),
        Err(ServingStatus::NotFound) => Err(failure_status),
        Err(route_status) => Err(route_status),
    }
}

/// `SparkRingServiceBackendHandleResidentSubmitResult` (credit release
/// propagates; rejections set the blocker and surface the result status).
pub fn handle_submit_result(
    core: &mut BackendCore,
    header: &IpcHeader,
    payload: &[u8],
) -> Result<SubmitResultOutcome, ServingStatus> {
    if header.payload_bytes != wire::IPC_SUBMIT_RESULT_BYTES as u32 {
        return Err(ServingStatus::AbiMismatch);
    }
    let result = IpcSubmitResult::decode(payload)?;
    core.resident.credits.release(CREDIT_DOMAIN_RESIDENT_RESERVATION, 1)?;
    let Some(status) = wire::status_from_u32(result.status) else {
        return Err(ServingStatus::AbiMismatch);
    };
    if status == ServingStatus::Ok {
        return Ok(SubmitResultOutcome { status, stats: result.stats });
    }
    core.resident.rejection_count += 1;
    let blocker = format!(
        "resident submission rejected status={} blocker={}",
        result.status, result.stats.blocker
    );
    core.set_blocker(&blocker);
    eprintln!(
        "ring_resident_submit_rejected status={} blocker={}",
        result.status, result.stats.blocker
    );
    Ok(SubmitResultOutcome { status, stats: result.stats })
}

/// The handled submit result: its status plus the reported daemon stats.
pub struct SubmitResultOutcome {
    pub status: ServingStatus,
    pub stats: IpcStats,
}

impl SubmitResultOutcome {
    fn into_status(self) -> Result<(), ServingStatus> {
        if self.status == ServingStatus::Ok {
            Ok(())
        } else {
            Err(self.status)
        }
    }
}
