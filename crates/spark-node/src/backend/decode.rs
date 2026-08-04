//! Decode dispatch path — port of the `node/backend.c`
//! `SparkRingServiceBackend*Decode*` functions.
//!
//! Two execution paths, exactly as in C:
//!
//! * **CUDA-resident** (`cuda_resident_attached`): chunk planning, credit
//!   check, pending registration, work-output forward, resident IPC submit
//!   (single SUBMIT_DECODE, or chunked SUBMIT_WORK for MTP verify), then the
//!   early-final-event drain decides `Ok` (completed inline) / `Pending`
//!   (the backend pump completes later).
//! * **Local-CUDA builder** (`Rank0NodeContext` seam): the builder decodes
//!   synchronously into `decode_result`, then the same register/forward/
//!   drain choreography.
//!
//! Callback-context discipline (deviation from the flat C, forced by the
//! Rust engine owning `&mut self` while the decode callback runs): the C's
//! in-callback `RequireResidentSubmitCredits` does ensure + response pump,
//! which re-enters the pending/engine failure paths. Here the backend pump
//! performs ensure + resident response pump every cycle *before* the
//! service pump schedules decodes, so the callback only *checks* credit
//! availability; a shortfall returns `Busy`, which the engine maps onto
//! `retry_decode_dispatch` — the same net effect one pump-cycle earlier.
//!
//! Resident decode payload stamping (deviation, flagged): the C
//! `SparkRingServiceBackendBuildDecodeResidentPayload` memsets the message
//! and never stamps `request_generation` / `step_generation` /
//! `step_chunk_*` / `transaction_phase`, yet its own
//! `SparkCudaResidentIpcValidateSubmitDecode` (and the resident daemon's
//! packet conversion) requires them non-zero — the C path cannot pass its
//! validator as written. The C test suite's `SparkTestFinalizeDecodeMessage`
//! shows the intended stamping; this port stamps at build time with
//! production values (lane `request_generation` = the request handle, as the
//! work-packet lane stamping does; `step_generation` derived nonzero exactly
//! as the C test finalizer derives it).

use spark_sched::work_control::{self, WorkControlPacket};
use spark_serve::request_api::dispatch::{
    DISPATCH_KIND_DECODE_BATCH, DISPATCH_KIND_PREFILL, DISPATCH_KIND_PREFILL_BATCH,
    DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH,
};
use spark_serve::serving_engine::bridge::DispatchKind;
use spark_serve::serving_engine::engine::{ServingDecodeDispatch, ServingDecodeResult};
use spark_serve::serving_engine::ServingStatus;

use super::adapt::{DispatchExtras, RequestApiAdapter};
use super::pending::CallbackDrain;
use super::resident;
use super::state::BackendCore;
use super::wire;

/// `SPARK_SERVING_MAX_DECODE_TOKENS_PER_LANE`.
const DECODE_TOKEN_STRIDE: u32 =
    spark_serve::serving_engine::bridge::MAX_DECODE_TOKENS_PER_LANE as u32;

/// `SPARK_REQUEST_API_DISPATCH_KIND_*` code for a bridge kind.
pub fn dispatch_kind_code(kind: DispatchKind) -> u32 {
    match kind {
        DispatchKind::Prefill => DISPATCH_KIND_PREFILL,
        DispatchKind::DecodeBatch => DISPATCH_KIND_DECODE_BATCH,
        DispatchKind::PrefillBatch => DISPATCH_KIND_PREFILL_BATCH,
        DispatchKind::SpeculativeVerifyBatch => DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH,
        DispatchKind::None => 0,
    }
}

/// `SparkRingServiceBackendDecodeIsMtpVerify`.
pub fn is_mtp_verify(decode_dispatch: &ServingDecodeDispatch) -> bool {
    (decode_dispatch.request_dispatch.flags
        & work_control::REQUEST_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY)
        != 0
}

/// The engine's borrowed decode dispatch plus the staged full-dispatch
/// extras, in work control's owned input shape
/// (`SparkServingDecodeDispatch` + `SparkRequestApiDispatch` in C).
pub fn to_work_control_decode_dispatch(
    decode_dispatch: &ServingDecodeDispatch,
    extras: &DispatchExtras,
) -> work_control::ServingDecodeDispatch {
    let dto = decode_dispatch.request_dispatch;
    work_control::ServingDecodeDispatch {
        dispatch_kind: dispatch_kind_code(decode_dispatch.dispatch_kind),
        request_count: decode_dispatch.request_count,
        active_sequence_count: decode_dispatch.active_sequence_count,
        request_dispatch: work_control::RequestApiDispatch {
            kind: dispatch_kind_code(dto.kind),
            flags: dto.flags,
            request_count: dto.request_count,
            highest_priority: extras.highest_priority,
            request_handles: dto.request_handles.clone(),
            request_ids: dto.request_ids.clone(),
            sequence_ids: extras.sequence_ids.clone(),
            prefill_decision_batch_bucket: extras.prefill_decision_batch_bucket,
            prefill_batch_decision_batch_bucket: extras.prefill_batch_decision_batch_bucket,
            decode_batch_decision_batch_bucket: extras.decode_batch_decision_batch_bucket,
            speculative_verifier_token_count: dto.speculative_verifier_token_count,
            mtp_draft_token_budget: dto.mtp_draft_token_budget,
        },
        kv_block_table_view: Some(work_control::DispatchKvBlockTableView {
            block_token_count: decode_dispatch.kv_block_table_view.block_token_count,
            lane_count: decode_dispatch.kv_block_table_view.lane_count,
            lane_stride: decode_dispatch.kv_block_table_view.lane_stride,
            lane_capacity: decode_dispatch.kv_block_table_view.lane_capacity,
        }),
        decode_view: work_control::DecodeDispatchView {
            lane_count: decode_dispatch.decode_view.lane_count,
            lanes: decode_dispatch
                .decode_view
                .lanes
                .iter()
                .map(|lane| work_control::DecodeDispatchLaneView {
                    request_index: lane.request_index,
                    sequence_position: lane.sequence_position,
                    context_token_count: lane.context_token_count,
                    request_slot_index: lane.request_slot_index,
                    request_id: lane.request_id,
                    sequence_id: lane.sequence_id,
                    request_handle: lane.request_handle,
                    mtp_resolution_base_position: lane.mtp_resolution_base_position,
                    mtp_resolution_proposed_token_count: lane
                        .mtp_resolution_proposed_token_count,
                    mtp_resolution_accepted_token_count: lane
                        .mtp_resolution_accepted_token_count,
                    mtp_resolution_committed_token_count: lane
                        .mtp_resolution_committed_token_count,
                    mtp_resolution_path_id: lane.mtp_resolution_path_id,
                })
                .collect(),
        },
        input_token_ids: decode_dispatch.input_token_ids.to_vec(),
        speculative_token_count: decode_dispatch.speculative_token_count,
        speculative_draft_token_ids: decode_dispatch
            .speculative_draft_token_ids
            .chunks(
                spark_serve::serving_engine::bridge::MAX_SPECULATIVE_TOKENS,
            )
            .map(|lane_row| lane_row.to_vec())
            .collect(),
    }
}

/// `SparkRingServiceBackendPlanDecodeChunks` — returns
/// `(maximum_lanes_per_chunk, chunk_count)`.
pub fn plan_decode_chunks(
    core: &BackendCore,
    decode_dispatch: &ServingDecodeDispatch,
) -> Result<(u32, u32), ServingStatus> {
    if decode_dispatch.request_count == 0 {
        return Err(ServingStatus::InvalidArgument);
    }
    let request_flags = decode_dispatch.request_dispatch.flags;
    let mut rows_per_lane = 1u32;
    if (request_flags
        & (work_control::REQUEST_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY
            | work_control::REQUEST_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY))
        != 0
    {
        if decode_dispatch.speculative_token_count == 0
            || decode_dispatch.speculative_token_count == u32::MAX
        {
            return Err(ServingStatus::InvalidArgument);
        }
        rows_per_lane = if (request_flags
            & work_control::REQUEST_DISPATCH_FLAG_MTP_TREE_VERIFY)
            != 0
        {
            decode_dispatch.request_dispatch.speculative_verifier_token_count
        } else {
            decode_dispatch.speculative_token_count + 1
        };
    }
    work_control::plan_execution_chunks(
        &core.work_control,
        decode_dispatch.request_count,
        rows_per_lane,
        core.rank_plan.execution_row_capacity,
    )
    .map_err(|_| ServingStatus::InvalidArgument)
}

/// `SparkRingServiceBackendForwardDecodeWork`.
pub fn forward_decode_work(
    core: &mut BackendCore,
    decode_dispatch: &ServingDecodeDispatch,
    extras: &DispatchExtras,
    pending_index: usize,
) -> Result<(), ServingStatus> {
    let (maximum_lanes_per_chunk, chunk_count) = plan_decode_chunks(core, decode_dispatch)?;
    if chunk_count as usize
        > core.work_output.queue_capacity() - core.work_output.queue_count()
    {
        return Err(ServingStatus::Busy);
    }
    let wc_dispatch = to_work_control_decode_dispatch(decode_dispatch, extras);
    let mut lane_offset = 0u32;
    for chunk_index in 0..chunk_count {
        let mut lane_count = decode_dispatch.request_count - lane_offset;
        if lane_count > maximum_lanes_per_chunk {
            lane_count = maximum_lanes_per_chunk;
        }
        let mut packet = work_control::build_decode_packet_range(
            &core.work_control,
            &wc_dispatch,
            lane_offset,
            lane_count,
            0,
        )
        .map_err(|_| ServingStatus::InvalidArgument)?;
        core.work_output
            .stamp_work_packet_chunk(&mut packet, chunk_index, chunk_count)?;
        core.work_output.validate_packet(&packet)?;
        core.pendings
            .record_decode_chunk(pending_index, lane_offset, &packet)?;
        core.work_output.enqueue_work_packet(&packet)?;
        lane_offset += lane_count;
    }
    if lane_offset != decode_dispatch.request_count {
        return Err(ServingStatus::InternalError);
    }
    match core.work_output.pump_work_output() {
        Ok(()) | Err(ServingStatus::Busy) => Ok(()),
        Err(status) => Err(status),
    }
}

// ---------------------------------------------------------------------------
// Resident SUBMIT_DECODE payload
// ---------------------------------------------------------------------------

/// Decode lane wire size (`SparkCudaResidentIpcDecodeLane`, 96 bytes).
const DECODE_LANE_BYTES: usize = 96;
/// Header through the transaction fields (11 u32 + pad + 3 u64 + 4 u32).
const DECODE_MESSAGE_HEADER_BYTES: usize = 88;

/// `SparkCudaResidentIpcValidateSubmitDecode` over the serialized message.
/// `maximum_lane_count` is the dispatch's lane count (the C call site's
/// third argument); `max_speculative_token_count` /
/// `max_lane_block_count` come from the work-control configuration
/// (`SPARK_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT` and
/// `SPARK_CUDA_RESIDENT_IPC_MAX_LANE_BLOCKS`).
pub fn validate_submit_decode_payload(
    message: &[u8],
    maximum_lane_count: u32,
    max_speculative_token_count: u32,
    max_lane_block_count: u32,
) -> Result<(), ServingStatus> {
    if message.len() < wire::IPC_SUBMIT_DECODE_HEADER_BYTES || maximum_lane_count == 0 {
        return Err(ServingStatus::InvalidArgument);
    }
    if wire::read_u32(message, 0) != wire::IPC_SUBMIT_DECODE_HEADER_BYTES as u32 {
        return Err(ServingStatus::AbiMismatch);
    }
    let control_generation = wire::read_u64(message, 48);
    let request_generation = wire::read_u64(message, 56);
    let step_generation = wire::read_u64(message, 64);
    let step_chunk_index = wire::read_u32(message, 72);
    let step_chunk_count = wire::read_u32(message, 76);
    let transaction_phase = wire::read_u32(message, 80);
    let reserved_transaction = wire::read_u32(message, 84);
    let lane_count = wire::read_u32(message, 16);
    let active_sequence_count = wire::read_u32(message, 20);
    let execution_batch_bucket = wire::read_u32(message, 24);
    let speculative_token_count = wire::read_u32(message, 28);
    let kv_block_token_count = wire::read_u32(message, 32);
    let kv_block_index_count = wire::read_u32(message, 36);
    let resident_flags = wire::read_u32(message, 40);
    let dispatch_kind = wire::read_u32(message, 12);
    if message.len() as u64
        != wire::IPC_SUBMIT_DECODE_HEADER_BYTES as u64 + u64::from(kv_block_index_count) * 4
        || control_generation == 0
        || request_generation == 0
        || step_generation == 0
        || step_chunk_count == 0
        || step_chunk_index >= step_chunk_count
        || !work_control::work_transaction::phase_is_valid(transaction_phase)
        || reserved_transaction != 0
        || lane_count == 0
        || lane_count > maximum_lane_count
        || active_sequence_count != lane_count
        || !work_control::batch_bucket_is_supported(execution_batch_bucket)
        || lane_count > execution_batch_bucket
        || kv_block_token_count == 0
    {
        return Err(ServingStatus::InvalidArgument);
    }
    if (resident_flags & !wire::IPC_SUBMIT_KNOWN_FLAGS) != 0 {
        return Err(ServingStatus::InvalidArgument);
    }
    let internal_kv_directory =
        (resident_flags & wire::IPC_SUBMIT_FLAG_INTERNAL_KV_DIRECTORY) != 0;
    if internal_kv_directory != (kv_block_index_count == 0) {
        return Err(ServingStatus::InvalidArgument);
    }
    if (dispatch_kind != DISPATCH_KIND_DECODE_BATCH
        && dispatch_kind != DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
        || speculative_token_count > max_speculative_token_count
        || (dispatch_kind == DISPATCH_KIND_DECODE_BATCH && speculative_token_count != 0)
        || (dispatch_kind == DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH
            && speculative_token_count == 0)
    {
        return Err(ServingStatus::InvalidArgument);
    }
    let expected_phase = if dispatch_kind == DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH {
        work_control::work_transaction::PHASE_VERIFY
    } else {
        work_control::work_transaction::PHASE_DECODE
    };
    if transaction_phase != expected_phase {
        return Err(ServingStatus::InvalidArgument);
    }
    let request_flags = wire::read_u32(message, 8);
    let expected_mtp_budget = work_control::select_mtp_draft_budget(
        dispatch_kind,
        request_flags,
        wire::read_u32(message, DECODE_MESSAGE_HEADER_BYTES + 44),
    )
    .map_err(|_| ServingStatus::InvalidArgument)?;
    let mut expected_block_offset = 0u32;
    for lane_index in 0..lane_count as usize {
        let lane = &message[DECODE_MESSAGE_HEADER_BYTES + lane_index * DECODE_LANE_BYTES..];
        let request_id = wire::read_u64(lane, 0);
        let request_generation = wire::read_u64(lane, 8);
        let sequence_id = wire::read_u64(lane, 16);
        let request_slot_index = wire::read_u32(lane, 32);
        let context_token_count = wire::read_u32(lane, 36);
        let mtp_draft_token_budget = wire::read_u32(lane, 44);
        let lane_speculative_token_count = wire::read_u32(lane, 48);
        let proposed = lane[52];
        let accepted = lane[53];
        let path_id = wire::read_u16(lane, 54);
        let kv_block_offset = wire::read_u32(lane, 56);
        let kv_block_count = wire::read_u32(lane, 60);
        if request_id == 0
            || request_generation == 0
            || sequence_id == 0
            || request_slot_index == u32::MAX
            || context_token_count == 0
            || mtp_draft_token_budget != expected_mtp_budget
            || lane_speculative_token_count != speculative_token_count
            || (proposed == 0
                && (accepted != 0
                    || u32::from(path_id) != work_control::mtp_tree::RESOLUTION_NONE))
            || u32::from(proposed) > max_speculative_token_count
            || !work_control::mtp_tree::resolution_is_valid(
                u32::from(proposed),
                u32::from(accepted),
                u32::from(path_id),
            )
            || kv_block_offset != expected_block_offset
            || (internal_kv_directory && kv_block_count != 0)
            || (!internal_kv_directory && kv_block_count == 0)
            || kv_block_count > max_lane_block_count
            || kv_block_count > kv_block_index_count - expected_block_offset
        {
            return Err(ServingStatus::InvalidArgument);
        }
        expected_block_offset += kv_block_count;
    }
    if expected_block_offset != kv_block_index_count
        || request_generation != wire::read_u64(message, DECODE_MESSAGE_HEADER_BYTES + 8)
    {
        return Err(ServingStatus::InvalidArgument);
    }
    Ok(())
}

/// `SparkRingServiceBackendBuildDecodeResidentPayload` (+ the stamping the
/// C validator requires; see module docs). Returns the serialized
/// SUBMIT_DECODE payload.
pub fn build_decode_resident_payload(
    core: &BackendCore,
    decode_dispatch: &ServingDecodeDispatch,
    extras: &DispatchExtras,
) -> Result<Vec<u8>, ServingStatus> {
    let kv_view = decode_dispatch.kv_block_table_view;
    let decode_view = decode_dispatch.decode_view;
    let dto = decode_dispatch.request_dispatch;
    let lane_count = decode_dispatch.active_sequence_count;
    if lane_count == 0
        || lane_count > core.work_control.max_lane_count
        || decode_dispatch.request_count != lane_count
        || decode_view.active_sequence_count != lane_count
        || decode_view.lane_count != lane_count
        || kv_view.lane_count != lane_count
        || kv_view.lane_capacity < lane_count
        || kv_view.block_token_count == 0
    {
        return Err(ServingStatus::InvalidArgument);
    }
    let dispatch_kind = dispatch_kind_code(decode_dispatch.dispatch_kind);
    if dispatch_kind != DISPATCH_KIND_DECODE_BATCH
        && dispatch_kind != DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH
    {
        return Err(ServingStatus::InvalidArgument);
    }
    let max_speculative_token_count = core.work_control.max_speculative_token_count();
    if (dispatch_kind == DISPATCH_KIND_DECODE_BATCH
        && decode_dispatch.speculative_token_count != 0)
        || decode_dispatch.speculative_token_count > max_speculative_token_count
    {
        return Err(ServingStatus::InvalidArgument);
    }
    if kv_view.lane_stride == 0 {
        return Err(ServingStatus::InvalidArgument);
    }
    let payload_bytes = wire::IPC_SUBMIT_DECODE_HEADER_BYTES;
    let mut message = vec![0u8; payload_bytes];
    wire::write_u32(&mut message, 0, wire::IPC_SUBMIT_DECODE_HEADER_BYTES as u32);
    wire::write_u32(&mut message, 4, extras.highest_priority);
    wire::write_u32(&mut message, 8, dto.flags);
    wire::write_u32(&mut message, 12, dispatch_kind);
    wire::write_u32(&mut message, 16, lane_count);
    wire::write_u32(&mut message, 20, lane_count);
    wire::write_u32(&mut message, 28, decode_dispatch.speculative_token_count);
    wire::write_u32(&mut message, 32, kv_view.block_token_count);
    wire::write_u32(&mut message, 36, 0);
    wire::write_u32(&mut message, 40, wire::IPC_SUBMIT_FLAG_INTERNAL_KV_DIRECTORY);
    wire::write_u64(&mut message, 48, core.session_id_base);
    let rows_per_lane = if dispatch_kind == DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH {
        dto.speculative_verifier_token_count
    } else {
        1
    };
    let execution_row_count = u64::from(lane_count) * u64::from(rows_per_lane);
    if rows_per_lane == 0 || execution_row_count > u64::from(u32::MAX) {
        return Err(ServingStatus::CapacityExceeded);
    }
    let wc_dispatch = to_work_control_decode_dispatch(decode_dispatch, extras);
    let execution_batch_bucket = work_control::select_execution_batch_bucket(
        &wc_dispatch.request_dispatch,
        execution_row_count as u32,
    )
    .map_err(|_| ServingStatus::InvalidArgument)?;
    wire::write_u32(&mut message, 24, execution_batch_bucket);
    let draft_stride = spark_serve::serving_engine::bridge::MAX_SPECULATIVE_TOKENS;
    for lane_index in 0..lane_count as usize {
        let lane = &decode_view.lanes[lane_index];
        if lane.request_id != dto.request_ids[lane_index]
            || lane.sequence_id != extras.sequence_ids[lane_index]
            || lane.request_slot_index != extras.request_slot_indices[lane_index]
        {
            return Err(ServingStatus::InvalidArgument);
        }
        let base = DECODE_MESSAGE_HEADER_BYTES + lane_index * DECODE_LANE_BYTES;
        let target = &mut message[base..base + DECODE_LANE_BYTES];
        wire::write_u64(target, 0, lane.request_id);
        // Deviation (see module docs): the C never sets the lane
        // `request_generation`; the work-packet lane stamping uses the
        // request handle, so the resident lane does too.
        wire::write_u64(target, 8, lane.request_handle);
        wire::write_u64(target, 16, lane.sequence_id);
        wire::write_u64(target, 24, u64::from(lane.sequence_position));
        wire::write_u32(target, 32, lane.request_slot_index);
        wire::write_u32(target, 36, lane.context_token_count);
        wire::write_u32(target, 40, decode_dispatch.input_token_ids[lane_index]);
        wire::write_u32(target, 44, dto.mtp_draft_token_budget);
        wire::write_u32(target, 48, decode_dispatch.speculative_token_count);
        target[52] = lane.mtp_resolution_proposed_token_count as u8;
        target[53] = lane.mtp_resolution_accepted_token_count as u8;
        wire::write_u16(target, 54, lane.mtp_resolution_path_id as u16);
        wire::write_u32(target, 56, 0);
        wire::write_u32(target, 60, 0);
        let draft_base = lane_index * draft_stride;
        for token_index in 0..draft_stride {
            wire::write_u32(
                target,
                64 + token_index * 4,
                decode_dispatch.speculative_draft_token_ids[draft_base + token_index],
            );
        }
    }
    // Deviation (see module docs): stamp the transaction fields the wire
    // validator and the resident daemon's packet conversion require, with
    // the C test finalizer's derivations.
    let lane0_request_generation = wire::read_u64(&message, DECODE_MESSAGE_HEADER_BYTES + 8);
    wire::write_u64(&mut message, 56, lane0_request_generation);
    let mut step_generation = 0x5100_0000_0000_0000u64
        ^ (u64::from(dispatch_kind) << 32)
        ^ lane0_request_generation
        ^ u64::from(lane_count);
    if step_generation == 0 {
        step_generation = 1;
    }
    wire::write_u64(&mut message, 64, step_generation);
    wire::write_u32(&mut message, 72, 0);
    wire::write_u32(&mut message, 76, 1);
    wire::write_u32(
        &mut message,
        80,
        if dispatch_kind == DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH {
            work_control::work_transaction::PHASE_VERIFY
        } else {
            work_control::work_transaction::PHASE_DECODE
        },
    );
    validate_submit_decode_payload(
        &message,
        lane_count,
        max_speculative_token_count,
        core.work_control.kv_block_capacity(),
    )?;
    Ok(message)
}

/// `SparkRingServiceBackendSubmitDecodeToResident`.
pub fn submit_decode_to_resident(
    core: &mut BackendCore,
    decode_dispatch: &ServingDecodeDispatch,
    extras: &DispatchExtras,
) -> Result<(), ServingStatus> {
    let payload = build_decode_resident_payload(core, decode_dispatch, extras)?;
    resident::submit_message(core, wire::IPC_KIND_SUBMIT_DECODE, &payload)
}

/// `SparkRingServiceBackendSubmitDecodeChunksToResident`.
pub fn submit_decode_chunks_to_resident(
    core: &mut BackendCore,
    decode_dispatch: &ServingDecodeDispatch,
    extras: &DispatchExtras,
) -> Result<(), ServingStatus> {
    if !is_mtp_verify(decode_dispatch) {
        return Err(ServingStatus::InvalidArgument);
    }
    let (maximum_lanes_per_chunk, chunk_count) = plan_decode_chunks(core, decode_dispatch)?;
    let wc_dispatch = to_work_control_decode_dispatch(decode_dispatch, extras);
    let mut lane_offset = 0u32;
    for chunk_index in 0..chunk_count {
        let mut lane_count = decode_dispatch.request_count - lane_offset;
        if lane_count > maximum_lanes_per_chunk {
            lane_count = maximum_lanes_per_chunk;
        }
        let mut packet = work_control::build_decode_packet_range(
            &core.work_control,
            &wc_dispatch,
            lane_offset,
            lane_count,
            0,
        )
        .map_err(|_| ServingStatus::InvalidArgument)?;
        core.work_output.stamp_work_packet(&mut packet)?;
        resident::submit_work(core, &packet, 0)?;
        lane_offset += lane_count;
    }
    if lane_offset == decode_dispatch.request_count {
        Ok(())
    } else {
        Err(ServingStatus::InternalError)
    }
}

// ---------------------------------------------------------------------------
// The decode callback
// ---------------------------------------------------------------------------
/// `SparkRingServiceBackendTraceDecodeSubmit`.
fn trace_decode_submit(core: &mut BackendCore, decode_dispatch: &ServingDecodeDispatch) -> u64 {
    if !core.trace_enabled {
        return 0;
    }
    let request_id = if decode_dispatch.request_dispatch.request_count != 0 {
        decode_dispatch.request_dispatch.request_ids[0]
    } else {
        0
    };
    let submit_ns = super::net::monotonic_ns();
    eprintln!(
        "ring_decode kind={} requests={} active={} request={}",
        dispatch_kind_code(decode_dispatch.dispatch_kind),
        decode_dispatch.request_count,
        decode_dispatch.active_sequence_count,
        request_id
    );
    if submit_ns != 0 && core.trace_last_decode_completion_ns != 0 {
        eprintln!(
            "ring_decode_gap_ns={} request={} kind={}",
            submit_ns - core.trace_last_decode_completion_ns,
            request_id,
            dispatch_kind_code(decode_dispatch.dispatch_kind)
        );
    }
    submit_ns
}

/// `SparkRingServiceBackendDecodeInner` (+ the `Decode` trace wrapper).
///
/// Callback context: no engine access (see module docs). `adapter` is used
/// to recover the full-dispatch extras for packet building.
pub fn decode_inner(
    core: &mut BackendCore,
    adapter: &RequestApiAdapter,
    decode_dispatch: &ServingDecodeDispatch,
    decode_result: &mut ServingDecodeResult,
) -> Result<(), ServingStatus> {
    if decode_dispatch.active_sequence_count == 0
        || decode_dispatch.active_sequence_count as usize
            > spark_serve::serving_engine::bridge::MAX_DISPATCH_REQUEST_COUNT
        || decode_dispatch.request_count != decode_dispatch.active_sequence_count
    {
        return Err(ServingStatus::InvalidArgument);
    }
    let extras = adapter.lookup_extras(decode_dispatch.request_dispatch)?;
    let trace_begin_ns = if core.trace_enabled {
        let begin = super::net::monotonic_ns();
        let request_id = if decode_dispatch.request_dispatch.request_count != 0 {
            decode_dispatch.request_dispatch.request_ids[0]
        } else {
            0
        };
        if begin != 0 {
            eprintln!("ring_trace decode_begin request={request_id}");
        }
        begin
    } else {
        0
    };
    let status = decode_inner_impl(core, decode_dispatch, &extras, decode_result);
    if core.trace_enabled && trace_begin_ns != 0 {
        let end = super::net::monotonic_ns();
        let status_code = match &status {
            Ok(()) => ServingStatus::Ok.code(),
            Err(error) => error.code(),
        };
        eprintln!(
            "ring_trace decode_end status={} dur_us={}",
            status_code,
            (end - trace_begin_ns) / 1000
        );
    }
    status
}

fn decode_inner_impl(
    core: &mut BackendCore,
    decode_dispatch: &ServingDecodeDispatch,
    extras: &DispatchExtras,
    decode_result: &mut ServingDecodeResult,
) -> Result<(), ServingStatus> {
    if core.cuda_resident_attached {
        let mut resident_submit_count = 1u32;
        if is_mtp_verify(decode_dispatch) {
            let (_maximum_lanes, chunk_count) = plan_decode_chunks(core, decode_dispatch)?;
            resident_submit_count = chunk_count;
        }
        // Callback-context credit check (see module docs): the backend pump
        // owns ensure + response pumping; a shortfall retries the dispatch.
        if !core.resident.is_connected()
            || core
                .resident
                .credits
                .available(wire::CREDIT_DOMAIN_RESIDENT_RESERVATION)
                < resident_submit_count
        {
            return Err(ServingStatus::Busy);
        }
        *decode_result = ServingDecodeResult::new(
            decode_dispatch.active_sequence_count,
            DECODE_TOKEN_STRIDE,
        );
        let mtp_verify = is_mtp_verify(decode_dispatch);
        let pending_index = core.pendings.register(decode_dispatch, decode_result)?;
        if !core
            .pendings
            .get(pending_index)
            .is_some_and(|pending| pending.active)
        {
            return Ok(());
        }
        let trace_submit_ns = trace_decode_submit(core, decode_dispatch);
        core.pendings.set_trace_submit_time(pending_index, trace_submit_ns);
        if let Err(status) = forward_decode_work(core, decode_dispatch, extras, pending_index) {
            core.pendings.free_slot(pending_index);
            return Err(status);
        }
        let submit_status = if mtp_verify {
            submit_decode_chunks_to_resident(core, decode_dispatch, extras)
        } else {
            submit_decode_to_resident(core, decode_dispatch, extras)
        };
        if let Err(status) = submit_status {
            core.pendings.free_slot(pending_index);
            return Err(status);
        }
        return match core.pendings.apply_early_events_for_callback(pending_index) {
            CallbackDrain::Completed => {
                core.pendings
                    .complete_for_callback(pending_index, decode_result);
                Ok(())
            }
            CallbackDrain::StillPending => Err(ServingStatus::Pending),
            CallbackDrain::Failed(status) => {
                core.pendings.free_slot(pending_index);
                Err(status)
            }
        };
    }
    if !core.rank0_builder.is_attached() {
        return Err(ServingStatus::ModuleNotValidated);
    }
    let (_maximum_lanes, chunk_count) = plan_decode_chunks(core, decode_dispatch)?;
    if chunk_count != 1 {
        return Err(ServingStatus::ModuleNotValidated);
    }
    // The builder seam is taken out of `core` so the idle pump can borrow
    // the core while the builder runs (decode itself does not idle-pump,
    // but the take/replace discipline matches the prefill path).
    let mut builder = std::mem::replace(
        &mut core.rank0_builder,
        Box::new(super::state::NullRank0NodeContext),
    );
    let builder_status = builder.decode(decode_dispatch, decode_result);
    core.rank0_builder = builder;
    if let Err(status) = builder_status {
        eprintln!("ring_decode_builder status={}", status.code());
        return Err(status);
    }
    let pending_index = core.pendings.register(decode_dispatch, decode_result)?;
    if !core
        .pendings
        .get(pending_index)
        .is_some_and(|pending| pending.active)
    {
        return Ok(());
    }
    let trace_submit_ns = trace_decode_submit(core, decode_dispatch);
    core.pendings.set_trace_submit_time(pending_index, trace_submit_ns);
    if let Err(status) = forward_decode_work(core, decode_dispatch, extras, pending_index) {
        core.pendings.free_slot(pending_index);
        eprintln!("ring_decode_forward status={}", status.code());
        return Err(status);
    }
    let drain = core.pendings.apply_early_events_for_callback(pending_index);
    if core.trace_enabled {
        eprintln!("ring_decode_pending_final begin");
    }
    match drain {
        CallbackDrain::Completed => {
            core.pendings
                .complete_for_callback(pending_index, decode_result);
            Ok(())
        }
        CallbackDrain::StillPending => Err(ServingStatus::Pending),
        CallbackDrain::Failed(status) => {
            core.pendings.free_slot(pending_index);
            Err(status)
        }
    }
}
