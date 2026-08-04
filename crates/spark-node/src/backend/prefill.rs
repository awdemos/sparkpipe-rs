//! Prefill and sequence-release paths — port of the `node/backend.c`
//! `SparkRingServiceBackendPrefill{,Inner}`, `...ForwardPrefillWork`,
//! `...SubmitPrefillToResident`, `...ForwardPrefillPacket`,
//! `...SubmitPrefillPacket`, and `...PumpSequenceReleases` /
//! `...SubmitReleaseToRank0`. (`...QueueSequenceRelease`,
//! `...BuildReleasePacket`, and the record pop live in
//! [`super::work_output`] with the release ring.)
//!
//! The prefill entry point runs in the engine's prefill-callback context;
//! like the decode path it cannot re-enter the engine, so the resident
//! credit requirement uses the callback-context variant (a missing
//! connection reports `Busy`; the engine leaves the prefill dispatch
//! scheduled and the backend pump reconnects).

use std::cell::RefCell;
use std::rc::Rc;

use spark_sched::work_control::{self, WorkControlPacket};
use spark_serve::request_api::dispatch::Dispatch as FullDispatch;
use spark_serve::serving_engine::bridge::{Dispatch, DispatchKind};
use spark_serve::serving_engine::engine::ServingPrefillDispatch;
use spark_serve::serving_engine::ServingStatus;

use super::adapt::RequestApiAdapter;
use super::decode::{dispatch_kind_code, map_work_control_error};
use super::resident;
use super::state::BackendCore;
use super::wire::{self, IPC_KIND_SUBMIT_PREFILL};

/// Convert the engine's borrowed prefill dispatch plus the staged full
/// dispatch into work control's owned input shape.
fn work_control_prefill_dispatch(
    prefill_dispatch: &ServingPrefillDispatch,
    full: &FullDispatch,
) -> Result<work_control::PromptPipelinePrefillDispatch, ServingStatus> {
    let lane_count = prefill_dispatch.lane_count as usize;
    let request_dispatch = work_control::RequestApiDispatch {
        kind: dispatch_kind_code(prefill_dispatch.dispatch_kind),
        flags: full.flags,
        request_count: full.request_count,
        highest_priority: full.highest_priority,
        request_handles: full.request_handles[..lane_count].to_vec(),
        request_ids: full.request_ids[..lane_count].to_vec(),
        sequence_ids: full.sequence_ids[..lane_count].to_vec(),
        prefill_decision_batch_bucket: full
            .prefill_decision
            .map_or(0, |decision| decision.batch_bucket),
        prefill_batch_decision_batch_bucket: full
            .prefill_batch_decision
            .map_or(0, |decision| decision.batch_bucket),
        decode_batch_decision_batch_bucket: full
            .decode_batch_decision
            .map_or(0, |decision| decision.batch_bucket),
        speculative_verifier_token_count: full.speculative_verifier_token_count,
        mtp_draft_token_budget: full.mtp_draft_token_budget,
    };
    let prefill_view = work_control::PrefillDispatchView {
        lane_count: prefill_dispatch.lane_count,
        prompt_token_count: prefill_dispatch.prompt_token_count,
        prompt_token_stride: prefill_dispatch.prompt_token_stride,
        lanes: prefill_dispatch
            .lanes
            .iter()
            .map(|lane| work_control::PrefillDispatchLaneView {
                request_index: lane.request_index,
                prompt_token_offset: lane.prompt_token_offset,
                prompt_token_count: lane.prompt_token_count,
                request_slot_index: lane.request_slot_index,
                request_id: lane.request_id,
                sequence_id: lane.sequence_id,
                request_handle: lane.request_handle,
            })
            .collect(),
    };
    Ok(work_control::PromptPipelinePrefillDispatch {
        dispatch_kind: dispatch_kind_code(prefill_dispatch.dispatch_kind),
        active_sequence_count: prefill_dispatch.active_sequence_count,
        lane_count: prefill_dispatch.lane_count,
        prompt_token_offset: prefill_dispatch.prompt_token_offset,
        prompt_token_count: prefill_dispatch.prompt_token_count,
        prompt_token_stride: prefill_dispatch.prompt_token_stride,
        host_token_stride: prefill_dispatch.host_token_stride,
        request_dispatch,
        prefill_view,
        host_token_ids: prefill_dispatch.host_token_ids.to_vec(),
        kv_block_table_view: Some(work_control::DispatchKvBlockTableView {
            block_token_count: prefill_dispatch.kv_block_table_view.block_token_count,
            lane_count: prefill_dispatch.kv_block_table_view.lane_count,
            lane_stride: prefill_dispatch.kv_block_table_view.lane_stride,
            lane_capacity: prefill_dispatch.kv_block_table_view.lane_capacity,
        }),
    })
}

/// Prefill-callback full-dispatch lookup (the callback carries no
/// request-dispatch DTO, so identity is matched on the lane handles/ids).
fn lookup_full(
    adapter: &Rc<RefCell<RequestApiAdapter>>,
    prefill_dispatch: &ServingPrefillDispatch,
) -> Result<FullDispatch, ServingStatus> {
    let lane_count = prefill_dispatch.lane_count as usize;
    let request_handles: Vec<u64> = prefill_dispatch.lanes[..lane_count]
        .iter()
        .map(|lane| lane.request_handle)
        .collect();
    let request_ids: Vec<u64> = prefill_dispatch.lanes[..lane_count]
        .iter()
        .map(|lane| lane.request_id)
        .collect();
    adapter.borrow().lookup_full_by_lanes(
        &[DispatchKind::Prefill, DispatchKind::PrefillBatch],
        &request_handles,
        &request_ids,
    )
}

/// `SparkRingServiceBackendForwardPrefillWork` (ring forwarding only; the
/// no-next-rank early-out and the execution-row capacity gate included).
pub fn forward_prefill_work(
    core: &mut BackendCore,
    prefill_dispatch: &ServingPrefillDispatch,
    full: &FullDispatch,
) -> Result<(), ServingStatus> {
    if !core.work_output.has_next() {
        return Ok(());
    }
    if prefill_dispatch.lane_count > core.rank_plan.execution_row_capacity {
        return Err(ServingStatus::CapacityExceeded);
    }
    let packet_input = work_control_prefill_dispatch(prefill_dispatch, full)?;
    let mut token_offset = 0u32;
    while token_offset < prefill_dispatch.prompt_token_count {
        let token_count = work_control::select_prefill_chunk(
            &core.work_control,
            &packet_input,
            token_offset,
            core.rank_plan.execution_row_capacity,
        )
        .map_err(map_work_control_error)?;
        let mut packet = work_control::build_prefill_packet(
            &core.work_control,
            &packet_input,
            token_offset,
            token_count,
        )
        .map_err(map_work_control_error)?;
        core.work_output.stamp_work_packet(&mut packet)?;
        work_control::validate_packet(
            &packet,
            &core.work_control,
            core.rank_plan.execution_row_capacity,
            core.config.max_pipeline_slot_count,
        )
        .map_err(map_work_control_error)?;
        core.work_output.enqueue_work_packet(&packet)?;
        match core.work_output.pump_work_output() {
            Ok(()) | Err(ServingStatus::Busy) => {}
            Err(status) => return Err(status),
        }
        token_offset += token_count;
    }
    Ok(())
}

/// `SparkRingServiceBackendForwardPrefillPacket` (validate + enqueue + pump
/// with `Busy` tolerated).
fn forward_prefill_packet(
    core: &mut BackendCore,
    packet: &WorkControlPacket,
) -> Result<(), ServingStatus> {
    work_control::validate_packet(
        packet,
        &core.work_control,
        core.rank_plan.execution_row_capacity,
        core.config.max_pipeline_slot_count,
    )
    .map_err(map_work_control_error)?;
    core.work_output.enqueue_work_packet(packet)?;
    match core.work_output.pump_work_output() {
        Ok(()) | Err(ServingStatus::Busy) => Ok(()),
        Err(status) => Err(status),
    }
}

/// `SparkRingServiceBackendSubmitPrefillPacket` (the blocker on failure is
/// part of the C contract).
fn submit_prefill_packet(
    core: &mut BackendCore,
    payload: &[u8],
) -> Result<(), ServingStatus> {
    if let Err(status) = resident::submit_message(core, IPC_KIND_SUBMIT_PREFILL, payload) {
        core.set_blocker("forwarded prefill packet was not queued locally");
        return Err(status);
    }
    Ok(())
}

/// `SparkRingServiceBackendSubmitPrefillToResident` (chunked; one
/// SUBMIT_PREFILL message per wave).
pub fn submit_prefill_to_resident(
    core: &mut BackendCore,
    prefill_dispatch: &ServingPrefillDispatch,
    full: &FullDispatch,
) -> Result<(), ServingStatus> {
    if prefill_dispatch.lane_count == 0
        || prefill_dispatch.lane_count != prefill_dispatch.active_sequence_count
        || prefill_dispatch.prompt_token_count == 0
        || prefill_dispatch.prompt_token_count > core.config.builder_max_prefill_tokens
    {
        return Err(ServingStatus::InvalidArgument);
    }
    if prefill_dispatch.lane_count > core.rank_plan.execution_row_capacity {
        return Err(ServingStatus::CapacityExceeded);
    }
    let packet_input = work_control_prefill_dispatch(prefill_dispatch, full)?;
    let mut token_offset = 0u32;
    while token_offset < prefill_dispatch.prompt_token_count {
        let token_count = work_control::select_prefill_chunk(
            &core.work_control,
            &packet_input,
            token_offset,
            core.rank_plan.execution_row_capacity,
        )
        .map_err(map_work_control_error)?;
        resident::require_credits_for_callback(core, 1)?;
        let mut packet = work_control::build_prefill_packet(
            &core.work_control,
            &packet_input,
            token_offset,
            token_count,
        )
        .map_err(map_work_control_error)?;
        core.work_output.stamp_work_packet(&mut packet)?;
        // `SparkCudaResidentIpcCalculateSubmitPrefillBytes`: 8-byte prefix
        // (descriptor_bytes, request_flags) + the packet wire bytes.
        let packet_bytes = wire::serialize_packet(&core.work_control, &packet)?;
        let descriptor_bytes = 8 + packet_bytes.len() as u32;
        let mut payload = Vec::with_capacity(descriptor_bytes as usize);
        payload.extend_from_slice(&descriptor_bytes.to_le_bytes());
        payload.extend_from_slice(&full.flags.to_le_bytes());
        payload.extend_from_slice(&packet_bytes);
        forward_prefill_packet(core, &packet)?;
        submit_prefill_packet(core, &payload)?;
        token_offset += token_count;
    }
    Ok(())
}

/// `SparkRingServiceBackendPrefillInner` (+ the `Prefill` trace wrapper,
/// [`prefill`]).
pub fn prefill_inner(
    core: &mut BackendCore,
    adapter: &Rc<RefCell<RequestApiAdapter>>,
    prefill_dispatch: &ServingPrefillDispatch,
) -> Result<(), ServingStatus> {
    if core.cuda_resident_attached {
        // Callback-context ensure: never connects (the C's Ensure reconnects
        // with engine access; here a missing connection reports Busy and the
        // backend pump reconnects).
        if core.resident.fd.is_none() {
            return Err(ServingStatus::Busy);
        }
        let full = lookup_full(adapter, prefill_dispatch)?;
        return submit_prefill_to_resident(core, prefill_dispatch, &full);
    }
    if !core.rank0_builder.is_attached() {
        return Err(ServingStatus::ModuleNotValidated);
    }
    eprintln!(
        "ring_prefill dispatch_kind={} active={} lanes={} offset={} tokens={}",
        dispatch_kind_code(prefill_dispatch.dispatch_kind),
        prefill_dispatch.active_sequence_count,
        prefill_dispatch.lane_count,
        prefill_dispatch.prompt_token_offset,
        prefill_dispatch.prompt_token_count
    );
    let full = lookup_full(adapter, prefill_dispatch)?;
    if let Err(status) = forward_prefill_work(core, prefill_dispatch, &full) {
        eprintln!("ring_prefill_forward status={}", status.code());
        return Err(status);
    }
    let status = {
        let builder = &mut core.rank0_builder;
        let work_output = &mut core.work_output;
        let mut idle_pump = move || work_output.pump_work_output();
        builder.prefill(prefill_dispatch, &mut idle_pump)
    };
    eprintln!(
        "ring_prefill_builder status={}",
        match &status {
            Ok(()) => ServingStatus::Ok.code(),
            Err(error) => error.code(),
        }
    );
    status
}

/// `SparkRingServiceBackendPrefill` (the trace wrapper).
pub fn prefill(
    core: &mut BackendCore,
    adapter: &Rc<RefCell<RequestApiAdapter>>,
    prefill_dispatch: &ServingPrefillDispatch,
) -> Result<(), ServingStatus> {
    let trace_begin_ns = if core.trace_enabled {
        super::net::monotonic_ns()
    } else {
        0
    };
    if trace_begin_ns != 0 {
        eprintln!(
            "ring_trace prefill_begin request={} offset={} count={}",
            prefill_dispatch.lanes.first().map_or(0, |lane| lane.request_id),
            prefill_dispatch.prompt_token_offset,
            prefill_dispatch.prompt_token_count
        );
    }
    let status = prefill_inner(core, adapter, prefill_dispatch);
    if trace_begin_ns != 0 {
        let dur_us = (super::net::monotonic_ns() - trace_begin_ns) / 1000;
        let status_code = match &status {
            Ok(()) => ServingStatus::Ok.code(),
            Err(error) => error.code(),
        };
        eprintln!("ring_trace prefill_end status={status_code} dur_us={dur_us}");
    }
    status
}

// ---------------------------------------------------------------------------
// Sequence releases
// ---------------------------------------------------------------------------

/// `SparkRingServiceBackendSubmitReleaseToRank0`: the resident path when
/// attached, otherwise the rank0 builder's `submit_work` entry point.
fn submit_release_to_rank0(
    core: &mut BackendCore,
    engine: &mut spark_serve::serving_engine::engine::ServingEngine,
    cancel_dispatch: &mut dyn FnMut(&mut Dispatch) -> Result<(), ServingStatus>,
    packet: &WorkControlPacket,
) -> Result<(), ServingStatus> {
    if core.cuda_resident_attached {
        return resident::submit_release(core, engine, cancel_dispatch, packet);
    }
    if !core.rank0_builder.is_attached() {
        return Err(ServingStatus::ModuleNotValidated);
    }
    core.rank0_builder.submit_work(packet)
}

/// `SparkRingServiceBackendPumpSequenceReleases` (one packet of up to
/// `max_lane_count` records per call; the queue-room `CapacityExceeded`
/// collapses to `Busy`, as in C).
pub fn pump_sequence_releases(
    core: &mut BackendCore,
    engine: &mut spark_serve::serving_engine::engine::ServingEngine,
    cancel_dispatch: &mut dyn FnMut(&mut Dispatch) -> Result<(), ServingStatus>,
) -> Result<(), ServingStatus> {
    if core.work_output.release_count() == 0 {
        return Ok(());
    }
    let mut lane_count = core.work_output.release_count() as u32;
    if lane_count > core.work_control.max_lane_count {
        lane_count = core.work_control.max_lane_count;
    }
    let packet = core.work_output.build_release_packet(
        lane_count,
        core.config.kv_block_tokens,
        core.config.max_blocks_per_sequence(),
        mtp_resolution_none(),
    )?;
    if let Err(status) = core.work_output.enqueue_work_packet(&packet) {
        if status == ServingStatus::CapacityExceeded {
            return Err(ServingStatus::Busy);
        }
        return Err(status);
    }
    match core.work_output.pump_work_output() {
        Ok(()) | Err(ServingStatus::Busy) => {}
        Err(status) => return Err(status),
    }
    submit_release_to_rank0(core, engine, cancel_dispatch, &packet)?;
    core.work_output.pop_release_records(lane_count as usize);
    Ok(())
}

/// `SPARK_MODEL_MTP_TREE_RESOLUTION_NONE` as the lane's u16 path id.
fn mtp_resolution_none() -> u16 {
    work_control::mtp_tree::RESOLUTION_NONE as u16
}
