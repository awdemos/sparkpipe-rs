//! Pending-decode pipeline — port of the `node/backend.c`
//! `SparkRingServiceBackendPendingDecode` arena, the early-final-event ring,
//! and the final-event completion/failure choreography.
//!
//! Two completion contexts exist, exactly as in C:
//!
//! * Backend-pump context ([`PendingDecodes::complete_pending_final_event`]):
//!   the engine and the request-API adapter are reachable, so failures run
//!   the full `SparkRingServiceBackendFailPendingDecode` (cancel dispatch +
//!   per-lane request failure events).
//! * Decode-callback context ([`PendingDecodes::apply_early_events_for_callback`]):
//!   the engine is mid-`pump` and cannot be re-entered from its own decode
//!   callback. The callback path therefore applies stashed early events to
//!   the pending slot only; a synchronous completion is reported so the
//!   callback can hand the filled result back through its `decode_result`
//!   out-parameter (the engine then completes the dispatch exactly once),
//!   and failures free the slot and surface the status, leaving the engine
//!   pump's error path to cancel the dispatch and fail the requests — the
//!   same net effect as the C's in-callback `FailPendingDecode`.
//!
//! Deviation note: the Rust `serving_engine::bridge::Dispatch` DTO omits
//! `SparkRequestApiDispatch.sequence_ids`, so each pending slot carries its
//! own per-lane `sequence_ids` captured from the decode view at
//! registration.

use spark_serve::request_api::dispatch::{
    DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY, DISPATCH_FLAG_DSPARK_TAP_CAPTURE,
};
use spark_serve::serving_engine::bridge::{self, Dispatch, MAX_DISPATCH_REQUEST_COUNT};
use spark_serve::serving_engine::engine::{
    ServingDecodeDispatch, ServingDecodeResult, ServingEngine,
};
use spark_serve::serving_engine::ServingStatus;

use super::wire::{
    self, DsparkDraftResult, FinalEvent, COMPLETION_TOKEN_CAPACITY, DRAFT_TOKEN_CAPACITY,
    DSPARK_ABI_VERSION, DSPARK_DRAFT_RESULT_BYTES,
};

/// `SPARK_SERVING_MAX_DECODE_TOKENS_PER_LANE` (bridge
/// [`bridge::MAX_DECODE_TOKENS_PER_LANE`]).
const MAX_DECODE_TOKENS_PER_LANE: usize = bridge::MAX_DECODE_TOKENS_PER_LANE;
/// `SPARK_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT`.
const MTP_MAX_DRAFT_TOKEN_COUNT: usize = bridge::MTP_MAX_DRAFT_TOKEN_COUNT;
/// Lane array bound (`SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT`).
const MAX_LANES: usize = MAX_DISPATCH_REQUEST_COUNT;

/// `SparkRingServiceBackendLaneTransaction`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LaneTransaction {
    pub control_generation: u64,
    pub transaction_id: u64,
    pub dispatch_generation: u64,
    pub request_generation: u64,
    pub sequence_position: u64,
    pub step_generation: u64,
    pub step_chunk_index: u32,
    pub step_chunk_count: u32,
    pub transaction_phase: u32,
    pub reserved0: u32,
}

/// `SparkRingServiceBackendPendingDecode` (`state == FREE` becomes
/// `active: bool`).
pub struct PendingDecode {
    pub active: bool,
    pub done_count: u32,
    pub trace_submit_time_ns: u64,
    lane_done: Vec<bool>,
    lane_final_event_fingerprints: Vec<u64>,
    lane_transactions: Vec<LaneTransaction>,
    dspark_draft_valid: Vec<bool>,
    dspark_drafts: Vec<DsparkDraftResult>,
    pub dispatch: Dispatch,
    /// Per-lane sequence ids (see module docs; from the decode view).
    pub sequence_ids: Vec<u64>,
    pub result: ServingDecodeResult,
}

impl PendingDecode {
    fn new() -> Self {
        PendingDecode {
            active: false,
            done_count: 0,
            trace_submit_time_ns: 0,
            lane_done: vec![false; MAX_LANES],
            lane_final_event_fingerprints: vec![0; MAX_LANES],
            lane_transactions: vec![LaneTransaction::default(); MAX_LANES],
            dspark_draft_valid: vec![false; MAX_LANES],
            dspark_drafts: vec![DsparkDraftResult::default(); MAX_LANES],
            dispatch: Dispatch::new(),
            sequence_ids: vec![0; MAX_LANES],
            result: ServingDecodeResult::new(0, 0),
        }
    }

    /// `memset(pending, 0, sizeof(*pending))` (allocation reuse: the big
    /// `Vec`s keep their capacity, as the C arena keeps its slot storage).
    fn reset(&mut self) {
        self.active = false;
        self.done_count = 0;
        self.trace_submit_time_ns = 0;
        self.lane_done.fill(false);
        self.lane_final_event_fingerprints.fill(0);
        self.lane_transactions.fill(LaneTransaction::default());
        self.dspark_draft_valid.fill(false);
        self.dspark_drafts.fill(DsparkDraftResult::default());
        self.dispatch = Dispatch::new();
        self.sequence_ids.fill(0);
        self.result = ServingDecodeResult::new(0, 0);
    }
}

/// Result of applying one final event to a located lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LaneApply {
    /// Lane recorded; the dispatch still waits for more lanes.
    Applied,
    /// Byte-identical duplicate of an already-recorded lane event.
    Duplicate,
    /// This was the last lane; the pending decode is complete.
    Completed,
}

/// Outcome of [`PendingDecodes::apply_early_events_for_callback`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallbackDrain {
    /// Events (if any) were applied; lanes remain outstanding.
    StillPending,
    /// All lanes resolved; `result` is ready for the callback's
    /// out-parameter and the slot can be freed by the caller.
    Completed,
    /// A lane event failed validation or reported a failure status; the
    /// caller frees the slot and returns the status to the engine.
    Failed(ServingStatus),
}

/// The pending-decode arena plus the early-final-event ring and the
/// final-event counters (`state->pending_decodes`,
/// `state->early_final_events*`, `state->final_event_*`).
pub struct PendingDecodes {
    slots: Vec<PendingDecode>,
    early_events: Vec<FinalEvent>,
    early_head: usize,
    early_count: usize,
    pub final_event_receive_count: u64,
    pub final_event_receive_error_count: u64,
    pub last_final_event_token_count: u32,
    pub last_final_event_token_ids: [u32; COMPLETION_TOKEN_CAPACITY],
}

impl PendingDecodes {
    pub fn new(pending_capacity: usize, early_event_capacity: usize) -> Self {
        PendingDecodes {
            slots: (0..pending_capacity).map(|_| PendingDecode::new()).collect(),
            early_events: vec![FinalEvent::default(); early_event_capacity],
            early_head: 0,
            early_count: 0,
            final_event_receive_count: 0,
            final_event_receive_error_count: 0,
            last_final_event_token_count: 0,
            last_final_event_token_ids: [0; COMPLETION_TOKEN_CAPACITY],
        }
    }

    pub fn pending_capacity(&self) -> usize {
        self.slots.len()
    }

    pub fn early_event_count(&self) -> usize {
        self.early_count
    }

    pub fn early_event_capacity(&self) -> usize {
        self.early_events.len()
    }

    /// Read access to a slot (tests, dspark seam).
    pub fn get(&self, index: usize) -> Option<&PendingDecode> {
        self.slots.get(index)
    }

    /// `pending->trace_submit_time_ns = ...` from the decode callback.
    pub fn set_trace_submit_time(&mut self, index: usize, trace_submit_time_ns: u64) {
        if let Some(pending) = self.slots.get_mut(index) {
            pending.trace_submit_time_ns = trace_submit_time_ns;
        }
    }

    /// Copy a slot's filled decode result into the callback's out-parameter
    /// (the callback-context synchronous-completion path).
    pub fn copy_result_into(&self, index: usize, decode_result: &mut ServingDecodeResult) {
        if let Some(pending) = self.slots.get(index) {
            decode_result.clone_from(&pending.result);
        }
    }

    /// `SparkRingServiceBackendFindFreePendingDecode`.
    pub fn find_free(&self) -> Option<usize> {
        self.slots.iter().position(|slot| !slot.active)
    }

    /// `SparkRingServiceBackendRegisterPendingDecode`.
    pub fn register(
        &mut self,
        decode_dispatch: &ServingDecodeDispatch,
        decode_result: &ServingDecodeResult,
    ) -> Result<usize, ServingStatus> {
        let request_count = decode_dispatch.request_dispatch.request_count as usize;
        if request_count == 0
            || request_count > MAX_LANES
            || decode_dispatch.decode_view.lane_count as usize != request_count
        {
            return Err(ServingStatus::InvalidArgument);
        }
        let index = self.find_free().ok_or(ServingStatus::Busy)?;
        let pending = &mut self.slots[index];
        pending.reset();
        pending.active = true;
        pending.dispatch.clone_from(decode_dispatch.request_dispatch);
        for (lane_index, lane) in
            decode_dispatch.decode_view.lanes.iter().take(request_count).enumerate()
        {
            pending.sequence_ids[lane_index] = lane.sequence_id;
        }
        pending.result.clone_from(decode_result);
        Ok(index)
    }

    /// `SparkRingServiceBackendFindDecodeLane`: `Ok((lane, transaction_matches))`
    /// or `Err(())` when no lane carries the event's request/sequence ids.
    fn find_decode_lane(pending: &PendingDecode, event: &FinalEvent) -> Result<(usize, bool), ()> {
        for lane_index in 0..pending.dispatch.request_count as usize {
            if pending.dispatch.request_ids[lane_index] != event.request_id
                || pending.sequence_ids[lane_index] != event.sequence_id
            {
                continue;
            }
            let transaction = &pending.lane_transactions[lane_index];
            let transaction_matches = transaction.control_generation == event.control_generation
                && transaction.transaction_id == event.transaction_id
                && transaction.dispatch_generation == event.dispatch_generation
                && transaction.request_generation == event.request_generation
                && transaction.sequence_position == event.sequence_position
                && transaction.step_generation == event.step_generation
                && transaction.step_chunk_index == event.step_chunk_index
                && transaction.step_chunk_count == event.step_chunk_count
                && transaction.transaction_phase == event.transaction_phase
                && transaction.reserved0 == 0;
            return Ok((lane_index, transaction_matches));
        }
        Err(())
    }

    /// `SparkRingServiceBackendFindPendingDecodeForEvent`.
    pub fn find_for_event(&self, event: &FinalEvent) -> Option<(usize, usize, bool)> {
        for (index, pending) in self.slots.iter().enumerate() {
            if !pending.active {
                continue;
            }
            if let Ok((lane_index, transaction_matches)) = Self::find_decode_lane(pending, event) {
                return Some((index, lane_index, transaction_matches));
            }
        }
        None
    }

    /// `SparkRingServiceBackendFindPendingDecodeForMalformedEvent`.
    pub fn find_for_malformed_event(&self, event: &FinalEvent) -> Option<usize> {
        if event.request_id == 0 || event.sequence_id == 0 {
            return None;
        }
        for (index, pending) in self.slots.iter().enumerate() {
            if !pending.active {
                continue;
            }
            for lane_index in 0..pending.dispatch.request_count as usize {
                if pending.dispatch.request_ids[lane_index] != event.request_id
                    || pending.sequence_ids[lane_index] != event.sequence_id
                {
                    continue;
                }
                if event.request_generation != 0
                    && pending.dispatch.request_handles[lane_index] != event.request_generation
                {
                    continue;
                }
                return Some(index);
            }
        }
        None
    }

    /// `SparkRingServiceBackendFindPendingDecodeForRequest`.
    pub fn find_for_request(&self, request_id: u64) -> Option<usize> {
        if request_id == 0 {
            return None;
        }
        for (index, pending) in self.slots.iter().enumerate() {
            if !pending.active {
                continue;
            }
            if pending.dispatch.request_ids[..pending.dispatch.request_count as usize]
                .contains(&request_id)
            {
                return Some(index);
            }
        }
        None
    }

    /// `SparkRingServiceBackendRecordDecodeChunk`.
    pub fn record_decode_chunk(
        &mut self,
        index: usize,
        lane_offset: u32,
        packet: &spark_sched::work_control::WorkControlPacket,
    ) -> Result<(), ServingStatus> {
        let pending = self.slots.get_mut(index).ok_or(ServingStatus::InvalidArgument)?;
        if !pending.active
            || lane_offset > pending.dispatch.request_count
            || packet.lane_count > pending.dispatch.request_count - lane_offset
        {
            return Err(ServingStatus::InvalidArgument);
        }
        for lane_index in 0..packet.lane_count as usize {
            let request_index = lane_offset as usize + lane_index;
            let lane = &packet.lanes[lane_index];
            if pending.dispatch.request_ids[request_index] != lane.request_id
                || pending.dispatch.request_handles[request_index] != lane.request_generation
                || pending.sequence_ids[request_index] != lane.sequence_id
            {
                return Err(ServingStatus::ValidationFailed);
            }
            pending.lane_transactions[request_index] = LaneTransaction {
                control_generation: packet.control_generation,
                transaction_id: packet.transaction_id,
                dispatch_generation: packet.dispatch_generation,
                request_generation: lane.request_generation,
                sequence_position: lane.sequence_position,
                step_generation: packet.step_generation,
                step_chunk_index: packet.step_chunk_index,
                step_chunk_count: packet.step_chunk_count,
                transaction_phase: packet.transaction_phase,
                reserved0: 0,
            };
        }
        Ok(())
    }

    /// `SparkRingServiceBackendRecordFinalEvent` (counters only). Callers
    /// pass the raw wire bytes too, since the magic/descriptor words are not
    /// retained in [`FinalEvent`].
    pub fn record_final_event(&mut self, event: &FinalEvent, wire_bytes: &[u8]) {
        if wire::read_u32(wire_bytes, 0) != wire::FINAL_EVENT_MAGIC
            || wire::read_u32(wire_bytes, 4) != wire::FINAL_EVENT_BYTES as u32
        {
            self.final_event_receive_error_count += 1;
            return;
        }
        if (event.completion_flags & wire::COMPLETION_FLAG_TOKEN_IDS) == 0 {
            self.final_event_receive_count += 1;
            self.last_final_event_token_count = 0;
            return;
        }
        let token_count = event.token_count.min(COMPLETION_TOKEN_CAPACITY as u32) as usize;
        self.last_final_event_token_ids[..token_count]
            .copy_from_slice(&event.token_ids[..token_count]);
        self.last_final_event_token_count = token_count as u32;
        self.final_event_receive_count += 1;
    }

    /// `SparkRingServiceBackendStashEarlyFinalEvent`. A successful stash
    /// reports `Err(Busy)`, exactly the C status convention.
    pub fn stash_early_final_event(&mut self, event: &FinalEvent) -> Result<(), ServingStatus> {
        event.validate_envelope()?;
        for event_index in 0..self.early_count {
            let ring_index = (self.early_head + event_index) % self.early_events.len();
            let queued_event = &self.early_events[ring_index];
            if !queued_event.identity_matches(event) {
                continue;
            }
            // C: `memcmp(queued_event, event, sizeof(*event)) == 0`. Both
            // events passed `validate_envelope` (reserved words are zero),
            // so comparing the re-encoded wire bytes is equivalent.
            return if queued_event.encode() == event.encode() {
                Err(ServingStatus::Busy)
            } else {
                Err(ServingStatus::ValidationFailed)
            };
        }
        if self.early_count >= self.early_events.len() {
            return Err(ServingStatus::Busy);
        }
        let tail = (self.early_head + self.early_count) % self.early_events.len();
        self.early_events[tail] = *event;
        self.early_count += 1;
        Err(ServingStatus::Busy)
    }

    /// `SparkRingServiceBackendDropEarlyFinalEvent`.
    fn drop_early_final_event(&mut self, event_index: usize) {
        if event_index >= self.early_count {
            return;
        }
        for shift_index in event_index + 1..self.early_count {
            let read_index = (self.early_head + shift_index) % self.early_events.len();
            let write_index = (self.early_head + shift_index - 1) % self.early_events.len();
            self.early_events[write_index] = self.early_events[read_index];
        }
        self.early_count -= 1;
    }

    /// The shared per-lane apply step of
    /// `SparkRingServiceBackendCompletePendingFinalEvent` (everything after
    /// the event has been located to a lane). On lane-level failure the
    /// status that should fail the pending decode is returned.
    fn apply_final_event_to_lane(
        &mut self,
        index: usize,
        lane_index: usize,
        transaction_matches: bool,
        event: &FinalEvent,
    ) -> Result<LaneApply, ServingStatus> {
        if !transaction_matches {
            return Err(ServingStatus::ValidationFailed);
        }
        if event.status != ServingStatus::Ok.code() {
            return Err(
                wire::status_from_u32(event.status).unwrap_or(ServingStatus::ValidationFailed)
            );
        }
        let event_fingerprint = event.fingerprint();
        if event_fingerprint == 0 {
            return Err(ServingStatus::InternalError);
        }
        let pending = &mut self.slots[index];
        if pending.lane_done[lane_index] {
            if pending.lane_final_event_fingerprints[lane_index] == event_fingerprint {
                return Ok(LaneApply::Duplicate);
            }
            return Err(ServingStatus::ValidationFailed);
        }
        if (event.completion_flags & wire::COMPLETION_FLAG_TOKEN_IDS) == 0
            || (((event.completion_flags & wire::COMPLETION_FLAG_DRAFT_TOKEN_IDS) != 0)
                != (event.draft_token_count != 0))
            || event.draft_token_count as usize > MTP_MAX_DRAFT_TOKEN_COUNT
        {
            return Err(ServingStatus::ValidationFailed);
        }
        let dspark_expected = (pending.dispatch.flags
            & (DISPATCH_FLAG_DSPARK_TAP_CAPTURE | DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY))
            != 0;
        if dspark_expected {
            if (event.extension_flags & wire::FINAL_EVENT_FLAG_DSPARK_DRAFT) == 0
                || event.dspark_draft.abi_version != DSPARK_ABI_VERSION
                || event.dspark_draft.descriptor_bytes != DSPARK_DRAFT_RESULT_BYTES as u32
                || event.dspark_draft.token_count == 0
            {
                return Err(ServingStatus::ValidationFailed);
            }
            pending.dspark_drafts[lane_index] = event.dspark_draft;
            pending.dspark_draft_valid[lane_index] = true;
        } else if (event.extension_flags & wire::FINAL_EVENT_FLAG_DSPARK_DRAFT) != 0 {
            return Err(ServingStatus::ValidationFailed);
        }
        let token_count = event.token_count as usize;
        if token_count == 0
            || token_count > MAX_DECODE_TOKENS_PER_LANE
            || token_count > COMPLETION_TOKEN_CAPACITY
        {
            return Err(ServingStatus::ValidationFailed);
        }
        let token_base = lane_index * bridge::MAX_DECODE_TOKENS_PER_LANE;
        pending.result.token_ids[token_base..token_base + token_count]
            .copy_from_slice(&event.token_ids[..token_count]);
        pending.result.token_counts[lane_index] = event.token_count;
        pending.result.draft_token_counts[lane_index] = event.draft_token_count;
        let draft_base = lane_index * MTP_MAX_DRAFT_TOKEN_COUNT;
        let draft_count = event.draft_token_count as usize;
        pending.result.draft_token_ids[draft_base..draft_base + draft_count]
            .copy_from_slice(&event.draft_token_ids[..draft_count]);
        pending.lane_final_event_fingerprints[lane_index] = event_fingerprint;
        pending.lane_done[lane_index] = true;
        pending.done_count += 1;
        if pending.done_count != pending.dispatch.request_count {
            return Ok(LaneApply::Applied);
        }
        Ok(LaneApply::Completed)
    }

    /// `SparkRingServiceBackendCompletePendingDecode`.
    pub fn complete_pending_decode(
        &mut self,
        index: usize,
        engine: &mut ServingEngine,
        trace_enabled: bool,
    ) -> Result<(), ServingStatus> {
        let pending = &mut self.slots[index];
        if trace_enabled {
            let completion_ns = super::net::monotonic_ns();
            let request_id = if pending.dispatch.request_count != 0 {
                pending.dispatch.request_ids[0]
            } else {
                0
            };
            if completion_ns != 0 && pending.trace_submit_time_ns != 0 {
                eprintln!(
                    "ring_decode_flight_ns={} kind={:?} request={}",
                    completion_ns - pending.trace_submit_time_ns,
                    pending.dispatch.kind,
                    request_id
                );
            }
        }
        let mut dispatch = std::mem::take(&mut pending.dispatch);
        let mut result = std::mem::replace(&mut pending.result, ServingDecodeResult::new(0, 0));
        let status = engine.complete_decode_dispatch(&mut dispatch, &mut result);
        self.slots[index].reset();
        status
    }

    /// `SparkRingServiceBackendFailPendingDecode`. `cancel_dispatch` is the
    /// adapter-mediated request-API cancel; the per-lane failure events go
    /// through the engine, as in C.
    pub fn fail_pending_decode(
        &mut self,
        index: usize,
        failure_status: ServingStatus,
        engine: &mut ServingEngine,
        cancel_dispatch: &mut dyn FnMut(&mut Dispatch) -> Result<(), ServingStatus>,
    ) -> Result<(), ServingStatus> {
        let pending = &mut self.slots[index];
        if pending.dispatch.request_count == 0 {
            return Err(ServingStatus::InvalidArgument);
        }
        let mut status = cancel_dispatch(&mut pending.dispatch);
        status?;
        status = Ok(());
        for lane_index in 0..pending.dispatch.request_count as usize {
            let route_status = engine.fail_request_by_request_id(
                pending.dispatch.request_ids[lane_index],
                failure_status,
            );
            if let Err(route_status) = route_status {
                if route_status != ServingStatus::NotFound && status.is_ok() {
                    status = Err(route_status);
                }
            }
        }
        self.slots[index].reset();
        status
    }

    /// Free a slot without touching the engine (decode-callback context and
    /// `SparkRingServiceBackendFailWorkPacketCohort`'s "lane lookup missed"
    /// case after the first fail cleared the cohort).
    pub fn free_slot(&mut self, index: usize) {
        if let Some(pending) = self.slots.get_mut(index) {
            pending.reset();
        }
    }

    /// `SparkRingServiceBackendFailInflightResidentDecodes`.
    pub fn fail_inflight(
        &mut self,
        failure_status: ServingStatus,
        engine: &mut ServingEngine,
        cancel_dispatch: &mut dyn FnMut(&mut Dispatch) -> Result<(), ServingStatus>,
    ) -> u32 {
        let mut failed_count = 0u32;
        for index in 0..self.slots.len() {
            if !self.slots[index].active {
                continue;
            }
            let _ = self.fail_pending_decode(index, failure_status, engine, cancel_dispatch);
            failed_count += 1;
        }
        failed_count
    }

    /// `SparkRingServiceBackendCompletePendingFinalEvent`
    /// (backend-pump context).
    pub fn complete_pending_final_event(
        &mut self,
        event: &FinalEvent,
        wire_bytes: &[u8],
        engine: &mut ServingEngine,
        cancel_dispatch: &mut dyn FnMut(&mut Dispatch) -> Result<(), ServingStatus>,
        trace_enabled: bool,
    ) -> Result<(), ServingStatus> {
        if let Err(status) = event.validate_envelope() {
            self.final_event_receive_error_count += 1;
            if let Some(index) = self.find_for_malformed_event(event) {
                return self.fail_pending_decode(
                    index,
                    ServingStatus::ValidationFailed,
                    engine,
                    cancel_dispatch,
                );
            }
            return Err(status);
        }
        self.record_final_event(event, wire_bytes);
        let Some((index, lane_index, transaction_matches)) = self.find_for_event(event) else {
            return self.stash_early_final_event(event);
        };
        match self.apply_final_event_to_lane(index, lane_index, transaction_matches, event) {
            Ok(LaneApply::Duplicate) => Ok(()),
            Ok(LaneApply::Applied) => {
                if trace_enabled {
                    trace_final_event(event, &self.slots[index].dispatch);
                }
                Err(ServingStatus::Busy)
            }
            Ok(LaneApply::Completed) => {
                if trace_enabled {
                    trace_final_event(event, &self.slots[index].dispatch);
                }
                self.complete_pending_decode(index, engine, trace_enabled)
            }
            Err(failure_status) => {
                if trace_enabled && event.status != ServingStatus::Ok.code() {
                    trace_final_event(event, &self.slots[index].dispatch);
                }
                self.fail_pending_decode(index, failure_status, engine, cancel_dispatch)
            }
        }
    }

    /// `SparkRingServiceBackendCompleteEarlyFinalEvents`
    /// (backend-pump context).
    pub fn complete_early_final_events(
        &mut self,
        index: usize,
        engine: &mut ServingEngine,
        cancel_dispatch: &mut dyn FnMut(&mut Dispatch) -> Result<(), ServingStatus>,
        trace_enabled: bool,
    ) -> Result<(), ServingStatus> {
        let mut event_index = 0usize;
        while self.slots[index].active && event_index < self.early_count {
            let ring_index = (self.early_head + event_index) % self.early_events.len();
            let event = self.early_events[ring_index];
            if Self::find_decode_lane(&self.slots[index], &event).is_err() {
                event_index += 1;
                continue;
            }
            self.drop_early_final_event(event_index);
            let wire_bytes = event.encode();
            let status = self.complete_pending_final_event(
                &event,
                &wire_bytes,
                engine,
                cancel_dispatch,
                trace_enabled,
            );
            if let Err(status) = status {
                if status != ServingStatus::Busy {
                    return Err(status);
                }
            }
        }
        if self.slots[index].active {
            Err(ServingStatus::Busy)
        } else {
            Ok(())
        }
    }

    /// Decode-callback-context drain of matching early events (see module
    /// docs): no engine access, so failures only mark the slot for the
    /// caller to free and completions hand the filled result back.
    pub fn apply_early_events_for_callback(&mut self, index: usize) -> CallbackDrain {
        let mut event_index = 0usize;
        while self.slots[index].active && event_index < self.early_count {
            let ring_index = (self.early_head + event_index) % self.early_events.len();
            let event = self.early_events[ring_index];
            let Ok((lane_index, transaction_matches)) =
                Self::find_decode_lane(&self.slots[index], &event)
            else {
                event_index += 1;
                continue;
            };
            self.drop_early_final_event(event_index);
            // The C drain re-enters `CompletePendingFinalEvent`, which
            // re-validates the envelope and re-records the counters; mirror
            // both (the counter double-record is the C behavior).
            let wire_bytes = event.encode();
            if event.validate_envelope().is_err() {
                return CallbackDrain::Failed(ServingStatus::ValidationFailed);
            }
            self.record_final_event(&event, &wire_bytes);
            match self.apply_final_event_to_lane(index, lane_index, transaction_matches, &event) {
                Ok(LaneApply::Duplicate) | Ok(LaneApply::Applied) => {}
                Ok(LaneApply::Completed) => return CallbackDrain::Completed,
                Err(status) => return CallbackDrain::Failed(status),
            }
        }
        CallbackDrain::StillPending
    }

    /// Decode-callback completion handoff: the filled result is cloned
    /// into the callback's out-parameter and the slot freed (the engine
    /// then completes the dispatch exactly once, as the C's in-callback
    /// `CompletePendingDecode` does).
    pub fn complete_for_callback(&mut self, index: usize, decode_result: &mut ServingDecodeResult) {
        if let Some(pending) = self.slots.get_mut(index) {
            decode_result.clone_from(&pending.result);
            pending.reset();
        }
    }

    /// `SparkRingServiceBackendDsparkDraft`: hand a captured draft to the
    /// speculator seam, consuming it.
    pub fn dspark_draft(
        &mut self,
        request_id: u64,
        sequence_id: u64,
        requested_token_count: u32,
    ) -> Result<DsparkDraftResult, ServingStatus> {
        if requested_token_count == 0 || requested_token_count as usize > DRAFT_TOKEN_CAPACITY {
            return Err(ServingStatus::InvalidArgument);
        }
        for index in 0..self.slots.len() {
            if !self.slots[index].active {
                continue;
            }
            let mut found_lane = None;
            for lane_index in 0..self.slots[index].dispatch.request_count as usize {
                if self.slots[index].dispatch.request_ids[lane_index] == request_id
                    && self.slots[index].sequence_ids[lane_index] == sequence_id
                    && self.slots[index].dspark_draft_valid[lane_index]
                {
                    found_lane = Some(lane_index);
                    break;
                }
            }
            let Some(lane_index) = found_lane else {
                continue;
            };
            let pending = &mut self.slots[index];
            let ready_draft = pending.dspark_drafts[lane_index];
            if requested_token_count > ready_draft.token_count {
                return Err(ServingStatus::InvalidArgument);
            }
            pending.dspark_draft_valid[lane_index] = false;
            let mut result = ready_draft;
            result.token_count = requested_token_count;
            for token_index in result.token_count as usize..DRAFT_TOKEN_CAPACITY {
                result.token_ids[token_index] = 0;
                result.confidence_milli[token_index] = 0;
            }
            pending.dspark_drafts[lane_index] = DsparkDraftResult::default();
            return Ok(result);
        }
        Err(ServingStatus::NotFound)
    }
}

/// `SparkRingServiceBackendTraceFinalEvent`.
fn trace_final_event(event: &FinalEvent, dispatch: &Dispatch) {
    eprint!(
        "ring_trace final_event request={} sequence={} position={} kind={:?} dispatch_flags=0x{:x} completion_flags=0x{:x} tokens={} ids=",
        event.request_id,
        event.sequence_id,
        event.sequence_position,
        dispatch.kind,
        dispatch.flags,
        event.completion_flags,
        event.token_count
    );
    for token_index in 0..event.token_count as usize {
        eprint!("{}{}", if token_index == 0 { "" } else { "," }, event.token_ids[token_index]);
    }
    eprintln!(" status={}", event.status);
}
