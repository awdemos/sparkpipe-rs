//! Request-API adapter — ports the C backend's direct
//! `SparkRequestApi*` calls onto the serving engine's [`ServingRequestApi`]
//! seam.
//!
//! In C the serving engine and the backend share one
//! `SparkRequestApiDispatch` struct (scheduler decisions included). The Rust
//! engine boundary deliberately carries a minimized DTO
//! ([`bridge::Dispatch`]) without the scheduler decisions or
//! `sequence_ids`; those are required by the concrete [`RequestApi`]
//! entry points. The adapter therefore keeps the full dispatch of every
//! schedule that has not yet completed, cancelled, or been retried, and
//! resolves each DTO the engine hands back to its full record by identity
//! (kind + lane handles + caller request ids).
//!
//! The engine owns the adapter behind `Box<dyn ServingRequestApi>`; the
//! backend keeps an `Rc<RefCell<_>>` clone so its pump-context failure paths
//! (`SparkRingServiceBackendFailPendingDecode`'s
//! `SparkRequestApiCancelDispatch`) reach the same instance. All borrows are
//! short and never cross an engine call, so the single-threaded `RefCell`
//! discipline cannot conflict (there is deliberately no async runtime here).

use std::cell::RefCell;
use std::rc::Rc;

use spark_serve::request_api::dispatch::{
    self as api, Dispatch as FullDispatch, DispatchKind as RequestApiDispatchKind,
};
use spark_serve::request_api::slot::SubmitRequest;
use spark_serve::request_api::{RequestApi, RequestApiError};
use spark_serve::serving_engine::bridge::{
    self, ApiSubmitRequest, Dispatch, DispatchKind, KvBlockTableView, PrefillDispatchLaneView,
    PrefillDispatchView, RequestApiCounters, RequestApiState, RequestCacheState, ServingRequestApi,
};
use spark_serve::serving_engine::ServingStatus;

fn bridge_kind(kind: RequestApiDispatchKind) -> DispatchKind {
    match kind {
        RequestApiDispatchKind::None => DispatchKind::None,
        RequestApiDispatchKind::Prefill => DispatchKind::Prefill,
        RequestApiDispatchKind::DecodeBatch => DispatchKind::DecodeBatch,
        RequestApiDispatchKind::PrefillBatch => DispatchKind::PrefillBatch,
        RequestApiDispatchKind::SpeculativeVerifyBatch => DispatchKind::SpeculativeVerifyBatch,
    }
}

/// Map the request-API error enum onto the flat status space (1:1 by
/// construction; both mirror `SparkStatus`).
fn map_error(error: RequestApiError) -> ServingStatus {
    match error {
        RequestApiError::InvalidArgument => ServingStatus::InvalidArgument,
        RequestApiError::CapacityExceeded => ServingStatus::CapacityExceeded,
        RequestApiError::NotFound => ServingStatus::NotFound,
        RequestApiError::Busy => ServingStatus::Busy,
        RequestApiError::InternalError => ServingStatus::InternalError,
        RequestApiError::ModuleNotValidated => ServingStatus::ModuleNotValidated,
        RequestApiError::HashMismatch => ServingStatus::HashMismatch,
    }
}

/// The adapter state shared between the engine (trait object) and the
/// backend (fail paths).
pub struct RequestApiAdapter {
    api: RequestApi,
    /// Full dispatches scheduled but not yet completed/cancelled/retried.
    inflight: Vec<FullDispatch>,
    inflight_capacity: usize,
}

/// The engine-facing handle: the `Box<dyn ServingRequestApi>` installed in
/// the serving engine configuration.
pub struct ServingRequestApiHandle {
    shared: Rc<RefCell<RequestApiAdapter>>,
}

/// [`RequestApiAdapter::lookup_extras`] result.
#[derive(Debug, Clone, Default)]
pub struct DispatchExtras {
    pub highest_priority: u32,
    pub prefill_decision_batch_bucket: u32,
    pub prefill_batch_decision_batch_bucket: u32,
    pub decode_batch_decision_batch_bucket: u32,
    pub sequence_ids: Vec<u64>,
    pub request_slot_indices: Vec<u32>,
}

/// Create the shared adapter plus the engine-facing trait object.
pub fn new_shared(
    api: RequestApi,
    inflight_capacity: usize,
) -> (Rc<RefCell<RequestApiAdapter>>, Box<dyn ServingRequestApi>) {
    let shared =
        Rc::new(RefCell::new(RequestApiAdapter { api, inflight: Vec::new(), inflight_capacity }));
    let handle = ServingRequestApiHandle { shared: Rc::clone(&shared) };
    (shared, Box::new(handle))
}

/// `SparkRequestApiDispatch` → DTO: exactly the fields the engine reads or
/// writes (the scheduler decisions stay behind the boundary, by design).
fn dto_from_full(full: &FullDispatch) -> Dispatch {
    let lane_capacity = bridge::MAX_DISPATCH_REQUEST_COUNT;
    let mut dto = Dispatch {
        accepted: full.accepted,
        kind: bridge_kind(full.kind),
        flags: full.flags,
        request_count: full.request_count,
        request_handles: full.request_handles.to_vec(),
        request_ids: full.request_ids.to_vec(),
        speculative_token_count: full.speculative_token_count,
        speculative_verifier_token_count: full.speculative_verifier_token_count,
        speculative_committed_token_counts: full.speculative_committed_token_counts.to_vec(),
        speculative_accepted_token_counts: full.speculative_accepted_token_counts.to_vec(),
        speculative_fallback_token_ids: full.speculative_fallback_token_ids.to_vec(),
        speculative_draft_token_ids: vec![0; lane_capacity * bridge::MAX_SPECULATIVE_TOKENS],
        mtp_draft_token_budget: full.mtp_draft_token_budget,
        decode_committed_token_counts: full.decode_committed_token_counts.to_vec(),
    };
    for lane in 0..lane_capacity {
        let base = lane * bridge::MAX_SPECULATIVE_TOKENS;
        dto.speculative_draft_token_ids[base..base + bridge::MAX_SPECULATIVE_TOKENS]
            .copy_from_slice(&full.speculative_draft_token_ids[lane]);
    }
    dto
}

/// Copy the engine-written DTO fields back onto the full dispatch before
/// delegating (completion outputs and any engine-clamped inputs).
fn sync_dto_into_full(dto: &Dispatch, full: &mut FullDispatch) {
    full.flags = dto.flags;
    full.speculative_token_count = dto.speculative_token_count;
    full.speculative_verifier_token_count = dto.speculative_verifier_token_count;
    full.speculative_committed_token_counts
        .copy_from_slice(&dto.speculative_committed_token_counts);
    full.speculative_accepted_token_counts.copy_from_slice(&dto.speculative_accepted_token_counts);
    full.speculative_fallback_token_ids.copy_from_slice(&dto.speculative_fallback_token_ids);
    full.decode_committed_token_counts.copy_from_slice(&dto.decode_committed_token_counts);
    for lane in 0..bridge::MAX_DISPATCH_REQUEST_COUNT {
        let base = lane * bridge::MAX_SPECULATIVE_TOKENS;
        full.speculative_draft_token_ids[lane].copy_from_slice(
            &dto.speculative_draft_token_ids[base..base + bridge::MAX_SPECULATIVE_TOKENS],
        );
    }
}

impl RequestApiAdapter {
    /// Concrete request-API access for the backend's initialization reads
    /// (batch width, counters).
    pub fn api(&mut self) -> &mut RequestApi {
        &mut self.api
    }

    /// Identity match: kind, lane count, per-lane request handles and caller
    /// request ids. In C the engine passes the same struct pointer back, so
    /// identity is pointer equality; the DTO boundary needs this instead.
    fn find_inflight(&self, dto: &Dispatch) -> Option<usize> {
        let lane_count = dto.request_count as usize;
        let dto_kind = request_api_kind(dto.kind);
        self.inflight.iter().position(|full| {
            full.kind == dto_kind
                && full.request_count == dto.request_count
                && full.request_handles[..lane_count] == dto.request_handles[..lane_count]
                && full.request_ids[..lane_count] == dto.request_ids[..lane_count]
        })
    }

    /// Clone the staged full dispatch matching the DTO. The packet builders
    /// (decode/prefill paths) need the priority, batch-bucket decisions,
    /// `sequence_ids`, and `request_slot_indices` the DTO omits (in C they
    /// read the same shared struct).
    pub fn lookup_full(&self, dto: &Dispatch) -> Result<FullDispatch, ServingStatus> {
        let index = self.find_inflight(dto).ok_or(ServingStatus::InvalidArgument)?;
        Ok(self.inflight[index].clone())
    }

    /// Prefill-callback lookup: the engine's prefill dispatch carries no
    /// request-dispatch DTO, so identity is matched by kind + per-lane
    /// request handles and caller request ids (pointer equality in C).
    pub fn lookup_full_by_lanes(
        &self,
        kinds: &[DispatchKind],
        request_handles: &[u64],
        request_ids: &[u64],
    ) -> Result<FullDispatch, ServingStatus> {
        let lane_count = request_handles.len();
        if lane_count == 0 || request_ids.len() != lane_count {
            return Err(ServingStatus::InvalidArgument);
        }
        self.inflight
            .iter()
            .find(|full| {
                kinds.iter().any(|kind| request_api_kind(*kind) == full.kind)
                    && full.request_count as usize == lane_count
                    && full.request_handles[..lane_count] == request_handles[..]
                    && full.request_ids[..lane_count] == request_ids[..]
            })
            .cloned()
            .ok_or(ServingStatus::InvalidArgument)
    }

    /// `SparkRequestApiScheduleNext` (+ DTO conversion).
    pub fn schedule_next(&mut self) -> Result<Dispatch, ServingStatus> {
        let mut full = FullDispatch::default();
        self.api.schedule_next(&mut full).map_err(map_error)?;
        if !full.accepted || full.kind == RequestApiDispatchKind::None {
            return Err(ServingStatus::InvalidArgument);
        }
        if self.inflight.len() >= self.inflight_capacity {
            // Unreachable in the C geometry (pending-decode capacity bounds
            // in-flight decodes and prefill completes synchronously); fail
            // loudly rather than corrupting identity tracking.
            return Err(ServingStatus::CapacityExceeded);
        }
        let dto = dto_from_full(&full);
        self.inflight.push(full);
        Ok(dto)
    }

    /// `SparkRequestApiCompleteDispatch`.
    pub fn complete_dispatch(&mut self, dto: &mut Dispatch) -> Result<(), ServingStatus> {
        let index = self.find_inflight(dto).ok_or(ServingStatus::InvalidArgument)?;
        let mut full = self.inflight.remove(index);
        sync_dto_into_full(dto, &mut full);
        self.api.complete_dispatch(&full).map_err(map_error)
    }

    /// `SparkRequestApiCancelDispatch`.
    pub fn cancel_dispatch(&mut self, dto: &mut Dispatch) -> Result<(), ServingStatus> {
        let index = self.find_inflight(dto).ok_or(ServingStatus::InvalidArgument)?;
        let full = self.inflight.remove(index);
        self.api.cancel_dispatch(&full).map_err(map_error)
    }

    /// `SparkRequestApiRetryDecodeDispatch` (the re-queued dispatch is
    /// re-scheduled from scratch, so its inflight record is dropped).
    pub fn retry_decode_dispatch(&mut self, dto: &mut Dispatch) -> Result<(), ServingStatus> {
        let index = self.find_inflight(dto).ok_or(ServingStatus::InvalidArgument)?;
        let full = self.inflight.remove(index);
        self.api.retry_decode_dispatch(&full).map_err(map_error)
    }

    /// The full-dispatch fields the packet builders need but the engine's
    /// DTO dropped (`SparkRequestApiDispatch` is shared in C): priority,
    /// the three decision batch buckets, and the per-lane sequence/slot
    /// indices. Looked up by the same identity rule as
    /// [`RequestApiAdapter::find_inflight`].
    pub fn lookup_extras(&self, dto: &Dispatch) -> Result<DispatchExtras, ServingStatus> {
        let index = self.find_inflight(dto).ok_or(ServingStatus::InvalidArgument)?;
        let full = &self.inflight[index];
        let lane_count = full.request_count as usize;
        Ok(DispatchExtras {
            highest_priority: full.highest_priority,
            prefill_decision_batch_bucket: full
                .prefill_decision
                .as_ref()
                .map(|decision| decision.batch_bucket)
                .unwrap_or(0),
            prefill_batch_decision_batch_bucket: full
                .prefill_batch_decision
                .as_ref()
                .map(|decision| decision.batch_bucket)
                .unwrap_or(0),
            decode_batch_decision_batch_bucket: full
                .decode_batch_decision
                .as_ref()
                .map(|decision| decision.batch_bucket)
                .unwrap_or(0),
            sequence_ids: full.sequence_ids[..lane_count].to_vec(),
            request_slot_indices: full.request_slot_indices[..lane_count].to_vec(),
        })
    }

    /// `SparkRequestApiDescribePrefillDispatch`.
    pub fn describe_prefill_dispatch(
        &mut self,
        dto: &Dispatch,
    ) -> Result<PrefillDispatchView, ServingStatus> {
        let index = self.find_inflight(dto).ok_or(ServingStatus::InvalidArgument)?;
        let view = api::describe_prefill_dispatch(&self.inflight[index]).map_err(map_error)?;
        Ok(PrefillDispatchView {
            kind: bridge_kind(view.kind),
            active_sequence_count: view.active_sequence_count,
            prompt_token_offset: view.prompt_token_offset,
            prompt_token_count: view.prompt_token_count,
            prompt_token_stride: view.prompt_token_stride,
            lane_count: view.lane_count,
            lanes: view
                .lanes
                .iter()
                .map(|lane| PrefillDispatchLaneView {
                    request_index: lane.request_index,
                    prompt_token_offset: lane.prompt_token_offset,
                    prompt_token_count: lane.prompt_token_count,
                    request_slot_index: lane.request_slot_index,
                    request_id: lane.request_id,
                    sequence_id: lane.sequence_id,
                    request_handle: lane.request_handle,
                })
                .collect(),
        })
    }

    /// `SparkRequestApiCopyPrefillDispatchTokenIds`.
    pub fn copy_prefill_dispatch_token_ids(
        &mut self,
        dto: &Dispatch,
        destination_token_ids: &mut [u32],
        destination_token_stride: u32,
        destination_lane_capacity: u32,
    ) -> Result<(), ServingStatus> {
        let index = self.find_inflight(dto).ok_or(ServingStatus::InvalidArgument)?;
        api::copy_prefill_dispatch_token_ids(
            &self.inflight[index],
            destination_token_ids,
            destination_token_stride,
            destination_lane_capacity,
        )
        .map_err(map_error)
    }

    /// `SparkRequestApiBuildDispatchKvBlockTableView`.
    #[allow(clippy::too_many_arguments)]
    pub fn build_dispatch_kv_block_table_view<'a>(
        &mut self,
        dto: &Dispatch,
        host_physical_block_indices: &'a mut [u32],
        execution_physical_block_indices: Option<&'a [u32]>,
        lane_stride: u32,
        lane_capacity: u32,
        lane_physical_block_counts: &'a mut [u32],
    ) -> Result<KvBlockTableView<'a>, ServingStatus> {
        let index = self.find_inflight(dto).ok_or(ServingStatus::InvalidArgument)?;
        // Borrow split: the inflight entry and the API are disjoint fields.
        let full = &self.inflight[index];
        let api = unsafe_full_borrow_helper(full);
        self.api
            .build_dispatch_kv_block_table_view(
                api,
                host_physical_block_indices,
                execution_physical_block_indices,
                lane_stride,
                lane_capacity,
                lane_physical_block_counts,
            )
            .map_err(map_error)
    }

    /// `SparkRequestApiDescribeDecodeDispatch`.
    pub fn describe_decode_dispatch(
        &mut self,
        dto: &Dispatch,
    ) -> Result<bridge::DecodeDispatchView, ServingStatus> {
        let index = self.find_inflight(dto).ok_or(ServingStatus::InvalidArgument)?;
        let view = self.api.describe_decode_dispatch(&self.inflight[index]).map_err(map_error)?;
        Ok(bridge::DecodeDispatchView {
            kind: bridge_kind(view.kind),
            active_sequence_count: view.active_sequence_count,
            lane_count: view.lane_count,
            speculative_token_count: view.speculative_token_count,
            lanes: view
                .lanes
                .iter()
                .map(|lane| bridge::DecodeDispatchLaneView {
                    request_index: lane.request_index,
                    sequence_position: lane.sequence_position,
                    context_token_count: lane.context_token_count,
                    request_slot_index: lane.request_slot_index,
                    request_id: lane.request_id,
                    sequence_id: lane.sequence_id,
                    request_handle: lane.request_handle,
                    mtp_resolution_base_position: lane.mtp_resolution_base_position,
                    mtp_resolution_proposed_token_count: lane.mtp_resolution_proposed_token_count,
                    mtp_resolution_accepted_token_count: lane.mtp_resolution_accepted_token_count,
                    mtp_resolution_committed_token_count: lane.mtp_resolution_committed_token_count,
                    mtp_resolution_path_id: lane.mtp_resolution_path_id,
                })
                .collect(),
        })
    }

    /// `SparkRequestApiResolveSpeculativeVerifyDispatch` (resolution lands
    /// on the full dispatch; the DTO-visible outputs are copied back).
    pub fn resolve_speculative_verify_dispatch(
        &mut self,
        dto: &mut Dispatch,
        verifier_token_ids: &[u32],
        verifier_lane_stride: u32,
        verifier_token_count: u32,
    ) -> Result<(), ServingStatus> {
        let index = self.find_inflight(dto).ok_or(ServingStatus::InvalidArgument)?;
        sync_dto_into_full(dto, &mut self.inflight[index]);
        let full = &mut self.inflight[index];
        self.api
            .resolve_speculative_verify_dispatch(
                full,
                verifier_token_ids,
                verifier_lane_stride,
                verifier_token_count,
            )
            .map_err(map_error)?;
        dto.speculative_committed_token_counts
            .copy_from_slice(&full.speculative_committed_token_counts);
        dto.speculative_accepted_token_counts
            .copy_from_slice(&full.speculative_accepted_token_counts);
        dto.speculative_fallback_token_ids.copy_from_slice(&full.speculative_fallback_token_ids);
        Ok(())
    }

    /// `SparkRequestApiArmMtpVerifyDispatch`.
    pub fn arm_mtp_verify_dispatch(
        &mut self,
        dto: &mut Dispatch,
        draft_token_ids: &[u32],
        draft_lane_stride: u32,
        draft_token_count: u32,
    ) -> Result<(), ServingStatus> {
        // The engine arms drafts for a dispatch that was just completed (and
        // thus already removed from inflight) — fall back to matching any
        // still-tracked record, else reconstruct nothing: the C entry point
        // reads only kind/flags/count/handles/budget, all DTO-carried.
        if let Some(index) = self.find_inflight(dto) {
            sync_dto_into_full(dto, &mut self.inflight[index]);
            let full = &self.inflight[index];
            return self
                .api
                .arm_mtp_verify_dispatch(
                    full,
                    draft_token_ids,
                    draft_lane_stride,
                    draft_token_count,
                )
                .map_err(map_error);
        }
        let full = full_from_dto(dto);
        self.api
            .arm_mtp_verify_dispatch(&full, draft_token_ids, draft_lane_stride, draft_token_count)
            .map_err(map_error)
    }
}

/// Reconstruct a decision-less full dispatch from the DTO for entry points
/// that read only DTO-carried fields (`ArmMtpVerifyDispatch` after the
/// completing dispatch already left the inflight table).
fn request_api_kind(kind: DispatchKind) -> RequestApiDispatchKind {
    match kind {
        DispatchKind::None => RequestApiDispatchKind::None,
        DispatchKind::Prefill => RequestApiDispatchKind::Prefill,
        DispatchKind::DecodeBatch => RequestApiDispatchKind::DecodeBatch,
        DispatchKind::PrefillBatch => RequestApiDispatchKind::PrefillBatch,
        DispatchKind::SpeculativeVerifyBatch => RequestApiDispatchKind::SpeculativeVerifyBatch,
    }
}

fn full_from_dto(dto: &Dispatch) -> FullDispatch {
    let mut full = FullDispatch {
        accepted: dto.accepted,
        kind: request_api_kind(dto.kind),
        flags: dto.flags,
        request_count: dto.request_count,
        speculative_token_count: dto.speculative_token_count,
        speculative_verifier_token_count: dto.speculative_verifier_token_count,
        mtp_draft_token_budget: dto.mtp_draft_token_budget,
        ..FullDispatch::default()
    };
    let lane_count = (dto.request_count as usize).min(bridge::MAX_DISPATCH_REQUEST_COUNT);
    full.request_handles[..lane_count].copy_from_slice(&dto.request_handles[..lane_count]);
    full.request_ids[..lane_count].copy_from_slice(&dto.request_ids[..lane_count]);
    full
}

/// Field-split borrow helper: `build_dispatch_kv_block_table_view` takes
/// `&mut self` on the API but only `&Dispatch` on the inflight entry, and
/// the two live in disjoint fields of the adapter. Reborrowing through a
/// shared reference keeps this safe-Rust; the function exists to make the
/// disjointness explicit at the call site.
fn unsafe_full_borrow_helper(full: &FullDispatch) -> &FullDispatch {
    full
}

impl ServingRequestApi for ServingRequestApiHandle {
    fn configuration_flags(&self) -> u32 {
        self.shared.borrow().api.configuration_flags
    }

    fn counters(&self) -> RequestApiCounters {
        let api = &self.shared.borrow().api;
        RequestApiCounters {
            queued_request_count: api.queued_request_count,
            completed_request_count: api.completed_request_count,
            cancelled_request_count: api.cancelled_request_count,
            jit_prefetch_dispatch_count: api.jit_prefetch_dispatch_count,
            jit_prefetch_block_count: api.jit_prefetch_block_count,
            async_jit_prefetch_start_count: api.async_jit_prefetch_start_count,
            async_jit_prefetch_poll_count: api.async_jit_prefetch_poll_count,
            async_jit_prefetch_completion_count: api.async_jit_prefetch_completion_count,
            prefix_family_dispatch_count: api.prefix_family_dispatch_count,
            prefix_family_member_count: api.prefix_family_member_count,
            prefix_family_saved_prompt_token_count: api.prefix_family_saved_prompt_token_count,
            mtp_draft_ready_count: api.mtp_draft_ready_count,
            mtp_accepted_draft_token_count: api.mtp_accepted_draft_token_count,
            mtp_committed_token_count: api.mtp_committed_token_count,
            mtp_rejected_token_count: api.mtp_rejected_token_count,
        }
    }

    fn submit(&mut self, request: &ApiSubmitRequest) -> Result<u64, ServingStatus> {
        self.shared
            .borrow_mut()
            .api
            .submit(&SubmitRequest {
                flags: request.flags,
                priority: request.priority,
                prompt_token_count: request.prompt_token_count,
                thinking_token_budget: request.thinking_token_budget,
                output_token_budget: request.output_token_budget,
                max_prefill_tokens_per_step: request.max_prefill_tokens_per_step,
                request_id: request.request_id,
                sequence_id: request.sequence_id,
                prompt_token_ids: request.prompt_token_ids.to_vec(),
            })
            .map_err(map_error)
    }

    fn request_cache_state(&self, handle: u64) -> Result<RequestCacheState, ServingStatus> {
        let cache_state =
            self.shared.borrow_mut().api.get_request_cache_state(handle).map_err(map_error)?;
        let state = match cache_state.state {
            0 => RequestApiState::Free,
            1 => RequestApiState::QueuedPrefill,
            2 => RequestApiState::RunningPrefill,
            3 => RequestApiState::ReadyDecode,
            4 => RequestApiState::RunningDecode,
            5 => RequestApiState::Completed,
            6 => RequestApiState::Cancelled,
            7 => RequestApiState::WaitingPrefixCohort,
            8 => RequestApiState::ReadySpeculativeVerify,
            9 => RequestApiState::RunningSpeculativeVerify,
            _ => return Err(ServingStatus::InternalError),
        };
        Ok(RequestCacheState {
            state,
            request_id: cache_state.request_id,
            sequence_id: cache_state.sequence_id,
        })
    }

    fn cancel_request(&mut self, handle: u64) -> Result<(), ServingStatus> {
        self.shared.borrow_mut().api.cancel_request(handle).map_err(map_error)
    }

    fn release_completed_request(&mut self, handle: u64) -> Result<(), ServingStatus> {
        self.shared.borrow_mut().api.release_completed_request(handle).map_err(map_error)
    }

    fn finish_request_generation(&mut self, handle: u64) -> Result<(), ServingStatus> {
        self.shared.borrow_mut().api.finish_request_generation(handle).map_err(map_error)
    }

    fn schedule_next(&mut self) -> Result<Dispatch, ServingStatus> {
        self.shared.borrow_mut().schedule_next()
    }

    fn describe_prefill_dispatch(
        &self,
        dispatch: &Dispatch,
    ) -> Result<PrefillDispatchView, ServingStatus> {
        self.shared.borrow_mut().describe_prefill_dispatch(dispatch)
    }

    fn copy_prefill_dispatch_token_ids(
        &self,
        dispatch: &Dispatch,
        destination_token_ids: &mut [u32],
        destination_token_stride: u32,
        destination_lane_capacity: u32,
    ) -> Result<(), ServingStatus> {
        self.shared.borrow_mut().copy_prefill_dispatch_token_ids(
            dispatch,
            destination_token_ids,
            destination_token_stride,
            destination_lane_capacity,
        )
    }

    fn build_dispatch_kv_block_table_view<'a>(
        &self,
        dispatch: &Dispatch,
        host_physical_block_indices: &'a mut [u32],
        execution_physical_block_indices: Option<&'a [u32]>,
        lane_stride: u32,
        lane_capacity: u32,
        lane_physical_block_counts: &'a mut [u32],
    ) -> Result<KvBlockTableView<'a>, ServingStatus> {
        self.shared.borrow_mut().build_dispatch_kv_block_table_view(
            dispatch,
            host_physical_block_indices,
            execution_physical_block_indices,
            lane_stride,
            lane_capacity,
            lane_physical_block_counts,
        )
    }

    fn describe_decode_dispatch(
        &self,
        dispatch: &Dispatch,
    ) -> Result<bridge::DecodeDispatchView, ServingStatus> {
        self.shared.borrow_mut().describe_decode_dispatch(dispatch)
    }

    fn resolve_speculative_verify_dispatch(
        &mut self,
        dispatch: &mut Dispatch,
        verifier_token_ids: &[u32],
        verifier_lane_stride: u32,
        verifier_token_count: u32,
    ) -> Result<(), ServingStatus> {
        self.shared.borrow_mut().resolve_speculative_verify_dispatch(
            dispatch,
            verifier_token_ids,
            verifier_lane_stride,
            verifier_token_count,
        )
    }

    fn complete_dispatch(&mut self, dispatch: &mut Dispatch) -> Result<(), ServingStatus> {
        self.shared.borrow_mut().complete_dispatch(dispatch)
    }

    fn arm_mtp_verify_dispatch(
        &mut self,
        dispatch: &mut Dispatch,
        draft_token_ids: &[u32],
        draft_lane_stride: u32,
        draft_token_count: u32,
    ) -> Result<(), ServingStatus> {
        self.shared.borrow_mut().arm_mtp_verify_dispatch(
            dispatch,
            draft_token_ids,
            draft_lane_stride,
            draft_token_count,
        )
    }

    fn retry_decode_dispatch(&mut self, dispatch: &mut Dispatch) -> Result<(), ServingStatus> {
        self.shared.borrow_mut().retry_decode_dispatch(dispatch)
    }

    fn cancel_dispatch(&mut self, dispatch: &mut Dispatch) -> Result<(), ServingStatus> {
        self.shared.borrow_mut().cancel_dispatch(dispatch)
    }
}
