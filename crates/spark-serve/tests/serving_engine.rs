//! Integration tests for the `serving_engine` port — a GPU-free port of
//! `tests/test_glm52_serving_engine.c`, `tests/test_glm52_service.c`, and
//! `tests/test_glm52_compat_api.c`.
//!
//! The C tests drive a real `SparkRequestApi` (scheduler + KV arena + prefix
//! cache). Here that concrete stack is replaced by [`FakeRequestApi`], a
//! scripted implementation of the [`ServingRequestApi`] boundary trait that
//! reproduces the exact scheduling behavior the C scenarios rely on:
//! 64-token prefill steps, prefill batching, decode batches of a configurable
//! target size, speculative-verify dispatches fed by armed MTP drafts, and
//! the `SparkMtpTreeResolve` acceptance algorithm from
//! `include/sparkpipe/spark_mtp_tree.h`.

use std::cell::RefCell;
use std::rc::Rc;

use spark_serve::serving_engine::bridge::{
    self, ApiSubmitRequest, DecodeDispatchLaneView, DecodeDispatchView, Dispatch, KvBlockTableView,
    PrefillDispatchLaneView, PrefillDispatchView, RequestApiCounters, RequestApiState,
    RequestCacheState, ServingRequestApi, CONFIGURATION_FLAG_DECODE_BATCHING,
    CONFIGURATION_FLAG_JIT_KV_PREFETCH, CONFIGURATION_FLAG_MTP_COMMIT,
    CONFIGURATION_FLAG_PREFILL_BATCHING, DISPATCH_FLAG_MTP_COMMIT,
    DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY, DISPATCH_FLAG_MTP_TREE_VERIFY,
    MAX_DECODE_TOKENS_PER_LANE, MAX_DISPATCH_REQUEST_COUNT,
};
use spark_serve::serving_engine::compat::{prepare_anthropic_json, prepare_openai_json};
use spark_serve::serving_engine::engine::{
    self, ServingDecodeResult, ServingEngine, ServingEngineConfiguration, ServingEventKind,
    ServingStats, SubmitTokenIdsRequest, ENGINE_DEFAULT_FLAGS,
    ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE, PUMP_FLAG_STOP_AFTER_ONE_DISPATCH,
    RUNTIME_CONTRACT_FLAG_INDEXSHARE_STAGE_BOUNDARY_STATE,
    RUNTIME_CONTRACT_FLAG_JIT_KV_PREFETCH_CONNECTED,
    RUNTIME_CONTRACT_FLAG_OVERLAPPED_STAGING_READY,
    RUNTIME_CONTRACT_FLAG_TAIL_WINDOW_VALIDATION_ONLY, RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS,
};
use spark_serve::serving_engine::service::{
    self, ServiceRuntime, ServiceStats, ServiceSubmitTokenIdsRequest, FRAME_KIND_SUBMIT_TOKEN_IDS,
    FRAME_SUBMIT_TOKENS_BODY_BYTES, SERVICE_ABI_VERSION,
};
use spark_serve::serving_engine::{CompatTextRequest, ServiceEventKind, ServingStatus};
use spark_text::prompt_pipeline::DispatchKind;

// ---------------------------------------------------------------------------
// Fixture constants (mirror the C test #defines).
// ---------------------------------------------------------------------------

const REQUEST_SLOT_COUNT: usize = 4;
const REQUEST_RECORD_CAPACITY: u32 = 4;
const REQUEST_TOKEN_STRIDE: u32 = 128;
const PREFILL_TOKEN_STRIDE: u32 = 64;
const PROMPT_TOKEN_COUNT: usize = 97;
const EVENT_CAPACITY: u32 = 16384;
const KV_BLOCK_TOKENS: u32 = 16;
const KV_BLOCK_LANE_STRIDE: u32 = 64;
const MAX_CONTEXT_TOKENS: u32 = 256;
const MTP_STOP_TOKEN_ID: u32 = 90008;

const MTP_TREE_CANDIDATE_COUNT: usize = 5;
const MTP_TREE_VERIFIER_ROW_COUNT: usize = 6;
const MTP_CANDIDATE_COUNT: usize = MTP_TREE_CANDIDATE_COUNT;
const MTP_VERIFIER_ROW_COUNT: usize = MTP_TREE_VERIFIER_ROW_COUNT;

/// `SPARK_REQUEST_API_CONFIGURATION_FLAG_MTP_FORCE_ENABLE`.
const CONFIGURATION_FLAG_MTP_FORCE_ENABLE: u32 = 0x0000_0200;

const FAKE_CONFIGURATION_FLAGS: u32 = CONFIGURATION_FLAG_JIT_KV_PREFETCH
    | CONFIGURATION_FLAG_DECODE_BATCHING
    | CONFIGURATION_FLAG_PREFILL_BATCHING;

/// Runtime contract flags used by the fixture configurations (the C test
/// fixture's production contract plus JIT-prefetch and staging bits).
const TEST_RUNTIME_CONTRACT_FLAGS: u32 = RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS
    | RUNTIME_CONTRACT_FLAG_JIT_KV_PREFETCH_CONNECTED
    | RUNTIME_CONTRACT_FLAG_OVERLAPPED_STAGING_READY;

/// Per-lane KV block-index capacity used for the staging arrays (scaled down
/// from the C fixture's full block-table capacity).
const KV_BLOCK_LANE_CAPACITY: u32 = KV_BLOCK_LANE_STRIDE;

fn fill_token_ids(token_count: usize, first_token_id: u32) -> Vec<u32> {
    (0..token_count as u32).map(|index| first_token_id + index).collect()
}

// ---------------------------------------------------------------------------
// MTP tree resolution (port of `SparkMtpTreeResolve`).
// ---------------------------------------------------------------------------

struct MtpTreeNode {
    parent_row: usize,
    depth: u32,
    candidate_index: usize,
    child_row_base: usize,
    child_count: usize,
}

const MTP_TREE_NODES: [MtpTreeNode; MTP_TREE_VERIFIER_ROW_COUNT] = [
    MtpTreeNode { parent_row: 0, depth: 0, candidate_index: 0, child_row_base: 1, child_count: 1 },
    MtpTreeNode { parent_row: 0, depth: 1, candidate_index: 0, child_row_base: 2, child_count: 2 },
    MtpTreeNode { parent_row: 1, depth: 2, candidate_index: 1, child_row_base: 4, child_count: 2 },
    MtpTreeNode { parent_row: 1, depth: 2, candidate_index: 2, child_row_base: 0, child_count: 0 },
    MtpTreeNode { parent_row: 2, depth: 3, candidate_index: 3, child_row_base: 0, child_count: 0 },
    MtpTreeNode { parent_row: 2, depth: 3, candidate_index: 4, child_row_base: 0, child_count: 0 },
];

/// `SparkMtpTreeResolve`: returns `(accepted_token_count,
/// committed_token_count, fallback_token_id)`.
fn mtp_tree_resolve(candidate_token_ids: &[u32], verifier_token_ids: &[u32]) -> (u32, u32, u32) {
    let mut current_row = 0usize;
    let mut path_id = 0usize;
    let mut matched = true;
    while matched {
        matched = false;
        let node = &MTP_TREE_NODES[current_row];
        for child_offset in 0..node.child_count {
            let child = &MTP_TREE_NODES[node.child_row_base + child_offset];
            if verifier_token_ids[current_row] != candidate_token_ids[child.candidate_index] {
                continue;
            }
            current_row = node.child_row_base + child_offset;
            path_id = current_row;
            matched = true;
            break;
        }
    }
    let _ = MTP_TREE_NODES[path_id].parent_row;
    let accepted_token_count = MTP_TREE_NODES[path_id].depth;
    let committed_token_count = accepted_token_count + 1;
    let fallback_token_id = verifier_token_ids[path_id];
    (accepted_token_count, committed_token_count, fallback_token_id)
}

// ---------------------------------------------------------------------------
// FakeRequestApi: scripted `ServingRequestApi` implementation.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct FakeSlot {
    state: RequestApiState,
    handle: u64,
    request_id: u64,
    sequence_id: u64,
    prompt_token_ids: Vec<u32>,
    output_token_budget: u32,
    prefilled_token_count: u32,
    generated_token_count: u32,
    pending_draft_token_ids: Vec<u32>,
}

impl FakeSlot {
    fn free() -> Self {
        FakeSlot {
            state: RequestApiState::Free,
            handle: 0,
            request_id: 0,
            sequence_id: 0,
            prompt_token_ids: Vec::new(),
            output_token_budget: 0,
            prefilled_token_count: 0,
            generated_token_count: 0,
            pending_draft_token_ids: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct FakeState {
    configuration_flags: u32,
    decode_batch_target: u32,
    slots: Vec<FakeSlot>,
    next_handle: u64,
    next_sequence_id: u64,
    running_request_count: u32,
    counters: RequestApiCounters,
}

#[derive(Clone)]
struct FakeRequestApi {
    state: Rc<RefCell<FakeState>>,
}

impl FakeRequestApi {
    fn new() -> Self {
        FakeRequestApi {
            state: Rc::new(RefCell::new(FakeState {
                configuration_flags: FAKE_CONFIGURATION_FLAGS,
                decode_batch_target: 2,
                slots: (0..REQUEST_SLOT_COUNT).map(|_| FakeSlot::free()).collect(),
                next_handle: 1,
                next_sequence_id: 7000,
                running_request_count: 0,
                counters: RequestApiCounters::default(),
            })),
        }
    }

    /// Mirrors the C tests poking `request_api.configuration_flags`.
    fn set_configuration_flags(&self, configuration_flags: u32) {
        self.state.borrow_mut().configuration_flags = configuration_flags;
    }

    /// Mirrors the C MTP test setting `MTP_COMMIT | MTP_FORCE_ENABLE`.
    fn enable_mtp(&self) {
        self.state.borrow_mut().configuration_flags |=
            CONFIGURATION_FLAG_MTP_COMMIT | CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    }

    /// Mirrors the C tests assigning `request_api.decode_batch_target`.
    fn set_decode_batch_target(&self, decode_batch_target: u32) {
        self.state.borrow_mut().decode_batch_target = decode_batch_target;
    }

    /// Slot state for assertions (`request_slots[i].state` in C).
    fn slot_state(&self, slot_index: usize) -> RequestApiState {
        self.state.borrow().slots[slot_index].state
    }

    /// `request_api.running_request_count` in C.
    fn running_request_count(&self) -> u32 {
        self.state.borrow().running_request_count
    }
}

fn find_slot(state: &FakeState, handle: u64) -> Option<usize> {
    state.slots.iter().position(|slot| slot.handle == handle && slot.state != RequestApiState::Free)
}

impl ServingRequestApi for FakeRequestApi {
    fn configuration_flags(&self) -> u32 {
        self.state.borrow().configuration_flags
    }

    fn counters(&self) -> RequestApiCounters {
        self.state.borrow().counters
    }

    fn submit(&mut self, request: &ApiSubmitRequest) -> Result<u64, ServingStatus> {
        let mut state = self.state.borrow_mut();
        let Some(slot_index) =
            state.slots.iter().position(|slot| slot.state == RequestApiState::Free)
        else {
            return Err(ServingStatus::CapacityExceeded);
        };
        let handle = state.next_handle;
        state.next_handle += 1;
        let sequence_id = if request.sequence_id != 0 {
            request.sequence_id
        } else {
            state.next_sequence_id += 1;
            state.next_sequence_id
        };
        state.slots[slot_index] = FakeSlot {
            state: RequestApiState::QueuedPrefill,
            handle,
            request_id: request.request_id,
            sequence_id,
            prompt_token_ids: request.prompt_token_ids.to_vec(),
            output_token_budget: request.output_token_budget,
            prefilled_token_count: 0,
            generated_token_count: 0,
            pending_draft_token_ids: Vec::new(),
        };
        state.counters.queued_request_count += 1;
        Ok(handle)
    }

    fn request_cache_state(&self, handle: u64) -> Result<RequestCacheState, ServingStatus> {
        let state = self.state.borrow();
        let Some(slot_index) = find_slot(&state, handle) else {
            return Err(ServingStatus::NotFound);
        };
        let slot = &state.slots[slot_index];
        Ok(RequestCacheState {
            state: slot.state,
            request_id: slot.request_id,
            sequence_id: slot.sequence_id,
        })
    }

    fn cancel_request(&mut self, handle: u64) -> Result<(), ServingStatus> {
        let mut state = self.state.borrow_mut();
        let Some(slot_index) = find_slot(&state, handle) else {
            return Err(ServingStatus::NotFound);
        };
        state.slots[slot_index].state = RequestApiState::Cancelled;
        state.counters.cancelled_request_count += 1;
        Ok(())
    }

    fn release_completed_request(&mut self, handle: u64) -> Result<(), ServingStatus> {
        let mut state = self.state.borrow_mut();
        let Some(slot_index) = find_slot(&state, handle) else {
            return Err(ServingStatus::NotFound);
        };
        state.slots[slot_index] = FakeSlot::free();
        Ok(())
    }

    fn finish_request_generation(&mut self, handle: u64) -> Result<(), ServingStatus> {
        let mut state = self.state.borrow_mut();
        let Some(slot_index) = find_slot(&state, handle) else {
            return Err(ServingStatus::NotFound);
        };
        state.slots[slot_index].state = RequestApiState::Completed;
        state.counters.completed_request_count += 1;
        Ok(())
    }

    fn schedule_next(&mut self) -> Result<Dispatch, ServingStatus> {
        let mut state = self.state.borrow_mut();

        // Prefill: batch every queued-prefill slot into one dispatch.
        let prefill_lanes: Vec<usize> = state
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.state == RequestApiState::QueuedPrefill)
            .map(|(index, _)| index)
            .collect();
        if !prefill_lanes.is_empty() {
            let mut dispatch = Dispatch::new();
            dispatch.accepted = true;
            dispatch.kind = if prefill_lanes.len() == 1 {
                DispatchKind::Prefill
            } else {
                DispatchKind::PrefillBatch
            };
            dispatch.request_count = prefill_lanes.len() as u32;
            for (lane_index, slot_index) in prefill_lanes.iter().enumerate() {
                let slot = &mut state.slots[*slot_index];
                dispatch.request_handles[lane_index] = slot.handle;
                dispatch.request_ids[lane_index] = slot.request_id;
                slot.state = RequestApiState::RunningPrefill;
            }
            return Ok(dispatch);
        }

        // Speculative verify: every slot with armed drafts.
        let verify_lanes: Vec<usize> = state
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.state == RequestApiState::ReadySpeculativeVerify)
            .map(|(index, _)| index)
            .collect();
        if !verify_lanes.is_empty() {
            let mut dispatch = Dispatch::new();
            dispatch.accepted = true;
            dispatch.kind = DispatchKind::SpeculativeVerifyBatch;
            dispatch.flags = DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY | DISPATCH_FLAG_MTP_TREE_VERIFY;
            dispatch.request_count = verify_lanes.len() as u32;
            dispatch.speculative_token_count = MTP_TREE_CANDIDATE_COUNT as u32;
            dispatch.speculative_verifier_token_count = MTP_TREE_VERIFIER_ROW_COUNT as u32;
            for (lane_index, slot_index) in verify_lanes.iter().enumerate() {
                let slot = &mut state.slots[*slot_index];
                dispatch.request_handles[lane_index] = slot.handle;
                dispatch.request_ids[lane_index] = slot.request_id;
                for (draft_index, draft_token_id) in slot.pending_draft_token_ids.iter().enumerate()
                {
                    dispatch.speculative_draft_token_ids
                        [lane_index * bridge::MAX_SPECULATIVE_TOKENS + draft_index] =
                        *draft_token_id;
                }
                slot.state = RequestApiState::RunningSpeculativeVerify;
            }
            state.running_request_count += verify_lanes.len() as u32;
            return Ok(dispatch);
        }

        // Decode batch: up to `decode_batch_target` ready slots.
        let decode_lanes: Vec<usize> = state
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.state == RequestApiState::ReadyDecode)
            .take(state.decode_batch_target as usize)
            .map(|(index, _)| index)
            .collect();
        if !decode_lanes.is_empty() {
            let mut dispatch = Dispatch::new();
            dispatch.accepted = true;
            dispatch.kind = DispatchKind::DecodeBatch;
            if (state.configuration_flags & CONFIGURATION_FLAG_MTP_COMMIT) != 0
                && (state.configuration_flags & CONFIGURATION_FLAG_MTP_FORCE_ENABLE) != 0
            {
                dispatch.flags = DISPATCH_FLAG_MTP_COMMIT;
                dispatch.mtp_draft_token_budget = MTP_TREE_CANDIDATE_COUNT as u32;
            }
            dispatch.request_count = decode_lanes.len() as u32;
            for (lane_index, slot_index) in decode_lanes.iter().enumerate() {
                let slot = &mut state.slots[*slot_index];
                dispatch.request_handles[lane_index] = slot.handle;
                dispatch.request_ids[lane_index] = slot.request_id;
                slot.state = RequestApiState::RunningDecode;
            }
            state.running_request_count += decode_lanes.len() as u32;
            return Ok(dispatch);
        }

        Err(ServingStatus::NotFound)
    }

    fn describe_prefill_dispatch(
        &self,
        dispatch: &Dispatch,
    ) -> Result<PrefillDispatchView, ServingStatus> {
        let state = self.state.borrow();
        let mut lanes = Vec::with_capacity(dispatch.request_count as usize);
        let mut first_offset = 0;
        let mut first_count = 0;
        for lane_index in 0..dispatch.request_count as usize {
            let Some(slot_index) = find_slot(&state, dispatch.request_handles[lane_index]) else {
                return Err(ServingStatus::NotFound);
            };
            let slot = &state.slots[slot_index];
            let prompt_token_count = slot.prompt_token_ids.len() as u32;
            let remaining = prompt_token_count - slot.prefilled_token_count;
            let step_token_count = remaining.min(PREFILL_TOKEN_STRIDE);
            if lane_index == 0 {
                first_offset = slot.prefilled_token_count;
                first_count = step_token_count;
            }
            lanes.push(PrefillDispatchLaneView {
                request_index: lane_index as u32,
                prompt_token_offset: slot.prefilled_token_count,
                prompt_token_count: step_token_count,
                request_slot_index: slot_index as u32,
                request_id: slot.request_id,
                sequence_id: slot.sequence_id,
                request_handle: slot.handle,
            });
        }
        Ok(PrefillDispatchView {
            kind: dispatch.kind,
            active_sequence_count: dispatch.request_count,
            prompt_token_offset: first_offset,
            prompt_token_count: first_count,
            prompt_token_stride: PREFILL_TOKEN_STRIDE,
            lane_count: dispatch.request_count,
            lanes,
        })
    }

    fn copy_prefill_dispatch_token_ids(
        &self,
        dispatch: &Dispatch,
        destination_token_ids: &mut [u32],
        destination_token_stride: u32,
        destination_lane_capacity: u32,
    ) -> Result<(), ServingStatus> {
        let state = self.state.borrow();
        if dispatch.request_count > destination_lane_capacity {
            return Err(ServingStatus::CapacityExceeded);
        }
        for lane_index in 0..dispatch.request_count as usize {
            let Some(slot_index) = find_slot(&state, dispatch.request_handles[lane_index]) else {
                return Err(ServingStatus::NotFound);
            };
            let slot = &state.slots[slot_index];
            let offset = slot.prefilled_token_count as usize;
            let step_token_count =
                (slot.prompt_token_ids.len() - offset).min(PREFILL_TOKEN_STRIDE as usize);
            let lane_base = lane_index * destination_token_stride as usize;
            destination_token_ids[lane_base..lane_base + step_token_count]
                .copy_from_slice(&slot.prompt_token_ids[offset..offset + step_token_count]);
            for word in &mut destination_token_ids
                [lane_base + step_token_count..lane_base + destination_token_stride as usize]
            {
                *word = 0;
            }
        }
        Ok(())
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
        let lane_count = dispatch.request_count as usize;
        if lane_count * lane_stride as usize > host_physical_block_indices.len()
            || lane_count > lane_physical_block_counts.len()
        {
            return Err(ServingStatus::CapacityExceeded);
        }
        let _ = execution_physical_block_indices;
        let _ = lane_capacity;
        for (lane_index, block_count) in
            lane_physical_block_counts.iter_mut().take(lane_count).enumerate()
        {
            let lane_base = lane_index * lane_stride as usize;
            for block_index in 0..4u32 {
                host_physical_block_indices[lane_base + block_index as usize] =
                    (lane_index as u32) * 1000 + block_index + 1;
            }
            *block_count = 4;
        }
        Ok(KvBlockTableView {
            block_token_count: KV_BLOCK_TOKENS,
            lane_count: dispatch.request_count,
            lane_stride,
            lane_capacity,
            physical_block_indices: None,
            lane_physical_block_counts: None,
            host_physical_block_indices: &host_physical_block_indices
                [..lane_count * lane_stride as usize],
            host_lane_physical_block_counts: &lane_physical_block_counts[..lane_count],
        })
    }

    fn describe_decode_dispatch(
        &self,
        dispatch: &Dispatch,
    ) -> Result<DecodeDispatchView, ServingStatus> {
        let state = self.state.borrow();
        let mut lanes = Vec::with_capacity(dispatch.request_count as usize);
        for lane_index in 0..dispatch.request_count as usize {
            let Some(slot_index) = find_slot(&state, dispatch.request_handles[lane_index]) else {
                return Err(ServingStatus::NotFound);
            };
            let slot = &state.slots[slot_index];
            let context_token_count =
                slot.prompt_token_ids.len() as u32 + slot.generated_token_count;
            lanes.push(DecodeDispatchLaneView {
                request_index: lane_index as u32,
                sequence_position: context_token_count - 1,
                context_token_count,
                request_slot_index: slot_index as u32,
                request_id: slot.request_id,
                sequence_id: slot.sequence_id,
                request_handle: slot.handle,
                mtp_resolution_base_position: 0,
                mtp_resolution_proposed_token_count: 0,
                mtp_resolution_accepted_token_count: 0,
                mtp_resolution_committed_token_count: 0,
                mtp_resolution_path_id: 0,
            });
        }
        Ok(DecodeDispatchView {
            kind: dispatch.kind,
            active_sequence_count: dispatch.request_count,
            lane_count: dispatch.request_count,
            speculative_token_count: dispatch.speculative_token_count,
            lanes,
        })
    }

    fn resolve_speculative_verify_dispatch(
        &mut self,
        dispatch: &mut Dispatch,
        verifier_token_ids: &[u32],
        verifier_lane_stride: u32,
        verifier_token_count: u32,
    ) -> Result<(), ServingStatus> {
        let mut state = self.state.borrow_mut();
        for lane_index in 0..dispatch.request_count as usize {
            let Some(slot_index) = find_slot(&state, dispatch.request_handles[lane_index]) else {
                return Err(ServingStatus::NotFound);
            };
            let slot = &mut state.slots[slot_index];
            let candidate_token_ids = slot.pending_draft_token_ids.clone();
            if candidate_token_ids.len() != MTP_TREE_CANDIDATE_COUNT {
                return Err(ServingStatus::InvalidArgument);
            }
            let verifier_base = lane_index * verifier_lane_stride as usize;
            let lane_verifier =
                &verifier_token_ids[verifier_base..verifier_base + verifier_token_count as usize];
            let (accepted_token_count, committed_token_count, fallback_token_id) =
                mtp_tree_resolve(&candidate_token_ids, lane_verifier);
            dispatch.speculative_accepted_token_counts[lane_index] = accepted_token_count;
            dispatch.speculative_committed_token_counts[lane_index] = committed_token_count;
            dispatch.speculative_fallback_token_ids[lane_index] = fallback_token_id;
            slot.pending_draft_token_ids.clear();
            state.counters.mtp_accepted_draft_token_count += accepted_token_count as u64;
            state.counters.mtp_committed_token_count += committed_token_count as u64;
            state.counters.mtp_rejected_token_count +=
                (MTP_TREE_CANDIDATE_COUNT as u32 - accepted_token_count) as u64;
        }
        Ok(())
    }

    fn complete_dispatch(&mut self, dispatch: &mut Dispatch) -> Result<(), ServingStatus> {
        let mut state = self.state.borrow_mut();
        match dispatch.kind {
            DispatchKind::Prefill | DispatchKind::PrefillBatch => {
                for lane_index in 0..dispatch.request_count as usize {
                    let Some(slot_index) = find_slot(&state, dispatch.request_handles[lane_index])
                    else {
                        return Err(ServingStatus::NotFound);
                    };
                    let slot = &mut state.slots[slot_index];
                    let prompt_token_count = slot.prompt_token_ids.len() as u32;
                    let step_token_count =
                        (prompt_token_count - slot.prefilled_token_count).min(PREFILL_TOKEN_STRIDE);
                    slot.prefilled_token_count += step_token_count;
                    if slot.prefilled_token_count == prompt_token_count {
                        slot.state = RequestApiState::ReadyDecode;
                        state.counters.queued_request_count =
                            state.counters.queued_request_count.saturating_sub(1);
                    } else {
                        slot.state = RequestApiState::QueuedPrefill;
                    }
                }
            }
            DispatchKind::DecodeBatch | DispatchKind::SpeculativeVerifyBatch => {
                for lane_index in 0..dispatch.request_count as usize {
                    let Some(slot_index) = find_slot(&state, dispatch.request_handles[lane_index])
                    else {
                        return Err(ServingStatus::NotFound);
                    };
                    let committed_token_count =
                        if dispatch.kind == DispatchKind::SpeculativeVerifyBatch {
                            dispatch.speculative_committed_token_counts[lane_index]
                        } else if (dispatch.flags & DISPATCH_FLAG_MTP_COMMIT) != 0 {
                            dispatch.decode_committed_token_counts[lane_index]
                        } else {
                            1
                        };
                    let slot = &mut state.slots[slot_index];
                    slot.generated_token_count += committed_token_count;
                    if slot.generated_token_count >= slot.output_token_budget {
                        // The C request API retires the slot when the output
                        // budget is consumed; the engine's budget-finished
                        // sweep then publishes the completion.
                        if slot.state != RequestApiState::Completed {
                            slot.state = RequestApiState::Completed;
                            state.counters.completed_request_count += 1;
                        }
                    } else if matches!(
                        slot.state,
                        RequestApiState::RunningDecode | RequestApiState::RunningSpeculativeVerify
                    ) {
                        slot.state = RequestApiState::ReadyDecode;
                    }
                }
                state.running_request_count =
                    state.running_request_count.saturating_sub(dispatch.request_count);
            }
            DispatchKind::None => return Err(ServingStatus::InvalidArgument),
        }
        Ok(())
    }

    fn arm_mtp_verify_dispatch(
        &mut self,
        dispatch: &mut Dispatch,
        draft_token_ids: &[u32],
        draft_lane_stride: u32,
        draft_token_count: u32,
    ) -> Result<(), ServingStatus> {
        if draft_token_count == 0 {
            return Err(ServingStatus::NotFound);
        }
        let mut state = self.state.borrow_mut();
        let mut armed_lane_count = 0u32;
        for lane_index in 0..dispatch.request_count as usize {
            let Some(slot_index) = find_slot(&state, dispatch.request_handles[lane_index]) else {
                continue;
            };
            let slot = &mut state.slots[slot_index];
            if slot.state != RequestApiState::ReadyDecode {
                continue;
            }
            let draft_base = lane_index * draft_lane_stride as usize;
            slot.pending_draft_token_ids =
                draft_token_ids[draft_base..draft_base + draft_token_count as usize].to_vec();
            slot.state = RequestApiState::ReadySpeculativeVerify;
            armed_lane_count += 1;
        }
        if armed_lane_count == 0 {
            return Err(ServingStatus::NotFound);
        }
        state.counters.mtp_draft_ready_count += 1;
        Ok(())
    }

    fn retry_decode_dispatch(&mut self, dispatch: &mut Dispatch) -> Result<(), ServingStatus> {
        let mut state = self.state.borrow_mut();
        for lane_index in 0..dispatch.request_count as usize {
            let Some(slot_index) = find_slot(&state, dispatch.request_handles[lane_index]) else {
                return Err(ServingStatus::NotFound);
            };
            let slot = &mut state.slots[slot_index];
            if slot.state == RequestApiState::RunningDecode {
                slot.state = RequestApiState::ReadyDecode;
            }
        }
        state.running_request_count =
            state.running_request_count.saturating_sub(dispatch.request_count);
        Ok(())
    }

    fn cancel_dispatch(&mut self, dispatch: &mut Dispatch) -> Result<(), ServingStatus> {
        let mut state = self.state.borrow_mut();
        for lane_index in 0..dispatch.request_count as usize {
            let Some(slot_index) = find_slot(&state, dispatch.request_handles[lane_index]) else {
                continue;
            };
            let slot = &mut state.slots[slot_index];
            if matches!(
                slot.state,
                RequestApiState::RunningDecode
                    | RequestApiState::RunningSpeculativeVerify
                    | RequestApiState::RunningPrefill
                    | RequestApiState::QueuedPrefill
            ) {
                slot.state = RequestApiState::Cancelled;
            }
        }
        state.running_request_count =
            state.running_request_count.saturating_sub(dispatch.request_count);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Callback context (port of `SparkTestServingCallbackContext`; the behavior
// swaps the C tests do by reassigning function pointers become selector
// flags).
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct CallbackContext {
    expected_prompt_tokens: Vec<u32>,
    alternate_expected_prompt_tokens: Option<Vec<u32>>,
    alternate_request_id: u64,
    fail_prefill: bool,
    decode_busy_once: bool,
    decode_pending: bool,
    mtp_decode: bool,
    /// Service-test decode: no position assertions, `710000+` token ids
    /// (mirrors `SparkTestServiceDecode`).
    service_decode: bool,
    prefill_callback_count: u32,
    decode_callback_count: u32,
    largest_prefill_lane_count: u32,
    saw_decode_kv_table: bool,
    transient_decode_busy_count: u32,
    pending_dispatch: Option<Dispatch>,
    pending_decode_callback_count: u32,
    release_callback_count: u32,
    released_request_id: u64,
    released_sequence_id: u64,
    released_token_count: u32,
}

type SharedContext = Rc<RefCell<CallbackContext>>;

// ---------------------------------------------------------------------------
// Prefill/decode/release callbacks (ports of the C fixture callbacks; the
// behavior-swap the C does by reassigning function pointers is driven by the
// selector flags on `CallbackContext`).
// ---------------------------------------------------------------------------

fn make_prefill_function(context: SharedContext) -> engine::PrefillFunction {
    Box::new(move |prefill_dispatch| {
        let mut context = context.borrow_mut();
        if context.fail_prefill {
            return Err(ServingStatus::InvalidArgument);
        }
        assert!(prefill_dispatch.lane_count != 0);
        assert!(prefill_dispatch.prompt_token_count != 0);
        assert!(prefill_dispatch.prompt_token_count <= PREFILL_TOKEN_STRIDE);
        assert_eq!(prefill_dispatch.kv_block_table_view.lane_count, prefill_dispatch.lane_count);
        context.largest_prefill_lane_count =
            context.largest_prefill_lane_count.max(prefill_dispatch.lane_count);
        // The C service fixture's prefill callback does not verify staged
        // tokens (`SparkTestServicePrefill`); the engine fixture does.
        if !context.service_decode {
            for lane_index in 0..prefill_dispatch.lane_count as usize {
                let lane = &prefill_dispatch.lanes[lane_index];
                let expected_tokens = match &context.alternate_expected_prompt_tokens {
                    Some(alternate) if lane.request_id == context.alternate_request_id => alternate,
                    _ => &context.expected_prompt_tokens,
                };
                let lane_start = lane_index * prefill_dispatch.host_token_stride as usize;
                for token_index in 0..lane.prompt_token_count as usize {
                    assert_eq!(
                        prefill_dispatch.host_token_ids[lane_start + token_index],
                        expected_tokens[lane.prompt_token_offset as usize + token_index],
                        "prefill staging mismatch lane={lane_index} token={token_index}"
                    );
                }
            }
        }
        context.prefill_callback_count += 1;
        Ok(())
    })
}

fn make_decode_function(context: SharedContext) -> engine::DecodeFunction {
    Box::new(move |decode_dispatch, decode_result| {
        let mut context = context.borrow_mut();
        assert_eq!(decode_dispatch.kv_block_table_view.lane_count, decode_dispatch.request_count);
        assert_eq!(decode_dispatch.decode_view.lane_count, decode_dispatch.request_count);

        if context.decode_busy_once && context.transient_decode_busy_count == 0 {
            context.transient_decode_busy_count += 1;
            return Err(ServingStatus::Busy);
        }
        if context.decode_pending {
            context.pending_dispatch = Some(decode_dispatch.request_dispatch.clone());
            context.pending_decode_callback_count += 1;
            return Err(ServingStatus::Pending);
        }

        *decode_result = ServingDecodeResult::new(
            decode_dispatch.request_count,
            MAX_DECODE_TOKENS_PER_LANE as u32,
        );

        if context.mtp_decode {
            const FIRST_CANDIDATES: [u32; MTP_CANDIDATE_COUNT] =
                [90001, 90002, 91002, 90003, 91003];
            const FIRST_VERIFIER: [u32; MTP_VERIFIER_ROW_COUNT] =
                [90001, 90002, 90003, 91012, 90004, 91014];
            const SECOND_CANDIDATES: [u32; MTP_CANDIDATE_COUNT] =
                [90005, 90006, 91006, 90007, 91007];
            const SECOND_VERIFIER: [u32; MTP_VERIFIER_ROW_COUNT] =
                [90005, 90006, 90007, 91022, 90008, 91024];

            if decode_dispatch.dispatch_kind == DispatchKind::DecodeBatch {
                assert_ne!(decode_dispatch.request_dispatch.flags & DISPATCH_FLAG_MTP_COMMIT, 0);
                assert_eq!(context.decode_callback_count, 0);
                assert_eq!(
                    decode_dispatch.request_dispatch.mtp_draft_token_budget,
                    MTP_CANDIDATE_COUNT as u32
                );
                for lane_index in 0..decode_dispatch.request_count as usize {
                    decode_result.token_counts[lane_index] = MTP_CANDIDATE_COUNT as u32 + 1;
                    decode_result.token_ids[lane_index * MAX_DECODE_TOKENS_PER_LANE] = 90000;
                    for (token_index, &token_id) in FIRST_CANDIDATES.iter().enumerate() {
                        decode_result.token_ids
                            [lane_index * MAX_DECODE_TOKENS_PER_LANE + token_index + 1] = token_id;
                    }
                }
            } else {
                assert_eq!(decode_dispatch.dispatch_kind, DispatchKind::SpeculativeVerifyBatch);
                assert_ne!(
                    decode_dispatch.request_dispatch.flags & DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY,
                    0
                );
                assert_ne!(
                    decode_dispatch.request_dispatch.flags & DISPATCH_FLAG_MTP_TREE_VERIFY,
                    0
                );
                assert!(context.decode_callback_count == 1 || context.decode_callback_count == 2);
                assert_eq!(
                    decode_dispatch.request_dispatch.speculative_token_count,
                    MTP_CANDIDATE_COUNT as u32
                );
                assert_eq!(
                    decode_dispatch.request_dispatch.speculative_verifier_token_count,
                    MTP_VERIFIER_ROW_COUNT as u32
                );
                let verifier_token_ids = if context.decode_callback_count == 1 {
                    FIRST_VERIFIER
                } else {
                    SECOND_VERIFIER
                };
                for lane_index in 0..decode_dispatch.request_count as usize {
                    decode_result.token_counts[lane_index] = MTP_VERIFIER_ROW_COUNT as u32;
                    for (token_index, &token_id) in verifier_token_ids.iter().enumerate() {
                        decode_result.token_ids
                            [lane_index * MAX_DECODE_TOKENS_PER_LANE + token_index] = token_id;
                    }
                    if context.decode_callback_count == 1 {
                        decode_result.draft_token_counts[lane_index] = MTP_CANDIDATE_COUNT as u32;
                        for (token_index, &token_id) in SECOND_CANDIDATES.iter().enumerate() {
                            decode_result.draft_token_ids
                                [lane_index * bridge::MTP_MAX_DRAFT_TOKEN_COUNT + token_index] =
                                token_id;
                        }
                    }
                }
            }
        } else if context.service_decode {
            // `SparkTestServiceDecode`: no position assertions.
            for lane_index in 0..decode_dispatch.request_count as usize {
                decode_result.token_counts[lane_index] = 1;
                decode_result.token_ids[lane_index * MAX_DECODE_TOKENS_PER_LANE] =
                    710000 + context.decode_callback_count + lane_index as u32;
            }
        } else {
            for lane_index in 0..decode_dispatch.request_count as usize {
                if context.decode_callback_count == 0 {
                    assert_eq!(
                        decode_dispatch.input_token_ids[lane_index],
                        context.expected_prompt_tokens[PROMPT_TOKEN_COUNT - 1]
                    );
                    assert_eq!(
                        decode_dispatch.decode_view.lanes[lane_index].sequence_position,
                        PROMPT_TOKEN_COUNT as u32 - 1
                    );
                    assert_eq!(
                        decode_dispatch.decode_view.lanes[lane_index].context_token_count,
                        PROMPT_TOKEN_COUNT as u32
                    );
                } else {
                    assert_eq!(
                        decode_dispatch.input_token_ids[lane_index],
                        90000 + context.decode_callback_count - 1 + lane_index as u32
                    );
                    assert_eq!(
                        decode_dispatch.decode_view.lanes[lane_index].sequence_position,
                        PROMPT_TOKEN_COUNT as u32 + context.decode_callback_count - 1
                    );
                    assert_eq!(
                        decode_dispatch.decode_view.lanes[lane_index].context_token_count,
                        PROMPT_TOKEN_COUNT as u32 + context.decode_callback_count
                    );
                }
                decode_result.token_counts[lane_index] = 1;
                decode_result.token_ids[lane_index * MAX_DECODE_TOKENS_PER_LANE] =
                    90000 + context.decode_callback_count + lane_index as u32;
            }
        }

        context.decode_callback_count += 1;
        context.saw_decode_kv_table = true;
        Ok(())
    })
}

fn make_release_sequence_function(context: SharedContext) -> engine::ReleaseSequenceFunction {
    Box::new(move |request_id, request_generation, sequence_id, token_count| {
        assert_ne!(request_id, 0);
        assert_ne!(request_generation, 0);
        assert_ne!(sequence_id, 0);
        assert_ne!(token_count, 0);
        let mut context = context.borrow_mut();
        context.release_callback_count += 1;
        context.released_request_id = request_id;
        context.released_sequence_id = sequence_id;
        context.released_token_count = token_count;
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Engine/service builders.
// ---------------------------------------------------------------------------

fn engine_configuration(
    request_api: FakeRequestApi,
    context: SharedContext,
    flags: u32,
    default_output_token_budget: u32,
    stop_token_ids: Vec<u32>,
) -> ServingEngineConfiguration {
    let dynamic_storage = (flags & ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE) != 0;
    ServingEngineConfiguration {
        flags,
        runtime_contract_flags: TEST_RUNTIME_CONTRACT_FLAGS,
        default_thinking_token_budget: 0,
        default_output_token_budget,
        default_max_prefill_tokens_per_step: PREFILL_TOKEN_STRIDE,
        max_context_tokens: MAX_CONTEXT_TOKENS,
        request_id_base: 0,
        request_api: Box::new(request_api),
        tokenizer: None,
        request_record_capacity: REQUEST_RECORD_CAPACITY,
        request_token_stride: if dynamic_storage { 0 } else { REQUEST_TOKEN_STRIDE },
        event_ring_capacity: EVENT_CAPACITY,
        host_prefill_token_stride: PREFILL_TOKEN_STRIDE,
        host_prefill_lane_capacity: MAX_DISPATCH_REQUEST_COUNT as u32,
        execution_physical_block_indices: None,
        kv_block_lane_stride: KV_BLOCK_LANE_CAPACITY,
        kv_block_lane_capacity: KV_BLOCK_LANE_CAPACITY,
        lane_count_capacity: MAX_DISPATCH_REQUEST_COUNT as u32,
        prefill_function: make_prefill_function(context.clone()),
        decode_function: make_decode_function(context.clone()),
        release_sequence_function: Some(make_release_sequence_function(context)),
        stop_token_ids,
    }
}

struct EngineFixture {
    engine: ServingEngine,
    request_api: FakeRequestApi,
    context: SharedContext,
}

fn build_engine_fixture(default_output_token_budget: u32) -> EngineFixture {
    let request_api = FakeRequestApi::new();
    let context: SharedContext = Rc::new(RefCell::new(CallbackContext {
        expected_prompt_tokens: fill_token_ids(PROMPT_TOKEN_COUNT, 81000),
        ..CallbackContext::default()
    }));
    let engine = ServingEngine::new(engine_configuration(
        request_api.clone(),
        context.clone(),
        0,
        default_output_token_budget,
        Vec::new(),
    ))
    .expect("engine init");
    EngineFixture { engine, request_api, context }
}

fn submit_request(
    engine: &mut ServingEngine,
    token_ids: &[u32],
    output_token_budget: u32,
    request_id: u64,
    sequence_id: u64,
) -> spark_serve::serving_engine::SubmitResult {
    let request = SubmitTokenIdsRequest {
        flags: 0,
        priority: 0,
        thinking_token_budget: 0,
        output_token_budget,
        max_prefill_tokens_per_step: 0,
        request_id,
        sequence_id,
        token_ids,
    };
    engine.submit_token_ids(&request).expect("submit")
}

// ---------------------------------------------------------------------------
// Serving engine tests (port of tests/test_glm52_serving_engine.c).
// ---------------------------------------------------------------------------

/// `SparkTestServingRejectsTailWindowRuntimeContract`.
#[test]
fn rejects_tail_window_runtime_contract() {
    // Tail-window validation-only vetoes production.
    {
        let request_api = FakeRequestApi::new();
        let context: SharedContext = Rc::new(RefCell::new(CallbackContext::default()));
        let mut configuration = engine_configuration(request_api, context, 0, 2, Vec::new());
        configuration.runtime_contract_flags = RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS
            | RUNTIME_CONTRACT_FLAG_TAIL_WINDOW_VALIDATION_ONLY;
        assert_eq!(ServingEngine::new(configuration).err(), Some(ServingStatus::InvalidArgument));
    }
    // Production contract but the request API lacks prefill batching.
    {
        let request_api = FakeRequestApi::new();
        request_api.set_configuration_flags(
            FAKE_CONFIGURATION_FLAGS & !CONFIGURATION_FLAG_PREFILL_BATCHING,
        );
        let context: SharedContext = Rc::new(RefCell::new(CallbackContext::default()));
        let mut configuration = engine_configuration(request_api, context, 0, 2, Vec::new());
        configuration.runtime_contract_flags = RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS;
        assert_eq!(ServingEngine::new(configuration).err(), Some(ServingStatus::InvalidArgument));
    }
    // Non-production contract (missing INDEXSHARE_STAGE_BOUNDARY_STATE).
    {
        let request_api = FakeRequestApi::new();
        let context: SharedContext = Rc::new(RefCell::new(CallbackContext::default()));
        let mut configuration = engine_configuration(request_api, context, 0, 2, Vec::new());
        configuration.runtime_contract_flags = RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS
            & !RUNTIME_CONTRACT_FLAG_INDEXSHARE_STAGE_BOUNDARY_STATE;
        assert_eq!(ServingEngine::new(configuration).err(), Some(ServingStatus::InvalidArgument));
    }
}

/// `SparkTestServingFireAndForgetPumpRunsFullPromptToDecode`.
#[test]
fn fire_and_forget_pump_runs_full_prompt_to_decode() {
    let mut fixture = build_engine_fixture(2);
    let prompt_tokens = fill_token_ids(PROMPT_TOKEN_COUNT, 81000);
    let submit_result = submit_request(&mut fixture.engine, &prompt_tokens, 0, 9001, 19001);
    assert_ne!(submit_result.request_handle, 0);
    assert_eq!(submit_result.output_token_budget, 2);

    let mut stats = ServingStats::default();
    let status = fixture.engine.pump(0, 16, Some(&mut stats));
    assert!(status == Ok(()) || status == Err(ServingStatus::NotFound));
    {
        let context = fixture.context.borrow();
        assert_eq!(context.prefill_callback_count, 2);
        assert_eq!(context.decode_callback_count, 2);
        assert!(context.saw_decode_kv_table);
    }
    assert_eq!(stats.prefill_dispatch_count, 2);
    assert_eq!(stats.prefill_token_count, PROMPT_TOKEN_COUNT as u64);
    assert_eq!(stats.decode_dispatch_count, 2);
    assert_eq!(stats.decoded_token_count, 2);

    let mut accepted_event_count = 0;
    let mut prefill_event_count = 0;
    let mut token_event_count = 0;
    let mut completion_event_count = 0;
    while let Ok(event) = fixture.engine.pop_event() {
        assert_eq!(event.request_id, 9001);
        assert_eq!(event.sequence_id, 19001);
        match event.kind {
            ServingEventKind::RequestAccepted => accepted_event_count += 1,
            ServingEventKind::PrefillProgress => prefill_event_count += 1,
            ServingEventKind::Token => {
                token_event_count += 1;
                assert!(event.token_id == 90000 || event.token_id == 90001);
            }
            ServingEventKind::RequestCompleted => completion_event_count += 1,
            _ => {}
        }
    }
    assert_eq!(accepted_event_count, 1);
    assert_eq!(prefill_event_count, 2);
    assert_eq!(token_event_count, 2);
    assert_eq!(completion_event_count, 1);
    let context = fixture.context.borrow();
    assert_eq!(context.release_callback_count, 1);
    assert_eq!(context.released_request_id, 9001);
    assert_eq!(context.released_sequence_id, 19001);
    assert_eq!(context.released_token_count, PROMPT_TOKEN_COUNT as u32 + 2);
}

/// `SparkTestServingPrefillFailureEmitsTerminalEvent`.
#[test]
fn prefill_failure_emits_terminal_event() {
    let mut fixture = build_engine_fixture(2);
    fixture.context.borrow_mut().fail_prefill = true;
    let prompt_tokens = fill_token_ids(PROMPT_TOKEN_COUNT, 81000);
    let submit_result = submit_request(&mut fixture.engine, &prompt_tokens, 0, 9101, 19101);

    let status = fixture.engine.pump(0, 16, None);
    assert_eq!(status, Err(ServingStatus::InvalidArgument));

    let mut cancelled_event_count = 0;
    while let Ok(event) = fixture.engine.pop_event() {
        if event.kind == ServingEventKind::RequestCancelled {
            cancelled_event_count += 1;
            assert_eq!(event.request_handle, submit_result.request_handle);
            assert_eq!(event.status, ServingStatus::InvalidArgument);
        }
    }
    assert_eq!(cancelled_event_count, 1);
    // The cancelled request can be released instead of leaking.
    assert_eq!(fixture.engine.release_completed_request(submit_result.request_handle), Ok(()));
}

/// `SparkTestServingFailRequestByRequestIdEmitsTerminalEvent`.
#[test]
fn fail_request_by_request_id_emits_terminal_event() {
    let mut fixture = build_engine_fixture(2);
    let prompt_tokens = fill_token_ids(PROMPT_TOKEN_COUNT, 81000);
    let submit_result = submit_request(&mut fixture.engine, &prompt_tokens, 0, 9201, 19201);

    assert_eq!(
        fixture.engine.fail_request_by_request_id(9201, ServingStatus::InternalError),
        Ok(())
    );
    // Idempotent on a terminal record.
    assert_eq!(
        fixture.engine.fail_request_by_request_id(9201, ServingStatus::InternalError),
        Ok(())
    );
    assert_eq!(
        fixture.engine.fail_request_by_request_id(424242, ServingStatus::InternalError),
        Err(ServingStatus::NotFound)
    );

    let mut cancelled_event_count = 0;
    while let Ok(event) = fixture.engine.pop_event() {
        if event.kind == ServingEventKind::RequestCancelled {
            cancelled_event_count += 1;
            assert_eq!(event.request_id, 9201);
            assert_eq!(event.request_handle, submit_result.request_handle);
            assert_eq!(event.status, ServingStatus::InternalError);
        }
    }
    assert_eq!(cancelled_event_count, 1);
    assert_eq!(fixture.engine.release_completed_request(submit_result.request_handle), Ok(()));

    let submit_result = submit_request(&mut fixture.engine, &prompt_tokens, 0, 9202, 19202);
    assert_ne!(submit_result.request_handle, 0);
    let mut stats = ServingStats::default();
    let status = fixture.engine.pump(0, 16, Some(&mut stats));
    assert!(status == Ok(()) || status == Err(ServingStatus::NotFound));
    assert_eq!(stats.prefill_token_count, PROMPT_TOKEN_COUNT as u64);
}

/// `SparkTestServingRetriesUnacceptedBusyDecode`.
#[test]
fn retries_unaccepted_busy_decode() {
    let mut fixture = build_engine_fixture(2);
    fixture.context.borrow_mut().decode_busy_once = true;
    let prompt_tokens = fill_token_ids(PROMPT_TOKEN_COUNT, 81000);
    submit_request(&mut fixture.engine, &prompt_tokens, 1, 9002, 19002);

    let mut stats = ServingStats::default();
    assert_eq!(fixture.engine.pump(0, 16, Some(&mut stats)), Err(ServingStatus::Busy));
    {
        let context = fixture.context.borrow();
        assert_eq!(context.transient_decode_busy_count, 1);
        assert_eq!(context.decode_callback_count, 0);
    }
    assert_eq!(fixture.request_api.slot_state(0), RequestApiState::ReadyDecode);
    assert_eq!(fixture.request_api.running_request_count(), 0);

    let status = fixture.engine.pump(0, 16, Some(&mut stats));
    assert!(status == Ok(()) || status == Err(ServingStatus::NotFound));
    assert_eq!(fixture.context.borrow().decode_callback_count, 1);
    assert_eq!(stats.completed_stream_count, 1);
}

/// `SparkTestServingPreservesAcceptedPendingDecode`.
#[test]
fn preserves_accepted_pending_decode() {
    let mut fixture = build_engine_fixture(2);
    fixture.context.borrow_mut().decode_pending = true;
    let prompt_tokens = fill_token_ids(PROMPT_TOKEN_COUNT, 81000);
    submit_request(&mut fixture.engine, &prompt_tokens, 1, 9003, 19003);

    let mut stats = ServingStats::default();
    assert_eq!(fixture.engine.pump(0, 16, Some(&mut stats)), Err(ServingStatus::Pending));
    let pending_dispatch =
        fixture.context.borrow_mut().pending_dispatch.take().expect("pending dispatch captured");
    assert_eq!(fixture.request_api.slot_state(0), RequestApiState::RunningDecode);
    assert_eq!(fixture.request_api.running_request_count(), 1);

    let mut dispatch = pending_dispatch;
    let mut decode_result = ServingDecodeResult::new(1, MAX_DECODE_TOKENS_PER_LANE as u32);
    decode_result.token_counts[0] = 1;
    decode_result.token_ids[0] = 90000;
    assert_eq!(fixture.engine.complete_decode_dispatch(&mut dispatch, &mut decode_result), Ok(()));
    assert_eq!(fixture.engine.get_stats().completed_stream_count, 1);
}

/// `SparkTestServingPumpsMultipleAcceptedPendingDecodes`.
#[test]
fn pumps_multiple_accepted_pending_decodes() {
    let mut fixture = build_engine_fixture(2);
    fixture.context.borrow_mut().decode_pending = true;
    fixture.request_api.set_decode_batch_target(1);
    let prompt_tokens = fill_token_ids(PROMPT_TOKEN_COUNT, 81000);
    for request_index in 0..2u64 {
        submit_request(
            &mut fixture.engine,
            &prompt_tokens,
            1,
            9100 + request_index,
            19100 + request_index,
        );
    }

    let mut stats = ServingStats::default();
    assert_eq!(fixture.engine.pump(0, 16, Some(&mut stats)), Err(ServingStatus::Pending));
    assert_eq!(fixture.context.borrow().pending_decode_callback_count, 2);
    assert_eq!(fixture.request_api.running_request_count(), 2);
}

/// `SparkTestServingPrefillBatchingIsInternal`.
#[test]
fn prefill_batching_is_internal() {
    let mut fixture = build_engine_fixture(2);
    let first_prompt = fill_token_ids(PREFILL_TOKEN_STRIDE as usize, 81000);
    let second_prompt = fill_token_ids(PREFILL_TOKEN_STRIDE as usize, 82000);
    {
        let mut context = fixture.context.borrow_mut();
        context.expected_prompt_tokens = first_prompt.clone();
        context.alternate_expected_prompt_tokens = Some(second_prompt.clone());
        context.alternate_request_id = 9102;
    }
    submit_request(&mut fixture.engine, &first_prompt, 1, 9101, 19101);
    submit_request(&mut fixture.engine, &second_prompt, 1, 9102, 19102);

    assert_eq!(fixture.engine.pump(PUMP_FLAG_STOP_AFTER_ONE_DISPATCH, 1, None), Ok(()));
    let context = fixture.context.borrow();
    assert_eq!(context.prefill_callback_count, 1);
    assert_eq!(context.largest_prefill_lane_count, 2);
}

/// `SparkTestServingMtpCommitStreamsMultiTokenLanes`.
#[test]
fn mtp_commit_streams_multi_token_lanes() {
    let request_api = FakeRequestApi::new();
    request_api.enable_mtp();
    let context: SharedContext = Rc::new(RefCell::new(CallbackContext {
        expected_prompt_tokens: fill_token_ids(PROMPT_TOKEN_COUNT, 81000),
        mtp_decode: true,
        ..CallbackContext::default()
    }));
    let mut engine = ServingEngine::new(engine_configuration(
        request_api,
        context.clone(),
        0,
        2,
        vec![MTP_STOP_TOKEN_ID],
    ))
    .expect("engine init");

    let prompt_tokens = fill_token_ids(PROMPT_TOKEN_COUNT, 81000);
    let submit_result = submit_request(&mut engine, &prompt_tokens, 9, 9301, 19301);
    assert_eq!(submit_result.output_token_budget, 9);

    let mut stats = ServingStats::default();
    let status = engine.pump(0, 16, Some(&mut stats));
    assert!(status == Ok(()) || status == Err(ServingStatus::NotFound));
    assert_eq!(context.borrow().decode_callback_count, 3);
    assert_eq!(stats.decode_dispatch_count, 3);
    assert_eq!(stats.decoded_token_count, 9);
    assert_eq!(stats.mtp_draft_token_count, (MTP_CANDIDATE_COUNT * 2) as u64);
    assert_eq!(stats.mtp_verify_dispatch_count, 2);
    assert_eq!(stats.mtp_draft_ready_count, 2);
    assert_eq!(stats.mtp_accepted_draft_token_count, 6);
    assert_eq!(stats.mtp_committed_token_count, 8);
    assert_eq!(stats.mtp_rejected_token_count, 4);

    let mut token_event_count = 0;
    let mut completion_event_count = 0;
    while let Ok(event) = engine.pop_event() {
        match event.kind {
            ServingEventKind::Token => {
                assert_eq!(event.token_id, 90000 + token_event_count);
                token_event_count += 1;
            }
            ServingEventKind::RequestCompleted => completion_event_count += 1,
            _ => {}
        }
    }
    assert_eq!(token_event_count, 9);
    assert_eq!(completion_event_count, 1);
}

/// `SparkTestServingDynamicTokenStorageGrowsAndRecycles`.
#[test]
fn dynamic_token_storage_grows_and_recycles() {
    let request_api = FakeRequestApi::new();
    let context: SharedContext = Rc::new(RefCell::new(CallbackContext {
        expected_prompt_tokens: fill_token_ids(PROMPT_TOKEN_COUNT, 81000),
        ..CallbackContext::default()
    }));
    let mut engine = ServingEngine::new(engine_configuration(
        request_api,
        context.clone(),
        ENGINE_DEFAULT_FLAGS | ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE,
        2,
        Vec::new(),
    ))
    .expect("engine init");
    assert_eq!(engine.free_record_head(), 0);

    let prompt_tokens = fill_token_ids(PROMPT_TOKEN_COUNT, 81000);
    let result = submit_request(&mut engine, &prompt_tokens, 7, 9401, 0);
    assert_ne!(result.sequence_id, 0);
    let handle = result.request_handle;
    assert_eq!(engine.free_record_head(), 1);
    assert!(engine.record_token_capacity(0).expect("record 0") >= PROMPT_TOKEN_COUNT as u32 + 7);

    assert_eq!(engine.cancel_request(handle), Ok(()));
    assert_eq!(engine.release_completed_request(handle), Ok(()));
    assert_eq!(engine.free_record_head(), 0);
    assert_eq!(engine.record_token_capacity(0), Some(0));
    let context = context.borrow();
    assert_eq!(context.release_callback_count, 1);
    assert_eq!(context.released_request_id, 9401);
    assert_eq!(context.released_sequence_id, result.sequence_id);
    assert_eq!(context.released_token_count, PROMPT_TOKEN_COUNT as u32);
}

// ---------------------------------------------------------------------------
// Service runtime tests (port of tests/test_glm52_service.c).
// ---------------------------------------------------------------------------

const SERVICE_PROMPT_TOKEN_COUNT: usize = 64;

fn build_service_fixture() -> (ServiceRuntime, SharedContext) {
    let request_api = FakeRequestApi::new();
    let context: SharedContext = Rc::new(RefCell::new(CallbackContext {
        expected_prompt_tokens: fill_token_ids(SERVICE_PROMPT_TOKEN_COUNT, 111000),
        service_decode: true,
        ..CallbackContext::default()
    }));
    let engine =
        ServingEngine::new(engine_configuration(request_api, context.clone(), 0, 1, Vec::new()))
            .expect("engine init");
    let service = ServiceRuntime::new(service::ServiceConfiguration {
        flags: 0,
        default_pump_dispatch_steps: 0,
        request_id_base: 0,
        serving_engine: engine,
        client_session_capacity: 4,
        request_map_capacity: 8,
        event_ring_capacity: EVENT_CAPACITY,
    })
    .expect("service init");
    (service, context)
}

/// `SparkTestServiceClientsUseInternalQueueing`.
#[test]
fn service_clients_use_internal_queueing() {
    let (mut service, context) = build_service_fixture();
    let first_client_id = service.register_client(100).expect("register first");
    let second_client_id = service.register_client(200).expect("register second");
    let first_prompt = fill_token_ids(SERVICE_PROMPT_TOKEN_COUNT, 111000);
    let second_prompt = fill_token_ids(SERVICE_PROMPT_TOKEN_COUNT, 222000);

    let submit_request = ServiceSubmitTokenIdsRequest {
        client_id: first_client_id,
        client_request_id: 11,
        token_ids: &first_prompt,
        ..ServiceSubmitTokenIdsRequest::default()
    };
    let submit_result = service.submit_token_ids(&submit_request).expect("first submit");
    assert_eq!(submit_result.client_id, first_client_id);
    assert_eq!(submit_result.client_request_id, 11);

    assert_eq!(service.submit_token_ids(&submit_request), Err(ServingStatus::Duplicate));

    let submit_request = ServiceSubmitTokenIdsRequest {
        client_id: second_client_id,
        client_request_id: 22,
        token_ids: &second_prompt,
        ..ServiceSubmitTokenIdsRequest::default()
    };
    service.submit_token_ids(&submit_request).expect("second submit");

    let mut stats = ServiceStats::default();
    assert_eq!(service.pump(16, Some(&mut stats)), Ok(()));
    {
        let context = context.borrow();
        assert_eq!(context.prefill_callback_count, 1);
        assert_eq!(context.largest_prefill_lane_count, 2);
        assert_eq!(context.decode_callback_count, 1);
        assert!(context.saw_decode_kv_table);
    }
    assert_eq!(stats.serving_stats.prefill_batch_dispatch_count, 1);
    assert_eq!(stats.serving_stats.decode_dispatch_count, 1);

    let mut first_token_event_count = 0;
    let mut second_token_event_count = 0;
    let mut completion_event_count = 0;
    while let Ok(event) = service.pop_event() {
        match event.kind {
            ServiceEventKind::Token => {
                if event.client_id == first_client_id {
                    first_token_event_count += 1;
                    assert_eq!(event.client_request_id, 11);
                } else if event.client_id == second_client_id {
                    second_token_event_count += 1;
                    assert_eq!(event.client_request_id, 22);
                }
            }
            ServiceEventKind::RequestCompleted => completion_event_count += 1,
            _ => {}
        }
    }
    assert_eq!(first_token_event_count, 1);
    assert_eq!(second_token_event_count, 1);
    assert_eq!(completion_event_count, 2);
}

/// `SparkTestServiceTokenFrameSubmitWorks`.
#[test]
fn service_token_frame_submit_works() {
    let (mut service, _context) = build_service_fixture();
    let client_id = service.register_client(300).expect("register");

    let token_count = 8usize;
    let mut body = vec![0u8; FRAME_SUBMIT_TOKENS_BODY_BYTES + token_count * 4];
    body[0..4].copy_from_slice(&SERVICE_ABI_VERSION.to_le_bytes());
    body[4..8].copy_from_slice(&(FRAME_SUBMIT_TOKENS_BODY_BYTES as u32).to_le_bytes());
    body[16..20].copy_from_slice(&1u32.to_le_bytes()); // output_token_budget
    body[24..28].copy_from_slice(&(token_count as u32).to_le_bytes());
    for token_index in 0..token_count {
        let offset = FRAME_SUBMIT_TOKENS_BODY_BYTES + token_index * 4;
        body[offset..offset + 4].copy_from_slice(&(88000u32 + token_index as u32).to_le_bytes());
    }

    let mut frame_header = service::initialize_frame_header(FRAME_KIND_SUBMIT_TOKEN_IDS);
    frame_header.client_id = client_id;
    frame_header.client_request_id = 44;
    frame_header.body_bytes = body.len() as u32;

    let submit_result = service
        .handle_submit_token_ids_frame(client_id, &frame_header, &body)
        .expect("frame submit");
    assert_eq!(submit_result.client_id, client_id);
    assert_eq!(submit_result.client_request_id, 44);
    assert_eq!(submit_result.prompt_token_count, 8);
}

// ---------------------------------------------------------------------------
// Compat API tests (port of tests/test_glm52_compat_api.c; the C file's
// chat-template-writer case is covered by spark-text's chat_template tests).
// ---------------------------------------------------------------------------

const CHAT_BEGIN: &str = "[gMASK]<sop><|system|>Reasoning Effort: Max";
const CHAT_END: &str = "<|assistant|><think>";
const NO_THINK_CHAT_BEGIN: &str = "[gMASK]<sop>";
const NO_THINK_CHAT_END: &str = "<|assistant|><think></think>";

/// `SparkTestCompatOpenAiChat`.
#[test]
fn compat_openai_chat() {
    let request_json = concat!(
        "{",
        "\"model\":\"glm-5.2\",",
        "\"priority\":123,",
        "\"max_tokens\":17,",
        "\"messages\":[",
        "{\"role\":\"system\",\"content\":\"You are terse.\"},",
        "{\"role\":\"user\",\"content\":\"Read this C code.\"}",
        "]",
        "}"
    );
    let expected =
        format!("{CHAT_BEGIN}<|system|>You are terse.<|user|>Read this C code.{CHAT_END}");
    let mut request = CompatTextRequest::new(512);
    request.client_id = 10;
    request.client_request_id = 20;
    assert_eq!(prepare_openai_json(request_json, &mut request), Ok(()));
    assert_eq!(request.output_token_budget, 17);
    assert_eq!(request.priority, 123);
    assert_eq!(request.text_bytes(), expected.len());
    assert_eq!(request.text(), expected);
}

/// `SparkTestCompatOpenAiPrompt`.
#[test]
fn compat_openai_prompt() {
    let request_json =
        "{\"model\":\"glm-5.2\",\"max_completion_tokens\":5,\"prompt\":\"plain prompt\"}";
    let mut request = CompatTextRequest::new(64);
    assert_eq!(prepare_openai_json(request_json, &mut request), Ok(()));
    assert_eq!(request.output_token_budget, 5);
    assert_eq!(request.text(), "plain prompt");
}

/// `SparkTestCompatOpenAiThinkingBudgets`.
#[test]
fn compat_openai_thinking_budgets() {
    let no_think_json =
        "{\"thinking_budget_tokens\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Answer.\"}]}";
    let think_json =
        "{\"thinking_token_budget\":1024,\"messages\":[{\"role\":\"user\",\"content\":\"Answer.\"}]}";
    let conflict_json = concat!(
        "{\"thinking_budget_tokens\":0,\"thinking_token_budget\":1024,",
        "\"messages\":[{\"role\":\"user\",\"content\":\"Answer.\"}]}"
    );

    let mut request = CompatTextRequest::new(256);
    assert_eq!(prepare_openai_json(no_think_json, &mut request), Ok(()));
    assert_eq!(request.thinking_token_budget, 0);
    assert_eq!(request.text(), format!("{NO_THINK_CHAT_BEGIN}<|user|>Answer.{NO_THINK_CHAT_END}"));

    let mut request = CompatTextRequest::new(256);
    assert_eq!(prepare_openai_json(think_json, &mut request), Ok(()));
    assert_eq!(request.thinking_token_budget, 1024);
    assert_eq!(request.text(), format!("{CHAT_BEGIN}<|user|>Answer.{CHAT_END}"));

    let mut request = CompatTextRequest::new(256);
    assert_eq!(
        prepare_openai_json(conflict_json, &mut request),
        Err(ServingStatus::InvalidArgument)
    );
}

/// `SparkTestCompatOpenAiChatWithFiles`.
#[test]
fn compat_openai_chat_with_files() {
    let request_json = concat!(
        "{",
        "\"model\":\"glm-5.2\",",
        "\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Use the attachment.\"}]}],",
        "\"files\":[{\"filename\":\"notes.txt\",\"content\":\"alpha\\nbeta\"}]",
        "}"
    );
    let mut request = CompatTextRequest::new(512);
    assert_eq!(prepare_openai_json(request_json, &mut request), Ok(()));
    let text = request.text().to_string();
    assert!(text.contains(&format!("{CHAT_BEGIN}<|user|>Use the attachment.")));
    assert!(text.contains("[uploaded file: notes.txt]"));
    assert!(text.contains("alpha\nbeta"));
    assert!(text.contains("[/uploaded file]"));
    assert!(text.contains(CHAT_END));
}

/// `SparkTestCompatAnthropicMessages`.
#[test]
fn compat_anthropic_messages() {
    let request_json = concat!(
        "{",
        "\"model\":\"glm-5.2\",",
        "\"priority\":456,",
        "\"system\":\"Stay exact.\",",
        "\"max_tokens\":9,",
        "\"messages\":[",
        "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"First\"},",
        "{\"type\":\"text\",\"text\":\" second\"}]},",
        "{\"role\":\"assistant\",\"content\":\"Ack\"}",
        "]",
        "}"
    );
    let expected = format!(
        "{CHAT_BEGIN}<|system|>Stay exact.<|user|>First second<|assistant|><think></think>Ack{CHAT_END}"
    );
    let mut request = CompatTextRequest::new(512);
    assert_eq!(prepare_anthropic_json(request_json, &mut request), Ok(()));
    assert_eq!(request.output_token_budget, 9);
    assert_eq!(request.priority, 456);
    assert_eq!(request.text(), expected);
}

/// `SparkTestCompatRejectsUnknownChatRole`.
#[test]
fn compat_rejects_unknown_chat_role() {
    let request_json = "{\"messages\":[{\"role\":\"alien\",\"content\":\"no\"}]}";
    let mut request = CompatTextRequest::new(256);
    assert_eq!(
        prepare_openai_json(request_json, &mut request),
        Err(ServingStatus::InvalidArgument)
    );
}

/// `SparkTestCompatRejectsSmallBuffer`.
#[test]
fn compat_rejects_small_buffer() {
    let request_json = "{\"messages\":[{\"role\":\"user\",\"content\":\"too long for buffer\"}]}";
    let mut request = CompatTextRequest::new(8);
    assert_eq!(
        prepare_openai_json(request_json, &mut request),
        Err(ServingStatus::CapacityExceeded)
    );
}
