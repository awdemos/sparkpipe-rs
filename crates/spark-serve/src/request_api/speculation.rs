//! Speculation seams and MTP commit logic — port of the
//! `spark_request_model.h` seam (the C's twelve-entry model-module link
//! surface, collapsing into the [`RequestModelSeam`] trait), the GLM52 dspark
//! request-model shim (`modules/glm52_dspark_draft_backend/source/
//! spark_glm52_request_model.c`), `SparkMtpTreeResolve` from
//! `include/sparkpipe/spark_mtp_tree.h`, and the MTP draft/verify/commit
//! policy in `api/request.c`.
//!
//! The dspark provider module itself (tap recording, trajectory buffer,
//! draft generation) is a model module and is not ported here; the trait
//! carries its per-request operations.

use spark_sched::stage_plan;
use spark_sched::work_control::mtp_tree;

use super::dispatch::{
    Dispatch, DispatchKind, DISPATCH_FLAG_MTP_COMMIT, DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY,
    DISPATCH_FLAG_MTP_TREE_VERIFY,
};
use super::slot::{
    Slot, REQUEST_FLAG_DISABLE_SPECULATION, STATE_COMPLETED, STATE_READY_DECODE,
    STATE_READY_SPECULATIVE_VERIFY, STATE_RUNNING_SPECULATIVE_VERIFY,
};
use super::{
    RequestApi, RequestApiError, CONFIGURATION_FLAG_MTP_FORCE_ENABLE, MAX_DISPATCH_REQUEST_COUNT,
    MAX_SPECULATIVE_TOKENS, MTP_COMMIT_EMA_DIVISOR, MTP_MAX_DRAFT_TOKEN_COUNT,
    MTP_REPROBE_INTERVAL, MTP_SUPPRESS_THRESHOLD_MILLI,
};

/// Speculative source: the drafter (`SPARK_REQUEST_MODEL_SPECULATIVE_SOURCE_DRAFTER`).
pub const SPECULATIVE_SOURCE_DRAFTER: u32 = 1;
/// Speculative source: MTP (`SPARK_REQUEST_MODEL_SPECULATIVE_SOURCE_MTP`).
pub const SPECULATIVE_SOURCE_MTP: u32 = 2;

/// Verify-result flag: every draft token accepted
/// (`SPARK_REQUEST_MODEL_VERIFY_RESULT_FLAG_ACCEPTED_ALL`).
pub const VERIFY_RESULT_FLAG_ACCEPTED_ALL: u32 = 0x0000_0001;
/// Verify-result flag: some draft token rejected
/// (`SPARK_REQUEST_MODEL_VERIFY_RESULT_FLAG_REJECTED`).
pub const VERIFY_RESULT_FLAG_REJECTED: u32 = 0x0000_0002;

/// MTP utility scale (`SPARK_GLM52_REQUEST_API_MTP_UTILITY_SCALE`).
const MTP_UTILITY_SCALE: u64 = 1000;
/// MTP utility margin (`SPARK_GLM52_REQUEST_API_MTP_UTILITY_MARGIN_SCALE`).
const MTP_UTILITY_MARGIN_SCALE: u64 = 1250;
/// Draft-chain work multiplier
/// (`SPARK_GLM52_REQUEST_API_MTP_DRAFT_CHAIN_WORK_MULTIPLIER`).
const MTP_DRAFT_CHAIN_WORK_MULTIPLIER: u64 = 2;
/// Prior committed token count
/// (`SPARK_GLM52_REQUEST_API_MTP_UTILITY_PRIOR_COMMITTED_TOKEN_COUNT`).
const MTP_UTILITY_PRIOR_COMMITTED_TOKEN_COUNT: u64 =
    (mtp_tree::MAX_COMMITTED_TOKEN_COUNT / 2) as u64;
/// Prior sample count
/// (`SPARK_GLM52_REQUEST_API_MTP_UTILITY_PRIOR_SAMPLE_COUNT` =
/// `SPARK_STAGE_PLAN_CURRENT_SPARK_COUNT`).
const MTP_UTILITY_PRIOR_SAMPLE_COUNT: u64 = stage_plan::CURRENT_SPARK_COUNT as u64;

/// Speculative draft result (`SparkRequestModelDraftResult`, minus ABI words).
#[derive(Debug, Clone, Default)]
pub struct DraftResult {
    pub flags: u32,
    pub token_count: u32,
    pub confidence_milli: [u32; MAX_SPECULATIVE_TOKENS as usize],
    pub token_ids: [u32; MAX_SPECULATIVE_TOKENS as usize],
}

/// Speculative verify result (`SparkRequestModelVerifyResult`, minus ABI words).
#[derive(Debug, Clone, Default)]
pub struct VerifyResult {
    pub flags: u32,
    pub proposed_token_count: u32,
    pub accepted_draft_token_count: u32,
    pub committed_token_count: u32,
    pub fallback_token_id: u32,
}

/// Draft request (`SparkGlm52DsparkDraftRequest`, minus ABI words) — the
/// arguments the seam's `ensure_draft` consumes.
#[derive(Debug, Clone, Copy, Default)]
pub struct DraftRequest {
    pub requested_token_count: u32,
    pub priority: u32,
    pub request_id: u64,
    pub sequence_id: u64,
    pub sequence_position: u64,
    pub tap_generation: u64,
}

/// The `spark_request_model.h` seam as a trait: the per-provider entries the
/// request API calls. `speculation_is_enabled` and
/// `get_slot_speculative_draft` / `mtp_outranks_plain_decode` stay on
/// [`RequestApi`] because the C implements them over the API's own state.
pub trait RequestModelSeam {
    /// `SparkRequestModelSpeculatorIsValid` — the C validates the provider's
    /// ABI header; the Rust seam validates the implementation's own
    /// invariants.
    fn is_valid(&self) -> bool;

    /// `SparkRequestModelSlotCanSpeculate` (the provider's policy:
    /// speculation enabled && !DISABLE_SPECULATION && (realtime policy ||
    /// underfilled-decode policy)).
    fn slot_can_speculate(&self, slot: &Slot) -> bool;

    /// `SparkRequestModelGetDraft`.
    fn get_draft(&self, sequence_id: u64) -> Result<DraftResult, RequestApiError>;

    /// `SparkRequestModelMarkVerifierTapsReady`; returns the tap generation.
    fn mark_verifier_taps_ready(
        &mut self,
        request_id: u64,
        sequence_id: u64,
        sequence_position: u64,
    ) -> Result<u64, RequestApiError>;

    /// `SparkRequestModelDefaultSpeculativeTokenCount`.
    fn default_speculative_token_count(&self) -> u32;

    /// `SparkGlm52DsparkEnsureDraft` (reached through
    /// `SparkRequestModelPrepareDraftForSlot` in C).
    fn ensure_draft(&mut self, request: &DraftRequest) -> Result<(), RequestApiError>;

    /// `SparkRequestModelCompleteVerify`.
    fn complete_verify(
        &mut self,
        sequence_id: u64,
        verify_result: &VerifyResult,
    ) -> Result<(), RequestApiError>;

    /// `SparkRequestModelResolveVerifierTokens` (the drafter's
    /// sequential-match resolver; MTP tree drafts bypass the seam — see
    /// [`mtp_tree_resolve`]).
    fn resolve_verifier_tokens(
        &mut self,
        draft_token_ids: &[u32],
        verifier_token_ids: &[u32],
    ) -> Result<VerifyResult, RequestApiError>;

    /// `SparkRequestModelCancelSequence`.
    fn cancel_sequence(&mut self, sequence_id: u64) -> Result<(), RequestApiError>;
}

/// `serving/spark_request_model_null.c` — the null provider (fail-closed).
#[derive(Debug, Default)]
pub struct NullSpeculator;

impl RequestModelSeam for NullSpeculator {
    fn is_valid(&self) -> bool {
        true
    }

    fn slot_can_speculate(&self, _slot: &Slot) -> bool {
        false
    }

    fn get_draft(&self, _sequence_id: u64) -> Result<DraftResult, RequestApiError> {
        Err(RequestApiError::NotFound)
    }

    fn mark_verifier_taps_ready(
        &mut self,
        _request_id: u64,
        _sequence_id: u64,
        _sequence_position: u64,
    ) -> Result<u64, RequestApiError> {
        Err(RequestApiError::NotFound)
    }

    fn default_speculative_token_count(&self) -> u32 {
        0
    }

    fn ensure_draft(&mut self, _request: &DraftRequest) -> Result<(), RequestApiError> {
        Err(RequestApiError::NotFound)
    }

    fn complete_verify(
        &mut self,
        _sequence_id: u64,
        _verify_result: &VerifyResult,
    ) -> Result<(), RequestApiError> {
        Ok(())
    }

    fn resolve_verifier_tokens(
        &mut self,
        _draft_token_ids: &[u32],
        _verifier_token_ids: &[u32],
    ) -> Result<VerifyResult, RequestApiError> {
        Err(RequestApiError::NotFound)
    }

    fn cancel_sequence(&mut self, _sequence_id: u64) -> Result<(), RequestApiError> {
        Ok(())
    }
}

/// `SparkMtpTreeResolution` (the resolver output).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MtpTreeResolution {
    pub path_id: u32,
    pub accepted_token_count: u32,
    pub committed_token_count: u32,
    pub fallback_row_index: u32,
}

/// `SparkMtpTreeResolve` (`include/sparkpipe/spark_mtp_tree.h`): walk the
/// fixed draft tree, matching each verifier row's token against the
/// candidate children. Token ids are validated against `output_vocab_count`
/// (`SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT` in C; a [`super::Configuration`]
/// field here).
pub fn mtp_tree_resolve(
    candidate_token_ids: &[u32; mtp_tree::CANDIDATE_COUNT as usize],
    verifier_token_ids: &[u32; mtp_tree::VERIFIER_ROW_COUNT as usize],
    output_vocab_count: u32,
) -> Result<MtpTreeResolution, RequestApiError> {
    for &token_id in candidate_token_ids.iter() {
        if token_id >= output_vocab_count {
            return Err(RequestApiError::InvalidArgument);
        }
    }
    for &token_id in verifier_token_ids.iter() {
        if token_id >= output_vocab_count {
            return Err(RequestApiError::InvalidArgument);
        }
    }
    let mut current_row = 0u32; // SPARK_MODEL_MTP_TREE_VERIFIER_INPUT_ROW
    let mut path_id = mtp_tree::RESOLUTION_NONE;
    let mut matched = true;
    while matched {
        matched = false;
        let node = mtp_tree::node_at(current_row).ok_or(RequestApiError::InternalError)?;
        for child_offset in 0..node.child_count {
            let child = mtp_tree::node_at(u32::from(node.child_row_base) + u32::from(child_offset))
                .ok_or(RequestApiError::InternalError)?;
            if verifier_token_ids[current_row as usize]
                != candidate_token_ids[child.candidate_index as usize]
            {
                continue;
            }
            current_row = u32::from(node.child_row_base) + u32::from(child_offset);
            path_id = current_row;
            matched = true;
            break;
        }
    }
    let accepted_token_count = mtp_tree::accepted_token_count(path_id);
    Ok(MtpTreeResolution {
        path_id,
        accepted_token_count,
        committed_token_count: accepted_token_count + 1,
        fallback_row_index: mtp_tree::fallback_row_index(path_id),
    })
}

// ---------------------------------------------------------------------------
// RequestApi MTP / dspark policy (ports of the C statics + public entry
// points that operate on slots).
// ---------------------------------------------------------------------------

impl RequestApi {
    /// `SparkRequestApiMtpResolvedRequestCount` — `(resolved_request_count,
    /// committed_token_count)`.
    pub(crate) fn mtp_resolved_request_count(&self) -> (u64, u64) {
        let proposed_token_count =
            self.mtp_accepted_draft_token_count + self.mtp_rejected_token_count;
        // A partially-resolved in-flight cycle leaves a non-zero remainder;
        // floor the completed cycle count instead of discarding every
        // sample (see the C comment).
        (
            proposed_token_count / u64::from(mtp_tree::CANDIDATE_COUNT),
            self.mtp_committed_token_count,
        )
    }

    /// `SparkRequestModelMtpExpectedCommittedTokensScaled`.
    fn mtp_expected_committed_tokens_scaled(&self) -> u64 {
        let (resolved_request_count, committed_token_count) = self.mtp_resolved_request_count();
        let weighted_committed_token_count = committed_token_count
            + MTP_UTILITY_PRIOR_SAMPLE_COUNT * MTP_UTILITY_PRIOR_COMMITTED_TOKEN_COUNT;
        let weighted_request_count = resolved_request_count + MTP_UTILITY_PRIOR_SAMPLE_COUNT;
        weighted_committed_token_count * MTP_UTILITY_SCALE / weighted_request_count
    }

    /// `SparkRequestModelMtpOutranksPlainDecode` (the GLM52 request-model
    /// shim's analytic comparison).
    pub(crate) fn mtp_outranks_plain_decode(
        &self,
        plain_request_count: u32,
        mtp_request_count: u32,
    ) -> bool {
        if plain_request_count == 0 || mtp_request_count == 0 {
            return false;
        }
        if self.configuration_flags & CONFIGURATION_FLAG_MTP_FORCE_ENABLE != 0 {
            return true;
        }
        let scheduler = self.scheduler.borrow();
        let plain_work_ns = match scheduler.estimate_decode_work_ns(
            plain_request_count,
            1,
            self.decode_execution_row_capacity,
        ) {
            Ok(work) => work,
            Err(_) => return false,
        };
        let mtp_work_ns = match scheduler.estimate_decode_work_ns(
            mtp_request_count,
            mtp_tree::VERIFIER_ROW_COUNT,
            self.decode_execution_row_capacity,
        ) {
            Ok(work) => work,
            Err(_) => return false,
        };
        // Compare plain decode against the full MTP cycle: the verify batch
        // plus the draft-chain dispatches that produced the candidates (see
        // the C comment).
        if plain_work_ns > u64::MAX / MTP_DRAFT_CHAIN_WORK_MULTIPLIER {
            return false;
        }
        let draft_chain_work_ns = plain_work_ns * MTP_DRAFT_CHAIN_WORK_MULTIPLIER;
        if mtp_work_ns > u64::MAX - draft_chain_work_ns {
            return false;
        }
        let mtp_cycle_work_ns = mtp_work_ns + draft_chain_work_ns;
        let mtp_expected_tokens_scaled = self.mtp_expected_committed_tokens_scaled();
        // The C multiplies in unbounded u64 (wrap); mirror that.
        mtp_expected_tokens_scaled
            .wrapping_mul(u64::from(mtp_request_count))
            .wrapping_mul(plain_work_ns)
            > MTP_UTILITY_MARGIN_SCALE
                .wrapping_mul(u64::from(plain_request_count))
                .wrapping_mul(mtp_cycle_work_ns)
    }

    /// `SparkRequestApiPendingSpeculativeTokenCount` (the C static).
    pub(crate) fn pending_speculative_token_count(
        &self,
        slot_index: u32,
    ) -> Result<u32, RequestApiError> {
        let slot = &self.slots[slot_index as usize];
        if !slot.is_schedulable_speculative_verify()
            && slot.state != STATE_RUNNING_SPECULATIVE_VERIFY
        {
            return Ok(0);
        }
        if slot.mtp_draft_token_count != 0 {
            if slot.mtp_draft_token_count > MTP_MAX_DRAFT_TOKEN_COUNT {
                return Err(RequestApiError::InvalidArgument);
            }
            return Ok(slot.mtp_draft_token_count);
        }
        if !self.speculation_is_enabled() {
            return Err(RequestApiError::NotFound);
        }
        let sequence_id = slot.sequence_id;
        let draft = self
            .model_speculator
            .as_ref()
            .ok_or(RequestApiError::NotFound)?
            .get_draft(sequence_id)?;
        if draft.token_count == 0 || draft.token_count > MAX_SPECULATIVE_TOKENS {
            return Err(RequestApiError::InvalidArgument);
        }
        Ok(draft.token_count)
    }

    /// `SparkRequestModelGetSlotMtpDraft` (the shim's slot-draft adapter).
    fn get_slot_mtp_draft(&self, slot_index: u32) -> Result<DraftResult, RequestApiError> {
        let slot = &self.slots[slot_index as usize];
        if slot.mtp_draft_token_count == 0 || slot.mtp_draft_token_count > MTP_MAX_DRAFT_TOKEN_COUNT
        {
            return Err(RequestApiError::NotFound);
        }
        let mut draft =
            DraftResult { token_count: slot.mtp_draft_token_count, ..DraftResult::default() };
        for index in 0..slot.mtp_draft_token_count as usize {
            // The C caps the loop at the draft array's extent (7) even
            // though the slot stores 8; the draft result only holds 7.
            if index >= MAX_SPECULATIVE_TOKENS as usize {
                break;
            }
            draft.token_ids[index] = slot.mtp_draft_token_ids[index];
            draft.confidence_milli[index] = 1000; // SPARK_GLM52_DSPARK_CONFIDENCE_MILLI_ONE
        }
        Ok(draft)
    }

    /// `SparkRequestModelGetSlotSpeculativeDraft` (the shim's source
    /// selection: MTP first unless the drafter is preferred).
    pub(crate) fn get_slot_speculative_draft(
        &self,
        slot_index: u32,
        preferred_source: u32,
    ) -> Result<(DraftResult, u32), RequestApiError> {
        let get_dspark_draft = |api: &RequestApi| -> Result<DraftResult, RequestApiError> {
            if !api.speculation_is_enabled() {
                return Err(RequestApiError::InvalidArgument);
            }
            let sequence_id = api.slots[slot_index as usize].sequence_id;
            api.model_speculator
                .as_ref()
                .ok_or(RequestApiError::InvalidArgument)?
                .get_draft(sequence_id)
        };
        if preferred_source != SPECULATIVE_SOURCE_DRAFTER {
            match self.get_slot_mtp_draft(slot_index) {
                Ok(draft) => return Ok((draft, SPECULATIVE_SOURCE_MTP)),
                Err(status) => {
                    if preferred_source == SPECULATIVE_SOURCE_MTP {
                        return Err(status);
                    }
                    {
                        let draft = get_dspark_draft(self)?;
                        return Ok((draft, SPECULATIVE_SOURCE_DRAFTER));
                    }
                }
            }
        }
        if let Ok(draft) = get_dspark_draft(self) {
            return Ok((draft, SPECULATIVE_SOURCE_DRAFTER));
        }
        match self.get_slot_mtp_draft(slot_index) {
            Ok(draft) => Ok((draft, SPECULATIVE_SOURCE_MTP)),
            Err(status) => Err(status),
        }
    }

    /// `SparkRequestApiDiscardMtpDraft`.
    pub(crate) fn discard_mtp_draft(&mut self, slot_index: u32) {
        let slot = &mut self.slots[slot_index as usize];
        if slot.state != STATE_READY_SPECULATIVE_VERIFY || slot.mtp_draft_token_count == 0 {
            return;
        }
        slot.mtp_draft_token_ids = [0; MTP_MAX_DRAFT_TOKEN_COUNT as usize];
        slot.mtp_draft_token_count = 0;
        slot.state = STATE_READY_DECODE;
    }

    /// `SparkRequestModelPrepareDraftForSlot` (the shim's draft preparation:
    /// mark verifier taps ready, then ensure a draft sized to the remaining
    /// decode budget).
    pub(crate) fn prepare_draft_for_slot(
        &mut self,
        slot_index: u32,
    ) -> Result<(), RequestApiError> {
        if !self.slot_can_speculate(slot_index)
            || self.slots[slot_index as usize].remaining_decode_budget() == 0
        {
            return Err(RequestApiError::NotFound);
        }
        let (request_id, sequence_id, sequence_position) = {
            let slot = &self.slots[slot_index as usize];
            (
                slot.request_id,
                slot.sequence_id,
                u64::from(slot.computed_prompt_token_count)
                    + u64::from(slot.completed_decode_token_count),
            )
        };
        let tap_generation = self
            .model_speculator
            .as_mut()
            .ok_or(RequestApiError::NotFound)?
            .mark_verifier_taps_ready(request_id, sequence_id, sequence_position)?;

        let default_token_count = self
            .model_speculator
            .as_ref()
            .ok_or(RequestApiError::NotFound)?
            .default_speculative_token_count();
        let mut requested_token_count = default_token_count;
        let remaining_budget = self.slots[slot_index as usize].remaining_decode_budget();
        if requested_token_count + 1 > remaining_budget {
            requested_token_count = remaining_budget - 1;
        }
        if requested_token_count == 0 {
            return Err(RequestApiError::NotFound);
        }
        let priority = self.slots[slot_index as usize].priority;
        let draft_request = DraftRequest {
            requested_token_count,
            priority,
            request_id,
            sequence_id,
            sequence_position,
            tap_generation,
        };
        let status = self
            .model_speculator
            .as_mut()
            .ok_or(RequestApiError::NotFound)?
            .ensure_draft(&draft_request);
        if status.is_ok() {
            self.slots[slot_index as usize].state = STATE_READY_SPECULATIVE_VERIFY;
            self.dspark_draft_ready_count += 1;
        }
        status
    }

    /// `SparkRequestApiDecodeBatchMtpBudget`.
    pub(crate) fn decode_batch_mtp_budget(&self, selected_slots: &[u32]) -> u32 {
        if !self.mtp_commit_is_enabled()
            || selected_slots.is_empty()
            || !self
                .mtp_outranks_plain_decode(selected_slots.len() as u32, selected_slots.len() as u32)
        {
            return 0;
        }
        let mut budget = MTP_MAX_DRAFT_TOKEN_COUNT;
        for &slot_index in selected_slots {
            let slot = &self.slots[slot_index as usize];
            if slot.flags & REQUEST_FLAG_DISABLE_SPECULATION != 0
                || slot.remaining_decode_budget() < mtp_tree::MAX_COMMITTED_TOKEN_COUNT + 1
            {
                return 0;
            }
            let lane_budget = mtp_tree::CANDIDATE_COUNT;
            if slot.mtp_next_draft_token_budget != lane_budget {
                return 0;
            }
            if lane_budget < budget {
                budget = lane_budget;
            }
        }
        budget
    }

    /// `SparkRequestApiConsumeDecodeBudget`.
    pub(crate) fn consume_decode_budget(slot: &mut Slot, committed_token_count: u32) {
        let mut consumed_token_count = 0u32;
        while consumed_token_count < committed_token_count
            && slot.remaining_thinking_token_budget != 0
        {
            slot.remaining_thinking_token_budget -= 1;
            consumed_token_count += 1;
        }
        while consumed_token_count < committed_token_count
            && slot.remaining_output_token_budget != 0
        {
            slot.remaining_output_token_budget -= 1;
            consumed_token_count += 1;
        }
    }

    /// `SparkRequestApiFinishSlotAfterDecode`. Errors from draft preparation
    /// are swallowed after restoring `READY_DECODE` (the C ignores them).
    pub(crate) fn finish_slot_after_decode(&mut self, slot_index: u32, committed_token_count: u32) {
        {
            let slot = &mut self.slots[slot_index as usize];
            slot.mtp_resolution_base_position = 0;
            slot.mtp_resolution_proposed_token_count = 0;
            slot.mtp_resolution_accepted_token_count = 0;
            slot.mtp_resolution_committed_token_count = 0;
            slot.mtp_resolution_path_id = mtp_tree::RESOLUTION_NONE;
            if slot.mtp_next_draft_token_budget == 0 && slot.mtp_probe_countdown != 0 {
                if slot.mtp_probe_countdown > committed_token_count {
                    slot.mtp_probe_countdown -= committed_token_count;
                } else {
                    slot.mtp_probe_countdown = 0;
                    slot.mtp_next_draft_token_budget = mtp_tree::CANDIDATE_COUNT;
                }
            }
            Self::consume_decode_budget(slot, committed_token_count);
            slot.completed_decode_token_count += committed_token_count;
        }
        self.running_request_count -= 1;
        let slot = &self.slots[slot_index as usize];
        if slot.remaining_thinking_token_budget == 0 && slot.remaining_output_token_budget == 0 {
            self.slots[slot_index as usize].state = STATE_COMPLETED;
            self.completed_request_count += 1;
            return;
        }
        self.slots[slot_index as usize].state = STATE_READY_DECODE;
        if let Err(error) = self.prepare_draft_for_slot(slot_index) {
            if error != RequestApiError::NotFound {
                self.slots[slot_index as usize].state = STATE_READY_DECODE;
            }
        }
    }

    /// `SparkRequestApiFinishSlotAfterSpeculativeVerify`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finish_slot_after_speculative_verify(
        &mut self,
        slot_index: u32,
        proposed_token_count: u32,
        accepted_draft_token_count: u32,
        committed_token_count: u32,
        fallback_token_id: u32,
        resolution_path_id: u32,
        mtp_verify: bool,
    ) -> Result<(), RequestApiError> {
        if proposed_token_count == 0
            || proposed_token_count > MAX_SPECULATIVE_TOKENS
            || accepted_draft_token_count > proposed_token_count
            || committed_token_count == 0
            || committed_token_count > proposed_token_count + 1
            || committed_token_count > self.slots[slot_index as usize].remaining_decode_budget()
        {
            return Err(RequestApiError::InvalidArgument);
        }
        let mut resolution_base_position = 0u64;
        if mtp_verify {
            let slot = &self.slots[slot_index as usize];
            if proposed_token_count != mtp_tree::CANDIDATE_COUNT
                || accepted_draft_token_count > mtp_tree::CONTEXT_EXTENSION
                || committed_token_count != accepted_draft_token_count + 1
                || committed_token_count > mtp_tree::MAX_COMMITTED_TOKEN_COUNT
                || resolution_path_id >= mtp_tree::RESOLUTION_COUNT
                || mtp_tree::accepted_token_count(resolution_path_id) != accepted_draft_token_count
                || slot.computed_prompt_token_count == 0
            {
                return Err(RequestApiError::InvalidArgument);
            }
            resolution_base_position = u64::from(slot.computed_prompt_token_count)
                + u64::from(slot.completed_decode_token_count)
                - 1;
        }

        let mut verify_result = VerifyResult {
            proposed_token_count,
            accepted_draft_token_count,
            committed_token_count,
            fallback_token_id,
            ..VerifyResult::default()
        };
        if accepted_draft_token_count == proposed_token_count {
            verify_result.flags |= VERIFY_RESULT_FLAG_ACCEPTED_ALL;
        } else {
            verify_result.flags |= VERIFY_RESULT_FLAG_REJECTED;
        }

        if mtp_verify {
            {
                let slot = &mut self.slots[slot_index as usize];
                if slot.mtp_draft_token_count != proposed_token_count {
                    return Err(RequestApiError::InvalidArgument);
                }
                // Exact C arithmetic: int32 truncation toward zero.
                let ema_delta =
                    (committed_token_count * 1000) as i32 - slot.mtp_commit_ema_milli as i32;
                slot.mtp_commit_ema_milli =
                    (slot.mtp_commit_ema_milli as i32 + ema_delta / MTP_COMMIT_EMA_DIVISOR) as u32;
                if slot.mtp_commit_ema_milli < MTP_SUPPRESS_THRESHOLD_MILLI {
                    slot.mtp_next_draft_token_budget = 0;
                    slot.mtp_probe_countdown = MTP_REPROBE_INTERVAL;
                } else {
                    slot.mtp_next_draft_token_budget = mtp_tree::CANDIDATE_COUNT;
                    slot.mtp_probe_countdown = 0;
                }
                slot.mtp_resolution_base_position = resolution_base_position;
                slot.mtp_resolution_proposed_token_count = proposed_token_count;
                slot.mtp_resolution_accepted_token_count = accepted_draft_token_count;
                slot.mtp_resolution_committed_token_count = committed_token_count;
                slot.mtp_resolution_path_id = resolution_path_id;
                slot.mtp_draft_token_ids = [0; MTP_MAX_DRAFT_TOKEN_COUNT as usize];
                slot.mtp_draft_token_count = 0;
            }
            self.mtp_accepted_draft_token_count += u64::from(accepted_draft_token_count);
            self.mtp_committed_token_count += u64::from(committed_token_count);
            if accepted_draft_token_count < proposed_token_count {
                self.mtp_rejected_token_count +=
                    u64::from(proposed_token_count - accepted_draft_token_count);
            }
        } else {
            {
                let slot = &mut self.slots[slot_index as usize];
                slot.mtp_resolution_base_position = 0;
                slot.mtp_resolution_proposed_token_count = 0;
                slot.mtp_resolution_accepted_token_count = 0;
                slot.mtp_resolution_committed_token_count = 0;
                if resolution_path_id != mtp_tree::RESOLUTION_NONE {
                    return Err(RequestApiError::InvalidArgument);
                }
                slot.mtp_resolution_path_id = mtp_tree::RESOLUTION_NONE;
                if slot.mtp_draft_token_count != 0 {
                    slot.mtp_draft_token_ids = [0; MTP_MAX_DRAFT_TOKEN_COUNT as usize];
                    slot.mtp_draft_token_count = 0;
                    if self.mtp_draft_ready_count != 0 {
                        self.mtp_draft_ready_count -= 1;
                    }
                }
            }
            let sequence_id = self.slots[slot_index as usize].sequence_id;
            self.model_speculator
                .as_mut()
                .ok_or(RequestApiError::NotFound)?
                .complete_verify(sequence_id, &verify_result)?;
            self.dspark_accepted_draft_token_count += u64::from(accepted_draft_token_count);
            self.dspark_committed_token_count += u64::from(committed_token_count);
            if accepted_draft_token_count < proposed_token_count {
                self.dspark_rejected_token_count +=
                    u64::from(proposed_token_count - accepted_draft_token_count);
            }
        }

        {
            let slot = &mut self.slots[slot_index as usize];
            Self::consume_decode_budget(slot, committed_token_count);
            slot.completed_decode_token_count += committed_token_count;
        }
        self.running_request_count -= 1;

        let slot = &self.slots[slot_index as usize];
        if slot.remaining_thinking_token_budget == 0 && slot.remaining_output_token_budget == 0 {
            self.slots[slot_index as usize].state = STATE_COMPLETED;
            self.completed_request_count += 1;
            return Ok(());
        }
        self.slots[slot_index as usize].state = STATE_READY_DECODE;
        if let Err(error) = self.prepare_draft_for_slot(slot_index) {
            if error != RequestApiError::NotFound {
                self.slots[slot_index as usize].state = STATE_READY_DECODE;
            }
        }
        Ok(())
    }

    /// `SparkRequestApiArmMtpVerifyDispatch`.
    pub fn arm_mtp_verify_dispatch(
        &mut self,
        completed_decode_dispatch: &Dispatch,
        draft_token_ids: &[u32],
        lane_stride: u32,
        draft_token_count: u32,
    ) -> Result<(), RequestApiError> {
        self.validate()?;
        let dispatch_is_mtp_producer = completed_decode_dispatch.kind == DispatchKind::DecodeBatch
            && completed_decode_dispatch.flags & DISPATCH_FLAG_MTP_COMMIT != 0;
        let dispatch_is_mtp_verify = completed_decode_dispatch.kind
            == DispatchKind::SpeculativeVerifyBatch
            && completed_decode_dispatch.flags & DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY != 0;
        if !completed_decode_dispatch.accepted
            || (!dispatch_is_mtp_producer && !dispatch_is_mtp_verify)
            || completed_decode_dispatch.request_count == 0
            || completed_decode_dispatch.request_count > MAX_DISPATCH_REQUEST_COUNT
            || draft_token_ids.is_empty()
            || draft_token_count == 0
            || draft_token_count > MTP_MAX_DRAFT_TOKEN_COUNT
            || (dispatch_is_mtp_producer
                && draft_token_count > completed_decode_dispatch.mtp_draft_token_budget)
            || lane_stride < draft_token_count
            || (draft_token_ids.len() as u64)
                < u64::from(completed_decode_dispatch.request_count - 1) * u64::from(lane_stride)
                    + u64::from(draft_token_count)
        {
            return Err(RequestApiError::InvalidArgument);
        }
        if draft_token_count != mtp_tree::CANDIDATE_COUNT
            || (dispatch_is_mtp_producer
                && completed_decode_dispatch.mtp_draft_token_budget != mtp_tree::CANDIDATE_COUNT)
            || (dispatch_is_mtp_verify
                && (completed_decode_dispatch.flags & DISPATCH_FLAG_MTP_TREE_VERIFY == 0
                    || completed_decode_dispatch.mtp_draft_token_budget
                        != mtp_tree::CANDIDATE_COUNT))
        {
            return Err(RequestApiError::ModuleNotValidated);
        }
        let arm_draft_token_count = mtp_tree::CANDIDATE_COUNT;
        for request_index in 0..completed_decode_dispatch.request_count as usize {
            let slot_index = self
                .find_slot_by_handle(completed_decode_dispatch.request_handles[request_index])
                .ok_or(RequestApiError::InvalidArgument)?;
            let slot = &self.slots[slot_index as usize];
            if slot.state == STATE_COMPLETED
                || slot.remaining_decode_budget() < mtp_tree::MAX_COMMITTED_TOKEN_COUNT
            {
                return Err(RequestApiError::NotFound);
            }
            if (slot.state != STATE_READY_DECODE && slot.state != STATE_READY_SPECULATIVE_VERIFY)
                || slot.mtp_draft_token_count != 0
            {
                return Err(RequestApiError::InvalidArgument);
            }
            if slot.mtp_next_draft_token_budget == 0 {
                return Err(RequestApiError::NotFound);
            }
            if slot.mtp_next_draft_token_budget != mtp_tree::CANDIDATE_COUNT {
                return Err(RequestApiError::ModuleNotValidated);
            }
        }
        for request_index in 0..completed_decode_dispatch.request_count as usize {
            let slot_index = self
                .find_slot_by_handle(completed_decode_dispatch.request_handles[request_index])
                .ok_or(RequestApiError::InvalidArgument)?;
            let base = request_index * lane_stride as usize;
            {
                let slot = &mut self.slots[slot_index as usize];
                slot.mtp_draft_token_ids = [0; MTP_MAX_DRAFT_TOKEN_COUNT as usize];
                slot.mtp_draft_token_ids[..arm_draft_token_count as usize]
                    .copy_from_slice(&draft_token_ids[base..base + arm_draft_token_count as usize]);
                slot.mtp_draft_token_count = arm_draft_token_count;
            }
            if self.slots[slot_index as usize].state == STATE_READY_SPECULATIVE_VERIFY
                && self.dspark_draft_ready_count != 0
            {
                self.dspark_draft_ready_count -= 1;
            }
            self.slots[slot_index as usize].state = STATE_READY_SPECULATIVE_VERIFY;
            self.mtp_draft_ready_count += 1;
        }
        Ok(())
    }

    /// `SparkRequestApiResolveSpeculativeVerifyDispatch`: writes the resolved
    /// accepted/committed/fallback/path metadata back into `dispatch`.
    pub fn resolve_speculative_verify_dispatch(
        &mut self,
        dispatch: &mut Dispatch,
        verifier_token_ids: &[u32],
        lane_stride: u32,
        verifier_token_count: u32,
    ) -> Result<(), RequestApiError> {
        self.validate()?;
        if !dispatch.accepted
            || dispatch.kind != DispatchKind::SpeculativeVerifyBatch
            || dispatch.request_count == 0
            || dispatch.request_count > MAX_DISPATCH_REQUEST_COUNT
            || dispatch.speculative_token_count == 0
            || dispatch.speculative_token_count > MAX_SPECULATIVE_TOKENS
            || verifier_token_ids.is_empty()
            || dispatch.speculative_verifier_token_count == 0
            || verifier_token_count != dispatch.speculative_verifier_token_count
            || lane_stride < verifier_token_count
            || (verifier_token_ids.len() as u64)
                < u64::from(dispatch.request_count - 1) * u64::from(lane_stride)
                    + u64::from(verifier_token_count)
        {
            return Err(RequestApiError::InvalidArgument);
        }
        for request_index in 0..dispatch.request_count as usize {
            let lane_base = request_index * lane_stride as usize;
            let lane_verifier =
                &verifier_token_ids[lane_base..lane_base + verifier_token_count as usize];
            let mut resolution_path_id = mtp_tree::RESOLUTION_NONE;
            let verify_result: VerifyResult;
            if dispatch.flags & DISPATCH_FLAG_MTP_TREE_VERIFY != 0 {
                if dispatch.flags & DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY == 0
                    || dispatch.speculative_token_count != mtp_tree::CANDIDATE_COUNT
                    || dispatch.speculative_verifier_token_count != mtp_tree::VERIFIER_ROW_COUNT
                    || dispatch.speculative_max_committed_token_count
                        != mtp_tree::MAX_COMMITTED_TOKEN_COUNT
                {
                    return Err(RequestApiError::InvalidArgument);
                }
                // SparkRequestApiResolveMtpTreeVerifierTokens.
                let mut candidate_token_ids = [0u32; mtp_tree::CANDIDATE_COUNT as usize];
                candidate_token_ids.copy_from_slice(
                    &dispatch.speculative_draft_token_ids[request_index]
                        [..mtp_tree::CANDIDATE_COUNT as usize],
                );
                let mut verifier_row = [0u32; mtp_tree::VERIFIER_ROW_COUNT as usize];
                verifier_row.copy_from_slice(lane_verifier);
                let resolution =
                    mtp_tree_resolve(&candidate_token_ids, &verifier_row, self.output_vocab_count)?;
                resolution_path_id = resolution.path_id;
                verify_result = VerifyResult {
                    flags: VERIFY_RESULT_FLAG_REJECTED,
                    proposed_token_count: mtp_tree::CANDIDATE_COUNT,
                    accepted_draft_token_count: resolution.accepted_token_count,
                    committed_token_count: resolution.committed_token_count,
                    fallback_token_id: lane_verifier[resolution.fallback_row_index as usize],
                };
            } else {
                let speculator = self.model_speculator.as_mut().ok_or(RequestApiError::NotFound)?;
                verify_result = speculator.resolve_verifier_tokens(
                    &dispatch.speculative_draft_token_ids[request_index]
                        [..dispatch.speculative_token_count as usize],
                    lane_verifier,
                )?;
            }
            dispatch.speculative_accepted_token_counts[request_index] =
                verify_result.accepted_draft_token_count;
            dispatch.speculative_committed_token_counts[request_index] =
                verify_result.committed_token_count;
            dispatch.speculative_fallback_token_ids[request_index] =
                verify_result.fallback_token_id;
            dispatch.speculative_resolution_path_ids[request_index] = resolution_path_id;
        }
        Ok(())
    }
}
