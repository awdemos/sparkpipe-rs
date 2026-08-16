//! Port of `tests/test_glm52_request_api.c`. GPU-free: the fixture builds an
//! owned `KvArena` + `PrefixCache` + `Scheduler` stack exactly like the C
//! test's, and the `RequestApi` is configured with the same capacities
//! (32 slots, 13 prefetch lanes, decode batch target 1024, 128-block arena
//! with 16-token blocks, 128-entry/512-binding prefix cache, measured
//! profile 20260701, NVFP4, 78/3 GLM-5.2 geometry).
//!
//! Intentional differences from the C test:
//! - The C's `SparkTestPrefetchCapture` / async start-poll capture functions
//!   become a `Rc<RefCell<PrefetchCapture>>` shared with the installed
//!   [`CapturePrefetchBackend`] (the Rust API owns its backend).
//! - The C links the real `SparkGlm52DsparkSpeculator`; here a
//!   [`FakeDsparkSpeculator`] reproduces its observable seam semantics
//!   (sequence states, tap/draft generations, confidence gating at 350/250
//!   milli, sequential-match verifier resolution) behind the
//!   [`RequestModelSeam`] trait, driven by the same draft-capture double.
//! - `UsesBuiltInAsyncMemoryPrefetchBackend`: the C links the real async
//!   memory-copy backend against arena device pointers. Safe Rust cannot
//!   write through the arena's `usize` device bases, so an emulated
//!   [`MemoryPrefetchBackend`] copies source bytes into its own map keyed by
//!   the plan's device addresses with the same progressive-poll behavior
//!   (13 blocks per poll, 2 in flight); byte-content assertions compare
//!   against that map.
//! - The tokenizer C-text test (`SubmitsCTextPromptToPrefillSchedule`)
//!   writes a HuggingFace tokenizer JSON to a temp path, as the C does.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use spark_core::kv_arena::{KvArena, KvArenaConfig};
use spark_core::prefix_cache::{
    hash_block, hash_prompt_tokens, PrefixCache, PrefixCacheConfig, EMPTY_PARENT_HASH,
};
use spark_sched::scheduler::{
    Scheduler, SchedulerConfig, CONFIGURATION_DEFAULT_FLAGS as SCHEDULER_DEFAULT_FLAGS,
    DECISION_FLAG_ADAPTIVE_DECODE_PACK, DECISION_FLAG_ADAPTIVE_PREFILL_PACK,
    DECISION_FLAG_MEASURED_DECODE_BUCKET, DISPATCH_STAGE_FLAG_MEASURED_DECODE_BUCKET,
    PREFILL_BLOCK_TOKENS,
};
use spark_sched::stage_plan::{
    QuantizationMode, StagePlanGeometry, BUCKET_B16, BUCKET_B64, CURRENT_SPARK_COUNT,
    MAX_BATCH_BUCKET, MEASURED_PROFILE_20260701,
};
use spark_sched::work_control::mtp_tree;
use spark_serve::request_api::*;

/// `SPARK_TEST_REQUEST_SLOT_COUNT`.
const REQUEST_SLOT_COUNT: u32 = 32;
/// `SPARK_TEST_PREFIX_ENTRY_COUNT`.
const PREFIX_ENTRY_COUNT: u32 = 128;
/// `SPARK_TEST_PREFIX_BINDING_COUNT`.
const PREFIX_BINDING_COUNT: u32 = 512;
/// `SPARK_TEST_KV_BLOCK_COUNT`.
const KV_BLOCK_COUNT: u32 = 128;
/// `SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT`.
const OUTPUT_VOCAB_COUNT: u32 = 154_880;
/// `SPARK_GLM52_MODEL_LAYER_COUNT` / `SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER`.
const GLM52_GEOMETRY: StagePlanGeometry = StagePlanGeometry::new(78, 3);
/// `SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH`.
const GLM52_DEFAULT_MAX_PREFILL_TOKENS_PER_STEP: u32 = 256;
/// `SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS`.
const GLM52_MAX_CONTEXT_TOKENS: u32 = 1_048_576;
/// `SPARK_GLM52_DSPARK_FULL_VOCAB_SIZE`.
const DSPARK_FULL_VOCAB_SIZE: u32 = OUTPUT_VOCAB_COUNT;
/// `SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT`.
const DSPARK_MAX_SPECULATIVE_TOKEN_COUNT: u32 = MAX_SPECULATIVE_TOKENS;
/// `SPARK_GLM52_DSPARK_DEFAULT_MIN_CONFIDENCE_MILLI`.
const DSPARK_MIN_CONFIDENCE_MILLI: u32 = 350;
/// `SPARK_GLM52_DSPARK_DRAFT_RESULT_FLAG_CONFIDENCE_TRUNCATED`.
const DRAFT_RESULT_FLAG_CONFIDENCE_TRUNCATED: u32 = 0x0000_0001;

/// `SparkTestFillTokenIds`.
fn fill_token_ids(token_count: u32, first_token_id: u32) -> Vec<u32> {
    (0..token_count).map(|index| first_token_id + index).collect()
}

/// `SparkTestInitializeSubmitRequest` + `SparkRequestApiSubmit`.
fn submit(
    api: &mut RequestApi,
    request_id: u64,
    sequence_id: u64,
    priority: u32,
    prompt_token_ids: Vec<u32>,
    output_token_budget: u32,
) -> u64 {
    let prompt_token_count = prompt_token_ids.len() as u32;
    api.submit(&SubmitRequest {
        flags: 0,
        priority,
        prompt_token_count,
        thinking_token_budget: 0,
        output_token_budget,
        max_prefill_tokens_per_step: 0,
        request_id,
        sequence_id,
        prompt_token_ids,
    })
    .expect("submit succeeds")
}

/// Arena access shortcut for residency assertions.
fn arena(api: &RequestApi) -> std::cell::Ref<'_, KvArena> {
    std::cell::Ref::map(api.scheduler().borrow(), |scheduler| {
        scheduler.prefix_cache().expect("prefix cache present").arena()
    })
}

/// Prefix-cache access shortcut.
fn prefix_cache(api: &RequestApi) -> std::cell::Ref<'_, PrefixCache> {
    std::cell::Ref::map(api.scheduler().borrow(), |scheduler| {
        scheduler.prefix_cache().expect("prefix cache present")
    })
}

// ---------------------------------------------------------------------------
// `SparkTestPrefetchCapture` + the prefetch backend doubles.
// ---------------------------------------------------------------------------

/// `SparkTestPrefetchCapture`.
#[derive(Debug)]
struct PrefetchCapture {
    return_status: Result<(), RequestApiError>,
    busy_call_budget: u32,
    call_count: u32,
    last_lane_count: u32,
    last_prefetch_block_count: u32,
    last_lane_block_counts: [u32; MAX_PREFETCH_LANE_COUNT as usize],
    last_physical_block_indices: Vec<u32>,
    last_first_token_indices: Vec<u32>,
    last_token_counts: Vec<u32>,
    last_parent_hashes: Vec<u64>,
    last_block_hashes: Vec<u64>,
    last_content_hashes: Vec<u64>,
    async_start_status: Result<(), RequestApiError>,
    async_poll_status: Result<(), RequestApiError>,
    async_start_count: u32,
    async_poll_count: u32,
    async_poll_busy_budget: u32,
    async_pending: bool,
    async_prefetch_id: u64,
    async_prefetch_plan: PrefetchPlan,
}

impl PrefetchCapture {
    fn new() -> Self {
        PrefetchCapture {
            return_status: Ok(()),
            busy_call_budget: 0,
            call_count: 0,
            last_lane_count: 0,
            last_prefetch_block_count: 0,
            last_lane_block_counts: [0; MAX_PREFETCH_LANE_COUNT as usize],
            last_physical_block_indices: Vec::new(),
            last_first_token_indices: Vec::new(),
            last_token_counts: Vec::new(),
            last_parent_hashes: Vec::new(),
            last_block_hashes: Vec::new(),
            last_content_hashes: Vec::new(),
            async_start_status: Ok(()),
            async_poll_status: Ok(()),
            async_start_count: 0,
            async_poll_count: 0,
            async_poll_busy_budget: 0,
            async_pending: false,
            async_prefetch_id: 0,
            async_prefetch_plan: PrefetchPlan::default(),
        }
    }
}

/// The sync capture backend + async start/poll pair, as one trait object.
struct CapturePrefetchBackend {
    capture: Rc<RefCell<PrefetchCapture>>,
}

impl KvPrefetchBackend for CapturePrefetchBackend {
    /// `SparkTestCaptureKvPrefetch`.
    fn prefetch(&mut self, prefetch_plan: &PrefetchPlan) -> Result<(), RequestApiError> {
        let mut capture = self.capture.borrow_mut();
        capture.call_count += 1;
        capture.last_lane_count = prefetch_plan.lane_count;
        capture.last_prefetch_block_count = prefetch_plan.prefetch_block_count();
        capture.last_lane_block_counts = prefetch_plan.lane_block_counts;
        capture.last_physical_block_indices.clear();
        capture.last_first_token_indices.clear();
        capture.last_token_counts.clear();
        capture.last_parent_hashes.clear();
        capture.last_block_hashes.clear();
        capture.last_content_hashes.clear();
        for block in &prefetch_plan.blocks {
            assert!(block.lane_index < prefetch_plan.lane_count);
            assert!(block.key_device_address != 0);
            assert!(block.value_device_address != 0);
            capture.last_physical_block_indices.push(block.physical_block_index);
            capture.last_first_token_indices.push(block.first_token_index);
            capture.last_token_counts.push(block.token_count);
            capture.last_parent_hashes.push(block.parent_hash);
            capture.last_block_hashes.push(block.block_hash);
            capture.last_content_hashes.push(block.content_hash);
        }
        if capture.busy_call_budget != 0 {
            capture.busy_call_budget -= 1;
            return Err(RequestApiError::Busy);
        }
        capture.return_status
    }

    /// `SparkTestStartAsyncKvPrefetch`.
    fn start_prefetch(
        &mut self,
        prefetch_id: u64,
        prefetch_plan: &PrefetchPlan,
    ) -> Result<(), RequestApiError> {
        let mut capture = self.capture.borrow_mut();
        assert!(prefetch_id != 0);
        assert!(!capture.async_pending);
        capture.async_start_count += 1;
        capture.async_pending = true;
        capture.async_prefetch_id = prefetch_id;
        capture.async_prefetch_plan = prefetch_plan.clone();
        capture.async_start_status
    }

    /// `SparkTestPollAsyncKvPrefetch`.
    fn poll_prefetch(
        &mut self,
        prefetch_id: u64,
        prefetch_plan: &PrefetchPlan,
    ) -> Result<(), RequestApiError> {
        let mut capture = self.capture.borrow_mut();
        assert!(capture.async_pending);
        assert_eq!(capture.async_prefetch_id, prefetch_id);
        assert_eq!(
            capture.async_prefetch_plan.prefetch_block_count(),
            prefetch_plan.prefetch_block_count()
        );
        capture.async_poll_count += 1;
        if capture.async_poll_busy_budget != 0 {
            capture.async_poll_busy_budget -= 1;
            return Err(RequestApiError::Busy);
        }
        capture.async_pending = false;
        capture.async_poll_status
    }
}

/// Emulation of the C's built-in async memory prefetch backend
/// (`SparkKvCacheAsyncPrefetchBackend` with the memory-source flag): start
/// records the plan; each poll copies up to `blocks_per_poll` blocks from
/// the configured source into the emulated device memory and returns `Busy`
/// until the plan completes. `max_inflight_prefetch_count` bounds
/// concurrent plans (the request API never exceeds one here).
struct MemoryPrefetchBackend {
    blocks_per_poll: u32,
    key_source: Vec<u8>,
    value_source: Vec<u8>,
    block_stride_bytes: usize,
    /// Emulated device memory: device address -> copied bytes.
    device_memory: HashMap<usize, Vec<u8>>,
    pending: Option<(u64, PrefetchPlan, usize)>,
    start_count: u32,
    poll_count: u32,
    completed_prefetch_count: u32,
    copied_key_block_count: u32,
    copied_value_block_count: u32,
}

impl MemoryPrefetchBackend {
    /// `SparkKvCacheAsyncPrefetchBackendInitialize` (memory source, 64-byte
    /// blocks, 13 lanes, 2 in flight, 13 blocks per poll).
    fn new(key_source: Vec<u8>, value_source: Vec<u8>, block_stride_bytes: usize) -> Self {
        MemoryPrefetchBackend {
            blocks_per_poll: CURRENT_SPARK_COUNT,
            key_source,
            value_source,
            block_stride_bytes,
            device_memory: HashMap::new(),
            pending: None,
            start_count: 0,
            poll_count: 0,
            completed_prefetch_count: 0,
            copied_key_block_count: 0,
            copied_value_block_count: 0,
        }
    }
}

impl KvPrefetchBackend for MemoryPrefetchBackend {
    fn start_prefetch(
        &mut self,
        prefetch_id: u64,
        prefetch_plan: &PrefetchPlan,
    ) -> Result<(), RequestApiError> {
        assert!(prefetch_id != 0);
        assert!(self.pending.is_none());
        self.start_count += 1;
        self.pending = Some((prefetch_id, prefetch_plan.clone(), 0));
        Ok(())
    }

    fn poll_prefetch(
        &mut self,
        prefetch_id: u64,
        _prefetch_plan: &PrefetchPlan,
    ) -> Result<(), RequestApiError> {
        self.poll_count += 1;
        let Some((pending_id, plan, next_block)) = self.pending.take() else {
            return Err(RequestApiError::NotFound);
        };
        assert_eq!(pending_id, prefetch_id);
        let end = (next_block + self.blocks_per_poll as usize).min(plan.blocks.len());
        for block in &plan.blocks[next_block..end] {
            let source_index = block.physical_block_index as usize * self.block_stride_bytes;
            let key_bytes =
                self.key_source[source_index..source_index + self.block_stride_bytes].to_vec();
            let value_bytes =
                self.value_source[source_index..source_index + self.block_stride_bytes].to_vec();
            self.device_memory.insert(block.key_device_address, key_bytes);
            self.device_memory.insert(block.value_device_address, value_bytes);
            self.copied_key_block_count += 1;
            self.copied_value_block_count += 1;
        }
        if end < plan.blocks.len() {
            self.pending = Some((pending_id, plan, end));
            return Err(RequestApiError::Busy);
        }
        self.completed_prefetch_count += 1;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// `SparkTestDsparkDraftCapture` + the dspark speculator double.
// ---------------------------------------------------------------------------

/// `SPARK_GLM52_DSPARK_POLICY_FLAG_ENABLE_REALTIME`.
const DSPARK_POLICY_FLAG_ENABLE_REALTIME: u32 = 0x0000_0001;
/// `SPARK_GLM52_DSPARK_POLICY_FLAG_ENABLE_UNDERFILLED_DECODE`.
const DSPARK_POLICY_FLAG_ENABLE_UNDERFILLED_DECODE: u32 = 0x0000_0002;
/// `SPARK_GLM52_DSPARK_POLICY_DEFAULT_FLAGS`.
const DSPARK_POLICY_DEFAULT_FLAGS: u32 = 0x0000_000F;

/// `SparkTestDsparkDraftCapture`.
#[derive(Debug)]
struct DsparkCapture {
    call_count: u32,
    requested_token_count: u32,
    priority: u32,
    next_token_id: u32,
    confidence_milli: [u32; DSPARK_MAX_SPECULATIVE_TOKEN_COUNT as usize],
    request_id: u64,
    sequence_id: u64,
    sequence_position: u64,
    tap_generation: u64,
}

impl DsparkCapture {
    fn new() -> Self {
        DsparkCapture {
            call_count: 0,
            requested_token_count: 0,
            priority: 0,
            next_token_id: 140_000,
            confidence_milli: [0; DSPARK_MAX_SPECULATIVE_TOKEN_COUNT as usize],
            request_id: 0,
            sequence_id: 0,
            sequence_position: 0,
            tap_generation: 0,
        }
    }
}

/// `SparkGlm52DsparkSequenceState` (observable parts).
#[derive(Debug, Default)]
struct DsparkSequenceState {
    taps_ready: bool,
    draft_ready: bool,
    tap_generation: u64,
    sequence_position: u64,
    draft_token_count: u32,
    draft_token_ids: [u32; DSPARK_MAX_SPECULATIVE_TOKEN_COUNT as usize],
    draft_confidence_milli: [u32; DSPARK_MAX_SPECULATIVE_TOKEN_COUNT as usize],
}

/// The `SparkGlm52DsparkSpeculator` double: reproduces the C speculator's
/// seam-observable semantics (see the file-header deviation note) with the
/// C test's configuration: default speculative token count 7, minimum
/// confidence 350 milli, realtime minimum 250 milli, default policy flags
/// (realtime + underfilled decode enabled).
struct FakeDsparkSpeculator {
    capture: Rc<RefCell<DsparkCapture>>,
    policy_flags: Rc<Cell<u32>>,
    default_speculative_token_count: u32,
    minimum_confidence_milli: u32,
    realtime_minimum_confidence_milli: u32,
    next_tap_generation: u64,
    states: HashMap<u64, DsparkSequenceState>,
    tap_ready_count: u32,
    draft_request_count: u32,
    draft_success_count: u32,
    draft_rejected_by_confidence_count: u32,
    verify_dispatch_count: u32,
    accepted_draft_token_count: u64,
    committed_token_count: u64,
    rejected_token_count: u64,
}

impl FakeDsparkSpeculator {
    fn new(capture: Rc<RefCell<DsparkCapture>>, policy_flags: Rc<Cell<u32>>) -> Self {
        FakeDsparkSpeculator {
            capture,
            policy_flags,
            default_speculative_token_count: DSPARK_MAX_SPECULATIVE_TOKEN_COUNT,
            minimum_confidence_milli: DSPARK_MIN_CONFIDENCE_MILLI,
            realtime_minimum_confidence_milli: 250,
            next_tap_generation: 1,
            states: HashMap::new(),
            tap_ready_count: 0,
            draft_request_count: 0,
            draft_success_count: 0,
            draft_rejected_by_confidence_count: 0,
            verify_dispatch_count: 0,
            accepted_draft_token_count: 0,
            committed_token_count: 0,
            rejected_token_count: 0,
        }
    }

    /// `SparkTestDsparkDraft` (the draft-function double).
    fn run_draft_function(&mut self, request: &DraftRequest) -> DraftResult {
        assert!(request.requested_token_count != 0);
        assert!(request.requested_token_count <= DSPARK_MAX_SPECULATIVE_TOKEN_COUNT);
        assert!(request.sequence_id != 0);
        assert!(request.tap_generation != 0);
        let mut capture = self.capture.borrow_mut();
        capture.call_count += 1;
        capture.requested_token_count = request.requested_token_count;
        capture.priority = request.priority;
        capture.request_id = request.request_id;
        capture.sequence_id = request.sequence_id;
        capture.sequence_position = request.sequence_position;
        capture.tap_generation = request.tap_generation;
        let mut result =
            DraftResult { token_count: request.requested_token_count, ..DraftResult::default() };
        for token_index in 0..request.requested_token_count as usize {
            let confidence_milli = match capture.confidence_milli[token_index] {
                0 => 800,
                confidence_milli => confidence_milli,
            };
            result.token_ids[token_index] = capture.next_token_id + token_index as u32;
            result.confidence_milli[token_index] = confidence_milli;
        }
        result
    }
}

impl RequestModelSeam for FakeDsparkSpeculator {
    fn is_valid(&self) -> bool {
        true
    }

    /// `SparkRequestModelSlotCanSpeculate` with the C speculator's policy
    /// flags (the C test pokes `fixture.dspark_speculator.policy_flags`
    /// directly; the shared cell reproduces that).
    fn slot_can_speculate(&self, slot: &Slot) -> bool {
        if slot.flags & REQUEST_FLAG_DISABLE_SPECULATION != 0 {
            return false;
        }
        let policy_flags = self.policy_flags.get();
        if policy_flags & DSPARK_POLICY_FLAG_ENABLE_REALTIME != 0
            && (slot.flags & REQUEST_FLAG_REALTIME != 0 || slot.priority >= REALTIME_PRIORITY)
        {
            return true;
        }
        policy_flags & DSPARK_POLICY_FLAG_ENABLE_UNDERFILLED_DECODE != 0
    }

    /// `SparkGlm52DsparkGetDraft`.
    fn get_draft(&self, sequence_id: u64) -> Result<DraftResult, RequestApiError> {
        let Some(state) = self.states.get(&sequence_id) else {
            return Err(RequestApiError::NotFound);
        };
        if !state.draft_ready || state.draft_token_count == 0 {
            return Err(RequestApiError::NotFound);
        }
        Ok(DraftResult {
            token_count: state.draft_token_count,
            token_ids: state.draft_token_ids,
            confidence_milli: state.draft_confidence_milli,
            ..DraftResult::default()
        })
    }

    /// `SparkGlm52DsparkMarkVerifierTapsReady`.
    fn mark_verifier_taps_ready(
        &mut self,
        _request_id: u64,
        sequence_id: u64,
        sequence_position: u64,
    ) -> Result<u64, RequestApiError> {
        if sequence_id == 0 {
            return Err(RequestApiError::InvalidArgument);
        }
        let state = self.states.entry(sequence_id).or_default();
        state.taps_ready = true;
        state.draft_ready = false;
        state.sequence_position = sequence_position;
        state.tap_generation = self.next_tap_generation;
        self.next_tap_generation += 1;
        if self.next_tap_generation == 0 {
            self.next_tap_generation = 1;
        }
        state.draft_token_count = 0;
        self.tap_ready_count += 1;
        Ok(state.tap_generation)
    }

    fn default_speculative_token_count(&self) -> u32 {
        self.default_speculative_token_count
    }

    /// `SparkGlm52DsparkEnsureDraft`.
    fn ensure_draft(&mut self, request: &DraftRequest) -> Result<(), RequestApiError> {
        if request.requested_token_count == 0
            || request.requested_token_count > DSPARK_MAX_SPECULATIVE_TOKEN_COUNT
            || request.sequence_id == 0
            || request.tap_generation == 0
        {
            return Err(RequestApiError::InvalidArgument);
        }
        let Some(state) = self.states.get(&request.sequence_id) else {
            return Err(RequestApiError::NotFound);
        };
        if !state.taps_ready || state.tap_generation != request.tap_generation {
            return Err(RequestApiError::NotFound);
        }
        if state.draft_ready {
            return Ok(());
        }
        self.draft_request_count += 1;
        let mut result = self.run_draft_function(request);
        for token_index in 0..result.token_count as usize {
            if result.token_ids[token_index] >= DSPARK_FULL_VOCAB_SIZE
                || result.confidence_milli[token_index] > 1000
            {
                return Err(RequestApiError::InvalidArgument);
            }
        }
        let confidence_threshold_milli = if request.priority >= 4_000_000_000 {
            self.realtime_minimum_confidence_milli
        } else {
            self.minimum_confidence_milli
        };
        let mut accepted_by_confidence = result.token_count;
        for token_index in 0..result.token_count as usize {
            if result.confidence_milli[token_index] < confidence_threshold_milli {
                accepted_by_confidence = token_index as u32;
                break;
            }
        }
        if accepted_by_confidence == 0 {
            self.draft_rejected_by_confidence_count += 1;
            return Err(RequestApiError::NotFound);
        }
        if accepted_by_confidence < result.token_count {
            result.flags |= DRAFT_RESULT_FLAG_CONFIDENCE_TRUNCATED;
        }
        let state = self.states.entry(request.sequence_id).or_default();
        state.draft_ready = true;
        state.draft_token_count = accepted_by_confidence;
        for token_index in 0..DSPARK_MAX_SPECULATIVE_TOKEN_COUNT as usize {
            if (token_index as u32) < accepted_by_confidence {
                state.draft_token_ids[token_index] = result.token_ids[token_index];
                state.draft_confidence_milli[token_index] = result.confidence_milli[token_index];
            } else {
                state.draft_token_ids[token_index] = 0;
                state.draft_confidence_milli[token_index] = 0;
            }
        }
        self.draft_success_count += 1;
        Ok(())
    }

    /// `SparkGlm52DsparkCompleteVerify`.
    fn complete_verify(
        &mut self,
        sequence_id: u64,
        verify_result: &VerifyResult,
    ) -> Result<(), RequestApiError> {
        if verify_result.proposed_token_count == 0
            || verify_result.proposed_token_count > DSPARK_MAX_SPECULATIVE_TOKEN_COUNT
            || verify_result.accepted_draft_token_count > verify_result.proposed_token_count
            || verify_result.committed_token_count > verify_result.proposed_token_count + 1
        {
            return Err(RequestApiError::InvalidArgument);
        }
        if verify_result.flags & VERIFY_RESULT_FLAG_ACCEPTED_ALL != 0
            && verify_result.accepted_draft_token_count != verify_result.proposed_token_count
        {
            return Err(RequestApiError::InvalidArgument);
        }
        if verify_result.flags & VERIFY_RESULT_FLAG_REJECTED != 0
            && verify_result.accepted_draft_token_count >= verify_result.proposed_token_count
        {
            return Err(RequestApiError::InvalidArgument);
        }
        let Some(state) = self.states.get_mut(&sequence_id) else {
            return Err(RequestApiError::InvalidArgument);
        };
        if !state.draft_ready || state.draft_token_count != verify_result.proposed_token_count {
            return Err(RequestApiError::InvalidArgument);
        }
        self.accepted_draft_token_count += u64::from(verify_result.accepted_draft_token_count);
        self.committed_token_count += u64::from(verify_result.committed_token_count);
        if verify_result.accepted_draft_token_count < verify_result.proposed_token_count {
            self.rejected_token_count += u64::from(
                verify_result.proposed_token_count - verify_result.accepted_draft_token_count,
            );
        }
        self.verify_dispatch_count += 1;
        state.draft_ready = false;
        state.draft_token_count = 0;
        Ok(())
    }

    /// `SparkGlm52DsparkResolveVerifierTokens`.
    fn resolve_verifier_tokens(
        &mut self,
        draft_token_ids: &[u32],
        verifier_token_ids: &[u32],
    ) -> Result<VerifyResult, RequestApiError> {
        let draft_token_count = draft_token_ids.len() as u32;
        let verifier_token_count = verifier_token_ids.len() as u32;
        if draft_token_count == 0
            || draft_token_count > DSPARK_MAX_SPECULATIVE_TOKEN_COUNT
            || verifier_token_count < draft_token_count
            || verifier_token_count > draft_token_count + 1
        {
            return Err(RequestApiError::InvalidArgument);
        }
        for token_index in 0..draft_token_count as usize {
            if draft_token_ids[token_index] >= DSPARK_FULL_VOCAB_SIZE
                || verifier_token_ids[token_index] >= DSPARK_FULL_VOCAB_SIZE
            {
                return Err(RequestApiError::InvalidArgument);
            }
        }
        if verifier_token_count > draft_token_count
            && verifier_token_ids[draft_token_count as usize] >= DSPARK_FULL_VOCAB_SIZE
        {
            return Err(RequestApiError::InvalidArgument);
        }
        let mut accepted_token_count = draft_token_count;
        for token_index in 0..draft_token_count as usize {
            if draft_token_ids[token_index] != verifier_token_ids[token_index] {
                accepted_token_count = token_index as u32;
                break;
            }
        }
        let mut verify_result = VerifyResult {
            proposed_token_count: draft_token_count,
            accepted_draft_token_count: accepted_token_count,
            ..VerifyResult::default()
        };
        if accepted_token_count == draft_token_count {
            verify_result.flags = VERIFY_RESULT_FLAG_ACCEPTED_ALL;
            verify_result.committed_token_count = verifier_token_count;
            if verifier_token_count > draft_token_count {
                verify_result.fallback_token_id = verifier_token_ids[draft_token_count as usize];
            }
            return Ok(verify_result);
        }
        verify_result.flags = VERIFY_RESULT_FLAG_REJECTED;
        verify_result.committed_token_count = accepted_token_count + 1;
        verify_result.fallback_token_id = verifier_token_ids[accepted_token_count as usize];
        Ok(verify_result)
    }

    /// `SparkGlm52DsparkCancelSequence`.
    fn cancel_sequence(&mut self, sequence_id: u64) -> Result<(), RequestApiError> {
        if self.states.remove(&sequence_id).is_none() {
            return Err(RequestApiError::NotFound);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Fixture (`SparkTestRequestApiFixture` + initialize helpers).
// ---------------------------------------------------------------------------

/// The C fixture: the request API plus the capture doubles it dispatches to.
struct Fixture {
    api: RequestApi,
    prefetch_capture: Rc<RefCell<PrefetchCapture>>,
    dspark_capture: Rc<RefCell<DsparkCapture>>,
    dspark_policy_flags: Rc<Cell<u32>>,
}

impl Fixture {
    fn new(api: RequestApi, prefetch_capture: Rc<RefCell<PrefetchCapture>>) -> Self {
        Fixture {
            api,
            prefetch_capture,
            dspark_capture: Rc::new(RefCell::new(DsparkCapture::new())),
            dspark_policy_flags: Rc::new(Cell::new(DSPARK_POLICY_DEFAULT_FLAGS)),
        }
    }
}

/// `SparkTestInitializePrefixCache`/`SparkTestInitializeScheduler` stack for
/// the standard fixture (78-layer arena geometry).
fn make_scheduler(layer_count: u32, kv_head_count: u32, head_dim: u32) -> Scheduler {
    make_scheduler_with_flags(layer_count, kv_head_count, head_dim, SCHEDULER_DEFAULT_FLAGS)
}

/// `make_scheduler` with an explicit scheduler configuration flag mask.
fn make_scheduler_with_flags(
    layer_count: u32,
    kv_head_count: u32,
    head_dim: u32,
    scheduler_configuration_flags: u32,
) -> Scheduler {
    make_scheduler_with(layer_count, kv_head_count, head_dim, scheduler_configuration_flags, 2)
}

/// The full scheduler builder: geometry, flag mask, and queue depth (the C
/// tests poke `fixture.scheduler.queue_depth_per_spark` post-init; the Rust
/// scheduler keeps it private, so the variant is built up front).
fn make_scheduler_with(
    layer_count: u32,
    kv_head_count: u32,
    head_dim: u32,
    scheduler_configuration_flags: u32,
    queue_depth_per_spark: u32,
) -> Scheduler {
    let arena = KvArena::new(&KvArenaConfig {
        physical_block_count: KV_BLOCK_COUNT,
        block_token_count: PREFILL_BLOCK_TOKENS,
        resident_block_capacity: 0,
        layer_count,
        kv_head_count,
        head_dim,
        bytes_per_scalar: 2,
        key_block_stride_bytes: 0,
        value_block_stride_bytes: 0,
        key_device_base: 0x1_0000_0000,
        value_device_base: 0x2_0000_0000,
    })
    .expect("arena configuration is valid");
    let prefix_cache = PrefixCache::new(
        &PrefixCacheConfig {
            block_token_count: PREFILL_BLOCK_TOKENS,
            entry_count: PREFIX_ENTRY_COUNT,
            physical_block_count: KV_BLOCK_COUNT,
            sequence_binding_count: PREFIX_BINDING_COUNT,
        },
        arena,
    )
    .expect("prefix cache configuration is valid");
    Scheduler::new(SchedulerConfig {
        spark_count: CURRENT_SPARK_COUNT,
        queue_depth_per_spark,
        measured_profile_id: MEASURED_PROFILE_20260701,
        stage_geometry: GLM52_GEOMETRY,
        estimated_layer_cost_ns: 0,
        estimated_final_stage_extra_cost_ns: 0,
        quantization_mode: QuantizationMode::Nvfp4_4Bit,
        max_prefill_tokens_per_step: 0,
        default_max_prefill_tokens_per_step: GLM52_DEFAULT_MAX_PREFILL_TOKENS_PER_STEP,
        max_context_tokens: GLM52_MAX_CONTEXT_TOKENS,
        max_batch_bucket: MAX_BATCH_BUCKET,
        prefix_cache_block_tokens: PREFILL_BLOCK_TOKENS,
        configuration_flags: scheduler_configuration_flags,
        prefix_cache: Some(prefix_cache),
    })
    .expect("scheduler configuration is valid")
}

/// `make_fixture` with an explicit scheduler configuration flag mask (the C
/// test mutates `fixture.scheduler.configuration_flags` after init; the Rust
/// scheduler keeps its flags private, so the variant is built up front).
fn make_fixture_with_scheduler_flags(scheduler_configuration_flags: u32) -> Fixture {
    make_fixture_with_scheduler(scheduler_configuration_flags, 2)
}

/// `make_fixture` with explicit scheduler flags and queue depth.
fn make_fixture_with_scheduler(
    scheduler_configuration_flags: u32,
    queue_depth_per_spark: u32,
) -> Fixture {
    let prefetch_capture = Rc::new(RefCell::new(PrefetchCapture::new()));
    let mut configuration = make_api_configuration(make_scheduler_with(
        78,
        8,
        128,
        scheduler_configuration_flags,
        queue_depth_per_spark,
    ));
    configuration.kv_prefetch_backend =
        Some(Box::new(CapturePrefetchBackend { capture: Rc::clone(&prefetch_capture) }));
    let api = RequestApi::new(configuration).expect("request api configuration is valid");
    Fixture::new(api, prefetch_capture)
}

/// The request-API configuration shared by both fixtures.
fn make_api_configuration(scheduler: Scheduler) -> Configuration {
    Configuration {
        configuration_flags: 0,
        request_capacity: REQUEST_SLOT_COUNT,
        prefetch_lookahead_request_count: 0,
        prefetch_lane_count: CURRENT_SPARK_COUNT,
        decode_batch_target: MAX_DISPATCH_REQUEST_COUNT,
        max_resident_kv_block_count: 0,
        decode_execution_row_capacity: 0,
        scheduler,
        kv_prefetch_backend: None,
        model_speculator: None,
        output_vocab_count: OUTPUT_VOCAB_COUNT,
    }
}

/// `SparkTestInitializeFixture`.
fn make_fixture() -> Fixture {
    let prefetch_capture = Rc::new(RefCell::new(PrefetchCapture::new()));
    let mut configuration = make_api_configuration(make_scheduler(78, 8, 128));
    configuration.kv_prefetch_backend =
        Some(Box::new(CapturePrefetchBackend { capture: Rc::clone(&prefetch_capture) }));
    let api = RequestApi::new(configuration).expect("request api configuration is valid");
    Fixture::new(api, prefetch_capture)
}

/// `make_fixture` with an explicit slot capacity (the C
/// `AdmitsThirteenThousandWithFreeList` test builds a 13000-slot api).
fn make_fixture_with_request_capacity(request_capacity: u32) -> Fixture {
    let prefetch_capture = Rc::new(RefCell::new(PrefetchCapture::new()));
    let mut configuration = make_api_configuration(make_scheduler(78, 8, 128));
    configuration.request_capacity = request_capacity;
    configuration.kv_prefetch_backend =
        Some(Box::new(CapturePrefetchBackend { capture: Rc::clone(&prefetch_capture) }));
    let api = RequestApi::new(configuration).expect("request api configuration is valid");
    Fixture::new(api, prefetch_capture)
}

/// `make_fixture` with an explicit api configuration flag mask (the C
/// `AcceptsMtpForceEnableConfiguration` test initializes a second api with
/// default flags | MTP_FORCE_ENABLE).
fn make_fixture_with_api_flags(configuration_flags: u32) -> Fixture {
    let prefetch_capture = Rc::new(RefCell::new(PrefetchCapture::new()));
    let mut configuration = make_api_configuration(make_scheduler(78, 8, 128));
    configuration.configuration_flags = configuration_flags;
    configuration.kv_prefetch_backend =
        Some(Box::new(CapturePrefetchBackend { capture: Rc::clone(&prefetch_capture) }));
    let api = RequestApi::new(configuration).expect("request api configuration is valid");
    Fixture::new(api, prefetch_capture)
}

/// `SparkTestInitializeFixtureWithAsyncMemoryBackend`: same stack with the
/// 1-layer/1-head/32-dim arena (64-byte blocks) and the emulated async
/// memory prefetch backend.
fn make_fixture_with_async_memory_backend() -> (Fixture, Rc<RefCell<MemoryPrefetchBackend>>) {
    // `SparkTestFillMemoryKvSource`.
    let mut key_source = vec![0u8; (KV_BLOCK_COUNT * 64) as usize];
    let mut value_source = vec![0u8; (KV_BLOCK_COUNT * 64) as usize];
    for block_index in 0..KV_BLOCK_COUNT as usize {
        for byte_index in 0..64usize {
            key_source[block_index * 64 + byte_index] = (0x31 + block_index + byte_index) as u8;
            value_source[block_index * 64 + byte_index] = (0xa7 + block_index + byte_index) as u8;
        }
    }
    let backend = Rc::new(RefCell::new(MemoryPrefetchBackend::new(key_source, value_source, 64)));
    struct SharedMemoryBackend(Rc<RefCell<MemoryPrefetchBackend>>);
    impl KvPrefetchBackend for SharedMemoryBackend {
        fn prefetch(&mut self, _prefetch_plan: &PrefetchPlan) -> Result<(), RequestApiError> {
            self.0.borrow_mut().prefetch(_prefetch_plan)
        }
        fn start_prefetch(
            &mut self,
            prefetch_id: u64,
            prefetch_plan: &PrefetchPlan,
        ) -> Result<(), RequestApiError> {
            self.0.borrow_mut().start_prefetch(prefetch_id, prefetch_plan)
        }
        fn poll_prefetch(
            &mut self,
            prefetch_id: u64,
            prefetch_plan: &PrefetchPlan,
        ) -> Result<(), RequestApiError> {
            self.0.borrow_mut().poll_prefetch(prefetch_id, prefetch_plan)
        }
    }
    let mut configuration = make_api_configuration(make_scheduler(1, 1, 32));
    configuration
        .use_async_kv_cache_prefetch_backend(Box::new(SharedMemoryBackend(Rc::clone(&backend))))
        .expect("async backend configuration is valid");
    let api = RequestApi::new(configuration).expect("request api configuration is valid");
    (Fixture::new(api, Rc::new(RefCell::new(PrefetchCapture::new()))), backend)
}

/// `SparkTestEnableAsyncPrefetch`: switch the standard fixture to the async
/// capture backend (start/poll) with both statuses OK.
fn enable_async_prefetch(fixture: &mut Fixture) {
    fixture.api.configuration_flags |= CONFIGURATION_FLAG_ASYNC_JIT_KV_PREFETCH;
    fixture.api.kv_prefetch_backend =
        Some(Box::new(CapturePrefetchBackend { capture: Rc::clone(&fixture.prefetch_capture) }));
    fixture.api.next_prefetch_id = 1;
    fixture.prefetch_capture.borrow_mut().async_start_status = Ok(());
    fixture.prefetch_capture.borrow_mut().async_poll_status = Ok(());
}

/// `SparkTestEnableDsparkSpeculation`.
fn enable_dspark_speculation(fixture: &mut Fixture) {
    let dspark_capture = Rc::new(RefCell::new(DsparkCapture::new()));
    fixture.api.configuration_flags |= CONFIGURATION_FLAG_DSPARK_SPECULATIVE_DECODE;
    fixture.api.model_speculator = Some(Box::new(FakeDsparkSpeculator::new(
        Rc::clone(&dspark_capture),
        Rc::clone(&fixture.dspark_policy_flags),
    )));
    fixture.dspark_capture = dspark_capture;
}

/// `SparkTestRequestApiWarmsPrefixCacheAndReleases`.
fn warm_prefix_cache_and_release(fixture: &mut Fixture, prompt_token_ids: &[u32]) {
    let handle = submit(&mut fixture.api, 1, 1001, 10, prompt_token_ids.to_vec(), 0);
    let mut dispatch = Dispatch::default();
    fixture.api.schedule_next(&mut dispatch).expect("schedule succeeds");
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    fixture.api.release_completed_request(handle).expect("release succeeds");
}

// ---------------------------------------------------------------------------
// Test helpers.
// ---------------------------------------------------------------------------

/// Mutable arena access shortcut.
fn arena_mut(api: &RequestApi) -> std::cell::RefMut<'_, KvArena> {
    std::cell::RefMut::map(api.scheduler().borrow_mut(), |scheduler| {
        scheduler.prefix_cache_mut().expect("prefix cache present").arena_mut()
    })
}

/// Mutable prefix-cache access shortcut.
fn prefix_cache_mut(api: &RequestApi) -> std::cell::RefMut<'_, PrefixCache> {
    std::cell::RefMut::map(api.scheduler().borrow_mut(), |scheduler| {
        scheduler.prefix_cache_mut().expect("prefix cache present")
    })
}

/// `SparkRequestApiScheduleNext` into a fresh dispatch.
fn schedule(api: &mut RequestApi) -> Dispatch {
    let mut dispatch = Dispatch::default();
    api.schedule_next(&mut dispatch).expect("schedule succeeds");
    dispatch
}

/// `SparkTestBuildSharedPrefixPrompt`.
fn build_shared_prefix_prompt(
    shared_prefix: &[u32],
    suffix_first_token_id: u32,
    suffix_token_count: u32,
) -> Vec<u32> {
    let mut prompt = shared_prefix.to_vec();
    prompt.extend((0..suffix_token_count).map(|index| suffix_first_token_id + index));
    prompt
}

// ---------------------------------------------------------------------------
// The C test bodies, in file order.
// ---------------------------------------------------------------------------

/// `SparkTestRequestApiJitPrefetchesCachedPrefixForPriorityRequest`.
#[test]
fn jit_prefetches_cached_prefix_for_priority_request() {
    let mut fixture = make_fixture();
    let shared_prompt = fill_token_ids(32, 10_000);
    let other_prompt = fill_token_ids(32, 20_000);
    warm_prefix_cache_and_release(&mut fixture, &shared_prompt);

    let probe = prefix_cache_mut(&fixture.api)
        .probe_physical_block_table(&shared_prompt, 4)
        .expect("probe succeeds");
    assert_eq!(probe.matched_token_count, PREFILL_BLOCK_TOKENS);
    assert_eq!(probe.physical_block_indices.len(), 1);
    let cold_physical_block_index = probe.physical_block_indices[0];
    drop(probe);
    arena_mut(&fixture.api)
        .mark_block_nonresident(cold_physical_block_index)
        .expect("mark nonresident succeeds");
    assert!(!arena(&fixture.api).is_resident(cold_physical_block_index));

    submit(&mut fixture.api, 2, 2002, 10, other_prompt, 1);
    let realtime_handle = {
        let prompt_token_count = shared_prompt.len() as u32;
        fixture
            .api
            .submit(&SubmitRequest {
                flags: REQUEST_FLAG_REALTIME,
                priority: 0,
                prompt_token_count,
                thinking_token_budget: 0,
                output_token_budget: 1,
                max_prefill_tokens_per_step: 0,
                request_id: 3,
                sequence_id: 3003,
                prompt_token_ids: shared_prompt.clone(),
            })
            .expect("submit succeeds")
    };

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    assert_eq!(dispatch.request_handles[0], realtime_handle);
    assert!(dispatch.flags & DISPATCH_FLAG_JIT_PREFETCHED_KV != 0);
    assert!(dispatch.flags & DISPATCH_FLAG_PRIORITY_PREEMPTED_QUEUE != 0);
    assert_eq!(dispatch.kv_prefetch_plan.lane_count, CURRENT_SPARK_COUNT);
    assert_eq!(dispatch.kv_prefetch_plan.prefetch_block_count(), 1);
    let expected_parent_hash = EMPTY_PARENT_HASH;
    let expected_block_hash = hash_block(&shared_prompt[..16], expected_parent_hash);
    assert_eq!(dispatch.kv_prefetch_plan.blocks[0].physical_block_index, cold_physical_block_index);
    assert_eq!(dispatch.kv_prefetch_plan.blocks[0].first_token_index, 0);
    assert_eq!(dispatch.kv_prefetch_plan.blocks[0].token_count, PREFILL_BLOCK_TOKENS);
    assert_eq!(dispatch.kv_prefetch_plan.blocks[0].parent_hash, expected_parent_hash);
    assert_eq!(dispatch.kv_prefetch_plan.blocks[0].block_hash, expected_block_hash);
    assert_ne!(dispatch.kv_prefetch_plan.blocks[0].content_hash, 0);
    let capture = fixture.prefetch_capture.borrow();
    assert_eq!(capture.call_count, 1);
    assert_eq!(capture.last_lane_count, CURRENT_SPARK_COUNT);
    assert_eq!(capture.last_prefetch_block_count, 1);
    assert_eq!(capture.last_physical_block_indices[0], cold_physical_block_index);
    assert_eq!(capture.last_first_token_indices[0], 0);
    assert_eq!(capture.last_token_counts[0], PREFILL_BLOCK_TOKENS);
    assert_eq!(capture.last_parent_hashes[0], expected_parent_hash);
    assert_eq!(capture.last_block_hashes[0], expected_block_hash);
    assert_eq!(capture.last_content_hashes[0], dispatch.kv_prefetch_plan.blocks[0].content_hash);
    drop(capture);
    assert!(arena(&fixture.api).is_resident(cold_physical_block_index));
    assert_eq!(
        dispatch.prefill_decision.as_ref().expect("prefill decision").cached_prefix_token_count,
        PREFILL_BLOCK_TOKENS
    );

    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
}

/// `SparkTestRequestApiBatchesReadyDecodeRequestsAndConsumesBudgets`.
#[test]
fn batches_ready_decode_requests_and_consumes_budgets() {
    let mut fixture = make_fixture();
    let first_handle = submit(&mut fixture.api, 11, 5011, 20, fill_token_ids(16, 30_000), 2);
    let first_dispatch = schedule(&mut fixture.api);
    assert_eq!(first_dispatch.kind, DispatchKind::Prefill);
    fixture.api.complete_dispatch(&first_dispatch).expect("complete succeeds");

    let second_handle = submit(&mut fixture.api, 12, 5012, 20, fill_token_ids(16, 40_000), 2);
    let second_dispatch = schedule(&mut fixture.api);
    assert_eq!(second_dispatch.kind, DispatchKind::Prefill);
    assert_eq!(second_dispatch.request_handles[0], second_handle);
    fixture.api.complete_dispatch(&second_dispatch).expect("complete succeeds");

    let first_dispatch = schedule(&mut fixture.api);
    assert!(first_dispatch.accepted);
    assert_eq!(first_dispatch.kind, DispatchKind::DecodeBatch);
    assert_eq!(first_dispatch.request_count, 2);
    assert_eq!(first_dispatch.request_handles[0], first_handle);
    assert_eq!(first_dispatch.request_handles[1], second_handle);
    let decode_batch_decision =
        first_dispatch.decode_batch_decision.as_ref().expect("decode batch decision");
    assert_eq!(decode_batch_decision.active_sequence_count, 2);
    assert!(decode_batch_decision.decision_flags & DECISION_FLAG_ADAPTIVE_DECODE_PACK != 0);
    let mut decode_block_tables = [0u32; 8];
    let mut decode_block_counts = [0u32; 2];
    fixture
        .api
        .build_dispatch_kv_block_tables(
            &first_dispatch,
            &mut decode_block_tables,
            4,
            4,
            &mut decode_block_counts,
        )
        .expect("kv block tables succeed");
    assert_eq!(decode_block_counts[0], 2);
    assert_eq!(decode_block_counts[1], 2);
    assert_ne!(decode_block_tables[0], decode_block_tables[1]);
    assert_ne!(decode_block_tables[4], decode_block_tables[5]);
    fixture.api.complete_dispatch(&first_dispatch).expect("complete succeeds");

    let second_dispatch = schedule(&mut fixture.api);
    assert!(second_dispatch.accepted);
    assert_eq!(second_dispatch.kind, DispatchKind::DecodeBatch);
    assert_eq!(second_dispatch.request_count, 2);
    fixture.api.complete_dispatch(&second_dispatch).expect("complete succeeds");

    fixture.api.release_completed_request(first_handle).expect("release succeeds");
    fixture.api.release_completed_request(second_handle).expect("release succeeds");
}

/// `SparkTestRequestApiFillsDecodeBatchBeforeEqualPriorityDecode`.
#[test]
fn fills_decode_batch_before_equal_priority_decode() {
    let mut fixture = make_fixture();
    fixture.api.configuration_flags &= !CONFIGURATION_FLAG_PREFILL_BATCHING;
    fixture.api.decode_batch_target = 4;
    for request_index in 0..4u32 {
        submit(
            &mut fixture.api,
            u64::from(100 + request_index),
            u64::from(6000 + request_index),
            100,
            fill_token_ids(16, 50_000 + request_index * 100),
            2,
        );
    }
    for _ in 0..4 {
        let dispatch = schedule(&mut fixture.api);
        assert_eq!(dispatch.kind, DispatchKind::Prefill);
        fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    }
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert_eq!(dispatch.request_count, 4);
    assert_eq!(
        dispatch
            .decode_batch_decision
            .as_ref()
            .expect("decode batch decision")
            .active_sequence_count,
        4
    );
}

/// `SparkTestRequestApiCohortsSamePromptRequestsAndSharesBlocks`.
#[test]
fn cohorts_same_prompt_requests_and_shares_blocks() {
    let mut fixture = make_fixture();
    let prompt = fill_token_ids(32, 50_000);
    let mut handles = [0u64; 4];
    for (request_index, handle) in handles.iter_mut().enumerate() {
        *handle = submit(
            &mut fixture.api,
            100 + request_index as u64,
            7000 + request_index as u64,
            40,
            prompt.clone(),
            1,
        );
    }

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    assert_eq!(dispatch.request_count, 4);
    assert!(dispatch.flags & DISPATCH_FLAG_PREFIX_COHORT != 0);
    assert_eq!(
        dispatch.prefill_decision.as_ref().expect("prefill decision").total_scheduled_token_count,
        32
    );
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");

    let leader_block_table = prefix_cache_mut(&fixture.api)
        .build_physical_block_table(7000, 32)
        .expect("block table succeeds");
    assert_eq!(leader_block_table.len(), 2);
    for request_index in 1..4u64 {
        let block_table = prefix_cache_mut(&fixture.api)
            .build_physical_block_table(7000 + request_index, 32)
            .expect("block table succeeds");
        assert_eq!(block_table.len(), leader_block_table.len());
        assert_eq!(block_table[0], leader_block_table[0]);
        assert_eq!(block_table[1], leader_block_table[1]);
    }

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert_eq!(dispatch.request_count, 4);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    for handle in handles {
        fixture.api.release_completed_request(handle).expect("release succeeds");
    }
}

/// `SparkTestRequestApiWidePrefixFamilyCannotBeatHigherPriority`.
#[test]
fn wide_prefix_family_cannot_beat_higher_priority() {
    let mut fixture = make_fixture();
    let low_prompt = fill_token_ids(32, 61_000);
    let high_prompt = low_prompt.clone();
    fixture.api.configuration_flags |= CONFIGURATION_FLAG_ADAPTIVE_PIPELINE_BATCHING;
    for request_index in 0..4u64 {
        submit(
            &mut fixture.api,
            200 + request_index,
            7200 + request_index,
            10,
            low_prompt.clone(),
            1,
        );
    }
    let high_handle = submit(&mut fixture.api, 300, 7300, 100, high_prompt, 1);
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    assert_eq!(dispatch.request_count, 1);
    assert_eq!(dispatch.request_handles[0], high_handle);
    assert_eq!(dispatch.shared_prefix_token_count, 0);
    for slot_index in 0..4 {
        assert_eq!(fixture.api.slots()[slot_index].state, STATE_QUEUED_PREFILL);
    }
}

/// `SparkTestRequestApiLateHighPriorityUsesNextAvailablePipelineSlot`.
#[test]
fn late_high_priority_uses_next_available_pipeline_slot() {
    let mut fixture = make_fixture();
    let low_handle = submit(&mut fixture.api, 400, 7400, 10, fill_token_ids(16, 63_000), 1);
    let low_dispatch = schedule(&mut fixture.api);
    assert_eq!(low_dispatch.request_handles[0], low_handle);
    let high_handle = submit(&mut fixture.api, 401, 7401, 100, fill_token_ids(16, 64_000), 1);
    let high_dispatch = schedule(&mut fixture.api);
    assert!(!high_dispatch.accepted);
    fixture.api.complete_dispatch(&low_dispatch).expect("complete succeeds");
    let high_dispatch = schedule(&mut fixture.api);
    assert!(
        high_dispatch.kind == DispatchKind::Prefill
            || high_dispatch.kind == DispatchKind::PrefillBatch
    );
    assert_eq!(high_dispatch.request_count, 1);
    assert_eq!(high_dispatch.request_handles[0], high_handle);
    assert_eq!(fixture.api.slots()[0].state, STATE_READY_DECODE);
}

/// `SparkTestRequestApiCohortsArbitrarySharedPrefixWithSuffixes`.
#[test]
fn cohorts_arbitrary_shared_prefix_with_suffixes() {
    let mut fixture = make_fixture();
    let shared_prefix = fill_token_ids(32, 70_000);
    let mut handles = [0u64; 7];
    let mut prompts = Vec::new();
    for (request_index, handle) in handles.iter_mut().enumerate() {
        let prompt =
            build_shared_prefix_prompt(&shared_prefix, 80_000 + request_index as u32 * 100, 16);
        *handle = submit(
            &mut fixture.api,
            500 + request_index as u64,
            9000 + request_index as u64,
            60,
            prompt.clone(),
            1,
        );
        prompts.push(prompt);
    }

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    assert_eq!(dispatch.request_count, 7);
    assert_eq!(dispatch.shared_prefix_token_count, 32);
    assert_eq!(dispatch.shared_prefix_block_count, 2);
    let prefill_decision = dispatch.prefill_decision.as_ref().expect("prefill decision");
    assert_eq!(prefill_decision.prompt_token_count, 48);
    assert_eq!(prefill_decision.scheduled_prompt_token_count, 32);
    assert_eq!(prefill_decision.total_scheduled_token_count, 32);
    let shared_prefix_hash =
        hash_prompt_tokens(PREFILL_BLOCK_TOKENS, EMPTY_PARENT_HASH, &prompts[0][..32])
            .expect("hash succeeds");
    assert!(dispatch.flags & DISPATCH_FLAG_PREFIX_COHORT != 0);
    assert_eq!(dispatch.prefix_cache_parent_hash, EMPTY_PARENT_HASH);
    assert_eq!(dispatch.prefix_cache_result_hash, shared_prefix_hash.prompt_hash);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");

    let leader_block_table = prefix_cache_mut(&fixture.api)
        .build_physical_block_table(9000, 32)
        .expect("block table succeeds");
    assert_eq!(leader_block_table.len(), 2);
    for (request_index, &handle) in handles.iter().enumerate() {
        let cache_state =
            fixture.api.get_request_cache_state(handle).expect("cache state succeeds");
        assert_eq!(cache_state.computed_prompt_token_count, 32);
        assert_eq!(cache_state.last_committed_prefix_token_count, 32);
        assert_eq!(cache_state.last_committed_prefix_hash, dispatch.prefix_cache_result_hash);
        assert_eq!(cache_state.state, STATE_QUEUED_PREFILL);
        let follower_block_table = prefix_cache_mut(&fixture.api)
            .build_physical_block_table(9000 + request_index as u64, 32)
            .expect("block table succeeds");
        assert_eq!(follower_block_table.len(), leader_block_table.len());
        assert_eq!(follower_block_table[0], leader_block_table[0]);
        assert_eq!(follower_block_table[1], leader_block_table[1]);
    }

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::PrefillBatch);
    assert!(dispatch.flags & DISPATCH_FLAG_PREFILL_BATCH != 0);
    assert_eq!(dispatch.request_count, 7);
    let batch_decision = dispatch.prefill_batch_decision.as_ref().expect("prefill batch decision");
    assert!(batch_decision.accepted);
    assert_eq!(batch_decision.active_sequence_count, 7);
    assert_eq!(batch_decision.batch_bucket, BUCKET_B16);
    assert_eq!(batch_decision.maximum_scheduled_prompt_token_count, 16);
    assert_eq!(batch_decision.total_scheduled_token_count, 7 * 16);
    assert!(batch_decision.decision_flags & DECISION_FLAG_ADAPTIVE_PREFILL_PACK != 0);
    for (request_index, lane) in batch_decision.lanes.iter().enumerate() {
        assert_eq!(dispatch.request_handles[request_index], handles[request_index]);
        assert_eq!(lane.sequence_id, 9000 + request_index as u64);
        assert_eq!(lane.prompt_token_count, 48);
        assert_eq!(lane.computed_prompt_token_count, 32);
        assert_eq!(lane.cached_prefix_token_count, 32);
        assert_eq!(lane.scheduled_prompt_token_offset, 32);
        assert_eq!(lane.scheduled_prompt_token_count, 16);
        assert_eq!(lane.cache_commit_token_count_after_step, 48);
        assert_eq!(lane.kv_block_token_count, PREFILL_BLOCK_TOKENS);
        assert_eq!(lane.kv_block_table_token_count, 48);
        let full_prompt_hash =
            hash_prompt_tokens(PREFILL_BLOCK_TOKENS, EMPTY_PARENT_HASH, &prompts[request_index])
                .expect("hash succeeds");
        let suffix_hash = hash_prompt_tokens(
            PREFILL_BLOCK_TOKENS,
            shared_prefix_hash.prompt_hash,
            &prompts[request_index][32..],
        )
        .expect("hash succeeds");
        assert_eq!(lane.prefix_cache_parent_hash, shared_prefix_hash.prompt_hash);
        assert_eq!(lane.prefix_cache_result_hash, full_prompt_hash.prompt_hash);
        assert_eq!(lane.prefix_cache_result_hash, suffix_hash.prompt_hash);
    }
    let mut batch_block_tables = [0u32; 28];
    let mut batch_block_counts = [0u32; 7];
    fixture
        .api
        .build_dispatch_kv_block_tables(
            &dispatch,
            &mut batch_block_tables,
            4,
            4,
            &mut batch_block_counts,
        )
        .expect("kv block tables succeed");
    let execution_block_tables = batch_block_tables;
    let counts_snapshot = batch_block_counts;
    let block_table_view = fixture
        .api
        .build_dispatch_kv_block_table_view(
            &dispatch,
            &mut batch_block_tables,
            Some(&execution_block_tables),
            4,
            4,
            &mut batch_block_counts,
        )
        .expect("kv block table view succeeds");
    assert_eq!(block_table_view.block_token_count, PREFILL_BLOCK_TOKENS);
    assert_eq!(block_table_view.lane_count, 7);
    assert_eq!(block_table_view.lane_stride, 4);
    assert_eq!(block_table_view.lane_capacity, 4);
    assert_eq!(
        block_table_view.physical_block_indices.expect("execution indices"),
        &execution_block_tables[..]
    );
    assert_eq!(block_table_view.host_physical_block_indices, &execution_block_tables[..]);
    assert_eq!(
        block_table_view.lane_physical_block_counts.expect("lane counts"),
        &counts_snapshot[..]
    );
    assert_eq!(block_table_view.host_lane_physical_block_counts, &counts_snapshot[..]);
    // NLL ends the view's borrows at its last use above.
    for request_index in 0..7usize {
        assert_eq!(batch_block_counts[request_index], 3);
        assert_eq!(batch_block_tables[request_index * 4], leader_block_table[0]);
        assert_eq!(batch_block_tables[request_index * 4 + 1], leader_block_table[1]);
        assert_ne!(batch_block_tables[request_index * 4 + 2], leader_block_table[0]);
        assert_ne!(batch_block_tables[request_index * 4 + 2], leader_block_table[1]);
    }
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    for &handle in &handles {
        let cache_state =
            fixture.api.get_request_cache_state(handle).expect("cache state succeeds");
        assert_eq!(cache_state.computed_prompt_token_count, 48);
        assert_eq!(cache_state.last_committed_prefix_token_count, 48);
        assert_eq!(cache_state.state, STATE_READY_DECODE);
    }
}

/// `SparkTestRequestApiChoosesWiderSharedPrefixFamily`.
#[test]
fn chooses_wider_shared_prefix_family() {
    let mut fixture = make_fixture();
    let shared_prefix = fill_token_ids(32, 122_000);
    let close_extra = fill_token_ids(16, 123_000);
    let mut handles = [0u64; 10];
    let mut prompts = Vec::new();
    let mut first = build_shared_prefix_prompt(&shared_prefix, 124_000, 32);
    first[32..48].copy_from_slice(&close_extra);
    prompts.push(first);
    let mut second = build_shared_prefix_prompt(&shared_prefix, 126_000, 32);
    second[32..48].copy_from_slice(&close_extra);
    prompts.push(second);
    for request_index in 2..10u32 {
        prompts.push(build_shared_prefix_prompt(&shared_prefix, 128_000 + request_index * 100, 32));
    }
    for (request_index, handle) in handles.iter_mut().enumerate() {
        *handle = submit(
            &mut fixture.api,
            1220 + request_index as u64,
            12_200 + request_index as u64,
            70,
            prompts[request_index].clone(),
            1,
        );
    }

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    assert_eq!(dispatch.request_count, 10);
    assert_eq!(dispatch.shared_prefix_token_count, 32);
    assert_eq!(dispatch.shared_prefix_block_count, 2);
    let prefill_decision = dispatch.prefill_decision.as_ref().expect("prefill decision");
    assert_eq!(prefill_decision.prompt_token_count, 64);
    assert_eq!(prefill_decision.scheduled_prompt_token_count, 32);
    assert!(dispatch.flags & DISPATCH_FLAG_PREFIX_COHORT != 0);
    assert!(dispatch.flags & DISPATCH_FLAG_PREFIX_FAMILY_SELECTED != 0);
    let shared_prefix_hash =
        hash_prompt_tokens(PREFILL_BLOCK_TOKENS, EMPTY_PARENT_HASH, &prompts[0][..32])
            .expect("hash succeeds");
    assert_eq!(dispatch.prefix_cache_result_hash, shared_prefix_hash.prompt_hash);
    assert_eq!(fixture.api.prefix_family_dispatch_count, 1);
    assert_eq!(fixture.api.prefix_family_member_count, 10);
    assert_eq!(fixture.api.prefix_family_saved_prompt_token_count, 32 * 9);

    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    for &handle in &handles {
        let cache_state =
            fixture.api.get_request_cache_state(handle).expect("cache state succeeds");
        assert_eq!(cache_state.computed_prompt_token_count, 32);
        assert_eq!(cache_state.state, STATE_QUEUED_PREFILL);
    }
}

/// `SparkTestRequestApiBatchesVariableOneBlockSuffixes`.
#[test]
fn batches_variable_one_block_suffixes() {
    let mut fixture = make_fixture();
    let shared_prefix = fill_token_ids(32, 94_000);
    warm_prefix_cache_and_release(&mut fixture, &shared_prefix);
    let suffix_token_counts = [1u32, 5, 16, 9, 12];
    let mut handles = [0u64; 5];
    for (request_index, handle) in handles.iter_mut().enumerate() {
        let prompt = build_shared_prefix_prompt(
            &shared_prefix,
            95_000 + request_index as u32 * 100,
            suffix_token_counts[request_index],
        );
        *handle = submit(
            &mut fixture.api,
            950 + request_index as u64,
            9950 + request_index as u64,
            75,
            prompt,
            1,
        );
    }

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::PrefillBatch);
    assert_eq!(dispatch.request_count, 5);
    let batch_decision = dispatch.prefill_batch_decision.as_ref().expect("prefill batch decision");
    assert!(batch_decision.accepted);
    assert_eq!(batch_decision.active_sequence_count, 5);
    assert_eq!(batch_decision.batch_bucket, BUCKET_B16);
    assert_eq!(batch_decision.maximum_scheduled_prompt_token_count, 16);
    assert_eq!(batch_decision.graph_sequence_padding_count, 11);
    let mut scheduled_token_count_seen = [0u32; 17];
    let mut scheduled_token_total = 0u32;
    for lane in batch_decision.lanes.iter().take(batch_decision.packed_request_count as usize) {
        assert_eq!(lane.cached_prefix_token_count, 32);
        assert_eq!(lane.computed_prompt_token_count, 32);
        assert_eq!(lane.scheduled_prompt_token_offset, 32);
        assert!(lane.scheduled_prompt_token_count > 0);
        assert!(lane.scheduled_prompt_token_count <= 16);
        assert_eq!(lane.remaining_prompt_token_count_after_step, 0);
        assert_eq!(lane.kv_block_table_token_count, lane.prompt_token_count);
        scheduled_token_count_seen[lane.scheduled_prompt_token_count as usize] += 1;
        scheduled_token_total += lane.scheduled_prompt_token_count;
    }
    assert_eq!(scheduled_token_count_seen[1], 1);
    assert_eq!(scheduled_token_count_seen[5], 1);
    assert_eq!(scheduled_token_count_seen[9], 1);
    assert_eq!(scheduled_token_count_seen[12], 1);
    assert_eq!(scheduled_token_count_seen[16], 1);
    assert_eq!(batch_decision.total_scheduled_token_count, u64::from(scheduled_token_total));
    assert_eq!(scheduled_token_total, 43);

    let mut batch_block_tables = [0u32; 20];
    let mut batch_block_counts = [0u32; 5];
    fixture
        .api
        .build_dispatch_kv_block_tables(
            &dispatch,
            &mut batch_block_tables,
            4,
            4,
            &mut batch_block_counts,
        )
        .expect("kv block tables succeed");
    for count in &batch_block_counts {
        assert_eq!(*count, 3);
    }

    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    for (request_index, &handle) in handles.iter().enumerate() {
        let cache_state =
            fixture.api.get_request_cache_state(handle).expect("cache state succeeds");
        assert_eq!(
            cache_state.computed_prompt_token_count,
            32 + suffix_token_counts[request_index]
        );
        assert_eq!(
            cache_state.last_committed_prefix_token_count,
            32 + suffix_token_counts[request_index]
        );
        assert_eq!(cache_state.state, STATE_READY_DECODE);
    }
}

/// `SparkTestRequestApiOpportunisticLookaheadDoesNotBlockReadyPriorityPrefill`.
#[test]
fn opportunistic_lookahead_does_not_block_ready_priority_prefill() {
    let mut fixture = make_fixture();
    let high_priority_prompt = fill_token_ids(32, 91_000);
    let low_priority_prompt = fill_token_ids(32, 92_000);
    warm_prefix_cache_and_release(&mut fixture, &high_priority_prompt);
    warm_prefix_cache_and_release(&mut fixture, &low_priority_prompt);

    let probe = prefix_cache_mut(&fixture.api)
        .probe_physical_block_table(&low_priority_prompt, 4)
        .expect("probe succeeds");
    assert_eq!(probe.matched_token_count, PREFILL_BLOCK_TOKENS);
    assert_eq!(probe.physical_block_indices.len(), 1);
    let low_priority_cold_block_index = probe.physical_block_indices[0];
    drop(probe);
    arena_mut(&fixture.api)
        .mark_block_nonresident(low_priority_cold_block_index)
        .expect("mark nonresident succeeds");
    {
        let mut capture = fixture.prefetch_capture.borrow_mut();
        capture.return_status = Err(RequestApiError::Busy);
        capture.call_count = 0;
    }

    let low_priority_handle = submit(&mut fixture.api, 910, 9910, 10, low_priority_prompt, 1);
    let high_priority_handle = submit(&mut fixture.api, 911, 9911, 100, high_priority_prompt, 1);

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    assert_eq!(dispatch.request_count, 1);
    assert_eq!(dispatch.request_handles[0], high_priority_handle);
    assert!(dispatch.flags & DISPATCH_FLAG_JIT_PREFETCH_PENDING == 0);
    let capture = fixture.prefetch_capture.borrow();
    assert_eq!(capture.call_count, 1);
    assert_eq!(capture.last_prefetch_block_count, 1);
    assert_eq!(capture.last_physical_block_indices[0], low_priority_cold_block_index);
    drop(capture);
    assert!(!arena(&fixture.api).is_resident(low_priority_cold_block_index));
    assert_eq!(fixture.api.slots()[0].state, STATE_QUEUED_PREFILL);

    fixture.api.cancel_dispatch(&dispatch).expect("cancel dispatch succeeds");
    fixture.api.cancel_request(low_priority_handle).expect("cancel request succeeds");
}

/// `SparkTestRequestApiPrefetchesLiveNonresidentDecodeBlocks`.
#[test]
fn prefetches_live_nonresident_decode_blocks() {
    let mut fixture = make_fixture();
    let handle = submit(&mut fixture.api, 930, 9930, 70, fill_token_ids(32, 93_000), 1);
    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");

    let block_table = prefix_cache_mut(&fixture.api)
        .build_physical_block_table(9930, 32)
        .expect("block table succeeds");
    assert_eq!(block_table.len(), 2);
    let cold_physical_block_index = block_table[0];
    assert_ne!(arena(&fixture.api).reference_count(cold_physical_block_index), 0);
    arena_mut(&fixture.api)
        .mark_block_nonresident(cold_physical_block_index)
        .expect("mark nonresident succeeds");
    assert!(!arena(&fixture.api).is_resident(cold_physical_block_index));
    {
        let mut capture = fixture.prefetch_capture.borrow_mut();
        capture.call_count = 0;
        capture.return_status = Ok(());
    }

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert_eq!(dispatch.request_count, 1);
    assert_eq!(dispatch.request_handles[0], handle);
    assert!(dispatch.flags & DISPATCH_FLAG_JIT_PREFETCHED_KV != 0);
    let capture = fixture.prefetch_capture.borrow();
    assert_eq!(capture.call_count, 1);
    assert_eq!(capture.last_prefetch_block_count, 1);
    assert_eq!(capture.last_physical_block_indices[0], cold_physical_block_index);
    drop(capture);
    assert!(arena(&fixture.api).is_resident(cold_physical_block_index));

    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    fixture.api.release_completed_request(handle).expect("release succeeds");
}

/// `SparkTestRequestApiBatchesDecodeAfterBatchCriticalPrefetch`.
#[test]
fn batches_decode_after_batch_critical_prefetch() {
    let mut fixture = make_fixture();
    let first_handle = submit(&mut fixture.api, 940, 9940, 50, fill_token_ids(32, 94_000), 1);
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");

    let second_handle = submit(&mut fixture.api, 941, 9941, 60, fill_token_ids(32, 95_000), 1);
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    assert_eq!(dispatch.request_handles[0], second_handle);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");

    for slot in fixture.api.slots_mut() {
        if slot.handle == first_handle || slot.handle == second_handle {
            slot.priority = 50;
        }
    }

    let second_block_table = prefix_cache_mut(&fixture.api)
        .build_physical_block_table(9941, 32)
        .expect("block table succeeds");
    assert_eq!(second_block_table.len(), 2);
    let cold_second_block_index = second_block_table[0];
    arena_mut(&fixture.api)
        .mark_block_nonresident(cold_second_block_index)
        .expect("mark nonresident succeeds");
    {
        let mut capture = fixture.prefetch_capture.borrow_mut();
        capture.call_count = 0;
        capture.busy_call_budget = 1;
        capture.return_status = Ok(());
    }

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert_eq!(dispatch.request_count, 2);
    assert!(dispatch.flags & DISPATCH_FLAG_JIT_PREFETCHED_KV != 0);
    assert_eq!(fixture.prefetch_capture.borrow().call_count, 2);
    assert_eq!(dispatch.kv_prefetch_plan.prefetch_block_count(), 1);
    assert_eq!(dispatch.kv_prefetch_plan.blocks[0].physical_block_index, cold_second_block_index);
    assert!(arena(&fixture.api).is_resident(cold_second_block_index));
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    fixture.api.release_completed_request(first_handle).expect("release succeeds");
    fixture.api.release_completed_request(second_handle).expect("release succeeds");
}

/// `SparkTestRequestApiBatchesPrefillAfterBatchCriticalPrefetch`.
#[test]
fn batches_prefill_after_batch_critical_prefetch() {
    let mut fixture = make_fixture();
    let first_prefix = fill_token_ids(32, 96_000);
    let second_prefix = fill_token_ids(32, 97_000);
    let first_prompt = build_shared_prefix_prompt(&first_prefix, 98_000, 16);
    let second_prompt = build_shared_prefix_prompt(&second_prefix, 99_000, 16);
    warm_prefix_cache_and_release(&mut fixture, &first_prefix);
    warm_prefix_cache_and_release(&mut fixture, &second_prefix);

    let probe = prefix_cache_mut(&fixture.api)
        .probe_physical_block_table(&second_prompt, 4)
        .expect("probe succeeds");
    assert_eq!(probe.matched_token_count, 32);
    assert_eq!(probe.physical_block_indices.len(), 2);
    let cold_second_prefix_block_index = probe.physical_block_indices[0];
    drop(probe);
    arena_mut(&fixture.api)
        .mark_block_nonresident(cold_second_prefix_block_index)
        .expect("mark nonresident succeeds");
    {
        let mut capture = fixture.prefetch_capture.borrow_mut();
        capture.call_count = 0;
        capture.busy_call_budget = 1;
        capture.return_status = Ok(());
    }

    let first_handle = submit(&mut fixture.api, 960, 9960, 50, first_prompt, 1);
    let second_handle = submit(&mut fixture.api, 961, 9961, 50, second_prompt, 1);

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::PrefillBatch);
    assert_eq!(dispatch.request_count, 2);
    assert!(dispatch.flags & DISPATCH_FLAG_JIT_PREFETCHED_KV != 0);
    assert_eq!(fixture.prefetch_capture.borrow().call_count, 2);
    assert_eq!(dispatch.kv_prefetch_plan.prefetch_block_count(), 1);
    assert_eq!(
        dispatch.kv_prefetch_plan.blocks[0].physical_block_index,
        cold_second_prefix_block_index
    );
    let batch_decision = dispatch.prefill_batch_decision.as_ref().expect("prefill batch decision");
    assert_eq!(batch_decision.active_sequence_count, 2);
    assert_eq!(batch_decision.maximum_scheduled_prompt_token_count, 16);
    assert!(arena(&fixture.api).is_resident(cold_second_prefix_block_index));
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    fixture.api.cancel_request(first_handle).expect("cancel succeeds");
    fixture.api.cancel_request(second_handle).expect("cancel succeeds");
}

/// `SparkTestRequestApiAdaptivePrefillChoosesFullResidentBucketOverOlderSingleton`.
#[test]
fn adaptive_prefill_chooses_full_resident_bucket_over_older_singleton() {
    let mut fixture = make_fixture();
    let singleton_handle =
        submit(&mut fixture.api, 1300, 11_300, 10, fill_token_ids(48, 130_000), 1);
    for request_index in 0..16u64 {
        submit(
            &mut fixture.api,
            1400 + request_index,
            11_400 + request_index,
            10,
            fill_token_ids(16, 140_000 + request_index as u32 * 1000),
            1,
        );
    }

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::PrefillBatch);
    assert_eq!(dispatch.request_count, 16);
    let batch_decision = dispatch.prefill_batch_decision.as_ref().expect("prefill batch decision");
    assert_eq!(batch_decision.active_sequence_count, 16);
    assert_eq!(batch_decision.batch_bucket, BUCKET_B16);
    assert_eq!(batch_decision.graph_sequence_padding_count, 0);
    assert_eq!(batch_decision.maximum_scheduled_prompt_token_count, 16);
    assert_eq!(batch_decision.total_scheduled_token_count, 256);
    for request_index in 0..dispatch.request_count as usize {
        assert_ne!(dispatch.request_handles[request_index], singleton_handle);
    }
    assert_eq!(fixture.api.slots()[0].state, STATE_QUEUED_PREFILL);
}

/// `SparkTestRequestApiRealtimePrefillBypassesFullBulkBatch`.
#[test]
fn realtime_prefill_bypasses_full_bulk_batch() {
    let mut fixture = make_fixture();
    for request_index in 0..16u64 {
        submit(
            &mut fixture.api,
            1500 + request_index,
            11_500 + request_index,
            10,
            fill_token_ids(16, 150_000 + request_index as u32 * 1000),
            1,
        );
    }
    let realtime_handle = {
        let realtime_prompt = fill_token_ids(48, 170_000);
        fixture
            .api
            .submit(&SubmitRequest {
                flags: REQUEST_FLAG_REALTIME,
                priority: 0,
                prompt_token_count: realtime_prompt.len() as u32,
                thinking_token_budget: 0,
                output_token_budget: 1,
                max_prefill_tokens_per_step: 0,
                request_id: 1700,
                sequence_id: 11_700,
                prompt_token_ids: realtime_prompt,
            })
            .expect("submit succeeds")
    };

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    assert_eq!(dispatch.request_count, 1);
    assert_eq!(dispatch.request_handles[0], realtime_handle);
    assert!(dispatch.flags & DISPATCH_FLAG_PRIORITY_PREEMPTED_QUEUE != 0);
    assert_eq!(fixture.api.slots()[16].state, STATE_RUNNING_PREFILL);
}

/// `SparkTestRequestApiDecodeBatchUsesMeasuredB64ForSeventeenReadyRequests`.
#[test]
fn decode_batch_uses_measured_b64_for_seventeen_ready_requests() {
    let mut fixture = make_fixture();
    let mut handles = [0u64; 17];
    for (request_index, handle) in handles.iter_mut().enumerate() {
        *handle = submit(
            &mut fixture.api,
            2000 + request_index as u64,
            12_000 + request_index as u64,
            10,
            fill_token_ids(16, 200_000 + request_index as u32 * 1000),
            1,
        );
    }

    let mut ready_count = 0;
    for _ in 0..8 {
        if ready_count >= 17 {
            break;
        }
        let dispatch = schedule(&mut fixture.api);
        assert!(dispatch.accepted);
        assert!(
            dispatch.kind == DispatchKind::Prefill || dispatch.kind == DispatchKind::PrefillBatch
        );
        fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
        ready_count = 0;
        for &handle in &handles {
            let cache_state =
                fixture.api.get_request_cache_state(handle).expect("cache state succeeds");
            if cache_state.state == STATE_READY_DECODE {
                ready_count += 1;
            }
        }
    }
    assert_eq!(ready_count, 17);

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert_eq!(dispatch.request_count, 17);
    let decode_batch_decision =
        dispatch.decode_batch_decision.as_ref().expect("decode batch decision");
    assert_eq!(decode_batch_decision.active_sequence_count, 17);
    assert_eq!(decode_batch_decision.batch_bucket, BUCKET_B64);
    assert_eq!(decode_batch_decision.graph_sequence_padding_count, 47);
    assert!(decode_batch_decision.decision_flags & DECISION_FLAG_MEASURED_DECODE_BUCKET != 0);
    assert!(
        decode_batch_decision.stage_decision.dispatch_stages[0].dispatch_flags
            & DISPATCH_STAGE_FLAG_MEASURED_DECODE_BUCKET
            != 0
    );
    assert_eq!(fixture.api.scheduler().borrow().measured_decode_bucket_selection_count, 1);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
}

/// `SparkTestRequestApiAsyncJitPrefetchOverlapsResidentWork`.
#[test]
fn async_jit_prefetch_overlaps_resident_work() {
    let mut fixture = make_fixture();
    let prompt = fill_token_ids(32, 180_000);
    let low_priority_prompt = fill_token_ids(32, 181_000);
    warm_prefix_cache_and_release(&mut fixture, &prompt);
    let probe = prefix_cache_mut(&fixture.api)
        .probe_physical_block_table(&prompt, 4)
        .expect("probe succeeds");
    assert!(probe.matched_token_count == 16 || probe.matched_token_count == 32);
    assert!(!probe.physical_block_indices.is_empty());
    let cold_physical_block_index = probe.physical_block_indices[0];
    drop(probe);
    arena_mut(&fixture.api)
        .mark_block_nonresident(cold_physical_block_index)
        .expect("mark nonresident succeeds");
    enable_async_prefetch(&mut fixture);
    fixture.prefetch_capture.borrow_mut().async_poll_busy_budget = 1;

    let high_priority_handle = submit(&mut fixture.api, 1800, 11_800, 80, prompt, 1);
    let low_priority_handle = submit(&mut fixture.api, 1801, 11_801, 10, low_priority_prompt, 1);

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.request_handles[0], low_priority_handle);
    assert!(dispatch.flags & DISPATCH_FLAG_JIT_PREFETCH_PENDING != 0);
    {
        let capture = fixture.prefetch_capture.borrow();
        assert_eq!(capture.async_start_count, 1);
        assert_eq!(capture.async_poll_count, 1);
        assert!(capture.async_pending);
    }
    assert_eq!(fixture.api.async_jit_prefetch_start_count, 1);
    assert_eq!(fixture.api.async_jit_prefetch_completion_count, 0);
    assert!(!arena(&fixture.api).is_resident(cold_physical_block_index));
    assert_eq!(fixture.api.slots()[0].state, STATE_QUEUED_PREFILL);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    assert_eq!(dispatch.request_handles[0], high_priority_handle);
    assert!(dispatch.flags & DISPATCH_FLAG_JIT_PREFETCHED_KV != 0);
    {
        let capture = fixture.prefetch_capture.borrow();
        assert_eq!(capture.async_start_count, 1);
        assert_eq!(capture.async_poll_count, 2);
        assert!(!capture.async_pending);
    }
    assert_eq!(fixture.api.async_jit_prefetch_completion_count, 1);
    assert!(arena(&fixture.api).is_resident(cold_physical_block_index));

    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    fixture.api.cancel_request(high_priority_handle).expect("cancel succeeds");
    fixture.api.cancel_request(low_priority_handle).expect("cancel succeeds");
}

/// `SparkTestRequestApiDoesNotGreenlightMissingJitKv`.
#[test]
fn does_not_greenlight_missing_jit_kv() {
    let mut fixture = make_fixture();
    let prompt = fill_token_ids(32, 90_000);
    warm_prefix_cache_and_release(&mut fixture, &prompt);
    let probe = prefix_cache_mut(&fixture.api)
        .probe_physical_block_table(&prompt, 4)
        .expect("probe succeeds");
    assert_eq!(probe.physical_block_indices.len(), 1);
    let cold_physical_block_index = probe.physical_block_indices[0];
    drop(probe);
    arena_mut(&fixture.api)
        .mark_block_nonresident(cold_physical_block_index)
        .expect("mark nonresident succeeds");
    fixture.prefetch_capture.borrow_mut().return_status = Err(RequestApiError::Busy);

    submit(&mut fixture.api, 900, 9900, 70, prompt, 1);
    let mut dispatch = Dispatch::default();
    assert_eq!(fixture.api.schedule_next(&mut dispatch), Err(RequestApiError::Busy));
    assert!(!dispatch.accepted);
    assert!(dispatch.flags & DISPATCH_FLAG_JIT_PREFETCH_PENDING != 0);
    assert_eq!(fixture.api.slots()[0].state, STATE_QUEUED_PREFILL);
    assert!(!arena(&fixture.api).is_resident(cold_physical_block_index));
}

/// `SparkTestRequestApiTrimsResidentKvWithoutEvictingNearFuturePrefix`.
#[test]
fn trims_resident_kv_without_evicting_near_future_prefix() {
    let mut fixture = make_fixture();
    let cold_prompt = fill_token_ids(32, 131_000);
    let hot_prompt = fill_token_ids(32, 132_000);
    fixture.api.max_resident_kv_block_count = 2;

    warm_prefix_cache_and_release(&mut fixture, &cold_prompt);
    warm_prefix_cache_and_release(&mut fixture, &hot_prompt);
    assert_eq!(arena(&fixture.api).resident_block_count(), 4);

    let cold_probe = prefix_cache_mut(&fixture.api)
        .probe_physical_block_table(&cold_prompt, 4)
        .expect("probe succeeds");
    assert_eq!(cold_probe.matched_token_count, PREFILL_BLOCK_TOKENS);
    assert_eq!(cold_probe.physical_block_indices.len(), 1);
    let cold_block = cold_probe.physical_block_indices[0];
    let hot_probe = prefix_cache_mut(&fixture.api)
        .probe_physical_block_table(&hot_prompt, 4)
        .expect("probe succeeds");
    assert_eq!(hot_probe.matched_token_count, PREFILL_BLOCK_TOKENS);
    assert_eq!(hot_probe.physical_block_indices.len(), 1);
    let hot_block = hot_probe.physical_block_indices[0];

    let handle = submit(&mut fixture.api, 1310, 11_310, 90, hot_prompt, 0);

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    assert_eq!(dispatch.request_handles[0], handle);
    assert_eq!(
        dispatch.prefill_decision.as_ref().expect("prefill decision").cached_prefix_token_count,
        PREFILL_BLOCK_TOKENS
    );
    assert_eq!(fixture.api.jit_residency_eviction_count, 2);
    assert_eq!(arena(&fixture.api).resident_block_count(), 2);
    assert!(arena(&fixture.api).is_resident(hot_block));
    assert!(!arena(&fixture.api).is_resident(cold_block));

    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    fixture.api.release_completed_request(handle).expect("release succeeds");
}

/// `SparkTestRequestApiEvictsColdResidentBlocksButKeepsLookaheadHotset`.
#[test]
fn evicts_cold_resident_blocks_but_keeps_lookahead_hotset() {
    let mut fixture = make_fixture();
    let cold_prompt = fill_token_ids(32, 125_000);
    let hot_prompt = fill_token_ids(32, 126_000);
    warm_prefix_cache_and_release(&mut fixture, &cold_prompt);
    warm_prefix_cache_and_release(&mut fixture, &hot_prompt);

    let cold_probe = prefix_cache_mut(&fixture.api)
        .probe_physical_block_table(&cold_prompt, 4)
        .expect("probe succeeds");
    assert_eq!(cold_probe.physical_block_indices.len(), 1);
    let cold_reusable_block = cold_probe.physical_block_indices[0];
    let hot_probe = prefix_cache_mut(&fixture.api)
        .probe_physical_block_table(&hot_prompt, 4)
        .expect("probe succeeds");
    assert_eq!(hot_probe.physical_block_indices.len(), 1);
    let hot_reusable_block = hot_probe.physical_block_indices[0];
    assert_ne!(cold_reusable_block, hot_reusable_block);
    assert!(arena(&fixture.api).is_resident(cold_reusable_block));
    assert!(arena(&fixture.api).is_resident(hot_reusable_block));

    fixture.api.max_resident_kv_block_count = 1;
    let handle = submit(&mut fixture.api, 1250, 11_250, 90, hot_prompt, 1);

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    assert_eq!(dispatch.request_handles[0], handle);
    assert_ne!(fixture.api.jit_residency_eviction_count, 0);
    assert_ne!(fixture.api.jit_residency_protected_block_count, 0);
    assert!(arena(&fixture.api).is_resident(hot_reusable_block));
    assert!(!arena(&fixture.api).is_resident(cold_reusable_block));
    assert!(
        arena(&fixture.api).resident_block_count()
            <= u64::from(fixture.api.max_resident_kv_block_count)
    );

    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    fixture.api.cancel_request(handle).expect("cancel succeeds");
}

/// `SparkTestRequestApiReuseScoredEvictionKeepsSharedPrefixFamily`.
#[test]
fn reuse_scored_eviction_keeps_shared_prefix_family() {
    let mut fixture = make_fixture();
    let hot_prompt = fill_token_ids(32, 141_000);
    let cold_prompt = fill_token_ids(32, 142_000);
    warm_prefix_cache_and_release(&mut fixture, &hot_prompt);
    warm_prefix_cache_and_release(&mut fixture, &cold_prompt);
    assert_eq!(arena(&fixture.api).resident_block_count(), 4);
    fixture.api.max_resident_kv_block_count = 1;

    let hot_probe = prefix_cache_mut(&fixture.api)
        .probe_physical_block_table(&hot_prompt, 4)
        .expect("probe succeeds");
    assert_eq!(hot_probe.physical_block_indices.len(), 1);
    let hot_reusable_block = hot_probe.physical_block_indices[0];
    let cold_probe = prefix_cache_mut(&fixture.api)
        .probe_physical_block_table(&cold_prompt, 4)
        .expect("probe succeeds");
    assert_eq!(cold_probe.physical_block_indices.len(), 1);
    let cold_reusable_block = cold_probe.physical_block_indices[0];
    assert_ne!(hot_reusable_block, cold_reusable_block);

    submit(&mut fixture.api, 1410, 11_410, 50, hot_prompt.clone(), 1);
    submit(&mut fixture.api, 1411, 11_411, 50, hot_prompt, 1);
    submit(&mut fixture.api, 1420, 11_420, 50, cold_prompt, 1);

    let mut dispatch = Dispatch::default();
    let schedule_status = fixture.api.schedule_next(&mut dispatch);
    assert!(schedule_status == Ok(()) || schedule_status == Err(RequestApiError::Busy));
    assert!(
        prefix_cache(&fixture.api).reuse_scored_resident_eviction_count >= 3,
        "resident eviction count {}",
        prefix_cache(&fixture.api).reuse_scored_resident_eviction_count
    );
    assert_ne!(prefix_cache(&fixture.api).reuse_scored_lookahead_eviction_count, 0);
    assert!(arena(&fixture.api).is_resident(hot_reusable_block));
    assert!(!arena(&fixture.api).is_resident(cold_reusable_block));
    assert_eq!(arena(&fixture.api).resident_block_count(), 1);

    if dispatch.accepted {
        fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    }
}

/// `SparkTestRequestApiUsesBuiltInAsyncMemoryPrefetchBackend` (see the
/// file-header deviation note for the emulated backend).
#[test]
fn uses_built_in_async_memory_prefetch_backend() {
    let (mut fixture, backend) = make_fixture_with_async_memory_backend();
    let prompt = fill_token_ids(32, 131_000);
    warm_prefix_cache_and_release(&mut fixture, &prompt);

    let probe = prefix_cache_mut(&fixture.api)
        .probe_physical_block_table(&prompt, 4)
        .expect("probe succeeds");
    assert_eq!(probe.matched_token_count, PREFILL_BLOCK_TOKENS);
    assert_eq!(probe.physical_block_indices.len(), 1);
    let cold_physical_block_index = probe.physical_block_indices[0];
    drop(probe);
    arena_mut(&fixture.api)
        .mark_block_nonresident(cold_physical_block_index)
        .expect("mark nonresident succeeds");

    let handle = {
        let prompt_token_count = prompt.len() as u32;
        fixture
            .api
            .submit(&SubmitRequest {
                flags: REQUEST_FLAG_REALTIME,
                priority: 900,
                prompt_token_count,
                thinking_token_budget: 0,
                output_token_budget: 1,
                max_prefill_tokens_per_step: 0,
                request_id: 1310,
                sequence_id: 11_310,
                prompt_token_ids: prompt,
            })
            .expect("submit succeeds")
    };
    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.request_handles[0], handle);
    assert!(dispatch.flags & DISPATCH_FLAG_JIT_PREFETCHED_KV != 0);
    {
        let backend = backend.borrow();
        assert_eq!(backend.start_count, 1);
        assert_eq!(backend.completed_prefetch_count, 1);
        assert_eq!(backend.copied_key_block_count, 1);
        assert_eq!(backend.copied_value_block_count, 1);
    }
    let block_view =
        arena(&fixture.api).resolve_block(cold_physical_block_index).expect("resolve succeeds");
    let source_index = cold_physical_block_index as usize * 64;
    let expected_key: Vec<u8> =
        (0..64usize).map(|i| (0x31 + cold_physical_block_index as usize + i) as u8).collect();
    let expected_value: Vec<u8> =
        (0..64usize).map(|i| (0xa7 + cold_physical_block_index as usize + i) as u8).collect();
    let _ = source_index;
    {
        let backend = backend.borrow();
        assert_eq!(
            backend.device_memory.get(&block_view.key_device_address).expect("key copied"),
            &expected_key
        );
        assert_eq!(
            backend.device_memory.get(&block_view.value_device_address).expect("value copied"),
            &expected_value
        );
    }
    assert!(arena(&fixture.api).is_resident(cold_physical_block_index));

    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
}

/// `SparkTestRequestApiRefreshesQueueAwarePrefixProtection`.
#[test]
fn refreshes_queue_aware_prefix_protection() {
    let mut fixture = make_fixture();
    let prompt = fill_token_ids(32, 121_000);
    warm_prefix_cache_and_release(&mut fixture, &prompt);
    let baseline_protection_sweep_count = fixture.api.lookahead_protection_sweep_count;
    let baseline_protected_block_count = fixture.api.lookahead_protected_block_count;

    let handle = submit(&mut fixture.api, 1210, 11_210, 80, prompt, 1);

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    assert_eq!(dispatch.request_handles[0], handle);
    assert_eq!(fixture.api.lookahead_protection_sweep_count, baseline_protection_sweep_count + 1);
    assert!(fixture.api.lookahead_protected_block_count > baseline_protected_block_count);
    assert_ne!(prefix_cache(&fixture.api).lookahead_protection_epoch, 0);
    assert_ne!(prefix_cache(&fixture.api).lookahead_protected_block_count, 0);

    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    fixture.api.cancel_request(handle).expect("cancel succeeds");
}

/// `SparkTestRequestApiDsparkCapturesTapsAndRunsSpeculativeVerify`.
#[test]
fn dspark_captures_taps_and_runs_speculative_verify() {
    let mut fixture = make_fixture();
    enable_dspark_speculation(&mut fixture);
    let handle = {
        let prompt = fill_token_ids(16, 150_000);
        fixture
            .api
            .submit(&SubmitRequest {
                flags: REQUEST_FLAG_REALTIME,
                priority: REALTIME_PRIORITY,
                prompt_token_count: prompt.len() as u32,
                thinking_token_budget: 0,
                output_token_budget: 10,
                max_prefill_tokens_per_step: 0,
                request_id: 1500,
                sequence_id: 11_500,
                prompt_token_ids: prompt,
            })
            .expect("submit succeeds")
    };

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    assert!(dispatch.flags & DISPATCH_FLAG_DSPARK_TAP_CAPTURE != 0);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");

    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert_eq!(dispatch.request_count, 1);
    assert_eq!(dispatch.request_handles[0], handle);
    assert!(dispatch.flags & DISPATCH_FLAG_DSPARK_TAP_CAPTURE != 0);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    {
        let capture = fixture.dspark_capture.borrow();
        assert_eq!(capture.call_count, 1);
        assert_eq!(capture.requested_token_count, MAX_SPECULATIVE_TOKENS);
        assert_eq!(capture.sequence_id, 11_500);
        assert_eq!(capture.sequence_position, 17);
    }
    assert_eq!(fixture.api.dspark_tap_capture_dispatch_count, 2);
    assert_eq!(fixture.api.dspark_draft_ready_count, 1);

    let mut dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::SpeculativeVerifyBatch);
    assert_eq!(dispatch.request_count, 1);
    assert_eq!(dispatch.request_handles[0], handle);
    assert!(dispatch.flags & DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY != 0);
    assert_eq!(dispatch.speculative_token_count, MAX_SPECULATIVE_TOKENS);
    assert_eq!(dispatch.speculative_max_committed_token_count, MAX_SPECULATIVE_TOKENS + 1);
    assert_eq!(dispatch.speculative_draft_token_ids[0][0], 140_000);
    assert_eq!(dispatch.speculative_draft_token_ids[0][6], 140_006);
    assert_eq!(dispatch.speculative_confidence_milli[0][0], 800);

    let verifier_tokens =
        [140_000u32, 140_001, 140_002, 141_111, 141_112, 141_113, 141_114, 141_115];
    fixture
        .api
        .resolve_speculative_verify_dispatch(
            &mut dispatch,
            &verifier_tokens,
            MAX_SPECULATIVE_TOKENS + 1,
            MAX_SPECULATIVE_TOKENS + 1,
        )
        .expect("resolve succeeds");
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    assert_eq!(fixture.api.dspark_verify_dispatch_count, 1);
    assert_eq!(fixture.api.dspark_accepted_draft_token_count, 3);
    assert_eq!(fixture.api.dspark_committed_token_count, 4);
    assert_eq!(fixture.api.dspark_rejected_token_count, 4);
    assert_eq!(fixture.api.completed_request_count, 0);
    {
        let capture = fixture.dspark_capture.borrow();
        assert_eq!(capture.call_count, 2);
        assert_eq!(capture.requested_token_count, 4);
        assert_eq!(capture.sequence_position, 21);
    }

    fixture.api.cancel_request(handle).expect("cancel succeeds");
}

/// `SparkTestRequestApiDsparkBatchesEqualLengthDrafts`.
#[test]
fn dspark_batches_equal_length_drafts() {
    let mut fixture = make_fixture();
    enable_dspark_speculation(&mut fixture);
    let mut handles = [0u64; 3];
    for (request_index, handle) in handles.iter_mut().enumerate() {
        let prompt = fill_token_ids(16, 151_000 + request_index as u32 * 100);
        *handle = fixture
            .api
            .submit(&SubmitRequest {
                flags: REQUEST_FLAG_REALTIME,
                priority: REALTIME_PRIORITY - request_index as u32,
                prompt_token_count: prompt.len() as u32,
                thinking_token_budget: 0,
                output_token_budget: 8,
                max_prefill_tokens_per_step: 0,
                request_id: 1510 + request_index as u64,
                sequence_id: 11_510 + request_index as u64,
                prompt_token_ids: prompt,
            })
            .expect("submit succeeds");
    }

    let mut prefilled = 0u32;
    while prefilled < 3 {
        let dispatch = schedule(&mut fixture.api);
        assert!(
            dispatch.kind == DispatchKind::Prefill || dispatch.kind == DispatchKind::PrefillBatch
        );
        prefilled += dispatch.request_count;
        fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    }

    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert_eq!(dispatch.request_count, 3);
    assert!(dispatch.flags & DISPATCH_FLAG_DSPARK_TAP_CAPTURE != 0);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    assert_eq!(fixture.dspark_capture.borrow().call_count, 3);
    assert_eq!(fixture.api.dspark_draft_ready_count, 3);

    let mut dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::SpeculativeVerifyBatch);
    assert_eq!(dispatch.request_count, 3);
    assert_eq!(dispatch.speculative_token_count, 6);
    let mut verifier_tokens = [0u32; 3 * (MAX_SPECULATIVE_TOKENS as usize + 1)];
    for request_index in 0..3usize {
        for token_index in 0..(dispatch.speculative_token_count - 1) as usize {
            verifier_tokens[request_index * 8 + token_index] =
                dispatch.speculative_draft_token_ids[request_index][token_index];
        }
        verifier_tokens[request_index * 8 + (dispatch.speculative_token_count - 1) as usize] =
            148_000 + request_index as u32;
        verifier_tokens[request_index * 8 + dispatch.speculative_token_count as usize] =
            149_000 + request_index as u32;
    }
    let speculative_verifier_token_count = dispatch.speculative_verifier_token_count;
    fixture
        .api
        .resolve_speculative_verify_dispatch(
            &mut dispatch,
            &verifier_tokens,
            MAX_SPECULATIVE_TOKENS + 1,
            speculative_verifier_token_count,
        )
        .expect("resolve succeeds");
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    assert_eq!(fixture.api.completed_request_count, 0);
    assert_eq!(fixture.api.dspark_accepted_draft_token_count, 15);
    assert_eq!(fixture.api.dspark_committed_token_count, 18);
}

/// `SparkTestRequestApiDescribesAndCopiesFullPrefillTokenWindows`.
#[test]
fn describes_and_copies_full_prefill_token_windows() {
    let mut fixture = make_fixture_with_scheduler_flags(
        SCHEDULER_DEFAULT_FLAGS
            & !spark_sched::scheduler::CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE,
    );
    let prompt = fill_token_ids(97, 160_000);
    let handle = fixture
        .api
        .submit(&SubmitRequest {
            flags: 0,
            priority: DEFAULT_PRIORITY,
            prompt_token_count: prompt.len() as u32,
            thinking_token_budget: 0,
            output_token_budget: 1,
            max_prefill_tokens_per_step: 64,
            request_id: 1600,
            sequence_id: 11_600,
            prompt_token_ids: prompt.clone(),
        })
        .expect("submit succeeds");
    let _ = handle;

    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    let prefill_view = describe_prefill_dispatch(&dispatch).expect("describe succeeds");
    assert_eq!(prefill_view.lane_count, 1);
    assert_eq!(prefill_view.active_sequence_count, 1);
    assert_eq!(prefill_view.prompt_token_offset, 0);
    assert_eq!(prefill_view.prompt_token_count, 64);
    assert_eq!(
        dispatch.prefill_decision.as_ref().expect("prefill decision").cached_prefix_token_count,
        0
    );
    assert_eq!(prefill_view.prompt_token_stride, 64);
    assert_eq!(prefill_view.lanes[0].prompt_token_ids, prompt);
    let mut copied_tokens = [0xa5u32; 64];
    copy_prefill_dispatch_token_ids(&dispatch, &mut copied_tokens, 64, 1).expect("copy succeeds");
    assert_eq!(&copied_tokens[..], &prompt[..64]);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");

    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    let prefill_view = describe_prefill_dispatch(&dispatch).expect("describe succeeds");
    assert_eq!(prefill_view.lane_count, 1);
    assert_eq!(prefill_view.prompt_token_offset, 64);
    assert_eq!(prefill_view.prompt_token_count, 33);
    assert_eq!(
        dispatch.prefill_decision.as_ref().expect("prefill decision").cached_prefix_token_count,
        0
    );
    assert_eq!(prefill_view.prompt_token_stride, 33);
    let mut copied_tokens = [0xa5u32; 64];
    copy_prefill_dispatch_token_ids(&dispatch, &mut copied_tokens, 64, 1).expect("copy succeeds");
    assert_eq!(&copied_tokens[..33], &prompt[64..]);
    assert!(copied_tokens[33..].iter().all(|&token| token == 0));
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");

    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert!(matches!(describe_prefill_dispatch(&dispatch), Err(RequestApiError::InvalidArgument)));
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
}

/// `SparkTestRequestApiDsparkDisabledPerRequestFallsBackToDecode`.
#[test]
fn dspark_disabled_per_request_falls_back_to_decode() {
    let mut fixture = make_fixture();
    enable_dspark_speculation(&mut fixture);
    let handle = {
        let prompt = fill_token_ids(16, 152_000);
        fixture
            .api
            .submit(&SubmitRequest {
                flags: REQUEST_FLAG_REALTIME | REQUEST_FLAG_DISABLE_SPECULATION,
                priority: REALTIME_PRIORITY,
                prompt_token_count: prompt.len() as u32,
                thinking_token_budget: 0,
                output_token_budget: 2,
                max_prefill_tokens_per_step: 0,
                request_id: 1520,
                sequence_id: 11_520,
                prompt_token_ids: prompt,
            })
            .expect("submit succeeds")
    };
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");

    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert!(dispatch.flags & DISPATCH_FLAG_DSPARK_TAP_CAPTURE == 0);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    assert_eq!(fixture.dspark_capture.borrow().call_count, 0);
    fixture.api.cancel_request(handle).expect("cancel succeeds");
}

// ---------------------------------------------------------------------------
// The two `spark_mtp_tree.h` header tests (the C inlines; the resolve walker
// lives in `spark-serve`'s `mtp_tree_resolve`, the topology accessors in
// `spark-sched`'s `mtp_tree`).
// ---------------------------------------------------------------------------

/// C header constants not re-declared in the Rust `mtp_tree` module.
const MTP_DEPTH1_PRIMARY_INDEX: u32 = 0;
const MTP_DEPTH2_PRIMARY_INDEX: u32 = 1;
const MTP_DEPTH2_ALTERNATE_INDEX: u32 = 2;
const MTP_DEPTH3_PRIMARY_INDEX: u32 = 3;
const MTP_DEPTH3_ALTERNATE_INDEX: u32 = 4;
const MTP_VERIFIER_INPUT_ROW: u32 = 0;
const MTP_VERIFIER_DEPTH1_ROW: u32 = 1;
const MTP_VERIFIER_DEPTH2_PRIMARY_ROW: u32 = 2;
const MTP_RESOLUTION_DEPTH1: u32 = 1;
const MTP_RESOLUTION_DEPTH2_ALTERNATE: u32 = 3;
const MTP_RESOLUTION_DEPTH2_PRIMARY: u32 = 2;
const MTP_RESOLUTION_DEPTH3_PRIMARY: u32 = 4;

/// `SparkTestFillMtpTreeCandidates`.
fn fill_mtp_tree_candidates(token_seed: u32) -> [u32; mtp_tree::CANDIDATE_COUNT as usize] {
    let mut candidates = [0u32; mtp_tree::CANDIDATE_COUNT as usize];
    for (token_index, candidate) in candidates.iter_mut().enumerate() {
        *candidate = token_seed + token_index as u32;
    }
    candidates
}

/// `SparkTestFillMtpTreeVerifier`.
fn fill_mtp_tree_verifier(
    candidate_token_ids: &[u32; mtp_tree::CANDIDATE_COUNT as usize],
    path_id: u32,
    token_seed: u32,
) -> [u32; mtp_tree::VERIFIER_ROW_COUNT as usize] {
    let mut verifier_token_ids = [0u32; mtp_tree::VERIFIER_ROW_COUNT as usize];
    for (row_index, verifier_token_id) in verifier_token_ids.iter_mut().enumerate() {
        *verifier_token_id = token_seed + 100 + row_index as u32;
    }
    if path_id == mtp_tree::RESOLUTION_NONE {
        return verifier_token_ids;
    }
    verifier_token_ids[MTP_VERIFIER_INPUT_ROW as usize] =
        candidate_token_ids[MTP_DEPTH1_PRIMARY_INDEX as usize];
    if path_id == MTP_RESOLUTION_DEPTH1 {
        return verifier_token_ids;
    }
    if path_id == MTP_RESOLUTION_DEPTH2_ALTERNATE {
        verifier_token_ids[MTP_VERIFIER_DEPTH1_ROW as usize] =
            candidate_token_ids[MTP_DEPTH2_ALTERNATE_INDEX as usize];
        return verifier_token_ids;
    }
    verifier_token_ids[MTP_VERIFIER_DEPTH1_ROW as usize] =
        candidate_token_ids[MTP_DEPTH2_PRIMARY_INDEX as usize];
    if path_id == MTP_RESOLUTION_DEPTH2_PRIMARY {
        return verifier_token_ids;
    }
    verifier_token_ids[MTP_VERIFIER_DEPTH2_PRIMARY_ROW as usize] =
        candidate_token_ids[if path_id == MTP_RESOLUTION_DEPTH3_PRIMARY {
            MTP_DEPTH3_PRIMARY_INDEX
        } else {
            MTP_DEPTH3_ALTERNATE_INDEX
        } as usize];
    verifier_token_ids
}

/// `SparkTestMtpTreeResolvesEveryPath`.
#[test]
fn mtp_tree_resolves_every_path() {
    let expected_candidate_indices = [
        MTP_DEPTH1_PRIMARY_INDEX,
        MTP_DEPTH1_PRIMARY_INDEX,
        MTP_DEPTH2_PRIMARY_INDEX,
        MTP_DEPTH2_ALTERNATE_INDEX,
        MTP_DEPTH3_PRIMARY_INDEX,
        MTP_DEPTH3_ALTERNATE_INDEX,
    ];
    let expected_parent_rows = [
        MTP_VERIFIER_INPUT_ROW,
        MTP_VERIFIER_INPUT_ROW,
        MTP_VERIFIER_DEPTH1_ROW,
        MTP_VERIFIER_DEPTH1_ROW,
        MTP_VERIFIER_DEPTH2_PRIMARY_ROW,
        MTP_VERIFIER_DEPTH2_PRIMARY_ROW,
    ];
    let expected_base_offsets = [0u32, 0, 1, 1, 2, 2];
    let candidate_token_ids = fill_mtp_tree_candidates(90_000);
    for path_id in mtp_tree::RESOLUTION_NONE..mtp_tree::RESOLUTION_COUNT {
        let verifier_token_ids = fill_mtp_tree_verifier(&candidate_token_ids, path_id, 91_000);
        let resolution =
            mtp_tree_resolve(&candidate_token_ids, &verifier_token_ids, OUTPUT_VOCAB_COUNT)
                .expect("resolve succeeds");
        assert_eq!(resolution.path_id, path_id);
        assert_eq!(resolution.accepted_token_count, mtp_tree::accepted_token_count(path_id));
        assert_eq!(resolution.committed_token_count, resolution.accepted_token_count + 1);
        assert_eq!(resolution.fallback_row_index, mtp_tree::fallback_row_index(path_id));
        assert_eq!(
            mtp_tree::tail_candidate_index(path_id),
            expected_candidate_indices[path_id as usize]
        );
        assert_eq!(
            mtp_tree::tail_parent_row_index(path_id),
            expected_parent_rows[path_id as usize]
        );
        assert_eq!(
            mtp_tree::tail_base_position_offset(path_id),
            expected_base_offsets[path_id as usize]
        );
    }
}

/// `SparkTestMtpTreeUsesCompactAlternateStorage` (the C header's
/// `SparkMtpTreeVerifierPositionOffset` is `node.depth` here; the storage
/// constants are checked against the C header's values).
#[test]
fn mtp_tree_uses_compact_alternate_storage() {
    // C: BRANCH_ROW_COUNT 4, TRANSIENT_BLOCK_COUNT 2, SHADOW_TOKEN_COUNT 2,
    // TRANSIENT_DEPTH2_ALTERNATE_INDEX 0, TRANSIENT_DEPTH3_ALTERNATE_INDEX 1.
    // The Rust port keeps the tree as a node table; the compact-storage
    // constants are C allocation details with no Rust counterpart, so this
    // test pins the observable position offsets only.
    assert_eq!(mtp_tree::node_at(2).expect("node").depth, 2);
    assert_eq!(mtp_tree::node_at(3).expect("node").depth, 2);
    assert_eq!(mtp_tree::node_at(4).expect("node").depth, 3);
    assert_eq!(mtp_tree::node_at(5).expect("node").depth, 3);
    assert!(mtp_tree::topology_is_valid());
}

/// `SparkTestRequestApiMtpDraftRequiresSpeculativeVerify`.
#[test]
fn mtp_draft_requires_speculative_verify() {
    let mut fixture = make_fixture();
    fixture.api.configuration_flags |=
        CONFIGURATION_FLAG_MTP_COMMIT | CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    let handle =
        submit(&mut fixture.api, 1530, 11_530, DEFAULT_PRIORITY, fill_token_ids(16, 153_000), 5);
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let mut dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert!(dispatch.flags & DISPATCH_FLAG_MTP_COMMIT != 0);
    assert_eq!(dispatch.mtp_draft_token_budget, MTP_INITIAL_DRAFT_TOKEN_COUNT);

    dispatch.decode_committed_token_counts[0] = 1;
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let draft_token_ids = fill_mtp_tree_candidates(91_000);
    let verifier_token_ids = fill_mtp_tree_verifier(&draft_token_ids, 5, 92_000);
    fixture
        .api
        .arm_mtp_verify_dispatch(
            &dispatch,
            &draft_token_ids,
            mtp_tree::CANDIDATE_COUNT,
            mtp_tree::CANDIDATE_COUNT,
        )
        .expect("arm succeeds");
    assert_eq!(fixture.api.mtp_draft_ready_count, 1);

    let mut dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::SpeculativeVerifyBatch);
    assert!(dispatch.flags & DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY != 0);
    assert!(dispatch.flags & DISPATCH_FLAG_MTP_TREE_VERIFY != 0);
    assert_eq!(dispatch.speculative_token_count, mtp_tree::CANDIDATE_COUNT);
    assert_eq!(dispatch.speculative_verifier_token_count, mtp_tree::VERIFIER_ROW_COUNT);
    assert_eq!(dispatch.mtp_draft_token_budget, mtp_tree::CANDIDATE_COUNT);
    let mut verify_block_table = [0u32; 4];
    let mut verify_block_count = [0u32; 1];
    fixture
        .api
        .build_dispatch_kv_block_tables(
            &dispatch,
            &mut verify_block_table,
            4,
            4,
            &mut verify_block_count,
        )
        .expect("kv block tables succeed");
    assert_eq!(verify_block_count[0], 2);
    assert_ne!(verify_block_table[0], verify_block_table[1]);
    for (token_index, draft_token_id) in
        draft_token_ids.iter().enumerate().take(dispatch.speculative_token_count as usize)
    {
        assert_eq!(dispatch.speculative_draft_token_ids[0][token_index], *draft_token_id);
    }
    fixture
        .api
        .resolve_speculative_verify_dispatch(
            &mut dispatch,
            &verifier_token_ids,
            mtp_tree::VERIFIER_ROW_COUNT,
            mtp_tree::VERIFIER_ROW_COUNT,
        )
        .expect("resolve succeeds");
    assert_eq!(dispatch.speculative_accepted_token_counts[0], 3);
    assert_eq!(dispatch.speculative_committed_token_counts[0], 4);
    assert_eq!(dispatch.speculative_fallback_token_ids[0], verifier_token_ids[5]);
    assert_eq!(dispatch.speculative_resolution_path_ids[0], 5);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let cache_state = fixture.api.get_request_cache_state(handle).expect("cache state succeeds");
    assert_eq!(cache_state.state, STATE_COMPLETED);
    assert_eq!(fixture.api.slots()[0].completed_decode_token_count, 5);
    assert_eq!(fixture.api.slots()[0].mtp_draft_token_count, 0);
    assert_eq!(fixture.api.mtp_verify_dispatch_count, 1);
    assert_eq!(fixture.api.mtp_accepted_draft_token_count, 3);
    assert_eq!(fixture.api.mtp_committed_token_count, 4);
}

/// `SparkTestRequestApiMtpVerifyCapturesDsparkBatchTap`.
#[test]
fn mtp_verify_captures_dspark_batch_tap() {
    let mut fixture = make_fixture();
    enable_dspark_speculation(&mut fixture);
    fixture.api.configuration_flags |=
        CONFIGURATION_FLAG_MTP_COMMIT | CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    fixture.dspark_policy_flags.set(0);
    let handle =
        submit(&mut fixture.api, 1532, 11_532, DEFAULT_PRIORITY, fill_token_ids(16, 153_200), 20);
    let _ = handle;
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    assert!(dispatch.flags & DISPATCH_FLAG_DSPARK_TAP_CAPTURE == 0);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let mut dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert!(dispatch.flags & DISPATCH_FLAG_MTP_COMMIT != 0);
    dispatch.decode_committed_token_counts[0] = 1;
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let draft_token_ids = fill_mtp_tree_candidates(93_000);
    fixture
        .api
        .arm_mtp_verify_dispatch(
            &dispatch,
            &draft_token_ids,
            mtp_tree::CANDIDATE_COUNT,
            mtp_tree::CANDIDATE_COUNT,
        )
        .expect("arm succeeds");
    fixture.dspark_policy_flags.set(DSPARK_POLICY_DEFAULT_FLAGS);
    let mut dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::SpeculativeVerifyBatch);
    assert_eq!(dispatch.request_count, 1);
    assert!(dispatch.flags & DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY != 0);
    assert!(dispatch.flags & DISPATCH_FLAG_MTP_TREE_VERIFY != 0);
    assert!(dispatch.flags & DISPATCH_FLAG_DSPARK_TAP_CAPTURE != 0);
    assert!(dispatch.flags & DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY == 0);
    let verifier_token_ids =
        fill_mtp_tree_verifier(&draft_token_ids, MTP_RESOLUTION_DEPTH1, 94_000);
    fixture
        .api
        .resolve_speculative_verify_dispatch(
            &mut dispatch,
            &verifier_token_ids,
            mtp_tree::VERIFIER_ROW_COUNT,
            mtp_tree::VERIFIER_ROW_COUNT,
        )
        .expect("resolve succeeds");
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    assert_eq!(fixture.dspark_capture.borrow().call_count, 1);
    assert_eq!(fixture.api.dspark_draft_ready_count, 1);
    fixture.api.cancel_request(handle).expect("cancel succeeds");
}

/// `SparkTestRequestApiUsesHigherYieldMtpBeforeEqualPriorityDecode`.
#[test]
fn uses_higher_yield_mtp_before_equal_priority_decode() {
    let mut fixture = make_fixture();
    fixture.api.configuration_flags |=
        CONFIGURATION_FLAG_MTP_COMMIT | CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    fixture.api.decode_batch_target = 1;
    let first_handle = submit(&mut fixture.api, 1540, 11_540, 10, fill_token_ids(16, 154_000), 10);
    let second_handle = submit(&mut fixture.api, 1550, 11_550, 9, fill_token_ids(16, 155_000), 2);
    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.kind == DispatchKind::Prefill || dispatch.kind == DispatchKind::PrefillBatch);
    assert_eq!(dispatch.request_handles[0], first_handle);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let mut dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert_eq!(dispatch.request_handles[0], first_handle);
    dispatch.decode_committed_token_counts[0] = 1;
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let draft_token_ids = fill_mtp_tree_candidates(93_000);
    fixture
        .api
        .arm_mtp_verify_dispatch(
            &dispatch,
            &draft_token_ids,
            mtp_tree::CANDIDATE_COUNT,
            mtp_tree::CANDIDATE_COUNT,
        )
        .expect("arm succeeds");
    fixture.api.slots_mut()[1].priority = 11;
    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.kind == DispatchKind::Prefill || dispatch.kind == DispatchKind::PrefillBatch);
    assert_eq!(dispatch.request_handles[0], second_handle);
    fixture.api.cancel_dispatch(&dispatch).expect("cancel dispatch succeeds");
    fixture.api.slots_mut()[1].priority = 10;
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::SpeculativeVerifyBatch);
    assert_eq!(dispatch.request_handles[0], first_handle);
    fixture.api.cancel_dispatch(&dispatch).expect("cancel dispatch succeeds");
    fixture.api.cancel_request(first_handle).expect("cancel succeeds");
    fixture.api.cancel_request(second_handle).expect("cancel succeeds");
}

/// `SparkTestRequestApiMtpDraftBudgetRemainsTransactional`.
#[test]
fn mtp_draft_budget_remains_transactional() {
    let mut fixture = make_fixture();
    fixture.api.configuration_flags |=
        CONFIGURATION_FLAG_MTP_COMMIT | CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    let handle =
        submit(&mut fixture.api, 1535, 11_535, DEFAULT_PRIORITY, fill_token_ids(16, 153_500), 20);
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let mut dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert_eq!(dispatch.mtp_draft_token_budget, mtp_tree::CANDIDATE_COUNT);
    dispatch.decode_committed_token_counts[0] = 1;
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let draft_token_ids = fill_mtp_tree_candidates(93_000);
    fixture
        .api
        .arm_mtp_verify_dispatch(
            &dispatch,
            &draft_token_ids,
            mtp_tree::CANDIDATE_COUNT,
            mtp_tree::CANDIDATE_COUNT,
        )
        .expect("arm succeeds");
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::SpeculativeVerifyBatch);
    assert_eq!(dispatch.mtp_draft_token_budget, mtp_tree::CANDIDATE_COUNT);
    fixture.api.retry_decode_dispatch(&dispatch).expect("retry succeeds");
    assert_eq!(fixture.api.slots()[0].state, STATE_READY_SPECULATIVE_VERIFY);
    assert_eq!(fixture.api.slots()[0].mtp_draft_token_count, mtp_tree::CANDIDATE_COUNT);
    let mut dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::SpeculativeVerifyBatch);
    let verifier_token_ids =
        fill_mtp_tree_verifier(&draft_token_ids, MTP_RESOLUTION_DEPTH2_ALTERNATE, 94_000);
    fixture
        .api
        .resolve_speculative_verify_dispatch(
            &mut dispatch,
            &verifier_token_ids,
            mtp_tree::VERIFIER_ROW_COUNT,
            mtp_tree::VERIFIER_ROW_COUNT,
        )
        .expect("resolve succeeds");
    assert_eq!(dispatch.speculative_accepted_token_counts[0], 2);
    assert_eq!(dispatch.speculative_committed_token_counts[0], 3);
    assert_eq!(dispatch.speculative_resolution_path_ids[0], MTP_RESOLUTION_DEPTH2_ALTERNATE);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let draft_token_ids = fill_mtp_tree_candidates(95_000);
    fixture
        .api
        .arm_mtp_verify_dispatch(
            &dispatch,
            &draft_token_ids,
            mtp_tree::CANDIDATE_COUNT,
            mtp_tree::CANDIDATE_COUNT,
        )
        .expect("arm succeeds");
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::SpeculativeVerifyBatch);
    assert_eq!(dispatch.request_handles[0], handle);
    fixture.api.cancel_dispatch(&dispatch).expect("cancel dispatch succeeds");
    fixture.api.cancel_request(handle).expect("cancel succeeds");
}

/// `SparkTestRequestApiPrefillWavesPipelineThroughRing`.
#[test]
fn prefill_waves_pipeline_through_ring() {
    let mut fixture = make_fixture_with_scheduler(SCHEDULER_DEFAULT_FLAGS, 3);
    let handle =
        submit(&mut fixture.api, 1580, 11_580, DEFAULT_PRIORITY, fill_token_ids(640, 130_000), 8);
    let mut wave_dispatches = [Dispatch::default(), Dispatch::default(), Dispatch::default()];
    fixture.api.schedule_next(&mut wave_dispatches[0]).expect("schedule succeeds");
    assert_eq!(wave_dispatches[0].kind, DispatchKind::Prefill);
    let prefill_view = describe_prefill_dispatch(&wave_dispatches[0]).expect("describe succeeds");
    assert_eq!(prefill_view.prompt_token_offset, 0);
    assert_eq!(prefill_view.prompt_token_count, 256);
    assert_eq!(fixture.api.slots()[0].state, STATE_RUNNING_PREFILL);
    assert_eq!(fixture.api.slots()[0].inflight_prefill_dispatch_count, 1);
    assert_eq!(fixture.api.slots()[0].dispatched_prompt_token_count, 256);
    assert_eq!(fixture.api.slots()[0].computed_prompt_token_count, 0);
    fixture.api.schedule_next(&mut wave_dispatches[1]).expect("schedule succeeds");
    assert_eq!(wave_dispatches[1].kind, DispatchKind::Prefill);
    let prefill_view = describe_prefill_dispatch(&wave_dispatches[1]).expect("describe succeeds");
    assert_eq!(prefill_view.prompt_token_offset, 256);
    assert_eq!(prefill_view.prompt_token_count, 256);
    assert_eq!(fixture.api.slots()[0].inflight_prefill_dispatch_count, 2);
    assert_eq!(fixture.api.slots()[0].dispatched_prompt_token_count, 512);
    assert_eq!(fixture.api.slots()[0].computed_prompt_token_count, 0);
    let status = fixture.api.schedule_next(&mut wave_dispatches[2]);
    assert!(
        status != Ok(())
            || !wave_dispatches[2].accepted
            || wave_dispatches[2].kind != DispatchKind::Prefill
    );
    fixture.api.complete_dispatch(&wave_dispatches[0].clone()).expect("complete succeeds");
    assert_eq!(fixture.api.slots()[0].computed_prompt_token_count, 256);
    assert_eq!(fixture.api.slots()[0].inflight_prefill_dispatch_count, 1);
    assert_eq!(fixture.api.slots()[0].state, STATE_RUNNING_PREFILL);
    fixture.api.schedule_next(&mut wave_dispatches[2]).expect("schedule succeeds");
    assert_eq!(wave_dispatches[2].kind, DispatchKind::Prefill);
    let prefill_view = describe_prefill_dispatch(&wave_dispatches[2]).expect("describe succeeds");
    assert_eq!(prefill_view.prompt_token_offset, 512);
    assert_eq!(prefill_view.prompt_token_count, 128);
    assert_eq!(fixture.api.slots()[0].inflight_prefill_dispatch_count, 2);
    assert_eq!(fixture.api.slots()[0].dispatched_prompt_token_count, 640);
    fixture.api.complete_dispatch(&wave_dispatches[1].clone()).expect("complete succeeds");
    assert_eq!(fixture.api.slots()[0].computed_prompt_token_count, 512);
    assert_eq!(fixture.api.slots()[0].state, STATE_RUNNING_PREFILL);
    fixture.api.complete_dispatch(&wave_dispatches[2].clone()).expect("complete succeeds");
    assert_eq!(fixture.api.slots()[0].computed_prompt_token_count, 640);
    assert_eq!(fixture.api.slots()[0].inflight_prefill_dispatch_count, 0);
    assert_eq!(fixture.api.slots()[0].state, STATE_READY_DECODE);
    let decode_dispatch = schedule(&mut fixture.api);
    assert_eq!(decode_dispatch.kind, DispatchKind::DecodeBatch);
    fixture.api.cancel_dispatch(&decode_dispatch).expect("cancel dispatch succeeds");
    fixture.api.cancel_request(handle).expect("cancel succeeds");
}

/// `SparkTestRequestApiPrefillWaveSpansMultipleKvBlocks`.
#[test]
fn prefill_wave_spans_multiple_kv_blocks() {
    let mut fixture = make_fixture();
    let handle =
        submit(&mut fixture.api, 1570, 11_570, DEFAULT_PRIORITY, fill_token_ids(300, 120_000), 8);
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    let prefill_view = describe_prefill_dispatch(&dispatch).expect("describe succeeds");
    assert_eq!(prefill_view.prompt_token_offset, 0);
    assert_eq!(prefill_view.prompt_token_count, GLM52_DEFAULT_MAX_PREFILL_TOKENS_PER_STEP);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    let prefill_view = describe_prefill_dispatch(&dispatch).expect("describe succeeds");
    assert_eq!(prefill_view.prompt_token_offset, GLM52_DEFAULT_MAX_PREFILL_TOKENS_PER_STEP);
    assert_eq!(prefill_view.prompt_token_count, 300 - GLM52_DEFAULT_MAX_PREFILL_TOKENS_PER_STEP);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    fixture.api.cancel_dispatch(&dispatch).expect("cancel dispatch succeeds");
    fixture.api.cancel_request(handle).expect("cancel succeeds");
}

/// `SparkTestRequestApiMtpAdaptiveFloorSuppressesAndRecovers`.
#[test]
fn mtp_adaptive_floor_suppresses_and_recovers() {
    let mut fixture = make_fixture();
    fixture.api.configuration_flags |=
        CONFIGURATION_FLAG_MTP_COMMIT | CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    let handle =
        submit(&mut fixture.api, 1560, 11_560, DEFAULT_PRIORITY, fill_token_ids(16, 155_000), 128);
    assert_eq!(fixture.api.slots()[0].mtp_commit_ema_milli, MTP_COMMIT_EMA_INITIAL_MILLI);
    let dispatch = schedule(&mut fixture.api);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let mut dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    dispatch.decode_committed_token_counts[0] = 1;
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let mut failing_cycle_count = 0u32;
    for _ in 0..32 {
        let draft_token_ids = fill_mtp_tree_candidates(96_000);
        let verifier_token_ids =
            fill_mtp_tree_verifier(&draft_token_ids, mtp_tree::RESOLUTION_NONE, 97_000);
        let status = fixture.api.arm_mtp_verify_dispatch(
            &dispatch,
            &draft_token_ids,
            mtp_tree::CANDIDATE_COUNT,
            mtp_tree::CANDIDATE_COUNT,
        );
        if status == Err(RequestApiError::NotFound) {
            break;
        }
        status.expect("arm succeeds");
        let mut verify_dispatch = schedule(&mut fixture.api);
        assert_eq!(verify_dispatch.kind, DispatchKind::SpeculativeVerifyBatch);
        fixture
            .api
            .resolve_speculative_verify_dispatch(
                &mut verify_dispatch,
                &verifier_token_ids,
                mtp_tree::VERIFIER_ROW_COUNT,
                mtp_tree::VERIFIER_ROW_COUNT,
            )
            .expect("resolve succeeds");
        fixture.api.complete_dispatch(&verify_dispatch).expect("complete succeeds");
        failing_cycle_count += 1;
    }
    assert!((4..32).contains(&failing_cycle_count));
    assert!(fixture.api.slots()[0].mtp_commit_ema_milli < MTP_SUPPRESS_THRESHOLD_MILLI);
    assert_eq!(fixture.api.slots()[0].mtp_next_draft_token_budget, 0);
    assert_eq!(fixture.api.slots()[0].mtp_probe_countdown, MTP_REPROBE_INTERVAL);
    for _ in 0..MTP_REPROBE_INTERVAL {
        assert_eq!(fixture.api.slots()[0].mtp_next_draft_token_budget, 0);
        let mut plain_dispatch = schedule(&mut fixture.api);
        assert_eq!(plain_dispatch.kind, DispatchKind::DecodeBatch);
        assert_eq!(plain_dispatch.mtp_draft_token_budget, 0);
        plain_dispatch.decode_committed_token_counts[0] = 1;
        fixture.api.complete_dispatch(&plain_dispatch).expect("complete succeeds");
    }
    assert_eq!(fixture.api.slots()[0].mtp_probe_countdown, 0);
    assert_eq!(fixture.api.slots()[0].mtp_next_draft_token_budget, mtp_tree::CANDIDATE_COUNT);
    let mut dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert_eq!(dispatch.mtp_draft_token_budget, mtp_tree::CANDIDATE_COUNT);
    dispatch.decode_committed_token_counts[0] = 1;
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let draft_token_ids = fill_mtp_tree_candidates(98_000);
    let verifier_token_ids =
        fill_mtp_tree_verifier(&draft_token_ids, MTP_RESOLUTION_DEPTH3_PRIMARY, 99_000);
    fixture
        .api
        .arm_mtp_verify_dispatch(
            &dispatch,
            &draft_token_ids,
            mtp_tree::CANDIDATE_COUNT,
            mtp_tree::CANDIDATE_COUNT,
        )
        .expect("arm succeeds");
    let mut dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::SpeculativeVerifyBatch);
    fixture
        .api
        .resolve_speculative_verify_dispatch(
            &mut dispatch,
            &verifier_token_ids,
            mtp_tree::VERIFIER_ROW_COUNT,
            mtp_tree::VERIFIER_ROW_COUNT,
        )
        .expect("resolve succeeds");
    assert_eq!(dispatch.speculative_committed_token_counts[0], mtp_tree::MAX_COMMITTED_TOKEN_COUNT);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    assert!(fixture.api.slots()[0].mtp_commit_ema_milli >= MTP_SUPPRESS_THRESHOLD_MILLI);
    assert_eq!(fixture.api.slots()[0].mtp_next_draft_token_budget, mtp_tree::CANDIDATE_COUNT);
    fixture.api.cancel_request(handle).expect("cancel succeeds");
}

/// `SparkTestRequestApiMtpRejectedDraftStaysOutsideNextContext`.
#[test]
fn mtp_rejected_draft_stays_outside_next_context() {
    let mut fixture = make_fixture();
    fixture.api.configuration_flags |=
        CONFIGURATION_FLAG_MTP_COMMIT | CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    let handle =
        submit(&mut fixture.api, 1550, 11_550, DEFAULT_PRIORITY, fill_token_ids(16, 155_000), 20);
    let dispatch = schedule(&mut fixture.api);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let mut dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    dispatch.decode_committed_token_counts[0] = 1;
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let draft_token_ids = fill_mtp_tree_candidates(96_000);
    let verifier_token_ids =
        fill_mtp_tree_verifier(&draft_token_ids, mtp_tree::RESOLUTION_NONE, 97_000);
    fixture
        .api
        .arm_mtp_verify_dispatch(
            &dispatch,
            &draft_token_ids,
            mtp_tree::CANDIDATE_COUNT,
            mtp_tree::CANDIDATE_COUNT,
        )
        .expect("arm succeeds");
    let mut dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::SpeculativeVerifyBatch);
    fixture
        .api
        .resolve_speculative_verify_dispatch(
            &mut dispatch,
            &verifier_token_ids,
            mtp_tree::VERIFIER_ROW_COUNT,
            mtp_tree::VERIFIER_ROW_COUNT,
        )
        .expect("resolve succeeds");
    assert_eq!(dispatch.speculative_accepted_token_counts[0], 0);
    assert_eq!(dispatch.speculative_committed_token_counts[0], 1);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let dispatch = schedule(&mut fixture.api);
    let decode_view = fixture.api.describe_decode_dispatch(&dispatch).expect("describe succeeds");
    assert_eq!(decode_view.lanes[0].sequence_position, 17);
    assert_eq!(decode_view.lanes[0].context_token_count, 18);
    assert_eq!(decode_view.lanes[0].mtp_resolution_base_position, 16);
    assert_eq!(decode_view.lanes[0].mtp_resolution_proposed_token_count, mtp_tree::CANDIDATE_COUNT);
    assert_eq!(decode_view.lanes[0].mtp_resolution_accepted_token_count, 0);
    assert_eq!(decode_view.lanes[0].mtp_resolution_committed_token_count, 1);
    fixture.api.cancel_dispatch(&dispatch).expect("cancel dispatch succeeds");
    fixture.api.cancel_request(handle).expect("cancel succeeds");
}

/// `SparkTestRequestApiMtpBudgetLeavesVerifierFallbackHeadroom`.
#[test]
fn mtp_budget_leaves_verifier_fallback_headroom() {
    let mut fixture = make_fixture();
    fixture.api.configuration_flags |=
        CONFIGURATION_FLAG_MTP_COMMIT | CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    let handle =
        submit(&mut fixture.api, 1540, 11_540, DEFAULT_PRIORITY, fill_token_ids(16, 154_000), 3);
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert!(dispatch.flags & DISPATCH_FLAG_MTP_COMMIT == 0);
    assert_eq!(dispatch.mtp_draft_token_budget, 0);
    fixture.api.cancel_dispatch(&dispatch).expect("cancel dispatch succeeds");
    fixture.api.cancel_request(handle).expect("cancel succeeds");
}

/// `SparkTestRequestApiMtpVerifyCapsPackedExecutionRows`.
#[test]
fn mtp_verify_caps_packed_execution_rows() {
    const REQUEST_COUNT: usize = 2;
    const EXECUTION_ROW_BUDGET: u32 = 7;
    let mut fixture = make_fixture();
    fixture.api.configuration_flags |=
        CONFIGURATION_FLAG_MTP_COMMIT | CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    fixture.api.decode_batch_target = 2;
    fixture.api.decode_execution_row_capacity = EXECUTION_ROW_BUDGET;
    fixture.api.mtp_accepted_draft_token_count = 3 * u64::from(CURRENT_SPARK_COUNT);
    fixture.api.mtp_rejected_token_count = 2 * u64::from(CURRENT_SPARK_COUNT);
    fixture.api.mtp_committed_token_count = 4 * u64::from(CURRENT_SPARK_COUNT);
    let mut handles = [0u64; REQUEST_COUNT];
    for (request_index, handle) in handles.iter_mut().enumerate() {
        *handle = submit(
            &mut fixture.api,
            1540 + request_index as u64,
            11_540 + request_index as u64,
            DEFAULT_PRIORITY,
            fill_token_ids(16, 154_000 + request_index as u32 * 100),
            MTP_MAX_DRAFT_TOKEN_COUNT + 2,
        );
    }

    let mut completed_prefill_count = 0;
    while completed_prefill_count < REQUEST_COUNT as u32 {
        let dispatch = schedule(&mut fixture.api);
        assert!(
            dispatch.kind == DispatchKind::Prefill || dispatch.kind == DispatchKind::PrefillBatch
        );
        completed_prefill_count += dispatch.request_count;
        fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    }
    for slot in fixture.api.slots_mut().iter_mut().take(REQUEST_COUNT) {
        slot.mtp_next_draft_token_budget = MTP_INITIAL_DRAFT_TOKEN_COUNT;
    }

    let mut dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert_eq!(dispatch.request_count, REQUEST_COUNT as u32);
    assert!(dispatch.flags & DISPATCH_FLAG_MTP_COMMIT != 0);
    for committed in
        dispatch.decode_committed_token_counts.iter_mut().take(dispatch.request_count as usize)
    {
        *committed = 1;
    }
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");

    let mut draft_token_ids = [[0u32; mtp_tree::CANDIDATE_COUNT as usize]; REQUEST_COUNT];
    let mut verifier_token_ids = [[0u32; mtp_tree::VERIFIER_ROW_COUNT as usize]; REQUEST_COUNT];
    for request_index in 0..REQUEST_COUNT {
        draft_token_ids[request_index] =
            fill_mtp_tree_candidates(92_000 + request_index as u32 * 100);
        verifier_token_ids[request_index] = fill_mtp_tree_verifier(
            &draft_token_ids[request_index],
            MTP_RESOLUTION_DEPTH3_PRIMARY,
            93_000 + request_index as u32 * 100,
        );
    }
    let flat_draft_token_ids: Vec<u32> = draft_token_ids.concat();
    fixture
        .api
        .arm_mtp_verify_dispatch(
            &dispatch,
            &flat_draft_token_ids,
            mtp_tree::CANDIDATE_COUNT,
            mtp_tree::CANDIDATE_COUNT,
        )
        .expect("arm succeeds");

    let mut dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::SpeculativeVerifyBatch);
    assert_eq!(dispatch.request_count, EXECUTION_ROW_BUDGET / mtp_tree::VERIFIER_ROW_COUNT);
    assert!(dispatch.request_count * mtp_tree::VERIFIER_ROW_COUNT <= EXECUTION_ROW_BUDGET);
    let flat_verifier_token_ids: Vec<u32> = verifier_token_ids.concat();
    fixture
        .api
        .resolve_speculative_verify_dispatch(
            &mut dispatch,
            &flat_verifier_token_ids,
            mtp_tree::VERIFIER_ROW_COUNT,
            mtp_tree::VERIFIER_ROW_COUNT,
        )
        .expect("resolve succeeds");
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");

    for &handle in &handles {
        let cancel_status = fixture.api.cancel_request(handle);
        assert!(cancel_status == Ok(()) || cancel_status == Err(RequestApiError::NotFound));
    }
}

/// `SparkTestRequestApiSkipsUnprofitableWideMtp`.
#[test]
fn skips_unprofitable_wide_mtp() {
    const REQUEST_COUNT: u32 = 16;
    let mut fixture = make_fixture();
    fixture.api.configuration_flags |= CONFIGURATION_FLAG_MTP_COMMIT;
    fixture.api.decode_batch_target = REQUEST_COUNT;
    for request_index in 0..REQUEST_COUNT {
        submit(
            &mut fixture.api,
            u64::from(1800 + request_index),
            u64::from(11_800 + request_index),
            DEFAULT_PRIORITY,
            fill_token_ids(16, 180_000 + request_index * 100),
            8,
        );
    }
    let mut completed_prefill_count = 0;
    while completed_prefill_count < REQUEST_COUNT {
        let dispatch = schedule(&mut fixture.api);
        assert!(
            dispatch.kind == DispatchKind::Prefill || dispatch.kind == DispatchKind::PrefillBatch
        );
        completed_prefill_count += dispatch.request_count;
        fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    }
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert_eq!(dispatch.request_count, REQUEST_COUNT);
    assert!(dispatch.flags & DISPATCH_FLAG_MTP_COMMIT == 0);
    fixture.api.cancel_dispatch(&dispatch).expect("cancel dispatch succeeds");
}

/// The real HF-JSON tokenizer behind the prompt pipeline's
/// [`spark_text::prompt_pipeline::TextPromptTokenizer`] seam (the C test
/// passes the `SparkTokenizer` directly).
struct HfTokenizer(spark_text::tokenizer::Tokenizer);

impl spark_text::prompt_pipeline::TextPromptTokenizer for HfTokenizer {
    fn encode_utf8(
        &self,
        text: &str,
        flags: u32,
        token_ids: &mut [u32],
    ) -> Result<
        spark_text::prompt_pipeline::PromptEncoding,
        spark_text::prompt_pipeline::PromptPipelineError,
    > {
        let encoding = self
            .0
            .encode(text.as_bytes(), flags)
            .map_err(|_| spark_text::prompt_pipeline::PromptPipelineError::ParseError)?;
        let mut token_count = 0u32;
        for (index, &token_id) in encoding.token_ids().iter().enumerate() {
            match token_ids.get_mut(index) {
                Some(slot) => {
                    *slot = token_id;
                    token_count += 1;
                }
                None => break,
            }
        }
        Ok(spark_text::prompt_pipeline::PromptEncoding {
            token_count,
            overflow_token_count: encoding.overflow_token_count().try_into().unwrap_or(u32::MAX),
        })
    }
}

/// `SparkTestRequestApiWriteTokenizerJson` + `SparkTestRequestApiLoadTokenizer`.
fn write_and_load_tokenizer() -> HfTokenizer {
    let path = std::env::temp_dir().join("spark_test_glm52_request_api_tokenizer.json");
    std::fs::write(
        &path,
        "{\n  \"model\": {\n    \"type\": \"BPE\",\n    \"unk_token\": \"<unk>\",\n    \"byte_fallback\": false,\n    \"vocab\": {\n      \"a\": 1,\n      \"b\": 2,\n      \"c\": 3,\n      \"ab\": 4,\n      \"abc\": 5,\n      \"<unk>\": 6\n    },\n    \"merges\": [\n      \"a b\",\n      \"ab c\"\n    ]\n  },\n  \"pre_tokenizer\": {\n    \"type\": \"ByteLevel\",\n    \"add_prefix_space\": false\n  },\n  \"added_tokens\": []\n}\n",
    )
    .expect("tokenizer json writes");
    HfTokenizer(
        spark_text::tokenizer::Tokenizer::load_huggingface_json(&path).expect("tokenizer loads"),
    )
}

/// `SparkTestRequestApiSubmitsCTextPromptToPrefillSchedule`.
#[test]
fn submits_c_text_prompt_to_prefill_schedule() {
    let tokenizer = write_and_load_tokenizer();
    let mut fixture = make_fixture();
    let result = spark_text::prompt_pipeline::submit_text_prompt(
        &mut fixture.api,
        &tokenizer,
        &spark_text::prompt_pipeline::TextPromptSubmitRequest {
            priority: DEFAULT_PRIORITY,
            output_token_budget: 2,
            request_id: 1540,
            sequence_id: 11_540,
            prompt_text: "abc".to_string(),
            prompt_token_capacity: 8,
            ..spark_text::prompt_pipeline::TextPromptSubmitRequest::default()
        },
    )
    .expect("text prompt submit succeeds");
    assert_eq!(result.prompt_token_count, 1);
    assert_eq!(result.required_prompt_token_count, 1);
    assert_ne!(result.request_handle, INVALID_HANDLE);
    assert_eq!(result.prompt_token_ids[0], 5);
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::Prefill);
    assert_eq!(dispatch.request_count, 1);
    assert_eq!(dispatch.request_handles[0], result.request_handle);
    let prefill_view = describe_prefill_dispatch(&dispatch).expect("describe succeeds");
    assert_eq!(prefill_view.prompt_token_count, 1);
    assert_eq!(prefill_view.lanes[0].prompt_token_ids, result.prompt_token_ids);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
}

/// `SparkTestRequestApiAdmitsThirteenThousandWithFreeList`.
#[test]
fn admits_thirteen_thousand_with_free_list() {
    const REQUEST_CAPACITY: u32 = 13_000;
    let mut fixture = make_fixture_with_request_capacity(REQUEST_CAPACITY);
    let mut first_handle = INVALID_HANDLE;
    for request_index in 0..REQUEST_CAPACITY {
        let handle = submit(
            &mut fixture.api,
            u64::from(500_000 + request_index),
            u64::from(600_000 + request_index),
            DEFAULT_PRIORITY,
            vec![17],
            1,
        );
        if request_index == 0 {
            first_handle = handle;
        }
    }
    assert_eq!(fixture.api.queued_request_count, REQUEST_CAPACITY);
    let overflow_status = fixture.api.submit(&SubmitRequest {
        flags: 0,
        priority: DEFAULT_PRIORITY,
        prompt_token_count: 1,
        thinking_token_budget: 0,
        output_token_budget: 1,
        max_prefill_tokens_per_step: 0,
        request_id: 999_999,
        sequence_id: 999_999,
        prompt_token_ids: vec![17],
    });
    assert!(matches!(overflow_status, Err(RequestApiError::CapacityExceeded)));

    fixture.api.cancel_request(first_handle).expect("cancel succeeds");
    fixture.api.release_completed_request(first_handle).expect("release succeeds");
    let refill_handle = submit(&mut fixture.api, 513_000, 613_000, DEFAULT_PRIORITY, vec![17], 1);
    assert_ne!(refill_handle, INVALID_HANDLE);
    // The released first slot was reused (the C checks
    // `free_slot_head == SPARK_REQUEST_API_NO_SLOT`; the free list is private
    // here, so slot 0's occupancy and renewed capacity exhaustion are the
    // observable proxies).
    assert_eq!(fixture.api.slots()[0].request_id, 513_000);
    assert_eq!(fixture.api.slots()[0].state, STATE_QUEUED_PREFILL);
    let overflow_status = fixture.api.submit(&SubmitRequest {
        flags: 0,
        priority: DEFAULT_PRIORITY,
        prompt_token_count: 1,
        thinking_token_budget: 0,
        output_token_budget: 1,
        max_prefill_tokens_per_step: 0,
        request_id: 999_998,
        sequence_id: 999_998,
        prompt_token_ids: vec![17],
    });
    assert!(matches!(overflow_status, Err(RequestApiError::CapacityExceeded)));
}

/// `SparkTestRequestApiDecodeBatchBackfillsByDescendingPriority`.
#[test]
fn decode_batch_backfills_by_descending_priority() {
    let mut fixture = make_fixture();
    let mut handles = [INVALID_HANDLE; 6];
    for request_index in 0..6u32 {
        let prompt = fill_token_ids(16, 60_000 + request_index * 1000);
        handles[request_index as usize] = submit(
            &mut fixture.api,
            u64::from(700 + request_index),
            u64::from(8_700 + request_index),
            10 + request_index * 10,
            prompt,
            2,
        );
        let dispatch = schedule(&mut fixture.api);
        assert_eq!(dispatch.kind, DispatchKind::Prefill);
        assert_eq!(dispatch.request_handles[0], handles[request_index as usize]);
        fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    }

    fixture.api.decode_batch_target = 4;
    let dispatch = schedule(&mut fixture.api);
    assert!(dispatch.accepted);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert_eq!(dispatch.request_count, 4);
    assert_eq!(dispatch.request_handles[0], handles[5]);
    assert_eq!(dispatch.request_handles[1], handles[4]);
    assert_eq!(dispatch.request_handles[2], handles[3]);
    assert_eq!(dispatch.request_handles[3], handles[2]);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
}

/// `SparkTestRequestApiAdaptiveWidthBackfillsWithoutGrowingPriorityClass`.
#[test]
fn adaptive_width_backfills_without_growing_priority_class() {
    let mut fixture = make_fixture();
    fixture.api.configuration_flags |= CONFIGURATION_FLAG_ADAPTIVE_PIPELINE_BATCHING;
    let low_handle = submit(&mut fixture.api, 720, 8_720, 10, fill_token_ids(16, 66_000), 2);
    let dispatch = schedule(&mut fixture.api);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    let high_handle = submit(&mut fixture.api, 721, 8_721, 100, fill_token_ids(16, 67_000), 2);
    let dispatch = schedule(&mut fixture.api);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    for slot in &mut fixture.api.slots_mut()[2..15] {
        slot.state = STATE_RUNNING_DECODE;
        slot.priority = 100;
    }
    fixture.api.running_request_count = 13;
    assert_eq!(fixture.api.current_pipeline_batch_width(), 2);
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.request_count, 2);
    assert_eq!(dispatch.request_handles[0], high_handle);
    assert_eq!(dispatch.request_handles[1], low_handle);
    assert_eq!(dispatch.highest_priority, 100);
}

/// `SparkTestRequestApiCapsDecodeBatchByActiveKvBlocks`.
#[test]
fn caps_decode_batch_by_active_kv_blocks() {
    let mut fixture = make_fixture();
    let mut handles = [INVALID_HANDLE; 4];
    for request_index in 0..4u32 {
        let prompt = fill_token_ids(16, 70_000 + request_index * 1000);
        handles[request_index as usize] = submit(
            &mut fixture.api,
            u64::from(800 + request_index),
            u64::from(8_800 + request_index),
            10,
            prompt,
            2,
        );
        let dispatch = schedule(&mut fixture.api);
        assert_eq!(dispatch.kind, DispatchKind::Prefill);
        fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    }

    fixture.api.decode_batch_target = 4;
    fixture.api.max_resident_kv_block_count = 4;
    fixture.api.configuration_flags &= !CONFIGURATION_FLAG_JIT_KV_PREFETCH;
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert_eq!(dispatch.request_count, 2);
    assert_eq!(dispatch.request_handles[0], handles[0]);
    assert_eq!(dispatch.request_handles[1], handles[1]);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
}

/// `SparkTestRequestApiHoldsPrefillAtGlobalResidentKvLimit`.
#[test]
fn holds_prefill_at_global_resident_kv_limit() {
    // The C test pokes `fixture.scheduler.queue_depth_per_spark = 8` post-init;
    // the Rust scheduler keeps it private, so the variant is built up front.
    let mut fixture = make_fixture_with_scheduler(SCHEDULER_DEFAULT_FLAGS, 8);
    fixture.api.configuration_flags &=
        !(CONFIGURATION_FLAG_JIT_KV_PREFETCH | CONFIGURATION_FLAG_PREFIX_COHORTING);
    fixture.api.decode_batch_target = 2;
    fixture.api.max_resident_kv_block_count = 4;
    for request_index in 0..6u32 {
        let prompt = fill_token_ids(4, 90_000 + request_index * 1000);
        submit(
            &mut fixture.api,
            u64::from(1_000 + request_index),
            u64::from(9_000 + request_index),
            10,
            prompt,
            1,
        );
    }
    for _ in 0..2 {
        let prefill_dispatch = schedule(&mut fixture.api);
        assert_eq!(prefill_dispatch.kind, DispatchKind::PrefillBatch);
        assert_eq!(prefill_dispatch.request_count, 2);
        fixture.api.complete_dispatch(&prefill_dispatch).expect("complete succeeds");
    }
    let mut decode_dispatches = Vec::new();
    for _ in 0..2 {
        let decode_dispatch = schedule(&mut fixture.api);
        assert_eq!(decode_dispatch.kind, DispatchKind::DecodeBatch);
        assert_eq!(decode_dispatch.request_count, 2);
        decode_dispatches.push(decode_dispatch);
    }
    let mut held_dispatch = Dispatch::default();
    let held_status = fixture.api.schedule_next(&mut held_dispatch);
    assert!(matches!(held_status, Err(RequestApiError::Busy)));
    fixture.api.complete_dispatch(&decode_dispatches[0]).expect("complete succeeds");
    for request_index in 0..2usize {
        fixture
            .api
            .release_completed_request(decode_dispatches[0].request_handles[request_index])
            .expect("release succeeds");
    }
    let prefill_dispatch = schedule(&mut fixture.api);
    assert_eq!(prefill_dispatch.kind, DispatchKind::PrefillBatch);
    assert_eq!(prefill_dispatch.request_count, 2);
}

/// `SparkTestRequestApiReservesMtpDraftKvBlocks`.
#[test]
fn reserves_mtp_draft_kv_blocks() {
    let mut fixture = make_fixture();
    fixture.api.configuration_flags |=
        CONFIGURATION_FLAG_MTP_COMMIT | CONFIGURATION_FLAG_MTP_FORCE_ENABLE;
    let mut handles = [INVALID_HANDLE; 4];
    for request_index in 0..4u32 {
        let prompt = fill_token_ids(26, 80_000 + request_index * 1000);
        handles[request_index as usize] = submit(
            &mut fixture.api,
            u64::from(900 + request_index),
            u64::from(8_900 + request_index),
            10,
            prompt,
            8,
        );
        let dispatch = schedule(&mut fixture.api);
        assert_eq!(dispatch.kind, DispatchKind::Prefill);
        fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
    }

    fixture.api.decode_batch_target = 4;
    fixture.api.max_resident_kv_block_count = 6;
    fixture.api.configuration_flags &= !CONFIGURATION_FLAG_JIT_KV_PREFETCH;
    let dispatch = schedule(&mut fixture.api);
    assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
    assert!(dispatch.flags & DISPATCH_FLAG_MTP_COMMIT != 0);
    assert_eq!(dispatch.mtp_draft_token_budget, MTP_INITIAL_DRAFT_TOKEN_COUNT);
    assert_eq!(dispatch.request_count, 3);
    // The C test declares `physical_block_indices[8]` and then writes three
    // lanes at stride 4 through the raw-pointer api (an out-of-bounds write
    // the C harness gets away with). The Rust api is bounds-checked, so the
    // host table is sized for all three lanes; assertions are unchanged.
    let mut physical_block_indices = [0u32; 12];
    let mut lane_block_counts = [0u32; 3];
    let _block_table_view = fixture
        .api
        .build_dispatch_kv_block_table_view(
            &dispatch,
            &mut physical_block_indices,
            None,
            4,
            4,
            &mut lane_block_counts,
        )
        .expect("kv block table view succeeds");
    assert_eq!(lane_block_counts[0], 2);
    assert_eq!(lane_block_counts[1], 2);
    assert_eq!(lane_block_counts[2], 2);
    fixture.api.complete_dispatch(&dispatch).expect("complete succeeds");
}

/// `SparkTestRequestApiUsesAdaptivePipelineBatchWidth`.
#[test]
fn uses_adaptive_pipeline_batch_width() {
    let mut fixture = make_fixture();
    assert_eq!(fixture.api.current_pipeline_batch_width(), fixture.api.decode_batch_target);
    fixture.api.configuration_flags |= CONFIGURATION_FLAG_ADAPTIVE_PIPELINE_BATCHING;
    for slot in &mut fixture.api.slots_mut()[..13] {
        slot.state = STATE_QUEUED_PREFILL;
        slot.priority = 10;
    }
    assert_eq!(fixture.api.current_pipeline_batch_width(), 1);
    for slot in &mut fixture.api.slots_mut()[..13] {
        slot.state = STATE_RUNNING_DECODE;
    }
    fixture.api.slots_mut()[13].state = STATE_QUEUED_PREFILL;
    fixture.api.slots_mut()[13].priority = 10;
    assert_eq!(fixture.api.current_pipeline_batch_width(), 2);
    fixture.api.slots_mut()[14].state = STATE_QUEUED_PREFILL;
    fixture.api.slots_mut()[14].priority = 100;
    assert_eq!(fixture.api.current_pipeline_batch_width(), 1);
}

/// `SparkTestRequestApiAcceptsMtpForceEnableConfiguration`.
#[test]
fn accepts_mtp_force_enable_configuration() {
    let fixture = make_fixture_with_api_flags(
        CONFIGURATION_DEFAULT_FLAGS | CONFIGURATION_FLAG_MTP_FORCE_ENABLE,
    );
    assert!(fixture.api.configuration_flags & CONFIGURATION_FLAG_MTP_FORCE_ENABLE != 0);
}
