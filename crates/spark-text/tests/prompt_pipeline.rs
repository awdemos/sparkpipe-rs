//! Tests for the prompt pipeline port (`text/prompt_pipeline.c` +
//! `text/prompt.c`).
//!
//! `FakeRequestApi` replays the scheduling behavior of the C test fixture
//! (`tests/test_glm52_prompt_pipeline.c`): one 97-token prompt submitted
//! with a 64-token per-step prefill limit, yielding prefill(0,64) →
//! prefill(64,33) → decode. Its `copy_prefill_dispatch_token_ids` zero-fills
//! per-lane padding exactly like `SparkRequestApiCopyPrefillDispatchTokenIds`.

use std::cell::{Cell, RefCell};

use spark_text::prompt_pipeline::{
    run, submit_text_prompt, DispatchKind, KvBlockTableView, PrefillDispatch, PrefillDispatchView,
    PromptEncoding, PromptPipelineConfiguration, PromptPipelineError, RequestApi, RequestDispatch,
    RunStats, SubmitRequest, TextPromptSubmitRequest, TextPromptTokenizer,
    RUN_FLAG_STOP_AFTER_FIRST_DECODE_DISPATCH,
};

const CONTEXT_TOKENS: usize = 97;
const PREFILL_TOKEN_STRIDE: u32 = 64;
const LANE_CAPACITY: u32 = 4;
const KV_BLOCK_LANES: u32 = 8;

/// Port of the C test fixture's scheduler+request-API behavior.
struct FakeRequestApi {
    prompt_tokens: Vec<u32>,
    max_prefill_tokens_per_step: u32,
    dispatched_prefill_offset: u32,
    decode_issued: bool,
    /// Currently scheduled (not yet completed) prefill slice.
    current_prefill: Option<(u32, u32)>,
    fail_complete: bool,
    fail_submit: bool,
    completed_dispatches: Vec<DispatchKind>,
    cancelled_dispatches: Vec<DispatchKind>,
    submitted_token_counts: Vec<u32>,
    next_handle: u64,
}

impl FakeRequestApi {
    fn new(prompt_tokens: Vec<u32>, max_prefill_tokens_per_step: u32) -> Self {
        FakeRequestApi {
            prompt_tokens,
            max_prefill_tokens_per_step,
            dispatched_prefill_offset: 0,
            decode_issued: false,
            current_prefill: None,
            fail_complete: false,
            fail_submit: false,
            completed_dispatches: Vec::new(),
            cancelled_dispatches: Vec::new(),
            submitted_token_counts: Vec::new(),
            next_handle: 9000,
        }
    }
}

impl RequestApi for FakeRequestApi {
    fn schedule_next(&mut self) -> Result<RequestDispatch, PromptPipelineError> {
        let remaining = self.prompt_tokens.len() as u32 - self.dispatched_prefill_offset;
        if remaining > 0 {
            let count = remaining.min(self.max_prefill_tokens_per_step);
            self.current_prefill = Some((self.dispatched_prefill_offset, count));
            self.dispatched_prefill_offset += count;
            return Ok(RequestDispatch { accepted: true, kind: DispatchKind::Prefill });
        }
        if !self.decode_issued {
            self.decode_issued = true;
            return Ok(RequestDispatch { accepted: true, kind: DispatchKind::DecodeBatch });
        }
        // Queue drained: a not-accepted dispatch (the C pipeline maps this
        // to SPARK_STATUS_BUSY).
        Ok(RequestDispatch { accepted: false, kind: DispatchKind::None })
    }

    fn describe_prefill_dispatch(
        &self,
        dispatch: &RequestDispatch,
    ) -> Result<PrefillDispatchView, PromptPipelineError> {
        if dispatch.kind != DispatchKind::Prefill && dispatch.kind != DispatchKind::PrefillBatch {
            return Err(PromptPipelineError::InvalidArgument);
        }
        let (offset, count) = self.current_prefill.ok_or(PromptPipelineError::InvalidArgument)?;
        Ok(PrefillDispatchView {
            kind: dispatch.kind,
            active_sequence_count: 1,
            prompt_token_offset: offset,
            prompt_token_count: count,
            prompt_token_stride: count,
            lane_count: 1,
        })
    }

    fn copy_prefill_dispatch_token_ids(
        &self,
        dispatch: &RequestDispatch,
        destination_token_ids: &mut [u32],
        destination_token_stride: u32,
        destination_lane_capacity: u32,
    ) -> Result<(), PromptPipelineError> {
        let view = self.describe_prefill_dispatch(dispatch)?;
        if destination_lane_capacity < view.lane_count
            || destination_token_stride < view.prompt_token_stride
        {
            return Err(PromptPipelineError::InvalidArgument);
        }
        let lane = &mut destination_token_ids[..destination_token_stride as usize];
        for (index, slot) in lane.iter_mut().enumerate() {
            *slot = if index < view.prompt_token_count as usize {
                self.prompt_tokens[view.prompt_token_offset as usize + index]
            } else {
                0
            };
        }
        Ok(())
    }

    fn build_dispatch_kv_block_table_view<'a>(
        &self,
        dispatch: &RequestDispatch,
        host_physical_block_indices: &'a mut [u32],
        execution_physical_block_indices: Option<&'a [u32]>,
        lane_stride: u32,
        lane_capacity: u32,
        lane_physical_block_counts: &'a mut [u32],
    ) -> Result<KvBlockTableView<'a>, PromptPipelineError> {
        let view = self.describe_prefill_dispatch(dispatch)?;
        let block_count = view.prompt_token_count.div_ceil(16);
        host_physical_block_indices[..block_count as usize]
            .iter_mut()
            .enumerate()
            .for_each(|(index, slot)| *slot = index as u32);
        lane_physical_block_counts[0] = block_count;
        Ok(KvBlockTableView {
            block_token_count: 16,
            lane_count: view.lane_count,
            lane_stride,
            lane_capacity,
            physical_block_indices: execution_physical_block_indices,
            lane_physical_block_counts: None,
            host_physical_block_indices,
            host_lane_physical_block_counts: lane_physical_block_counts,
        })
    }

    fn complete_dispatch(&mut self, dispatch: &RequestDispatch) -> Result<(), PromptPipelineError> {
        if self.fail_complete {
            return Err(PromptPipelineError::InternalError);
        }
        self.completed_dispatches.push(dispatch.kind);
        Ok(())
    }

    fn cancel_dispatch(&mut self, dispatch: &RequestDispatch) -> Result<(), PromptPipelineError> {
        self.cancelled_dispatches.push(dispatch.kind);
        Ok(())
    }

    fn submit(&mut self, request: &SubmitRequest) -> Result<u64, PromptPipelineError> {
        if self.fail_submit {
            return Err(PromptPipelineError::Busy);
        }
        assert!(request.prompt_token_ids.len() as u32 <= request.prompt_token_count);
        self.submitted_token_counts.push(request.prompt_token_count);
        let handle = self.next_handle;
        self.next_handle += 1;
        Ok(handle)
    }
}

fn fill_token_ids(count: u32, first_token_id: u32) -> Vec<u32> {
    (0..count).map(|index| first_token_id + index).collect()
}

struct Fixture {
    prefill_token_staging: Vec<u32>,
    physical_block_indices: Vec<u32>,
    lane_physical_block_counts: Vec<u32>,
}

impl Fixture {
    fn new() -> Self {
        Fixture {
            prefill_token_staging: vec![0; (PREFILL_TOKEN_STRIDE * LANE_CAPACITY) as usize],
            physical_block_indices: vec![0; (KV_BLOCK_LANES * LANE_CAPACITY) as usize],
            lane_physical_block_counts: vec![0; LANE_CAPACITY as usize],
        }
    }

    fn configuration(&mut self, run_flags: u32) -> PromptPipelineConfiguration<'_> {
        PromptPipelineConfiguration {
            run_flags,
            max_dispatch_steps: 0,
            host_prefill_token_ids: &mut self.prefill_token_staging,
            host_prefill_token_stride: PREFILL_TOKEN_STRIDE,
            host_prefill_lane_capacity: LANE_CAPACITY,
            host_physical_block_indices: &mut self.physical_block_indices,
            execution_physical_block_indices: None,
            kv_block_lane_stride: KV_BLOCK_LANES,
            kv_block_lane_capacity: KV_BLOCK_LANES,
            lane_physical_block_counts: &mut self.lane_physical_block_counts,
            lane_count_capacity: LANE_CAPACITY,
        }
    }
}

/// Port of `SparkTestPromptPipelineRunsPrefillThenDecodeWithoutPython`.
#[test]
fn runs_prefill_then_decode_without_python() {
    let prompt_tokens = fill_token_ids(CONTEXT_TOKENS as u32, 70_000);
    let mut api = FakeRequestApi::new(prompt_tokens.clone(), PREFILL_TOKEN_STRIDE);
    let mut fixture = Fixture::new();
    let mut configuration = fixture.configuration(RUN_FLAG_STOP_AFTER_FIRST_DECODE_DISPATCH);
    let mut stats = RunStats::default();

    let prefill_callback_count = Cell::new(0u32);
    let decode_callback_count = Cell::new(0u32);
    let expected_tokens = RefCell::new(prompt_tokens);

    let mut prefill = |dispatch: &PrefillDispatch| {
        let callback_count = prefill_callback_count.get();
        assert_eq!(dispatch.active_sequence_count, 1);
        assert_eq!(dispatch.lane_count, 1);
        assert_eq!(dispatch.dispatch_kind, DispatchKind::Prefill);

        let expected_offset = if callback_count == 0 { 0 } else { 64 };
        let expected_count = if callback_count == 0 { 64 } else { 33 };
        assert_eq!(dispatch.prompt_token_offset, expected_offset);
        assert_eq!(dispatch.prompt_token_count, expected_count);
        assert_eq!(dispatch.prompt_token_stride, expected_count);
        assert_eq!(dispatch.step_index, callback_count);

        let expected = expected_tokens.borrow();
        for (index, token) in dispatch.host_token_ids[..expected_count as usize].iter().enumerate()
        {
            assert_eq!(*token, expected[expected_offset as usize + index]);
        }
        // Per-lane padding is zero-filled (SparkRequestApiCopyPrefillDispatchTokenIds).
        for index in expected_count as usize..dispatch.host_token_stride as usize {
            assert_eq!(dispatch.host_token_ids[index], 0);
        }

        assert_eq!(dispatch.kv_block_table_view.lane_count, 1);
        assert!(!dispatch.kv_block_table_view.host_physical_block_indices.is_empty());
        assert_eq!(
            dispatch.kv_block_table_view.host_lane_physical_block_counts[0],
            expected_count.div_ceil(16)
        );

        prefill_callback_count.set(callback_count + 1);
        Ok(())
    };
    let mut decode = |dispatch: &RequestDispatch| {
        assert_eq!(dispatch.kind, DispatchKind::DecodeBatch);
        decode_callback_count.set(decode_callback_count.get() + 1);
        Ok(())
    };

    run(&mut api, &mut configuration, &mut prefill, &mut decode, &mut stats).unwrap();

    assert_eq!(stats.completed_dispatch_count, 3);
    assert_eq!(stats.prefill_dispatch_count, 2);
    assert_eq!(stats.decode_dispatch_count, 1);
    assert_eq!(stats.speculative_verify_dispatch_count, 0);
    assert_eq!(stats.prefill_token_count, CONTEXT_TOKENS as u32);
    assert_eq!(stats.maximum_prefill_token_count, 64);
    assert_eq!(stats.maximum_prefill_lane_count, 1);
    assert_eq!(stats.last_dispatch_kind, DispatchKind::DecodeBatch);
    assert!(stats.reached_decode_dispatch);
    assert_eq!(prefill_callback_count.get(), 2);
    assert_eq!(decode_callback_count.get(), 1);
    assert_eq!(
        api.completed_dispatches,
        vec![DispatchKind::Prefill, DispatchKind::Prefill, DispatchKind::DecodeBatch]
    );
    assert!(api.cancelled_dispatches.is_empty());
}

#[test]
fn speculative_verify_dispatch_counts_separately() {
    struct VerifyApi {
        steps: u32,
    }
    impl RequestApi for VerifyApi {
        fn schedule_next(&mut self) -> Result<RequestDispatch, PromptPipelineError> {
            self.steps += 1;
            Ok(match self.steps {
                1 => RequestDispatch { accepted: true, kind: DispatchKind::DecodeBatch },
                2 => RequestDispatch { accepted: true, kind: DispatchKind::SpeculativeVerifyBatch },
                _ => RequestDispatch { accepted: false, kind: DispatchKind::None },
            })
        }
        fn describe_prefill_dispatch(
            &self,
            _dispatch: &RequestDispatch,
        ) -> Result<PrefillDispatchView, PromptPipelineError> {
            Err(PromptPipelineError::InvalidArgument)
        }
        fn copy_prefill_dispatch_token_ids(
            &self,
            _dispatch: &RequestDispatch,
            _destination_token_ids: &mut [u32],
            _destination_token_stride: u32,
            _destination_lane_capacity: u32,
        ) -> Result<(), PromptPipelineError> {
            Err(PromptPipelineError::InvalidArgument)
        }
        fn build_dispatch_kv_block_table_view<'a>(
            &self,
            _dispatch: &RequestDispatch,
            host: &'a mut [u32],
            execution: Option<&'a [u32]>,
            lane_stride: u32,
            lane_capacity: u32,
            counts: &'a mut [u32],
        ) -> Result<KvBlockTableView<'a>, PromptPipelineError> {
            Ok(KvBlockTableView {
                block_token_count: 16,
                lane_count: 0,
                lane_stride,
                lane_capacity,
                physical_block_indices: execution,
                lane_physical_block_counts: None,
                host_physical_block_indices: host,
                host_lane_physical_block_counts: counts,
            })
        }
        fn complete_dispatch(
            &mut self,
            _dispatch: &RequestDispatch,
        ) -> Result<(), PromptPipelineError> {
            Ok(())
        }
        fn cancel_dispatch(
            &mut self,
            _dispatch: &RequestDispatch,
        ) -> Result<(), PromptPipelineError> {
            Ok(())
        }
        fn submit(&mut self, _request: &SubmitRequest) -> Result<u64, PromptPipelineError> {
            Err(PromptPipelineError::Unsupported)
        }
    }

    let mut api = VerifyApi { steps: 0 };
    let mut fixture = Fixture::new();
    let mut configuration = fixture.configuration(0);
    let mut stats = RunStats::default();
    let mut prefill = |_dispatch: &PrefillDispatch| Ok(());
    let mut decode = |_dispatch: &RequestDispatch| Ok(());

    // Without the stop flag, the loop ends on the drained queue (BUSY).
    let status = run(&mut api, &mut configuration, &mut prefill, &mut decode, &mut stats);
    assert_eq!(status, Err(PromptPipelineError::Busy));
    assert_eq!(stats.decode_dispatch_count, 1);
    assert_eq!(stats.speculative_verify_dispatch_count, 1);
    assert!(stats.reached_decode_dispatch);
}

#[test]
fn not_accepted_dispatch_maps_to_busy() {
    struct EmptyApi;
    impl RequestApi for EmptyApi {
        fn schedule_next(&mut self) -> Result<RequestDispatch, PromptPipelineError> {
            Ok(RequestDispatch { accepted: false, kind: DispatchKind::None })
        }
        fn describe_prefill_dispatch(
            &self,
            _d: &RequestDispatch,
        ) -> Result<PrefillDispatchView, PromptPipelineError> {
            unreachable!()
        }
        fn copy_prefill_dispatch_token_ids(
            &self,
            _d: &RequestDispatch,
            _t: &mut [u32],
            _s: u32,
            _c: u32,
        ) -> Result<(), PromptPipelineError> {
            unreachable!()
        }
        fn build_dispatch_kv_block_table_view<'a>(
            &self,
            _d: &RequestDispatch,
            _h: &'a mut [u32],
            _e: Option<&'a [u32]>,
            _s: u32,
            _c: u32,
            _n: &'a mut [u32],
        ) -> Result<KvBlockTableView<'a>, PromptPipelineError> {
            unreachable!()
        }
        fn complete_dispatch(&mut self, _d: &RequestDispatch) -> Result<(), PromptPipelineError> {
            unreachable!()
        }
        fn cancel_dispatch(&mut self, _d: &RequestDispatch) -> Result<(), PromptPipelineError> {
            unreachable!()
        }
        fn submit(&mut self, _r: &SubmitRequest) -> Result<u64, PromptPipelineError> {
            unreachable!()
        }
    }

    let mut api = EmptyApi;
    let mut fixture = Fixture::new();
    let mut configuration = fixture.configuration(RUN_FLAG_STOP_AFTER_FIRST_DECODE_DISPATCH);
    let mut stats = RunStats::default();
    let mut prefill = |_dispatch: &PrefillDispatch| Ok(());
    let mut decode = |_dispatch: &RequestDispatch| Ok(());
    let status = run(&mut api, &mut configuration, &mut prefill, &mut decode, &mut stats);
    assert_eq!(status, Err(PromptPipelineError::Busy));
    assert_eq!(stats, RunStats::default());
}

#[test]
fn zero_lane_capacity_is_invalid() {
    let prompt_tokens = fill_token_ids(CONTEXT_TOKENS as u32, 70_000);
    let mut api = FakeRequestApi::new(prompt_tokens, PREFILL_TOKEN_STRIDE);
    let mut fixture = Fixture::new();
    let mut configuration = fixture.configuration(0);
    // Stride 64 fits, but a tiny lane capacity does not.
    configuration.host_prefill_lane_capacity = 0;
    configuration.host_prefill_token_ids = &mut [];
    assert_eq!(
        run(
            &mut api,
            &mut configuration,
            &mut |_dispatch: &PrefillDispatch| Ok(()),
            &mut |_dispatch: &RequestDispatch| Ok(()),
            &mut RunStats::default(),
        ),
        Err(PromptPipelineError::InvalidArgument)
    );
}

#[test]
fn prefill_stride_overflow_maps_to_capacity_exceeded() {
    let prompt_tokens = fill_token_ids(CONTEXT_TOKENS as u32, 70_000);
    let mut api = FakeRequestApi::new(prompt_tokens, PREFILL_TOKEN_STRIDE);
    let mut fixture = Fixture::new();
    let mut configuration = fixture.configuration(0);
    // Host stride 32 < dispatch stride 64 -> SPARK_STATUS_CAPACITY_EXCEEDED.
    configuration.host_prefill_token_stride = 32;
    let mut stats = RunStats::default();
    let status = run(
        &mut api,
        &mut configuration,
        &mut |_dispatch: &PrefillDispatch| Ok(()),
        &mut |_dispatch: &RequestDispatch| Ok(()),
        &mut stats,
    );
    assert_eq!(status, Err(PromptPipelineError::CapacityExceeded));
    // The failed dispatch was cancelled, and stats were still delivered.
    assert_eq!(api.cancelled_dispatches, vec![DispatchKind::Prefill]);
    assert_eq!(stats.completed_dispatch_count, 1);
    assert_eq!(stats.prefill_dispatch_count, 0);
}

#[test]
fn prefill_callback_failure_cancels_dispatch() {
    let prompt_tokens = fill_token_ids(CONTEXT_TOKENS as u32, 70_000);
    let mut api = FakeRequestApi::new(prompt_tokens, PREFILL_TOKEN_STRIDE);
    let mut fixture = Fixture::new();
    let mut configuration = fixture.configuration(RUN_FLAG_STOP_AFTER_FIRST_DECODE_DISPATCH);
    let mut stats = RunStats::default();
    let status = run(
        &mut api,
        &mut configuration,
        &mut |_dispatch: &PrefillDispatch| Err(PromptPipelineError::InternalError),
        &mut |_dispatch: &RequestDispatch| Ok(()),
        &mut stats,
    );
    assert_eq!(status, Err(PromptPipelineError::InternalError));
    assert_eq!(api.cancelled_dispatches, vec![DispatchKind::Prefill]);
    assert!(api.completed_dispatches.is_empty());
    assert_eq!(stats.completed_dispatch_count, 1);
    assert!(!stats.reached_decode_dispatch);
}

#[test]
fn decode_callback_failure_cancels_dispatch() {
    let prompt_tokens = fill_token_ids(CONTEXT_TOKENS as u32, 70_000);
    let mut api = FakeRequestApi::new(prompt_tokens, PREFILL_TOKEN_STRIDE);
    let mut fixture = Fixture::new();
    let mut configuration = fixture.configuration(RUN_FLAG_STOP_AFTER_FIRST_DECODE_DISPATCH);
    let mut stats = RunStats::default();
    let status = run(
        &mut api,
        &mut configuration,
        &mut |_dispatch: &PrefillDispatch| Ok(()),
        &mut |_dispatch: &RequestDispatch| Err(PromptPipelineError::InternalError),
        &mut stats,
    );
    assert_eq!(status, Err(PromptPipelineError::InternalError));
    assert_eq!(api.completed_dispatches, vec![DispatchKind::Prefill, DispatchKind::Prefill]);
    assert_eq!(api.cancelled_dispatches, vec![DispatchKind::DecodeBatch]);
    assert_eq!(stats.prefill_dispatch_count, 2);
    // The decode failure happens before decode counters/reached flag update,
    // exactly as in the C invoke-decode ordering.
    assert_eq!(stats.decode_dispatch_count, 0);
    assert!(!stats.reached_decode_dispatch);
}

#[test]
fn complete_dispatch_failure_propagates() {
    let prompt_tokens = fill_token_ids(CONTEXT_TOKENS as u32, 70_000);
    let mut api = FakeRequestApi::new(prompt_tokens, PREFILL_TOKEN_STRIDE);
    api.fail_complete = true;
    let mut fixture = Fixture::new();
    let mut configuration = fixture.configuration(RUN_FLAG_STOP_AFTER_FIRST_DECODE_DISPATCH);
    let mut stats = RunStats::default();
    let status = run(
        &mut api,
        &mut configuration,
        &mut |_dispatch: &PrefillDispatch| Ok(()),
        &mut |_dispatch: &RequestDispatch| Ok(()),
        &mut stats,
    );
    assert_eq!(status, Err(PromptPipelineError::InternalError));
    assert_eq!(stats.prefill_dispatch_count, 1);
}

#[test]
fn exhausted_step_budget_maps_to_capacity_exceeded() {
    let prompt_tokens = fill_token_ids(CONTEXT_TOKENS as u32, 70_000);
    let mut api = FakeRequestApi::new(prompt_tokens, PREFILL_TOKEN_STRIDE);
    let mut fixture = Fixture::new();
    let mut configuration = fixture.configuration(0);
    configuration.max_dispatch_steps = 1;
    let mut stats = RunStats::default();
    let status = run(
        &mut api,
        &mut configuration,
        &mut |_dispatch: &PrefillDispatch| Ok(()),
        &mut |_dispatch: &RequestDispatch| Ok(()),
        &mut stats,
    );
    assert_eq!(status, Err(PromptPipelineError::CapacityExceeded));
    assert_eq!(stats.prefill_dispatch_count, 1);
    assert_eq!(stats.prefill_token_count, 64);
}

#[test]
fn rejects_invalid_configurations() {
    let mut api = FakeRequestApi::new(vec![1], 1);
    let mut fixture = Fixture::new();
    let mut stats = RunStats::default();

    // Unknown run flag.
    let mut configuration = fixture.configuration(0x0000_0002);
    assert_eq!(
        run(&mut api, &mut configuration, &mut |_| Ok(()), &mut |_| Ok(()), &mut stats),
        Err(PromptPipelineError::InvalidArgument)
    );

    // kv_block_lane_stride < kv_block_lane_capacity.
    let mut configuration = fixture.configuration(0);
    configuration.kv_block_lane_stride = 2;
    assert_eq!(
        run(&mut api, &mut configuration, &mut |_| Ok(()), &mut |_| Ok(()), &mut stats),
        Err(PromptPipelineError::InvalidArgument)
    );

    // Undersized staging buffer.
    let mut configuration = fixture.configuration(0);
    configuration.host_prefill_token_ids = &mut [];
    assert_eq!(
        run(&mut api, &mut configuration, &mut |_| Ok(()), &mut |_| Ok(()), &mut stats),
        Err(PromptPipelineError::InvalidArgument)
    );
}

// ---------------------------------------------------------------------------
// text/prompt.c — submit_text_prompt
// ---------------------------------------------------------------------------

/// Stub tokenizer standing in for the concurrent `src/tokenizer.rs` port:
/// one token per whitespace-separated word, ids assigned in order.
struct StubTokenizer {
    fail: bool,
}

impl TextPromptTokenizer for StubTokenizer {
    fn encode_utf8(
        &self,
        text: &str,
        _flags: u32,
        token_ids: &mut [u32],
    ) -> Result<PromptEncoding, PromptPipelineError> {
        if self.fail {
            return Err(PromptPipelineError::ParseError);
        }
        let mut token_count = 0u32;
        let mut overflow_token_count = 0u32;
        for (index, _word) in text.split_whitespace().enumerate() {
            match token_ids.get_mut(index) {
                Some(slot) => {
                    *slot = 500 + index as u32;
                    token_count += 1;
                }
                None => overflow_token_count += 1,
            }
        }
        Ok(PromptEncoding { token_count, overflow_token_count })
    }
}

fn submit_request(prompt_text: &str, capacity: u32) -> TextPromptSubmitRequest {
    TextPromptSubmitRequest {
        output_token_budget: 1,
        max_prefill_tokens_per_step: PREFILL_TOKEN_STRIDE,
        request_id: 701,
        sequence_id: 1701,
        prompt_text: prompt_text.to_string(),
        prompt_token_capacity: capacity,
        ..TextPromptSubmitRequest::default()
    }
}

#[test]
fn submit_text_prompt_encodes_and_submits() {
    let mut api = FakeRequestApi::new(Vec::new(), PREFILL_TOKEN_STRIDE);
    let tokenizer = StubTokenizer { fail: false };
    let result = submit_text_prompt(&mut api, &tokenizer, &submit_request("a b c", 8)).unwrap();
    assert_eq!(result.prompt_token_count, 3);
    assert_eq!(result.required_prompt_token_count, 3);
    assert_eq!(result.prompt_token_ids, vec![500, 501, 502]);
    assert_ne!(result.request_handle, 0);
    assert_eq!(api.submitted_token_counts, vec![3]);
}

#[test]
fn submit_text_prompt_reports_overflow_in_required_count() {
    let mut api = FakeRequestApi::new(Vec::new(), PREFILL_TOKEN_STRIDE);
    let tokenizer = StubTokenizer { fail: false };
    let result = submit_text_prompt(&mut api, &tokenizer, &submit_request("a b c", 2)).unwrap();
    assert_eq!(result.prompt_token_count, 2);
    assert_eq!(result.required_prompt_token_count, 3);
    assert_eq!(result.prompt_token_ids, vec![500, 501]);
    // The C submits with the *required* count (caller is expected to size
    // storage accordingly).
    assert_eq!(api.submitted_token_counts, vec![3]);
}

#[test]
fn submit_text_prompt_rejects_zero_capacity_and_unknown_flags() {
    let mut api = FakeRequestApi::new(Vec::new(), PREFILL_TOKEN_STRIDE);
    let tokenizer = StubTokenizer { fail: false };

    let mut request = submit_request("a", 0);
    assert_eq!(
        submit_text_prompt(&mut api, &tokenizer, &request).unwrap_err(),
        PromptPipelineError::InvalidArgument
    );

    request = submit_request("a", 4);
    request.tokenizer_encode_flags = 0x0000_0010;
    assert_eq!(
        submit_text_prompt(&mut api, &tokenizer, &request).unwrap_err(),
        PromptPipelineError::InvalidArgument
    );
}

#[test]
fn submit_text_prompt_rejects_empty_encoding() {
    let mut api = FakeRequestApi::new(Vec::new(), PREFILL_TOKEN_STRIDE);
    let tokenizer = StubTokenizer { fail: false };
    // Empty prompt text encodes to zero tokens -> SPARK_STATUS_INVALID_ARGUMENT.
    assert_eq!(
        submit_text_prompt(&mut api, &tokenizer, &submit_request("", 4)).unwrap_err(),
        PromptPipelineError::InvalidArgument
    );
    assert!(api.submitted_token_counts.is_empty());
}

#[test]
fn submit_text_prompt_propagates_encode_failure() {
    let mut api = FakeRequestApi::new(Vec::new(), PREFILL_TOKEN_STRIDE);
    let tokenizer = StubTokenizer { fail: true };
    assert_eq!(
        submit_text_prompt(&mut api, &tokenizer, &submit_request("a", 4)).unwrap_err(),
        PromptPipelineError::ParseError
    );
    assert!(api.submitted_token_counts.is_empty());
}

#[test]
fn submit_text_prompt_propagates_submit_failure() {
    let mut api = FakeRequestApi::new(Vec::new(), PREFILL_TOKEN_STRIDE);
    api.fail_submit = true;
    let tokenizer = StubTokenizer { fail: false };
    assert_eq!(
        submit_text_prompt(&mut api, &tokenizer, &submit_request("a", 4)).unwrap_err(),
        PromptPipelineError::Busy
    );
}
