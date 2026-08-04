//! Port of `tests/test_glm52_scheduler.c`. GPU-free: the C test runs against
//! a stub-backed prefix cache; here the owned [`PrefixCache`] + [`KvArena`]
//! pair plays the same role. The geometry the C test compiles in from
//! `SPARK_GLM52_MODEL_*` (78 layers, first routed layer 3, 256-token default
//! prefill dispatch, 1M-token context) is supplied explicitly through
//! [`SchedulerConfig`], matching the Rust port's config-fields deviation.
//!
//! Intentional differences from the C test:
//! - The "quantization mode 99 is rejected" case is dropped: the Rust config
//!   takes the [`QuantizationMode`] enum, so an unknown mode is
//!   unrepresentable rather than validated.
//! - Where the C re-initializes the scheduler over the same cache, each
//!   [`Scheduler::new`] here gets a fresh cache (the scheduler owns it);
//!   those paths never touch the cache, so this is observationally
//!   equivalent.
//! - Decode-batch validation failures return `Err` without a decision (the C
//!   also fills a rejected batch decision); the test asserts the error and
//!   the unchanged statistics instead.

use spark_core::kv_arena::{KvArena, KvArenaConfig};
use spark_core::prefix_cache::{PrefixCache, PrefixCacheConfig};
use spark_sched::scheduler::{
    Scheduler, SchedulerConfig, SchedulerError, SchedulerRequest, CONFIGURATION_DEFAULT_FLAGS,
    CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE,
    CONFIGURATION_FLAG_MEASURED_DECODE_BUCKET_SELECTION, DECISION_FLAG_ADAPTIVE_DECODE_PACK,
    DECISION_FLAG_CUDAGRAPH_PADDING, DECISION_FLAG_DECODE_BYPASS_PREFILL,
    DECISION_FLAG_DECODE_STEP, DECISION_FLAG_MEASURED_DECODE_BUCKET, DECISION_FLAG_PREFILL_CHUNK,
    DECISION_FLAG_PREFILL_FINAL_CHUNK, DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT,
    DECISION_FLAG_PREFIX_CACHE_USED, DISPATCH_STAGE_FLAG_ADAPTIVE_DECODE_PACK,
    DISPATCH_STAGE_FLAG_DECODE, DISPATCH_STAGE_FLAG_DECODE_BYPASS_PREFILL,
    DISPATCH_STAGE_FLAG_MEASURED_DECODE_BUCKET, DISPATCH_STAGE_FLAG_PREFILL_CHUNK,
    PREFILL_BLOCK_TOKENS,
};
use spark_sched::stage_plan::{
    QuantizationMode, StagePlanGeometry, BUCKET_B16, BUCKET_B32, BUCKET_B64, CURRENT_SPARK_COUNT,
    MAX_BATCH_BUCKET, MEASURED_PROFILE_20260701,
};

/// The GLM-5.2 geometry the C test gets from `SPARK_GLM52_MODEL_*`.
const GLM52_GEOMETRY: StagePlanGeometry =
    StagePlanGeometry { layer_count: 78, first_routed_layer: 3 };
/// `SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH`.
const GLM52_DEFAULT_MAX_PREFILL_TOKENS_PER_STEP: u32 = 256;
/// `SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS`.
const GLM52_MAX_CONTEXT_TOKENS: u32 = 1_048_576;

/// `SparkTestFillTokenIds`.
fn fill_token_ids(token_count: u32, base_token_id: u32) -> Vec<u32> {
    (0..token_count).map(|index| base_token_id + index).collect()
}

/// `SparkTestBuildSharedPrefixPrompt`.
fn build_shared_prefix_prompt(
    shared_prefix_token_ids: &[u32],
    suffix_base_token_id: u32,
    suffix_token_count: u32,
) -> Vec<u32> {
    let mut prompt = shared_prefix_token_ids.to_vec();
    prompt.extend((0..suffix_token_count).map(|index| suffix_base_token_id + index));
    prompt
}

/// `SparkTestInitializePrefixCache`.
fn make_prefix_cache(entry_count: u32, binding_count: u32) -> PrefixCache {
    let arena = KvArena::new(&KvArenaConfig {
        physical_block_count: entry_count,
        block_token_count: PREFILL_BLOCK_TOKENS,
        resident_block_capacity: 0,
        layer_count: 78,
        kv_head_count: 8,
        head_dim: 128,
        bytes_per_scalar: 2,
        key_block_stride_bytes: 0,
        value_block_stride_bytes: 0,
        key_device_base: 0x1_0000_0000,
        value_device_base: 0x2_0000_0000,
    })
    .expect("arena configuration is valid");
    PrefixCache::new(
        &PrefixCacheConfig {
            block_token_count: PREFILL_BLOCK_TOKENS,
            entry_count,
            physical_block_count: entry_count,
            sequence_binding_count: binding_count,
        },
        arena,
    )
    .expect("prefix cache configuration is valid")
}

/// `SparkTestInitializeSchedulerConfiguration`.
fn make_config(quantization_mode: QuantizationMode, prefix_cache: PrefixCache) -> SchedulerConfig {
    SchedulerConfig {
        spark_count: CURRENT_SPARK_COUNT,
        queue_depth_per_spark: 1,
        measured_profile_id: MEASURED_PROFILE_20260701,
        stage_geometry: GLM52_GEOMETRY,
        estimated_layer_cost_ns: 0,
        estimated_final_stage_extra_cost_ns: 0,
        quantization_mode,
        max_prefill_tokens_per_step: 0,
        default_max_prefill_tokens_per_step: GLM52_DEFAULT_MAX_PREFILL_TOKENS_PER_STEP,
        max_context_tokens: GLM52_MAX_CONTEXT_TOKENS,
        max_batch_bucket: MAX_BATCH_BUCKET,
        prefix_cache_block_tokens: PREFILL_BLOCK_TOKENS,
        configuration_flags: CONFIGURATION_DEFAULT_FLAGS,
        prefix_cache: Some(prefix_cache),
    }
}

/// `SparkTestInitializeDecodeRequest`.
fn decode_request(active_sequence_count: u32) -> SchedulerRequest<'static> {
    SchedulerRequest::decode(active_sequence_count)
}

/// `SparkTestInitializePrefillRequest`.
fn prefill_request<'a>(
    active_sequence_count: u32,
    prompt_token_count: u32,
    sequence_id: u64,
    prompt_token_ids: &'a [u32],
) -> SchedulerRequest<'a> {
    SchedulerRequest::prefill(
        active_sequence_count,
        prompt_token_count,
        sequence_id,
        prompt_token_ids,
    )
}

/// `SparkTestGlm52SchedulerAdmitsCurrentSparkRingDecode`.
#[test]
fn admits_current_spark_ring_decode() {
    let cache = make_prefix_cache(128, 512);
    let configuration = make_config(QuantizationMode::Nvfp4_4Bit, cache);
    let mut scheduler = Scheduler::new(configuration).expect("configuration is valid");
    assert_eq!(scheduler.configuration_flags(), CONFIGURATION_DEFAULT_FLAGS);
    assert_eq!(scheduler.max_prefill_tokens_per_step(), GLM52_DEFAULT_MAX_PREFILL_TOKENS_PER_STEP);
    assert_eq!(scheduler.prefix_cache_block_tokens(), PREFILL_BLOCK_TOKENS);

    let request = decode_request(64);
    let decision = scheduler.admit(&request).expect("admission returns");
    assert!(decision.accepted);
    assert_eq!(decision.batch_bucket, BUCKET_B64);
    assert_eq!(decision.quantization_mode, QuantizationMode::Nvfp4_4Bit);
    assert_eq!(decision.stage_count, CURRENT_SPARK_COUNT);
    assert_ne!(decision.estimated_critical_path_ns, 0);
    assert_eq!(decision.decision_flags, DECISION_FLAG_DECODE_STEP);
    assert_eq!(decision.total_scheduled_token_count, 64);
    assert_eq!(decision.graph_sequence_padding_count, 0);
    assert_eq!(scheduler.scheduled_decode_token_count, 64);
    assert_eq!(decision.stage_plan.stages[0].first_layer_index, 0);
    assert_eq!(decision.stage_plan.stages[0].layer_count, 6);
    for stage_index in 0..decision.stage_count as usize {
        assert_eq!(decision.dispatch_stages[stage_index].spark_index, stage_index as u32);
        assert_eq!(
            decision.dispatch_stages[stage_index].dispatch_flags,
            DISPATCH_STAGE_FLAG_DECODE
        );
        assert_ne!(decision.dispatch_stages[stage_index].estimated_service_time_ns, 0);
        assert_eq!(scheduler.spark_inflight_counts()[stage_index], 1);
    }

    let decision = scheduler.admit(&request).expect("admission returns");
    assert!(!decision.accepted);
    assert_eq!(decision.rejected_status, Some(SchedulerError::Busy));
    assert_eq!(scheduler.rejected_count, 1);

    // Completing a rejected decision is invalid.
    assert_eq!(scheduler.complete(&decision), Err(SchedulerError::InvalidArgument));

    let request = decode_request(64);
    let decision = scheduler.admit(&request).expect("admission returns");
    assert!(!decision.accepted);

    // Re-initialize with queue depth 2 (fresh cache; see module docs).
    let cache = make_prefix_cache(128, 512);
    let mut configuration = make_config(QuantizationMode::Nvfp4_4Bit, cache);
    configuration.queue_depth_per_spark = 2;
    let mut scheduler = Scheduler::new(configuration).expect("configuration is valid");
    let request = decode_request(64);
    let decision = scheduler.admit(&request).expect("admission returns");
    assert!(decision.accepted);
    scheduler.complete(&decision).expect("complete succeeds");
    for stage_index in 0..decision.stage_count as usize {
        assert_eq!(scheduler.spark_inflight_counts()[stage_index], 0);
    }
}

/// `SparkTestGlm52SchedulerSupportsFp8AndPrefill`.
#[test]
fn supports_fp8_and_prefill() {
    let tokens = fill_token_ids(32, 3000);
    let cache = make_prefix_cache(128, 512);
    let mut configuration = make_config(QuantizationMode::Fp8E4m3_8Bit, cache);
    configuration.queue_depth_per_spark = 2;
    let mut scheduler = Scheduler::new(configuration).expect("configuration is valid");

    let request = decode_request(16);
    let decode_decision = scheduler.admit(&request).expect("admission returns");
    assert!(decode_decision.accepted);
    assert_eq!(decode_decision.batch_bucket, BUCKET_B16);
    assert_eq!(decode_decision.graph_sequence_padding_count, 0);
    assert_eq!(decode_decision.quantization_mode, QuantizationMode::Fp8E4m3_8Bit);
    scheduler.complete(&decode_decision).expect("complete succeeds");

    let request = prefill_request(33, 32, 11, &tokens);
    let prefill_decision = scheduler.admit(&request).expect("admission returns");
    assert!(prefill_decision.accepted);
    assert_eq!(prefill_decision.batch_bucket, BUCKET_B64);
    assert_eq!(prefill_decision.graph_sequence_padding_count, 31);
    assert_ne!(prefill_decision.decision_flags & DECISION_FLAG_CUDAGRAPH_PADDING, 0);
    assert_ne!(prefill_decision.decision_flags & DECISION_FLAG_PREFILL_FINAL_CHUNK, 0);
    assert_ne!(prefill_decision.decision_flags & DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT, 0);
    assert_eq!(prefill_decision.scheduled_prompt_token_count, 32);
    assert_eq!(prefill_decision.prefill_block_count, 2);
    assert_eq!(prefill_decision.cache_commit_token_count_after_step, 32);
    assert_eq!(prefill_decision.total_scheduled_token_count, 1056);
    assert!(
        prefill_decision.estimated_critical_path_ns >= decode_decision.estimated_critical_path_ns
    );

    scheduler.complete(&prefill_decision).expect("complete succeeds");
    assert_eq!(scheduler.prefix_cache().expect("cache present").inserted_block_count, 2);
    scheduler.release_sequence(11).expect("release succeeds");
}

/// `SparkTestGlm52SchedulerUsesVllmStyleChunkedPrefill`.
#[test]
fn uses_vllm_style_chunked_prefill() {
    let tokens = fill_token_ids(1024, 4000);
    let cache = make_prefix_cache(128, 512);
    let mut configuration = make_config(QuantizationMode::Nvfp4_4Bit, cache);
    configuration.max_prefill_tokens_per_step = 128;
    let mut scheduler = Scheduler::new(configuration).expect("configuration is valid");

    let mut request = prefill_request(64, 1024, 21, &tokens);
    request.max_scheduled_prompt_token_count = 256;
    let decision = scheduler.admit(&request).expect("admission returns");
    assert!(decision.accepted);
    assert_eq!(decision.scheduled_prompt_token_offset, 0);
    assert_eq!(decision.scheduled_prompt_token_count, 128);
    assert_eq!(decision.remaining_prompt_token_count_after_step, 896);
    assert_eq!(decision.prefill_block_count, 8);
    assert_eq!(decision.total_scheduled_token_count, 8192);
    assert_ne!(decision.decision_flags & DECISION_FLAG_PREFILL_CHUNK, 0);
    assert_eq!(decision.decision_flags & DECISION_FLAG_PREFILL_FINAL_CHUNK, 0);
    assert_ne!(decision.dispatch_stages[0].dispatch_flags & DISPATCH_STAGE_FLAG_PREFILL_CHUNK, 0);
    assert_eq!(scheduler.chunked_prefill_count, 1);
    assert_eq!(scheduler.scheduled_prefill_token_count, 8192);
    let chunk_critical_path_ns = decision.estimated_critical_path_ns;

    scheduler.complete(&decision).expect("complete succeeds");
    assert_eq!(scheduler.prefix_cache().expect("cache present").inserted_block_count, 8);

    let request = prefill_request(64, 1024, 21, &tokens);
    let decision = scheduler.admit(&request).expect("admission returns");
    assert!(decision.accepted);
    assert_eq!(decision.cached_prefix_token_count, 128);
    assert_eq!(decision.scheduled_prompt_token_offset, 128);
    assert_eq!(decision.scheduled_prompt_token_count, 128);
    assert_eq!(decision.remaining_prompt_token_count_after_step, 768);
    assert_ne!(decision.decision_flags & DECISION_FLAG_PREFILL_CHUNK, 0);
    assert_eq!(decision.decision_flags & DECISION_FLAG_PREFILL_FINAL_CHUNK, 0);
    assert_eq!(decision.estimated_critical_path_ns, chunk_critical_path_ns);
    scheduler.complete(&decision).expect("complete succeeds");
    scheduler.release_sequence(21).expect("release succeeds");
}

/// `SparkTestGlm52SchedulerUsesIntegratedPrefixCacheAdmission`.
#[test]
fn uses_integrated_prefix_cache_admission() {
    let tokens = fill_token_ids(1024, 5000);
    let short_tokens = fill_token_ids(100, 7000);
    let mut cache = make_prefix_cache(128, 512);
    cache.commit_prompt(77, &tokens[..768]).expect("commit succeeds");
    cache.commit_prompt(78, &short_tokens).expect("commit succeeds");
    let mut configuration = make_config(QuantizationMode::Nvfp4_4Bit, cache);
    configuration.max_prefill_tokens_per_step = 256;
    let mut scheduler = Scheduler::new(configuration).expect("configuration is valid");

    let request = prefill_request(32, 1024, 88, &tokens);
    let decision = scheduler.admit(&request).expect("admission returns");
    assert!(decision.accepted);
    assert_eq!(decision.cached_prefix_token_count, 768);
    assert_eq!(decision.prefix_cache_block_count, 48);
    assert_eq!(decision.scheduled_prompt_token_offset, 768);
    assert_eq!(decision.scheduled_prompt_token_count, 256);
    assert_eq!(decision.remaining_prompt_token_count_after_step, 0);
    assert_eq!(decision.prefill_block_count, 16);
    assert_ne!(decision.decision_flags & DECISION_FLAG_PREFIX_CACHE_USED, 0);
    assert_ne!(decision.decision_flags & DECISION_FLAG_PREFILL_FINAL_CHUNK, 0);
    assert_eq!(scheduler.prefix_cache_hit_token_count, 24576);
    assert_eq!(scheduler.scheduled_prefill_token_count, 8192);
    scheduler.complete(&decision).expect("complete succeeds");
    scheduler.release_sequence(88).expect("release succeeds");

    let request = prefill_request(1, 100, 89, &short_tokens);
    let decision = scheduler.admit(&request).expect("admission returns");
    assert!(decision.accepted);
    assert_eq!(decision.cached_prefix_token_count, 96);
    assert_eq!(decision.prefix_cache_block_count, 6);
    assert_eq!(decision.scheduled_prompt_token_offset, 96);
    assert_eq!(decision.scheduled_prompt_token_count, 4);
    assert_eq!(decision.remaining_prompt_token_count_after_step, 0);
    assert_eq!(decision.prefill_block_count, 1);
    scheduler.complete(&decision).expect("complete succeeds");
    scheduler.release_sequence(89).expect("release succeeds");
}

/// `SparkTestGlm52SchedulerDisablesCrossSequencePrefixReuse`.
#[test]
fn disables_cross_sequence_prefix_reuse() {
    let tokens = fill_token_ids(64, 7500);
    let cache = make_prefix_cache(128, 512);
    let mut configuration = make_config(QuantizationMode::Fp8E4m3_8Bit, cache);
    configuration.max_prefill_tokens_per_step = 32;
    configuration.configuration_flags &= !CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE;
    let mut scheduler = Scheduler::new(configuration).expect("configuration is valid");

    let request = prefill_request(1, 64, 601, &tokens);
    let decision = scheduler.admit(&request).expect("admission returns");
    assert_eq!(decision.scheduled_prompt_token_offset, 0);
    let first_blocks =
        [decision.kv_physical_block_indices[0], decision.kv_physical_block_indices[1]];
    scheduler.complete(&decision).expect("complete succeeds");

    let mut request = prefill_request(1, 64, 601, &tokens);
    request.computed_prompt_token_count = 32;
    let decision = scheduler.admit(&request).expect("admission returns");
    assert_eq!(decision.cached_prefix_token_count, 0);
    assert_eq!(decision.scheduled_prompt_token_offset, 32);
    assert_eq!(decision.kv_physical_block_indices[0], first_blocks[0]);
    assert_eq!(decision.kv_physical_block_indices[1], first_blocks[1]);
    scheduler.complete(&decision).expect("complete succeeds");

    let request = prefill_request(1, 64, 602, &tokens);
    let decision = scheduler.admit(&request).expect("admission returns");
    assert_eq!(decision.cached_prefix_token_count, 0);
    assert_eq!(decision.scheduled_prompt_token_offset, 0);
    assert_ne!(decision.kv_physical_block_indices[0], first_blocks[0]);
    scheduler.cancel(&decision).expect("cancel succeeds");
    scheduler.release_sequence(601).expect("release succeeds");
}

/// `SparkTestGlm52SchedulerInterleavesPrefillAndDecode`.
#[test]
fn interleaves_prefill_and_decode() {
    let tokens = fill_token_ids(1024, 9000);
    let cache = make_prefix_cache(128, 512);
    let mut configuration = make_config(QuantizationMode::Nvfp4_4Bit, cache);
    configuration.queue_depth_per_spark = 2;
    configuration.max_prefill_tokens_per_step = 256;
    let mut scheduler = Scheduler::new(configuration).expect("configuration is valid");

    let request = prefill_request(64, 1024, 31, &tokens);
    let prefill_decision = scheduler.admit(&request).expect("admission returns");
    assert!(prefill_decision.accepted);
    assert_ne!(prefill_decision.decision_flags & DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT, 0);
    assert_eq!(scheduler.interleaved_prefill_admission_count, 1);

    let request = prefill_request(64, 1024, 32, &tokens);
    let rejected_prefill_decision = scheduler.admit(&request).expect("admission returns");
    assert!(!rejected_prefill_decision.accepted);
    assert_eq!(rejected_prefill_decision.rejected_status, Some(SchedulerError::Busy));

    let request = decode_request(64);
    let decode_decision = scheduler.admit(&request).expect("admission returns");
    assert!(decode_decision.accepted);
    assert_ne!(decode_decision.decision_flags & DECISION_FLAG_DECODE_BYPASS_PREFILL, 0);
    assert_ne!(
        decode_decision.dispatch_stages[0].dispatch_flags
            & DISPATCH_STAGE_FLAG_DECODE_BYPASS_PREFILL,
        0
    );
    assert_eq!(scheduler.decode_bypass_admission_count, 1);

    scheduler.complete(&decode_decision).expect("complete succeeds");
    scheduler.complete(&prefill_decision).expect("complete succeeds");
    scheduler.release_sequence(31).expect("release succeeds");
}

/// `SparkTestGlm52SchedulerFillsCurrentSparkPipeline`.
#[test]
fn fills_current_spark_pipeline() {
    let tokens = fill_token_ids(16, 9500);
    let cache = make_prefix_cache(128, 512);
    let mut configuration = make_config(QuantizationMode::Fp8E4m3_8Bit, cache);
    configuration.queue_depth_per_spark = CURRENT_SPARK_COUNT + 1;
    let mut scheduler = Scheduler::new(configuration).expect("configuration is valid");

    for _cohort_index in 0..CURRENT_SPARK_COUNT - 1 {
        let request = decode_request(64);
        let decision = scheduler.admit(&request).expect("admission returns");
        assert!(decision.accepted);
    }

    let request = prefill_request(1, 16, 51, &tokens);
    let prefill_decision = scheduler.admit(&request).expect("admission returns");
    assert!(prefill_decision.accepted);
    scheduler.complete(&prefill_decision).expect("complete succeeds");

    let request = decode_request(64);
    let decision = scheduler.admit(&request).expect("admission returns");
    assert!(decision.accepted);
    assert_eq!(scheduler.spark_inflight_counts()[0], CURRENT_SPARK_COUNT);

    let request = prefill_request(1, 16, 52, &tokens);
    let decision = scheduler.admit(&request).expect("admission returns");
    assert!(!decision.accepted);
    assert_eq!(decision.rejected_status, Some(SchedulerError::Busy));
    scheduler.release_sequence(51).expect("release succeeds");
}

/// `SparkTestGlm52SchedulerPacksDecodeRequestsIntoSingleGraphDecision`.
#[test]
fn packs_decode_requests_into_single_graph_decision() {
    let cache = make_prefix_cache(128, 512);
    let mut configuration = make_config(QuantizationMode::Nvfp4_4Bit, cache);
    configuration.queue_depth_per_spark = 1;
    let mut scheduler = Scheduler::new(configuration).expect("configuration is valid");

    let requests: Vec<_> = (0..8).map(|_| decode_request(1)).collect();
    let batch_decision = scheduler.admit_decode_batch(&requests).expect("admission returns");
    assert!(batch_decision.accepted);
    assert_eq!(batch_decision.source_request_count, 8);
    assert_eq!(batch_decision.packed_request_count, 8);
    assert_eq!(batch_decision.active_sequence_count, 8);
    assert_eq!(batch_decision.batch_bucket, BUCKET_B16);
    assert_eq!(batch_decision.graph_sequence_capacity, BUCKET_B16);
    assert_eq!(batch_decision.graph_sequence_padding_count, 8);
    assert_eq!(batch_decision.total_scheduled_token_count, 8);
    assert_ne!(batch_decision.decision_flags & DECISION_FLAG_ADAPTIVE_DECODE_PACK, 0);
    assert_ne!(batch_decision.stage_decision.decision_flags & DECISION_FLAG_DECODE_STEP, 0);
    assert_ne!(
        batch_decision.stage_decision.decision_flags & DECISION_FLAG_ADAPTIVE_DECODE_PACK,
        0
    );
    assert_eq!(scheduler.admitted_count, 1);
    assert_eq!(scheduler.scheduled_decode_token_count, 8);
    assert_eq!(scheduler.adaptive_decode_pack_admission_count, 1);
    assert_eq!(scheduler.adaptive_decode_pack_request_count, 8);
    assert_eq!(scheduler.adaptive_decode_pack_padding_token_count, 8);

    for request_index in 0..8usize {
        assert_eq!(
            batch_decision.packed_requests[request_index].request_index,
            request_index as u32
        );
        assert_eq!(
            batch_decision.packed_requests[request_index].active_sequence_offset,
            request_index as u32
        );
        assert_eq!(batch_decision.packed_requests[request_index].active_sequence_count, 1);
        assert_eq!(batch_decision.packed_requests[request_index].scheduled_token_count, 1);
    }
    for stage_index in 0..batch_decision.stage_decision.stage_count as usize {
        assert_ne!(
            batch_decision.stage_decision.dispatch_stages[stage_index].dispatch_flags
                & DISPATCH_STAGE_FLAG_ADAPTIVE_DECODE_PACK,
            0
        );
        assert_eq!(scheduler.spark_inflight_counts()[stage_index], 1);
    }

    scheduler.complete_decode_batch(&batch_decision).expect("complete succeeds");
    for stage_index in 0..batch_decision.stage_decision.stage_count as usize {
        assert_eq!(scheduler.spark_inflight_counts()[stage_index], 0);
    }
    assert_eq!(scheduler.completed_count, 1);
}

/// `SparkTestGlm52SchedulerDecodeBatchFillsMaxBucketFromOversubscribedQueue`.
#[test]
fn decode_batch_fills_max_bucket_from_oversubscribed_queue() {
    let cache = make_prefix_cache(128, 512);
    let configuration = make_config(QuantizationMode::Fp8E4m3_8Bit, cache);
    let mut scheduler = Scheduler::new(configuration).expect("configuration is valid");

    let requests: Vec<_> = (0..1040).map(|_| decode_request(1)).collect();
    let batch_decision = scheduler.admit_decode_batch(&requests).expect("admission returns");
    assert!(batch_decision.accepted);
    assert_eq!(batch_decision.source_request_count, 1040);
    assert_eq!(batch_decision.packed_request_count, MAX_BATCH_BUCKET);
    assert_eq!(batch_decision.active_sequence_count, MAX_BATCH_BUCKET);
    assert_eq!(batch_decision.batch_bucket, MAX_BATCH_BUCKET);
    assert_eq!(batch_decision.graph_sequence_padding_count, 0);
    assert_eq!(
        batch_decision.packed_requests[(MAX_BATCH_BUCKET - 1) as usize].request_index,
        MAX_BATCH_BUCKET - 1
    );
    assert_eq!(
        batch_decision.packed_requests[(MAX_BATCH_BUCKET - 1) as usize].active_sequence_offset,
        MAX_BATCH_BUCKET - 1
    );
    assert_eq!(batch_decision.stage_decision.quantization_mode, QuantizationMode::Fp8E4m3_8Bit);
    scheduler.complete_decode_batch(&batch_decision).expect("complete succeeds");
}

/// `SparkTestGlm52SchedulerUsesMeasuredDecodeBucketForMidSizedBatch`.
#[test]
fn uses_measured_decode_bucket_for_mid_sized_batch() {
    let cache = make_prefix_cache(128, 512);
    let mut configuration = make_config(QuantizationMode::Nvfp4_4Bit, cache);
    configuration.queue_depth_per_spark = 2;
    let mut scheduler = Scheduler::new(configuration).expect("configuration is valid");

    let measured_requests: Vec<_> = (0..17).map(|_| decode_request(1)).collect();
    let measured_decision =
        scheduler.admit_decode_batch(&measured_requests).expect("admission returns");
    assert!(measured_decision.accepted);
    assert_eq!(measured_decision.active_sequence_count, 17);
    assert_eq!(measured_decision.batch_bucket, BUCKET_B64);
    assert_eq!(measured_decision.graph_sequence_capacity, BUCKET_B64);
    assert_eq!(measured_decision.graph_sequence_padding_count, 47);
    assert_ne!(measured_decision.decision_flags & DECISION_FLAG_MEASURED_DECODE_BUCKET, 0);
    assert_ne!(
        measured_decision.stage_decision.decision_flags & DECISION_FLAG_MEASURED_DECODE_BUCKET,
        0
    );
    assert_ne!(
        measured_decision.stage_decision.dispatch_stages[0].dispatch_flags
            & DISPATCH_STAGE_FLAG_MEASURED_DECODE_BUCKET,
        0
    );
    assert_eq!(scheduler.measured_decode_bucket_selection_count, 1);
    assert_eq!(scheduler.measured_decode_bucket_padding_token_count, 47);
    let measured_critical_path_ns = measured_decision.estimated_critical_path_ns;
    scheduler.complete_decode_batch(&measured_decision).expect("complete succeeds");

    // Legacy minimal-bucket selection (fresh cache; see module docs).
    let cache = make_prefix_cache(128, 512);
    let mut configuration = make_config(QuantizationMode::Nvfp4_4Bit, cache);
    configuration.queue_depth_per_spark = 2;
    configuration.configuration_flags =
        CONFIGURATION_DEFAULT_FLAGS & !CONFIGURATION_FLAG_MEASURED_DECODE_BUCKET_SELECTION;
    let mut scheduler = Scheduler::new(configuration).expect("configuration is valid");
    let legacy_requests: Vec<_> = (0..17).map(|_| decode_request(1)).collect();
    let legacy_decision =
        scheduler.admit_decode_batch(&legacy_requests).expect("admission returns");
    assert!(legacy_decision.accepted);
    assert_eq!(legacy_decision.active_sequence_count, 17);
    assert_eq!(legacy_decision.batch_bucket, BUCKET_B32);
    assert_eq!(legacy_decision.graph_sequence_padding_count, 15);
    assert_eq!(legacy_decision.decision_flags & DECISION_FLAG_MEASURED_DECODE_BUCKET, 0);
    assert!(measured_critical_path_ns < legacy_decision.estimated_critical_path_ns);
    scheduler.complete_decode_batch(&legacy_decision).expect("complete succeeds");
}

/// `SparkTestGlm52SchedulerRejectsPrefillInDecodeBatch`.
#[test]
fn rejects_prefill_in_decode_batch() {
    let tokens = fill_token_ids(16, 12000);
    let cache = make_prefix_cache(128, 512);
    let configuration = make_config(QuantizationMode::Nvfp4_4Bit, cache);
    let mut scheduler = Scheduler::new(configuration).expect("configuration is valid");

    let requests = [decode_request(1), prefill_request(1, 16, 99, &tokens)];
    // The C also fills a rejected batch decision (accepted=0,
    // rejected_status=INVALID_ARGUMENT); only the error is observable here.
    assert!(matches!(
        scheduler.admit_decode_batch(&requests),
        Err(SchedulerError::InvalidArgument)
    ));
    assert_eq!(scheduler.admitted_count, 0);
}

/// `SparkTestGlm52SchedulerExposesKvBlockTableAndCancelsReservation`.
#[test]
fn exposes_kv_block_table_and_cancels_reservation() {
    let tokens = fill_token_ids(48, 13000);
    let cache = make_prefix_cache(128, 512);
    let mut configuration = make_config(QuantizationMode::Nvfp4_4Bit, cache);
    configuration.max_prefill_tokens_per_step = 48;
    let mut scheduler = Scheduler::new(configuration).expect("configuration is valid");

    let request = prefill_request(16, 48, 501, &tokens);
    let decision = scheduler.admit(&request).expect("admission returns");
    assert!(decision.accepted);
    assert_eq!(decision.kv_block_token_count, PREFILL_BLOCK_TOKENS);
    assert_eq!(decision.kv_physical_block_count, 3);
    assert_eq!(decision.kv_pending_physical_block_count, 3);
    assert_eq!(decision.kv_cached_physical_block_count, 0);
    assert_ne!(decision.prefix_cache_reservation_epoch, 0);

    let physical_blocks = scheduler.build_kv_block_table(&decision).expect("table builds");
    assert_eq!(physical_blocks.len(), 3);
    assert_eq!(physical_blocks[..], decision.kv_physical_block_indices[..3]);

    let lookup = scheduler
        .prefix_cache_mut()
        .expect("cache present")
        .probe_prompt(502, &tokens)
        .expect("probe succeeds");
    assert_eq!(lookup.matched_token_count, 0);

    scheduler.cancel(&decision).expect("cancel succeeds");
    assert_eq!(scheduler.kv_block_cancel_count, 1);
    let lookup = scheduler
        .prefix_cache_mut()
        .expect("cache present")
        .probe_prompt(503, &tokens)
        .expect("probe succeeds");
    assert_eq!(lookup.matched_token_count, 0);

    let request = prefill_request(16, 48, 504, &tokens);
    let decision = scheduler.admit(&request).expect("admission returns");
    assert!(decision.accepted);
    scheduler.complete(&decision).expect("complete succeeds");
    let lookup = scheduler
        .prefix_cache_mut()
        .expect("cache present")
        .probe_prompt(505, &tokens)
        .expect("probe succeeds");
    assert_eq!(lookup.matched_token_count, 32);
    scheduler.release_sequence(504).expect("release succeeds");
}

/// `SparkTestGlm52SchedulerBuildsBatchedPrefillKvTables`.
#[test]
fn builds_batched_prefill_kv_tables() {
    let shared_prefix = fill_token_ids(32, 15000);
    let cache = make_prefix_cache(128, 512);
    let mut configuration = make_config(QuantizationMode::Nvfp4_4Bit, cache);
    configuration.max_prefill_tokens_per_step = 48;
    let mut scheduler = Scheduler::new(configuration).expect("configuration is valid");

    let warm_request = prefill_request(1, 32, 601, &shared_prefix);
    let warm_decision = scheduler.admit(&warm_request).expect("admission returns");
    assert!(warm_decision.accepted);
    scheduler.complete(&warm_decision).expect("complete succeeds");
    let shared_physical_blocks =
        scheduler.build_kv_block_table(&warm_decision).expect("table builds");
    assert_eq!(shared_physical_blocks.len(), 2);

    let prompts: Vec<Vec<u32>> = (0..3u32)
        .map(|request_index| {
            build_shared_prefix_prompt(&shared_prefix, 16000 + request_index * 100, 16)
        })
        .collect();
    let batch_requests: Vec<_> = (0..3u64)
        .map(|request_index| {
            prefill_request(1, 48, 701 + request_index, &prompts[request_index as usize])
        })
        .collect();
    let batch_decision = scheduler.admit_prefill_batch(&batch_requests).expect("admission returns");
    assert!(batch_decision.accepted);
    assert_eq!(batch_decision.active_sequence_count, 3);
    assert_eq!(batch_decision.maximum_scheduled_prompt_token_count, 16);
    assert_eq!(batch_decision.total_scheduled_token_count, 3 * 16);
    for request_index in 0..3usize {
        assert_eq!(batch_decision.lanes[request_index].cached_prefix_token_count, 32);
        assert_eq!(batch_decision.lanes[request_index].scheduled_prompt_token_count, 16);
        assert_eq!(batch_decision.lanes[request_index].kv_block_table_token_count, 48);
    }

    let lane_tables =
        scheduler.build_prefill_batch_kv_block_tables(&batch_decision, 4).expect("tables build");
    for lane_table in &lane_tables {
        assert_eq!(lane_table.len(), 3);
        assert_eq!(lane_table[0], shared_physical_blocks[0]);
        assert_eq!(lane_table[1], shared_physical_blocks[1]);
        assert_ne!(lane_table[2], shared_physical_blocks[0]);
        assert_ne!(lane_table[2], shared_physical_blocks[1]);
    }
    scheduler.complete_prefill_batch(&batch_decision).expect("complete succeeds");
    for request_index in 0..3u64 {
        scheduler.release_sequence(701 + request_index).expect("release succeeds");
    }
    scheduler.release_sequence(601).expect("release succeeds");
}

/// `SparkTestGlm52SchedulerRejectsInvalidInputs`. The C's first case (unknown
/// quantization mode 99) is unrepresentable with the [`QuantizationMode`]
/// enum and is not ported.
#[test]
fn rejects_invalid_inputs() {
    let tokens = fill_token_ids(16, 11000);

    let cache = make_prefix_cache(32, 64);
    let mut configuration = make_config(QuantizationMode::Auto, cache);
    configuration.configuration_flags = 0xffff_ffff;
    assert_eq!(Scheduler::new(configuration).err(), Some(SchedulerError::InvalidArgument));

    // PREFIX_CACHE flag set but no cache supplied.
    let mut configuration = make_config(QuantizationMode::Auto, make_prefix_cache(32, 64));
    configuration.prefix_cache = None;
    assert_eq!(Scheduler::new(configuration).err(), Some(SchedulerError::InvalidArgument));

    let cache = make_prefix_cache(32, 64);
    let configuration = make_config(QuantizationMode::Auto, cache);
    let mut scheduler = Scheduler::new(configuration).expect("configuration is valid");
    assert_eq!(scheduler.quantization_mode(), QuantizationMode::Nvfp4_4Bit);

    let request = decode_request(MAX_BATCH_BUCKET + 1);
    let decision = scheduler.admit(&request).expect("admission returns");
    assert!(!decision.accepted);
    assert_eq!(decision.rejected_status, Some(SchedulerError::CapacityExceeded));

    let mut request = prefill_request(1, 16, 41, &tokens);
    request.computed_prompt_token_count = 16;
    assert!(matches!(scheduler.admit(&request), Err(SchedulerError::InvalidArgument)));

    let mut request = decode_request(1);
    request.cached_prefix_token_count = 16;
    assert!(matches!(scheduler.admit(&request), Err(SchedulerError::InvalidArgument)));

    let mut request = prefill_request(1, 16, 42, &tokens);
    request.cached_prefix_token_count = 16;
    assert!(matches!(scheduler.admit(&request), Err(SchedulerError::InvalidArgument)));

    let request = prefill_request(1, 16, 0, &tokens);
    assert!(matches!(scheduler.admit(&request), Err(SchedulerError::InvalidArgument)));
}

/// `SparkTestGlm52SchedulerSelectsPipelineBatchWidth`. The C builds a
/// memset scheduler with only `spark_count` set; a fully initialized
/// scheduler with the same spark count behaves identically.
#[test]
fn selects_pipeline_batch_width() {
    let cache = make_prefix_cache(32, 64);
    let configuration = make_config(QuantizationMode::Nvfp4_4Bit, cache);
    let scheduler = Scheduler::new(configuration).expect("configuration is valid");
    assert_eq!(scheduler.spark_count(), CURRENT_SPARK_COUNT);
    assert_eq!(scheduler.select_pipeline_batch_width(0, 256), 0);
    assert_eq!(scheduler.select_pipeline_batch_width(1, 256), 1);
    assert_eq!(scheduler.select_pipeline_batch_width(4, 256), 1);
    assert_eq!(scheduler.select_pipeline_batch_width(13, 256), 1);
    assert_eq!(scheduler.select_pipeline_batch_width(14, 256), 2);
    assert_eq!(scheduler.select_pipeline_batch_width(92, 256), 8);
    assert_eq!(scheduler.select_pipeline_batch_width(184, 256), 15);
    assert_eq!(scheduler.select_pipeline_batch_width(256, 256), 20);
    assert_eq!(scheduler.select_pipeline_batch_width(13312, 256), 256);
}

/// `SparkTestGlm52SchedulerEstimatesExpandedDecodeWork`.
#[test]
fn estimates_expanded_decode_work() {
    let cache = make_prefix_cache(128, 512);
    let configuration = make_config(QuantizationMode::Fp8E4m3_8Bit, cache);
    let scheduler = Scheduler::new(configuration).expect("configuration is valid");

    let plain_b1_work_ns = scheduler.estimate_decode_work_ns(1, 1, 112).expect("estimate succeeds");
    let mtp_b1_work_ns = scheduler.estimate_decode_work_ns(1, 6, 112).expect("estimate succeeds");
    assert_eq!(plain_b1_work_ns, mtp_b1_work_ns);
    let plain_b16_work_ns =
        scheduler.estimate_decode_work_ns(16, 1, 112).expect("estimate succeeds");
    let mtp_b16_work_ns = scheduler.estimate_decode_work_ns(16, 6, 112).expect("estimate succeeds");
    assert!(mtp_b16_work_ns > plain_b16_work_ns);
    let plain_b1024_work_ns =
        scheduler.estimate_decode_work_ns(1024, 1, 7168).expect("estimate succeeds");
    let mtp_b1024_work_ns =
        scheduler.estimate_decode_work_ns(1024, 6, 7168).expect("estimate succeeds");
    assert!(mtp_b1024_work_ns > plain_b1024_work_ns);
    assert_eq!(scheduler.estimate_decode_work_ns(1, 0, 112), Err(SchedulerError::InvalidArgument));
}
