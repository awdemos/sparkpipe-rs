//! GPU-free port of `tests/test_glm52_long_context.c`, plus coverage for
//! the lane-batch entry point the C test does not exercise.
//!
//! The C test baked the GLM52 defaults (1M context, 2048 selected tokens)
//! in through the header macros; here they are passed to
//! `LongContextPolicy::new` explicitly, as the model contract would.

use spark_sched::long_context::*;

/// GLM52's `SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS`, supplied by config.
const DEFAULT_MAX_CONTEXT_TOKENS: u32 = 1_048_576;
/// GLM52's `SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT`, supplied by config.
const DEFAULT_SELECTED_TOKEN_CAPACITY: u32 = 2048;

fn default_policy() -> LongContextPolicy {
    LongContextPolicy::new(DEFAULT_MAX_CONTEXT_TOKENS, DEFAULT_SELECTED_TOKEN_CAPACITY)
}

#[test]
fn default_policy_is_bounded() {
    let policy = default_policy();
    assert_eq!(policy.validate(), Ok(()));
    assert_eq!(policy.max_context_tokens, DEFAULT_MAX_CONTEXT_TOKENS);
    assert_eq!(policy.selected_token_capacity, 2048);
    assert_eq!(policy.maximum_decode_scan_token_count, 2048);
    assert_ne!(policy.policy_flags & POLICY_FLAG_REQUIRE_BOUNDED_DECODE, 0);
    assert_eq!(policy.policy_mode, PolicyMode::BoundedWindow);
}

#[test]
fn builds_bounded_decode_selection_for_256k() {
    let policy = default_policy();
    let mut selected_token_indices = vec![0u32; DEFAULT_SELECTED_TOKEN_CAPACITY as usize];

    let decode_plan = policy
        .build_decode_selection(262_144, &mut selected_token_indices)
        .expect("bounded selection");

    assert_ne!(decode_plan.flags & DECODE_PLAN_FLAG_BOUNDED_SELECTION, 0);
    assert_ne!(decode_plan.flags & DECODE_PLAN_FLAG_LONG_CONTEXT, 0);
    assert_ne!(decode_plan.flags & DECODE_PLAN_FLAG_CONTEXT_TRUNCATED, 0);
    assert_eq!(decode_plan.context_token_count, 262_144);
    assert_eq!(decode_plan.selected_token_count, DEFAULT_SELECTED_TOKEN_CAPACITY);
    assert!(decode_plan.selected_block_count < decode_plan.kv_block_count_for_context);
    assert!(decode_plan.selected_block_count <= 2048);
    assert!(decode_plan.avoided_full_scan_token_reads > 250_000);
    assert_eq!(selected_token_indices[0], 0);

    let mut saw_last_context_token = false;
    for &token_index in &selected_token_indices[..decode_plan.selected_token_count as usize] {
        assert!(token_index < 262_144);
        if token_index == 262_143 {
            saw_last_context_token = true;
        }
    }
    assert!(saw_last_context_token);
}

#[test]
fn builds_chunked_prefill_plan_for_256k() {
    let policy = default_policy();
    let prefill_plan = policy.build_prefill_plan(262_144, 256).expect("prefill plan");

    assert_ne!(prefill_plan.flags & PREFILL_PLAN_FLAG_CHUNKED_PREFILL, 0);
    assert_ne!(prefill_plan.flags & PREFILL_PLAN_FLAG_LONG_CONTEXT, 0);
    assert_eq!(prefill_plan.prefill_chunk_count, 1024);
    assert_eq!(prefill_plan.kv_block_count_for_prompt, 16_384);
    assert_eq!(prefill_plan.total_prompt_token_visits, 262_144);
}

#[test]
fn full_scan_requires_explicit_unsafe_policy() {
    let mut policy = default_policy();
    let mut selected_token_indices = vec![0u32; DEFAULT_SELECTED_TOKEN_CAPACITY as usize];

    policy.policy_mode = PolicyMode::FullContextScan;
    policy.policy_flags &= !POLICY_FLAG_REQUIRE_BOUNDED_DECODE;
    assert_eq!(
        policy.build_decode_selection(262_144, &mut selected_token_indices),
        Err(LongContextError::InvalidArgument)
    );

    policy.policy_flags |= POLICY_FLAG_ALLOW_FULL_CONTEXT_SCAN;
    assert_eq!(
        policy.build_decode_selection(262_144, &mut selected_token_indices),
        Err(LongContextError::CapacityExceeded)
    );
}

#[test]
fn lane_batch_builds_one_plan_per_lane() {
    let policy = default_policy();
    let context_token_counts = [512u32, 262_144u32];
    let stride = DEFAULT_SELECTED_TOKEN_CAPACITY as usize;
    let capacity = DEFAULT_SELECTED_TOKEN_CAPACITY as usize;
    let mut selected_token_indices = vec![0u32; stride * context_token_counts.len()];

    let decode_plans = policy
        .build_decode_selection_for_lane_batch(
            &context_token_counts,
            &mut selected_token_indices,
            stride,
            capacity,
        )
        .expect("lane batch");

    assert_eq!(decode_plans.len(), 2);

    // Lane 0 fits the capacity: full selection (flagged bounded, per the
    // C BuildFullSelection in bounded-window mode), padded, not truncated.
    let short = &decode_plans[0];
    assert_eq!(short.selected_token_count, 512);
    assert_ne!(short.flags & DECODE_PLAN_FLAG_BOUNDED_SELECTION, 0);
    assert_eq!(short.flags & DECODE_PLAN_FLAG_FULL_CONTEXT_SCAN, 0);
    assert_ne!(short.flags & DECODE_PLAN_FLAG_SELECTION_PADDED, 0);
    assert_eq!(short.flags & DECODE_PLAN_FLAG_CONTEXT_TRUNCATED, 0);
    assert_eq!(selected_token_indices[0], 0);
    assert_eq!(selected_token_indices[511], 511);
    assert_eq!(selected_token_indices[512], INVALID_TOKEN_ID);

    // Lane 1 is the bounded 256k case, written at the lane-1 stride offset.
    let long = &decode_plans[1];
    assert_eq!(long.context_token_count, 262_144);
    assert_ne!(long.flags & DECODE_PLAN_FLAG_BOUNDED_SELECTION, 0);
    assert_ne!(long.flags & DECODE_PLAN_FLAG_CONTEXT_TRUNCATED, 0);
    assert_eq!(long.selected_token_count, DEFAULT_SELECTED_TOKEN_CAPACITY);
    let lane1 = &selected_token_indices[stride..stride + long.selected_token_count as usize];
    assert!(lane1.contains(&262_143));
    assert!(lane1.iter().all(|&index| index < 262_144));

    // A short index buffer is rejected instead of overrun (C would index
    // out of bounds).
    let mut too_small = vec![0u32; stride];
    assert_eq!(
        policy.build_decode_selection_for_lane_batch(
            &context_token_counts,
            &mut too_small,
            stride,
            capacity,
        ),
        Err(LongContextError::InvalidArgument)
    );
}
