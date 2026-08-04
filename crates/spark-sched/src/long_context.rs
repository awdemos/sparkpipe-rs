//! Long-context scheduling helpers — port of `scheduler/long_context.c`
//! (API declared in `include/sparkpipe/spark_long_context.h`).
//!
//! The policy chooses, per decode step, which KV tokens a bounded-capacity
//! attention kernel reads: a sink prefix, an evenly strided sample of the
//! middle, and the recent tail — plus a chunked-prefill plan for long
//! prompts. All arithmetic and flag semantics mirror the C exactly.
//!
//! Rust-port deviations from the C surface (the behavior is otherwise
//! signature-faithful):
//!   - No GLM52 constants are baked in. The C
//!     `SparkLongContextInitializeDefaultPolicy` aliased
//!     `SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS` (1M) and
//!     `SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT` (2048) into the
//!     defaults; here [`LongContextPolicy::new`] takes
//!     `default_max_context_tokens` / `default_selected_token_capacity`
//!     from the model contract / runtime configuration.
//!   - ABI plumbing (`abi_version`, `descriptor_bytes`, reserved words) is
//!     dropped; the C validation checks against those fields have no Rust
//!     counterpart. `policy_mode` is a real enum, so the C "unknown mode"
//!     rejection is unrepresentable rather than validated.
//!   - `0`-means-default normalization is preserved field-for-field
//!     ([`LongContextPolicy::normalized`], port of
//!     `SparkLongContextNormalizePolicyCopy`), so callers may set only the
//!     fields they care about, exactly as in C — except `max_context_tokens`
//!     and `selected_token_capacity`, whose C zero-defaults were the GLM52
//!     constants; a 0 there is rejected by validation instead.
//!   - Selection buffers are caller-provided mutable slices whose length is
//!     the capacity; the plan is returned by value. Where the C filled the
//!     out-plan and then returned `SPARK_STATUS_CAPACITY_EXCEEDED` (bounded
//!     selection larger than the scan budget), the Rust port returns `Err`
//!     without yielding the partial plan.
//!   - The lane-batch entry point returns `Vec<DecodePlan>` instead of
//!     writing a caller-provided plan array, and (unlike the C, which would
//!     index out of bounds) rejects an undersized index buffer with
//!     [`LongContextError::InvalidArgument`].

/// Sentinel written into unused selection slots
/// (SPARK_LONG_CONTEXT_INVALID_TOKEN_ID).
pub const INVALID_TOKEN_ID: u32 = 0xffff_ffff;

/// Hard ceiling on the selected-token capacity
/// (SPARK_LONG_CONTEXT_MAX_SELECTED_TOKEN_CAPACITY).
pub const MAX_SELECTED_TOKEN_CAPACITY: u32 = 4096;

/// Default KV block size in tokens (SPARK_LONG_CONTEXT_DEFAULT_BLOCK_TOKEN_COUNT).
pub const DEFAULT_BLOCK_TOKEN_COUNT: u32 = 16;
/// Default recent-tail tokens kept by bounded selection
/// (SPARK_LONG_CONTEXT_DEFAULT_RECENT_TOKEN_COUNT).
pub const DEFAULT_RECENT_TOKEN_COUNT: u32 = 1536;
/// Default sink-prefix tokens kept by bounded selection
/// (SPARK_LONG_CONTEXT_DEFAULT_SINK_TOKEN_COUNT).
pub const DEFAULT_SINK_TOKEN_COUNT: u32 = 128;
/// Default strided middle-sample tokens (SPARK_LONG_CONTEXT_DEFAULT_STRIDE_SAMPLE_TOKEN_COUNT).
pub const DEFAULT_STRIDE_SAMPLE_TOKEN_COUNT: u32 = 384;
/// Default context length at/above which plans are flagged long-context
/// (SPARK_LONG_CONTEXT_DEFAULT_LONG_CONTEXT_THRESHOLD).
pub const DEFAULT_LONG_CONTEXT_THRESHOLD: u32 = 8192;

/// Policy flag: bounded decode selection is required
/// (SPARK_LONG_CONTEXT_POLICY_FLAG_REQUIRE_BOUNDED_DECODE).
pub const POLICY_FLAG_REQUIRE_BOUNDED_DECODE: u32 = 0x0000_0001;
/// Policy flag: include the sink prefix (SPARK_LONG_CONTEXT_POLICY_FLAG_INCLUDE_SINK_TOKENS).
pub const POLICY_FLAG_INCLUDE_SINK_TOKENS: u32 = 0x0000_0002;
/// Policy flag: include strided middle samples
/// (SPARK_LONG_CONTEXT_POLICY_FLAG_INCLUDE_STRIDED_MIDDLE_TOKENS).
pub const POLICY_FLAG_INCLUDE_STRIDED_MIDDLE_TOKENS: u32 = 0x0000_0004;
/// Policy flag: include the recent tail (SPARK_LONG_CONTEXT_POLICY_FLAG_INCLUDE_RECENT_TOKENS).
pub const POLICY_FLAG_INCLUDE_RECENT_TOKENS: u32 = 0x0000_0008;
/// Policy flag: reject contexts beyond `max_context_tokens`
/// (SPARK_LONG_CONTEXT_POLICY_FLAG_FAIL_ON_CONTEXT_OVERFLOW).
pub const POLICY_FLAG_FAIL_ON_CONTEXT_OVERFLOW: u32 = 0x0000_0010;
/// Policy flag: permit full-context-scan mode
/// (SPARK_LONG_CONTEXT_POLICY_FLAG_ALLOW_FULL_CONTEXT_SCAN).
pub const POLICY_FLAG_ALLOW_FULL_CONTEXT_SCAN: u32 = 0x0000_0020;
/// Default policy flags (SPARK_LONG_CONTEXT_POLICY_DEFAULT_FLAGS).
pub const POLICY_DEFAULT_FLAGS: u32 = POLICY_FLAG_REQUIRE_BOUNDED_DECODE
    | POLICY_FLAG_INCLUDE_SINK_TOKENS
    | POLICY_FLAG_INCLUDE_STRIDED_MIDDLE_TOKENS
    | POLICY_FLAG_INCLUDE_RECENT_TOKENS
    | POLICY_FLAG_FAIL_ON_CONTEXT_OVERFLOW;
/// All recognized policy flags (SPARK_LONG_CONTEXT_POLICY_KNOWN_FLAGS).
pub const POLICY_KNOWN_FLAGS: u32 = POLICY_DEFAULT_FLAGS | POLICY_FLAG_ALLOW_FULL_CONTEXT_SCAN;

/// Decode-plan flag: selection is a bounded subset
/// (SPARK_LONG_CONTEXT_DECODE_PLAN_FLAG_BOUNDED_SELECTION).
pub const DECODE_PLAN_FLAG_BOUNDED_SELECTION: u32 = 0x0000_0001;
/// Decode-plan flag: selection is the full context
/// (SPARK_LONG_CONTEXT_DECODE_PLAN_FLAG_FULL_CONTEXT_SCAN).
pub const DECODE_PLAN_FLAG_FULL_CONTEXT_SCAN: u32 = 0x0000_0002;
/// Decode-plan flag: context meets the long-context threshold
/// (SPARK_LONG_CONTEXT_DECODE_PLAN_FLAG_LONG_CONTEXT).
pub const DECODE_PLAN_FLAG_LONG_CONTEXT: u32 = 0x0000_0004;
/// Decode-plan flag: context was truncated to fit the selection
/// (SPARK_LONG_CONTEXT_DECODE_PLAN_FLAG_CONTEXT_TRUNCATED).
pub const DECODE_PLAN_FLAG_CONTEXT_TRUNCATED: u32 = 0x0000_0008;
/// Decode-plan flag: selection buffer was padded with INVALID_TOKEN_ID
/// (SPARK_LONG_CONTEXT_DECODE_PLAN_FLAG_SELECTION_PADDED).
pub const DECODE_PLAN_FLAG_SELECTION_PADDED: u32 = 0x0000_0010;

/// Prefill-plan flag: prompt is split across multiple steps
/// (SPARK_LONG_CONTEXT_PREFILL_PLAN_FLAG_CHUNKED_PREFILL).
pub const PREFILL_PLAN_FLAG_CHUNKED_PREFILL: u32 = 0x0000_0001;
/// Prefill-plan flag: prompt meets the long-context threshold
/// (SPARK_LONG_CONTEXT_PREFILL_PLAN_FLAG_LONG_CONTEXT).
pub const PREFILL_PLAN_FLAG_LONG_CONTEXT: u32 = 0x0000_0002;

/// Errors mirroring the `SparkStatus` codes returned by the C entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LongContextError {
    /// SPARK_STATUS_INVALID_ARGUMENT.
    #[error("invalid argument")]
    InvalidArgument,
    /// SPARK_STATUS_CAPACITY_EXCEEDED.
    #[error("capacity exceeded")]
    CapacityExceeded,
}

/// Selection policy mode (SPARK_LONG_CONTEXT_POLICY_MODE_*).
///
/// The C default when `policy_mode == 0` is [`PolicyMode::BoundedWindow`];
/// invalid mode values are unrepresentable here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyMode {
    /// SPARK_LONG_CONTEXT_POLICY_MODE_BOUNDED_WINDOW.
    BoundedWindow,
    /// SPARK_LONG_CONTEXT_POLICY_MODE_FULL_CONTEXT_SCAN.
    FullContextScan,
}

/// Long-context selection policy (port of `SparkLongContextPolicy`).
///
/// A field value of `0` selects the default for that field at
/// normalization time, exactly as in C — except `max_context_tokens` and
/// `selected_token_capacity`, whose defaults are the values passed to
/// [`LongContextPolicy::new`]. `policy_flags == 0` selects
/// [`POLICY_DEFAULT_FLAGS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LongContextPolicy {
    pub policy_mode: PolicyMode,
    pub policy_flags: u32,
    pub max_context_tokens: u32,
    pub selected_token_capacity: u32,
    pub block_token_count: u32,
    pub recent_token_count: u32,
    pub sink_token_count: u32,
    pub stride_sample_token_count: u32,
    pub maximum_decode_scan_token_count: u32,
    pub prefill_chunk_token_count: u32,
    pub long_context_threshold_token_count: u32,
}

/// Per-step decode selection plan (port of `SparkLongContextDecodePlan`,
/// minus ABI/reserved fields).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecodePlan {
    pub flags: u32,
    pub context_token_count: u32,
    pub selected_token_count: u32,
    pub selected_token_capacity: u32,
    pub padded_token_count: u32,
    pub selected_block_count: u32,
    pub kv_block_token_count: u32,
    pub kv_block_count_for_context: u32,
    pub first_recent_token_index: u32,
    pub maximum_decode_scan_token_count: u32,
    pub sink_token_count: u32,
    pub stride_sample_token_count: u32,
    pub recent_token_count: u32,
    pub estimated_attention_token_reads: u64,
    pub avoided_full_scan_token_reads: u64,
}

/// Chunked-prefill plan (port of `SparkLongContextPrefillPlan`, minus
/// ABI/reserved fields).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PrefillPlan {
    pub flags: u32,
    pub prompt_token_count: u32,
    pub max_prefill_tokens_per_step: u32,
    pub prefill_chunk_count: u32,
    pub final_chunk_token_count: u32,
    pub kv_block_token_count: u32,
    pub kv_block_count_for_prompt: u32,
    pub total_prompt_token_visits: u64,
}

/// `SparkCeilDivU32`; `0` denominator yields `0`. Computed in `u64` so a
/// `u32::MAX` numerator cannot wrap the way the C addition does.
fn ceil_div_u32(numerator: u32, denominator: u32) -> u32 {
    if denominator == 0 {
        return 0;
    }
    (u64::from(numerator).div_ceil(u64::from(denominator))) as u32
}

impl LongContextPolicy {
    /// Port of `SparkLongContextInitializeDefaultPolicy`.
    ///
    /// The two model-derived defaults (maximum context length and DSA
    /// selected-token capacity) are parameters instead of baked-in GLM52
    /// constants; every other default matches the C header.
    pub fn new(default_max_context_tokens: u32, default_selected_token_capacity: u32) -> Self {
        Self {
            policy_mode: PolicyMode::BoundedWindow,
            policy_flags: POLICY_DEFAULT_FLAGS,
            max_context_tokens: default_max_context_tokens,
            selected_token_capacity: default_selected_token_capacity,
            block_token_count: DEFAULT_BLOCK_TOKEN_COUNT,
            recent_token_count: DEFAULT_RECENT_TOKEN_COUNT,
            sink_token_count: DEFAULT_SINK_TOKEN_COUNT,
            stride_sample_token_count: DEFAULT_STRIDE_SAMPLE_TOKEN_COUNT,
            maximum_decode_scan_token_count: default_selected_token_capacity,
            prefill_chunk_token_count: default_selected_token_capacity,
            long_context_threshold_token_count: DEFAULT_LONG_CONTEXT_THRESHOLD,
        }
    }

    /// Port of `SparkLongContextNormalizePolicyCopy`: apply the
    /// `0`-means-default rules field by field. `selected_token_capacity`
    /// is normalized before `maximum_decode_scan_token_count`, as in C,
    /// because the latter defaults to the former.
    ///
    /// Deviation: in C, `max_context_tokens == 0` and
    /// `selected_token_capacity == 0` normalize to the baked-in GLM52
    /// defaults. Those defaults are constructor parameters here, so a 0 in
    /// either field is left as-is and rejected by [`Self::validate`].
    fn normalized(&self) -> Self {
        let mut policy = *self;
        if policy.policy_flags == 0 {
            policy.policy_flags = POLICY_DEFAULT_FLAGS;
        }
        if policy.block_token_count == 0 {
            policy.block_token_count = DEFAULT_BLOCK_TOKEN_COUNT;
        }
        if policy.recent_token_count == 0 {
            policy.recent_token_count = DEFAULT_RECENT_TOKEN_COUNT;
        }
        if policy.sink_token_count == 0 {
            policy.sink_token_count = DEFAULT_SINK_TOKEN_COUNT;
        }
        if policy.stride_sample_token_count == 0 {
            policy.stride_sample_token_count = DEFAULT_STRIDE_SAMPLE_TOKEN_COUNT;
        }
        if policy.maximum_decode_scan_token_count == 0 {
            policy.maximum_decode_scan_token_count = policy.selected_token_capacity;
        }
        if policy.prefill_chunk_token_count == 0 {
            policy.prefill_chunk_token_count = policy.selected_token_capacity;
        }
        if policy.long_context_threshold_token_count == 0 {
            policy.long_context_threshold_token_count = DEFAULT_LONG_CONTEXT_THRESHOLD;
        }
        policy
    }

    /// Port of `SparkLongContextValidatePolicy`.
    pub fn validate(&self) -> Result<(), LongContextError> {
        let policy = self.normalized();
        if (policy.policy_flags & !POLICY_KNOWN_FLAGS) != 0
            || policy.max_context_tokens == 0
            || policy.selected_token_capacity == 0
            || policy.selected_token_capacity > MAX_SELECTED_TOKEN_CAPACITY
            || policy.block_token_count == 0
            || policy.maximum_decode_scan_token_count == 0
            || policy.maximum_decode_scan_token_count > policy.selected_token_capacity
            || policy.prefill_chunk_token_count == 0
        {
            return Err(LongContextError::InvalidArgument);
        }
        if (policy.policy_flags & POLICY_FLAG_REQUIRE_BOUNDED_DECODE) != 0
            && policy.policy_mode == PolicyMode::FullContextScan
            && (policy.policy_flags & POLICY_FLAG_ALLOW_FULL_CONTEXT_SCAN) == 0
        {
            return Err(LongContextError::InvalidArgument);
        }
        if (policy.policy_flags & POLICY_FLAG_INCLUDE_SINK_TOKENS) == 0
            && (policy.policy_flags & POLICY_FLAG_INCLUDE_STRIDED_MIDDLE_TOKENS) == 0
            && (policy.policy_flags & POLICY_FLAG_INCLUDE_RECENT_TOKENS) == 0
        {
            return Err(LongContextError::InvalidArgument);
        }
        Ok(())
    }

    /// Port of `SparkLongContextBuildPrefillPlan`.
    ///
    /// `max_prefill_tokens_per_step == 0` selects the policy's
    /// (normalized) prefill chunk size, as in C.
    pub fn build_prefill_plan(
        &self,
        prompt_token_count: u32,
        max_prefill_tokens_per_step: u32,
    ) -> Result<PrefillPlan, LongContextError> {
        if prompt_token_count == 0 {
            return Err(LongContextError::InvalidArgument);
        }
        self.validate()?;
        let policy = self.normalized();
        if prompt_token_count > policy.max_context_tokens
            && (policy.policy_flags & POLICY_FLAG_FAIL_ON_CONTEXT_OVERFLOW) != 0
        {
            return Err(LongContextError::CapacityExceeded);
        }
        let max_prefill_tokens_per_step = if max_prefill_tokens_per_step == 0 {
            policy.prefill_chunk_token_count
        } else {
            max_prefill_tokens_per_step
        };
        let prefill_chunk_count = ceil_div_u32(prompt_token_count, max_prefill_tokens_per_step);
        let mut final_chunk_token_count = prompt_token_count % max_prefill_tokens_per_step;
        if final_chunk_token_count == 0 {
            final_chunk_token_count = max_prefill_tokens_per_step;
        }
        let mut flags = 0;
        if prefill_chunk_count > 1 {
            flags |= PREFILL_PLAN_FLAG_CHUNKED_PREFILL;
        }
        if prompt_token_count >= policy.long_context_threshold_token_count {
            flags |= PREFILL_PLAN_FLAG_LONG_CONTEXT;
        }
        Ok(PrefillPlan {
            flags,
            prompt_token_count,
            max_prefill_tokens_per_step,
            prefill_chunk_count,
            final_chunk_token_count,
            kv_block_token_count: policy.block_token_count,
            kv_block_count_for_prompt: ceil_div_u32(prompt_token_count, policy.block_token_count),
            total_prompt_token_visits: u64::from(prompt_token_count),
        })
    }

    /// Port of `SparkLongContextBuildDecodeSelection`.
    ///
    /// `selected_token_indices.len()` is the selection capacity; it is
    /// clamped to the policy's selected-token capacity as in C. On success
    /// the first `plan.selected_token_count` entries hold the selected
    /// token indices and the rest of the clamped capacity is padded with
    /// [`INVALID_TOKEN_ID`].
    pub fn build_decode_selection(
        &self,
        context_token_count: u32,
        selected_token_indices: &mut [u32],
    ) -> Result<DecodePlan, LongContextError> {
        if context_token_count == 0 || selected_token_indices.is_empty() {
            return Err(LongContextError::InvalidArgument);
        }
        self.validate()?;
        let policy = self.normalized();
        let selected_token_capacity =
            (selected_token_indices.len() as u32).min(policy.selected_token_capacity);
        if context_token_count > policy.max_context_tokens
            && (policy.policy_flags & POLICY_FLAG_FAIL_ON_CONTEXT_OVERFLOW) != 0
        {
            return Err(LongContextError::CapacityExceeded);
        }
        let mut decode_plan = DecodePlan {
            context_token_count,
            selected_token_capacity,
            kv_block_token_count: policy.block_token_count,
            kv_block_count_for_context: ceil_div_u32(context_token_count, policy.block_token_count),
            maximum_decode_scan_token_count: policy.maximum_decode_scan_token_count,
            ..DecodePlan::default()
        };
        if context_token_count >= policy.long_context_threshold_token_count {
            decode_plan.flags |= DECODE_PLAN_FLAG_LONG_CONTEXT;
        }

        if policy.policy_mode == PolicyMode::FullContextScan {
            if (policy.policy_flags & POLICY_FLAG_ALLOW_FULL_CONTEXT_SCAN) == 0 {
                return Err(LongContextError::InvalidArgument);
            }
            return build_full_selection(
                &policy,
                context_token_count,
                selected_token_indices,
                selected_token_capacity,
                decode_plan,
            );
        }

        if context_token_count <= selected_token_capacity
            && context_token_count <= policy.maximum_decode_scan_token_count
        {
            return build_full_selection(
                &policy,
                context_token_count,
                selected_token_indices,
                selected_token_capacity,
                decode_plan,
            );
        }

        build_bounded_selection(
            &policy,
            context_token_count,
            selected_token_indices,
            selected_token_capacity,
            &mut decode_plan,
        );
        if decode_plan.selected_token_count > policy.maximum_decode_scan_token_count {
            return Err(LongContextError::CapacityExceeded);
        }
        Ok(decode_plan)
    }

    /// Port of `SparkLongContextBuildDecodeSelectionForLaneBatch`.
    ///
    /// One selection is built per entry of `context_token_counts`; lane
    /// `i` writes into
    /// `selected_token_indices[i * stride .. i * stride + capacity]`.
    /// Plans are returned in lane order instead of filling a caller
    /// buffer. Unlike the C, an undersized index buffer is rejected with
    /// [`LongContextError::InvalidArgument`] rather than overrunning it.
    pub fn build_decode_selection_for_lane_batch(
        &self,
        context_token_counts: &[u32],
        selected_token_indices: &mut [u32],
        selected_token_stride: usize,
        selected_token_capacity: usize,
    ) -> Result<Vec<DecodePlan>, LongContextError> {
        let lane_count = context_token_counts.len();
        if lane_count == 0
            || selected_token_stride == 0
            || selected_token_capacity == 0
            || selected_token_stride < selected_token_capacity
        {
            return Err(LongContextError::InvalidArgument);
        }
        let required_len = (lane_count - 1) * selected_token_stride + selected_token_capacity;
        if selected_token_indices.len() < required_len {
            return Err(LongContextError::InvalidArgument);
        }
        let mut decode_plans = Vec::with_capacity(lane_count);
        for (lane_index, &context_token_count) in context_token_counts.iter().enumerate() {
            let lane_start = lane_index * selected_token_stride;
            let lane_indices =
                &mut selected_token_indices[lane_start..lane_start + selected_token_capacity];
            let decode_plan = self.build_decode_selection(context_token_count, lane_indices)?;
            decode_plans.push(decode_plan);
        }
        Ok(decode_plans)
    }
}

/// Port of `SparkLongContextBuildFullSelection`. On success the selection
/// holds every context token, padded to capacity; returns
/// [`LongContextError::CapacityExceeded`] when the context does not fit
/// the capacity or the scan budget.
fn build_full_selection(
    policy: &LongContextPolicy,
    context_token_count: u32,
    selected_token_indices: &mut [u32],
    selected_token_capacity: u32,
    mut decode_plan: DecodePlan,
) -> Result<DecodePlan, LongContextError> {
    if context_token_count > selected_token_capacity
        || context_token_count > policy.maximum_decode_scan_token_count
    {
        return Err(LongContextError::CapacityExceeded);
    }
    let capacity = selected_token_capacity as usize;
    for (token_index, slot) in selected_token_indices[..capacity].iter_mut().enumerate() {
        *slot = token_index as u32;
    }
    for slot in selected_token_indices[context_token_count as usize..capacity].iter_mut() {
        *slot = INVALID_TOKEN_ID;
    }
    decode_plan.selected_token_count = context_token_count;
    decode_plan.padded_token_count = selected_token_capacity - context_token_count;
    decode_plan.selected_block_count = ceil_div_u32(context_token_count, policy.block_token_count);
    decode_plan.estimated_attention_token_reads = u64::from(context_token_count);
    if context_token_count < selected_token_capacity {
        decode_plan.flags |= DECODE_PLAN_FLAG_SELECTION_PADDED;
    }
    if policy.policy_mode == PolicyMode::FullContextScan {
        decode_plan.flags |= DECODE_PLAN_FLAG_FULL_CONTEXT_SCAN;
    } else {
        decode_plan.flags |= DECODE_PLAN_FLAG_BOUNDED_SELECTION;
    }
    Ok(decode_plan)
}

/// Port of `SparkLongContextAppendUniqueToken`: append unless the token
/// is already selected, the capacity is exhausted, or the token is the
/// invalid sentinel.
fn append_unique_token(selection: &mut Vec<u32>, selected_token_capacity: u32, token_index: u32) {
    if selection.len() as u32 >= selected_token_capacity || token_index == INVALID_TOKEN_ID {
        return;
    }
    if selection.contains(&token_index) {
        return;
    }
    selection.push(token_index);
}

/// Port of `SparkLongContextCountUniqueBlocks`: distinct KV blocks among
/// the selected (non-sentinel) tokens.
fn count_unique_blocks(selection: &[u32], block_token_count: u32) -> u32 {
    let mut blocks = std::collections::HashSet::new();
    for &token_index in selection {
        if token_index != INVALID_TOKEN_ID {
            blocks.insert(token_index / block_token_count);
        }
    }
    blocks.len() as u32
}

/// Port of `SparkLongContextBuildBoundedSelection`: sink prefix, evenly
/// strided middle sample, then the recent tail, deduplicated and padded
/// to capacity.
fn build_bounded_selection(
    policy: &LongContextPolicy,
    context_token_count: u32,
    selected_token_indices: &mut [u32],
    selected_token_capacity: u32,
    decode_plan: &mut DecodePlan,
) {
    let mut selection: Vec<u32> = Vec::with_capacity(selected_token_capacity as usize);
    let mut sink_count = 0;
    let mut recent_count = 0;
    let mut stride_count = 0;

    if (policy.policy_flags & POLICY_FLAG_INCLUDE_SINK_TOKENS) != 0 {
        sink_count = policy.sink_token_count.min(context_token_count).min(selected_token_capacity);
        for token_index in 0..sink_count {
            append_unique_token(&mut selection, selected_token_capacity, token_index);
        }
    }

    if (policy.policy_flags & POLICY_FLAG_INCLUDE_RECENT_TOKENS) != 0
        && (selection.len() as u32) < selected_token_capacity
    {
        recent_count = policy
            .recent_token_count
            .min(selected_token_capacity - selection.len() as u32)
            .min(context_token_count);
    }
    let mut recent_begin = context_token_count.saturating_sub(recent_count);
    if recent_begin < sink_count {
        recent_begin = sink_count;
    }

    let middle_begin = sink_count;
    let middle_end = recent_begin;
    let middle_token_count = middle_end.saturating_sub(middle_begin);
    if (policy.policy_flags & POLICY_FLAG_INCLUDE_STRIDED_MIDDLE_TOKENS) != 0
        && middle_token_count != 0
        && (selection.len() as u32) < selected_token_capacity
    {
        stride_count = policy
            .stride_sample_token_count
            .min(selected_token_capacity - selection.len() as u32)
            .min(middle_token_count);
        for token_index in 0..stride_count {
            let numerator = (u64::from(token_index) + 1) * u64::from(middle_token_count);
            let mut selected_middle_token =
                middle_begin + (numerator / (u64::from(stride_count) + 1)) as u32;
            if selected_middle_token >= middle_end {
                selected_middle_token = middle_end - 1;
            }
            append_unique_token(&mut selection, selected_token_capacity, selected_middle_token);
        }
    }

    for token_index in recent_begin..context_token_count {
        if selection.len() as u32 >= selected_token_capacity {
            break;
        }
        append_unique_token(&mut selection, selected_token_capacity, token_index);
    }

    // Write the selection out and pad to capacity (SparkLongContextPadSelection).
    let capacity = selected_token_capacity as usize;
    selected_token_indices[..selection.len()].copy_from_slice(&selection);
    for slot in selected_token_indices[selection.len()..capacity].iter_mut() {
        *slot = INVALID_TOKEN_ID;
    }

    let selected_token_count = selection.len() as u32;
    decode_plan.flags |= DECODE_PLAN_FLAG_BOUNDED_SELECTION;
    decode_plan.selected_token_count = selected_token_count;
    decode_plan.padded_token_count = selected_token_capacity - selected_token_count;
    decode_plan.selected_block_count = count_unique_blocks(&selection, policy.block_token_count);
    decode_plan.first_recent_token_index = recent_begin;
    decode_plan.sink_token_count = sink_count;
    decode_plan.stride_sample_token_count = stride_count;
    decode_plan.recent_token_count = context_token_count - recent_begin;
    decode_plan.estimated_attention_token_reads = u64::from(selected_token_count);
    decode_plan.avoided_full_scan_token_reads =
        u64::from(context_token_count.saturating_sub(selected_token_count));
    if selected_token_count < selected_token_capacity {
        decode_plan.flags |= DECODE_PLAN_FLAG_SELECTION_PADDED;
    }
    if context_token_count > selected_token_count {
        decode_plan.flags |= DECODE_PLAN_FLAG_CONTEXT_TRUNCATED;
    }
}
