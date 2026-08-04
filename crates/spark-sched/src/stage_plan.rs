//! Stage planner — port of `scheduler/stage_plan.c`
//! (API declared in `include/sparkpipe/spark_stage_plan.h`).
//!
//! Splits a model's layer range into pipeline stages under the family cut
//! rules (dense prefix stays whole in stage zero, bounded routed layers per
//! stage), either from an explicit layer-count table, from uniform costs, or
//! by minimizing the maximum stage cost over a measured/estimated per-layer
//! cost profile with an exact dynamic program.
//!
//! Deliberate deviations from C:
//! - Geometry (`layer_count`, `first_routed_layer`) is a runtime parameter;
//!   no GLM-5.2 constants are baked in (the C derives its limits from
//!   `SPARK_GLM52_MODEL_*`). The measured cost profiles below are the C
//!   tables verbatim — they describe a 13x6-layer ring measurement and only
//!   fill as many per-layer entries as the caller's geometry declares, so a
//!   geometry whose `layer_count` exceeds the measured span yields zero-cost
//!   layers that [`build_balanced_with_final_cost`] rejects, exactly like the
//!   C path reading past the measured span.
//! - `Vec` replaces the fixed C arrays bounded by
//!   `SPARK_STAGE_PLAN_MAX_LAYER_COUNT` (128) and
//!   `SPARK_STAGE_PLAN_MAX_STAGE_COUNT` (13); both capacities remain as
//!   validation limits ([`MAX_LAYER_COUNT`], [`MAX_STAGE_COUNT`]).
//! - The C error-buffer strings are carried on the error variants instead of
//!   an out-parameter.

/// Stage-plan ABI version (`SPARK_STAGE_PLAN_ABI_VERSION`).
pub const ABI_VERSION: u32 = 1;
/// Wire descriptor size of the C `SparkStagePlan` struct
/// (`SPARK_STAGE_PLAN_DESCRIPTOR_BYTES`, `sizeof(SparkStagePlan)` = 4 u32
/// header words + 13 stages x 16 bytes). Kept so serialized plans stay
/// byte-compatible with the C layout.
pub const DESCRIPTOR_BYTES: u32 = 16 + MAX_STAGE_COUNT * 16;
/// Maximum layers any family geometry may declare
/// (`SPARK_STAGE_PLAN_MAX_LAYER_COUNT`).
pub const MAX_LAYER_COUNT: u32 = 128;
/// Maximum routed layers one stage may cover
/// (`SPARK_STAGE_PLAN_MAX_ROUTED_LAYERS_PER_STAGE`).
pub const MAX_ROUTED_LAYERS_PER_STAGE: u32 = 8;
/// Spark count of the current production ring
/// (`SPARK_STAGE_PLAN_CURRENT_SPARK_COUNT`).
pub const CURRENT_SPARK_COUNT: u32 = 13;
/// Maximum stages in a plan (`SPARK_STAGE_PLAN_MAX_STAGE_COUNT`).
pub const MAX_STAGE_COUNT: u32 = CURRENT_SPARK_COUNT;
/// Pipeline inflight request capacity
/// (`SPARK_STAGE_PLAN_PIPELINE_INFLIGHT_REQUEST_CAPACITY`).
pub const PIPELINE_INFLIGHT_REQUEST_CAPACITY: u32 = CURRENT_SPARK_COUNT * MAX_BATCH_BUCKET;
/// Measured cost profile captured 2026-07-01
/// (`SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701`).
pub const MEASURED_PROFILE_20260701: u32 = 20260701;
/// Profile zero: no measurements yet; the scheduler builds a balanced plan
/// from one uniform per-layer estimate
/// (`SPARK_STAGE_PLAN_PROFILE_UNIFORM_ESTIMATED`).
pub const PROFILE_UNIFORM_ESTIMATED: u32 = 0;

/// Batch bucket B16 (`SPARK_STAGE_PLAN_BUCKET_B16`).
pub const BUCKET_B16: u32 = 16;
/// Batch bucket B32 (`SPARK_STAGE_PLAN_BUCKET_B32`).
pub const BUCKET_B32: u32 = 32;
/// Batch bucket B64 (`SPARK_STAGE_PLAN_BUCKET_B64`).
pub const BUCKET_B64: u32 = 64;
/// Batch bucket B128 (`SPARK_STAGE_PLAN_BUCKET_B128`).
pub const BUCKET_B128: u32 = 128;
/// Batch bucket B256 (`SPARK_STAGE_PLAN_BUCKET_B256`).
pub const BUCKET_B256: u32 = 256;
/// Batch bucket B512 (`SPARK_STAGE_PLAN_BUCKET_B512`).
pub const BUCKET_B512: u32 = 512;
/// Batch bucket B1024 (`SPARK_STAGE_PLAN_BUCKET_B1024`).
pub const BUCKET_B1024: u32 = 1024;
/// Largest supported batch bucket (`SPARK_STAGE_PLAN_MAX_BATCH_BUCKET`).
pub const MAX_BATCH_BUCKET: u32 = BUCKET_B1024;
/// All supported batch buckets (`SPARK_STAGE_PLAN_BATCH_BUCKETS`).
pub const BATCH_BUCKETS: [u32; 7] =
    [BUCKET_B16, BUCKET_B32, BUCKET_B64, BUCKET_B128, BUCKET_B256, BUCKET_B512, BUCKET_B1024];

/// Stage flag: emits the final token (`SPARK_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN`).
pub const STAGE_FLAG_FINAL_TOKEN: u32 = 0x0000_0001;
/// Stage flag: consumes hidden state (`SPARK_STAGE_PLAN_STAGE_FLAG_INPUT_HIDDEN`).
pub const STAGE_FLAG_INPUT_HIDDEN: u32 = 0x0000_0002;
/// Stage flag: emits hidden state (`SPARK_STAGE_PLAN_STAGE_FLAG_OUTPUT_HIDDEN`).
pub const STAGE_FLAG_OUTPUT_HIDDEN: u32 = 0x0000_0004;
/// Stage flag: covers the dense prefix (`SPARK_STAGE_PLAN_STAGE_FLAG_DENSE_PREFIX`).
pub const STAGE_FLAG_DENSE_PREFIX: u32 = 0x0000_0008;
/// Mask of every defined stage flag (`SPARK_STAGE_PLAN_STAGE_KNOWN_FLAGS`).
pub const STAGE_KNOWN_FLAGS: u32 = STAGE_FLAG_FINAL_TOKEN
    | STAGE_FLAG_INPUT_HIDDEN
    | STAGE_FLAG_OUTPUT_HIDDEN
    | STAGE_FLAG_DENSE_PREFIX;

/// Sentinel cost for unreachable DP states (`SPARK_STAGE_PLAN_UNREACHABLE_COST`).
const UNREACHABLE_COST: u64 = u64::MAX / 4;

/// Model geometry for stage planning (`SparkStagePlanGeometry`). Every entry
/// point that reasons about layer ranges takes these values instead of
/// compiling one family in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagePlanGeometry {
    pub layer_count: u32,
    pub first_routed_layer: u32,
}

/// One pipeline stage's layer range (`SparkStagePlanStage`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StagePlanStage {
    pub first_layer_index: u32,
    pub layer_count: u32,
    pub flags: u32,
    pub reserved: u32,
}

/// A full stage plan (`SparkStagePlan`). `stages` holds exactly
/// `stage_count` entries when produced by the builders (the C version is a
/// fixed array of [`MAX_STAGE_COUNT`] slots of which `stage_count` are
/// live).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagePlan {
    pub abi_version: u32,
    pub descriptor_bytes: u32,
    pub stage_count: u32,
    pub reserved: u32,
    pub stages: Vec<StagePlanStage>,
}

impl StagePlan {
    /// Fresh zeroed plan with the ABI fields set (`SparkStagePlanReset`).
    fn reset() -> Self {
        StagePlan {
            abi_version: ABI_VERSION,
            descriptor_bytes: DESCRIPTOR_BYTES,
            stage_count: 0,
            reserved: 0,
            stages: Vec::new(),
        }
    }

    /// Sum of `layer_count` over the live stages
    /// (`SparkStagePlanTotalLayerCount`).
    pub fn total_layer_count(&self) -> u32 {
        self.stages[..self.stage_count.min(self.stages.len() as u32) as usize]
            .iter()
            .map(|stage| stage.layer_count)
            .sum()
    }
}

/// A loaded cost profile: per-layer cost in nanoseconds plus the extra cost
/// charged to the final (token-emitting) stage. `layer_cost_ns` is sized to
/// the geometry's `layer_count` (the C version writes into a
/// [`MAX_LAYER_COUNT`]-entry out array).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CostProfile {
    pub layer_cost_ns: Vec<u64>,
    pub final_stage_extra_cost_ns: u64,
}

/// Execution chunk shape (`SparkStagePlanExecutionChunkShape` outputs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionChunkShape {
    pub maximum_sequences_per_chunk: u32,
    pub chunk_count: u32,
}

/// Quantization mode selector (`SPARK_STAGE_PLAN_QUANTIZATION_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    /// `SPARK_STAGE_PLAN_QUANTIZATION_AUTO` — resolves to
    /// [`QuantizationMode::Nvfp4_4Bit`].
    Auto,
    /// `SPARK_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT`.
    Nvfp4_4Bit,
    /// `SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT`.
    Fp8E4m3_8Bit,
}

impl QuantizationMode {
    /// `SparkStagePlanNormalizeQuantizationMode`.
    fn normalize(self) -> Result<Self, StagePlanError> {
        match self {
            QuantizationMode::Auto => Ok(QuantizationMode::Nvfp4_4Bit),
            QuantizationMode::Nvfp4_4Bit | QuantizationMode::Fp8E4m3_8Bit => Ok(self),
        }
    }
}

/// Errors mirroring the `SparkStatus` codes returned by the C entry points;
/// each variant carries the message the C version writes into its error
/// buffer.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StagePlanError {
    /// `SPARK_STATUS_INVALID_ARGUMENT`.
    #[error("invalid argument: {0}")]
    InvalidArgument(&'static str),
    /// `SPARK_STATUS_CAPACITY_EXCEEDED`.
    #[error("capacity exceeded: {0}")]
    CapacityExceeded(&'static str),
    /// `SPARK_STATUS_ABI_MISMATCH`.
    #[error("ABI mismatch: {0}")]
    AbiMismatch(&'static str),
    /// `SPARK_STATUS_INTERNAL_ERROR`.
    #[error("internal error: {0}")]
    InternalError(&'static str),
}

/// `SparkStagePlanBatchBucketIsSupported`.
pub fn batch_bucket_is_supported(batch_bucket: u32) -> bool {
    BATCH_BUCKETS.contains(&batch_bucket)
}

/// `SparkStagePlanSelectBatchBucketValue` — returns 0 when no bucket fits.
fn select_batch_bucket_value(active_sequence_count: u32) -> u32 {
    if active_sequence_count == 0 || active_sequence_count > MAX_BATCH_BUCKET {
        return 0;
    }
    for bucket in BATCH_BUCKETS {
        if active_sequence_count <= bucket {
            return bucket;
        }
    }
    BUCKET_B1024
}

/// `SparkRoutedLayerCountForRange`.
pub fn routed_layer_count_for_range(
    first_layer_index: u32,
    layer_count: u32,
    first_routed_layer: u32,
    total_layer_count: u32,
) -> u32 {
    let range_end = first_layer_index + layer_count;
    let routed_begin = first_layer_index.max(first_routed_layer);
    let routed_end = range_end.min(total_layer_count);
    routed_end.saturating_sub(routed_begin)
}

/// `SparkStagePlanLayerRangeIsValid`.
fn layer_range_is_valid(
    geometry: &StagePlanGeometry,
    first_layer_index: u32,
    layer_count: u32,
) -> bool {
    if layer_count == 0 || first_layer_index >= geometry.layer_count {
        return false;
    }
    if layer_count > geometry.layer_count - first_layer_index {
        return false;
    }
    let range_end = first_layer_index + layer_count;
    // The dense prefix stays whole in stage zero - a rule that only
    // means something when a routed region exists. A fully dense model
    // (first_routed == layer_count) may cut anywhere.
    if geometry.first_routed_layer < geometry.layer_count {
        if first_layer_index != 0 && first_layer_index < geometry.first_routed_layer {
            return false;
        }
        if range_end < geometry.first_routed_layer {
            return false;
        }
    }
    let routed_layer_count = routed_layer_count_for_range(
        first_layer_index,
        layer_count,
        geometry.first_routed_layer,
        geometry.layer_count,
    );
    routed_layer_count <= MAX_ROUTED_LAYERS_PER_STAGE
}

/// `SparkStagePlanAssignStageFlags`.
fn assign_stage_flags(geometry: &StagePlanGeometry, stage_plan: &mut StagePlan) {
    let stage_count = stage_plan.stage_count as usize;
    for (stage_index, stage) in stage_plan.stages[..stage_count].iter_mut().enumerate() {
        stage.flags = STAGE_FLAG_INPUT_HIDDEN;
        if stage_index + 1 < stage_count {
            stage.flags |= STAGE_FLAG_OUTPUT_HIDDEN;
        } else {
            stage.flags |= STAGE_FLAG_FINAL_TOKEN;
        }
        if stage.first_layer_index == 0 && stage.layer_count >= geometry.first_routed_layer {
            stage.flags |= STAGE_FLAG_DENSE_PREFIX;
        }
    }
}

/// `SparkStagePlanValidate`.
pub fn validate(
    geometry: &StagePlanGeometry,
    stage_plan: &StagePlan,
) -> Result<(), StagePlanError> {
    if stage_plan.abi_version != ABI_VERSION || stage_plan.descriptor_bytes != DESCRIPTOR_BYTES {
        return Err(StagePlanError::AbiMismatch("stage plan ABI mismatch"));
    }
    if stage_plan.stage_count == 0 || stage_plan.stage_count > MAX_STAGE_COUNT {
        return Err(StagePlanError::InvalidArgument("stage count is outside supported range"));
    }
    if stage_plan.stages.len() < stage_plan.stage_count as usize {
        return Err(StagePlanError::InvalidArgument(
            "stage table holds fewer entries than stage count",
        ));
    }

    let stage_count = stage_plan.stage_count as usize;
    let mut expected_first_layer_index = 0u32;
    let mut final_stage_count = 0u32;
    for (stage_index, stage) in stage_plan.stages[..stage_count].iter().enumerate() {
        if stage.flags & !STAGE_KNOWN_FLAGS != 0 {
            return Err(StagePlanError::InvalidArgument("stage contains unknown flags"));
        }
        if stage.first_layer_index != expected_first_layer_index {
            return Err(StagePlanError::InvalidArgument("stage layers are not contiguous"));
        }
        if !layer_range_is_valid(geometry, stage.first_layer_index, stage.layer_count) {
            return Err(StagePlanError::InvalidArgument(
                "stage layer range violates GLM-5.2 cut rules",
            ));
        }
        if stage.flags & STAGE_FLAG_FINAL_TOKEN != 0 {
            final_stage_count += 1;
            if stage_index + 1 != stage_count {
                return Err(StagePlanError::InvalidArgument(
                    "final-token stage is not the last stage",
                ));
            }
        } else if stage.flags & STAGE_FLAG_OUTPUT_HIDDEN == 0 {
            return Err(StagePlanError::InvalidArgument("non-final stage must emit hidden state"));
        }
        expected_first_layer_index = stage.first_layer_index + stage.layer_count;
    }

    if expected_first_layer_index != geometry.layer_count {
        return Err(StagePlanError::InvalidArgument(
            "stage plan does not cover all GLM-5.2 layers",
        ));
    }
    if final_stage_count != 1 {
        return Err(StagePlanError::InvalidArgument(
            "stage plan must contain exactly one final-token stage",
        ));
    }
    Ok(())
}

/// `SparkStagePlanBuildFromLayerCounts`. `stage_count` is the length of
/// `layer_counts`.
pub fn build_from_layer_counts(
    geometry: &StagePlanGeometry,
    layer_counts: &[u32],
) -> Result<StagePlan, StagePlanError> {
    let stage_count = layer_counts.len();
    if stage_count == 0 || stage_count > MAX_STAGE_COUNT as usize {
        return Err(StagePlanError::InvalidArgument("invalid table-driven stage-plan input"));
    }
    let mut stage_plan = StagePlan::reset();
    stage_plan.stage_count = stage_count as u32;
    let mut first_layer_index = 0u32;
    for &layer_count in layer_counts {
        if layer_count == 0 || layer_count > geometry.layer_count - first_layer_index {
            return Err(StagePlanError::InvalidArgument(
                "table-driven stage layer count is invalid",
            ));
        }
        stage_plan.stages.push(StagePlanStage {
            first_layer_index,
            layer_count,
            flags: 0,
            reserved: 0,
        });
        first_layer_index += layer_count;
    }
    assign_stage_flags(geometry, &mut stage_plan);
    validate(geometry, &stage_plan)?;
    Ok(stage_plan)
}

/// `SparkStagePlanBuildBalancedWithFinalCost`: exact DP minimizing the
/// maximum per-stage cost (final stage additionally charged
/// `final_stage_extra_cost_ns`). `layer_cost_ns` must hold at least
/// `geometry.layer_count` entries.
pub fn build_balanced_with_final_cost(
    geometry: &StagePlanGeometry,
    layer_cost_ns: &[u64],
    final_stage_extra_cost_ns: u64,
    stage_count: u32,
) -> Result<StagePlan, StagePlanError> {
    let layer_count = geometry.layer_count as usize;
    if layer_cost_ns.len() < layer_count || stage_count == 0 || stage_count > MAX_STAGE_COUNT {
        return Err(StagePlanError::InvalidArgument("invalid balanced stage-plan input"));
    }

    let mut prefix_cost_ns = vec![0u64; layer_count + 1];
    for layer_index in 0..layer_count {
        if layer_cost_ns[layer_index] == 0 {
            return Err(StagePlanError::InvalidArgument("layer cost cannot be zero"));
        }
        prefix_cost_ns[layer_index + 1] = prefix_cost_ns[layer_index] + layer_cost_ns[layer_index];
        if prefix_cost_ns[layer_index + 1] < prefix_cost_ns[layer_index] {
            return Err(StagePlanError::CapacityExceeded("layer cost prefix overflow"));
        }
    }

    let stage_rows = stage_count as usize + 1;
    let mut best_cost = vec![vec![UNREACHABLE_COST; layer_count + 1]; stage_rows];
    let mut best_split = vec![vec![u32::MAX; layer_count + 1]; stage_rows];
    best_cost[0][0] = 0;

    for stage_index in 1..=stage_count as usize {
        for layer_index in 1..=layer_count {
            for split_layer_index in 0..layer_index {
                if best_cost[stage_index - 1][split_layer_index] == UNREACHABLE_COST {
                    continue;
                }
                if !layer_range_is_valid(
                    geometry,
                    split_layer_index as u32,
                    (layer_index - split_layer_index) as u32,
                ) {
                    continue;
                }
                let mut segment_cost =
                    prefix_cost_ns[layer_index] - prefix_cost_ns[split_layer_index];
                if stage_index == stage_count as usize {
                    if segment_cost > u64::MAX - final_stage_extra_cost_ns {
                        continue;
                    }
                    segment_cost += final_stage_extra_cost_ns;
                }
                let candidate_cost =
                    best_cost[stage_index - 1][split_layer_index].max(segment_cost);
                if candidate_cost < best_cost[stage_index][layer_index] {
                    best_cost[stage_index][layer_index] = candidate_cost;
                    best_split[stage_index][layer_index] = split_layer_index as u32;
                }
            }
        }
    }

    if best_cost[stage_count as usize][layer_count] == UNREACHABLE_COST {
        return Err(StagePlanError::CapacityExceeded("stage count cannot satisfy the cut rules"));
    }

    let mut stage_plan = StagePlan::reset();
    stage_plan.stage_count = stage_count;
    stage_plan.stages = vec![StagePlanStage::default(); stage_count as usize];
    let mut current_layer_index = layer_count;
    for stage_index in (1..=stage_count as usize).rev() {
        let split_layer_index = best_split[stage_index][current_layer_index];
        if split_layer_index == u32::MAX {
            return Err(StagePlanError::InternalError("stage-plan backtrack failed"));
        }
        stage_plan.stages[stage_index - 1].first_layer_index = split_layer_index;
        stage_plan.stages[stage_index - 1].layer_count =
            (current_layer_index - split_layer_index as usize) as u32;
        current_layer_index = split_layer_index as usize;
    }
    assign_stage_flags(geometry, &mut stage_plan);
    validate(geometry, &stage_plan)?;
    Ok(stage_plan)
}

/// `SparkStagePlanBuildBalanced` — balanced build with no extra final-stage
/// cost.
pub fn build_balanced(
    geometry: &StagePlanGeometry,
    layer_cost_ns: &[u64],
    stage_count: u32,
) -> Result<StagePlan, StagePlanError> {
    build_balanced_with_final_cost(geometry, layer_cost_ns, 0, stage_count)
}

/// `SparkStagePlanBuildUniform` — every layer costs 1 ns.
pub fn build_uniform(
    geometry: &StagePlanGeometry,
    stage_count: u32,
) -> Result<StagePlan, StagePlanError> {
    let layer_cost_ns = vec![1u64; geometry.layer_count as usize];
    build_balanced(geometry, &layer_cost_ns, stage_count)
}

/// `SparkStagePlanStoreUniformSegmentCost`: spread a segment cost over its
/// layers, giving the remainder to the earliest layers. Writes are clipped
/// to the destination length (see module docs).
fn store_uniform_segment_cost(
    segment_cost_ns: u64,
    first_layer_index: u32,
    layer_count: u32,
    layer_cost_ns: &mut [u64],
) {
    let base_layer_cost_ns = segment_cost_ns / u64::from(layer_count);
    let remainder_ns = segment_cost_ns % u64::from(layer_count);
    for layer_offset in 0..layer_count {
        let index = (first_layer_index + layer_offset) as usize;
        if index >= layer_cost_ns.len() {
            break;
        }
        layer_cost_ns[index] =
            base_layer_cost_ns + u64::from(u64::from(layer_offset) < remainder_ns);
    }
}

/// `SparkStagePlanScaleCostProfile`: ceiling division scaling by
/// `numerator / denominator`.
fn scale_cost_profile(
    layer_cost_ns: &mut [u64],
    final_stage_extra_cost_ns: &mut u64,
    numerator: u32,
    denominator: u32,
) {
    for cost in layer_cost_ns.iter_mut() {
        *cost = (*cost * u64::from(numerator)).div_ceil(u64::from(denominator));
    }
    *final_stage_extra_cost_ns =
        (*final_stage_extra_cost_ns * u64::from(numerator)).div_ceil(u64::from(denominator));
}

/// `SparkStagePlanLoadMeasuredB64CostProfile`.
fn load_measured_b64_cost_profile(layer_cost_ns: &mut [u64]) -> u64 {
    const FIRST_LAYER_INDEX: [u32; 13] = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72];
    const LAYER_COUNT: [u32; 13] = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6];
    const STAGE_COST_NS: [u64; 13] = [
        50660288, 45685889, 45232480, 45782816, 44223711, 45062784, 45439968, 45055391, 45370304,
        46190688, 44552225, 45824320, 46449792,
    ];
    for stage_index in 0..13usize {
        store_uniform_segment_cost(
            STAGE_COST_NS[stage_index],
            FIRST_LAYER_INDEX[stage_index],
            LAYER_COUNT[stage_index],
            layer_cost_ns,
        );
    }
    0
}

/// `SparkStagePlanLoadMeasuredB128CostProfile`.
fn load_measured_b128_cost_profile(layer_cost_ns: &mut [u64]) -> u64 {
    const STAGE_COST_NS: [u64; 13] = [
        177756500, 188399000, 193653016, 197361562, 195107453, 195583047, 192805688, 194702031,
        195701188, 193026547, 196165188, 190992359, 194542891,
    ];
    for (stage_index, &stage_cost_ns) in STAGE_COST_NS.iter().enumerate() {
        store_uniform_segment_cost(stage_cost_ns, stage_index as u32 * 6, 6, layer_cost_ns);
    }
    0
}

/// `SparkStagePlanLoadMeasuredB32CostProfile`.
fn load_measured_b32_cost_profile(layer_cost_ns: &mut [u64]) -> u64 {
    const STAGE_COST_NS: [u64; 13] = [
        39691000, 35142000, 33496000, 36755000, 38837000, 41760000, 46160000, 56283000, 60862000,
        69126000, 72429000, 81610000, 73314000,
    ];
    for (stage_index, &stage_cost_ns) in STAGE_COST_NS.iter().enumerate() {
        store_uniform_segment_cost(stage_cost_ns, stage_index as u32 * 6, 6, layer_cost_ns);
    }
    39342000
}

/// `SparkStagePlanLoadEstimatedLargeBatchCostProfile`: scale the B128
/// measurement up to a larger supported bucket.
fn load_estimated_large_batch_cost_profile(
    batch_bucket: u32,
    layer_cost_ns: &mut [u64],
    final_stage_extra_cost_ns: &mut u64,
) -> Result<(), StagePlanError> {
    if batch_bucket < BUCKET_B128 || !batch_bucket_is_supported(batch_bucket) {
        return Err(StagePlanError::InvalidArgument("invalid large-batch profile bucket"));
    }
    *final_stage_extra_cost_ns = load_measured_b128_cost_profile(layer_cost_ns);
    scale_cost_profile(layer_cost_ns, final_stage_extra_cost_ns, batch_bucket, BUCKET_B128);
    Ok(())
}

/// `SparkStagePlanLoadUniformCostProfile`.
pub fn load_uniform_cost_profile(
    geometry: &StagePlanGeometry,
    layer_cost_ns_estimate: u64,
    final_stage_extra_cost_ns_estimate: u64,
) -> Result<CostProfile, StagePlanError> {
    if layer_cost_ns_estimate == 0
        || geometry.layer_count == 0
        || geometry.layer_count > MAX_LAYER_COUNT
    {
        return Err(StagePlanError::InvalidArgument("invalid uniform cost-profile input"));
    }
    Ok(CostProfile {
        layer_cost_ns: vec![layer_cost_ns_estimate; geometry.layer_count as usize],
        final_stage_extra_cost_ns: final_stage_extra_cost_ns_estimate,
    })
}

/// `SparkStagePlanLoadMeasuredCostProfileForQuantization`.
pub fn load_measured_cost_profile_for_quantization(
    geometry: &StagePlanGeometry,
    measured_profile_id: u32,
    batch_bucket: u32,
    quantization_mode: QuantizationMode,
) -> Result<CostProfile, StagePlanError> {
    // Normalization also rejects the modes the C switch refuses.
    quantization_mode
        .normalize()
        .map_err(|_| StagePlanError::InvalidArgument("invalid quantization mode"))?;
    if measured_profile_id != MEASURED_PROFILE_20260701 {
        return Err(StagePlanError::InvalidArgument("unknown measured profile"));
    }

    let mut layer_cost_ns = vec![0u64; geometry.layer_count as usize];
    let final_stage_extra_cost_ns = if batch_bucket == BUCKET_B64 {
        load_measured_b64_cost_profile(&mut layer_cost_ns)
    } else if batch_bucket == BUCKET_B128 {
        load_measured_b128_cost_profile(&mut layer_cost_ns)
    } else if batch_bucket > BUCKET_B128 && batch_bucket_is_supported(batch_bucket) {
        let mut extra = 0u64;
        load_estimated_large_batch_cost_profile(batch_bucket, &mut layer_cost_ns, &mut extra)?;
        extra
    } else if batch_bucket == BUCKET_B32 || batch_bucket == BUCKET_B16 {
        load_measured_b32_cost_profile(&mut layer_cost_ns)
    } else {
        return Err(StagePlanError::InvalidArgument("unsupported batch bucket"));
    };
    Ok(CostProfile { layer_cost_ns, final_stage_extra_cost_ns })
}

/// `SparkStagePlanLoadMeasuredCostProfile` (quantization auto).
pub fn load_measured_cost_profile(
    geometry: &StagePlanGeometry,
    measured_profile_id: u32,
    batch_bucket: u32,
) -> Result<CostProfile, StagePlanError> {
    load_measured_cost_profile_for_quantization(
        geometry,
        measured_profile_id,
        batch_bucket,
        QuantizationMode::Auto,
    )
}

/// `SparkStagePlanBuildMeasuredBalancedForQuantization`.
pub fn build_measured_balanced_for_quantization(
    geometry: &StagePlanGeometry,
    measured_profile_id: u32,
    batch_bucket: u32,
    quantization_mode: QuantizationMode,
    stage_count: u32,
) -> Result<StagePlan, StagePlanError> {
    let profile = load_measured_cost_profile_for_quantization(
        geometry,
        measured_profile_id,
        batch_bucket,
        quantization_mode,
    )
    .map_err(|error| match error {
        StagePlanError::InvalidArgument(_) => {
            StagePlanError::InvalidArgument("measured stage-plan profile is unavailable")
        }
        other => other,
    })?;
    build_balanced_with_final_cost(
        geometry,
        &profile.layer_cost_ns,
        profile.final_stage_extra_cost_ns,
        stage_count,
    )
}

/// `SparkStagePlanBuildMeasuredBalanced` (quantization auto).
pub fn build_measured_balanced(
    geometry: &StagePlanGeometry,
    measured_profile_id: u32,
    batch_bucket: u32,
    stage_count: u32,
) -> Result<StagePlan, StagePlanError> {
    build_measured_balanced_for_quantization(
        geometry,
        measured_profile_id,
        batch_bucket,
        QuantizationMode::Auto,
        stage_count,
    )
}

/// `SparkStagePlanBuildMeasuredB64RingExact`: the measured 13x6-layer ring
/// cut, used when the current spark count matches the measurement.
fn build_measured_b64_ring_exact(
    geometry: &StagePlanGeometry,
) -> Result<StagePlan, StagePlanError> {
    const FIRST_LAYER_INDEX: [u32; 13] = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72];
    let mut stage_plan = StagePlan::reset();
    stage_plan.stage_count = CURRENT_SPARK_COUNT;
    for &first_layer_index in FIRST_LAYER_INDEX.iter() {
        stage_plan.stages.push(StagePlanStage {
            first_layer_index,
            layer_count: 6,
            flags: 0,
            reserved: 0,
        });
    }
    assign_stage_flags(geometry, &mut stage_plan);
    validate(geometry, &stage_plan)?;
    Ok(stage_plan)
}

/// `SparkStagePlanBuildCurrentSparkMeasuredBalancedForQuantization`.
pub fn build_current_spark_measured_balanced_for_quantization(
    geometry: &StagePlanGeometry,
    measured_profile_id: u32,
    batch_bucket: u32,
    quantization_mode: QuantizationMode,
) -> Result<StagePlan, StagePlanError> {
    let normalized_quantization_mode = quantization_mode.normalize().map_err(|_| {
        StagePlanError::InvalidArgument("invalid measured stage-plan quantization mode")
    })?;
    if measured_profile_id == MEASURED_PROFILE_20260701
        && batch_bucket >= BUCKET_B64
        && batch_bucket_is_supported(batch_bucket)
        && (normalized_quantization_mode == QuantizationMode::Nvfp4_4Bit
            || normalized_quantization_mode == QuantizationMode::Fp8E4m3_8Bit)
    {
        return build_measured_b64_ring_exact(geometry);
    }
    build_measured_balanced_for_quantization(
        geometry,
        measured_profile_id,
        batch_bucket,
        normalized_quantization_mode,
        CURRENT_SPARK_COUNT,
    )
}

/// `SparkStagePlanBuildCurrentSparkMeasuredBalanced` (quantization auto).
pub fn build_current_spark_measured_balanced(
    geometry: &StagePlanGeometry,
    measured_profile_id: u32,
    batch_bucket: u32,
) -> Result<StagePlan, StagePlanError> {
    build_current_spark_measured_balanced_for_quantization(
        geometry,
        measured_profile_id,
        batch_bucket,
        QuantizationMode::Auto,
    )
}

/// `SparkStagePlanSelectBatchBucket`.
pub fn select_batch_bucket(active_sequence_count: u32) -> Result<u32, StagePlanError> {
    if active_sequence_count == 0 {
        return Err(StagePlanError::InvalidArgument("active sequence count is zero"));
    }
    let bucket = select_batch_bucket_value(active_sequence_count);
    if bucket == 0 {
        return Err(StagePlanError::CapacityExceeded(
            "active sequence count exceeds the maximum batch bucket",
        ));
    }
    Ok(bucket)
}

/// `SparkStagePlanExecutionChunkShape`.
pub fn execution_chunk_shape(
    logical_sequence_count: u32,
    rows_per_sequence: u32,
    execution_row_capacity: u32,
) -> Result<ExecutionChunkShape, StagePlanError> {
    if logical_sequence_count == 0 || rows_per_sequence == 0 || execution_row_capacity == 0 {
        return Err(StagePlanError::InvalidArgument("invalid execution chunk-shape input"));
    }
    let execution_row_capacity = execution_row_capacity.min(MAX_BATCH_BUCKET);
    let mut maximum_sequences_per_chunk = execution_row_capacity / rows_per_sequence;
    if maximum_sequences_per_chunk == 0 {
        return Err(StagePlanError::CapacityExceeded(
            "one sequence does not fit the execution row capacity",
        ));
    }
    if maximum_sequences_per_chunk > logical_sequence_count {
        maximum_sequences_per_chunk = logical_sequence_count;
    }
    let mut chunk_count = logical_sequence_count / maximum_sequences_per_chunk;
    if logical_sequence_count % maximum_sequences_per_chunk != 0 {
        chunk_count += 1;
    }
    Ok(ExecutionChunkShape { maximum_sequences_per_chunk, chunk_count })
}
