#ifndef SPARKPIPE_SPARK_STAGE_PLAN_H
#define SPARKPIPE_SPARK_STAGE_PLAN_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_STAGE_PLAN_ABI_VERSION 1u
// Model geometry is a runtime parameter now: every entry that reasons
// about layer ranges takes the family's values instead of compiling one
// family in. The capacity below bounds every family's layer count.
#define SPARK_STAGE_PLAN_MAX_LAYER_COUNT 128u

typedef struct SparkStagePlanGeometry
{
    uint32_t layer_count;
    uint32_t first_routed_layer;
} SparkStagePlanGeometry;
#define SPARK_STAGE_PLAN_MAX_ROUTED_LAYERS_PER_STAGE 8u
#define SPARK_STAGE_PLAN_CURRENT_SPARK_COUNT 13u
#define SPARK_STAGE_PLAN_PIPELINE_INFLIGHT_REQUEST_CAPACITY \
    (SPARK_STAGE_PLAN_CURRENT_SPARK_COUNT * \
     SPARK_STAGE_PLAN_MAX_BATCH_BUCKET)
#define SPARK_STAGE_PLAN_MAX_STAGE_COUNT \
    SPARK_STAGE_PLAN_CURRENT_SPARK_COUNT
#define SPARK_STAGE_PLAN_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkStagePlan))
#define SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701 20260701u
// Profile zero: no measurements yet for this family. The scheduler
// builds a balanced plan from one uniform per-layer estimate that the
// wiring supplies - the bring-up admission path for every new model.
#define SPARK_STAGE_PLAN_PROFILE_UNIFORM_ESTIMATED 0u

#define SPARK_STAGE_PLAN_QUANTIZATION_AUTO 0u
#define SPARK_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT 1u
#define SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT 2u

#define SPARK_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN 0x00000001u
#define SPARK_STAGE_PLAN_STAGE_FLAG_INPUT_HIDDEN 0x00000002u
#define SPARK_STAGE_PLAN_STAGE_FLAG_OUTPUT_HIDDEN 0x00000004u
#define SPARK_STAGE_PLAN_STAGE_FLAG_DENSE_PREFIX 0x00000008u
#define SPARK_STAGE_PLAN_STAGE_KNOWN_FLAGS \
    (SPARK_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN | \
     SPARK_STAGE_PLAN_STAGE_FLAG_INPUT_HIDDEN | \
     SPARK_STAGE_PLAN_STAGE_FLAG_OUTPUT_HIDDEN | \
     SPARK_STAGE_PLAN_STAGE_FLAG_DENSE_PREFIX)

#define SPARK_STAGE_PLAN_BUCKET_B16 16u
#define SPARK_STAGE_PLAN_BUCKET_B32 32u
#define SPARK_STAGE_PLAN_BUCKET_B64 64u
#define SPARK_STAGE_PLAN_BUCKET_B128 128u
#define SPARK_STAGE_PLAN_BUCKET_B256 256u
#define SPARK_STAGE_PLAN_BUCKET_B512 512u
#define SPARK_STAGE_PLAN_BUCKET_B1024 1024u
#define SPARK_STAGE_PLAN_MAX_BATCH_BUCKET \
    SPARK_STAGE_PLAN_BUCKET_B1024
#define SPARK_STAGE_PLAN_BATCH_BUCKETS \
    SPARK_STAGE_PLAN_BUCKET_B16, \
    SPARK_STAGE_PLAN_BUCKET_B32, \
    SPARK_STAGE_PLAN_BUCKET_B64, \
    SPARK_STAGE_PLAN_BUCKET_B128, \
    SPARK_STAGE_PLAN_BUCKET_B256, \
    SPARK_STAGE_PLAN_BUCKET_B512, \
    SPARK_STAGE_PLAN_BUCKET_B1024

static inline uint32_t SparkStagePlanBatchBucketIsSupported(
    uint32_t batch_bucket)
{
    if (batch_bucket == SPARK_STAGE_PLAN_BUCKET_B16 ||
        batch_bucket == SPARK_STAGE_PLAN_BUCKET_B32 ||
        batch_bucket == SPARK_STAGE_PLAN_BUCKET_B64 ||
        batch_bucket == SPARK_STAGE_PLAN_BUCKET_B128 ||
        batch_bucket == SPARK_STAGE_PLAN_BUCKET_B256 ||
        batch_bucket == SPARK_STAGE_PLAN_BUCKET_B512 ||
        batch_bucket == SPARK_STAGE_PLAN_BUCKET_B1024)
    {
        return 1u;
    }
    return 0u;
}

static inline uint32_t SparkStagePlanSelectBatchBucketValue(
    uint32_t active_sequence_count)
{
    if (active_sequence_count == 0u ||
        active_sequence_count > SPARK_STAGE_PLAN_MAX_BATCH_BUCKET)
    {
        return 0u;
    }
    if (active_sequence_count <= SPARK_STAGE_PLAN_BUCKET_B16)
    {
        return SPARK_STAGE_PLAN_BUCKET_B16;
    }
    if (active_sequence_count <= SPARK_STAGE_PLAN_BUCKET_B32)
    {
        return SPARK_STAGE_PLAN_BUCKET_B32;
    }
    if (active_sequence_count <= SPARK_STAGE_PLAN_BUCKET_B64)
    {
        return SPARK_STAGE_PLAN_BUCKET_B64;
    }
    if (active_sequence_count <= SPARK_STAGE_PLAN_BUCKET_B128)
    {
        return SPARK_STAGE_PLAN_BUCKET_B128;
    }
    if (active_sequence_count <= SPARK_STAGE_PLAN_BUCKET_B256)
    {
        return SPARK_STAGE_PLAN_BUCKET_B256;
    }
    if (active_sequence_count <= SPARK_STAGE_PLAN_BUCKET_B512)
    {
        return SPARK_STAGE_PLAN_BUCKET_B512;
    }
    return SPARK_STAGE_PLAN_BUCKET_B1024;
}

typedef struct SparkStagePlanStage
{
    uint32_t first_layer_index;
    uint32_t layer_count;
    uint32_t flags;
    uint32_t reserved;
} SparkStagePlanStage;

typedef struct SparkStagePlan
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t stage_count;
    uint32_t reserved;
    SparkStagePlanStage stages[SPARK_STAGE_PLAN_MAX_STAGE_COUNT];
} SparkStagePlan;

SparkStatus SparkStagePlanValidate(
    const SparkStagePlanGeometry *geometry,
    const SparkStagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkStagePlanBuildFromLayerCounts(
    const SparkStagePlanGeometry *geometry,
    const uint32_t *layer_counts,
    uint32_t stage_count,
    SparkStagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkStagePlanBuildUniform(
    const SparkStagePlanGeometry *geometry,
    uint32_t stage_count,
    SparkStagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkStagePlanBuildBalanced(
    const SparkStagePlanGeometry *geometry,
    const uint64_t *layer_cost_ns,
    uint32_t stage_count,
    SparkStagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkStagePlanBuildBalancedWithFinalCost(
    const SparkStagePlanGeometry *geometry,
    const uint64_t *layer_cost_ns,
    uint64_t final_stage_extra_cost_ns,
    uint32_t stage_count,
    SparkStagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkStagePlanLoadUniformCostProfile(
    const SparkStagePlanGeometry *geometry,
    uint64_t layer_cost_ns_estimate,
    uint64_t final_stage_extra_cost_ns_estimate,
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT],
    uint64_t *final_stage_extra_cost_ns_out);

SparkStatus SparkStagePlanLoadMeasuredCostProfile(
    const SparkStagePlanGeometry *geometry,
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT],
    uint64_t *final_stage_extra_cost_ns_out);

SparkStatus SparkStagePlanLoadMeasuredCostProfileForQuantization(
    const SparkStagePlanGeometry *geometry,
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint32_t quantization_mode,
    uint64_t layer_cost_ns[SPARK_STAGE_PLAN_MAX_LAYER_COUNT],
    uint64_t *final_stage_extra_cost_ns_out);

SparkStatus SparkStagePlanBuildMeasuredBalanced(
    const SparkStagePlanGeometry *geometry,
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint32_t stage_count,
    SparkStagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkStagePlanBuildMeasuredBalancedForQuantization(
    const SparkStagePlanGeometry *geometry,
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint32_t quantization_mode,
    uint32_t stage_count,
    SparkStagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkStagePlanBuildCurrentSparkMeasuredBalanced(
    const SparkStagePlanGeometry *geometry,
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    SparkStagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkStagePlanBuildCurrentSparkMeasuredBalancedForQuantization(
    const SparkStagePlanGeometry *geometry,
    uint32_t measured_profile_id,
    uint32_t batch_bucket,
    uint32_t quantization_mode,
    SparkStagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkStagePlanSelectBatchBucket(
    uint32_t active_sequence_count,
    uint32_t *bucket_out);

SparkStatus SparkStagePlanExecutionChunkShape(
    uint32_t logical_sequence_count,
    uint32_t rows_per_sequence,
    uint32_t execution_row_capacity,
    uint32_t *maximum_sequences_per_chunk_out,
    uint32_t *chunk_count_out);

static inline uint32_t SparkStagePlanTotalLayerCount(const SparkStagePlan *stage_plan)
{
	uint32_t stage_index,total;
	total = 0u;
	for (stage_index = 0u; stage_index < stage_plan->stage_count; ++stage_index)
		total += stage_plan->stages[stage_index].layer_count;
	return total;
}

#ifdef __cplusplus
}
#endif


static inline uint32_t SparkRoutedLayerCountForRange(uint32_t first_layer_index, uint32_t layer_count, uint32_t first_routed_layer, uint32_t total_layer_count)
{
	uint32_t range_end = first_layer_index + layer_count;
	uint32_t routed_begin = first_layer_index > first_routed_layer ? first_layer_index : first_routed_layer;
	uint32_t routed_end = range_end < total_layer_count ? range_end : total_layer_count;
	return routed_end <= routed_begin ? 0u : routed_end - routed_begin;
}
#endif
