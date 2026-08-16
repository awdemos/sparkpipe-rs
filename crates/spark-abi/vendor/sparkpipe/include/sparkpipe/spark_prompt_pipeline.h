#ifndef SPARKPIPE_SPARK_GLM52_PROMPT_PIPELINE_H
#define SPARKPIPE_SPARK_GLM52_PROMPT_PIPELINE_H

#include <stdint.h>

#include "sparkpipe/spark_glm52_kv_cache.h"
#include "sparkpipe/spark_request_api.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_PROMPT_PIPELINE_ABI_VERSION 1u
#define SPARK_PROMPT_PIPELINE_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkPromptPipelineConfiguration))
#define SPARK_PROMPT_PIPELINE_PREFILL_DISPATCH_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkPromptPipelinePrefillDispatch))
#define SPARK_PROMPT_PIPELINE_RUN_STATS_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkPromptPipelineRunStats))

#define SPARK_PROMPT_PIPELINE_RUN_FLAG_STOP_AFTER_FIRST_DECODE_DISPATCH \
    0x00000001u
#define SPARK_PROMPT_PIPELINE_RUN_KNOWN_FLAGS \
    SPARK_PROMPT_PIPELINE_RUN_FLAG_STOP_AFTER_FIRST_DECODE_DISPATCH
#define SPARK_PROMPT_PIPELINE_DEFAULT_MAX_DISPATCH_STEPS 4096u

typedef struct SparkPromptPipelinePrefillDispatch
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t step_index;
    uint32_t dispatch_kind;
    uint32_t active_sequence_count;
    uint32_t lane_count;
    uint32_t prompt_token_offset;
    uint32_t prompt_token_count;
    uint32_t prompt_token_stride;
    uint32_t host_token_stride;
    uint32_t reserved0;
    uint32_t reserved1;
    const SparkRequestApiDispatch *request_dispatch;
    const SparkRequestApiPrefillDispatchView *prefill_view;
    const uint32_t *host_token_ids;
    const SparkKvBlockTableView *kv_block_table_view;
} SparkPromptPipelinePrefillDispatch;

typedef SparkStatus (*SparkPromptPipelinePrefillFunction)(
    void *context,
    const SparkPromptPipelinePrefillDispatch *prefill_dispatch);

typedef SparkStatus (*SparkPromptPipelineDecodeFunction)(
    void *context,
    const SparkRequestApiDispatch *decode_dispatch);

typedef struct SparkPromptPipelineConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t run_flags;
    uint32_t max_dispatch_steps;
    SparkRequestApi *request_api;
    uint32_t *host_prefill_token_ids;
    uint32_t host_prefill_token_stride;
    uint32_t host_prefill_lane_capacity;
    uint32_t *host_physical_block_indices;
    const uint32_t *execution_physical_block_indices;
    uint32_t kv_block_lane_stride;
    uint32_t kv_block_lane_capacity;
    uint32_t *lane_physical_block_counts;
    uint32_t lane_count_capacity;
    SparkPromptPipelinePrefillFunction prefill_function;
    SparkPromptPipelineDecodeFunction decode_function;
    void *callback_context;
} SparkPromptPipelineConfiguration;

typedef struct SparkPromptPipelineRunStats
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t completed_dispatch_count;
    uint32_t prefill_dispatch_count;
    uint32_t decode_dispatch_count;
    uint32_t speculative_verify_dispatch_count;
    uint32_t prefill_token_count;
    uint32_t maximum_prefill_token_count;
    uint32_t maximum_prefill_lane_count;
    uint32_t last_dispatch_kind;
    uint32_t reached_decode_dispatch;
    uint32_t reserved0;
} SparkPromptPipelineRunStats;

void SparkPromptPipelineInitializeRunStats(
    SparkPromptPipelineRunStats *stats);

SparkStatus SparkPromptPipelineRun(
    const SparkPromptPipelineConfiguration *configuration,
    SparkPromptPipelineRunStats *stats);

#ifdef __cplusplus
}
#endif

#endif
