#ifndef SPARKPIPE_SPARK_GLM52_LONG_CONTEXT_H
#define SPARKPIPE_SPARK_GLM52_LONG_CONTEXT_H

#include <stdint.h>

#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_LONG_CONTEXT_ABI_VERSION 1u
#define SPARK_LONG_CONTEXT_POLICY_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkLongContextPolicy))
#define SPARK_LONG_CONTEXT_DECODE_PLAN_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkLongContextDecodePlan))
#define SPARK_LONG_CONTEXT_PREFILL_PLAN_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkLongContextPrefillPlan))

#define SPARK_LONG_CONTEXT_DEFAULT_MAX_CONTEXT_TOKENS \
    SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS
#define SPARK_LONG_CONTEXT_DEFAULT_SELECTED_TOKEN_CAPACITY \
    SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT
#define SPARK_LONG_CONTEXT_MAX_SELECTED_TOKEN_CAPACITY 4096u
#define SPARK_LONG_CONTEXT_DEFAULT_BLOCK_TOKEN_COUNT 16u
#define SPARK_LONG_CONTEXT_DEFAULT_RECENT_TOKEN_COUNT 1536u
#define SPARK_LONG_CONTEXT_DEFAULT_SINK_TOKEN_COUNT 128u
#define SPARK_LONG_CONTEXT_DEFAULT_STRIDE_SAMPLE_TOKEN_COUNT 384u
#define SPARK_LONG_CONTEXT_DEFAULT_LONG_CONTEXT_THRESHOLD 8192u
#define SPARK_LONG_CONTEXT_INVALID_TOKEN_ID 0xffffffffu

#define SPARK_LONG_CONTEXT_POLICY_MODE_BOUNDED_WINDOW 1u
#define SPARK_LONG_CONTEXT_POLICY_MODE_FULL_CONTEXT_SCAN 2u

#define SPARK_LONG_CONTEXT_POLICY_FLAG_REQUIRE_BOUNDED_DECODE 0x00000001u
#define SPARK_LONG_CONTEXT_POLICY_FLAG_INCLUDE_SINK_TOKENS 0x00000002u
#define SPARK_LONG_CONTEXT_POLICY_FLAG_INCLUDE_STRIDED_MIDDLE_TOKENS 0x00000004u
#define SPARK_LONG_CONTEXT_POLICY_FLAG_INCLUDE_RECENT_TOKENS 0x00000008u
#define SPARK_LONG_CONTEXT_POLICY_FLAG_FAIL_ON_CONTEXT_OVERFLOW 0x00000010u
#define SPARK_LONG_CONTEXT_POLICY_FLAG_ALLOW_FULL_CONTEXT_SCAN 0x00000020u
#define SPARK_LONG_CONTEXT_POLICY_DEFAULT_FLAGS \
    (SPARK_LONG_CONTEXT_POLICY_FLAG_REQUIRE_BOUNDED_DECODE | \
     SPARK_LONG_CONTEXT_POLICY_FLAG_INCLUDE_SINK_TOKENS | \
     SPARK_LONG_CONTEXT_POLICY_FLAG_INCLUDE_STRIDED_MIDDLE_TOKENS | \
     SPARK_LONG_CONTEXT_POLICY_FLAG_INCLUDE_RECENT_TOKENS | \
     SPARK_LONG_CONTEXT_POLICY_FLAG_FAIL_ON_CONTEXT_OVERFLOW)
#define SPARK_LONG_CONTEXT_POLICY_KNOWN_FLAGS \
    (SPARK_LONG_CONTEXT_POLICY_DEFAULT_FLAGS | \
     SPARK_LONG_CONTEXT_POLICY_FLAG_ALLOW_FULL_CONTEXT_SCAN)

#define SPARK_LONG_CONTEXT_DECODE_PLAN_FLAG_BOUNDED_SELECTION 0x00000001u
#define SPARK_LONG_CONTEXT_DECODE_PLAN_FLAG_FULL_CONTEXT_SCAN 0x00000002u
#define SPARK_LONG_CONTEXT_DECODE_PLAN_FLAG_LONG_CONTEXT 0x00000004u
#define SPARK_LONG_CONTEXT_DECODE_PLAN_FLAG_CONTEXT_TRUNCATED 0x00000008u
#define SPARK_LONG_CONTEXT_DECODE_PLAN_FLAG_SELECTION_PADDED 0x00000010u

#define SPARK_LONG_CONTEXT_PREFILL_PLAN_FLAG_CHUNKED_PREFILL 0x00000001u
#define SPARK_LONG_CONTEXT_PREFILL_PLAN_FLAG_LONG_CONTEXT 0x00000002u


typedef struct SparkLongContextPolicy
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t policy_mode;
    uint32_t policy_flags;
    uint32_t max_context_tokens;
    uint32_t selected_token_capacity;
    uint32_t block_token_count;
    uint32_t recent_token_count;
    uint32_t sink_token_count;
    uint32_t stride_sample_token_count;
    uint32_t maximum_decode_scan_token_count;
    uint32_t prefill_chunk_token_count;
    uint32_t long_context_threshold_token_count;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} SparkLongContextPolicy;

typedef struct SparkLongContextDecodePlan
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t context_token_count;
    uint32_t selected_token_count;
    uint32_t selected_token_capacity;
    uint32_t padded_token_count;
    uint32_t selected_block_count;
    uint32_t kv_block_token_count;
    uint32_t kv_block_count_for_context;
    uint32_t first_recent_token_index;
    uint32_t maximum_decode_scan_token_count;
    uint32_t sink_token_count;
    uint32_t stride_sample_token_count;
    uint32_t recent_token_count;
    uint32_t reserved0;
    uint64_t estimated_attention_token_reads;
    uint64_t avoided_full_scan_token_reads;
} SparkLongContextDecodePlan;

typedef struct SparkLongContextPrefillPlan
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t prompt_token_count;
    uint32_t max_prefill_tokens_per_step;
    uint32_t prefill_chunk_count;
    uint32_t final_chunk_token_count;
    uint32_t kv_block_token_count;
    uint32_t kv_block_count_for_prompt;
    uint32_t reserved0;
    uint64_t total_prompt_token_visits;
} SparkLongContextPrefillPlan;

void SparkLongContextInitializeDefaultPolicy(
    SparkLongContextPolicy *policy);

SparkStatus SparkLongContextValidatePolicy(
    const SparkLongContextPolicy *policy);

SparkStatus SparkLongContextBuildPrefillPlan(
    const SparkLongContextPolicy *policy,
    uint32_t prompt_token_count,
    uint32_t max_prefill_tokens_per_step,
    SparkLongContextPrefillPlan *prefill_plan);

SparkStatus SparkLongContextBuildDecodeSelection(
    const SparkLongContextPolicy *policy,
    uint32_t context_token_count,
    uint32_t *selected_token_indices,
    uint32_t selected_token_capacity,
    SparkLongContextDecodePlan *decode_plan);

SparkStatus SparkLongContextBuildDecodeSelectionForLaneBatch(
    const SparkLongContextPolicy *policy,
    const uint32_t *context_token_counts,
    uint32_t lane_count,
    uint32_t *selected_token_indices,
    uint32_t selected_token_stride,
    uint32_t selected_token_capacity,
    SparkLongContextDecodePlan *decode_plans,
    uint32_t decode_plan_capacity);

#ifdef __cplusplus
}
#endif

#endif
