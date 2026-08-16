#ifndef SPARKPIPE_SPARK_GLM52_SCHEDULER_H
#define SPARKPIPE_SPARK_GLM52_SCHEDULER_H

#include <stdint.h>

#include "sparkpipe/spark_prefix_cache.h"
#include "sparkpipe/spark_stage_plan.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SCHEDULER_ABI_VERSION 1u
#define SPARK_SCHEDULER_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkScheduler))
#define SPARK_SCHEDULER_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSchedulerConfiguration))
#define SPARK_SCHEDULER_REQUEST_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSchedulerRequest))
#define SPARK_SCHEDULER_DECISION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSchedulerDecision))
#define SPARK_SCHEDULER_BATCH_REQUEST_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSchedulerBatchRequest))
#define SPARK_SCHEDULER_PREFILL_BATCH_REQUEST_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSchedulerPrefillBatchRequest))
#define SPARK_SCHEDULER_PACKED_REQUEST_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSchedulerPackedRequest))
#define SPARK_SCHEDULER_PREFILL_BATCH_LANE_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSchedulerPrefillBatchLane))
#define SPARK_SCHEDULER_BATCH_DECISION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSchedulerBatchDecision))
#define SPARK_SCHEDULER_PREFILL_BATCH_DECISION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkSchedulerPrefillBatchDecision))
#define SPARK_SCHEDULER_MAX_SPARK_COUNT \
    SPARK_STAGE_PLAN_CURRENT_SPARK_COUNT
#define SPARK_SCHEDULER_MAX_PACKED_REQUEST_COUNT \
    SPARK_STAGE_PLAN_MAX_BATCH_BUCKET
#define SPARK_SCHEDULER_BATCH_REQUEST_CAPACITY_FACTOR 2u
#define SPARK_SCHEDULER_MAX_BATCH_REQUEST_COUNT \
    (SPARK_STAGE_PLAN_MAX_BATCH_BUCKET * \
     SPARK_SCHEDULER_BATCH_REQUEST_CAPACITY_FACTOR)
#define SPARK_SCHEDULER_DEFAULT_QUEUE_DEPTH_PER_SPARK 1u
#define SPARK_SCHEDULER_PREFILL_BLOCK_TOKENS 16u
#define SPARK_SCHEDULER_DEFAULT_MAX_PREFILL_TOKENS_PER_STEP \
    SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH
#define SPARK_SCHEDULER_MAX_CONTEXT_TOKENS \
    SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS
#define SPARK_SCHEDULER_KV_BLOCK_TABLE_CAPACITY \
    (SPARK_SCHEDULER_MAX_CONTEXT_TOKENS / \
     SPARK_SCHEDULER_PREFILL_BLOCK_TOKENS)

#define SPARK_SCHEDULER_CONFIGURATION_FLAG_CHUNKED_PREFILL 0x00000001u
#define SPARK_SCHEDULER_CONFIGURATION_FLAG_PREFIX_CACHE 0x00000002u
#define SPARK_SCHEDULER_CONFIGURATION_FLAG_CUDAGRAPH_PADDING 0x00000004u
#define SPARK_SCHEDULER_CONFIGURATION_FLAG_PREFILL_DECODE_INTERLEAVE 0x00000008u
#define SPARK_SCHEDULER_CONFIGURATION_FLAG_KV_CACHE_REQUIRED 0x00000010u
#define SPARK_SCHEDULER_CONFIGURATION_FLAG_MEASURED_DECODE_BUCKET_SELECTION 0x00000020u
#define SPARK_SCHEDULER_CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE 0x00000040u
#define SPARK_SCHEDULER_CONFIGURATION_DEFAULT_FLAGS \
    (SPARK_SCHEDULER_CONFIGURATION_FLAG_CHUNKED_PREFILL | \
     SPARK_SCHEDULER_CONFIGURATION_FLAG_PREFIX_CACHE | \
     SPARK_SCHEDULER_CONFIGURATION_FLAG_CUDAGRAPH_PADDING | \
     SPARK_SCHEDULER_CONFIGURATION_FLAG_PREFILL_DECODE_INTERLEAVE | \
     SPARK_SCHEDULER_CONFIGURATION_FLAG_KV_CACHE_REQUIRED | \
     SPARK_SCHEDULER_CONFIGURATION_FLAG_MEASURED_DECODE_BUCKET_SELECTION | \
     SPARK_SCHEDULER_CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE)
#define SPARK_SCHEDULER_CONFIGURATION_KNOWN_FLAGS \
    SPARK_SCHEDULER_CONFIGURATION_DEFAULT_FLAGS

#define SPARK_SCHEDULER_REQUEST_FLAG_DECODE 0x00000001u
#define SPARK_SCHEDULER_REQUEST_FLAG_PREFILL 0x00000002u
#define SPARK_SCHEDULER_REQUEST_KNOWN_FLAGS \
    (SPARK_SCHEDULER_REQUEST_FLAG_DECODE | \
     SPARK_SCHEDULER_REQUEST_FLAG_PREFILL)

#define SPARK_SCHEDULER_DECISION_FLAG_PREFILL_CHUNK 0x00000001u
#define SPARK_SCHEDULER_DECISION_FLAG_PREFILL_FINAL_CHUNK 0x00000002u
#define SPARK_SCHEDULER_DECISION_FLAG_PREFIX_CACHE_USED 0x00000004u
#define SPARK_SCHEDULER_DECISION_FLAG_CUDAGRAPH_PADDING 0x00000008u
#define SPARK_SCHEDULER_DECISION_FLAG_DECODE_STEP 0x00000010u
#define SPARK_SCHEDULER_DECISION_FLAG_PREFILL_STEP 0x00000020u
#define SPARK_SCHEDULER_DECISION_FLAG_PREFILL_RESERVED_DECODE_SLOT 0x00000040u
#define SPARK_SCHEDULER_DECISION_FLAG_DECODE_BYPASS_PREFILL 0x00000080u
#define SPARK_SCHEDULER_DECISION_FLAG_ADAPTIVE_DECODE_PACK 0x00000100u
#define SPARK_SCHEDULER_DECISION_FLAG_ADAPTIVE_PREFILL_PACK 0x00000200u
#define SPARK_SCHEDULER_DECISION_FLAG_MEASURED_DECODE_BUCKET 0x00000400u

#define SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_DECODE 0x00000001u
#define SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL 0x00000002u
#define SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL_CHUNK 0x00000004u
#define SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL_FINAL_CHUNK 0x00000008u
#define SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_CUDAGRAPH_PADDING 0x00000010u
#define SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_PREFILL_RESERVED_DECODE_SLOT 0x00000020u
#define SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_DECODE_BYPASS_PREFILL 0x00000040u
#define SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_ADAPTIVE_DECODE_PACK 0x00000080u
#define SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_ADAPTIVE_PREFILL_PACK 0x00000100u
#define SPARK_SCHEDULER_DISPATCH_STAGE_FLAG_MEASURED_DECODE_BUCKET 0x00000200u

#define SPARK_SCHEDULER_NO_PROMPT_LIMIT 0u

typedef struct SparkSchedulerConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t spark_count;
    uint32_t queue_depth_per_spark;
    uint32_t measured_profile_id;
    // The model's layer geometry - stage planning and cost profiles read
    // this instead of any compiled-in family.
    SparkStagePlanGeometry stage_geometry;
    // Used only when measured_profile_id is PROFILE_UNIFORM_ESTIMATED:
    // one per-layer cost estimate for the family, wiring-supplied.
    uint64_t estimated_layer_cost_ns;
    uint64_t estimated_final_stage_extra_cost_ns;
    uint32_t quantization_mode;
    uint32_t max_prefill_tokens_per_step;
    uint32_t prefix_cache_block_tokens;
    uint32_t configuration_flags;
    uint32_t reserved;
    SparkPrefixCache *prefix_cache;
} SparkSchedulerConfiguration;

typedef struct SparkSchedulerRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t active_sequence_count;
    uint32_t prompt_token_count;
    uint32_t flags;
    uint32_t computed_prompt_token_count;
    uint32_t cached_prefix_token_count;
    uint32_t max_scheduled_prompt_token_count;
    uint32_t reserved;
    uint64_t sequence_id;
    const uint32_t *prompt_token_ids;
} SparkSchedulerRequest;

typedef struct SparkSchedulerBatchRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t request_count;
    uint32_t reserved;
    const SparkSchedulerRequest *requests;
} SparkSchedulerBatchRequest;

typedef struct SparkSchedulerPrefillBatchRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t request_count;
    uint32_t reserved;
    const SparkSchedulerRequest *requests;
} SparkSchedulerPrefillBatchRequest;

typedef struct SparkSchedulerPackedRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t request_index;
    uint32_t active_sequence_offset;
    uint32_t active_sequence_count;
    uint32_t request_flags;
    uint32_t scheduled_token_count;
    uint32_t reserved;
    uint64_t total_scheduled_token_count;
} SparkSchedulerPackedRequest;

typedef struct SparkSchedulerDispatchStage
{
    uint32_t spark_index;
    uint32_t batch_bucket;
    uint32_t first_layer_index;
    uint32_t layer_count;
    uint32_t stage_flags;
    uint32_t dispatch_flags;
    uint32_t active_sequence_count;
    uint32_t graph_sequence_capacity;
    uint32_t graph_sequence_padding_count;
    uint32_t scheduled_prompt_token_offset;
    uint32_t scheduled_prompt_token_count;
    uint32_t cached_prefix_token_count;
    uint64_t estimated_service_time_ns;
} SparkSchedulerDispatchStage;

typedef struct SparkSchedulerDecision
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t accepted;
    uint32_t batch_bucket;
    uint32_t quantization_mode;
    uint32_t spark_count;
    uint32_t stage_count;
    uint32_t rejected_status;
    uint32_t decision_flags;
    uint32_t active_sequence_count;
    uint32_t graph_sequence_capacity;
    uint32_t graph_sequence_padding_count;
    uint32_t prompt_token_count;
    uint32_t computed_prompt_token_count;
    uint32_t cached_prefix_token_count;
    uint32_t prefix_cache_block_count;
    uint32_t scheduled_prompt_token_offset;
    uint32_t scheduled_prompt_token_count;
    uint32_t remaining_prompt_token_count_after_step;
    uint32_t prefill_block_count;
    uint32_t cache_commit_token_count_after_step;
    uint32_t kv_block_token_count;
    uint32_t kv_physical_block_count;
    uint32_t kv_cached_physical_block_count;
    uint32_t kv_pending_physical_block_count;
    uint32_t kv_block_table_token_count;
    uint64_t prefix_cache_reservation_epoch;
    uint64_t prefix_cache_parent_hash;
    uint64_t prefix_cache_result_hash;
    uint64_t sequence_id;
    const uint32_t *prompt_token_ids;
    uint64_t total_scheduled_token_count;
    uint64_t estimated_critical_path_ns;
    uint32_t kv_physical_block_indices[
        SPARK_SCHEDULER_KV_BLOCK_TABLE_CAPACITY];
    SparkStagePlan stage_plan;
    SparkSchedulerDispatchStage dispatch_stages[
        SPARK_SCHEDULER_MAX_SPARK_COUNT];
} SparkSchedulerDecision;

typedef struct SparkSchedulerPrefillBatchLane
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t request_index;
    uint32_t active_sequence_offset;
    uint32_t active_sequence_count;
    uint32_t prompt_token_count;
    uint32_t computed_prompt_token_count;
    uint32_t cached_prefix_token_count;
    uint32_t scheduled_prompt_token_offset;
    uint32_t scheduled_prompt_token_count;
    uint32_t remaining_prompt_token_count_after_step;
    uint32_t cache_commit_token_count_after_step;
    uint32_t prefix_cache_block_count;
    uint32_t kv_block_token_count;
    uint32_t kv_physical_block_count;
    uint32_t kv_cached_physical_block_count;
    uint32_t kv_pending_physical_block_count;
    uint32_t kv_block_table_token_count;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t prefix_cache_reservation_epoch;
    uint64_t prefix_cache_parent_hash;
    uint64_t prefix_cache_result_hash;
    uint64_t sequence_id;
    const uint32_t *prompt_token_ids;
} SparkSchedulerPrefillBatchLane;

typedef struct SparkSchedulerBatchDecision
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t accepted;
    uint32_t rejected_status;
    uint32_t source_request_count;
    uint32_t packed_request_count;
    uint32_t batch_bucket;
    uint32_t active_sequence_count;
    uint32_t graph_sequence_capacity;
    uint32_t graph_sequence_padding_count;
    uint32_t decision_flags;
    uint32_t reserved;
    uint64_t total_scheduled_token_count;
    uint64_t estimated_critical_path_ns;
    SparkSchedulerDecision stage_decision;
    SparkSchedulerPackedRequest packed_requests[
        SPARK_SCHEDULER_MAX_PACKED_REQUEST_COUNT];
} SparkSchedulerBatchDecision;

typedef struct SparkSchedulerPrefillBatchDecision
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t accepted;
    uint32_t rejected_status;
    uint32_t source_request_count;
    uint32_t packed_request_count;
    uint32_t batch_bucket;
    uint32_t active_sequence_count;
    uint32_t graph_sequence_capacity;
    uint32_t graph_sequence_padding_count;
    uint32_t decision_flags;
    uint32_t maximum_scheduled_prompt_token_count;
    uint64_t total_scheduled_token_count;
    uint64_t estimated_critical_path_ns;
    SparkSchedulerDecision stage_decision;
    SparkSchedulerPrefillBatchLane lanes[
        SPARK_SCHEDULER_MAX_PACKED_REQUEST_COUNT];
} SparkSchedulerPrefillBatchDecision;

typedef struct SparkScheduler
{
    SparkStagePlanGeometry stage_geometry;
    uint64_t estimated_layer_cost_ns;
    uint64_t estimated_final_stage_extra_cost_ns;
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t spark_count;
    uint32_t queue_depth_per_spark;
    uint32_t measured_profile_id;
    uint32_t quantization_mode;
    uint32_t max_prefill_tokens_per_step;
    uint32_t prefix_cache_block_tokens;
    uint32_t configuration_flags;
    uint32_t reserved;
    SparkPrefixCache *prefix_cache;
    uint32_t spark_inflight_counts[SPARK_SCHEDULER_MAX_SPARK_COUNT];
    uint32_t prefill_demand;
    uint64_t admitted_count;
    uint64_t rejected_count;
    uint64_t completed_count;
    uint64_t scheduled_decode_token_count;
    uint64_t scheduled_prefill_token_count;
    uint64_t prefix_cache_hit_token_count;
    uint64_t chunked_prefill_count;
    uint64_t interleaved_prefill_admission_count;
    uint64_t decode_bypass_admission_count;
    uint64_t adaptive_decode_pack_admission_count;
    uint64_t adaptive_decode_pack_request_count;
    uint64_t adaptive_decode_pack_padding_token_count;
    uint64_t adaptive_prefill_pack_admission_count;
    uint64_t adaptive_prefill_pack_request_count;
    uint64_t adaptive_prefill_pack_padding_token_count;
    uint64_t measured_decode_bucket_selection_count;
    uint64_t measured_decode_bucket_padding_token_count;
    uint64_t kv_block_reservation_count;
    uint64_t kv_block_reservation_token_count;
    uint64_t kv_block_cancel_count;
} SparkScheduler;

SparkStatus SparkSchedulerInitialize(
    SparkScheduler *scheduler,
    const SparkSchedulerConfiguration *configuration);

uint32_t SparkSchedulerSelectPipelineBatchWidth(
    const SparkScheduler *scheduler,
    uint32_t active_request_count,
    uint32_t batch_capacity);

SparkStatus SparkSchedulerEstimateDecodeWorkNs(
    const SparkScheduler *scheduler,
    uint32_t logical_sequence_count,
    uint32_t rows_per_sequence,
    uint32_t execution_row_capacity,
    uint64_t *estimated_work_ns_out);

void SparkSchedulerSetPrefillDemand(
    SparkScheduler *scheduler,
    uint32_t prefill_demand);
SparkStatus SparkSchedulerAdmit(
    SparkScheduler *scheduler,
    const SparkSchedulerRequest *request,
    SparkSchedulerDecision *decision);

SparkStatus SparkSchedulerComplete(
    SparkScheduler *scheduler,
    const SparkSchedulerDecision *decision);

SparkStatus SparkSchedulerAdmitDecodeBatch(
    SparkScheduler *scheduler,
    const SparkSchedulerBatchRequest *batch_request,
    SparkSchedulerBatchDecision *batch_decision);

SparkStatus SparkSchedulerAdmitPrefillBatch(
    SparkScheduler *scheduler,
    const SparkSchedulerPrefillBatchRequest *batch_request,
    SparkSchedulerPrefillBatchDecision *batch_decision);

SparkStatus SparkSchedulerCompleteDecodeBatch(
    SparkScheduler *scheduler,
    const SparkSchedulerBatchDecision *batch_decision);

SparkStatus SparkSchedulerCancelDecodeBatch(
    SparkScheduler *scheduler,
    const SparkSchedulerBatchDecision *batch_decision);

SparkStatus SparkSchedulerCompletePrefillBatch(
    SparkScheduler *scheduler,
    const SparkSchedulerPrefillBatchDecision *batch_decision);

SparkStatus SparkSchedulerCancel(
    SparkScheduler *scheduler,
    const SparkSchedulerDecision *decision);

SparkStatus SparkSchedulerCancelPrefillBatch(
    SparkScheduler *scheduler,
    const SparkSchedulerPrefillBatchDecision *batch_decision);

SparkStatus SparkSchedulerBuildKvBlockTable(
    SparkScheduler *scheduler,
    const SparkSchedulerDecision *decision,
    uint32_t *physical_block_indices,
    uint32_t physical_block_capacity,
    uint32_t *physical_block_count_out);

SparkStatus SparkSchedulerBuildPrefillBatchKvBlockTables(
    SparkScheduler *scheduler,
    const SparkSchedulerPrefillBatchDecision *batch_decision,
    uint32_t *physical_block_indices,
    uint32_t lane_stride,
    uint32_t lane_capacity,
    uint32_t *lane_physical_block_counts,
    uint32_t lane_count_capacity);

SparkStatus SparkSchedulerReleaseSequence(
    SparkScheduler *scheduler,
    uint64_t sequence_id);

#ifdef __cplusplus
}
#endif

#endif
