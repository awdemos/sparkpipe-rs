#pragma once

#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_glm52_kv_cache.h"
#include "sparkpipe/spark_ring_node_context_builder.h"
#include "sparkpipe/spark_ring_work_control.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_CUDA_RESIDENT_IPC_ABI_VERSION 24u
#define SPARK_CUDA_RESIDENT_IPC_MAGIC 0x52445543u
#define SPARK_CUDA_RESIDENT_IPC_MAX_LANE_BLOCKS \
    (SPARK_GLM52_KV_CONTEXT_TOKENS / SPARK_GLM52_KV_BLOCK_TOKENS)
#define SPARK_CUDA_RESIDENT_IPC_HEADER_BYTES \
    ((uint32_t)sizeof(SparkCudaResidentIpcHeader))
#define SPARK_CUDA_RESIDENT_IPC_HELLO_BYTES \
    ((uint32_t)sizeof(SparkCudaResidentIpcHello))
#define SPARK_CUDA_RESIDENT_IPC_SUBMIT_WORK_BYTES \
    ((uint32_t)sizeof(SparkCudaResidentIpcSubmitWork))
#define SPARK_CUDA_RESIDENT_IPC_SUBMIT_WORK_PREFIX_BYTES \
	((uint32_t)offsetof(SparkCudaResidentIpcSubmitWork,work_packet))
#define SPARK_CUDA_RESIDENT_IPC_COMPLETION_BYTES \
    ((uint32_t)sizeof(SparkCudaResidentIpcCompletion))
#define SPARK_CUDA_RESIDENT_IPC_SUBMIT_RESULT_BYTES \
    ((uint32_t)sizeof(SparkCudaResidentIpcSubmitResult))
#define SPARK_CUDA_RESIDENT_IPC_QUERY_BYTES \
    ((uint32_t)sizeof(SparkCudaResidentIpcQuery))
#define SPARK_CUDA_RESIDENT_IPC_STATS_BYTES \
    ((uint32_t)sizeof(SparkCudaResidentIpcStats))
#define SPARK_CUDA_RESIDENT_IPC_SUBMIT_PREFILL_BYTES \
    ((uint32_t)sizeof(SparkCudaResidentIpcSubmitPrefill))
#define SPARK_CUDA_RESIDENT_IPC_SUBMIT_PREFILL_PREFIX_BYTES \
	((uint32_t)offsetof(SparkCudaResidentIpcSubmitPrefill,work_packet))
#define SPARK_CUDA_RESIDENT_IPC_SUBMIT_DECODE_BYTES \
    SPARK_CUDA_RESIDENT_IPC_SUBMIT_DECODE_HEADER_BYTES
#define SPARK_CUDA_RESIDENT_IPC_SUBMIT_DECODE_HEADER_BYTES \
    ((uint32_t)offsetof(SparkCudaResidentIpcSubmitDecode, \
        kv_physical_block_indices))
#define SPARK_CUDA_RESIDENT_IPC_MAX_DECODE_PAYLOAD_BYTES \
    (SPARK_CUDA_RESIDENT_IPC_SUBMIT_DECODE_HEADER_BYTES + \
     (SPARK_RING_WORK_CONTROL_MAX_LANE_COUNT * \
      SPARK_CUDA_RESIDENT_IPC_MAX_LANE_BLOCKS * (uint32_t)sizeof(uint32_t)))
#define SPARK_CUDA_RESIDENT_IPC_MAX_CONTROL_PAYLOAD_BYTES \
    SPARK_CUDA_RESIDENT_IPC_MAX_DECODE_PAYLOAD_BYTES

#define SPARK_CUDA_RESIDENT_IPC_KIND_HELLO 1u
#define SPARK_CUDA_RESIDENT_IPC_KIND_HELLO_ACK 2u
#define SPARK_CUDA_RESIDENT_IPC_KIND_SUBMIT_WORK 3u
#define SPARK_CUDA_RESIDENT_IPC_KIND_SUBMIT_RESULT 4u
#define SPARK_CUDA_RESIDENT_IPC_KIND_COMPLETION 5u
#define SPARK_CUDA_RESIDENT_IPC_KIND_QUERY 6u
#define SPARK_CUDA_RESIDENT_IPC_KIND_STATS 7u
#define SPARK_CUDA_RESIDENT_IPC_KIND_SHUTDOWN 8u
#define SPARK_CUDA_RESIDENT_IPC_KIND_ERROR 9u
#define SPARK_CUDA_RESIDENT_IPC_KIND_SUBMIT_PREFILL 10u
#define SPARK_CUDA_RESIDENT_IPC_KIND_SUBMIT_DECODE 11u

#define SPARK_CUDA_RESIDENT_IPC_SUBMIT_FLAG_INTERNAL_KV_DIRECTORY \
    0x00000001u
#define SPARK_CUDA_RESIDENT_IPC_SUBMIT_KNOWN_FLAGS \
    SPARK_CUDA_RESIDENT_IPC_SUBMIT_FLAG_INTERNAL_KV_DIRECTORY
#define SPARK_CUDA_RESIDENT_IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT \
    0x00000001u
#define SPARK_CUDA_RESIDENT_IPC_SUBMIT_WORK_KNOWN_FLAGS \
    SPARK_CUDA_RESIDENT_IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT

#define SPARK_CUDA_RESIDENT_IPC_STATE_EMPTY 0u
#define SPARK_CUDA_RESIDENT_IPC_STATE_LOADING 1u
#define SPARK_CUDA_RESIDENT_IPC_STATE_READY 2u
#define SPARK_CUDA_RESIDENT_IPC_STATE_DRAINING 3u
#define SPARK_CUDA_RESIDENT_IPC_STATE_FAILED 4u

#define SPARK_CUDA_RESIDENT_IPC_FLAG_DRIVER_RESIDENT 0x00000001u
#define SPARK_CUDA_RESIDENT_IPC_FLAG_BUILDER_RESIDENT 0x00000002u
#define SPARK_CUDA_RESIDENT_IPC_FLAG_TRANSPORT_RESIDENT 0x00000004u
#define SPARK_CUDA_RESIDENT_IPC_FLAG_CUDA_STATE_RESIDENT 0x00000008u

#define SPARK_CUDA_RESIDENT_IPC_ERROR_TEXT_BYTES 160u

typedef struct SparkCudaResidentIpcHeader
{
    uint32_t magic;
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t kind;
    uint32_t payload_bytes;
    uint32_t rank_index;
    uint64_t sequence_number;
} SparkCudaResidentIpcHeader;

typedef struct SparkCudaResidentIpcReader
{
    SparkCudaResidentIpcHeader header;
    uint32_t header_offset;
    uint32_t payload_offset;
    uint32_t header_ready;
} SparkCudaResidentIpcReader;

typedef struct SparkCudaResidentIpcHello
{
    uint32_t descriptor_bytes;
    uint32_t rank_index;
    uint32_t rank_count;
    uint32_t expected_cuda_generation;
    uint64_t control_generation;
    uint64_t process_id;
} SparkCudaResidentIpcHello;

typedef struct SparkCudaResidentIpcSubmitWork
{
    uint32_t descriptor_bytes;
    uint32_t flags;
    SparkRingWorkControlPacket work_packet;
} SparkCudaResidentIpcSubmitWork;

typedef struct SparkCudaResidentIpcCompletion
{
    uint32_t descriptor_bytes;
    uint32_t flags;
    SparkModelDriverCompletion completion;
	SparkGlm52DsparkDraftResult dspark_draft;
} SparkCudaResidentIpcCompletion;

#define SPARK_CUDA_RESIDENT_IPC_COMPLETION_FLAG_DSPARK_DRAFT 0x00000001u
#define SPARK_CUDA_RESIDENT_IPC_COMPLETION_KNOWN_FLAGS \
	SPARK_CUDA_RESIDENT_IPC_COMPLETION_FLAG_DSPARK_DRAFT

typedef struct SparkCudaResidentIpcQuery
{
    uint32_t descriptor_bytes;
    uint32_t reserved0;
} SparkCudaResidentIpcQuery;

typedef struct SparkCudaResidentIpcStats
{
    uint32_t descriptor_bytes;
    uint32_t state;
    uint32_t capability_flags;
    uint32_t rank_index;
    uint32_t max_active_sequence_count;
    uint32_t active_submission_count;
    uint32_t available_dispatch_slot_count;
    uint32_t private_queue_pressure;
    uint64_t submitted_count;
    uint64_t completed_count;
    uint64_t rejected_count;
    uint64_t resident_sequence_count;
    uint64_t resident_token_count;
    uint32_t kv_nvme_enabled;
    uint32_t kv_physical_block_capacity;
    uint32_t kv_logical_block_capacity;
    uint32_t kv_nvme_mode;
    uint64_t kv_logical_block_count;
    uint64_t kv_resident_block_count;
    uint64_t kv_swapped_block_count;
    uint64_t kv_nvme_record_bytes;
    uint64_t kv_nvme_store_count;
    uint64_t kv_nvme_load_count;
    uint64_t kv_nvme_write_bytes;
    uint64_t kv_nvme_read_bytes;
    uint64_t kv_nvme_synchronous_wait_count;
    uint64_t kv_nvme_batch_flush_count;
    uint64_t kv_nvme_maximum_batch_operation_count;
    uint64_t kv_resident_bytes_per_token;
    uint64_t kv_resident_pool_bytes;
    uint64_t kv_nvme_capacity_bytes;
    uint64_t kv_compact_selected_mla_working_set_bytes;
    uint32_t kv_nvme_batch_block_capacity;
    uint32_t kv_nvme_pending_store_count;
    uint32_t kv_nvme_pending_load_count;
    uint32_t kv_nvme_clean_evict_count;
    uint32_t work_queue_depth;
    uint32_t work_queue_capacity;
    uint32_t builder_pending_work;
    uint32_t resident_driver_inflight;
    uint64_t work_queue_accepted_count;
    uint64_t work_queue_submit_count;
    uint64_t work_queue_error_count;
    uint64_t asynchronous_submit_count;
    uint64_t asynchronous_completion_count;
    uint64_t asynchronous_failure_count;
    uint32_t logical_lane_capacity;
    uint32_t execution_row_capacity;
    uint32_t last_layer_major_logical_lane_count;
    uint32_t last_layer_major_rows_per_lane;
    uint32_t last_layer_major_execution_row_count;
    uint32_t moe_backend_kind;
    uint32_t moe_bound_layer_count;
    uint32_t moe_expected_layer_count;
    uint32_t fp8_scaled_gemm_bound_plan_count;
    uint32_t fp8_scaled_gemm_expected_plan_count;
    uint32_t model_quantization_mode;
    uint64_t layer_major_submit_count;
    uint64_t layer_major_completion_count;
    uint64_t layer_major_failure_count;
	uint64_t cuda_total_bytes;
	uint64_t cuda_initial_free_bytes;
	uint64_t cuda_current_free_bytes;
	uint64_t cuda_consumed_bytes;
	uint64_t cuda_builder_allocation_bytes;
	uint64_t cuda_largest_allocation_bytes;
	uint64_t host_mapped_allocation_bytes;
    uint64_t cuda_generation;
    uint64_t control_generation;
    char blocker[SPARK_CUDA_RESIDENT_IPC_ERROR_TEXT_BYTES];
} SparkCudaResidentIpcStats;

typedef struct SparkCudaResidentIpcSubmitResult
{
    uint32_t descriptor_bytes;
    uint32_t status;
    SparkCudaResidentIpcStats stats;
} SparkCudaResidentIpcSubmitResult;

typedef struct SparkCudaResidentIpcSubmitPrefill
{
    uint32_t descriptor_bytes;
    uint32_t request_flags;
    SparkRingWorkControlPacket work_packet;
} SparkCudaResidentIpcSubmitPrefill;

typedef struct SparkCudaResidentIpcDecodeLane
{
    uint64_t request_id;
    uint64_t request_generation;
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint32_t request_slot_index;
    uint32_t context_token_count;
    uint32_t input_token_id;
    uint32_t mtp_draft_token_budget;
    uint32_t speculative_token_count;
    uint8_t mtp_resolution_proposed_token_count;
    uint8_t mtp_resolution_accepted_token_count;
    uint16_t mtp_resolution_path_id;
    uint32_t kv_block_offset;
    uint32_t kv_block_count;
    uint32_t speculative_draft_token_ids[
        SPARK_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT];
} SparkCudaResidentIpcDecodeLane;

typedef struct SparkCudaResidentIpcSubmitDecode
{
    uint32_t descriptor_bytes;
    uint32_t highest_priority;
    uint32_t request_flags;
    uint32_t dispatch_kind;
    uint32_t lane_count;
    uint32_t active_sequence_count;
    uint32_t execution_batch_bucket;
    uint32_t speculative_token_count;
    uint32_t kv_block_token_count;
    uint32_t kv_block_index_count;
    uint32_t resident_flags;
    uint64_t control_generation;
    uint64_t request_generation;
    uint64_t step_generation;
    uint32_t step_chunk_index;
    uint32_t step_chunk_count;
    uint32_t transaction_phase;
    uint32_t reserved_transaction;
    SparkCudaResidentIpcDecodeLane
        lanes[SPARK_RING_WORK_CONTROL_MAX_LANE_COUNT];
    uint32_t kv_physical_block_indices[];
} SparkCudaResidentIpcSubmitDecode;

typedef union SparkCudaResidentIpcAnyPayload
{
    SparkCudaResidentIpcHello hello;
    SparkCudaResidentIpcSubmitWork submit_work;
    SparkCudaResidentIpcSubmitResult submit_result;
    SparkCudaResidentIpcCompletion completion;
    SparkCudaResidentIpcQuery query;
    SparkCudaResidentIpcStats stats;
    SparkCudaResidentIpcSubmitPrefill submit_prefill;
} SparkCudaResidentIpcAnyPayload;

void SparkCudaResidentIpcInitializeHeader(
    SparkCudaResidentIpcHeader *header,
    uint32_t kind,
    uint32_t rank_index,
    uint64_t sequence_number,
    uint32_t payload_bytes);
SparkStatus SparkCudaResidentIpcValidateHeader(
    const SparkCudaResidentIpcHeader *header,
    uint32_t expected_kind,
    uint32_t maximum_payload_bytes);
void SparkCudaResidentIpcReaderReset(
    SparkCudaResidentIpcReader *reader);
SparkStatus SparkCudaResidentIpcReadHeader(
    SparkCudaResidentIpcReader *reader,
    int32_t fd,
    uint32_t maximum_payload_bytes);
SparkStatus SparkCudaResidentIpcReadPayload(
    SparkCudaResidentIpcReader *reader,
    int32_t fd,
    uint8_t *payload,
    uint32_t payload_capacity);
uint32_t SparkCudaResidentIpcCalculateSubmitWorkBytes(
	const SparkRingWorkControlPacket *work_packet);
SparkStatus SparkCudaResidentIpcInitializeSubmitWork(
    SparkCudaResidentIpcSubmitWork *message,
    const SparkRingWorkControlPacket *work_packet,
    uint32_t flags);
SparkStatus SparkCudaResidentIpcValidateSubmitWork(
    const SparkCudaResidentIpcSubmitWork *message,
    uint32_t payload_bytes);
uint32_t SparkCudaResidentIpcCalculateSubmitPrefillBytes(
	const SparkRingWorkControlPacket *work_packet);
SparkStatus SparkCudaResidentIpcDecodePayloadBytes(
    uint32_t kv_block_index_count,
    uint32_t *payload_bytes_out);
SparkStatus SparkCudaResidentIpcValidateSubmitDecode(
    const SparkCudaResidentIpcSubmitDecode *message,
    uint32_t payload_bytes,
    uint32_t maximum_lane_count);

#ifdef __cplusplus
}
#endif
