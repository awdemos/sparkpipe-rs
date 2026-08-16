#pragma once

#include <stdint.h>

#include "sparkpipe/spark_ring_runtime.h"
#include "sparkpipe/spark_ring_work_control.h"
#include "sparkpipe/spark_serving_engine.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_RING_NODE_CONTEXT_BUILDER_ABI_VERSION 18u
#define SPARK_RING_NODE_CONTEXT_BUILDER_DEFAULT_RESIDENT_SEQUENCE_COUNT \
	16384u
#define SPARK_RING_NODE_CONTEXT_BUILDER_MAX_PREFILL_TOKENS \
	SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH
#define SPARK_RING_NODE_CONTEXT_BUILDER_DEFAULT_NVME_BLOCK_CAPACITY \
	1048576u
#define SPARK_RING_NODE_CONTEXT_BUILDER_DEFAULT_NVME_BATCH_BLOCK_COUNT \
	32u
#define SPARK_RING_NODE_CONTEXT_BUILDER_MAX_NVME_BATCH_BLOCK_COUNT \
	128u
#define SPARK_RING_NODE_CONTEXT_BUILDER_INTERFACE_SYMBOL \
	"SparkGlm52RingNodeContextBuilderGetInterface"
#define SPARK_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkRingNodeContextBuilderConfiguration))
#define SPARK_RING_NODE_CONTEXT_BUILDER_RESULT_BYTES \
	((uint32_t)sizeof(SparkRingNodeContextBuilderResult))
#define SPARK_RING_NODE_CONTEXT_BUILDER_INTERFACE_BYTES \
	((uint32_t)sizeof(SparkRingNodeContextBuilderInterface))
#define SPARK_RING_NODE_CONTEXT_BUILDER_KV_STATS_BYTES \
	((uint32_t)sizeof(SparkRingNodeContextBuilderKvStats))
#define SPARK_RING_NODE_CONTEXT_BUILDER_MOE_BACKEND_NONE \
	SPARK_RING_RUNTIME_MOE_BACKEND_NONE
#define SPARK_RING_NODE_CONTEXT_BUILDER_MOE_BACKEND_FP8_FLASHINFER_GROUPED \
	SPARK_RING_RUNTIME_MOE_BACKEND_FP8_FLASHINFER_GROUPED
#define SPARK_RING_NODE_CONTEXT_BUILDER_MOE_BACKEND_NVFP4_B12X \
	SPARK_RING_RUNTIME_MOE_BACKEND_NVFP4_B12X
#define SPARK_RING_NODE_CONTEXT_BUILDER_NVME_MODE_DISABLED 0u
#define SPARK_RING_NODE_CONTEXT_BUILDER_NVME_MODE_SYNCHRONOUS_FULL_HISTORY 1u
#define SPARK_RING_NODE_CONTEXT_BUILDER_NVME_MODE_BATCHED_COHORT_JIT 2u
#define SPARK_RING_NODE_CONTEXT_BUILDER_NVME_MODE_ASYNC_SELECTED_JIT 3u

#define SPARK_RING_NODE_CONTEXT_BUILDER_CAP_RESIDENT_MOE_PACKS 0x00000001u
#define SPARK_RING_NODE_CONTEXT_BUILDER_CAP_STAGE_SLICE 0x00000002u
#define SPARK_RING_NODE_CONTEXT_BUILDER_CAP_RANK0_TOKEN_INPUT 0x00000004u
#define SPARK_RING_NODE_CONTEXT_BUILDER_CAP_RANK_WORK_DISPATCH 0x00000008u
#define SPARK_RING_NODE_CONTEXT_BUILDER_CAP_DSPARK_DRAFT 0x00000010u
#define SPARK_RING_NODE_CONTEXT_BUILDER_CAP_MTP_DRAFT 0x00000020u
#define SPARK_RING_NODE_CONTEXT_BUILDER_CAP_NVME_KV 0x00000040u
#define SPARK_RING_NODE_CONTEXT_BUILDER_CAP_ASYNC_WORK 0x00000080u
#define SPARK_RING_NODE_CONTEXT_BUILDER_CAP_LAYER_MAJOR_MTP_VERIFY 0x00000100u
#define SPARK_RING_NODE_CONTEXT_BUILDER_CAP_CONTROL_GENERATION_RESET 0x00000200u
#define SPARK_RING_NODE_CONTEXT_BUILDER_CAP_KV_LOOKAHEAD_PREFETCH 0x00000400u
#define SPARK_RING_NODE_CONTEXT_BUILDER_CAP_MOONCAKE_KV 0x00000800u
#define SPARK_RING_NODE_CONTEXT_BUILDER_REQUIRED_PRODUCTION_CAPS \
	(SPARK_RING_NODE_CONTEXT_BUILDER_CAP_RESIDENT_MOE_PACKS | \
	 SPARK_RING_NODE_CONTEXT_BUILDER_CAP_STAGE_SLICE | \
	 SPARK_RING_NODE_CONTEXT_BUILDER_CAP_RANK0_TOKEN_INPUT | \
	 SPARK_RING_NODE_CONTEXT_BUILDER_CAP_RANK_WORK_DISPATCH | \
	 SPARK_RING_NODE_CONTEXT_BUILDER_CAP_DSPARK_DRAFT | \
	 SPARK_RING_NODE_CONTEXT_BUILDER_CAP_MTP_DRAFT | \
	 SPARK_RING_NODE_CONTEXT_BUILDER_CAP_NVME_KV | \
	 SPARK_RING_NODE_CONTEXT_BUILDER_CAP_ASYNC_WORK | \
	 SPARK_RING_NODE_CONTEXT_BUILDER_CAP_LAYER_MAJOR_MTP_VERIFY | \
	 SPARK_RING_NODE_CONTEXT_BUILDER_CAP_CONTROL_GENERATION_RESET | \
	 SPARK_RING_NODE_CONTEXT_BUILDER_CAP_KV_LOOKAHEAD_PREFETCH | \
	 SPARK_RING_NODE_CONTEXT_BUILDER_CAP_MOONCAKE_KV)

#define SPARK_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_DSPARK \
	0x00000001u
#define SPARK_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_MTP \
	0x00000002u
#define SPARK_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_NVME_KV \
	0x00000004u
#define SPARK_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_MOONCAKE_KV \
	0x00000008u
#define SPARK_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_KNOWN_FLAGS \
	(SPARK_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_DSPARK | \
	 SPARK_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_MTP | \
	 SPARK_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_NVME_KV | \
	 SPARK_RING_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_MOONCAKE_KV)

typedef struct SparkRingNodeContextBuilderConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t rank_index;
	uint32_t max_active_sequence_count;
	uint32_t port_base;
	uint32_t kv_pool_token_capacity;
	uint32_t maximum_resident_sequence_count;
	const char *moe_pack_root;
	const char *stagepack_root;
	const char *embedding_pack_path;
	const char *node_target;
	const char *dspark_manifest_path;
	const char *dspark_config_path;
	const char *dspark_safetensors_path;
	const char *kv_nvme_path;
	const char *kv_store_module_path;
	const char *kv_store_service_address;
	const char *kv_store_ipc_socket_path;
	uint32_t dspark_maximum_lane_count;
	uint32_t dspark_maximum_context_token_count;
	uint32_t kv_nvme_block_capacity;
	uint32_t kv_nvme_batch_block_count;
	uint32_t kv_store_block_capacity;
	uint32_t kv_store_batch_block_count;
	uint32_t kv_store_worker_count;
	uint32_t kv_store_lookahead_packet_count;
	uint64_t kv_store_model_fingerprint;
	uint64_t kv_store_layout_fingerprint;
	uint64_t kv_store_client_memory_pool_bytes;
	uint64_t kv_store_local_buffer_bytes;
	const SparkRingRuntimeRankPlan *rank_plan;
} SparkRingNodeContextBuilderConfiguration;

typedef struct SparkRingNodeContextBuilderResult
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t rank_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t hidden_dimension;
	uint32_t vocabulary_size;
	void *node_context;
	const void *embedding_weight_bf16;
	void *private_state;
} SparkRingNodeContextBuilderResult;

typedef struct SparkRingNodeContextBuilderKvStats
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t nvme_enabled;
	uint32_t nvme_mode;
	uint32_t physical_block_capacity;
	uint32_t logical_block_capacity;
	uint32_t logical_block_count;
	uint32_t resident_block_count;
	uint32_t swapped_block_count;
	uint64_t nvme_record_bytes;
	uint64_t nvme_store_count;
	uint64_t nvme_load_count;
	uint64_t nvme_write_bytes;
	uint64_t nvme_read_bytes;
	uint64_t nvme_synchronous_wait_count;
	uint64_t nvme_batch_flush_count;
	uint64_t nvme_maximum_batch_operation_count;
	uint64_t resident_bytes_per_token;
	uint64_t resident_pool_bytes;
	uint64_t nvme_capacity_bytes;
	uint64_t compact_selected_mla_working_set_bytes;
	uint32_t nvme_batch_block_capacity;
	uint32_t nvme_pending_store_count;
	uint32_t nvme_pending_load_count;
	uint32_t nvme_clean_evict_count;
	uint32_t pending_work_active;
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
	uint64_t asynchronous_submit_count;
	uint64_t asynchronous_completion_count;
	uint64_t asynchronous_failure_count;
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
} SparkRingNodeContextBuilderKvStats;

typedef SparkStatus (*SparkRingNodeContextBuilderInitializeFunction)(
	const SparkRingNodeContextBuilderConfiguration *configuration,
	void **builder_state);
typedef void (*SparkRingNodeContextBuilderDestroyFunction)(
	void *builder_state);
typedef SparkStatus (*SparkRingNodeContextBuilderBuildFunction)(
	void *builder_state,
	SparkRingNodeContextBuilderResult *result);
typedef void (*SparkRingNodeContextBuilderDestroyResultFunction)(
	void *builder_state,
	SparkRingNodeContextBuilderResult *result);
typedef SparkStatus (*SparkRingNodeContextBuilderAttachDriverFunction)(
	void *builder_state,
	const SparkModelDriverInterface *driver_interface,
	void *driver_instance,
	const SparkModelDriverProgramDescriptor *program,
	SparkHiddenTransportSession *output_transport_session);
typedef SparkStatus (*SparkRingNodeContextBuilderIdlePumpFunction)(
	void *idle_pump_context);
typedef SparkStatus (*SparkRingNodeContextBuilderPrefillFunction)(
	void *builder_state,
	const SparkPromptPipelinePrefillDispatch *prefill_dispatch,
	SparkRingNodeContextBuilderIdlePumpFunction idle_pump_function,
	void *idle_pump_context);
typedef SparkStatus (*SparkRingNodeContextBuilderDecodeFunction)(
	void *builder_state,
	const SparkServingDecodeDispatch *decode_dispatch,
	SparkServingDecodeResult *decode_result);
typedef SparkStatus (*SparkRingNodeContextBuilderSubmitWorkFunction)(
	void *builder_state,
	const SparkRingWorkControlPacket *work_packet,
	SparkHiddenTransportSession *input_transport_session,
	SparkHiddenTransportSession *output_transport_session,
	SparkModelDriverCompletionFunction completion_function,
	void *completion_context);
typedef SparkStatus (*SparkRingNodeContextBuilderPrefetchWorkFunction)(
	void *builder_state,
	const SparkRingWorkControlPacket *work_packets,
	uint32_t work_packet_count);
typedef SparkStatus (*SparkRingNodeContextBuilderProgressFunction)(
	void *builder_state);
typedef SparkStatus (*SparkRingNodeContextBuilderTakeDsparkDraftFunction)(
	void *builder_state,
	SparkGlm52DsparkDraftResult *draft_result);
typedef SparkStatus (*SparkRingNodeContextBuilderGetKvStatsFunction)(
	void *builder_state,
	SparkRingNodeContextBuilderKvStats *stats);
typedef SparkStatus (*SparkRingNodeContextBuilderResetControlGenerationFunction)(
	void *builder_state,
	uint64_t control_generation);

typedef struct SparkRingNodeContextBuilderInterface
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t capability_flags;
	uint32_t reserved0;
	SparkRingNodeContextBuilderInitializeFunction initialize;
	SparkRingNodeContextBuilderDestroyFunction destroy;
	SparkRingNodeContextBuilderBuildFunction build;
	SparkRingNodeContextBuilderDestroyResultFunction destroy_result;
	SparkRingNodeContextBuilderAttachDriverFunction attach_driver;
	SparkRingNodeContextBuilderPrefillFunction prefill;
	SparkRingNodeContextBuilderDecodeFunction decode;
	SparkRingNodeContextBuilderSubmitWorkFunction submit_work;
	SparkRingNodeContextBuilderPrefetchWorkFunction prefetch_work;
	SparkRingNodeContextBuilderProgressFunction progress;
	SparkRingNodeContextBuilderTakeDsparkDraftFunction take_dspark_draft;
	SparkRingNodeContextBuilderGetKvStatsFunction get_kv_stats;
	SparkRingNodeContextBuilderResetControlGenerationFunction
		reset_control_generation;
} SparkRingNodeContextBuilderInterface;

typedef const SparkRingNodeContextBuilderInterface *(
	*SparkGlm52RingNodeContextBuilderGetInterfaceFunction)(void);

typedef struct SparkRingNodeContextBuilderDynamicLibrary
{
	void *dynamic_library;
	SparkRingNodeContextBuilderInterface builder_interface;
} SparkRingNodeContextBuilderDynamicLibrary;

SparkStatus SparkRingNodeContextBuilderValidateInterface(
	const SparkRingNodeContextBuilderInterface *builder_interface,
	uint32_t required_capability_flags);
SparkStatus SparkRingNodeContextBuilderValidateResult(
	const SparkRingNodeContextBuilderResult *result,
	const SparkRingRuntimeRankPlan *rank_plan);
SparkStatus SparkRingNodeContextBuilderLoadInterfaceFromSharedObject(
	const char *shared_object_path,
	uint32_t required_capability_flags,
	SparkRingNodeContextBuilderDynamicLibrary *library);
void SparkRingNodeContextBuilderUnloadInterface(
	SparkRingNodeContextBuilderDynamicLibrary *library);

#ifdef __cplusplus
}
#endif
