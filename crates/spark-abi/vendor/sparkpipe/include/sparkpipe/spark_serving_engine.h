#ifndef SPARKPIPE_SPARK_GLM52_SERVING_ENGINE_H
#define SPARKPIPE_SPARK_GLM52_SERVING_ENGINE_H

#include <stdint.h>

#include "sparkpipe/spark_prompt_pipeline.h"
#include "sparkpipe/spark_request_api.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SERVING_ENGINE_ABI_VERSION 4u
#define SPARK_SERVING_ENGINE_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServingEngineConfiguration))
#define SPARK_SERVING_ENGINE_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServingEngine))
#define SPARK_SERVING_SUBMIT_TEXT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServingSubmitTextRequest))
#define SPARK_SERVING_SUBMIT_TOKENS_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServingSubmitTokenIdsRequest))
#define SPARK_SERVING_SUBMIT_RESULT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServingSubmitResult))
#define SPARK_SERVING_DECODE_DISPATCH_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServingDecodeDispatch))
#define SPARK_SERVING_DECODE_RESULT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServingDecodeResult))
#define SPARK_SERVING_REQUEST_RECORD_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServingRequestRecord))
#define SPARK_SERVING_EVENT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServingEvent))
#define SPARK_SERVING_STATS_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServingStats))

#define SPARK_SERVING_MAX_STOP_TOKEN_IDS 8u
#define SPARK_SERVING_MAX_DECODE_TOKENS_PER_LANE \
    (SPARK_REQUEST_MODEL_MAX_SPECULATIVE_TOKENS + 1u)
#define SPARK_SERVING_DEFAULT_OUTPUT_TOKEN_BUDGET 1024u
#define SPARK_SERVING_DEFAULT_MAX_CONTEXT_TOKENS \
    SPARK_SCHEDULER_MAX_CONTEXT_TOKENS
#define SPARK_SERVING_DEFAULT_MAX_PUMP_STEPS 256u
#define SPARK_SERVING_DEFAULT_REQUEST_ID_BASE 1000000000ull

#define SPARK_SERVING_ENGINE_FLAG_REQUIRE_PRODUCTION_RUNTIME_CONTRACT \
    0x00000001u
#define SPARK_SERVING_ENGINE_FLAG_AUTO_RELEASE_COMPLETED_REQUESTS \
    0x00000002u
#define SPARK_SERVING_ENGINE_FLAG_CLAMP_BUDGET_TO_CONTEXT \
    0x00000004u
#define SPARK_SERVING_ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE \
    0x00000008u
#define SPARK_SERVING_ENGINE_DEFAULT_FLAGS \
    (SPARK_SERVING_ENGINE_FLAG_REQUIRE_PRODUCTION_RUNTIME_CONTRACT | \
     SPARK_SERVING_ENGINE_FLAG_AUTO_RELEASE_COMPLETED_REQUESTS | \
     SPARK_SERVING_ENGINE_FLAG_CLAMP_BUDGET_TO_CONTEXT)
#define SPARK_SERVING_ENGINE_KNOWN_FLAGS \
    (SPARK_SERVING_ENGINE_DEFAULT_FLAGS | \
     SPARK_SERVING_ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE)

#define SPARK_SERVING_SUBMIT_FLAG_REALTIME \
    SPARK_REQUEST_API_REQUEST_FLAG_REALTIME
#define SPARK_SERVING_SUBMIT_FLAG_DISABLE_SPECULATION \
    SPARK_REQUEST_API_REQUEST_FLAG_DISABLE_SPECULATION
#define SPARK_SERVING_SUBMIT_KNOWN_FLAGS \
    (SPARK_SERVING_SUBMIT_FLAG_REALTIME | \
     SPARK_SERVING_SUBMIT_FLAG_DISABLE_SPECULATION)

#define SPARK_SERVING_RUNTIME_CONTRACT_FLAG_PREFILL_ACCEPTS_BULK_TOKEN_WINDOWS \
    0x00000001u
#define SPARK_SERVING_RUNTIME_CONTRACT_FLAG_PREFILL_WRITES_RESIDENT_KV \
    0x00000002u
#define SPARK_SERVING_RUNTIME_CONTRACT_FLAG_DECODE_CONSUMES_RESIDENT_KV \
    0x00000004u
#define SPARK_SERVING_RUNTIME_CONTRACT_FLAG_DECODE_RETURNS_TOKEN_IDS \
    0x00000008u
#define SPARK_SERVING_RUNTIME_CONTRACT_FLAG_USES_REQUEST_KV_BLOCK_TABLES \
    0x00000010u
#define SPARK_SERVING_RUNTIME_CONTRACT_FLAG_INTERNAL_BATCHING_ENABLED \
    0x00000020u
#define SPARK_SERVING_RUNTIME_CONTRACT_FLAG_JIT_KV_PREFETCH_CONNECTED \
    0x00000040u
#define SPARK_SERVING_RUNTIME_CONTRACT_FLAG_OVERLAPPED_STAGING_READY \
    0x00000080u
#define SPARK_SERVING_RUNTIME_CONTRACT_FLAG_OFFICIAL_DSA_INDEXSHARE \
    0x00000100u
#define SPARK_SERVING_RUNTIME_CONTRACT_FLAG_BOUNDED_LONG_CONTEXT_ATTENTION \
    0x00000200u
#define SPARK_SERVING_RUNTIME_CONTRACT_FLAG_INDEXSHARE_STAGE_BOUNDARY_STATE \
    0x00000400u
#define SPARK_SERVING_RUNTIME_CONTRACT_FLAG_MLA_COMPRESSED_KV_CACHE \
    0x00000800u
#define SPARK_SERVING_RUNTIME_CONTRACT_FLAG_TAIL_WINDOW_VALIDATION_ONLY \
    0x80000000u
#define SPARK_SERVING_RUNTIME_CONTRACT_PRODUCTION_REQUIRED_FLAGS \
    (SPARK_SERVING_RUNTIME_CONTRACT_FLAG_PREFILL_ACCEPTS_BULK_TOKEN_WINDOWS | \
     SPARK_SERVING_RUNTIME_CONTRACT_FLAG_PREFILL_WRITES_RESIDENT_KV | \
     SPARK_SERVING_RUNTIME_CONTRACT_FLAG_DECODE_CONSUMES_RESIDENT_KV | \
     SPARK_SERVING_RUNTIME_CONTRACT_FLAG_DECODE_RETURNS_TOKEN_IDS | \
     SPARK_SERVING_RUNTIME_CONTRACT_FLAG_USES_REQUEST_KV_BLOCK_TABLES | \
     SPARK_SERVING_RUNTIME_CONTRACT_FLAG_INTERNAL_BATCHING_ENABLED | \
     SPARK_SERVING_RUNTIME_CONTRACT_FLAG_OFFICIAL_DSA_INDEXSHARE | \
     SPARK_SERVING_RUNTIME_CONTRACT_FLAG_BOUNDED_LONG_CONTEXT_ATTENTION | \
     SPARK_SERVING_RUNTIME_CONTRACT_FLAG_INDEXSHARE_STAGE_BOUNDARY_STATE | \
     SPARK_SERVING_RUNTIME_CONTRACT_FLAG_MLA_COMPRESSED_KV_CACHE)
#define SPARK_SERVING_RUNTIME_CONTRACT_CURRENT_IMPLEMENTED_FLAGS \
    (SPARK_SERVING_RUNTIME_CONTRACT_FLAG_PREFILL_WRITES_RESIDENT_KV | \
     SPARK_SERVING_RUNTIME_CONTRACT_FLAG_DECODE_CONSUMES_RESIDENT_KV | \
     SPARK_SERVING_RUNTIME_CONTRACT_FLAG_DECODE_RETURNS_TOKEN_IDS | \
     SPARK_SERVING_RUNTIME_CONTRACT_FLAG_USES_REQUEST_KV_BLOCK_TABLES)

#define SPARK_SERVING_EVENT_KIND_NONE 0u
#define SPARK_SERVING_EVENT_KIND_REQUEST_ACCEPTED 1u
#define SPARK_SERVING_EVENT_KIND_PREFILL_PROGRESS 2u
#define SPARK_SERVING_EVENT_KIND_TOKEN 3u
#define SPARK_SERVING_EVENT_KIND_REQUEST_COMPLETED 4u
#define SPARK_SERVING_EVENT_KIND_REQUEST_CANCELLED 5u
#define SPARK_SERVING_EVENT_KIND_ERROR 6u
#define SPARK_SERVING_EVENT_KIND_BACKPRESSURE 7u

#define SPARK_SERVING_DECODE_RESULT_FLAG_FINISH_REQUEST 0x00000001u
#define SPARK_SERVING_DECODE_RESULT_FLAG_TOKEN_STREAM_SUPPRESSED 0x00000002u

#define SPARK_SERVING_REQUEST_RECORD_STATE_FREE 0u
#define SPARK_SERVING_REQUEST_RECORD_STATE_SUBMITTED 1u
#define SPARK_SERVING_REQUEST_RECORD_STATE_COMPLETED 2u
#define SPARK_SERVING_REQUEST_RECORD_STATE_CANCELLED 3u

#define SPARK_SERVING_PUMP_FLAG_STOP_AFTER_ONE_DISPATCH 0x00000001u
#define SPARK_SERVING_PUMP_KNOWN_FLAGS \
    SPARK_SERVING_PUMP_FLAG_STOP_AFTER_ONE_DISPATCH
#define SPARK_SERVING_RECORD_HASH_SLOTS 4096u
#define SPARK_SERVING_NO_RECORD_SLOT UINT32_MAX

typedef uint64_t SparkServingRequestHandle;

typedef struct SparkServingRequestRecord
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t state;
    uint32_t flags;
    uint32_t prompt_token_count;
    uint32_t token_capacity;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t streamed_decode_token_count;
    uint32_t reserved0;
    uint64_t request_id;
    uint64_t sequence_id;
    SparkServingRequestHandle request_handle;
    uint32_t handle_hash_next;
    uint32_t free_record_next;
    uint32_t *token_ids;
} SparkServingRequestRecord;

typedef struct SparkServingEvent
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t kind;
    uint32_t flags;
    uint32_t status;
    uint32_t token_id;
    uint32_t token_index;
    uint32_t token_count;
    uint32_t prompt_token_offset;
    uint32_t prompt_token_count;
    uint32_t dispatch_kind;
    uint32_t dispatch_flags;
    uint64_t request_id;
    uint64_t sequence_id;
    SparkServingRequestHandle request_handle;
} SparkServingEvent;

typedef struct SparkServingSubmitTokenIdsRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t priority;
    uint32_t token_count;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t max_prefill_tokens_per_step;
    uint64_t request_id;
    uint64_t sequence_id;
    const uint32_t *token_ids;
} SparkServingSubmitTokenIdsRequest;

typedef struct SparkServingSubmitTextRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t priority;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t max_prefill_tokens_per_step;
    uint32_t tokenizer_encode_flags;
    uint64_t request_id;
    uint64_t sequence_id;
    const char *text;
    uint32_t text_bytes;
} SparkServingSubmitTextRequest;

typedef struct SparkServingSubmitResult
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t prompt_token_count;
    uint32_t required_token_capacity;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint64_t request_id;
    uint64_t sequence_id;
    SparkServingRequestHandle request_handle;
} SparkServingSubmitResult;

typedef struct SparkServingDecodeDispatch
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t dispatch_kind;
    uint32_t request_count;
    uint32_t active_sequence_count;
    uint32_t speculative_token_index;
    uint32_t reserved1;
    uint32_t reserved2;
    const SparkRequestApiDispatch *request_dispatch;
    const SparkKvBlockTableView *kv_block_table_view;
    const SparkRequestApiDecodeDispatchView *decode_view;
    uint32_t input_token_ids[SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint32_t speculative_token_count;
    uint32_t speculative_draft_token_ids[SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT][
        SPARK_REQUEST_MODEL_MAX_SPECULATIVE_TOKENS];
} SparkServingDecodeDispatch;

typedef struct SparkServingDecodeResult
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t lane_count;
    uint32_t token_stride;
    uint32_t token_counts[SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint32_t lane_flags[SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint32_t token_ids[SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT][
        SPARK_SERVING_MAX_DECODE_TOKENS_PER_LANE];
    uint32_t draft_token_counts[
        SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT];
    uint32_t draft_token_ids[SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT][
        SPARK_REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT];
} SparkServingDecodeResult;

typedef SparkStatus (*SparkServingPrefillFunction)(
    void *context,
    const SparkPromptPipelinePrefillDispatch *prefill_dispatch);

typedef SparkStatus (*SparkServingDecodeFunction)(
    void *context,
    const SparkServingDecodeDispatch *decode_dispatch,
    SparkServingDecodeResult *decode_result);
typedef SparkStatus (*SparkServingReleaseSequenceFunction)(
    void *context,
    uint64_t request_id,
    uint64_t request_generation,
    uint64_t sequence_id,
    uint32_t token_count);

typedef struct SparkServingStats
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t live_request_count;
    uint32_t queued_request_count;
    uint32_t completed_request_count;
    uint32_t cancelled_request_count;
    uint32_t event_count;
    uint32_t event_capacity;
    uint32_t dropped_event_count;
    uint32_t last_status;
    uint32_t maximum_prefill_active_sequence_count;
    uint32_t maximum_prefill_lane_count;
    uint32_t maximum_decode_active_sequence_count;
    uint32_t maximum_decode_lane_count;
    uint64_t submitted_request_count;
    uint64_t accepted_request_count;
    uint64_t prefill_dispatch_count;
    uint64_t prefill_batch_dispatch_count;
    uint64_t prefill_token_count;
    uint64_t decode_dispatch_count;
    uint64_t decoded_token_count;
    uint64_t mtp_draft_token_count;
    uint64_t mtp_verify_dispatch_count;
    uint64_t mtp_draft_ready_count;
    uint64_t mtp_accepted_draft_token_count;
    uint64_t mtp_committed_token_count;
    uint64_t mtp_rejected_token_count;
    uint64_t completed_stream_count;
    uint64_t jit_prefetch_dispatch_count;
    uint64_t jit_prefetch_block_count;
    uint64_t async_jit_prefetch_start_count;
    uint64_t async_jit_prefetch_poll_count;
    uint64_t async_jit_prefetch_completion_count;
    uint64_t prefix_family_dispatch_count;
    uint64_t prefix_family_member_count;
    uint64_t prefix_family_saved_prompt_token_count;
} SparkServingStats;

typedef struct SparkServingEngineConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t runtime_contract_flags;
    uint32_t default_thinking_token_budget;
    uint32_t default_output_token_budget;
    uint32_t default_max_prefill_tokens_per_step;
    uint32_t max_context_tokens;
    uint64_t request_id_base;
    SparkRequestApi *request_api;
    const SparkTokenizer *tokenizer;
    SparkServingRequestRecord *request_records;
    uint32_t request_record_capacity;
    uint32_t *request_token_storage;
    uint32_t request_token_stride;
    SparkServingEvent *event_ring;
    uint32_t event_ring_capacity;
    uint32_t *host_prefill_token_ids;
    uint32_t host_prefill_token_stride;
    uint32_t host_prefill_lane_capacity;
    uint32_t *host_physical_block_indices;
    const uint32_t *execution_physical_block_indices;
    uint32_t kv_block_lane_stride;
    uint32_t kv_block_lane_capacity;
    uint32_t *lane_physical_block_counts;
    uint32_t lane_count_capacity;
    SparkServingPrefillFunction prefill_function;
    SparkServingDecodeFunction decode_function;
    SparkServingReleaseSequenceFunction release_sequence_function;
    void *callback_context;
    const uint32_t *stop_token_ids;
    uint32_t stop_token_id_count;
    uint32_t reserved0;
} SparkServingEngineConfiguration;

typedef struct SparkServingEngine
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t runtime_contract_flags;
    uint32_t default_thinking_token_budget;
    uint32_t default_output_token_budget;
    uint32_t default_max_prefill_tokens_per_step;
    uint32_t max_context_tokens;
    uint64_t next_generated_request_id;
    SparkRequestApi *request_api;
    const SparkTokenizer *tokenizer;
    SparkTokenizerWorkspace tokenizer_workspace;
    SparkServingRequestRecord *request_records;
    uint32_t request_record_capacity;
    uint32_t free_record_head;
    uint32_t *request_token_storage;
    uint32_t request_token_stride;
    SparkServingEvent *event_ring;
    uint32_t event_ring_capacity;
    uint32_t event_read_index;
    uint32_t event_write_index;
    uint32_t event_count;
    uint32_t dropped_event_count;
    uint32_t *host_prefill_token_ids;
    uint32_t host_prefill_token_stride;
    uint32_t host_prefill_lane_capacity;
    uint32_t *host_physical_block_indices;
    const uint32_t *execution_physical_block_indices;
    uint32_t kv_block_lane_stride;
    uint32_t kv_block_lane_capacity;
    uint32_t *lane_physical_block_counts;
    uint32_t lane_count_capacity;
    SparkServingPrefillFunction prefill_function;
    SparkServingDecodeFunction decode_function;
    SparkServingReleaseSequenceFunction release_sequence_function;
    void *callback_context;
    uint32_t stop_token_ids[SPARK_SERVING_MAX_STOP_TOKEN_IDS];
    uint32_t stop_token_id_count;
    uint32_t request_handle_hash_heads[
        SPARK_SERVING_RECORD_HASH_SLOTS];
    SparkServingStats stats;
} SparkServingEngine;

void SparkServingInitializeSubmitTextRequest(
    SparkServingSubmitTextRequest *request);

void SparkServingInitializeSubmitTokenIdsRequest(
    SparkServingSubmitTokenIdsRequest *request);

void SparkServingInitializeDecodeResult(
    SparkServingDecodeResult *decode_result,
    uint32_t lane_count,
    uint32_t token_stride);

SparkStatus SparkServingEngineInitialize(
    SparkServingEngine *engine,
    const SparkServingEngineConfiguration *configuration);

// Releases resources the engine owns internally. The engine's request records,
// token storage, and other buffers remain caller-owned and are not touched; this
// frees only the persistent tokenizer workspace the engine allocates to keep the
// piece cache warm across requests. Safe to call on a zero-initialized engine and
// safe to call more than once.
void SparkServingEngineDestroy(
    SparkServingEngine *engine);

SparkStatus SparkServingEngineSubmitTokenIds(
    SparkServingEngine *engine,
    const SparkServingSubmitTokenIdsRequest *request,
    SparkServingSubmitResult *result);

SparkStatus SparkServingEngineSubmitText(
    SparkServingEngine *engine,
    const SparkServingSubmitTextRequest *request,
    SparkServingSubmitResult *result);

SparkStatus SparkServingEnginePump(
    SparkServingEngine *engine,
    uint32_t pump_flags,
    uint32_t max_dispatch_steps,
    SparkServingStats *stats);

SparkStatus SparkServingEngineCompletePrefillDispatch(
    SparkServingEngine *engine,
    const SparkRequestApiDispatch *dispatch);
SparkStatus SparkServingEngineCompleteDecodeDispatch(
    SparkServingEngine *engine,
    SparkRequestApiDispatch *dispatch,
    SparkServingDecodeResult *decode_result);

SparkStatus SparkServingEnginePopEvent(
    SparkServingEngine *engine,
    SparkServingEvent *event_out);

SparkStatus SparkServingEngineGetStats(
    SparkServingEngine *engine,
    SparkServingStats *stats_out);

SparkStatus SparkServingEngineCancelRequest(
    SparkServingEngine *engine,
    SparkServingRequestHandle request_handle);

SparkStatus SparkServingEngineFailRequestByRequestId(
    SparkServingEngine *engine,
    uint64_t request_id,
    SparkStatus failure_status);

SparkStatus SparkServingEngineReleaseCompletedRequest(
    SparkServingEngine *engine,
    SparkServingRequestHandle request_handle);

#ifdef __cplusplus
}
#endif

#endif
