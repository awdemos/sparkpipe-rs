#pragma once

#include <stdint.h>

#include "sparkpipe/spark_serving_engine.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SERVICE_ABI_VERSION 1u
#define SPARK_SERVICE_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServiceConfiguration))
#define SPARK_SERVICE_RUNTIME_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServiceRuntime))
#define SPARK_SERVICE_CLIENT_SESSION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServiceClientSession))
#define SPARK_SERVICE_REQUEST_MAP_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServiceRequestMap))
#define SPARK_SERVICE_SUBMIT_TEXT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServiceSubmitTextRequest))
#define SPARK_SERVICE_SUBMIT_TOKENS_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServiceSubmitTokenIdsRequest))
#define SPARK_SERVICE_SUBMIT_RESULT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServiceSubmitResult))
#define SPARK_SERVICE_EVENT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServiceEvent))
#define SPARK_SERVICE_STATS_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServiceStats))
#define SPARK_SERVICE_FRAME_HEADER_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServiceFrameHeader))
#define SPARK_SERVICE_FRAME_SUBMIT_TEXT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServiceSubmitTextFrameBody))
#define SPARK_SERVICE_FRAME_SUBMIT_TOKENS_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServiceSubmitTokenIdsFrameBody))
#define SPARK_SERVICE_FRAME_CANCEL_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkServiceCancelFrameBody))

#define SPARK_SERVICE_DEFAULT_PUMP_DISPATCH_STEPS 256u
#define SPARK_SERVICE_DEFAULT_REQUEST_ID_BASE 5000000000ull
#define SPARK_SERVICE_FRAME_MAGIC 0x35504B53u
#define SPARK_SERVICE_MAX_FRAME_BODY_BYTES (128u * 1024u * 1024u)
#define SPARK_SERVICE_MAX_TEXT_BYTES (64u * 1024u * 1024u)
#define SPARK_SERVICE_MAX_TOKEN_FRAME_COUNT \
    (SPARK_SCHEDULER_MAX_CONTEXT_TOKENS)
#define SPARK_SERVICE_CLIENT_HASH_SLOTS 1024u
#define SPARK_SERVICE_REQUEST_MAP_HASH_SLOTS 4096u
#define SPARK_SERVICE_NO_HASH_SLOT UINT32_MAX

#define SPARK_SERVICE_CONFIGURATION_FLAG_AUTO_RELEASE_COMPLETED_MAPPINGS \
    0x00000001u
#define SPARK_SERVICE_CONFIGURATION_FLAG_DRAIN_ENGINE_EVENTS_BEFORE_PUMP \
    0x00000002u
#define SPARK_SERVICE_CONFIGURATION_DEFAULT_FLAGS \
    (SPARK_SERVICE_CONFIGURATION_FLAG_AUTO_RELEASE_COMPLETED_MAPPINGS | \
     SPARK_SERVICE_CONFIGURATION_FLAG_DRAIN_ENGINE_EVENTS_BEFORE_PUMP)
#define SPARK_SERVICE_CONFIGURATION_KNOWN_FLAGS \
    SPARK_SERVICE_CONFIGURATION_DEFAULT_FLAGS

#define SPARK_SERVICE_CLIENT_STATE_FREE 0u
#define SPARK_SERVICE_CLIENT_STATE_CONNECTED 1u
#define SPARK_SERVICE_CLIENT_STATE_DRAINING 2u

#define SPARK_SERVICE_REQUEST_STATE_FREE 0u
#define SPARK_SERVICE_REQUEST_STATE_LIVE 1u
#define SPARK_SERVICE_REQUEST_STATE_COMPLETED 2u
#define SPARK_SERVICE_REQUEST_STATE_CANCELLED 3u

#define SPARK_SERVICE_EVENT_KIND_NONE 0u
#define SPARK_SERVICE_EVENT_KIND_REQUEST_ACCEPTED 1u
#define SPARK_SERVICE_EVENT_KIND_PREFILL_PROGRESS 2u
#define SPARK_SERVICE_EVENT_KIND_TOKEN 3u
#define SPARK_SERVICE_EVENT_KIND_REQUEST_COMPLETED 4u
#define SPARK_SERVICE_EVENT_KIND_REQUEST_CANCELLED 5u
#define SPARK_SERVICE_EVENT_KIND_ERROR 6u
#define SPARK_SERVICE_EVENT_KIND_BACKPRESSURE 7u
#define SPARK_SERVICE_EVENT_KIND_CLIENT_CONNECTED 8u
#define SPARK_SERVICE_EVENT_KIND_CLIENT_DISCONNECTED 9u
#define SPARK_SERVICE_EVENT_KIND_STATS 10u

#define SPARK_SERVICE_FRAME_KIND_SUBMIT_TEXT 1u
#define SPARK_SERVICE_FRAME_KIND_SUBMIT_TOKEN_IDS 2u
#define SPARK_SERVICE_FRAME_KIND_CANCEL_REQUEST 3u
#define SPARK_SERVICE_FRAME_KIND_PING 4u
#define SPARK_SERVICE_FRAME_KIND_EVENT 100u
#define SPARK_SERVICE_FRAME_KIND_SUBMIT_ACK 101u
#define SPARK_SERVICE_FRAME_KIND_ERROR 102u
#define SPARK_SERVICE_FRAME_KIND_PONG 103u

#define SPARK_SERVICE_FRAME_FLAG_REALTIME \
    SPARK_SERVING_SUBMIT_FLAG_REALTIME
#define SPARK_SERVICE_FRAME_FLAG_DISABLE_SPECULATION \
    SPARK_SERVING_SUBMIT_FLAG_DISABLE_SPECULATION
#define SPARK_SERVICE_FRAME_KNOWN_SUBMIT_FLAGS \
    SPARK_SERVING_SUBMIT_KNOWN_FLAGS

typedef uint64_t SparkServiceClientId;
typedef uint64_t SparkServiceRequestId;

typedef struct SparkServiceClientSession
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t state;
    uint32_t flags;
    SparkServiceClientId client_id;
    uint64_t user_cookie;
    uint64_t accepted_request_count;
    uint64_t completed_request_count;
    uint32_t client_hash_next;
    uint32_t reserved0;
} SparkServiceClientSession;

typedef struct SparkServiceRequestMap
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t state;
    uint32_t flags;
    SparkServiceClientId client_id;
    SparkServiceRequestId client_request_id;
    uint64_t serving_request_id;
    uint64_t sequence_id;
    SparkServingRequestHandle serving_request_handle;
    uint32_t client_request_hash_next;
    uint32_t serving_handle_hash_next;
} SparkServiceRequestMap;

typedef struct SparkServiceEvent
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
    SparkServiceClientId client_id;
    SparkServiceRequestId client_request_id;
    uint64_t serving_request_id;
    uint64_t sequence_id;
    SparkServingRequestHandle serving_request_handle;
} SparkServiceEvent;

typedef struct SparkServiceSubmitTextRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t priority;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t max_prefill_tokens_per_step;
    uint32_t tokenizer_encode_flags;
    SparkServiceClientId client_id;
    SparkServiceRequestId client_request_id;
    uint64_t sequence_id;
    const char *text;
    uint32_t text_bytes;
    uint32_t reserved0;
} SparkServiceSubmitTextRequest;

typedef struct SparkServiceSubmitTokenIdsRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t priority;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t max_prefill_tokens_per_step;
    uint32_t token_count;
    SparkServiceClientId client_id;
    SparkServiceRequestId client_request_id;
    uint64_t sequence_id;
    const uint32_t *token_ids;
} SparkServiceSubmitTokenIdsRequest;

typedef struct SparkServiceSubmitResult
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t prompt_token_count;
    uint32_t required_token_capacity;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    SparkServiceClientId client_id;
    SparkServiceRequestId client_request_id;
    uint64_t serving_request_id;
    uint64_t sequence_id;
    SparkServingRequestHandle serving_request_handle;
} SparkServiceSubmitResult;

typedef struct SparkServiceStats
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t connected_client_count;
    uint32_t live_request_count;
    uint32_t completed_request_mapping_count;
    uint32_t event_count;
    uint32_t event_capacity;
    uint32_t dropped_event_count;
    uint32_t last_status;
    uint32_t reserved0;
    uint64_t submitted_request_count;
    uint64_t accepted_request_count;
    uint64_t forwarded_event_count;
    uint64_t engine_pump_count;
    SparkServingStats serving_stats;
} SparkServiceStats;

typedef struct SparkServiceConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t default_pump_dispatch_steps;
    uint64_t request_id_base;
    SparkServingEngine *serving_engine;
    SparkServiceClientSession *client_sessions;
    uint32_t client_session_capacity;
    SparkServiceRequestMap *request_maps;
    uint32_t request_map_capacity;
    SparkServiceEvent *event_ring;
    uint32_t event_ring_capacity;
} SparkServiceConfiguration;

typedef struct SparkServiceRuntime
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t default_pump_dispatch_steps;
    uint64_t next_generated_request_id;
    uint64_t next_generated_client_id;
    SparkServingEngine *serving_engine;
    SparkServiceClientSession *client_sessions;
    uint32_t client_session_capacity;
    SparkServiceRequestMap *request_maps;
    uint32_t request_map_capacity;
    SparkServiceEvent *event_ring;
    uint32_t event_ring_capacity;
    uint32_t event_read_index;
    uint32_t event_write_index;
    uint32_t event_count;
    uint32_t dropped_event_count;
    uint32_t client_hash_heads[SPARK_SERVICE_CLIENT_HASH_SLOTS];
    uint32_t client_request_hash_heads[
        SPARK_SERVICE_REQUEST_MAP_HASH_SLOTS];
    uint32_t serving_handle_hash_heads[
        SPARK_SERVICE_REQUEST_MAP_HASH_SLOTS];
    SparkServiceStats stats;
} SparkServiceRuntime;

typedef struct SparkServiceFrameHeader
{
    uint32_t magic;
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t kind;
    uint32_t flags;
    uint32_t body_bytes;
    SparkServiceClientId client_id;
    SparkServiceRequestId client_request_id;
} SparkServiceFrameHeader;

typedef struct SparkServiceSubmitTextFrameBody
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t priority;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t max_prefill_tokens_per_step;
    uint32_t tokenizer_encode_flags;
    uint32_t text_bytes;
    uint64_t sequence_id;
} SparkServiceSubmitTextFrameBody;

typedef struct SparkServiceSubmitTokenIdsFrameBody
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t priority;
    uint32_t thinking_token_budget;
    uint32_t output_token_budget;
    uint32_t max_prefill_tokens_per_step;
    uint32_t token_count;
    uint32_t reserved0;
    uint64_t sequence_id;
} SparkServiceSubmitTokenIdsFrameBody;

typedef struct SparkServiceCancelFrameBody
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    SparkServingRequestHandle serving_request_handle;
    SparkServiceRequestId client_request_id;
} SparkServiceCancelFrameBody;

void SparkServiceInitializeSubmitTextRequest(
    SparkServiceSubmitTextRequest *request);

void SparkServiceInitializeSubmitTokenIdsRequest(
    SparkServiceSubmitTokenIdsRequest *request);

void SparkServiceInitializeFrameHeader(
    SparkServiceFrameHeader *frame_header,
    uint32_t frame_kind);

SparkStatus SparkServiceValidateFrameHeader(
    const SparkServiceFrameHeader *frame_header,
    uint32_t maximum_body_bytes);

SparkStatus SparkServiceInitialize(
    SparkServiceRuntime *service,
    const SparkServiceConfiguration *configuration);

SparkStatus SparkServiceRegisterClient(
    SparkServiceRuntime *service,
    uint64_t user_cookie,
    SparkServiceClientId *client_id_out);

SparkStatus SparkServiceDisconnectClient(
    SparkServiceRuntime *service,
    SparkServiceClientId client_id);

SparkStatus SparkServiceSubmitTokenIds(
    SparkServiceRuntime *service,
    const SparkServiceSubmitTokenIdsRequest *request,
    SparkServiceSubmitResult *result);

SparkStatus SparkServiceSubmitText(
    SparkServiceRuntime *service,
    const SparkServiceSubmitTextRequest *request,
    SparkServiceSubmitResult *result);

SparkStatus SparkServiceHandleSubmitTokenIdsFrame(
    SparkServiceRuntime *service,
    SparkServiceClientId client_id,
    const SparkServiceFrameHeader *frame_header,
    const void *body,
    uint32_t body_bytes,
    SparkServiceSubmitResult *result);

SparkStatus SparkServiceHandleSubmitTextFrame(
    SparkServiceRuntime *service,
    SparkServiceClientId client_id,
    const SparkServiceFrameHeader *frame_header,
    const void *body,
    uint32_t body_bytes,
    SparkServiceSubmitResult *result);

SparkStatus SparkServiceHandleCancelFrame(
    SparkServiceRuntime *service,
    SparkServiceClientId client_id,
    const SparkServiceFrameHeader *frame_header,
    const void *body,
    uint32_t body_bytes);

SparkStatus SparkServiceBuildEventFrame(
    const SparkServiceEvent *event,
    SparkServiceFrameHeader *frame_header,
    SparkServiceEvent *frame_body);

SparkStatus SparkServiceCancelRequest(
    SparkServiceRuntime *service,
    SparkServiceClientId client_id,
    SparkServiceRequestId client_request_id);

SparkStatus SparkServicePump(
    SparkServiceRuntime *service,
    uint32_t max_dispatch_steps,
    SparkServiceStats *stats_out);

SparkStatus SparkServicePopEvent(
    SparkServiceRuntime *service,
    SparkServiceEvent *event_out);

SparkStatus SparkServiceGetStats(
    SparkServiceRuntime *service,
    SparkServiceStats *stats_out);

#ifdef __cplusplus
}
#endif
