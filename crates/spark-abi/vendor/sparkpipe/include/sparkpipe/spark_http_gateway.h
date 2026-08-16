#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_compat_api.h"
#include "sparkpipe/spark_service_backend.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_HTTP_GATEWAY_ABI_VERSION 1u
#define SPARK_HTTP_GATEWAY_REQUEST_BYTES \
    ((uint32_t)sizeof(SparkHttpGatewayRequest))
#define SPARK_HTTP_GATEWAY_RESPONSE_BYTES \
    ((uint32_t)sizeof(SparkHttpGatewayResponse))

#define SPARK_HTTP_GATEWAY_ROUTE_NONE 0u
#define SPARK_HTTP_GATEWAY_ROUTE_DEMO_UI 1u
#define SPARK_HTTP_GATEWAY_ROUTE_HEALTH 2u
#define SPARK_HTTP_GATEWAY_ROUTE_OPENAI_CHAT 3u
#define SPARK_HTTP_GATEWAY_ROUTE_OPENAI_COMPLETIONS 4u
#define SPARK_HTTP_GATEWAY_ROUTE_ANTHROPIC_MESSAGES 5u
#define SPARK_HTTP_GATEWAY_ROUTE_CORS_PREFLIGHT 6u

#define SPARK_HTTP_GATEWAY_RESPONSE_FLAG_STREAM 0x00000001u

#define SPARK_HTTP_GATEWAY_DEFAULT_MAX_CONTEXT_TOKENS \
    SPARK_SERVICE_MAX_TOKEN_FRAME_COUNT
#define SPARK_HTTP_GATEWAY_DEFAULT_MAX_UPLOAD_BYTES \
    SPARK_SERVICE_MAX_TEXT_BYTES

typedef struct SparkHttpGatewayRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    const char *method;
    const char *path;
    const char *body;
    uint32_t body_bytes;
    const char *authorization;
    uint32_t authorization_bytes;
} SparkHttpGatewayRequest;

typedef struct SparkHttpGatewayResponse
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t status_code;
    uint32_t flags;
    const char *content_type;
    char *body;
    uint32_t body_capacity;
    uint32_t body_bytes;
} SparkHttpGatewayResponse;

void SparkHttpGatewayInitializeRequest(
    SparkHttpGatewayRequest *request);

void SparkHttpGatewayInitializeResponse(
    SparkHttpGatewayResponse *response,
    char *body,
    uint32_t body_capacity);

uint32_t SparkHttpGatewayRoute(
    const SparkHttpGatewayRequest *request);

SparkStatus SparkHttpGatewayBuildDemoUi(
    SparkHttpGatewayResponse *response);

SparkStatus SparkHttpGatewayBuildHealth(
    SparkHttpGatewayResponse *response,
    uint32_t runtime_initialized,
    uint32_t local_control_ready);

SparkStatus SparkHttpGatewayBuildBackendUnavailable(
    SparkHttpGatewayResponse *response,
    uint32_t stream);

SparkStatus SparkHttpGatewayBuildRequestTimeout(
    SparkHttpGatewayResponse *response,
    uint32_t stream);

SparkStatus SparkHttpGatewayBuildUnauthorized(
    SparkHttpGatewayResponse *response);

SparkStatus SparkHttpGatewayBuildNotFound(
    SparkHttpGatewayResponse *response);

SparkStatus SparkHttpGatewayBuildCorsPreflight(
    SparkHttpGatewayResponse *response);

uint32_t SparkHttpGatewayBodyRequestsStream(
    const char *body,
    uint32_t body_bytes);

uint32_t SparkHttpGatewayAuthorizationMatches(
    const SparkHttpGatewayRequest *request,
    const char *api_key);

SparkStatus SparkHttpGatewayBuildServiceHealth(
    SparkHttpGatewayResponse *response,
    const SparkServiceStats *stats,
    const SparkServiceBackendView *backend_view);

SparkStatus SparkHttpGatewaySubmitJsonToService(
    SparkServiceRuntime *service,
    const SparkHttpGatewayRequest *request,
    SparkServiceClientId client_id,
    SparkServiceRequestId client_request_id,
    SparkGlm52CompatTextRequest *compat_request,
    SparkServiceSubmitResult *submit_result,
    SparkHttpGatewayResponse *response);

SparkStatus SparkHttpGatewayBuildSubmitAccepted(
    SparkHttpGatewayResponse *response,
    const SparkServiceSubmitResult *submit_result,
    uint32_t stream);

SparkStatus SparkHttpGatewayBuildServiceEventStream(
    SparkHttpGatewayResponse *response,
    const SparkServiceEvent *service_event,
    const SparkTokenizer *tokenizer);

#ifdef __cplusplus
}
#endif
