#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_stage_plan.h"
#include "sparkpipe/spark_glm52_dspark.h"
#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_PRODUCTION_TOPOLOGY_ABI_VERSION 1u
#define SPARK_PRODUCTION_TOPOLOGY_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkProductionTopology))
#define SPARK_PRODUCTION_TOPOLOGY_SELECTED_TOKEN_COUNT \
    SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT
#define SPARK_PRODUCTION_TOPOLOGY_INDEXSHARE_GROUP_LAYER_COUNT \
    SPARK_GLM52_MODEL_DSA_INDEX_SHARE_GROUP_LAYER_COUNT
#define SPARK_PRODUCTION_TOPOLOGY_FIRST_FULL_INDEXER_LAYER_COUNT \
    SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER
#define SPARK_PRODUCTION_TOPOLOGY_INDEX_SKIP_TOPK_OFFSET \
    SPARK_GLM52_MODEL_DSA_INDEX_SKIP_TOPK_OFFSET
#define SPARK_PRODUCTION_TOPOLOGY_MAX_INDEXSHARE_SIDEBANDS \
    ((SPARK_STAGE_PLAN_MAX_STAGE_COUNT - 1u) + \
     SPARK_GLM52_DSPARK_AUX_LAYER_COUNT)

#define SPARK_PRODUCTION_TOPOLOGY_FLAG_OFFICIAL_DSA_INDEXSHARE \
    0x00000001u
#define SPARK_PRODUCTION_TOPOLOGY_FLAG_BOUNDED_LONG_CONTEXT_ATTENTION \
    0x00000002u
#define SPARK_PRODUCTION_TOPOLOGY_FLAG_INDEXSHARE_STAGE_BOUNDARY_STATE \
    0x00000004u
#define SPARK_PRODUCTION_TOPOLOGY_FLAG_MLA_COMPRESSED_KV_CACHE \
    0x00000008u
#define SPARK_PRODUCTION_TOPOLOGY_PRODUCTION_REQUIRED_FLAGS \
    (SPARK_PRODUCTION_TOPOLOGY_FLAG_OFFICIAL_DSA_INDEXSHARE | \
     SPARK_PRODUCTION_TOPOLOGY_FLAG_BOUNDED_LONG_CONTEXT_ATTENTION | \
     SPARK_PRODUCTION_TOPOLOGY_FLAG_INDEXSHARE_STAGE_BOUNDARY_STATE | \
     SPARK_PRODUCTION_TOPOLOGY_FLAG_MLA_COMPRESSED_KV_CACHE)

#define SPARK_PRODUCTION_TOPOLOGY_SIDEBAND_FLAG_SELECTED_TOKEN_INDICES \
    0x00000001u
#define SPARK_PRODUCTION_TOPOLOGY_SIDEBAND_FLAG_DEVICE_TO_DEVICE \
    0x00000002u
#define SPARK_PRODUCTION_TOPOLOGY_SIDEBAND_FLAG_DSPARK_HIDDEN_TAP \
    0x00000004u

typedef struct SparkProductionTopologyIndexShareSideBand
{
    uint32_t source_layer_index;
    uint32_t group_end_layer_exclusive;
    uint32_t export_stage_index;
    uint32_t import_stage_index;
    uint32_t first_imported_consumer_layer_index;
    uint32_t imported_consumer_layer_count;
    uint32_t selected_token_count;
    uint32_t active_sequence_capacity;
    uint64_t payload_bytes;
    uint32_t flags;
    uint32_t reserved0;
} SparkProductionTopologyIndexShareSideBand;

typedef struct SparkProductionTopologyStage
{
    uint32_t first_layer_index;
    uint32_t layer_count;
    uint32_t stage_plan_flags;
    uint32_t exported_sideband_count;
    uint32_t imported_sideband_count;
    uint32_t first_exported_sideband_index;
    uint32_t first_imported_sideband_index;
    uint32_t reserved0;
} SparkProductionTopologyStage;

typedef struct SparkProductionTopology
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t topology_flags;
    uint32_t stage_count;
    uint32_t selected_token_count;
    uint32_t active_sequence_capacity;
    uint32_t kv_block_token_count;
    uint32_t mla_cache_element_count;
    uint32_t indexshare_sideband_count;
    uint32_t reserved0;
    SparkProductionTopologyStage stages[
        SPARK_STAGE_PLAN_MAX_STAGE_COUNT];
    SparkProductionTopologyIndexShareSideBand indexshare_sidebands[
        SPARK_PRODUCTION_TOPOLOGY_MAX_INDEXSHARE_SIDEBANDS];
} SparkProductionTopology;

uint32_t SparkDsaIndexShareSourceLayer(uint32_t layer_index, uint32_t total_layer_count);

SparkStatus SparkDsaIndexShareFindGroupEndLayerExclusive(
    uint32_t layer_index,
    uint32_t total_layer_count,
    uint32_t *group_end_layer_exclusive_out);

SparkStatus SparkProductionTopologyBuild(
    const SparkStagePlanGeometry *geometry,
    const SparkStagePlan *stage_plan,
    uint32_t active_sequence_capacity,
    uint32_t selected_token_count,
    uint32_t kv_block_token_count,
    uint32_t mla_cache_element_count,
    SparkProductionTopology *topology,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkProductionTopologyHopSidebandLayout(
    const SparkProductionTopology *topology,
    uint32_t export_stage_index,
    uint32_t *sideband_kind_bits_out,
    uint32_t *sideband_bytes_per_sequence_out);

SparkStatus SparkProductionTopologyArmHopSidebandPacket(
    const SparkProductionTopology *topology,
    uint32_t export_stage_index,
    void *sideband_payload,
    SparkHiddenTransportPacket *packet);

SparkStatus SparkProductionTopologyValidate(
    const SparkProductionTopology *topology,
    char *error_buffer,
    uint32_t error_buffer_bytes);

#ifdef __cplusplus
}
#endif
