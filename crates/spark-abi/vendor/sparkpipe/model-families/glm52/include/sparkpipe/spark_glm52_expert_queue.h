#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#define SPARK_GLM52_EXPERT_QUEUE_ABI_VERSION 2u
#define SPARK_GLM52_EXPERT_QUEUE_MAX_LAYERS 8u
#define SPARK_GLM52_EXPERT_QUEUE_MAX_EXPERTS 256u
#define SPARK_GLM52_EXPERT_QUEUE_MAX_ROWS 1048576u
#define SPARK_GLM52_EXPERT_QUEUE_MAX_FIRING_ROWS 1024u

typedef enum SparkGlm52ExpertQueueMode
{
	SPARK_GLM52_EXPERT_QUEUE_MODE_INVALID = 0,
	SPARK_GLM52_EXPERT_QUEUE_MODE_LOW_LATENCY = 1,
	SPARK_GLM52_EXPERT_QUEUE_MODE_SEALED_BATCH = 2
} SparkGlm52ExpertQueueMode;

typedef struct SparkGlm52ExpertQueueRow
{
	uint64_t row_id;
	uint64_t arrival_ns;
	uint32_t list_next;
} SparkGlm52ExpertQueueRow;

typedef struct SparkGlm52ExpertQueueSlot
{
	uint32_t head;
	uint32_t tail;
	uint32_t count;
	uint64_t oldest_arrival_ns;
} SparkGlm52ExpertQueueSlot;

typedef struct SparkGlm52ExpertQueueConfiguration
{
	uint32_t abi_version;
	uint32_t layer_count;
	uint32_t expert_count;
	uint32_t firing_threshold_rows;
	uint64_t firing_deadline_ns;
	SparkGlm52ExpertQueueMode mode;
	uint32_t reserved_u32[3];
} SparkGlm52ExpertQueueConfiguration;

typedef struct SparkGlm52ExpertQueue
{
	uint32_t abi_version;
	uint32_t layer_count;
	uint32_t expert_count;
	uint32_t firing_threshold_rows;
	uint64_t firing_deadline_ns;
	SparkGlm52ExpertQueueMode mode;
	uint32_t free_head;
	uint32_t free_high_water;
	uint32_t enqueued_row_count;
	uint32_t layer_enqueued_row_count[SPARK_GLM52_EXPERT_QUEUE_MAX_LAYERS];
	uint32_t layer_sealed[SPARK_GLM52_EXPERT_QUEUE_MAX_LAYERS];
	uint64_t firing_count;
	uint64_t fired_row_count;
	SparkGlm52ExpertQueueSlot slots[SPARK_GLM52_EXPERT_QUEUE_MAX_LAYERS][SPARK_GLM52_EXPERT_QUEUE_MAX_EXPERTS];
	SparkGlm52ExpertQueueRow rows[SPARK_GLM52_EXPERT_QUEUE_MAX_ROWS];
} SparkGlm52ExpertQueue;

typedef struct SparkGlm52ExpertQueueFiring
{
	uint32_t layer_index;
	uint32_t expert_index;
	uint32_t row_count;
	uint64_t row_ids[SPARK_GLM52_EXPERT_QUEUE_MAX_FIRING_ROWS];
} SparkGlm52ExpertQueueFiring;

SparkStatus SparkGlm52ExpertQueueInitialize(
	SparkGlm52ExpertQueue *queue,
	const SparkGlm52ExpertQueueConfiguration *configuration);
SparkStatus SparkGlm52ExpertQueueSetFiringThreshold(
	SparkGlm52ExpertQueue *queue,
	uint32_t firing_threshold_rows);
SparkStatus SparkGlm52ExpertQueueEnqueueRow(
	SparkGlm52ExpertQueue *queue,
	uint32_t layer_index,
	uint32_t expert_index,
	uint64_t row_id,
	uint64_t arrival_ns);
SparkStatus SparkGlm52ExpertQueueSealLayer(
	SparkGlm52ExpertQueue *queue,
	uint32_t layer_index);
SparkStatus SparkGlm52ExpertQueueNextFiring(
	SparkGlm52ExpertQueue *queue,
	uint64_t now_ns,
	SparkGlm52ExpertQueueFiring *firing_out);
