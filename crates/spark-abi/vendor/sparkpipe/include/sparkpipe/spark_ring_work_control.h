#pragma once

#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_distributed_work.h"
#include "sparkpipe/spark_glm52_dspark.h"
#include "sparkpipe/spark_glm52_kv_cache.h"
#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_mtp_tree.h"
#include "sparkpipe/spark_serving_engine.h"
#include "sparkpipe/spark_stage_plan.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_RING_WORK_CONTROL_ABI_VERSION 16u
#define SPARK_RING_WORK_CONTROL_PACKET_MAGIC 0x35574350u
#define SPARK_RING_WORK_CONTROL_STANDALONE_GENERATION UINT64_C(1)
#define SPARK_RING_WORK_CONTROL_PACKET_BYTES \
	((uint32_t)sizeof(SparkRingWorkControlPacket))
#define SPARK_RING_WORK_CONTROL_PACKET_PREFIX_BYTES \
	((uint32_t)offsetof(SparkRingWorkControlPacket,lanes))
#define SPARK_RING_WORK_CONTROL_KV_STATE_BYTES \
	((uint32_t)sizeof(SparkRingWorkControlKvState))
#define SPARK_RING_WORK_CONTROL_LANE_BYTES \
	((uint32_t)sizeof(SparkRingWorkControlLane))
#define SPARK_RING_WORK_CONTROL_MAX_ACTIVE_SEQUENCE_COUNT \
	SPARK_STAGE_PLAN_MAX_BATCH_BUCKET
#define SPARK_RING_WORK_CONTROL_KV_CONTEXT_TOKEN_CAPACITY \
	SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS
#define SPARK_RING_WORK_CONTROL_KV_BLOCK_CAPACITY \
	(SPARK_RING_WORK_CONTROL_KV_CONTEXT_TOKEN_CAPACITY / \
	 SPARK_GLM52_KV_BLOCK_TOKENS)

#define SPARK_RING_WORK_CONTROL_FLAG_PREFILL 0x00000001u
#define SPARK_RING_WORK_CONTROL_FLAG_MTP_DRAFT 0x00000002u
#define SPARK_RING_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE 0x00000004u
#define SPARK_RING_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY 0x00000008u
#define SPARK_RING_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY 0x00000010u
#define SPARK_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES 0x00000020u
#define SPARK_RING_WORK_CONTROL_FLAG_MTP_RESOLVE 0x00000040u
#define SPARK_RING_WORK_CONTROL_FLAG_MTP_TREE_VERIFY 0x00000080u
#define SPARK_RING_WORK_CONTROL_KNOWN_FLAGS \
	(SPARK_RING_WORK_CONTROL_FLAG_PREFILL | \
	 SPARK_RING_WORK_CONTROL_FLAG_MTP_DRAFT | \
	 SPARK_RING_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE | \
	 SPARK_RING_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY | \
	 SPARK_RING_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY | \
	 SPARK_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES | \
	 SPARK_RING_WORK_CONTROL_FLAG_MTP_RESOLVE | \
	 SPARK_RING_WORK_CONTROL_FLAG_MTP_TREE_VERIFY)
#define SPARK_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT \
	((SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT > \
	  SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT) ? \
	 SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT : \
	 SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT)
#define SPARK_RING_WORK_CONTROL_MAX_LANE_COUNT 1024u
#define SPARK_RING_WORK_CONTROL_MAX_PREFILL_TOKENS_PER_PACKET \
	SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH
#define SPARK_RING_WORK_CONTROL_INVALID_REQUEST_SLOT UINT32_MAX

#define SPARK_RING_KV_ENTRY_MISSING 0u
#define SPARK_RING_KV_ENTRY_IN_FLIGHT 1u
#define SPARK_RING_KV_ENTRY_RESIDENT 2u
#define SPARK_RING_KV_ENTRY_TRANSIENT 3u


#define SPARK_RING_KV_DIRECTORY_RESIDENCY_GPU 1u
#define SPARK_RING_KV_DIRECTORY_RESIDENCY_NVME 2u
#define SPARK_RING_KV_INVALID_BLOCK_INDEX UINT32_MAX

// 128-bit block identity. The low bit of `high` is the domain tag: 1 marks a
// private key derived from (sequence_id,logical_block_index), 0 marks a content
// key derived from the token prefix a block holds. The two domains can never
// alias, and an all-zero key is reserved to mean "empty slot".
typedef struct SparkRingKvKey
{
	uint64_t low;
	uint64_t high;
} SparkRingKvKey;

typedef SparkStatus (*SparkRingWorkControlKvSwapStoreFunction)(
	void *context,
	SparkRingKvKey key,
	uint32_t physical_block_index,
	uint32_t backing_block_index);
typedef SparkStatus (*SparkRingWorkControlKvSwapLoadFunction)(
	void *context,
	SparkRingKvKey key,
	uint32_t physical_block_index,
	uint32_t backing_block_index);

// One sequence slot. Names the block record it resolves to and owns no storage,
// so many sequence slots may name one block.
typedef struct SparkRingWorkControlKvDirectoryEntry
{
	SparkRingKvKey key;
	SparkRingKvKey block_key;
} SparkRingWorkControlKvDirectoryEntry;

// One block. Sole owner of residency, backing storage and the share refcount,
// so an eviction is observed by every sequence that names this block.
typedef struct SparkRingWorkControlKvBlockEntry
{
	SparkRingKvKey key;
	uint32_t physical_block_index;
	uint32_t backing_block_index;
	uint32_t reference_count;
	uint32_t residency_state : 2;
	uint32_t backing_valid : 1;
	uint32_t prefetch_mark : 29;
} SparkRingWorkControlKvBlockEntry;

typedef struct SparkRingWorkControlKvPrefetchEntry
{
	SparkRingKvKey key;
	uint32_t backing_block_index;
} SparkRingWorkControlKvPrefetchEntry;

typedef struct SparkRingWorkControlLane
{
	uint64_t request_id;
	uint64_t request_generation;
	uint64_t step_generation;
	uint64_t sequence_id;
	uint64_t sequence_position;
	uint32_t request_slot_index;
	uint32_t context_token_count;
	uint32_t input_token_id;
	uint32_t mtp_draft_token_count;
	uint32_t speculative_token_count;
	uint8_t mtp_resolution_proposed_token_count;
	uint8_t mtp_resolution_accepted_token_count;
	uint16_t mtp_resolution_path_id;
	uint32_t speculative_draft_token_ids[
		SPARK_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT];
} SparkRingWorkControlLane;

typedef struct SparkRingWorkControlPacket
{
	uint32_t magic;
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint64_t request_id;
	uint64_t sequence_id;
	uint64_t sequence_position;
	uint64_t deadline_time_ns;
	uint64_t control_generation;
	uint64_t transaction_id;
	uint64_t dispatch_generation;
	uint64_t request_generation;
	uint64_t step_generation;
	uint32_t step_chunk_index;
	uint32_t step_chunk_count;
	uint32_t transaction_phase;
	uint32_t reserved_transaction;
	uint32_t active_sequence_count;
	uint32_t new_token_count;
	uint32_t pipeline_slot;
	uint32_t priority;
	uint32_t block_token_count;
	uint32_t kv_block_table_token_count;
	uint32_t max_blocks_per_sequence;
	uint32_t mtp_draft_token_count;
	uint32_t input_token_id;
	uint32_t speculative_token_count;
	uint32_t speculative_token_index;
	uint32_t speculative_draft_token_ids[
		SPARK_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT];
	uint32_t lane_count;
	uint32_t rows_per_lane;
	uint32_t execution_row_count;
	uint32_t execution_batch_bucket;
	uint32_t prefill_token_ids[
		SPARK_RING_WORK_CONTROL_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkRingWorkControlLane
		lanes[SPARK_RING_WORK_CONTROL_MAX_LANE_COUNT];
} SparkRingWorkControlPacket;

typedef struct SparkRingWorkControlKvState
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t lane_capacity;
	uint32_t lane_stride;
	uint32_t block_token_count;
	uint32_t table_entry_capacity;
	uint32_t physical_block_capacity;
	uint32_t directory_capacity;
	uint32_t block_entry_capacity;
	uint32_t lane_block_key_stride;
	uint32_t next_physical_block_index;
	uint32_t backing_block_capacity;
	uint32_t free_backing_block_head;
	uint32_t directory_entry_count;
	uint32_t block_entry_count;
	uint32_t swapped_block_count;
	uint32_t clean_evict_count;
	uint64_t epoch;
	uint64_t control_generation;
	uint64_t control_generation_reset_count;
	uint32_t *physical_block_indices;
	uint32_t *lane_physical_block_counts;
	uint8_t *physical_block_states;
	SparkRingKvKey *physical_block_keys;
	uint64_t *physical_block_last_used_epochs;
	uint32_t *physical_block_pin_counts;
	SparkRingWorkControlKvDirectoryEntry *directory_entries;
	SparkRingWorkControlKvBlockEntry *block_entries;
	const SparkRingKvKey *lane_block_keys;
	uint32_t *backing_block_free_next;
	SparkRingWorkControlKvSwapStoreFunction swap_store_function;
	SparkRingWorkControlKvSwapLoadFunction swap_load_function;
	void *swap_context;
	uint32_t missing_block_count;
	uint32_t in_flight_block_count;
	uint32_t resident_block_count;
	uint32_t allocated_physical_block_count;
	uint64_t swap_store_count;
	uint64_t swap_load_count;
	uint32_t prefetch_generation;
	uint64_t share_hit_count;
	uint64_t share_admit_count;
} SparkRingWorkControlKvState;

uint32_t SparkRingWorkControlTransactionPhase(
	const SparkRingWorkControlPacket *packet);
uint64_t SparkRingWorkControlPacketFingerprint(
	const SparkRingWorkControlPacket *packet);
SparkStatus SparkRingWorkControlFinalizeTransaction(
	SparkRingWorkControlPacket *packet,
	uint64_t control_generation,
	uint32_t step_chunk_index,
	uint32_t step_chunk_count);
SparkStatus SparkRingWorkControlSetTransactionIdentity(
	SparkRingWorkControlPacket *packet,
	uint64_t control_generation,
	uint64_t transaction_id,
	uint64_t dispatch_generation,
	uint64_t step_generation);
SparkStatus SparkRingWorkControlGetTransactionIdentity(
	const SparkRingWorkControlPacket *packet,
	SparkDistributedWorkIdentity *identity_out);
SparkStatus SparkRingWorkControlValidatePacket(
	const SparkRingWorkControlPacket *packet,
	uint32_t max_active_sequence_count,
	uint32_t max_pipeline_slot_count);
uint32_t SparkRingWorkControlCalculatePacketBytes(
	uint32_t active_sequence_count);
SparkStatus SparkRingWorkControlSelectExecutionBatchBucket(
	const SparkRequestApiDispatch *request_dispatch,
	uint32_t batch_lane_or_row_count,
	uint32_t *batch_bucket_out);
SparkStatus SparkRingWorkControlSelectMtpDraftBudget(
	uint32_t dispatch_kind,
	uint32_t request_flags,
	uint32_t requested_budget,
	uint32_t *mtp_budget_out);
SparkStatus SparkRingWorkControlBuildDecodePacket(
	const SparkServingDecodeDispatch *decode_dispatch,
	uint32_t speculative_token_index,
	SparkRingWorkControlPacket *packet);
SparkStatus SparkRingWorkControlBuildDecodePacketRange(
	const SparkServingDecodeDispatch *decode_dispatch,
	uint32_t lane_offset,
	uint32_t lane_count,
	uint32_t speculative_token_index,
	SparkRingWorkControlPacket *packet);
SparkStatus SparkRingWorkControlBuildPrefillPacket(
	const SparkPromptPipelinePrefillDispatch *prefill_dispatch,
	uint32_t token_offset,
	uint32_t token_count,
	SparkRingWorkControlPacket *packet);
SparkStatus SparkRingWorkControlSelectPrefillChunk(
	const SparkPromptPipelinePrefillDispatch *prefill_dispatch,
	uint32_t token_offset,
	uint32_t maximum_execution_row_count,
	uint32_t *token_count_out);
SparkStatus SparkRingWorkControlInitializeKvState(
	SparkRingWorkControlKvState *state,
	uint32_t lane_capacity,
	uint32_t lane_stride,
	uint32_t block_token_count,
	uint32_t physical_block_capacity,
	uint32_t directory_capacity,
	uint32_t block_entry_capacity,
	uint32_t *physical_block_indices,
	uint32_t *lane_physical_block_counts,
	uint8_t *physical_block_states,
	SparkRingKvKey *physical_block_keys,
	uint64_t *physical_block_last_used_epochs,
	SparkRingWorkControlKvDirectoryEntry *directory_entries,
	SparkRingWorkControlKvBlockEntry *block_entries);

// Supply per-lane content keys for the blocks the next packet will acquire.
// Leaving this unset, or leaving a key zero, keeps a block private, so sharing
// is opt-in per block and the default behaviour is unchanged.
SparkStatus SparkRingWorkControlConfigureKvSharing(
	SparkRingWorkControlKvState *state,
	const SparkRingKvKey *lane_block_keys,
	uint32_t lane_block_key_stride);

uint32_t SparkRingWorkControlKvKeyEqual(
	SparkRingKvKey left,
	SparkRingKvKey right);

SparkRingKvKey SparkRingWorkControlPrivateKey(
	uint64_t sequence_id,
	uint32_t logical_block_index);

SparkRingKvKey SparkRingWorkControlContentKey(
	uint64_t digest_low,
	uint64_t digest_high);

uint64_t SparkRingWorkControlKvCommittedFrontier(
	const SparkRingWorkControlLane *lane);
SparkStatus SparkRingWorkControlConfigureKvSwap(
	SparkRingWorkControlKvState *state,
	uint32_t backing_block_capacity,
	uint32_t *backing_block_free_next,
	SparkRingWorkControlKvSwapStoreFunction swap_store_function,
	SparkRingWorkControlKvSwapLoadFunction swap_load_function,
	void *swap_context);
SparkStatus SparkRingWorkControlConfigureKvPins(
	SparkRingWorkControlKvState *state,
	uint32_t *physical_block_pin_counts);
SparkStatus SparkRingWorkControlAdvanceKvGeneration(
	SparkRingWorkControlKvState *state,
	uint64_t control_generation);
SparkStatus SparkRingWorkControlPinPhysicalBlock(
	SparkRingWorkControlKvState *state,
	uint32_t physical_block_index);
SparkStatus SparkRingWorkControlUnpinPhysicalBlock(
	SparkRingWorkControlKvState *state,
	uint32_t physical_block_index);
SparkStatus SparkRingWorkControlAcquireTransientPhysicalBlock(
	SparkRingWorkControlKvState *state,
	uint32_t *physical_block_index_out);
SparkStatus SparkRingWorkControlReleaseTransientPhysicalBlock(
	SparkRingWorkControlKvState *state,
	uint32_t physical_block_index);
uint32_t SparkRingWorkControlBlockCount(
	uint32_t token_count,
	uint32_t block_token_count);
SparkStatus SparkRingWorkControlCollectKvPrefetchEntries(
	const SparkRingWorkControlPacket *packets,
	uint32_t packet_count,
	SparkRingWorkControlKvState *state,
	SparkRingWorkControlKvPrefetchEntry *entries,
	uint32_t entry_capacity,
	uint32_t *entry_count_out);
SparkStatus SparkRingWorkControlPlanExecutionChunks(
	uint32_t logical_lane_count,
	uint32_t rows_per_lane,
	uint32_t execution_row_capacity,
	uint32_t *maximum_lanes_per_chunk_out,
	uint32_t *chunk_count_out);
SparkStatus SparkRingWorkControlBuildHostKvBlockTable(
	const SparkRingWorkControlPacket *packet,
	SparkRingWorkControlKvState *state,
	SparkKvBlockTableView *view);
SparkStatus SparkRingWorkControlCommitHostKvBlockTable(
	const SparkRingWorkControlPacket *packet,
	SparkRingWorkControlKvState *state);
SparkStatus SparkRingWorkControlCancelHostKvBlockTable(
	const SparkRingWorkControlPacket *packet,
	SparkRingWorkControlKvState *state);
SparkStatus SparkRingWorkControlReleaseSequence(
	SparkRingWorkControlKvState *state,
	uint64_t sequence_id,
	uint32_t logical_block_count);
SparkStatus SparkRingWorkControlReleasePacketSequences(
	const SparkRingWorkControlPacket *packet,
	SparkRingWorkControlKvState *state);

#ifdef __cplusplus
}
#endif
