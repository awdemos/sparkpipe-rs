#ifndef SPARKPIPE_SPARK_K3_TP_SHARD_TABLE_H
#define SPARKPIPE_SPARK_K3_TP_SHARD_TABLE_H

#include "sparkpipe/spark_tp_shard.h"

// The K3 classification table over the common shard engine, keyed by the
// PACK MANIFEST field names tools/k3_pack.py emits - the loader classifies
// what it reads, and what it reads is the manifest. tools/k3_shard.py holds
// the same partition in python for offline pre-sharding; the k3-shard-table
// gate proves the two tables are one table twice, so a drift in either is a
// red suite, not a mis-sliced rank.
//
// The classes follow docs/DRY_LEDGER.md's K3 departures, on pack format V2
// (docs/K3_PACK_FORMAT_V2.md): the six KDA input projections ride as TWO
// fused tensors - kda_qkv_beta_weight splits per section on whole heads
// (OUTPUT_DIM_HEADS), kda_decay_gate_down_weight REPLICATES with the other
// low-rank bottlenecks and the kv_a latent path; concatenated gate|up
// tensors and the interleaved expert tensors carry their own grid-aware
// slicing in the loader (engine classes cover the bf16 projections; the
// MXFP4 expert tensors classify as EXPERT_* and the pack loader slices
// whole 16-neuron cells for w1 and whole 128-element k-tiles for w2, scales
// co-tiled, so there are no scale tensors to classify).

static uint32_t SparkK3TpShardIs(const char *name, const char *suffix)
{
	return SparkTpShardNameEndsWith(name,suffix);
}

static SparkTpShardClass SparkK3TpShardClassifyTensor(const char *name)
{
	if (SparkK3TpShardIs(name,"attn_norm_weight") || SparkK3TpShardIs(name,"mlp_norm_weight") ||
		SparkK3TpShardIs(name,"attnres_attn_weight") || SparkK3TpShardIs(name,"attnres_mlp_weight") ||
		SparkK3TpShardIs(name,"router_weight") || SparkK3TpShardIs(name,"router_bias") ||
		SparkK3TpShardIs(name,"kda_decay_gate_down_weight") ||
		SparkK3TpShardIs(name,"kda_decay_bias") || SparkK3TpShardIs(name,"kda_head_log_scale") ||
		SparkK3TpShardIs(name,"kda_out_norm_weight") || SparkK3TpShardIs(name,"mla_q_down_weight") ||
		SparkK3TpShardIs(name,"mla_q_norm_weight") || SparkK3TpShardIs(name,"mla_kv_a_weight") ||
		SparkK3TpShardIs(name,"mla_kv_a_norm_weight") || SparkK3TpShardIs(name,"mla_gate_down_weight") ||
		SparkK3TpShardIs(name,"routed_norm_weight") || SparkK3TpShardIs(name,"model.norm.weight") ||
		SparkK3TpShardIs(name,"model.attnres_out_weight"))
		return SPARK_TP_SHARD_CLASS_REPLICATED;
	if (SparkK3TpShardIs(name,"kda_qkv_beta_weight") ||
		SparkK3TpShardIs(name,"kda_q_conv_weight") ||
		SparkK3TpShardIs(name,"kda_k_conv_weight") || SparkK3TpShardIs(name,"kda_v_conv_weight") ||
		SparkK3TpShardIs(name,"kda_decay_up_weight") || SparkK3TpShardIs(name,"kda_gate_up_weight") ||
		SparkK3TpShardIs(name,"mla_q_up_weight") ||
		SparkK3TpShardIs(name,"mla_kv_b_value_weight") || SparkK3TpShardIs(name,"mla_gate_up_weight"))
		return SPARK_TP_SHARD_CLASS_OUTPUT_DIM_HEADS;
	if (SparkK3TpShardIs(name,"kda_out_weight") || SparkK3TpShardIs(name,"mla_out_weight"))
		return SPARK_TP_SHARD_CLASS_INPUT_DIM_HEADS;
	if (SparkK3TpShardIs(name,"routed_down_weight") || SparkK3TpShardIs(name,"model.embed_tokens.weight") ||
		SparkK3TpShardIs(name,"lm_head.weight"))
		return SPARK_TP_SHARD_CLASS_OUTPUT_DIM;
	if (SparkK3TpShardIs(name,"routed_up_weight"))
		return SPARK_TP_SHARD_CLASS_INPUT_DIM;
	// gate|up concatenations and the interleaved expert w1: classified here
	// so the loader knows they are NOT plain splits; the slicing itself is
	// the loader's cell-aware path, mirrored by tools/k3_shard.py.
	if (SparkK3TpShardIs(name,"shared_w1_weight") || SparkK3TpShardIs(name,"dense_gate_up_weight") ||
		SparkK3TpShardIs(name,"expert_w1_weight"))
		return SPARK_TP_SHARD_CLASS_CONCAT_OUTPUT;
	if (SparkK3TpShardIs(name,"shared_w2_weight") || SparkK3TpShardIs(name,"dense_down_weight") ||
		SparkK3TpShardIs(name,"expert_w2_weight"))
		return SPARK_TP_SHARD_CLASS_INPUT_DIM;
	return SPARK_TP_SHARD_CLASS_UNKNOWN;
}

#endif
