#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_glm52_tp_shard.h"

#include <string.h>

// THE TABLE IS THE MODEL'S; THE ENGINE IS EVERYONE'S. This file is what
// remains of the glm shard module after the split: the name-to-class table,
// the one head-block irregularity (q_b's output block), and wrappers that
// keep the original API standing so no consumer moved. The arithmetic, the
// validation, the hash and the refusal live in serving/spark_tp_shard.c.

SparkTpShardClass SparkGlm52TpShardClassifyTensor(const char *tensor_name)
{
	// Head-structured attention projections: q_b and kv_b split their output
	// dimension on whole-head boundaries; o_proj splits its input dimension the
	// same way and closes with the layer all-reduce.
	if (SparkTpShardNameEndsWith(tensor_name,"self_attn.q_b_proj.weight"))
		return SPARK_TP_SHARD_CLASS_OUTPUT_DIM_HEADS;
	if (SparkTpShardNameEndsWith(tensor_name,"self_attn.kv_b_proj.weight"))
		return SPARK_TP_SHARD_CLASS_OUTPUT_DIM_HEADS;
	if (SparkTpShardNameEndsWith(tensor_name,"self_attn.o_proj.weight"))
		return SPARK_TP_SHARD_CLASS_INPUT_DIM_HEADS;
	// Dense and shared-expert MLP: gate and up split the intermediate output
	// dimension, down splits the intermediate input dimension.
	if (SparkTpShardNameEndsWith(tensor_name,"mlp.gate_proj.weight") ||
		SparkTpShardNameEndsWith(tensor_name,"mlp.up_proj.weight") ||
		SparkTpShardNameEndsWith(tensor_name,"mlp.shared_experts.gate_proj.weight") ||
		SparkTpShardNameEndsWith(tensor_name,"mlp.shared_experts.up_proj.weight"))
		return SPARK_TP_SHARD_CLASS_OUTPUT_DIM;
	if (SparkTpShardNameEndsWith(tensor_name,"mlp.down_proj.weight") ||
		SparkTpShardNameEndsWith(tensor_name,"mlp.shared_experts.down_proj.weight"))
		return SPARK_TP_SHARD_CLASS_INPUT_DIM;
	// MLA latent paths are head-agnostic and replicate on every rank, which is
	// what keeps the latent KV cache identical everywhere. The router and its
	// correction bias replicate so every rank routes identically. Norms, the
	// embedding, and the MTP projections replicate in phase one.
	if (SparkTpShardNameEndsWith(tensor_name,"self_attn.q_a_proj.weight") ||
		SparkTpShardNameEndsWith(tensor_name,"self_attn.kv_a_proj_with_mqa.weight") ||
		SparkTpShardNameEndsWith(tensor_name,"self_attn.q_a_layernorm.weight") ||
		SparkTpShardNameEndsWith(tensor_name,"self_attn.kv_a_layernorm.weight") ||
		SparkTpShardNameEndsWith(tensor_name,"self_attn.indexer.k_norm.weight") ||
		SparkTpShardNameEndsWith(tensor_name,"self_attn.indexer.k_norm.bias") ||
		SparkTpShardNameEndsWith(tensor_name,"self_attn.indexer.weights_proj.weight") ||
		SparkTpShardNameEndsWith(tensor_name,"self_attn.indexer.wk.weight") ||
		SparkTpShardNameEndsWith(tensor_name,"self_attn.indexer.wq_b.weight") ||
		SparkTpShardNameEndsWith(tensor_name,"mlp.gate.weight") ||
		SparkTpShardNameEndsWith(tensor_name,"mlp.gate.e_score_correction_bias") ||
		SparkTpShardNameEndsWith(tensor_name,"input_layernorm.weight") ||
		SparkTpShardNameEndsWith(tensor_name,"post_attention_layernorm.weight") ||
		SparkTpShardNameEndsWith(tensor_name,"model.norm.weight") ||
		SparkTpShardNameEndsWith(tensor_name,"model.embed_tokens.weight") ||
		SparkTpShardNameEndsWith(tensor_name,"enorm.weight") ||
		SparkTpShardNameEndsWith(tensor_name,"hnorm.weight") ||
		SparkTpShardNameEndsWith(tensor_name,"eh_proj.weight"))
		return SPARK_TP_SHARD_CLASS_REPLICATED;
	return SPARK_TP_SHARD_CLASS_UNKNOWN;
}


static uint64_t SparkGlm52TpShardHeadBlock(const char *tensor_name, const SparkTpModelGeometry *geometry)
{
	if (SparkTpShardNameEndsWith(tensor_name,"self_attn.q_b_proj.weight") != 0u)
		return geometry->q_b_head_block;
	if (SparkTpShardNameEndsWith(tensor_name,"self_attn.kv_b_proj.weight") != 0u)
		return geometry->kv_b_head_block;
	return geometry->o_proj_head_block;
}

SparkStatus SparkGlm52TpShardComputeView(const SparkGlm52StagePackTensorSpec *spec, const SparkTpShapeDescriptor *shape, const SparkTpModelGeometry *geometry, SparkTpShardView *view_out)
{
	return SparkTpShardComputeView(SparkGlm52TpShardClassifyTensor, SparkGlm52TpShardHeadBlock,spec,shape,geometry,view_out);
}

uint64_t SparkGlm52TpShardGeometryHash(const SparkGlm52StagePackTensorSpec *spec, const SparkTpShapeDescriptor *shape, const SparkTpShardView *view)
{
	return SparkTpShardGeometryHash(spec,shape,view);
}

SparkStatus SparkGlm52TpShardReadTensor(const char *stagepack_root, const SparkGlm52StagePackTensorSpec *spec, const SparkTpShapeDescriptor *shape, const SparkTpModelGeometry *geometry, void *destination, uint64_t destination_bytes, SparkTpShardView *view_out)
{
	return SparkTpShardReadTensor(SparkGlm52TpShardClassifyTensor, SparkGlm52TpShardHeadBlock,stagepack_root,spec,shape,geometry,destination,destination_bytes,view_out);
}
