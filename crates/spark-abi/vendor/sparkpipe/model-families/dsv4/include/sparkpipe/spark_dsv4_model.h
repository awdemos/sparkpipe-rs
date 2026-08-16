// DeepSeek V4's model geometry for the host tiers, mirroring the values
// in inference/llms/deepseek_v4/config.h. The firmware config is the
// source of truth; a conformance check holds the two in lockstep.
#ifndef SPARKPIPE_SPARK_DSV4_MODEL_H
#define SPARKPIPE_SPARK_DSV4_MODEL_H

#define SPARK_DSV4_MODEL_HIDDEN_DIMENSION 4096u
#define SPARK_DSV4_MODEL_LAYER_COUNT 43u
#define SPARK_DSV4_MODEL_VOCAB_COUNT 129280u
#define SPARK_DSV4_MODEL_ATTENTION_HEAD_COUNT 64u
#define SPARK_DSV4_MODEL_KV_HEAD_COUNT 1u
#define SPARK_DSV4_MODEL_HEAD_DIMENSION 512u
#define SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT 256u

#endif
