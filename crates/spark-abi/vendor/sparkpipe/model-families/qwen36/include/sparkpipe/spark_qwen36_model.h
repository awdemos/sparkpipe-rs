// Qwen 3.6's model geometry for the host tiers, mirroring the values in
// inference/llms/qwen_3_6/config.h. The firmware config is the source
// of truth; a conformance check holds the two in lockstep.
#ifndef SPARKPIPE_SPARK_QWEN36_MODEL_H
#define SPARKPIPE_SPARK_QWEN36_MODEL_H

#define SPARK_QWEN36_MODEL_HIDDEN_DIMENSION 5120u
#define SPARK_QWEN36_MODEL_LAYER_COUNT 64u
#define SPARK_QWEN36_MODEL_VOCAB_COUNT 248320u
#define SPARK_QWEN36_MODEL_ATTENTION_HEAD_COUNT 24u
#define SPARK_QWEN36_MODEL_KV_HEAD_COUNT 4u
#define SPARK_QWEN36_MODEL_HEAD_DIMENSION 256u
#define SPARK_QWEN36_MODEL_GDN_KEY_HEAD_COUNT 16u
#define SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT 48u

#endif
