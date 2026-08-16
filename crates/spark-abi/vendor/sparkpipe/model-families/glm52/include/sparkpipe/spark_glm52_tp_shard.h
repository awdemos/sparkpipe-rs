#ifndef SPARKPIPE_SPARK_GLM52_TP_SHARD_H
#define SPARKPIPE_SPARK_GLM52_TP_SHARD_H

#include <stdint.h>

#include "sparkpipe/spark_tp_shard.h"
#include "sparkpipe/spark_glm52_stagepack.h"

// The glm table over the common shard engine. Types, classes, validation and
// arithmetic are include/sparkpipe/spark_tp_shard.h's; this header keeps the
// original entry points standing on top of them so consumers did not move.

SparkTpShardClass SparkGlm52TpShardClassifyTensor(const char *tensor_name);

SparkStatus SparkGlm52TpShardComputeView(const SparkGlm52StagePackTensorSpec *spec, const SparkTpShapeDescriptor *shape, const SparkTpModelGeometry *geometry, SparkTpShardView *view_out);

uint64_t SparkGlm52TpShardGeometryHash(const SparkGlm52StagePackTensorSpec *spec, const SparkTpShapeDescriptor *shape, const SparkTpShardView *view);

SparkStatus SparkGlm52TpShardReadTensor(const char *stagepack_root, const SparkGlm52StagePackTensorSpec *spec, const SparkTpShapeDescriptor *shape, const SparkTpModelGeometry *geometry, void *destination, uint64_t destination_bytes, SparkTpShardView *view_out);

#endif
