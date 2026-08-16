#ifndef SPARKPIPE_SPARK_OPTIONS_H
#define SPARKPIPE_SPARK_OPTIONS_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "sparkpipe/spark_status.h"

// Declarative command-line options: one table row per option instead of one
// if-strcmp block per option. A row names the flag, the kind, the byte
// offset of its destination in the caller's configuration struct, the
// per-option error code the old chains already assigned, and - for the
// U32 kind - an optional companion is_set flag's offset. The parser is
// ~40 lines once; the twin 147- and 270-line chains in the two node
// daemons become tables the eye can diff against --help.

#define SPARK_OPTION_NO_SET_FLAG 0xffffu

typedef enum SparkOptionKind
{
	SPARK_OPTION_U32 = 1,
	SPARK_OPTION_U64 = 2,
	SPARK_OPTION_STRING = 3,
	SPARK_OPTION_FLAG = 4,
	SPARK_OPTION_QUANT = 5
} SparkOptionKind;

typedef struct SparkOption
{
	const char *name;
	uint16_t kind;
	int16_t error;
	uint16_t destination_offset;
	uint16_t set_flag_offset;
} SparkOption;

// the one custom kind both daemons share; defined by ring_runtime
SparkStatus SparkRingRuntimeParseQuantizationMode(const char *text, uint32_t *mode);
int32_t SparkNetParseU32(const char *text, uint32_t *value);

static inline int32_t SparkOptionsParseU64(const char *text, uint64_t *value)
{
	char *end;
	unsigned long long parsed;
	if (text == 0 || text[0] == 0)
		return -1;
	parsed = strtoull(text, &end, 10);
	if (*end != 0)
		return -1;
	*value = (uint64_t)parsed;
	return 0;
}

// Applies argv[*index] against the table. Returns 0 on a match (advancing
// *index past any value), the option's own error on a malformed value, and
// 1 when no row matches so the caller can try its residual specials.
static inline int32_t SparkOptionsApply(const SparkOption *options, uint32_t option_count, void *configuration, int argc, char **argv, int32_t *index)
{
	uint32_t at;
	for (at = 0u; at < option_count; ++at)
	{
		const SparkOption *option = &options[at];
		uint8_t *destination = (uint8_t *)configuration + option->destination_offset;
		if (strcmp(argv[*index], option->name) != 0)
			continue;
		if (option->kind == SPARK_OPTION_FLAG)
		{
			*(uint32_t *)destination = 1u;
			return 0;
		}
		if ((*index + 1) >= argc)
			return option->error;
		if (option->kind == SPARK_OPTION_STRING)
			*(const char **)destination = argv[*index + 1];
		else if (option->kind == SPARK_OPTION_U32)
		{
			if (SparkNetParseU32(argv[*index + 1], (uint32_t *)destination) < 0)
				return option->error;
			if (option->set_flag_offset != SPARK_OPTION_NO_SET_FLAG)
				*(uint32_t *)((uint8_t *)configuration + option->set_flag_offset) = 1u;
		}
		else if (option->kind == SPARK_OPTION_U64)
		{
			if (SparkOptionsParseU64(argv[*index + 1], (uint64_t *)destination) < 0)
				return option->error;
		}
		else if (option->kind == SPARK_OPTION_QUANT)
		{
			if (SparkRingRuntimeParseQuantizationMode(argv[*index + 1], (uint32_t *)destination) != SPARK_STATUS_OK)
				return option->error;
		}
		*index += 1;
		return 0;
	}
	return 1;
}

#endif
