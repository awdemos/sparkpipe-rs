// The request-model seam contract: twelve entries every provider
// implements. The glm drafter module answers them with a real
// speculator; the null provider answers them fail-closed for families
// without one. The symbol-parity gate keeps every provider's extern
// set equal to this contract.
#ifndef SPARKPIPE_SPARK_REQUEST_MODEL_H
#define SPARKPIPE_SPARK_REQUEST_MODEL_H

#include "sparkpipe/spark_request_api.h"

#ifdef __cplusplus
extern "C" {
#endif

uint32_t SparkRequestModelSlotCanSpeculate(const SparkRequestApi *api, const SparkRequestApiSlot *slot);
uint32_t SparkRequestModelSpeculatorIsValid(const struct SparkRequestModelSpeculator *speculator);
SparkStatus SparkRequestModelGetDraft(SparkRequestApi *api, uint64_t sequence_id, SparkRequestModelDraftResult *result);
SparkStatus SparkRequestModelMarkVerifierTapsReady(SparkRequestApi *api, uint64_t request_id, uint64_t sequence_id, uint64_t sequence_position, uint64_t *tap_generation_out);
uint32_t SparkRequestModelDefaultSpeculativeTokenCount(const SparkRequestApi *api);
SparkStatus SparkRequestModelCompleteVerify(SparkRequestApi *api, uint64_t sequence_id, const SparkRequestModelVerifyResult *verify_result);
SparkStatus SparkRequestModelResolveVerifierTokens(const uint32_t *draft_token_ids, uint32_t draft_token_count, const uint32_t *verifier_token_ids, uint32_t verifier_token_count, SparkRequestModelVerifyResult *verify_result);
SparkStatus SparkRequestModelPrepareDraftForSlot(SparkRequestApi *api, SparkRequestApiSlot *slot);
SparkStatus SparkRequestModelCancelSequence(SparkRequestApi *api, uint64_t sequence_id);
uint32_t SparkRequestModelSpeculationIsEnabled(
    const SparkRequestApi *api);
SparkStatus SparkRequestModelGetSlotSpeculativeDraft(
    SparkRequestApi *api,
    const SparkRequestApiSlot *slot,
    uint32_t preferred_source,
    SparkRequestModelDraftResult *draft_result,
    uint32_t *source_out);
uint32_t SparkRequestModelMtpOutranksPlainDecode(
    const SparkRequestApi *api,
    uint32_t plain_request_count,
    uint32_t mtp_request_count);

#ifdef __cplusplus
}
#endif

#endif
