#ifndef SPARKPIPE_SPARK_GLM52_RUNTIME_CONTRACT_H
#define SPARKPIPE_SPARK_GLM52_RUNTIME_CONTRACT_H

#include <string.h>

#include "sparkpipe/spark_model_runtime.h"

static inline SparkModelRuntimeContract SparkGlm52RuntimeContract(void)
{
    SparkModelRuntimeContract contract;

    memset(&contract,0,sizeof(contract));
    contract.abi_version = SPARK_MODEL_RUNTIME_ABI_VERSION;
    contract.descriptor_bytes = SPARK_MODEL_RUNTIME_CONTRACT_BYTES;
    contract.contract_id = "glm-5.2-fp8-expert-bf16-rest-v1";
    contract.precision_policy.expert_weight_precision =
        SPARK_MODEL_RUNTIME_PRECISION_FP8_E4M3;
    contract.precision_policy.expert_activation_precision =
        SPARK_MODEL_RUNTIME_PRECISION_BF16;
    contract.precision_policy.nonexpert_weight_precision =
        SPARK_MODEL_RUNTIME_PRECISION_BF16;
    contract.precision_policy.nonexpert_activation_precision =
        SPARK_MODEL_RUNTIME_PRECISION_BF16;
    contract.precision_policy.accumulator_precision =
        SPARK_MODEL_RUNTIME_PRECISION_FP32;
    contract.required_operation_mask =
        SPARK_MODEL_RUNTIME_OPERATION_GLOBAL_ATTENTION |
        SPARK_MODEL_RUNTIME_OPERATION_DENSE_MLP |
        SPARK_MODEL_RUNTIME_OPERATION_ROUTER |
        SPARK_MODEL_RUNTIME_OPERATION_ROUTED_MOE |
        SPARK_MODEL_RUNTIME_OPERATION_SPARSE_INDEX |
        SPARK_MODEL_RUNTIME_OPERATION_OUTPUT_HEAD;
    return contract;
}

static inline SparkStatus SparkGlm52RuntimeValidateContract(
    const SparkModelRuntimeContract *contract)
{
    SparkModelRuntimeContract expected_contract;

    expected_contract = SparkGlm52RuntimeContract();
    return SparkModelRuntimeContractsMatch(contract,&expected_contract) != 0u ?
        SPARK_STATUS_OK : SPARK_STATUS_VALIDATION_FAILED;
}

#endif
