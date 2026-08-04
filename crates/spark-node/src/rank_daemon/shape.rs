//! Shape-driven node configuration derivation.
//!
//! Generic (model-agnostic) port of `SparkGlm52ShapeDeriveNodeConfig` from
//! `model-families/glm52/src/spark_glm52_shape_config.c`. All model geometry
//! arrives through [`TpModelGeometry`] / [`ShapeModelInputs`] — no GLM52
//! constants (port deviation #1). The `configuration_hash` is bit-exact with
//! the C derivation: FNV-1a (seed `1469598103934665603`, exactly as the C
//! call site — one digit shorter than the standard offset basis) over the
//! C struct byte layouts in little-endian order, pinned by parity tests
//! against values produced by the C tree.

use super::status::{Result, SparkStatus};

/// C: `SPARK_TP_SHARD_ABI_VERSION`.
pub const TP_SHARD_ABI_VERSION: u32 = 1;
/// C: `SPARK_GLM52_SHAPE_CONFIG_ABI_VERSION`.
pub const SHAPE_CONFIG_ABI_VERSION: u32 = 1;

/// Hash seed used by the C `SparkGlm52ShapeDeriveNodeConfig` call site
/// (note: *not* the standard FNV-1a 64-bit offset basis — the C literal is
/// `1469598103934665603u`).
const SHAPE_HASH_SEED: u64 = 1469598103934665603;
const FNV_PRIME: u64 = 1099511628211;

/// C: `SparkHashBytes` (FNV-1a continuation over raw bytes).
fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

/// C: `SparkTpShapeDescriptor` — the inference shape a node serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpShapeDescriptor {
    pub abi_version: u32,
    pub tp_degree: u32,
    pub tp_rank: u32,
    pub pp_stage_count: u32,
    pub pp_stage_index: u32,
}

impl TpShapeDescriptor {
    pub fn new(tp_degree: u32, tp_rank: u32, pp_stage_count: u32, pp_stage_index: u32) -> Self {
        Self {
            abi_version: TP_SHARD_ABI_VERSION,
            tp_degree,
            tp_rank,
            pp_stage_count,
            pp_stage_index,
        }
    }

    /// C-layout serialization (5 × u32) for the configuration hash.
    fn hash_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(20);
        push_u32(&mut bytes, self.abi_version);
        push_u32(&mut bytes, self.tp_degree);
        push_u32(&mut bytes, self.tp_rank);
        push_u32(&mut bytes, self.pp_stage_count);
        push_u32(&mut bytes, self.pp_stage_index);
        bytes
    }
}

/// C: `SparkTpModelGeometry` — head-block geometry for the attention
/// projections, supplied by the caller from the model contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpModelGeometry {
    pub abi_version: u32,
    /// C ref (GLM52): 64.
    pub head_count: u32,
    /// C ref (GLM52): qk_nope + rope = 256.
    pub q_b_head_block: u32,
    /// C ref (GLM52): qk_nope + value = 448.
    pub kv_b_head_block: u32,
    /// C ref (GLM52): value per head = 256.
    pub o_proj_head_block: u32,
}

impl TpModelGeometry {
    pub fn new(
        head_count: u32,
        q_b_head_block: u32,
        kv_b_head_block: u32,
        o_proj_head_block: u32,
    ) -> Self {
        Self {
            abi_version: TP_SHARD_ABI_VERSION,
            head_count,
            q_b_head_block,
            kv_b_head_block,
            o_proj_head_block,
        }
    }

    fn hash_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(20);
        push_u32(&mut bytes, self.abi_version);
        push_u32(&mut bytes, self.head_count);
        push_u32(&mut bytes, self.q_b_head_block);
        push_u32(&mut bytes, self.kv_b_head_block);
        push_u32(&mut bytes, self.o_proj_head_block);
        bytes
    }
}

/// C: `SparkGlm52ShapeModelInputs` (model-generic despite the C name).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeModelInputs {
    pub abi_version: u32,
    pub total_layer_count: u32,
    pub hidden_dimension: u32,
    pub moe_intermediate_dimension: u32,
    pub dense_intermediate_dimension: u32,
    pub kv_latent_plus_rope_dimension: u32,
    pub kv_bytes_per_element: u32,
}

impl ShapeModelInputs {
    pub fn new(
        total_layer_count: u32,
        hidden_dimension: u32,
        moe_intermediate_dimension: u32,
        dense_intermediate_dimension: u32,
        kv_latent_plus_rope_dimension: u32,
        kv_bytes_per_element: u32,
    ) -> Self {
        Self {
            abi_version: SHAPE_CONFIG_ABI_VERSION,
            total_layer_count,
            hidden_dimension,
            moe_intermediate_dimension,
            dense_intermediate_dimension,
            kv_latent_plus_rope_dimension,
            kv_bytes_per_element,
        }
    }

    fn hash_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(28);
        push_u32(&mut bytes, self.abi_version);
        push_u32(&mut bytes, self.total_layer_count);
        push_u32(&mut bytes, self.hidden_dimension);
        push_u32(&mut bytes, self.moe_intermediate_dimension);
        push_u32(&mut bytes, self.dense_intermediate_dimension);
        push_u32(&mut bytes, self.kv_latent_plus_rope_dimension);
        push_u32(&mut bytes, self.kv_bytes_per_element);
        bytes
    }
}

/// C: `SparkGlm52ShapeNodeConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ShapeNodeConfig {
    pub abi_version: u32,
    pub first_layer_index: u32,
    pub layer_count: u32,
    pub heads_per_rank: u32,
    pub moe_intermediate_per_rank: u32,
    pub dense_intermediate_per_rank: u32,
    pub q_b_output_per_rank: u32,
    pub kv_b_output_per_rank: u32,
    pub o_proj_input_per_rank: u32,
    pub reserved0: u32,
    pub kv_bytes_per_token: u64,
    pub configuration_hash: u64,
}

impl ShapeNodeConfig {
    /// C-layout serialization of `[0, offsetof(configuration_hash))`
    /// (10 × u32 + 1 × u64 = 48 bytes) for the configuration hash.
    fn hash_prefix_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(48);
        push_u32(&mut bytes, self.abi_version);
        push_u32(&mut bytes, self.first_layer_index);
        push_u32(&mut bytes, self.layer_count);
        push_u32(&mut bytes, self.heads_per_rank);
        push_u32(&mut bytes, self.moe_intermediate_per_rank);
        push_u32(&mut bytes, self.dense_intermediate_per_rank);
        push_u32(&mut bytes, self.q_b_output_per_rank);
        push_u32(&mut bytes, self.kv_b_output_per_rank);
        push_u32(&mut bytes, self.o_proj_input_per_rank);
        push_u32(&mut bytes, self.reserved0);
        bytes.extend_from_slice(&self.kv_bytes_per_token.to_le_bytes());
        bytes
    }
}

/// C: `SparkGlm52ShapeDeriveNodeConfig`. Fails closed on the same conditions,
/// in the same order: ABI versions, unsupported TP degree, rank/stage out of
/// range, uneven layer split, indivisible head/intermediate dimensions.
pub fn derive_node_config(
    shape: &TpShapeDescriptor,
    geometry: &TpModelGeometry,
    inputs: &ShapeModelInputs,
) -> Result<ShapeNodeConfig> {
    if shape.abi_version != TP_SHARD_ABI_VERSION
        || geometry.abi_version != TP_SHARD_ABI_VERSION
        || inputs.abi_version != SHAPE_CONFIG_ABI_VERSION
    {
        return Err(SparkStatus::InvalidArgument);
    }
    if !matches!(shape.tp_degree, 1 | 2 | 4 | 8 | 16) {
        return Err(SparkStatus::InvalidArgument);
    }
    if shape.tp_rank >= shape.tp_degree
        || shape.pp_stage_count == 0
        || shape.pp_stage_index >= shape.pp_stage_count
    {
        return Err(SparkStatus::InvalidArgument);
    }
    if inputs.total_layer_count == 0 || inputs.total_layer_count % shape.pp_stage_count != 0 {
        return Err(SparkStatus::InvalidArgument);
    }
    if geometry.head_count == 0
        || geometry.head_count % shape.tp_degree != 0
        || inputs.moe_intermediate_dimension % shape.tp_degree != 0
        || inputs.dense_intermediate_dimension % shape.tp_degree != 0
    {
        return Err(SparkStatus::InvalidArgument);
    }
    let mut config =
        ShapeNodeConfig { abi_version: SHAPE_CONFIG_ABI_VERSION, ..ShapeNodeConfig::default() };
    config.layer_count = inputs.total_layer_count / shape.pp_stage_count;
    config.first_layer_index = shape.pp_stage_index * config.layer_count;
    config.heads_per_rank = geometry.head_count / shape.tp_degree;
    config.moe_intermediate_per_rank = inputs.moe_intermediate_dimension / shape.tp_degree;
    config.dense_intermediate_per_rank = inputs.dense_intermediate_dimension / shape.tp_degree;
    config.q_b_output_per_rank = config.heads_per_rank * geometry.q_b_head_block;
    config.kv_b_output_per_rank = config.heads_per_rank * geometry.kv_b_head_block;
    config.o_proj_input_per_rank = config.heads_per_rank * geometry.o_proj_head_block;
    // Every TP rank of a stage runs all of the stage's layers against the
    // full head-agnostic latent; the replication is across the TP group.
    config.kv_bytes_per_token = u64::from(config.layer_count)
        * u64::from(inputs.kv_latent_plus_rope_dimension)
        * u64::from(inputs.kv_bytes_per_element);
    let mut hash = SHAPE_HASH_SEED;
    hash = hash_bytes(hash, &shape.hash_bytes());
    hash = hash_bytes(hash, &geometry.hash_bytes());
    hash = hash_bytes(hash, &inputs.hash_bytes());
    hash = hash_bytes(hash, &config.hash_prefix_bytes());
    config.configuration_hash = hash;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GLM52-reference geometry, supplied through config per port
    /// deviation #1 (values from the C tree's `spark_glm52_model.h`).
    fn glm52_geometry() -> TpModelGeometry {
        TpModelGeometry::new(64, 192 + 64, 192 + 256, 256)
    }

    fn glm52_inputs() -> ShapeModelInputs {
        ShapeModelInputs::new(78, 6144, 2048, 12288, 512 + 64, 1)
    }

    #[test]
    fn derive_tp1_stage5_matches_c_hash() {
        let shape = TpShapeDescriptor::new(1, 0, 13, 5);
        let config = derive_node_config(&shape, &glm52_geometry(), &glm52_inputs()).unwrap();
        // Reference value computed by the C SparkGlm52ShapeDeriveNodeConfig.
        assert_eq!(config.configuration_hash, 2005884353472861467);
        assert_eq!(config.first_layer_index, 30);
        assert_eq!(config.layer_count, 6);
        assert_eq!(config.kv_bytes_per_token, 3456);
    }

    #[test]
    fn derive_tp2_stage3of6_matches_c_hash() {
        let shape = TpShapeDescriptor::new(2, 1, 6, 3);
        let config = derive_node_config(&shape, &glm52_geometry(), &glm52_inputs()).unwrap();
        assert_eq!(config.configuration_hash, 9169625037557488130);
        assert_eq!(config.first_layer_index, 39);
        assert_eq!(config.layer_count, 13);
        assert_eq!(config.heads_per_rank, 32);
        assert_eq!(config.moe_intermediate_per_rank, 1024);
        assert_eq!(config.kv_bytes_per_token, 7488);
    }

    #[test]
    fn derive_rejects_invalid_shapes() {
        let geometry = glm52_geometry();
        let inputs = glm52_inputs();
        // Unsupported TP degree.
        assert_eq!(
            derive_node_config(&TpShapeDescriptor::new(3, 0, 13, 0), &geometry, &inputs),
            Err(SparkStatus::InvalidArgument)
        );
        // Rank out of range.
        assert_eq!(
            derive_node_config(&TpShapeDescriptor::new(2, 2, 13, 0), &geometry, &inputs),
            Err(SparkStatus::InvalidArgument)
        );
        // Stage index out of range / zero stage count.
        assert_eq!(
            derive_node_config(&TpShapeDescriptor::new(1, 0, 13, 13), &geometry, &inputs),
            Err(SparkStatus::InvalidArgument)
        );
        assert_eq!(
            derive_node_config(&TpShapeDescriptor::new(1, 0, 0, 0), &geometry, &inputs),
            Err(SparkStatus::InvalidArgument)
        );
        // Uneven layer split (78 % 5 != 0).
        assert_eq!(
            derive_node_config(&TpShapeDescriptor::new(1, 0, 5, 0), &geometry, &inputs),
            Err(SparkStatus::InvalidArgument)
        );
        // TP degree 16 is legal and divides every GLM52 dimension; it
        // derives (heads 4/rank, MoE intermediate 128/rank).
        let config =
            derive_node_config(&TpShapeDescriptor::new(16, 0, 13, 0), &geometry, &inputs).unwrap();
        assert_eq!(config.heads_per_rank, 4);
        assert_eq!(config.moe_intermediate_per_rank, 128);
        // Wrong ABI versions.
        let mut bad_shape = TpShapeDescriptor::new(1, 0, 13, 0);
        bad_shape.abi_version = 99;
        assert_eq!(
            derive_node_config(&bad_shape, &geometry, &inputs),
            Err(SparkStatus::InvalidArgument)
        );
    }
}
