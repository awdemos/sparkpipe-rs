//! Build `RingModelGeometry` from a Kimi K2.7 contract.
//!
//! This is the GPU-free milestone geometry builder. Deployment topology
//! (stage count, host table, pack-file naming) is not yet present in the
//! contract, so we keep the GLM52 reference defaults for those fields and
//! derive the model-specific dimensions from the k27 sections.

use spark_model::K27Contract;
use spark_sched::stage_plan::MAX_BATCH_BUCKET;

use crate::rank_daemon::ring_runtime::{RingModelGeometry, RingPackLayout};
use crate::rank_daemon::shape::{ShapeModelInputs, TpModelGeometry};
use crate::rank_daemon::status::{Result, SparkStatus};

/// Placeholder deployment topology for the GPU-free milestone. The contract
/// does not yet carry a deployment table, so we reuse the GLM52 reference
/// host naming and derive a stage count that divides the layer count and
/// fits the scheduler's `MAX_STAGE_COUNT` bound. For k27 this yields a
/// single stage because 61 is prime and `MAX_STAGE_COUNT == 13`.
const DEFAULT_HOST_PREFIX: &str = "10.10.100.";
const DEFAULT_HOST_INDEX_BASE: u32 = 10;
const DEFAULT_DSA_SELECTED_TOKEN_COUNT: u32 = 2048;
const DEFAULT_MAX_SPECULATIVE_ROWS_PER_LANE: u32 = 8;
const BF16_BYTES_PER_ELEMENT: u32 = 2;

/// Distribute `layer_count` layers across `stage_count` stages as evenly as
/// possible. Every stage gets at least `layer_count / stage_count` layers and
/// the first `layer_count % stage_count` stages get one extra.
fn choose_stage_count(layer_count: u32) -> u32 {
    let max = spark_sched::stage_plan::MAX_STAGE_COUNT.min(layer_count);
    for candidate in (1..=max).rev() {
        if layer_count % candidate == 0 {
            return candidate;
        }
    }
    1
}

fn distribute_layers(layer_count: u32, stage_count: u32) -> Result<Vec<u32>> {
    if stage_count == 0 || layer_count == 0 || layer_count < stage_count {
        return Err(SparkStatus::InvalidArgument);
    }
    let base = layer_count / stage_count;
    let extra = layer_count % stage_count;
    let mut counts = Vec::with_capacity(stage_count as usize);
    for stage_index in 0..stage_count {
        let mut count = base;
        if stage_index < extra {
            count += 1;
        }
        counts.push(count);
    }
    Ok(counts)
}

/// Build a `RingModelGeometry` from the k27 contract.
///
/// The model-specific fields come directly from the contract. Deployment
/// topology defaults are documented above and will move into the contract or
/// a separate cluster config once that work lands.
pub fn ring_model_geometry(contract: &K27Contract) -> Result<RingModelGeometry> {
    let layer_count =
        u32::try_from(contract.model.layer_count).map_err(|_| SparkStatus::InvalidArgument)?;
    let first_routed_layer = u32::try_from(contract.model.first_routed_layer)
        .map_err(|_| SparkStatus::InvalidArgument)?;
    let hidden_dimension =
        u32::try_from(contract.model.hidden_dimension).map_err(|_| SparkStatus::InvalidArgument)?;
    let moe_intermediate_dimension = u32::try_from(contract.moe.expert_intermediate_dimension)
        .map_err(|_| SparkStatus::InvalidArgument)?;
    let dense_intermediate_dimension = u32::try_from(contract.moe.dense_intermediate_dimension)
        .map_err(|_| SparkStatus::InvalidArgument)?;
    let kv_latent_plus_rope_dimension = u32::try_from(
        contract
            .mla
            .kv_lora_rank
            .checked_add(contract.mla.qk_unrotated_dimension)
            .ok_or(SparkStatus::InvalidArgument)?,
    )
    .map_err(|_| SparkStatus::InvalidArgument)?;
    let query_head_count =
        u32::try_from(contract.mla.query_head_count).map_err(|_| SparkStatus::InvalidArgument)?;
    let qk_nope_dimension =
        u32::try_from(contract.mla.qk_nope_dimension).map_err(|_| SparkStatus::InvalidArgument)?;
    let qk_unrotated_dimension = u32::try_from(contract.mla.qk_unrotated_dimension)
        .map_err(|_| SparkStatus::InvalidArgument)?;
    let value_head_dimension = u32::try_from(contract.mla.value_head_dimension)
        .map_err(|_| SparkStatus::InvalidArgument)?;
    let maximum_context_tokens = u32::try_from(contract.model.maximum_context_tokens)
        .map_err(|_| SparkStatus::InvalidArgument)?;

    let stage_count = choose_stage_count(layer_count);
    let default_stage_layer_counts = distribute_layers(layer_count, stage_count)?;

    let geometry = RingModelGeometry {
        layer_count,
        first_routed_layer,
        weight_layer_count: layer_count + 1,
        hidden_dimension,
        hidden_bf16_bytes_per_sequence: hidden_dimension * BF16_BYTES_PER_ELEMENT,
        maximum_context_tokens,
        dsa_selected_token_count: DEFAULT_DSA_SELECTED_TOKEN_COUNT,
        dsa_selected_index_bytes_per_sequence: DEFAULT_DSA_SELECTED_TOKEN_COUNT * 4,
        max_speculative_rows_per_lane: DEFAULT_MAX_SPECULATIVE_ROWS_PER_LANE,
        max_batch_bucket: MAX_BATCH_BUCKET,
        shape_inputs: ShapeModelInputs::new(
            layer_count,
            hidden_dimension,
            moe_intermediate_dimension,
            dense_intermediate_dimension,
            kv_latent_plus_rope_dimension,
            BF16_BYTES_PER_ELEMENT,
        ),
        tp_geometry: TpModelGeometry::new(
            query_head_count,
            qk_nope_dimension + qk_unrotated_dimension,
            qk_nope_dimension + value_head_dimension,
            value_head_dimension,
        ),
        stage_count,
        default_stage_layer_counts,
        host_prefix: DEFAULT_HOST_PREFIX.to_string(),
        host_index_base: DEFAULT_HOST_INDEX_BASE,
        pack_layout: RingPackLayout::default(),
        max_routed_layers_per_stage: layer_count - first_routed_layer,
    };
    geometry.validate()?;
    Ok(geometry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_model::ContractDocument;
    use std::path::PathBuf;

    fn k27_contract() -> K27Contract {
        let root = std::env::var("SPARKPIPE_C_ROOT").map(PathBuf::from).unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join("..")
                .join("sparkpipe")
        });
        let document = ContractDocument::from_path(
            &root.join("model_contracts").join("k27_authoritative.json"),
        )
        .expect("k27 contract must parse");
        document.as_k27().expect("k27 contract must validate")
    }

    #[test]
    fn geometry_validates() {
        let contract = k27_contract();
        let geometry = ring_model_geometry(&contract).expect("geometry must validate");
        assert_eq!(geometry.layer_count, 61);
        assert_eq!(geometry.first_routed_layer, 1);
        assert_eq!(geometry.hidden_dimension, 7168);
        assert_eq!(geometry.hidden_bf16_bytes_per_sequence, 14336);
        assert_eq!(geometry.stage_count, 1);
        assert_eq!(geometry.default_stage_layer_counts, vec![61]);
        assert_eq!(geometry.shape_inputs.kv_latent_plus_rope_dimension, 576);
        assert_eq!(geometry.tp_geometry.head_count, 64);
        assert_eq!(geometry.tp_geometry.q_b_head_block, 192);
        assert_eq!(geometry.tp_geometry.kv_b_head_block, 256);
        assert_eq!(geometry.tp_geometry.o_proj_head_block, 128);
    }
}
