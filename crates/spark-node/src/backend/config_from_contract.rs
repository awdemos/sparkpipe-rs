//! Build `BackendConfig` from a Kimi K2.7 contract.
//!
//! Like [`super::k27_geometry`], this is the GPU-free milestone builder.
//! Capacity defaults that are not in the contract (request/event capacity,
//! work-queue size, port base, etc.) mirror the GLM52 reference values and
//! will move into a deployment/cluster config later.

use spark_model::K27Contract;
use spark_serve::serving_engine::ServingStatus;

use super::k27_geometry::ring_model_geometry;
use super::state::BackendConfig;

/// Default capacities that are not yet modeled in the k27 contract.
const DEFAULT_KV_POOL_TOKENS: u32 = 4_194_304;
const DEFAULT_PREFILL_WAVE_TOKENS: u32 = 256;
const DEFAULT_REQUEST_CAPACITY: u32 = 13_312;
const DEFAULT_EVENT_CAPACITY: u32 = 16_384;
const DEFAULT_WORK_QUEUE_CAPACITY: usize = 512;
const DEFAULT_MAX_PIPELINE_SLOT_COUNT: u32 = 1_024;
const DEFAULT_MAX_RESIDENT_SEQUENCE_COUNT: u32 = 16_384;
const DEFAULT_DEFAULT_MAX_ACTIVE: u32 = 1_024;
const DEFAULT_PORT_BASE: u32 = 52_100;
const DEFAULT_OUTPUT_TOKEN_BUDGET: u32 = 1_024;
const DEFAULT_PREFETCH_LANE_COUNT: u32 = 13;
const DEFAULT_MTP_DRAFT_TOKEN_COUNT: u32 = 6;
const DEFAULT_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT: u32 = 7;
const DEFAULT_MEASURED_PROFILE_ID: u32 = spark_sched::stage_plan::MEASURED_PROFILE_20260701;
const DEFAULT_METADATA_KEY_BASE: u64 = 0x1_0000_0000;
const DEFAULT_METADATA_VALUE_BASE: u64 = 0x2_0000_0000;

impl BackendConfig {
    /// Build a backend configuration from a Kimi K2.7 contract.
    ///
    /// # Geometry derivation
    /// - Model dimensions come from the contract (`model`, `mla`, `moe`).
    /// - KV arena uses a single "head" whose dimension is the full MLA latent
    ///   size (`kv_lora_rank + qk_unrotated_dimension`) because the arena's
    ///   `kv_head_count * head_dim` product is the per-token-per-layer byte
    ///   stride. This will be revisited when the CUDA backend consumes MLA
    ///   latent caches directly.
    /// - Deployment topology (stage count, host table, pack naming) uses the
    ///   GLM52-reference placeholder defaults.
    pub fn from_k27_contract(contract: &K27Contract) -> Result<Self, ServingStatus> {
        let geometry = ring_model_geometry(contract).map_err(|_| ServingStatus::InvalidArgument)?;

        let context_tokens = u32::try_from(contract.model.maximum_context_tokens)
            .map_err(|_| ServingStatus::InvalidArgument)?;
        let kv_block_tokens = u32::try_from(contract.cache.kv_page_slots)
            .map_err(|_| ServingStatus::InvalidArgument)?;
        let output_vocab_count = u32::try_from(contract.model.vocabulary_size)
            .map_err(|_| ServingStatus::InvalidArgument)?;
        let end_of_text = u32::try_from(contract.tokens.end_of_text)
            .map_err(|_| ServingStatus::InvalidArgument)?;
        let kv_latent_dim = u32::try_from(
            contract
                .mla
                .kv_lora_rank
                .checked_add(contract.mla.qk_unrotated_dimension)
                .ok_or(ServingStatus::InvalidArgument)?,
        )
        .map_err(|_| ServingStatus::InvalidArgument)?;

        let config = BackendConfig {
            geometry,
            context_tokens,
            kv_block_tokens,
            kv_pool_tokens: DEFAULT_KV_POOL_TOKENS,
            prefill_wave_tokens: DEFAULT_PREFILL_WAVE_TOKENS,
            builder_max_prefill_tokens: DEFAULT_PREFILL_WAVE_TOKENS,
            request_capacity: DEFAULT_REQUEST_CAPACITY,
            event_capacity: DEFAULT_EVENT_CAPACITY,
            work_queue_capacity: DEFAULT_WORK_QUEUE_CAPACITY,
            mtp_draft_token_count: DEFAULT_MTP_DRAFT_TOKEN_COUNT,
            dspark_max_speculative_token_count: DEFAULT_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT,
            output_vocab_count,
            kv_head_count: 1,
            kv_head_dim: kv_latent_dim,
            prefetch_lane_count: DEFAULT_PREFETCH_LANE_COUNT,
            max_pipeline_slot_count: DEFAULT_MAX_PIPELINE_SLOT_COUNT,
            max_resident_sequence_count: DEFAULT_MAX_RESIDENT_SEQUENCE_COUNT,
            default_max_active: DEFAULT_DEFAULT_MAX_ACTIVE,
            default_port_base: DEFAULT_PORT_BASE,
            default_output_token_budget: DEFAULT_OUTPUT_TOKEN_BUDGET,
            measured_profile_id: DEFAULT_MEASURED_PROFILE_ID,
            stop_token_ids: vec![end_of_text],
            metadata_key_base: DEFAULT_METADATA_KEY_BASE,
            metadata_value_base: DEFAULT_METADATA_VALUE_BASE,
        };
        config.validate()?;
        Ok(config)
    }
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
    fn config_validates() {
        let contract = k27_contract();
        let config = BackendConfig::from_k27_contract(&contract).expect("config must validate");
        assert_eq!(config.context_tokens, 262_144);
        assert_eq!(config.kv_block_tokens, 64);
        assert_eq!(config.output_vocab_count, 163_840);
        assert_eq!(config.kv_head_count, 1);
        assert_eq!(config.kv_head_dim, 576);
        assert_eq!(config.stop_token_ids, vec![163_586]);
        assert_eq!(config.mtp_draft_token_count, DEFAULT_MTP_DRAFT_TOKEN_COUNT);
        assert_eq!(
            config.dspark_max_speculative_token_count,
            DEFAULT_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT
        );
    }
}
