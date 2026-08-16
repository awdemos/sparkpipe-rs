//! Typed views of the Kimi K2.7 (KimiK25 text backbone) model contract.
//!
//! These structs mirror the sections of `k27_authoritative.json` and are
//! extracted from a generic [`crate::ContractDocument`] via
//! [`ContractDocument::as_k27`](crate::ContractDocument::as_k27).

use serde::Deserialize;

/// Provenance metadata for the contract.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct K27Sources {
    #[serde(default)]
    pub checkpoint_config: String,
    #[serde(default)]
    pub parameter_counts: String,
    #[serde(default)]
    pub license: String,
}

/// Wrapper bookkeeping: the published checkpoint is multimodal, but the serving
/// engine consumes only the text backbone.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct K27MultimodalWrapper {
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub text_architecture: String,
    #[serde(default)]
    pub text_model_type: String,
    #[serde(default)]
    pub media_placeholder_token_id: u64,
}

/// Core model geometry.
#[derive(Debug, Clone, Deserialize)]
pub struct K27Model {
    pub total_parameters: u64,
    pub active_parameters: u64,
    pub hidden_dimension: u64,
    pub layer_count: u64,
    pub vocabulary_size: u64,
    pub maximum_context_tokens: u64,
    pub rms_norm_epsilon: f64,
    pub first_routed_layer: u64,
    pub tie_word_embeddings: bool,
}

/// Multi-head latent attention geometry.
#[derive(Debug, Clone, Deserialize)]
pub struct K27Mla {
    pub query_head_count: u64,
    pub query_lora_rank: u64,
    pub kv_lora_rank: u64,
    pub qk_nope_dimension: u64,
    pub qk_unrotated_dimension: u64,
    pub value_head_dimension: u64,
    pub output_gate: bool,
    pub uses_nope: bool,
    pub attention_bias: bool,
}

/// RoPE / YaRN configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct K27Rope {
    #[serde(rename = "type")]
    pub rope_type: String,
    pub theta: f64,
    pub factor: f64,
    pub original_max_position_embeddings: u64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub mscale: f64,
    pub mscale_all_dim: f64,
    #[serde(default)]
    pub note: String,
}

/// Mixture-of-experts configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct K27Moe {
    pub routed_expert_count: u64,
    pub experts_per_token: u64,
    pub shared_expert_count: u64,
    pub expert_intermediate_dimension: u64,
    pub dense_intermediate_dimension: u64,
    pub routed_scaling_factor: f64,
    pub router_activation: String,
    pub top_k_method: String,
    pub router_group_count: u64,
    pub renormalize_selected_probabilities: bool,
    pub activation: String,
    pub moe_layer_frequency: u64,
}

/// Weight and activation quantization scheme.
#[derive(Debug, Clone, Deserialize)]
pub struct K27Quantization {
    pub routed_expert_weight_format: String,
    pub routed_expert_group_size: u64,
    pub routed_expert_symmetric: bool,
    pub routed_expert_quant_method: String,
    pub non_expert_weight_format: String,
    pub non_expert_activation_format: String,
    pub accumulator_format: String,
    pub kv_cache_format: String,
    pub quantized_components: Vec<String>,
    pub unquantized_patterns: Vec<String>,
    #[serde(default)]
    pub note: String,
}

/// Speculation / MTP configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct K27Speculation {
    pub base_checkpoint_mtp_layer_count: u64,
    #[serde(default)]
    pub note: String,
}

/// KV-cache policy constants.
#[derive(Debug, Clone, Deserialize)]
pub struct K27Cache {
    pub kv_element_bits: u64,
    pub kv_page_slots: u64,
    #[serde(default)]
    pub note: String,
}

/// Special token ids.
#[derive(Debug, Clone, Deserialize)]
pub struct K27Tokens {
    pub begin_of_sentence: u64,
    pub end_of_text: u64,
    pub pad: u64,
}

/// Qualification / readiness metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct K27Qualification {
    pub cuda_target: String,
    pub status: String,
    pub production_ready: bool,
}

/// A fully typed Kimi K2.7 contract.
#[derive(Debug, Clone)]
pub struct K27Contract {
    pub schema_version: Option<u64>,
    pub model_id: Option<String>,
    pub architecture: Option<String>,
    pub sources: K27Sources,
    pub multimodal_wrapper: K27MultimodalWrapper,
    pub model: K27Model,
    pub mla: K27Mla,
    pub rope: K27Rope,
    pub moe: K27Moe,
    pub quantization: K27Quantization,
    pub speculation: K27Speculation,
    pub cache: K27Cache,
    pub tokens: K27Tokens,
    pub qualification: K27Qualification,
}
