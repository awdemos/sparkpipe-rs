//! Verify every contract in the C tree's model_contracts/ parses and yields
//! geometry. This is the serde replacement gate for `runtime/json.c` parsing.

use std::path::PathBuf;

use spark_model::{ContractDocument, ModelFamily};

fn contracts_dir() -> PathBuf {
    let root = std::env::var("SPARKPIPE_C_ROOT").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("..").join("sparkpipe")
    });
    root.join("model_contracts")
}

#[test]
fn all_c_tree_contracts_parse() {
    let dir = contracts_dir();
    let mut parsed = 0usize;
    for entry in std::fs::read_dir(&dir).expect("model_contracts dir must exist") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();

        // These are registries/policies, not model contracts — they have no
        // geometry by design and get their own typed loaders later.
        if matches!(
            name.as_str(),
            "spark_hardware_assumption_bindings.json"
                | "spark_hardware_questions.json"
                | "must_work_targets.json"
        ) {
            continue;
        }

        let document = ContractDocument::from_path(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        parsed += 1;

        if let Some(version) = document.schema_version {
            assert!(version >= 1, "{name}: schema_version must be >= 1");
        }
    }
    assert!(parsed >= 9, "expected at least 9 contracts, parsed {parsed}");
}

#[test]
fn k3_authoritative_geometry() {
    let document =
        ContractDocument::from_path(&contracts_dir().join("k3_authoritative.json")).unwrap();
    assert_eq!(document.model_id.as_deref(), Some("moonshotai/Kimi-K3-MXFP4"));
    assert_eq!(document.architecture.as_deref(), Some("KimiK3ForCausalLM"));

    let geometry = document.geometry().expect("k3 must have geometry");
    assert_eq!(geometry.total_parameters, Some(2_780_000_000_000));
    assert_eq!(geometry.active_parameters, Some(104_200_000_000));
    assert_eq!(geometry.hidden_dimension, Some(7168));
    assert_eq!(geometry.layer_count, Some(93));
    assert_eq!(geometry.vocabulary_size, Some(163_840));
    assert_eq!(geometry.maximum_context_tokens, Some(1_048_576));

    // Family sections stay available as structured JSON.
    assert!(document.section("kda").is_some());
    assert!(document.section("mla").is_some());
    assert!(document.section("moe").is_some());
}

#[test]
fn k27_authoritative_geometry() {
    let document =
        ContractDocument::from_path(&contracts_dir().join("k27_authoritative.json")).unwrap();
    assert_eq!(document.model_id.as_deref(), Some("moonshotai/Kimi-K2.7-Code"));
    assert_eq!(document.architecture.as_deref(), Some("KimiK25ForConditionalGeneration"));

    let geometry = document.geometry().expect("k27 must have geometry");
    assert_eq!(geometry.total_parameters, Some(1_026_242_052_096));
    assert_eq!(geometry.active_parameters, Some(32_695_320_576));
    assert_eq!(geometry.hidden_dimension, Some(7168));
    assert_eq!(geometry.layer_count, Some(61));
    assert_eq!(geometry.vocabulary_size, Some(163_840));
    assert_eq!(geometry.maximum_context_tokens, Some(262_144));
    assert_eq!(geometry.first_routed_layer, Some(1));

    // DeepSeek-V3-family sections stay available as structured JSON.
    assert!(document.section("mla").is_some());
    assert!(document.section("rope").is_some());
    assert!(document.section("moe").is_some());
    assert!(document.section("quantization").is_some());
}

#[test]
fn k27_contract_typed_view() {
    let document =
        ContractDocument::from_path(&contracts_dir().join("k27_authoritative.json")).unwrap();
    let k27 = document.as_k27().expect("k27 contract must deserialize");

    assert_eq!(k27.schema_version, Some(1));
    assert_eq!(k27.model_id.as_deref(), Some("moonshotai/Kimi-K2.7-Code"));
    assert_eq!(k27.architecture.as_deref(), Some("KimiK25ForConditionalGeneration"));

    assert_eq!(k27.model.total_parameters, 1_026_242_052_096);
    assert_eq!(k27.model.active_parameters, 32_695_320_576);
    assert_eq!(k27.model.hidden_dimension, 7168);
    assert_eq!(k27.model.layer_count, 61);
    assert_eq!(k27.model.vocabulary_size, 163_840);
    assert_eq!(k27.model.maximum_context_tokens, 262_144);
    assert_eq!(k27.model.first_routed_layer, 1);
    assert!(!k27.model.tie_word_embeddings);

    assert_eq!(k27.mla.query_head_count, 64);
    assert_eq!(k27.mla.query_lora_rank, 1536);
    assert_eq!(k27.mla.kv_lora_rank, 512);
    assert_eq!(k27.mla.qk_nope_dimension, 128);
    assert_eq!(k27.mla.qk_unrotated_dimension, 64);
    assert_eq!(k27.mla.value_head_dimension, 128);
    assert!(!k27.mla.output_gate);
    assert!(k27.mla.uses_nope);
    assert!(!k27.mla.attention_bias);

    assert_eq!(k27.rope.rope_type, "yarn");
    assert_eq!(k27.rope.theta, 50000.0);
    assert_eq!(k27.rope.factor, 64.0);
    assert_eq!(k27.rope.original_max_position_embeddings, 4096);

    assert_eq!(k27.moe.routed_expert_count, 384);
    assert_eq!(k27.moe.experts_per_token, 8);
    assert_eq!(k27.moe.shared_expert_count, 1);
    assert_eq!(k27.moe.expert_intermediate_dimension, 2048);
    assert_eq!(k27.moe.dense_intermediate_dimension, 18432);
    assert_eq!(k27.moe.router_activation, "sigmoid");
    assert_eq!(k27.moe.top_k_method, "noaux_tc");
    assert!(k27.moe.renormalize_selected_probabilities);

    assert_eq!(k27.quantization.routed_expert_weight_format, "int4_pack_quantized");
    assert_eq!(k27.quantization.routed_expert_group_size, 32);
    assert!(k27.quantization.routed_expert_symmetric);
    assert_eq!(k27.quantization.kv_cache_format, "bf16");
    assert_eq!(k27.quantization.quantized_components, vec!["routed_experts"]);

    assert_eq!(k27.speculation.base_checkpoint_mtp_layer_count, 0);
    assert_eq!(k27.cache.kv_page_slots, 64);

    assert_eq!(k27.tokens.begin_of_sentence, 163_584);
    assert_eq!(k27.tokens.end_of_text, 163_586);
    assert_eq!(k27.tokens.pad, 163_839);

    assert_eq!(k27.qualification.cuda_target, "sm_121a");
    assert!(!k27.qualification.production_ready);
}

#[test]
fn k27_family_dispatch() {
    let document =
        ContractDocument::from_path(&contracts_dir().join("k27_authoritative.json")).unwrap();
    let family = ModelFamily::from_document(&document).expect("k27 must dispatch");
    match family {
        ModelFamily::K27(k27) => {
            assert_eq!(k27.model_id.as_deref(), Some("moonshotai/Kimi-K2.7-Code"));
        }
    }
}

#[test]
fn family_dispatch_rejects_unknown_architecture() {
    let document = ContractDocument::from_str(
        r#"{
            "schema_version": 1,
            "model_id": "x",
            "architecture": "UnknownLM",
            "model": {"layer_count": 1}
        }"#,
        "synthetic",
    )
    .unwrap();
    let err = ModelFamily::from_document(&document).expect_err("unknown architecture must fail");
    assert!(err.to_string().contains("unsupported architecture"), "{err}");
}

#[test]
fn glm52_flat_dialect_geometry() {
    let document = ContractDocument::from_path(&contracts_dir().join("glm52.json")).unwrap();
    let geometry = document.geometry().expect("glm52 flat dialect must yield geometry");
    assert!(geometry.layer_count.is_some(), "glm52 geometry: {geometry:?}");
}

#[test]
fn rejects_bad_schema_version() {
    let err = ContractDocument::from_str(
        r#"{"schema_version": 0, "model_id": "x", "model": {"layer_count": 1}}"#,
        "synthetic",
    )
    .expect_err("schema_version 0 must be rejected");
    assert!(err.to_string().contains("schema_version"), "{err}");
}

#[test]
fn rejects_missing_geometry() {
    let err = ContractDocument::from_str(r#"{"schema_version": 1, "model_id": "x"}"#, "synthetic")
        .expect_err("geometry-free document must be rejected");
    assert!(err.to_string().contains("geometry"), "{err}");
}
