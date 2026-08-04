//! Verify every contract in the C tree's model_contracts/ parses and yields
//! geometry. This is the serde replacement gate for `runtime/json.c` parsing.

use std::path::PathBuf;

use spark_model::ContractDocument;

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
