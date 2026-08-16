//! Model contract documents.
//!
//! Envelope invariants (fail-closed, mirroring the C schema gates):
//! - `schema_version` must be a positive integer when present
//! - `model_id`, when present, must be a non-empty string
//! - every contract must carry model geometry either in a `model` section
//!   (k3/dsv4/qwen36/mimo25 style) or as flat keys (glm52/k3 non-authoritative
//!   style); `geometry()` extracts a best-effort normalized view

use serde::Deserialize;
use std::path::Path;
use thiserror::Error;

use crate::k27::{
    K27Cache, K27Contract, K27Mla, K27Model, K27Moe, K27MultimodalWrapper, K27Qualification,
    K27Quantization, K27Rope, K27Speculation, K27Tokens,
};

#[derive(Debug, Error)]
pub enum ContractError {
    #[error("io error reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("JSON parse error in {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("contract {path}: schema_version must be a positive integer, found {found}")]
    BadSchemaVersion { path: String, found: String },

    #[error("contract {path}: model_id must be a non-empty string")]
    BadModelId { path: String },

    #[error("contract {path}: no model geometry found (neither a 'model' section nor flat geometry keys)")]
    MissingGeometry { path: String },

    #[error("contract {path}: missing required section '{section}'")]
    MissingSection { path: String, section: String },

    #[error("contract {path}: section '{section}' has invalid shape: {source}")]
    BadSection {
        path: String,
        section: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("contract {path}: missing architecture field")]
    MissingArchitecture { path: String },

    #[error("contract {path}: unsupported architecture '{architecture}'")]
    UnsupportedArchitecture { path: String, architecture: String },
}

/// Normalized model geometry, extracted from either the `model` section or
/// flat contract keys. All fields optional — contract dialects differ, and
/// absence is meaningful to the family validators that consume this.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelGeometry {
    #[serde(default)]
    pub total_parameters: Option<u64>,
    #[serde(default)]
    pub active_parameters: Option<u64>,
    #[serde(default)]
    pub hidden_dimension: Option<u64>,
    #[serde(default)]
    pub layer_count: Option<u64>,
    #[serde(default)]
    pub vocabulary_size: Option<u64>,
    #[serde(default)]
    pub maximum_context_tokens: Option<u64>,
    #[serde(default)]
    pub first_routed_layer: Option<u64>,
}

/// A parsed, envelope-validated contract document.
#[derive(Debug)]
pub struct ContractDocument {
    pub schema_version: Option<u64>,
    pub model_id: Option<String>,
    pub architecture: Option<String>,
    raw: serde_json::Value,
}

impl ContractDocument {
    pub fn from_str(text: &str, path: &str) -> Result<Self, ContractError> {
        let raw: serde_json::Value = serde_json::from_str(text)
            .map_err(|source| ContractError::Parse { path: path.to_string(), source })?;
        Self::from_value(raw, path)
    }

    pub fn from_path(path: &Path) -> Result<Self, ContractError> {
        let text = std::fs::read_to_string(path)
            .map_err(|source| ContractError::Io { path: path.display().to_string(), source })?;
        Self::from_str(&text, &path.display().to_string())
    }

    fn from_value(raw: serde_json::Value, path: &str) -> Result<Self, ContractError> {
        let schema_version = match raw.get("schema_version") {
            None => None,
            Some(v) => match v.as_u64().filter(|n| *n > 0) {
                Some(n) => Some(n),
                None => {
                    return Err(ContractError::BadSchemaVersion {
                        path: path.to_string(),
                        found: v.to_string(),
                    })
                }
            },
        };
        let model_id = match raw.get("model_id") {
            None => None,
            Some(v) => match v.as_str().filter(|s| !s.is_empty()) {
                Some(s) => Some(s.to_string()),
                None => return Err(ContractError::BadModelId { path: path.to_string() }),
            },
        };
        let architecture = raw.get("architecture").and_then(|v| v.as_str()).map(|s| s.to_string());

        let document = Self { schema_version, model_id, architecture, raw };
        if document.geometry().is_none() {
            return Err(ContractError::MissingGeometry { path: path.to_string() });
        }
        Ok(document)
    }

    /// The full raw document for family-specific section access.
    pub fn raw(&self) -> &serde_json::Value {
        &self.raw
    }

    /// A named section (e.g. `kda`, `mla`, `moe`, `precision`) if present.
    pub fn section(&self, name: &str) -> Option<&serde_json::Value> {
        self.raw.get(name)
    }

    /// Normalized geometry from the `model` section, or from flat keys for
    /// the glm52-style dialect.
    pub fn geometry(&self) -> Option<ModelGeometry> {
        if let Some(model) = self.raw.get("model") {
            return serde_json::from_value(model.clone()).ok();
        }
        // Flat dialect: lift geometry-ish keys from the top level.
        let flat = serde_json::json!({
            "total_parameters": self.raw.get("total_parameters"),
            "active_parameters": self.raw.get("active_parameters"),
            "hidden_dimension": self.raw.get("hidden_dimension"),
            "layer_count": self.raw.get("layer_count"),
            "vocabulary_size": self.raw.get("vocabulary_size")
                .or_else(|| self.raw.get("vocab_size")),
            "maximum_context_tokens": self.raw.get("maximum_context_tokens"),
            "first_routed_layer": self.raw.get("first_routed_layer"),
        });
        let geometry: ModelGeometry = serde_json::from_value(flat).ok()?;
        if geometry.layer_count.is_some() || geometry.hidden_dimension.is_some() {
            Some(geometry)
        } else {
            None
        }
    }

    /// Extract a typed Kimi K2.7 contract view.
    ///
    /// All required sections (`multimodal_wrapper`, `model`, `mla`, `rope`,
    /// `moe`, `quantization`, `speculation`, `cache`, `tokens`, `qualification`)
    /// must be present and well-formed. Optional top-level metadata
    /// (`schema_version`, `model_id`, `architecture`, `sources`) is preserved
    /// when present.
    pub fn as_k27(&self) -> Result<K27Contract, ContractError> {
        let path = self.model_id.as_deref().unwrap_or("unknown");

        fn take_section<T: serde::de::DeserializeOwned>(
            raw: &serde_json::Value,
            path: &str,
            name: &str,
        ) -> Result<T, ContractError> {
            let value = raw.get(name).ok_or_else(|| ContractError::MissingSection {
                path: path.to_string(),
                section: name.to_string(),
            })?;
            serde_json::from_value(value.clone()).map_err(|source| ContractError::BadSection {
                path: path.to_string(),
                section: name.to_string(),
                source,
            })
        }

        Ok(K27Contract {
            schema_version: self.schema_version,
            model_id: self.model_id.clone(),
            architecture: self.architecture.clone(),
            sources: self
                .raw
                .get("sources")
                .map(|v| serde_json::from_value(v.clone()))
                .transpose()
                .map_err(|source| ContractError::BadSection {
                    path: path.to_string(),
                    section: "sources".to_string(),
                    source,
                })?
                .unwrap_or_default(),
            multimodal_wrapper: take_section::<K27MultimodalWrapper>(
                &self.raw,
                path,
                "multimodal_wrapper",
            )?,
            model: take_section::<K27Model>(&self.raw, path, "model")?,
            mla: take_section::<K27Mla>(&self.raw, path, "mla")?,
            rope: take_section::<K27Rope>(&self.raw, path, "rope")?,
            moe: take_section::<K27Moe>(&self.raw, path, "moe")?,
            quantization: take_section::<K27Quantization>(&self.raw, path, "quantization")?,
            speculation: take_section::<K27Speculation>(&self.raw, path, "speculation")?,
            cache: take_section::<K27Cache>(&self.raw, path, "cache")?,
            tokens: take_section::<K27Tokens>(&self.raw, path, "tokens")?,
            qualification: take_section::<K27Qualification>(&self.raw, path, "qualification")?,
        })
    }
}
