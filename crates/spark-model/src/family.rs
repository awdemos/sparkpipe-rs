//! Lightweight model-family dispatcher.
//!
//! Contracts are heterogeneous across families (glm52 is flat, k3 carries `kda`,
//! k27 carries DeepSeek-V3-family `mla`/`moe` sections, etc.). This module
//! selects the appropriate typed view from a parsed envelope.

use crate::{ContractDocument, ContractError, K27Contract};

/// A typed view over a supported model-family contract.
#[derive(Debug, Clone)]
pub enum ModelFamily {
    /// Kimi K2.7 text backbone (`KimiK25ForConditionalGeneration`).
    K27(K27Contract),
}

impl ModelFamily {
    /// Select a typed family view from an envelope-validated contract.
    ///
    /// Selection is currently by `architecture`;glm52/k3 support can be added
    /// here when their typed section views land.
    pub fn from_document(document: &ContractDocument) -> Result<Self, ContractError> {
        match document.architecture.as_deref() {
            Some("KimiK25ForConditionalGeneration") => Ok(ModelFamily::K27(document.as_k27()?)),
            Some(other) => Err(ContractError::UnsupportedArchitecture {
                path: document.model_id.clone().unwrap_or_default(),
                architecture: other.to_string(),
            }),
            None => Err(ContractError::MissingArchitecture {
                path: document.model_id.clone().unwrap_or_default(),
            }),
        }
    }
}
