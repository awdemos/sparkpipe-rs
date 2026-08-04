//! `spark-model`: model contracts and family descriptors.
//!
//! Replaces the C tree's `runtime/json.c` + `spark_model_description` parsing
//! for contract documents (`model_contracts/*.json`). The contract files are
//! heterogeneous per family (k3 carries `kda`/`mla`/`attnres`, glm52 is flat,
//! dsv4 carries `hyper_connections`), so this crate validates the common
//! envelope and keeps sections as structured JSON; typed per-family views are
//! added alongside the family that consumes them (k27 first, Phase 5).

pub mod contract;

pub use contract::{ContractDocument, ContractError, ModelGeometry};
