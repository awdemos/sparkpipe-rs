//! First k27-contract-driven end-to-end token generation test.
//!
//! Loads `k27_authoritative.json`, builds the backend from the contract, and
//! drives a mock rank-0 compute seam through a register → submit → pump loop
//! until a `Token` event is produced.

use std::path::PathBuf;

use spark_model::ContractDocument;
use spark_node::backend::state::BackendConfig;
use spark_node::backend::test_support::{reference_service_config, MockRank0NodeContext};
use spark_node::backend::RingServiceBackend;
use spark_serve::serving_engine::service::{
    ServiceEventKind, ServiceStats, ServiceSubmitTokenIdsRequest,
};
use spark_serve::serving_engine::{ServiceBackend, ServingStatus};

fn contracts_dir() -> PathBuf {
    let root = std::env::var("SPARKPIPE_C_ROOT").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("..").join("sparkpipe")
    });
    root.join("model_contracts")
}

#[test]
fn k27_contract_driven_token_generation_with_mock_rank0() {
    let document = ContractDocument::from_path(&contracts_dir().join("k27_authoritative.json"))
        .expect("k27 contract must parse");
    let k27 = document.as_k27().expect("k27 contract must validate");
    let backend_config =
        BackendConfig::from_k27_contract(&k27).expect("backend config must validate");
    let service_config = reference_service_config(&backend_config);

    let mut backend = RingServiceBackend::new(
        backend_config,
        service_config,
        Some(Box::new(MockRank0NodeContext { decode_token: 42 })),
    )
    .expect("backend must initialize");

    let service = backend.service().expect("service must be available");
    let client_id = service.register_client(1).expect("client must register");
    service
        .submit_token_ids(&ServiceSubmitTokenIdsRequest {
            flags: 0,
            priority: 0,
            thinking_token_budget: 0,
            output_token_budget: 4,
            max_prefill_tokens_per_step: 0,
            client_id,
            client_request_id: 1,
            sequence_id: 0,
            token_ids: &[100],
        })
        .expect("request must submit");

    let mut stats = ServiceStats::default();
    for _ in 0..32 {
        let status = backend.pump(4, &mut stats);
        assert!(
            status == Ok(()) || status == Err(ServingStatus::Busy),
            "pump failed: {:?}",
            status
        );
        let service = backend.service().expect("service must be available");
        while let Ok(event) = service.pop_event() {
            if event.kind == ServiceEventKind::Token {
                assert_eq!(event.token_id, 42);
                return;
            }
        }
    }
    panic!("expected a Token event within 32 pump iterations");
}
