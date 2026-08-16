//! Shared test helpers for the backend integration tests.
//!
//! These are re-exported from the library only under `#[cfg(test)]` so
//! integration tests can build a backend with the same mock rank-0 seam used
//! by the unit tests.

use spark_serve::serving_engine::backend::ServiceBackendConfiguration;
use spark_serve::serving_engine::bridge::MAX_DECODE_TOKENS_PER_LANE;
use spark_serve::serving_engine::engine::{
    ServingDecodeDispatch, ServingDecodeResult, ServingPrefillDispatch,
};
use spark_serve::serving_engine::ServingStatus;

use crate::backend::state::{BackendConfig, Rank0NodeContext};
use crate::rank_daemon::ring_runtime::QuantizationMode;

/// A `ServiceBackendConfiguration` that exercises the backend with the
/// reference quantization mode and no external sockets.
pub fn reference_service_config(backend_config: &BackendConfig) -> ServiceBackendConfiguration {
    ServiceBackendConfiguration {
        flags: 0,
        max_active_sequence_count: backend_config.default_max_active,
        port_base: backend_config.default_port_base,
        kv_logical_block_capacity: 0,
        model_quantization_mode: QuantizationMode::Fp8E4m3.code(),
        ..Default::default()
    }
}

/// Deterministic rank-0 compute seam for the GPU-free full-stack milestone.
pub struct MockRank0NodeContext {
    pub decode_token: u32,
}

impl Rank0NodeContext for MockRank0NodeContext {
    fn prefill(
        &mut self,
        _prefill_dispatch: &ServingPrefillDispatch,
        _idle_pump: &mut dyn FnMut() -> Result<(), ServingStatus>,
    ) -> Result<(), ServingStatus> {
        Ok(())
    }

    fn decode(
        &mut self,
        decode_dispatch: &ServingDecodeDispatch,
        decode_result: &mut ServingDecodeResult,
    ) -> Result<(), ServingStatus> {
        let lane_count = decode_dispatch.decode_view.lane_count as usize;
        for lane in 0..lane_count {
            decode_result.token_counts[lane] = 1;
            decode_result.token_ids[lane * MAX_DECODE_TOKENS_PER_LANE] = self.decode_token;
        }
        Ok(())
    }

    fn submit_work(
        &mut self,
        _packet: &spark_sched::work_control::WorkControlPacket,
    ) -> Result<(), ServingStatus> {
        Ok(())
    }

    fn is_attached(&self) -> bool {
        true
    }
}
