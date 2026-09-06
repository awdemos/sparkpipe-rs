//! Per-rank serving backend pump — port of the C tree's `node/backend.c`.
//!
//! This is the integration root: it constructs the request API, serving
//! engine, and service runtime, then implements the [`ServiceBackend`] seam
//! from `spark-serve`. The actual compute paths (rank-0 local CUDA builder,
//! CUDA-resident IPC, next-rank work output) live in submodules and are
//! driven through the trait seams defined in [`state`].
//!
//! The integration root and its helpers are currently only exercised by unit
//! tests; suppress dead-code warnings until the top-level binary wires them.
#![allow(dead_code)]

pub mod adapt;
pub mod config_from_contract;
pub mod k27_geometry;
pub mod net;
pub mod pending;
pub mod resident;
pub mod state;
pub mod test_support;
pub mod wire;
pub mod work_output;

pub use adapt::{new_shared, DispatchExtras, RequestApiAdapter, ServingRequestApiHandle};
pub use state::{
    build_kv_arena, build_prefix_cache, build_scheduler, BackendConfig, BackendCore,
    NullRank0NodeContext, NullResidentDecodeStageRunner, OutputTransport, Rank0NodeContext,
    ResidentDecodeStageRunner,
};

use std::cell::RefCell;
use std::rc::Rc;

use spark_serve::request_api::prefetch::KvPrefetchBackend;
use spark_serve::request_api::{Configuration as RequestApiConfiguration, RequestApi};
use spark_serve::serving_engine::backend::{
    ServiceBackend, ServiceBackendConfiguration, ServiceBackendPollDescriptor, ServiceBackendView,
    CAPABILITY_POLL_DESCRIPTORS, CAPABILITY_RING_RUNTIME, CAPABILITY_SERVICE_RUNTIME,
    CAPABILITY_TOKENIZER, CONFIGURATION_FLAG_DSPARK, CONFIGURATION_FLAG_MTP,
    CONFIGURATION_KNOWN_FLAGS, POLL_READ,
};
use spark_serve::serving_engine::bridge::{self, Dispatch};
use spark_serve::serving_engine::engine::{
    DecodeFunction, PrefillFunction, ReleaseSequenceFunction, ServingDecodeDispatch,
    ServingDecodeResult, ServingEngine, ServingEngineConfiguration, ServingPrefillDispatch,
    ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE,
};
use spark_serve::serving_engine::service::{ServiceConfiguration, ServiceRuntime, ServiceStats};
use spark_serve::serving_engine::ServingStatus;
use spark_text::tokenizer::Tokenizer;
use tracing;

use crate::rank_daemon::ring_runtime::{build_rank_plan, QuantizationMode};

struct NullKvPrefetchBackend;
impl KvPrefetchBackend for NullKvPrefetchBackend {}

/// `SparkRingServiceBackendState` as an owned Rust value implementing the
/// node-runtime backend trait.
pub struct RingServiceBackend {
    core: Rc<RefCell<BackendCore>>,
    service: ServiceRuntime,
    adapter: Rc<RefCell<RequestApiAdapter>>,
}

impl RingServiceBackend {
    /// `SparkRingServiceBackendInitialize`.
    ///
    /// The `service_configuration` carries the runtime-facing options; the
    /// `backend_config` carries the model/geometry capacities. Both are kept
    /// separate so the geometry stays contract-driven and no GLM52 constants
    /// leak into the generic backend.
    ///
    /// `rank0_builder` is the rank-0 local compute seam. For the GPU-free
    /// milestone this is typically a test mock or the future CUDA builder
    /// wrapper; when `None` the backend reports `Busy` on prefill/decode until
    /// a builder or CUDA-resident connection is attached.
    pub fn new(
        backend_config: BackendConfig,
        service_configuration: ServiceBackendConfiguration,
        rank0_builder: Option<Box<dyn Rank0NodeContext>>,
    ) -> Result<Self, ServingStatus> {
        backend_config.validate().inspect_err(|&e| {
            tracing::error!("backend config validate failed: {:?}", e);
        })?;

        if (service_configuration.flags & !CONFIGURATION_KNOWN_FLAGS) != 0 {
            return Err(ServingStatus::AbiMismatch);
        }

        let kv_logical_block_capacity = if service_configuration.kv_logical_block_capacity != 0 {
            service_configuration.kv_logical_block_capacity
        } else {
            backend_config.gpu_block_count()
        };

        let quantization_mode =
            QuantizationMode::from_code(service_configuration.model_quantization_mode)
                .ok_or(ServingStatus::InvalidArgument)?;
        let rank_plan = build_rank_plan(
            &backend_config.geometry,
            0,
            backend_config.geometry.max_batch_bucket,
            service_configuration.port_base,
            quantization_mode,
        )
        .inspect_err(|&e| {
            tracing::error!("build_rank_plan failed: {:?}", e);
        })?;

        let work_control_config = backend_config.work_control_config();
        let arena =
            build_kv_arena(&backend_config, kv_logical_block_capacity).inspect_err(|&e| {
                tracing::error!("build_kv_arena failed: {:?}", e);
            })?;
        let prefix_cache = build_prefix_cache(&backend_config, kv_logical_block_capacity, arena)
            .inspect_err(|&e| {
                tracing::error!("build_prefix_cache failed: {:?}", e);
            })?;
        let scheduler = build_scheduler(&backend_config, quantization_mode, prefix_cache)
            .inspect_err(|&e| {
                tracing::error!("build_scheduler failed: {:?}", e);
            })?;

        let mut request_api_configuration = RequestApiConfiguration {
            configuration_flags: 0, // normalizes to defaults
            request_capacity: backend_config.request_capacity,
            prefetch_lookahead_request_count: 0,
            prefetch_lane_count: backend_config.prefetch_lane_count,
            decode_batch_target: 0,
            max_resident_kv_block_count: 0,
            decode_execution_row_capacity: 0,
            scheduler,
            kv_prefetch_backend: None,
            model_speculator: None,
            output_vocab_count: backend_config.output_vocab_count,
        };
        request_api_configuration
            .use_async_kv_cache_prefetch_backend(Box::new(NullKvPrefetchBackend))
            .map_err(map_request_api_error)?;
        let request_api = RequestApi::new(request_api_configuration).map_err(|e| {
            tracing::error!("RequestApi::new failed: {:?}", e);
            map_request_api_error(e)
        })?;

        let tokenizer = match &service_configuration.tokenizer_path {
            Some(path) if !path.is_empty() => Tokenizer::load_compiled_file(path)
                .or_else(|_| Tokenizer::load_huggingface_json(path))
                .ok(),
            _ => None,
        };

        let pending_capacity = backend_config.pipeline_cohort_capacity();
        let early_capacity = backend_config.final_event_pump_budget();
        let work_output = work_output::WorkOutput::new(
            work_control_config.clone(),
            state::monotonic_ns(),
            false,
            String::new(),
            0,
            rank_plan.execution_row_capacity,
            backend_config.max_pipeline_slot_count,
            backend_config.work_queue_capacity,
            backend_config.request_capacity as usize,
        );

        let resident_socket_path =
            service_configuration.cuda_resident_socket_path.as_deref().unwrap_or("");
        let resident =
            resident::ResidentConnection::new(resident_socket_path, state::monotonic_ns());

        let core = Rc::new(RefCell::new(BackendCore {
            config: backend_config.clone(),
            work_control: work_control_config,
            rank_plan,
            final_event_listen_port: service_configuration.port_base,
            session_id_base: state::monotonic_ns(),
            speculation_enabled: (service_configuration.flags & CONFIGURATION_FLAG_DSPARK) != 0,
            mtp_enabled: (service_configuration.flags & CONFIGURATION_FLAG_MTP) != 0,
            kv_logical_block_capacity,
            kv_physical_block_capacity: backend_config.gpu_block_count(),
            trace_enabled: state::environment_text("SPARKPIPE_RING_TRACE") == "1",
            trace_last_decode_completion_ns: 0,
            first_blocker: None,
            initialized: true,
            rank0_runtime_ready: rank0_builder.is_some(),
            service_runtime_ready: false,
            cuda_resident_attached: false,
            work_output,
            pendings: pending::PendingDecodes::new(pending_capacity, early_capacity),
            resident,
            rank0_builder: rank0_builder.unwrap_or_else(|| Box::new(NullRank0NodeContext)),
            runner: Box::new(NullResidentDecodeStageRunner),
            output_transport: None,
            transport_capability_flags: 0,
            transport_shared_object_path: service_configuration
                .transport_shared_object_path
                .unwrap_or_default(),
            final_event_listen_fd: None,
            final_event_socket: None,
            final_event_read_buffer: [0u8; wire::FINAL_EVENT_BYTES],
            final_event_read_offset: 0,
        }));

        // Build the engine with the adapter; callbacks close over Rc clones so
        // the single-threaded engine can reach the shared state when it calls
        // them during pump().
        let (adapter, request_api_box) = new_shared(request_api, pending_capacity);
        let core_for_prefill = Rc::clone(&core);
        let core_for_decode = Rc::clone(&core);
        let core_for_release = Rc::clone(&core);

        let engine_config = ServingEngineConfiguration {
            flags: ENGINE_FLAG_DYNAMIC_REQUEST_TOKEN_STORAGE,
            runtime_contract_flags: 0,
            default_thinking_token_budget: 0,
            default_output_token_budget: backend_config.default_output_token_budget,
            default_max_prefill_tokens_per_step: backend_config.prefill_wave_tokens,
            max_context_tokens: backend_config.context_tokens,
            request_id_base: 0,
            request_api: request_api_box,
            tokenizer,
            request_record_capacity: backend_config.request_capacity,
            request_token_stride: 0, // dynamic storage
            event_ring_capacity: backend_config.event_capacity,
            host_prefill_token_stride: backend_config.prefill_wave_tokens,
            host_prefill_lane_capacity: backend_config.pipeline_cohort_capacity() as u32,
            execution_physical_block_indices: None,
            kv_block_lane_stride: backend_config.max_blocks_per_sequence(),
            kv_block_lane_capacity: backend_config.max_blocks_per_sequence(),
            lane_count_capacity: bridge::MAX_DISPATCH_REQUEST_COUNT as u32,
            prefill_function: make_prefill_function(core_for_prefill),
            decode_function: make_decode_function(core_for_decode),
            release_sequence_function: Some(make_release_function(core_for_release)),
            stop_token_ids: backend_config.stop_token_ids.clone(),
        };
        let engine = ServingEngine::new(engine_config).inspect_err(|&e| {
            tracing::error!("ServingEngine::new failed: {:?}", e);
        })?;

        let service = ServiceRuntime::new(ServiceConfiguration {
            flags: 0,
            default_pump_dispatch_steps: 1,
            request_id_base: 0,
            serving_engine: engine,
            client_session_capacity: backend_config.default_max_active,
            request_map_capacity: backend_config.request_capacity,
            event_ring_capacity: backend_config.event_capacity,
        })
        .inspect_err(|&e| {
            tracing::error!("ServiceRuntime::new failed: {:?}", e);
        })?;

        core.borrow_mut().service_runtime_ready = true;
        Ok(RingServiceBackend { core, service, adapter })
    }
}

impl ServiceBackend for RingServiceBackend {
    fn capability_flags(&self) -> u32 {
        CAPABILITY_SERVICE_RUNTIME
            | CAPABILITY_RING_RUNTIME
            | CAPABILITY_TOKENIZER
            | CAPABILITY_POLL_DESCRIPTORS
    }

    fn get_view(&mut self) -> Result<ServiceBackendView, ServingStatus> {
        let core = self.core.borrow();
        Ok(ServiceBackendView {
            runtime_initialized: core.initialized,
            local_control_ready: core.rank0_runtime_ready,
            configured_kv_context_limit_tokens: core.config.context_tokens,
            configured_max_active_sequences: core.config.default_max_active,
            transport_capability_flags: core.transport_capability_flags,
            speculation_configuration_flags: {
                let mut flags = 0u32;
                if core.speculation_enabled {
                    flags |= CONFIGURATION_FLAG_DSPARK;
                }
                if core.mtp_enabled {
                    flags |= CONFIGURATION_FLAG_MTP;
                }
                flags
            },
            request_api_configuration_flags: self.adapter.borrow_mut().api().configuration_flags,
            adaptive_decode_batch_width: core.config.pipeline_cohort_capacity() as u32,
            decode_batch_capacity: bridge::MAX_DISPATCH_REQUEST_COUNT as u32,
            prefill_wave_token_count: core.config.prefill_wave_tokens,
            release_generation: 0,
            first_blocker: core.first_blocker.clone(),
            release_id: None,
            release_git_commit: None,
            transport_shared_object_path: if core.transport_shared_object_path.is_empty() {
                None
            } else {
                Some(core.transport_shared_object_path.clone())
            },
        })
    }

    fn pump(
        &mut self,
        max_dispatch_steps: u32,
        stats_out: &mut ServiceStats,
    ) -> Result<(), ServingStatus> {
        // Mirror the C pump order: drain resident/work/release/final events,
        // then run the service engine if both runtimes are ready.
        let mut cancel =
            |dispatch: &mut Dispatch| self.adapter.borrow_mut().cancel_dispatch(dispatch);
        {
            let mut core = self.core.borrow_mut();
            let _ =
                resident::pump_responses(&mut core, self.service.serving_engine_mut(), &mut cancel);
            let _ = core.work_output.pump_work_output();
            core.work_output.pop_release_records(bridge::MAX_DISPATCH_REQUEST_COUNT);
            let _ = core.runner.progress();

            let trace_enabled = core.trace_enabled;
            let active_count = core.pendings.pending_capacity();
            for index in 0..active_count {
                if core.pendings.get(index).is_some_and(|p| p.active) {
                    let _ = core.pendings.complete_early_final_events(
                        index,
                        self.service.serving_engine_mut(),
                        &mut cancel,
                        trace_enabled,
                    );
                }
            }
        }

        let can_dispatch = {
            let core = self.core.borrow();
            core.service_runtime_ready
                && core.rank0_runtime_ready
                && core.first_blocker.is_none()
                && core.pendings.find_free().is_some()
        };

        if can_dispatch {
            self.service.pump(max_dispatch_steps, Some(stats_out))
        } else {
            *stats_out = ServiceStats::default();
            let core = self.core.borrow();
            if core.service_runtime_ready
                && core.rank0_runtime_ready
                && core.first_blocker.is_none()
            {
                Err(ServingStatus::Busy)
            } else {
                Ok(())
            }
        }
    }

    fn poll_descriptors(&mut self) -> Result<Vec<ServiceBackendPollDescriptor>, ServingStatus> {
        let mut core = self.core.borrow_mut();
        let mut descriptors = Vec::new();
        if let Some(fd) = core.resident.fd.as_ref() {
            descriptors
                .push(ServiceBackendPollDescriptor { fd: net::raw_fd(fd), events: POLL_READ });
        }
        let final_fd = core.final_event_socket.as_ref().or(core.final_event_listen_fd.as_ref());
        if let Some(fd) = final_fd {
            descriptors
                .push(ServiceBackendPollDescriptor { fd: net::raw_fd(fd), events: POLL_READ });
        }
        if let Some(transport) = core.output_transport.as_mut() {
            for (fd, events) in transport.poll_descriptors() {
                descriptors.push(ServiceBackendPollDescriptor { fd, events });
            }
        }
        Ok(descriptors)
    }

    fn service(&mut self) -> Option<&mut ServiceRuntime> {
        Some(&mut self.service)
    }
}

fn map_request_api_error(error: spark_serve::request_api::RequestApiError) -> ServingStatus {
    use spark_serve::request_api::RequestApiError;
    match error {
        RequestApiError::InvalidArgument => ServingStatus::InvalidArgument,
        RequestApiError::CapacityExceeded => ServingStatus::CapacityExceeded,
        RequestApiError::NotFound => ServingStatus::NotFound,
        RequestApiError::Busy => ServingStatus::Busy,
        RequestApiError::InternalError => ServingStatus::InternalError,
        RequestApiError::ModuleNotValidated => ServingStatus::ModuleNotValidated,
        RequestApiError::HashMismatch => ServingStatus::HashMismatch,
    }
}

fn make_prefill_function(core: Rc<RefCell<BackendCore>>) -> PrefillFunction {
    Box::new(move |prefill_dispatch: &ServingPrefillDispatch| {
        let mut core = core.borrow_mut();
        if core.rank0_builder.is_attached() {
            // The idle pump is called by the builder while it waits for the
            // resident/work-output plane to drain. For the GPU-free milestone
            // the builder is synchronous, so a no-op idle pump is sufficient.
            let mut idle_pump = || Ok(());
            core.rank0_builder.prefill(prefill_dispatch, &mut idle_pump)
        } else {
            Err(ServingStatus::Busy)
        }
    })
}

fn make_decode_function(core: Rc<RefCell<BackendCore>>) -> DecodeFunction {
    Box::new(
        move |decode_dispatch: &ServingDecodeDispatch, decode_result: &mut ServingDecodeResult| {
            let mut core = core.borrow_mut();
            if core.rank0_builder.is_attached() {
                core.rank0_builder.decode(decode_dispatch, decode_result)
            } else {
                // TODO: resident async path — register pending decode and
                // submit the resident payload, returning Busy until the final
                // event completes it on a later pump.
                Err(ServingStatus::Busy)
            }
        },
    )
}

fn make_release_function(core: Rc<RefCell<BackendCore>>) -> ReleaseSequenceFunction {
    Box::new(
        move |_request_id: u64, _request_generation: u64, _sequence_id: u64, _token_count: u32| {
            // TODO: queue a work-control release packet through work_output
            // when the backend has a next-rank neighbor; for the single-rank
            // GPU-free milestone the local request API already owns release.
            let _ = core.borrow_mut().work_output.queue_count();
            Ok(())
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_serve::serving_engine::service::{ServiceEventKind, ServiceSubmitTokenIdsRequest};
    use test_support::{reference_service_config, MockRank0NodeContext};

    #[test]
    fn backend_config_validates() {
        BackendConfig::reference().validate().unwrap();
    }

    #[test]
    fn backend_inits_and_reports_view() {
        let backend_config = BackendConfig::reference();
        let service_config = reference_service_config(&backend_config);
        let mut backend = RingServiceBackend::new(backend_config, service_config, None).unwrap();
        let view = backend.get_view().unwrap();
        assert!(view.runtime_initialized);
        assert_eq!(view.configured_kv_context_limit_tokens, 1_048_576);
    }

    #[test]
    fn backend_poll_descriptors_are_empty_when_quiescent() {
        let backend_config = BackendConfig::reference();
        let service_config = reference_service_config(&backend_config);
        let mut backend = RingServiceBackend::new(backend_config, service_config, None).unwrap();
        let descriptors = backend.poll_descriptors().unwrap();
        assert!(descriptors.is_empty());
    }

    #[test]
    fn full_stack_token_generation_with_mock_rank0() {
        let backend_config = BackendConfig::reference();
        let service_config = reference_service_config(&backend_config);
        let mut backend = RingServiceBackend::new(
            backend_config,
            service_config,
            Some(Box::new(MockRank0NodeContext { decode_token: 42 })),
        )
        .unwrap();

        let service = backend.service().unwrap();
        let client_id = service.register_client(1).unwrap();
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
            .unwrap();

        let mut stats = ServiceStats::default();
        for _ in 0..32 {
            let status = backend.pump(4, &mut stats);
            assert!(
                status == Ok(()) || status == Err(ServingStatus::Busy),
                "pump failed: {:?}",
                status
            );
            {
                let service = backend.service().unwrap();
                while let Ok(event) = service.pop_event() {
                    if event.kind == ServiceEventKind::Token {
                        assert_eq!(event.token_id, 42);
                        return;
                    }
                }
            }
        }
        panic!("expected a Token event within 32 pump iterations");
    }
}
