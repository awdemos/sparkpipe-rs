//! Backend configuration, seam traits, and the shared core state.
//!
//! Every GLM52-derived capacity the C bakes in as `SPARK_GLM52_*` /
//! `SPARK_RING_SERVICE_BACKEND_*` macros is a [`BackendConfig`] field here
//! (hard rule: no GLM52 constants in Rust code); the C values are documented
//! per field as the reference configuration. [`BackendConfig::reference`]
//! builds exactly that configuration.
//!
//! The seams (task requirement 3): the GLM52-specific moving parts behind
//! traits so the model-family modules can stay C-side:
//!
//! * [`ResidentDecodeStageRunner`] — `SparkResidentDecodeStageProductionRunner*`.
//! * [`Rank0NodeContext`] — the node-context builder interface's
//!   prefill/decode/submit_work entry points (local-CUDA rank0 path).
//! * [`OutputTransport`] — the hidden transport session's poll descriptors.

use std::os::unix::io::OwnedFd;

use spark_core::kv_arena::{KvArena, KvArenaConfig};
use spark_core::prefix_cache::{PrefixCache, PrefixCacheConfig};
use spark_sched::scheduler::{
    Scheduler, SchedulerConfig, CONFIGURATION_DEFAULT_FLAGS,
    CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE,
};
use spark_sched::stage_plan::QuantizationMode as SchedulerQuantizationMode;
use spark_sched::stage_plan::{self, StagePlanGeometry};
use spark_sched::work_control::{WorkControlConfig, WorkControlPacket};
use spark_serve::serving_engine::engine::{
    ServingDecodeDispatch, ServingDecodeResult, ServingPrefillDispatch,
};
use spark_serve::serving_engine::ServingStatus;

use super::net;
use super::pending::PendingDecodes;
use super::resident::ResidentConnection;
use super::wire::FINAL_EVENT_BYTES;
use super::work_output::{WorkOutput, PREFILL_RESERVE_CAPACITY};
use crate::rank_daemon::ring_runtime::{QuantizationMode, RankPlan, RingModelGeometry};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Backend configuration (`BackendConfig`). Field docs name the C macro and
/// its GLM52 value; nothing here is read from GLM52 headers at build time.
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Ring model geometry (`SparkGlm52RingRuntime*` tables in C).
    pub geometry: RingModelGeometry,
    /// `SPARK_RING_SERVICE_BACKEND_CONTEXT_TOKENS` (GLM52: 1048576).
    pub context_tokens: u32,
    /// `SPARK_RING_SERVICE_BACKEND_KV_BLOCK_TOKENS` (GLM52: 64).
    pub kv_block_tokens: u32,
    /// `SPARK_GLM52_KV_POOL_TOKENS` (GLM52: 4194304).
    pub kv_pool_tokens: u32,
    /// `SPARK_RING_SERVICE_BACKEND_PREFILL_WAVE_TOKENS` (GLM52: 256).
    pub prefill_wave_tokens: u32,
    /// `SPARK_RING_SERVICE_BACKEND_PREFILL_TOKENS` /
    /// `SPARK_RING_NODE_CONTEXT_BUILDER_MAX_PREFILL_TOKENS` (GLM52: 256).
    pub builder_max_prefill_tokens: u32,
    /// `SPARK_RING_SERVICE_BACKEND_REQUEST_CAPACITY`
    /// (= `SPARK_STAGE_PLAN_PIPELINE_INFLIGHT_REQUEST_CAPACITY`, 13312).
    pub request_capacity: u32,
    /// `SPARK_RING_SERVICE_BACKEND_EVENT_CAPACITY` (16384).
    pub event_capacity: u32,
    /// `SPARK_RING_SERVICE_BACKEND_WORK_QUEUE_CAPACITY`
    /// (`MAX_PREFILL_TOKENS * 2` = 512).
    pub work_queue_capacity: usize,
    /// `SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT` (GLM52: 6).
    pub mtp_draft_token_count: u32,
    /// `SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT` (GLM52: 7).
    pub dspark_max_speculative_token_count: u32,
    /// `SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT` (GLM52: 154880).
    pub output_vocab_count: u32,
    /// KV heads of the arena descriptor (GLM52: 8).
    pub kv_head_count: u32,
    /// KV head dim of the arena descriptor (GLM52: 128).
    pub kv_head_dim: u32,
    /// `SPARK_KV_CACHE_MAX_PREFETCH_LANE_COUNT` (13).
    pub prefetch_lane_count: u32,
    /// `SPARK_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT` (1024).
    pub max_pipeline_slot_count: u32,
    /// `SPARK_RING_NODE_CONTEXT_BUILDER_DEFAULT_RESIDENT_SEQUENCE_COUNT`
    /// (16384).
    pub max_resident_sequence_count: u32,
    /// `SPARK_RING_SERVICE_BACKEND_DEFAULT_MAX_ACTIVE` (1024).
    pub default_max_active: u32,
    /// `SPARK_RING_SERVICE_BACKEND_DEFAULT_PORT_BASE` (52100).
    pub default_port_base: u32,
    /// Engine `default_output_token_budget` (C: 1024).
    pub default_output_token_budget: u32,
    /// `SPARK_STAGE_PLAN_MEASURED_PROFILE_20260701`.
    pub measured_profile_id: u32,
    /// `SPARK_GLM52_MODEL_EOS_TOKEN_IDS_INITIALIZER`
    /// (GLM52: {154820, 154827, 154829}).
    pub stop_token_ids: Vec<u32>,
    /// `SPARK_RING_SERVICE_BACKEND_METADATA_KEY_BASE` (0x100000000).
    pub metadata_key_base: u64,
    /// `SPARK_RING_SERVICE_BACKEND_METADATA_VALUE_BASE` (0x200000000).
    pub metadata_value_base: u64,
}

impl BackendConfig {
    /// `SPARK_RING_SERVICE_BACKEND_GPU_BLOCK_COUNT`.
    pub fn gpu_block_count(&self) -> u32 {
        self.kv_pool_tokens / self.kv_block_tokens
    }

    /// `SPARK_RING_SERVICE_BACKEND_KV_BLOCK_COUNT` (per-sequence blocks).
    pub fn kv_block_count(&self) -> u32 {
        self.context_tokens / self.kv_block_tokens
    }

    /// `SPARK_RING_SERVICE_BACKEND_MAX_BLOCKS_PER_SEQUENCE`.
    pub fn max_blocks_per_sequence(&self) -> u32 {
        self.kv_block_count()
    }

    /// `SPARK_RING_SERVICE_BACKEND_PIPELINE_COHORT_CAPACITY`
    /// (= the ring's spark count).
    pub fn pipeline_cohort_capacity(&self) -> usize {
        self.geometry.stage_count as usize
    }

    /// `SPARK_RING_SERVICE_BACKEND_QUEUE_DEPTH_PER_SPARK`.
    pub fn queue_depth_per_spark(&self) -> usize {
        self.pipeline_cohort_capacity() + PREFILL_RESERVE_CAPACITY as usize
    }

    /// `SPARK_RING_SERVICE_BACKEND_PREFIX_BINDING_COUNT`.
    pub fn prefix_binding_count(&self) -> u32 {
        self.kv_block_count() + self.request_capacity
    }

    /// Final-event pump budget (`PENDING_DECODE_CAPACITY *
    /// MAX_DISPATCH_REQUEST_COUNT`).
    pub fn final_event_pump_budget(&self) -> usize {
        self.pipeline_cohort_capacity() * spark_sched::stage_plan::MAX_BATCH_BUCKET as usize
    }

    /// The work-control configuration view of these capacities.
    pub fn work_control_config(&self) -> WorkControlConfig {
        WorkControlConfig {
            mtp_draft_token_count: self.mtp_draft_token_count,
            dspark_max_speculative_token_count: self.dspark_max_speculative_token_count,
            maximum_context_tokens: self.context_tokens,
            kv_block_tokens: self.kv_block_tokens,
            max_prefill_tokens_per_packet: self.prefill_wave_tokens,
            output_vocab_count: self.output_vocab_count,
            max_lane_count: spark_sched::stage_plan::MAX_BATCH_BUCKET,
            max_active_sequence_count: spark_sched::stage_plan::MAX_BATCH_BUCKET,
            cohort_capacity: self.geometry.stage_count,
        }
    }

    /// Cross-field validation mirroring the C `_Static_assert`s and the
    /// derived-capacity relations.
    pub fn validate(&self) -> Result<(), ServingStatus> {
        if self.context_tokens == 0
            || self.kv_block_tokens == 0
            || self.context_tokens % self.kv_block_tokens != 0
            || self.kv_pool_tokens == 0
            || self.kv_pool_tokens % self.kv_block_tokens != 0
            || self.prefill_wave_tokens == 0
            || self.builder_max_prefill_tokens == 0
            || self.request_capacity == 0
            || self.event_capacity == 0
            || self.work_queue_capacity == 0
            || self.kv_head_count == 0
            || self.kv_head_dim == 0
            || self.prefetch_lane_count == 0
            || self.max_pipeline_slot_count == 0
            || self.default_max_active == 0
            || self.stop_token_ids.is_empty()
        {
            return Err(ServingStatus::InvalidArgument);
        }
        self.geometry.validate().map_err(|_| ServingStatus::InvalidArgument)?;
        self.work_control_config().validate().map_err(|_| ServingStatus::InvalidArgument)?;
        // _Static_assert: scheduler depth holds every decode cohort plus the
        // prefill reserve.
        if self.queue_depth_per_spark()
            != self.pipeline_cohort_capacity() + PREFILL_RESERVE_CAPACITY as usize
        {
            return Err(ServingStatus::InvalidArgument);
        }
        // _Static_assert: the work queue holds a complete sequence-release
        // drain.
        let max_lane_count = spark_sched::stage_plan::MAX_BATCH_BUCKET as usize;
        if self.request_capacity as usize / max_lane_count
            + usize::from(self.request_capacity as usize % max_lane_count != 0)
            > self.work_queue_capacity
        {
            return Err(ServingStatus::InvalidArgument);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Seams
// ---------------------------------------------------------------------------

/// `SparkResidentDecodeStageProductionRunnerProgress` seam (GLM52 runner
/// module stays C-side / spark-sys-side).
pub trait ResidentDecodeStageRunner {
    /// `SparkResidentDecodeStageProductionRunnerProgress` (status ignored by
    /// the pump, as in C).
    fn progress(&mut self) -> Result<(), ServingStatus>;
}

/// Null runner (rank0 not ready / tests).
pub struct NullResidentDecodeStageRunner;

impl ResidentDecodeStageRunner for NullResidentDecodeStageRunner {
    fn progress(&mut self) -> Result<(), ServingStatus> {
        Ok(())
    }
}

/// The rank0 node-context builder seam: the three builder entry points the
/// backend invokes on the local-CUDA path
/// (`builder_interface.prefill/decode/submit_work`).
pub trait Rank0NodeContext {
    /// `builder_interface.prefill` (the idle pump keeps the work-output
    /// plane draining while the builder works).
    fn prefill(
        &mut self,
        prefill_dispatch: &ServingPrefillDispatch,
        idle_pump: &mut dyn FnMut() -> Result<(), ServingStatus>,
    ) -> Result<(), ServingStatus>;
    /// `builder_interface.decode` (fills `decode_result` synchronously).
    fn decode(
        &mut self,
        decode_dispatch: &ServingDecodeDispatch,
        decode_result: &mut ServingDecodeResult,
    ) -> Result<(), ServingStatus>;
    /// `builder_interface.submit_work` (sequence releases to rank0).
    fn submit_work(&mut self, packet: &WorkControlPacket) -> Result<(), ServingStatus>;
    /// Whether the seam is attached (`builder_state != 0` and the entry
    /// points exist, in C terms).
    fn is_attached(&self) -> bool;
}

/// Null builder: every entry point reports `ModuleNotValidated`, exactly
/// the C "builder not loaded" status.
pub struct NullRank0NodeContext;

impl Rank0NodeContext for NullRank0NodeContext {
    fn prefill(
        &mut self,
        _prefill_dispatch: &ServingPrefillDispatch,
        _idle_pump: &mut dyn FnMut() -> Result<(), ServingStatus>,
    ) -> Result<(), ServingStatus> {
        Err(ServingStatus::ModuleNotValidated)
    }

    fn decode(
        &mut self,
        _decode_dispatch: &ServingDecodeDispatch,
        _decode_result: &mut ServingDecodeResult,
    ) -> Result<(), ServingStatus> {
        Err(ServingStatus::ModuleNotValidated)
    }

    fn submit_work(&mut self, _packet: &WorkControlPacket) -> Result<(), ServingStatus> {
        Err(ServingStatus::ModuleNotValidated)
    }

    fn is_attached(&self) -> bool {
        false
    }
}

/// The output hidden-transport session's poll-descriptor seam
/// (`SparkHiddenTransportGetPollDescriptors`, mapped onto the service poll
/// event bits).
pub trait OutputTransport {
    /// Poll descriptors as `(fd, events)` with events in
    /// `spark_serve::serving_engine::backend::POLL_READ`/`POLL_WRITE` bits.
    fn poll_descriptors(&mut self) -> Vec<(i32, u32)> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// Shared core state (the callback context)
// ---------------------------------------------------------------------------

/// `SparkRingServiceBackendState`'s callback-reachable half: everything the
/// prefill/decode/release callbacks and the backend pump share. The engine
/// callbacks reach it through `Rc<RefCell<BackendCore>>`; the engine itself
/// stays outside (see `adapt` module docs for the borrow discipline).
pub struct BackendCore {
    pub config: BackendConfig,
    pub work_control: WorkControlConfig,
    /// `state->rank_plan` (built during init; rank0 in this port).
    pub rank_plan: RankPlan,
    /// `state->final_event_route.listen_port` (bind for the listener).
    pub final_event_listen_port: u32,
    /// `state->session_id_base`.
    pub session_id_base: u64,
    /// `state->speculation_enabled`.
    pub speculation_enabled: bool,
    /// `state->mtp_enabled`.
    pub mtp_enabled: bool,
    /// `state->kv_logical_block_capacity`.
    pub kv_logical_block_capacity: u32,
    /// `state->kv_physical_block_capacity`.
    pub kv_physical_block_capacity: u32,
    /// `state->trace_enabled` (`SPARKPIPE_RING_TRACE`).
    pub trace_enabled: bool,
    /// `state->trace_last_decode_completion_ns`.
    pub trace_last_decode_completion_ns: u64,
    /// `state->first_blocker` (`None` == the C empty string).
    pub first_blocker: Option<String>,
    /// `state->initialized`.
    pub initialized: bool,
    /// `state->rank0_runtime_ready`.
    pub rank0_runtime_ready: bool,
    /// `state->service_runtime_ready`.
    pub service_runtime_ready: bool,
    /// `state->cuda_resident_attached`.
    pub cuda_resident_attached: bool,
    /// The work-output plane (queue + socket + release ring).
    pub work_output: WorkOutput,
    /// Pending decodes + early final events + final-event counters.
    pub pendings: PendingDecodes,
    /// CUDA-resident IPC connection state.
    pub resident: ResidentConnection,
    /// Rank0 node-context builder seam (local-CUDA path).
    pub rank0_builder: Box<dyn Rank0NodeContext>,
    /// Resident decode stage production runner seam.
    pub runner: Box<dyn ResidentDecodeStageRunner>,
    /// Output hidden transport session seam (poll descriptors only).
    pub output_transport: Option<Box<dyn OutputTransport>>,
    /// `state->transport_library.transport_interface.capability_flags`.
    pub transport_capability_flags: u32,
    /// `state->transport_shared_object_path`.
    pub transport_shared_object_path: String,
    /// Final-event listener/socket and split-read buffer.
    pub final_event_listen_fd: Option<OwnedFd>,
    pub final_event_socket: Option<OwnedFd>,
    pub final_event_read_buffer: [u8; FINAL_EVENT_BYTES],
    pub final_event_read_offset: usize,
}

impl BackendCore {
    /// `SparkRingServiceBackendSetBlocker` (first blocker wins).
    pub fn set_blocker(&mut self, message: &str) {
        if self.first_blocker.is_none() {
            self.first_blocker = Some(message.to_string());
        }
    }

    /// `SparkRingServiceBackendAcceptFinalEventSocket`.
    pub fn accept_final_event_socket(&mut self) {
        let Some(listen_fd) = &self.final_event_listen_fd else {
            return;
        };
        if self.final_event_socket.is_some() {
            return;
        }
        match net::accept_nonblocking(net::raw_fd(listen_fd)) {
            Ok(Some(fd)) => {
                if net::set_nonblocking(net::raw_fd(&fd)).is_err() {
                    self.pendings.final_event_receive_error_count += 1;
                    return;
                }
                self.final_event_socket = Some(fd);
            }
            Ok(None) => {}
            Err(_) => {
                self.pendings.final_event_receive_error_count += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Init helpers (the storage/engine chain of `InitializeServiceRuntime`)
// ---------------------------------------------------------------------------

/// `SparkRingServiceBackendInitializeKvArena` (the metadata descriptor is a
/// host-only bookkeeping view in the Rust port, as in the C service
/// backend's arena).
pub fn build_kv_arena(
    config: &BackendConfig,
    kv_logical_block_capacity: u32,
) -> Result<KvArena, ServingStatus> {
    KvArena::new(&KvArenaConfig {
        physical_block_count: kv_logical_block_capacity,
        block_token_count: config.kv_block_tokens,
        resident_block_capacity: 0,
        layer_count: config.geometry.layer_count,
        kv_head_count: config.kv_head_count,
        head_dim: config.kv_head_dim,
        bytes_per_scalar: 2,
        key_block_stride_bytes: 0,
        value_block_stride_bytes: 0,
        key_device_base: config.metadata_key_base as usize,
        value_device_base: config.metadata_value_base as usize,
    })
    .map_err(|_| ServingStatus::InvalidArgument)
}

/// `SparkRingServiceBackendInitializePrefixCache`.
pub fn build_prefix_cache(
    config: &BackendConfig,
    kv_logical_block_capacity: u32,
    arena: KvArena,
) -> Result<PrefixCache, ServingStatus> {
    PrefixCache::new(
        &PrefixCacheConfig {
            block_token_count: config.kv_block_tokens,
            entry_count: kv_logical_block_capacity,
            physical_block_count: kv_logical_block_capacity,
            sequence_binding_count: kv_logical_block_capacity + config.request_capacity,
        },
        arena,
    )
    .map_err(|_| ServingStatus::InvalidArgument)
}

/// `SparkRingServiceBackendInitializeScheduler` (including the
/// `SPARKPIPE_DISABLE_PREFIX_REUSE` kill-switch).
pub fn build_scheduler(
    config: &BackendConfig,
    quantization_mode: QuantizationMode,
    prefix_cache: PrefixCache,
) -> Result<Scheduler, ServingStatus> {
    let mut configuration_flags = CONFIGURATION_DEFAULT_FLAGS;
    if std::env::var_os("SPARKPIPE_DISABLE_PREFIX_REUSE").is_some() {
        configuration_flags &= !CONFIGURATION_FLAG_CROSS_SEQUENCE_PREFIX_REUSE;
        eprintln!("ring_config prefix_reuse=DISABLED by SPARKPIPE_DISABLE_PREFIX_REUSE");
    }
    let scheduler_quantization = match quantization_mode {
        QuantizationMode::Nvfp4 => SchedulerQuantizationMode::Nvfp4_4Bit,
        QuantizationMode::Fp8E4m3 => SchedulerQuantizationMode::Fp8E4m3_8Bit,
    };
    Scheduler::new(SchedulerConfig {
        spark_count: config.geometry.stage_count,
        queue_depth_per_spark: config.queue_depth_per_spark() as u32,
        measured_profile_id: config.measured_profile_id,
        stage_geometry: StagePlanGeometry {
            layer_count: config.geometry.layer_count,
            first_routed_layer: config.geometry.first_routed_layer,
        },
        estimated_layer_cost_ns: 0,
        estimated_final_stage_extra_cost_ns: 0,
        quantization_mode: scheduler_quantization,
        max_prefill_tokens_per_step: config.prefill_wave_tokens,
        default_max_prefill_tokens_per_step: config.prefill_wave_tokens,
        max_context_tokens: config.context_tokens,
        max_batch_bucket: stage_plan::MAX_BATCH_BUCKET,
        prefix_cache_block_tokens: config.kv_block_tokens,
        configuration_flags,
        prefix_cache: Some(prefix_cache),
    })
    .map_err(|_| ServingStatus::InvalidArgument)
}

/// `SparkRingServiceBackendEnvironmentU64`.
pub fn environment_u64(name: &str) -> u64 {
    std::env::var(name).ok().and_then(|text| text.parse::<u64>().ok()).unwrap_or(0)
}

/// `SparkRingServiceBackendEnvironmentText` (`""` when unset).
pub fn environment_text(name: &str) -> String {
    std::env::var(name).unwrap_or_default()
}

/// `SparkNetMonotonicNs` re-export for the backend root.
pub fn monotonic_ns() -> u64 {
    net::monotonic_ns()
}

#[cfg(test)]
impl BackendConfig {
    /// A reference GLM52-backed configuration for tests and documentation.
    /// All values mirror the C macros the backend used to bake in; production
    /// callers build this from a model contract instead.
    pub fn reference() -> Self {
        use crate::rank_daemon::ring_runtime::{RingModelGeometry, RingPackLayout};
        BackendConfig {
            geometry: RingModelGeometry {
                layer_count: 78,
                first_routed_layer: 3,
                weight_layer_count: 79,
                hidden_dimension: 6144,
                hidden_bf16_bytes_per_sequence: 12288,
                maximum_context_tokens: 1_048_576,
                dsa_selected_token_count: 2048,
                dsa_selected_index_bytes_per_sequence: 8192,
                max_speculative_rows_per_lane: 8,
                max_batch_bucket: 1024,
                shape_inputs: crate::rank_daemon::shape::ShapeModelInputs::new(
                    78,
                    6144,
                    2048,
                    12288,
                    512 + 64,
                    1,
                ),
                tp_geometry: crate::rank_daemon::shape::TpModelGeometry::new(
                    64,
                    192 + 64,
                    192 + 256,
                    256,
                ),
                stage_count: 13,
                default_stage_layer_counts: vec![6; 13],
                host_prefix: "10.10.100.".to_string(),
                host_index_base: 10,
                pack_layout: RingPackLayout::default(),
            },
            context_tokens: 1_048_576,
            kv_block_tokens: 64,
            kv_pool_tokens: 4_194_304,
            prefill_wave_tokens: 256,
            builder_max_prefill_tokens: 256,
            request_capacity: 13_312,
            event_capacity: 16_384,
            work_queue_capacity: 512,
            mtp_draft_token_count: 6,
            dspark_max_speculative_token_count: 7,
            output_vocab_count: 154_880,
            kv_head_count: 8,
            kv_head_dim: 128,
            prefetch_lane_count: 13,
            max_pipeline_slot_count: 1_024,
            max_resident_sequence_count: 16_384,
            default_max_active: 1_024,
            default_port_base: 52_100,
            default_output_token_budget: 1_024,
            measured_profile_id: spark_sched::stage_plan::MEASURED_PROFILE_20260701,
            stop_token_ids: vec![154_820, 154_827, 154_829],
            metadata_key_base: 0x1_0000_0000,
            metadata_value_base: 0x2_0000_0000,
        }
    }
}
