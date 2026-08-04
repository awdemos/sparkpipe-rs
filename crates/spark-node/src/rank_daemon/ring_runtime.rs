//! Ring runtime rank planning (port of `node/rank_runtime.c` and
//! `include/sparkpipe/spark_ring_runtime.h`, plus the hidden-transport
//! endpoint validation from `ring/transport/hidden_transport.c` it calls).
//!
//! Every model-derived value the C pulls from `SPARK_GLM52_MODEL_*` arrives
//! through [`RingModelGeometry`] (port deviation #1 — no GLM52 constants in
//! Rust). The C reference values are noted per field. Error-text side
//! channels (`SparkReportError` buffers) are dropped; the `Err` status is
//! the same code the C returns.

use spark_sched::stage_plan::{self, StagePlanGeometry};

use super::resident_ipc::DsparkDraftResult;
use super::shape::{
    derive_node_config, ShapeModelInputs, ShapeNodeConfig, TpModelGeometry, TpShapeDescriptor,
};
use super::status::{from_stage_plan, Result, SparkStatus};

/// C: `SPARK_RING_RUNTIME_ABI_VERSION`.
pub const RING_RUNTIME_ABI_VERSION: u32 = 6;
/// C: `SPARK_RING_RUNTIME_RANK_PLAN_DESCRIPTOR_BYTES` (C `sizeof`, probe).
pub const RANK_PLAN_DESCRIPTOR_BYTES: u32 = 472;
/// C: `SPARK_RING_RUNTIME_FINAL_EVENT_ROUTE_DESCRIPTOR_BYTES`.
pub const FINAL_EVENT_ROUTE_DESCRIPTOR_BYTES: u32 = 120;
/// C: `SPARK_RING_RUNTIME_FINAL_EVENT_DESCRIPTOR_BYTES`.
pub const FINAL_EVENT_DESCRIPTOR_BYTES: u32 = 272;
/// C: `SPARK_RING_RUNTIME_FINAL_EVENT_MAGIC`.
pub const FINAL_EVENT_MAGIC: u32 = 0x3545_4650;
/// C: `SPARK_RING_RUNTIME_FINAL_EVENT_FLAG_DSPARK_DRAFT`.
pub const FINAL_EVENT_FLAG_DSPARK_DRAFT: u32 = 0x0000_0001;
/// C: `SPARK_RING_RUNTIME_FINAL_EVENT_KNOWN_FLAGS`.
pub const FINAL_EVENT_KNOWN_FLAGS: u32 = FINAL_EVENT_FLAG_DSPARK_DRAFT;

/// C: `SPARK_RING_RUNTIME_HOST_NAME_BYTES` (including NUL).
pub const HOST_NAME_BYTES: usize = 16;
/// C: `SPARK_RING_RUNTIME_ROUTE_NAME_BYTES` (including NUL).
pub const ROUTE_NAME_BYTES: usize = 64;
/// C: `SPARK_RING_RUNTIME_FINAL_EVENT_ROUTE_NAME_BYTES`.
pub const FINAL_EVENT_ROUTE_NAME_BYTES: usize = 64;
/// C: `SPARK_RING_RUNTIME_PACK_PATH_BYTES` (including NUL).
pub const PACK_PATH_BYTES: usize = 512;
/// C: `SPARK_RING_RUNTIME_DEFAULT_PORT_BASE`.
pub const DEFAULT_PORT_BASE: u32 = 52100;
/// C: `SPARK_RING_RUNTIME_FINAL_EVENT_PORT_OFFSET`.
pub const FINAL_EVENT_PORT_OFFSET: u32 = 200;

pub const RANK_FLAG_HAS_PREVIOUS: u32 = 0x0000_0001;
pub const RANK_FLAG_HAS_NEXT: u32 = 0x0000_0002;
pub const RANK_FLAG_FINAL_STAGE: u32 = 0x0000_0004;
pub const RANK_FLAG_DENSE_PREFIX: u32 = 0x0000_0008;
pub const RANK_KNOWN_FLAGS: u32 =
    RANK_FLAG_HAS_PREVIOUS | RANK_FLAG_HAS_NEXT | RANK_FLAG_FINAL_STAGE | RANK_FLAG_DENSE_PREFIX;

/// C: `SPARK_RING_RUNTIME_MOE_BACKEND_NONE`.
pub const MOE_BACKEND_NONE: u32 = 0;
/// C: `SPARK_RING_RUNTIME_MOE_BACKEND_FP8_FLASHINFER_GROUPED`.
pub const MOE_BACKEND_FP8_FLASHINFER_GROUPED: u32 = 1;
/// C: `SPARK_RING_RUNTIME_MOE_BACKEND_NVFP4_B12X`.
pub const MOE_BACKEND_NVFP4_B12X: u32 = 3;

/// C: `SPARK_HIDDEN_TRANSPORT_ABI_VERSION`.
pub const HIDDEN_TRANSPORT_ABI_VERSION: u32 = 3;
/// C: `SPARK_HIDDEN_TRANSPORT_ENDPOINT_BYTES` (C `sizeof`, probe).
pub const HIDDEN_TRANSPORT_ENDPOINT_BYTES: u32 = 56;
/// C: `SPARK_HIDDEN_TRANSPORT_PERSISTENT_RING_MODULE_ID`.
pub const PERSISTENT_RING_MODULE_ID: &str = "spark.hidden_transport.persistent_ring.device.v1";
/// C: `SPARK_HIDDEN_TRANSPORT_REQUIRED_PIPELINE_HOST_STAGED_CAPS`.
pub const REQUIRED_PIPELINE_HOST_STAGED_CAPS: u32 = 0x67;
/// C: `SPARK_HIDDEN_TRANSPORT_REQUIRED_PRODUCTION_CAPS`.
pub const REQUIRED_PRODUCTION_CAPS: u32 = 0x7f;
/// C: `SPARK_HIDDEN_TRANSPORT_REQUIRED_SIMULATION_CAPS`.
pub const REQUIRED_SIMULATION_CAPS: u32 = 0x8000_0061;
/// C: `SPARK_HIDDEN_TRANSPORT_CAP_SIMULATION_ONLY`.
pub const CAP_SIMULATION_ONLY: u32 = 0x8000_0000;
/// C: `SPARK_HIDDEN_TRANSPORT_CAP_SPARK_HOST_PINNED_RDMA`.
pub const CAP_SPARK_HOST_PINNED_RDMA: u32 = 0x0000_0200;
/// C: `SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT`.
pub const BF16_BYTES_PER_ELEMENT: u32 = 2;

/// C: `SPARK_TP_COLLECTIVE_MAX_STEPS`.
pub const TP_COLLECTIVE_MAX_STEPS: usize = 4;

/// C: `SPARK_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT`.
pub const QUANTIZATION_NVFP4_4BIT: u32 = 1;
/// C: `SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT`.
pub const QUANTIZATION_FP8_E4M3_8BIT: u32 = 2;

/// C: the two pack manifests the daemon distinguishes
/// (`SPARK_RING_RUNTIME_FP8_PACK_MANIFEST` / `..._B12X_PACK_MANIFEST`).
pub const FP8_PACK_MANIFEST: &str = "fp8_moe_pack_manifest.json";
pub const B12X_PACK_MANIFEST: &str = "resident_moe_pack_manifest.json";

/// C: `SPARK_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN` (re-exported for plan flags).
const STAGE_FLAG_FINAL_TOKEN: u32 = 0x0000_0001;
/// C: `SPARK_STAGE_PLAN_STAGE_FLAG_DENSE_PREFIX`.
const STAGE_FLAG_DENSE_PREFIX: u32 = 0x0000_0008;

/// C: the `quantization_mode` value space (`spark_stage_plan.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    /// C: `SPARK_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT` ("fp8").
    Fp8E4m3,
    /// C: `SPARK_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT` ("nvfp4").
    Nvfp4,
}

impl QuantizationMode {
    /// The C wire/config code.
    pub fn code(self) -> u32 {
        match self {
            QuantizationMode::Fp8E4m3 => QUANTIZATION_FP8_E4M3_8BIT,
            QuantizationMode::Nvfp4 => QUANTIZATION_NVFP4_4BIT,
        }
    }

    /// C code to mode; `None` for any other value.
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            QUANTIZATION_FP8_E4M3_8BIT => Some(QuantizationMode::Fp8E4m3),
            QUANTIZATION_NVFP4_4BIT => Some(QuantizationMode::Nvfp4),
            _ => None,
        }
    }

    /// C: `SparkRingRuntimeParseQuantizationMode`.
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "fp8" => Ok(QuantizationMode::Fp8E4m3),
            "nvfp4" => Ok(QuantizationMode::Nvfp4),
            _ => Err(SparkStatus::InvalidArgument),
        }
    }

    /// C: `SparkRingRuntimeQuantizationModeName`.
    pub fn name(self) -> &'static str {
        match self {
            QuantizationMode::Fp8E4m3 => "fp8",
            QuantizationMode::Nvfp4 => "nvfp4",
        }
    }

    /// C: `SparkRingRuntimeValidateFp8PlanCounts`.
    pub fn validate_fp8_plan_counts(
        self,
        bound_plan_count: u32,
        expected_plan_count: u32,
    ) -> Result<()> {
        match self {
            QuantizationMode::Fp8E4m3 => {
                if expected_plan_count != 0 && bound_plan_count == expected_plan_count {
                    Ok(())
                } else {
                    Err(SparkStatus::ModuleNotValidated)
                }
            }
            QuantizationMode::Nvfp4 => {
                if bound_plan_count == 0 && expected_plan_count == 0 {
                    Ok(())
                } else {
                    Err(SparkStatus::ModuleNotValidated)
                }
            }
        }
    }

    /// C: `SparkRingRuntimeExpectedMoeBackendKind`.
    pub fn expected_moe_backend_kind(self) -> u32 {
        match self {
            QuantizationMode::Fp8E4m3 => MOE_BACKEND_FP8_FLASHINFER_GROUPED,
            QuantizationMode::Nvfp4 => MOE_BACKEND_NVFP4_B12X,
        }
    }
}

/// Pack-file naming layout (C bakes the GLM52 file naming convention into
/// `SparkRingRuntimeBuildMoePackPath`; here it is configuration). The
/// patterns use `{root}`, `{layer:04}`, and `{tag}` placeholders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingPackLayout {
    /// C ref: `glm52_layer_%04u_fp8_moe%s.spfp8`.
    pub fp8_pack_pattern: String,
    /// C ref: `glm52_layer_%04u_b12x_moe%s.spb12x`.
    pub b12x_pack_pattern: String,
    /// C: `SPARK_RING_RUNTIME_FP8_PACK_MANIFEST`.
    pub fp8_manifest: String,
    /// C: `SPARK_RING_RUNTIME_B12X_PACK_MANIFEST`.
    pub b12x_manifest: String,
}

impl Default for RingPackLayout {
    fn default() -> Self {
        Self {
            fp8_pack_pattern: "{root}/glm52_layer_{layer:04}_fp8_moe{tag}.spfp8".to_string(),
            b12x_pack_pattern: "{root}/glm52_layer_{layer:04}_b12x_moe{tag}.spb12x".to_string(),
            fp8_manifest: FP8_PACK_MANIFEST.to_string(),
            b12x_manifest: B12X_PACK_MANIFEST.to_string(),
        }
    }
}

impl RingPackLayout {
    fn pack_pattern(&self, mode: QuantizationMode) -> &str {
        match mode {
            QuantizationMode::Fp8E4m3 => &self.fp8_pack_pattern,
            QuantizationMode::Nvfp4 => &self.b12x_pack_pattern,
        }
    }

    fn manifest(&self, mode: QuantizationMode) -> &str {
        match mode {
            QuantizationMode::Fp8E4m3 => &self.fp8_manifest,
            QuantizationMode::Nvfp4 => &self.b12x_manifest,
        }
    }
}

/// Model/deployment geometry the C bakes in as `SPARK_GLM52_MODEL_*`,
/// `SPARK_RING_RUNTIME_STAGE_COUNT`, the fixed host table, and the fixed
/// stage layer-count table. All values are configuration; the C reference
/// values are noted per field.
#[derive(Debug, Clone)]
pub struct RingModelGeometry {
    /// C ref: `SPARK_GLM52_MODEL_LAYER_COUNT` (78).
    pub layer_count: u32,
    /// C ref: `SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER` (3).
    pub first_routed_layer: u32,
    /// C ref: `SPARK_GLM52_MODEL_WEIGHT_LAYER_COUNT` (79).
    pub weight_layer_count: u32,
    /// C ref: `SPARK_GLM52_MODEL_HIDDEN_DIMENSION` (6144).
    pub hidden_dimension: u32,
    /// C ref: `SPARK_GLM52_MODEL_HIDDEN_BF16_BYTES` (12288).
    pub hidden_bf16_bytes_per_sequence: u32,
    /// C ref: `SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS` (1048576).
    pub maximum_context_tokens: u32,
    /// C ref: `SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT` (2048).
    pub dsa_selected_token_count: u32,
    /// C ref: `SPARK_GLM52_MODEL_DSA_SELECTED_INDEX_BYTES` (8192).
    pub dsa_selected_index_bytes_per_sequence: u32,
    /// C ref: `SPARK_GLM52_MODEL_MAX_SPECULATIVE_ROWS_PER_LANE` (8).
    pub max_speculative_rows_per_lane: u32,
    /// C ref: `SPARK_STAGE_PLAN_MAX_BATCH_BUCKET` (1024).
    pub max_batch_bucket: u32,
    /// Shape-derivation inputs (C: `SparkGlm52ShapeModelInputs`).
    pub shape_inputs: ShapeModelInputs,
    /// TP head-block geometry (C: `SparkTpModelGeometryFromModel`).
    pub tp_geometry: TpModelGeometry,
    /// C ref: `SPARK_RING_RUNTIME_STAGE_COUNT` (13).
    pub stage_count: u32,
    /// C ref: `SparkGlm52RingRuntimeDefaultLayerCounts` (13 × 6).
    pub default_stage_layer_counts: Vec<u32>,
    /// C ref: the `"10.10.100."` host prefix.
    pub host_prefix: String,
    /// C ref: the `10u + rank_index` host index base.
    pub host_index_base: u32,
    /// Pack-file naming layout.
    pub pack_layout: RingPackLayout,
}

impl RingModelGeometry {
    /// Cross-field consistency check (the C encodes these as compile-time
    /// relations; here they are validated once at bring-up).
    pub fn validate(&self) -> Result<()> {
        if self.layer_count == 0
            || self.weight_layer_count == 0
            || self.hidden_dimension == 0
            || self.hidden_bf16_bytes_per_sequence != self.hidden_dimension * BF16_BYTES_PER_ELEMENT
            || self.maximum_context_tokens == 0
            || self.dsa_selected_token_count == 0
            || self.max_speculative_rows_per_lane == 0
            || self.max_batch_bucket == 0
            || self.stage_count == 0
            || self.stage_count as usize > stage_plan::MAX_STAGE_COUNT as usize
            || self.default_stage_layer_counts.len() != self.stage_count as usize
            || self.default_stage_layer_counts.iter().copied().sum::<u32>() != self.layer_count
            || self.shape_inputs.total_layer_count != self.layer_count
            || self.shape_inputs.hidden_dimension != self.hidden_dimension
            || self.host_prefix.is_empty()
        {
            return Err(SparkStatus::InvalidArgument);
        }
        Ok(())
    }

    /// C: `SPARK_RING_RUNTIME_LAYER_MAJOR_TRANSPORT_BYTES_PER_ROW`.
    pub fn layer_major_transport_bytes_per_row(&self) -> u64 {
        u64::from(self.hidden_bf16_bytes_per_sequence)
            + u64::from(self.dsa_selected_index_bytes_per_sequence)
    }
}

/// C: `SparkRingRuntimeDsaCandidateBucket`.
pub fn dsa_candidate_bucket(geometry: &RingModelGeometry, context_token_count: u32) -> u32 {
    if context_token_count == 0 || context_token_count > geometry.maximum_context_tokens {
        return 0;
    }
    let mut candidate_count = geometry.dsa_selected_token_count;
    while candidate_count < context_token_count && candidate_count < geometry.maximum_context_tokens
    {
        candidate_count <<= 1;
    }
    if candidate_count > geometry.maximum_context_tokens {
        candidate_count = geometry.maximum_context_tokens;
    }
    candidate_count
}

/// C: `SparkRingRuntimeExecutionRowCapacity`.
pub fn execution_row_capacity(geometry: &RingModelGeometry, logical_lane_capacity: u32) -> u32 {
    if logical_lane_capacity == 0 || logical_lane_capacity > geometry.max_batch_bucket {
        return 0;
    }
    let capacity =
        u64::from(logical_lane_capacity) * u64::from(geometry.max_speculative_rows_per_lane);
    capacity.min(u64::from(geometry.max_batch_bucket)) as u32
}

/// C: `SparkRingRuntimeRankHostName` (`10.10.100.{base + rank}` in the C
/// deployment table; prefix and base are configuration here).
pub fn rank_host_name(geometry: &RingModelGeometry, rank_index: u32) -> Result<String> {
    if rank_index >= geometry.stage_count {
        return Err(SparkStatus::InvalidArgument);
    }
    let host_name = format!("{}{}", geometry.host_prefix, geometry.host_index_base + rank_index);
    if host_name.len() >= HOST_NAME_BYTES {
        return Err(SparkStatus::CapacityExceeded);
    }
    Ok(host_name)
}

/// C: `SparkRingRuntimeFormatRoute` (`{left}_to_{right}_hidden`).
fn format_route(left: &str, right: &str) -> Result<String> {
    let route = format!("{left}_to_{right}_hidden");
    if route.len() >= ROUTE_NAME_BYTES {
        return Err(SparkStatus::CapacityExceeded);
    }
    Ok(route)
}

/// C: `SparkHiddenTransportEndpoint`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HiddenTransportEndpoint {
    pub abi_version: u32,
    pub descriptor_bytes: u32,
    pub capability_flags: u32,
    pub hidden_dimension: u32,
    pub bytes_per_sequence: u32,
    pub max_active_sequence_count: u32,
    pub max_packet_bytes: u64,
    pub validated_latency_ns: u64,
    pub transport_module_id: String,
    pub route_name: String,
}

/// C: `SparkHiddenTransportCapabilitiesMeetPipelineHostStaged` /
/// `...MeetProduction` / `...MeetSimulation` / `...AreSimulationOnly`.
fn capabilities_meet_pipeline_host_staged(flags: u32) -> bool {
    flags & REQUIRED_PIPELINE_HOST_STAGED_CAPS == REQUIRED_PIPELINE_HOST_STAGED_CAPS
}

fn capabilities_meet_production(flags: u32) -> bool {
    flags & REQUIRED_PRODUCTION_CAPS == REQUIRED_PRODUCTION_CAPS
}

fn capabilities_meet_simulation(flags: u32) -> bool {
    flags & REQUIRED_SIMULATION_CAPS == REQUIRED_SIMULATION_CAPS
}

fn capabilities_are_simulation_only(flags: u32) -> bool {
    flags & CAP_SIMULATION_ONLY != 0
}

/// C: `SparkHiddenTransportValidateEndpoint`.
pub fn validate_endpoint(endpoint: &HiddenTransportEndpoint) -> Result<()> {
    if endpoint.abi_version != HIDDEN_TRANSPORT_ABI_VERSION
        || endpoint.descriptor_bytes != HIDDEN_TRANSPORT_ENDPOINT_BYTES
    {
        return Err(SparkStatus::AbiMismatch);
    }
    if capabilities_are_simulation_only(endpoint.capability_flags) {
        if !capabilities_meet_simulation(endpoint.capability_flags)
            || capabilities_meet_production(endpoint.capability_flags)
        {
            return Err(SparkStatus::InvalidArgument);
        }
    } else if !capabilities_meet_production(endpoint.capability_flags)
        && !capabilities_meet_pipeline_host_staged(endpoint.capability_flags)
    {
        return Err(SparkStatus::InvalidArgument);
    }
    if endpoint.transport_module_id.is_empty()
        || endpoint.route_name.is_empty()
        || endpoint.hidden_dimension == 0
        || endpoint.bytes_per_sequence == 0
        || endpoint.max_active_sequence_count == 0
        || endpoint.max_packet_bytes == 0
    {
        return Err(SparkStatus::InvalidArgument);
    }
    if endpoint.bytes_per_sequence != endpoint.hidden_dimension * BF16_BYTES_PER_ELEMENT {
        return Err(SparkStatus::InvalidArgument);
    }
    let maximum_payload_bytes =
        u64::from(endpoint.bytes_per_sequence) * u64::from(endpoint.max_active_sequence_count);
    if maximum_payload_bytes > endpoint.max_packet_bytes {
        return Err(SparkStatus::CapacityExceeded);
    }
    Ok(())
}

/// C: `SparkRingRuntimeInitializeEndpoint`.
fn initialize_endpoint(
    geometry: &RingModelGeometry,
    max_active_sequence_count: u32,
    route_name: &str,
) -> HiddenTransportEndpoint {
    HiddenTransportEndpoint {
        abi_version: HIDDEN_TRANSPORT_ABI_VERSION,
        descriptor_bytes: HIDDEN_TRANSPORT_ENDPOINT_BYTES,
        capability_flags: REQUIRED_PIPELINE_HOST_STAGED_CAPS,
        hidden_dimension: geometry.hidden_dimension,
        bytes_per_sequence: geometry.hidden_bf16_bytes_per_sequence,
        max_active_sequence_count,
        max_packet_bytes: geometry.layer_major_transport_bytes_per_row()
            * u64::from(max_active_sequence_count),
        validated_latency_ns: 0,
        transport_module_id: PERSISTENT_RING_MODULE_ID.to_string(),
        route_name: route_name.to_string(),
    }
}

/// C: `SparkRingRuntimeRankPlan`. The C fixed char arrays are `String`s
/// (byte limits enforced at build); the endpoints are `Some` exactly when
/// the matching `HAS_PREVIOUS`/`HAS_NEXT` flag is set.
#[derive(Debug, Clone)]
pub struct RankPlan {
    pub abi_version: u32,
    pub descriptor_bytes: u32,
    pub rank_index: u32,
    pub flags: u32,
    pub first_layer_index: u32,
    pub layer_count: u32,
    pub previous_rank_index: u32,
    pub next_rank_index: u32,
    pub listen_port: u32,
    pub next_port: u32,
    pub logical_lane_capacity: u32,
    pub maximum_speculative_rows_per_lane: u32,
    pub execution_row_capacity: u32,
    pub hidden_dimension: u32,
    pub bytes_per_sequence: u32,
    pub quantization_mode: QuantizationMode,
    pub tp_degree: u32,
    pub tp_rank: u32,
    pub pp_stage_count: u32,
    pub pp_stage_index: u32,
    pub tp_collective_listen_port: u32,
    pub reserved_shape: u32,
    pub shape_configuration_hash: u64,
    pub max_packet_bytes: u64,
    pub host_name: String,
    pub previous_host_name: String,
    pub next_host_name: String,
    pub tp_peer_host_names: Vec<String>,
    pub tp_peer_ports: Vec<u32>,
    pub input_route_name: String,
    pub output_route_name: String,
    pub input_endpoint: Option<HiddenTransportEndpoint>,
    pub output_endpoint: Option<HiddenTransportEndpoint>,
}

impl RankPlan {
    fn blank(quantization_mode: QuantizationMode) -> Self {
        Self {
            abi_version: RING_RUNTIME_ABI_VERSION,
            descriptor_bytes: RANK_PLAN_DESCRIPTOR_BYTES,
            rank_index: 0,
            flags: 0,
            first_layer_index: 0,
            layer_count: 0,
            previous_rank_index: u32::MAX,
            next_rank_index: u32::MAX,
            listen_port: 0,
            next_port: 0,
            logical_lane_capacity: 0,
            maximum_speculative_rows_per_lane: 0,
            execution_row_capacity: 0,
            hidden_dimension: 0,
            bytes_per_sequence: 0,
            quantization_mode,
            tp_degree: 0,
            tp_rank: 0,
            pp_stage_count: 0,
            pp_stage_index: 0,
            tp_collective_listen_port: 0,
            reserved_shape: 0,
            shape_configuration_hash: 0,
            max_packet_bytes: 0,
            host_name: String::new(),
            previous_host_name: String::new(),
            next_host_name: String::new(),
            tp_peer_host_names: Vec::new(),
            tp_peer_ports: Vec::new(),
            input_route_name: String::new(),
            output_route_name: String::new(),
            input_endpoint: None,
            output_endpoint: None,
        }
    }

    /// Fills the pipeline-neighbor links (previous/next host, route, and
    /// endpoint) shared by the fixed-plan and shape-plan builders. `rank` is
    /// the pipeline position (C: rank index for the fixed plan, PP stage
    /// index for the shape plan); `previous_node`/`next_node` are the linear
    /// node indices of the neighbors.
    #[allow(clippy::too_many_arguments)]
    fn link_pipeline(
        &mut self,
        geometry: &RingModelGeometry,
        pipeline_position: u32,
        pipeline_count: u32,
        previous_node: u32,
        next_node: u32,
        next_port: u32,
    ) -> Result<()> {
        if pipeline_position > 0 {
            self.flags |= RANK_FLAG_HAS_PREVIOUS;
            self.previous_rank_index = previous_node;
            self.previous_host_name = rank_host_name(geometry, self.previous_rank_index)?;
            self.input_route_name = format_route(&self.previous_host_name, &self.host_name)?;
            self.input_endpoint = Some(initialize_endpoint(
                geometry,
                self.execution_row_capacity,
                &self.input_route_name,
            ));
        }
        if pipeline_position + 1 < pipeline_count {
            self.flags |= RANK_FLAG_HAS_NEXT;
            self.next_rank_index = next_node;
            self.next_port = next_port;
            self.next_host_name = rank_host_name(geometry, self.next_rank_index)?;
            self.output_route_name = format_route(&self.host_name, &self.next_host_name)?;
            self.output_endpoint = Some(initialize_endpoint(
                geometry,
                self.execution_row_capacity,
                &self.output_route_name,
            ));
        }
        Ok(())
    }
}

/// C: `SparkRingRuntimeShapeNodeConfig` (static) — one call site for the
/// geometry and inputs so plan, sharder, and packs cannot disagree.
fn shape_node_config(
    geometry: &RingModelGeometry,
    shape: &TpShapeDescriptor,
) -> Result<ShapeNodeConfig> {
    derive_node_config(shape, &geometry.tp_geometry, &geometry.shape_inputs)
}

/// C: `SparkRingRuntimeBuildFixedStagePlan`.
pub fn build_fixed_stage_plan(geometry: &RingModelGeometry) -> Result<stage_plan::StagePlan> {
    stage_plan::build_from_layer_counts(
        &StagePlanGeometry {
            layer_count: geometry.layer_count,
            first_routed_layer: geometry.first_routed_layer,
        },
        &geometry.default_stage_layer_counts,
    )
    .map_err(|error| from_stage_plan(&error))
}

/// C: `SparkRingRuntimeBuildRankPlan`.
pub fn build_rank_plan(
    geometry: &RingModelGeometry,
    rank_index: u32,
    logical_lane_capacity: u32,
    port_base: u32,
    quantization_mode: QuantizationMode,
) -> Result<RankPlan> {
    if logical_lane_capacity == 0
        || logical_lane_capacity > geometry.max_batch_bucket
        || rank_index >= geometry.stage_count
        || port_base > 65535 - geometry.stage_count
    {
        return Err(SparkStatus::InvalidArgument);
    }
    let stage_plan = build_fixed_stage_plan(geometry)?;
    let stage = stage_plan.stages[rank_index as usize];
    let mut rank_plan = RankPlan::blank(quantization_mode);
    rank_plan.rank_index = rank_index;
    rank_plan.first_layer_index = stage.first_layer_index;
    rank_plan.layer_count = stage.layer_count;
    rank_plan.listen_port = port_base + rank_index;
    rank_plan.logical_lane_capacity = logical_lane_capacity;
    rank_plan.maximum_speculative_rows_per_lane = geometry.max_speculative_rows_per_lane;
    rank_plan.execution_row_capacity = execution_row_capacity(geometry, logical_lane_capacity);
    rank_plan.hidden_dimension = geometry.hidden_dimension;
    rank_plan.bytes_per_sequence = geometry.hidden_bf16_bytes_per_sequence;
    rank_plan.tp_degree = 1;
    rank_plan.tp_rank = 0;
    rank_plan.pp_stage_count = geometry.stage_count;
    rank_plan.pp_stage_index = rank_index;
    rank_plan.tp_collective_listen_port = 0;
    {
        let shape = TpShapeDescriptor::new(1, 0, geometry.stage_count, rank_index);
        let shape_config = shape_node_config(geometry, &shape)?;
        if shape_config.first_layer_index != rank_plan.first_layer_index
            || shape_config.layer_count != rank_plan.layer_count
        {
            // C: "RING fixed stage plan disagrees with shape derivation".
            return Err(SparkStatus::ValidationFailed);
        }
        rank_plan.shape_configuration_hash = shape_config.configuration_hash;
    }
    rank_plan.max_packet_bytes = geometry.layer_major_transport_bytes_per_row()
        * u64::from(rank_plan.execution_row_capacity);
    rank_plan.host_name = rank_host_name(geometry, rank_index)?;
    rank_plan.link_pipeline(
        geometry,
        rank_index,
        geometry.stage_count,
        rank_index.wrapping_sub(1),
        rank_index + 1,
        port_base + rank_index + 1,
    )?;
    if stage.flags & STAGE_FLAG_FINAL_TOKEN != 0 {
        rank_plan.flags |= RANK_FLAG_FINAL_STAGE;
    }
    if stage.flags & STAGE_FLAG_DENSE_PREFIX != 0 {
        rank_plan.flags |= RANK_FLAG_DENSE_PREFIX;
    }
    validate_rank_plan(geometry, &rank_plan)?;
    Ok(rank_plan)
}

/// C: `SparkRingRuntimeBuildShapeRankPlan`.
pub fn build_shape_rank_plan(
    geometry: &RingModelGeometry,
    shape: &TpShapeDescriptor,
    logical_lane_capacity: u32,
    port_base: u32,
    tp_port_base: u32,
    quantization_mode: QuantizationMode,
) -> Result<RankPlan> {
    let shape_config = shape_node_config(geometry, shape)?;
    let node_index = shape.pp_stage_index * shape.tp_degree + shape.tp_rank;
    let mut rank_plan = RankPlan::blank(quantization_mode);
    rank_plan.rank_index = node_index;
    rank_plan.first_layer_index = shape_config.first_layer_index;
    rank_plan.layer_count = shape_config.layer_count;
    rank_plan.listen_port = port_base + node_index;
    rank_plan.logical_lane_capacity = logical_lane_capacity;
    rank_plan.maximum_speculative_rows_per_lane = geometry.max_speculative_rows_per_lane;
    rank_plan.execution_row_capacity = execution_row_capacity(geometry, logical_lane_capacity);
    rank_plan.hidden_dimension = geometry.hidden_dimension;
    rank_plan.bytes_per_sequence = geometry.hidden_bf16_bytes_per_sequence;
    rank_plan.tp_degree = shape.tp_degree;
    rank_plan.tp_rank = shape.tp_rank;
    rank_plan.pp_stage_count = shape.pp_stage_count;
    rank_plan.pp_stage_index = shape.pp_stage_index;
    rank_plan.tp_collective_listen_port = tp_port_base + node_index;
    rank_plan.shape_configuration_hash = shape_config.configuration_hash;
    rank_plan.max_packet_bytes = geometry.layer_major_transport_bytes_per_row()
        * u64::from(rank_plan.execution_row_capacity);
    rank_plan.host_name =
        rank_host_name(geometry, node_index).map_err(|_| SparkStatus::InvalidArgument)?;
    rank_plan.link_pipeline(
        geometry,
        shape.pp_stage_index,
        shape.pp_stage_count,
        node_index.wrapping_sub(shape.tp_degree),
        node_index + shape.tp_degree,
        port_base + node_index + shape.tp_degree,
    )?;
    // Collective peers: same-stage partners whose ranks differ in one bit.
    let mut step_count = 0u32;
    while (shape.tp_degree >> (step_count + 1)) != 0 {
        step_count += 1;
    }
    for step_index in 0..step_count {
        let partner_rank = shape.tp_rank ^ (1 << step_index);
        let partner_node = shape.pp_stage_index * shape.tp_degree + partner_rank;
        rank_plan.tp_peer_host_names.push(rank_host_name(geometry, partner_node)?);
        rank_plan.tp_peer_ports.push(tp_port_base + partner_node);
    }
    if shape.pp_stage_index == 0 {
        rank_plan.flags |= RANK_FLAG_DENSE_PREFIX;
    }
    if shape.pp_stage_index + 1 == shape.pp_stage_count {
        rank_plan.flags |= RANK_FLAG_FINAL_STAGE;
    }
    validate_rank_plan(geometry, &rank_plan)?;
    Ok(rank_plan)
}

/// C: `SparkRingRuntimeValidateRankPlan`.
pub fn validate_rank_plan(geometry: &RingModelGeometry, rank_plan: &RankPlan) -> Result<()> {
    let shape = TpShapeDescriptor::new(
        rank_plan.tp_degree,
        rank_plan.tp_rank,
        rank_plan.pp_stage_count,
        rank_plan.pp_stage_index,
    );
    let shape_config = shape_node_config(geometry, &shape)?;
    if rank_plan.abi_version != RING_RUNTIME_ABI_VERSION
        || rank_plan.descriptor_bytes != RANK_PLAN_DESCRIPTOR_BYTES
        || rank_plan.rank_index
            != rank_plan.pp_stage_index * rank_plan.tp_degree + rank_plan.tp_rank
        || rank_plan.rank_index >= rank_plan.pp_stage_count * rank_plan.tp_degree
        || (rank_plan.flags & !RANK_KNOWN_FLAGS) != 0
        || rank_plan.first_layer_index != shape_config.first_layer_index
        || rank_plan.layer_count != shape_config.layer_count
        || rank_plan.shape_configuration_hash != shape_config.configuration_hash
        || rank_plan.host_name.is_empty()
        || rank_plan.logical_lane_capacity == 0
        || rank_plan.logical_lane_capacity > geometry.max_batch_bucket
        || rank_plan.maximum_speculative_rows_per_lane != geometry.max_speculative_rows_per_lane
        || rank_plan.execution_row_capacity
            != execution_row_capacity(geometry, rank_plan.logical_lane_capacity)
        || rank_plan.hidden_dimension != geometry.hidden_dimension
        || rank_plan.bytes_per_sequence != geometry.hidden_bf16_bytes_per_sequence
        || rank_plan.max_packet_bytes
            != geometry.layer_major_transport_bytes_per_row()
                * u64::from(rank_plan.execution_row_capacity)
    {
        return Err(SparkStatus::InvalidArgument);
    }
    if rank_plan.flags & RANK_FLAG_HAS_PREVIOUS != 0 {
        let endpoint = rank_plan.input_endpoint.as_ref().ok_or(SparkStatus::InvalidArgument)?;
        validate_endpoint(endpoint)?;
    }
    if rank_plan.flags & RANK_FLAG_HAS_NEXT != 0 {
        let endpoint = rank_plan.output_endpoint.as_ref().ok_or(SparkStatus::InvalidArgument)?;
        validate_endpoint(endpoint)?;
    }
    if rank_plan.pp_stage_index == 0 && rank_plan.flags & RANK_FLAG_HAS_PREVIOUS != 0 {
        return Err(SparkStatus::InvalidArgument);
    }
    if rank_plan.pp_stage_index + 1 == rank_plan.pp_stage_count
        && rank_plan.flags & RANK_FLAG_HAS_NEXT != 0
    {
        return Err(SparkStatus::InvalidArgument);
    }
    Ok(())
}

/// C: `SparkRingRuntimeBuildMoePackPath`.
pub fn build_moe_pack_path(
    geometry: &RingModelGeometry,
    pack_root: &str,
    quantization_mode: QuantizationMode,
    layer_index: u32,
    tp_degree: u32,
    tp_rank: u32,
) -> Result<String> {
    if tp_degree == 0 || tp_rank >= tp_degree {
        return Err(SparkStatus::InvalidArgument);
    }
    let shape_tag =
        if tp_degree == 1 { String::new() } else { format!("_tp{tp_degree}r{tp_rank}") };
    if pack_root.is_empty() || layer_index >= geometry.weight_layer_count {
        return Err(SparkStatus::InvalidArgument);
    }
    let pack_path = geometry
        .pack_layout
        .pack_pattern(quantization_mode)
        .replace("{root}", pack_root)
        .replace("{layer:04}", &format!("{layer_index:04}"))
        .replace("{tag}", &shape_tag);
    if pack_path.len() >= PACK_PATH_BYTES {
        return Err(SparkStatus::CapacityExceeded);
    }
    Ok(pack_path)
}

/// Filesystem presence probe for pack validation (C: `stat` + `st_size > 0`
/// in `SparkRingRuntimePathIsPresent`). The seam follows the crate's trait
/// pattern (`KvPrefetchBackend`, `SwitchTier`, `SwapDevice`).
pub trait PackFileSystem {
    /// True when `path` exists and has nonzero size.
    fn path_is_present(&self, path: &str) -> bool;
}

/// C behavior over `std::fs` (`stat` succeeds and `st_size > 0`).
pub struct StdPackFileSystem;

impl PackFileSystem for StdPackFileSystem {
    fn path_is_present(&self, path: &str) -> bool {
        if path.is_empty() {
            return false;
        }
        std::fs::metadata(path).map(|metadata| metadata.len() > 0).unwrap_or(false)
    }
}

/// C: `SparkRingRuntimeValidateStageMoePackFiles`.
pub fn validate_stage_moe_pack_files(
    geometry: &RingModelGeometry,
    file_system: &dyn PackFileSystem,
    rank_plan: &RankPlan,
    pack_root: &str,
) -> Result<()> {
    validate_rank_plan(geometry, rank_plan)?;
    if pack_root.is_empty() {
        return Err(SparkStatus::InvalidArgument);
    }
    let manifests = [
        geometry.pack_layout.manifest(QuantizationMode::Fp8E4m3),
        geometry.pack_layout.manifest(QuantizationMode::Nvfp4),
    ];
    let selected_manifest_index = match rank_plan.quantization_mode {
        QuantizationMode::Fp8E4m3 => 0usize,
        QuantizationMode::Nvfp4 => 1usize,
    };
    let manifest_path = format!("{pack_root}/{}", manifests[selected_manifest_index]);
    if manifest_path.len() >= PACK_PATH_BYTES {
        return Err(SparkStatus::CapacityExceeded);
    }
    if !file_system.path_is_present(&manifest_path) {
        // C: "resident MoE pack manifest is missing".
        return Err(SparkStatus::NotFound);
    }
    for (manifest_index, manifest) in manifests.iter().enumerate() {
        if manifest_index == selected_manifest_index {
            continue;
        }
        let foreign_manifest_path = format!("{pack_root}/{manifest}");
        if foreign_manifest_path.len() >= PACK_PATH_BYTES {
            return Err(SparkStatus::CapacityExceeded);
        }
        if file_system.path_is_present(&foreign_manifest_path) {
            // C: "resident MoE pack root mixes quantization formats".
            return Err(SparkStatus::ModuleNotValidated);
        }
    }
    for layer_index in
        rank_plan.first_layer_index..rank_plan.first_layer_index + rank_plan.layer_count
    {
        if layer_index < geometry.first_routed_layer {
            continue;
        }
        let pack_path = build_moe_pack_path(
            geometry,
            pack_root,
            rank_plan.quantization_mode,
            layer_index,
            rank_plan.tp_degree,
            rank_plan.tp_rank,
        )?;
        if !file_system.path_is_present(&pack_path) {
            // C: "resident MoE layer pack is missing".
            return Err(SparkStatus::NotFound);
        }
    }
    Ok(())
}

/// C: `SparkRingRuntimeFinalEventRoute`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalEventRoute {
    pub abi_version: u32,
    pub descriptor_bytes: u32,
    pub source_rank_index: u32,
    pub sink_rank_index: u32,
    pub listen_port: u32,
    pub connect_port: u32,
    pub source_host_name: String,
    pub sink_host_name: String,
    pub route_name: String,
}

/// C: `SparkRingRuntimeFinalEvent` (272 bytes on the wire).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FinalEvent {
    pub magic: u32,
    pub descriptor_bytes: u32,
    /// C: `SparkStatus status` as its code.
    pub status: u32,
    pub program_id: u32,
    pub driver_dispatch_slot: u32,
    pub accepted_token_count: u32,
    pub completion_flags: u32,
    pub token_count: u32,
    pub token_ids: [u32; 8],
    pub draft_token_count: u32,
    pub draft_token_ids: [u32; 8],
    pub request_id: u64,
    pub sequence_id: u64,
    pub sequence_position: u64,
    pub service_time_ns: u64,
    pub control_generation: u64,
    pub transaction_id: u64,
    pub dispatch_generation: u64,
    pub request_generation: u64,
    pub step_generation: u64,
    pub step_chunk_index: u32,
    pub step_chunk_count: u32,
    pub transaction_phase: u32,
    pub reserved_transaction: u32,
    pub extension_flags: u32,
    pub reserved0: u32,
    pub dspark_draft: DsparkDraftResult,
}

/// C: `SparkRingRuntimeBuildFinalEventRoute`.
pub fn build_final_event_route(
    geometry: &RingModelGeometry,
    port_base: u32,
) -> Result<FinalEventRoute> {
    if port_base > 65535 - FINAL_EVENT_PORT_OFFSET {
        return Err(SparkStatus::InvalidArgument);
    }
    let route = FinalEventRoute {
        abi_version: RING_RUNTIME_ABI_VERSION,
        descriptor_bytes: FINAL_EVENT_ROUTE_DESCRIPTOR_BYTES,
        source_rank_index: geometry.stage_count - 1,
        sink_rank_index: 0,
        listen_port: port_base + FINAL_EVENT_PORT_OFFSET,
        connect_port: port_base + FINAL_EVENT_PORT_OFFSET,
        source_host_name: rank_host_name(geometry, geometry.stage_count - 1)?,
        sink_host_name: rank_host_name(geometry, 0)?,
        route_name: String::new(),
    };
    let route_name = format!("{}_to_{}_final_events", route.source_host_name, route.sink_host_name);
    if route_name.len() >= FINAL_EVENT_ROUTE_NAME_BYTES {
        return Err(SparkStatus::CapacityExceeded);
    }
    let route = FinalEventRoute { route_name, ..route };
    validate_final_event_route(geometry, &route)?;
    Ok(route)
}

/// C: `SparkRingRuntimeValidateFinalEventRoute`.
pub fn validate_final_event_route(
    geometry: &RingModelGeometry,
    route: &FinalEventRoute,
) -> Result<()> {
    let expected_source_host = rank_host_name(geometry, geometry.stage_count - 1)?;
    let expected_sink_host = rank_host_name(geometry, 0)?;
    let expected_route_name =
        format!("{expected_source_host}_to_{expected_sink_host}_final_events");
    if route.abi_version != RING_RUNTIME_ABI_VERSION
        || route.descriptor_bytes != FINAL_EVENT_ROUTE_DESCRIPTOR_BYTES
        || route.source_rank_index != geometry.stage_count - 1
        || route.sink_rank_index != 0
        || route.listen_port == 0
        || route.connect_port != route.listen_port
        || route.source_host_name != expected_source_host
        || route.sink_host_name != expected_sink_host
        || route.route_name != expected_route_name
    {
        return Err(SparkStatus::InvalidArgument);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::shape::{ShapeModelInputs, TpModelGeometry};
    use super::*;
    use std::collections::HashSet;

    /// GLM52-production geometry, supplied through config per deviation #1
    /// (values from the C tree: spark_glm52_model.h, rank_runtime.c, the
    /// `10.10.100.x` host table, and the 13×6 fixed stage table).
    pub fn glm52_geometry() -> RingModelGeometry {
        RingModelGeometry {
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
            shape_inputs: ShapeModelInputs::new(78, 6144, 2048, 12288, 512 + 64, 1),
            tp_geometry: TpModelGeometry::new(64, 192 + 64, 192 + 256, 256),
            stage_count: 13,
            default_stage_layer_counts: vec![6; 13],
            host_prefix: "10.10.100.".to_string(),
            host_index_base: 10,
            pack_layout: RingPackLayout::default(),
        }
    }

    #[test]
    fn geometry_validates() {
        assert_eq!(glm52_geometry().validate(), Ok(()));
        let mut bad = glm52_geometry();
        bad.default_stage_layer_counts = vec![5; 13];
        assert_eq!(bad.validate(), Err(SparkStatus::InvalidArgument));
    }

    #[test]
    fn dsa_bucket_matches_c_logic() {
        let geometry = glm52_geometry();
        assert_eq!(dsa_candidate_bucket(&geometry, 0), 0);
        assert_eq!(dsa_candidate_bucket(&geometry, 1_048_577), 0);
        assert_eq!(dsa_candidate_bucket(&geometry, 1), 2048);
        assert_eq!(dsa_candidate_bucket(&geometry, 2048), 2048);
        assert_eq!(dsa_candidate_bucket(&geometry, 2049), 4096);
        assert_eq!(dsa_candidate_bucket(&geometry, 600_000), 1_048_576);
        assert_eq!(dsa_candidate_bucket(&geometry, 1_048_576), 1_048_576);
    }

    #[test]
    fn execution_rows_clamp_to_bucket() {
        let geometry = glm52_geometry();
        assert_eq!(execution_row_capacity(&geometry, 0), 0);
        assert_eq!(execution_row_capacity(&geometry, 1025), 0);
        assert_eq!(execution_row_capacity(&geometry, 1), 8);
        assert_eq!(execution_row_capacity(&geometry, 128), 1024);
        assert_eq!(execution_row_capacity(&geometry, 1024), 1024);
    }

    #[test]
    fn quantization_round_trip() {
        assert_eq!(QuantizationMode::parse("fp8"), Ok(QuantizationMode::Fp8E4m3));
        assert_eq!(QuantizationMode::parse("nvfp4"), Ok(QuantizationMode::Nvfp4));
        assert_eq!(QuantizationMode::parse("int8"), Err(SparkStatus::InvalidArgument));
        assert_eq!(QuantizationMode::Fp8E4m3.name(), "fp8");
        assert_eq!(QuantizationMode::Nvfp4.name(), "nvfp4");
        assert_eq!(QuantizationMode::from_code(2), Some(QuantizationMode::Fp8E4m3));
        assert_eq!(QuantizationMode::from_code(1), Some(QuantizationMode::Nvfp4));
        assert_eq!(QuantizationMode::from_code(0), None);
    }

    #[test]
    fn fp8_plan_counts_and_backend_kind() {
        assert_eq!(QuantizationMode::Fp8E4m3.validate_fp8_plan_counts(4, 4), Ok(()));
        assert_eq!(
            QuantizationMode::Fp8E4m3.validate_fp8_plan_counts(3, 4),
            Err(SparkStatus::ModuleNotValidated)
        );
        assert_eq!(
            QuantizationMode::Fp8E4m3.validate_fp8_plan_counts(0, 0),
            Err(SparkStatus::ModuleNotValidated)
        );
        assert_eq!(QuantizationMode::Nvfp4.validate_fp8_plan_counts(0, 0), Ok(()));
        assert_eq!(
            QuantizationMode::Nvfp4.validate_fp8_plan_counts(1, 0),
            Err(SparkStatus::ModuleNotValidated)
        );
        assert_eq!(
            QuantizationMode::Fp8E4m3.expected_moe_backend_kind(),
            MOE_BACKEND_FP8_FLASHINFER_GROUPED
        );
        assert_eq!(QuantizationMode::Nvfp4.expected_moe_backend_kind(), MOE_BACKEND_NVFP4_B12X);
    }

    #[test]
    fn host_names_follow_the_table() {
        let geometry = glm52_geometry();
        assert_eq!(rank_host_name(&geometry, 0).unwrap(), "10.10.100.10");
        assert_eq!(rank_host_name(&geometry, 12).unwrap(), "10.10.100.22");
        assert_eq!(rank_host_name(&geometry, 13).unwrap_err(), SparkStatus::InvalidArgument);
    }

    #[test]
    fn fixed_rank_plans_validate_for_every_rank() {
        let geometry = glm52_geometry();
        for rank in 0..13u32 {
            let plan = build_rank_plan(
                &geometry,
                rank,
                1024,
                DEFAULT_PORT_BASE,
                QuantizationMode::Fp8E4m3,
            )
            .unwrap_or_else(|status| panic!("rank {rank} failed: {status}"));
            assert_eq!(plan.rank_index, rank);
            assert_eq!(plan.first_layer_index, rank * 6);
            assert_eq!(plan.layer_count, 6);
            assert_eq!(plan.listen_port, DEFAULT_PORT_BASE + rank);
            assert_eq!(plan.execution_row_capacity, 1024);
            assert_eq!(plan.flags & RANK_FLAG_HAS_PREVIOUS != 0, rank > 0);
            assert_eq!(plan.flags & RANK_FLAG_HAS_NEXT != 0, rank < 12);
            assert_eq!(plan.flags & RANK_FLAG_DENSE_PREFIX != 0, rank == 0);
            assert_eq!(plan.flags & RANK_FLAG_FINAL_STAGE != 0, rank == 12);
            if rank > 0 {
                let endpoint = plan.input_endpoint.as_ref().unwrap();
                assert_eq!(
                    endpoint.route_name,
                    format!("{}_to_{}_hidden", plan.previous_host_name, plan.host_name)
                );
                assert_eq!(validate_endpoint(endpoint), Ok(()));
            }
        }
        // Shape-derivation agreement pins the fixed plan to the shape hash.
        let plan =
            build_rank_plan(&geometry, 5, 64, DEFAULT_PORT_BASE, QuantizationMode::Nvfp4).unwrap();
        assert_eq!(plan.shape_configuration_hash, 2005884353472861467);
        assert_eq!(plan.execution_row_capacity, 512);
    }

    #[test]
    fn build_rank_plan_rejects_bad_arguments() {
        let geometry = glm52_geometry();
        assert_eq!(
            build_rank_plan(&geometry, 13, 64, DEFAULT_PORT_BASE, QuantizationMode::Fp8E4m3)
                .unwrap_err(),
            SparkStatus::InvalidArgument
        );
        assert_eq!(
            build_rank_plan(&geometry, 0, 0, DEFAULT_PORT_BASE, QuantizationMode::Fp8E4m3)
                .unwrap_err(),
            SparkStatus::InvalidArgument
        );
        assert_eq!(
            build_rank_plan(&geometry, 0, 1025, DEFAULT_PORT_BASE, QuantizationMode::Fp8E4m3)
                .unwrap_err(),
            SparkStatus::InvalidArgument
        );
        assert_eq!(
            build_rank_plan(&geometry, 0, 64, 65535 - 12, QuantizationMode::Fp8E4m3).unwrap_err(),
            SparkStatus::InvalidArgument
        );
    }

    #[test]
    fn shape_rank_plan_links_tp_peers() {
        let geometry = glm52_geometry();
        let shape = TpShapeDescriptor::new(2, 1, 6, 3);
        let plan = build_shape_rank_plan(
            &geometry,
            &shape,
            64,
            DEFAULT_PORT_BASE,
            61000,
            QuantizationMode::Fp8E4m3,
        )
        .unwrap();
        // node index = 3 * 2 + 1 = 7
        assert_eq!(plan.rank_index, 7);
        assert_eq!(plan.first_layer_index, 39);
        assert_eq!(plan.layer_count, 13);
        assert_eq!(plan.shape_configuration_hash, 9169625037557488130);
        assert_eq!(plan.host_name, "10.10.100.17");
        assert_eq!(plan.previous_rank_index, 5);
        assert_eq!(plan.next_rank_index, 9);
        assert_eq!(plan.tp_collective_listen_port, 61007);
        assert_eq!(plan.tp_peer_host_names, vec!["10.10.100.16".to_string()]);
        assert_eq!(plan.tp_peer_ports, vec![61006]);
        assert_eq!(plan.flags & RANK_FLAG_DENSE_PREFIX, 0);
        assert_eq!(plan.flags & RANK_FLAG_FINAL_STAGE, 0);
        let last = build_shape_rank_plan(
            &geometry,
            &TpShapeDescriptor::new(2, 0, 6, 5),
            64,
            DEFAULT_PORT_BASE,
            61000,
            QuantizationMode::Fp8E4m3,
        )
        .unwrap();
        assert_ne!(last.flags & RANK_FLAG_FINAL_STAGE, 0);
        assert_eq!(last.flags & RANK_FLAG_HAS_NEXT, 0);
    }

    #[test]
    fn validate_rank_plan_catches_tampering() {
        let geometry = glm52_geometry();
        let plan = build_rank_plan(&geometry, 4, 64, DEFAULT_PORT_BASE, QuantizationMode::Fp8E4m3)
            .unwrap();
        let mut bad = plan.clone();
        bad.flags |= 0x100;
        assert_eq!(validate_rank_plan(&geometry, &bad), Err(SparkStatus::InvalidArgument));
        let mut bad = plan.clone();
        bad.shape_configuration_hash ^= 1;
        assert_eq!(validate_rank_plan(&geometry, &bad), Err(SparkStatus::InvalidArgument));
        let mut bad = plan.clone();
        bad.execution_row_capacity += 8;
        assert_eq!(validate_rank_plan(&geometry, &bad), Err(SparkStatus::InvalidArgument));
        let mut bad = plan.clone();
        bad.host_name.clear();
        assert_eq!(validate_rank_plan(&geometry, &bad), Err(SparkStatus::InvalidArgument));
        let mut bad = plan;
        bad.input_endpoint = None;
        assert_eq!(validate_rank_plan(&geometry, &bad), Err(SparkStatus::InvalidArgument));
    }

    #[test]
    fn moe_pack_paths_match_c_naming() {
        let geometry = glm52_geometry();
        assert_eq!(
            build_moe_pack_path(&geometry, "/packs", QuantizationMode::Fp8E4m3, 7, 1, 0).unwrap(),
            "/packs/glm52_layer_0007_fp8_moe.spfp8"
        );
        assert_eq!(
            build_moe_pack_path(&geometry, "/packs", QuantizationMode::Nvfp4, 77, 2, 1).unwrap(),
            "/packs/glm52_layer_0077_b12x_moe_tp2r1.spb12x"
        );
        assert_eq!(
            build_moe_pack_path(&geometry, "", QuantizationMode::Fp8E4m3, 7, 1, 0).unwrap_err(),
            SparkStatus::InvalidArgument
        );
        assert_eq!(
            build_moe_pack_path(&geometry, "/packs", QuantizationMode::Fp8E4m3, 79, 1, 0)
                .unwrap_err(),
            SparkStatus::InvalidArgument
        );
        assert_eq!(
            build_moe_pack_path(&geometry, "/packs", QuantizationMode::Fp8E4m3, 7, 2, 2)
                .unwrap_err(),
            SparkStatus::InvalidArgument
        );
    }

    struct FakeFs {
        present: HashSet<String>,
    }

    impl PackFileSystem for FakeFs {
        fn path_is_present(&self, path: &str) -> bool {
            self.present.contains(path)
        }
    }

    fn full_pack_fs(geometry: &RingModelGeometry, plan: &RankPlan) -> FakeFs {
        let mut present = HashSet::new();
        present.insert(format!("/packs/{}", FP8_PACK_MANIFEST));
        for layer in plan.first_layer_index..plan.first_layer_index + plan.layer_count {
            if layer < geometry.first_routed_layer {
                continue;
            }
            present.insert(
                build_moe_pack_path(geometry, "/packs", plan.quantization_mode, layer, 1, 0)
                    .unwrap(),
            );
        }
        FakeFs { present }
    }

    #[test]
    fn stage_moe_pack_validation() {
        let geometry = glm52_geometry();
        let plan = build_rank_plan(&geometry, 0, 64, DEFAULT_PORT_BASE, QuantizationMode::Fp8E4m3)
            .unwrap();
        let fs = full_pack_fs(&geometry, &plan);
        assert_eq!(validate_stage_moe_pack_files(&geometry, &fs, &plan, "/packs"), Ok(()));
        // Rank 0 serves the dense prefix layers 0..6; routed packs start at
        // layer 3.
        // Missing manifest.
        let empty = FakeFs { present: HashSet::new() };
        assert_eq!(
            validate_stage_moe_pack_files(&geometry, &empty, &plan, "/packs"),
            Err(SparkStatus::NotFound)
        );
        // Mixed quantization formats.
        let mut mixed = full_pack_fs(&geometry, &plan);
        mixed.present.insert(format!("/packs/{B12X_PACK_MANIFEST}"));
        assert_eq!(
            validate_stage_moe_pack_files(&geometry, &mixed, &plan, "/packs"),
            Err(SparkStatus::ModuleNotValidated)
        );
        // Missing one layer pack.
        let mut missing = full_pack_fs(&geometry, &plan);
        missing.present.remove("/packs/glm52_layer_0003_fp8_moe.spfp8");
        assert_eq!(
            validate_stage_moe_pack_files(&geometry, &missing, &plan, "/packs"),
            Err(SparkStatus::NotFound)
        );
        // Empty root.
        assert_eq!(
            validate_stage_moe_pack_files(&geometry, &fs, &plan, "").unwrap_err(),
            SparkStatus::InvalidArgument
        );
    }

    #[test]
    fn final_event_route_matches_c_literals() {
        let geometry = glm52_geometry();
        let route = build_final_event_route(&geometry, DEFAULT_PORT_BASE).unwrap();
        assert_eq!(route.source_rank_index, 12);
        assert_eq!(route.sink_rank_index, 0);
        assert_eq!(route.listen_port, DEFAULT_PORT_BASE + 200);
        assert_eq!(route.connect_port, route.listen_port);
        assert_eq!(route.source_host_name, "10.10.100.22");
        assert_eq!(route.sink_host_name, "10.10.100.10");
        assert_eq!(route.route_name, "10.10.100.22_to_10.10.100.10_final_events");
        assert_eq!(
            build_final_event_route(&geometry, 65535 - 199).unwrap_err(),
            SparkStatus::InvalidArgument
        );
        let mut bad = route.clone();
        bad.connect_port += 1;
        assert_eq!(
            validate_final_event_route(&geometry, &bad).unwrap_err(),
            SparkStatus::InvalidArgument
        );
    }

    #[test]
    fn endpoint_validation_matrix() {
        let geometry = glm52_geometry();
        let endpoint = initialize_endpoint(&geometry, 8, "a_to_b_hidden");
        assert_eq!(validate_endpoint(&endpoint), Ok(()));
        let mut bad = endpoint.clone();
        bad.capability_flags = 0;
        assert_eq!(validate_endpoint(&bad), Err(SparkStatus::InvalidArgument));
        let mut bad = endpoint.clone();
        bad.max_packet_bytes = 1;
        assert_eq!(validate_endpoint(&bad), Err(SparkStatus::CapacityExceeded));
        let mut bad = endpoint.clone();
        bad.bytes_per_sequence = 42;
        assert_eq!(validate_endpoint(&bad), Err(SparkStatus::InvalidArgument));
        let mut bad = endpoint;
        bad.descriptor_bytes = 0;
        assert_eq!(validate_endpoint(&bad), Err(SparkStatus::AbiMismatch));
    }
}
