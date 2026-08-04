//! Service backend interface — port of `spark_service_backend.h`.
//!
//! This is the seam to the node runtime. In C the interface is a struct of
//! function pointers obtained by `dlopen`ing a backend module
//! (`SparkServiceBackendLoadInterfaceFromSharedObject`); here it is the
//! [`ServiceBackend`] trait. The dynamic-library loader — `dlopen`/`dlsym`,
//! the `SparkServiceBackendGetInterface` symbol, ABI-word validation —
//! involves `unsafe` FFI and belongs to `spark-sys` (Phase 4); this module
//! deliberately stays safe-Rust only.
//!
//! The C's null-function-pointer checks in
//! `SparkServiceBackendValidateInterface` are unrepresentable for a trait
//! object; [`validate_backend_capabilities`] ports the capability-flag half
//! of that validation.

use super::service::{ServiceRuntime, ServiceStats};
use super::status::ServingStatus;

/// `SPARK_SERVICE_BACKEND_ABI_VERSION`.
pub const SERVICE_BACKEND_ABI_VERSION: u32 = 10;
/// `SPARK_SERVICE_BACKEND_INTERFACE_SYMBOL`.
pub const SERVICE_BACKEND_INTERFACE_SYMBOL: &str = "SparkServiceBackendGetInterface";

/// `SPARK_SERVICE_BACKEND_CONFIGURATION_FLAG_DSPARK`.
pub const CONFIGURATION_FLAG_DSPARK: u32 = 0x0000_0001;
/// `SPARK_SERVICE_BACKEND_CONFIGURATION_FLAG_MTP`.
pub const CONFIGURATION_FLAG_MTP: u32 = 0x0000_0002;
/// `SPARK_SERVICE_BACKEND_CONFIGURATION_KNOWN_FLAGS`.
pub const CONFIGURATION_KNOWN_FLAGS: u32 = CONFIGURATION_FLAG_DSPARK | CONFIGURATION_FLAG_MTP;

/// `SPARK_SERVICE_BACKEND_CAPABILITY_SERVICE_RUNTIME`.
pub const CAPABILITY_SERVICE_RUNTIME: u32 = 0x0000_0001;
/// `SPARK_SERVICE_BACKEND_CAPABILITY_RING_RUNTIME`.
pub const CAPABILITY_RING_RUNTIME: u32 = 0x0000_0002;
/// `SPARK_SERVICE_BACKEND_CAPABILITY_TOKENIZER`.
pub const CAPABILITY_TOKENIZER: u32 = 0x0000_0004;
/// `SPARK_SERVICE_BACKEND_CAPABILITY_POLL_DESCRIPTORS`.
pub const CAPABILITY_POLL_DESCRIPTORS: u32 = 0x0000_0008;
/// `SPARK_SERVICE_BACKEND_REQUIRED_PRODUCTION_CAPS`.
pub const REQUIRED_PRODUCTION_CAPS: u32 =
    CAPABILITY_SERVICE_RUNTIME | CAPABILITY_RING_RUNTIME | CAPABILITY_POLL_DESCRIPTORS;

/// `SPARK_SERVICE_BACKEND_POLL_READ`.
pub const POLL_READ: u32 = 0x0000_0001;
/// `SPARK_SERVICE_BACKEND_POLL_WRITE`.
pub const POLL_WRITE: u32 = 0x0000_0002;

/// `SparkServiceBackendValidateInterface`, capability half.
///
/// Returns `Err(InvalidArgument)` when `capability_flags` does not cover
/// `required_capability_flags` (the C additionally validates the ABI words
/// and rejects null function pointers; both belong to the `spark-sys` FFI
/// loader / the type system here).
pub fn validate_backend_capabilities(
    capability_flags: u32,
    required_capability_flags: u32,
) -> Result<(), ServingStatus> {
    if (capability_flags & required_capability_flags) != required_capability_flags {
        return Err(ServingStatus::InvalidArgument);
    }
    Ok(())
}

/// Backend configuration (`SparkServiceBackendConfiguration`).
#[derive(Debug, Clone, Default)]
pub struct ServiceBackendConfiguration {
    /// `SPARK_SERVICE_BACKEND_CONFIGURATION_FLAG_*` bits.
    pub flags: u32,
    /// Maximum concurrently active sequences.
    pub max_active_sequence_count: u32,
    /// Base port for node interconnect.
    pub port_base: u32,
    /// Logical KV block capacity.
    pub kv_logical_block_capacity: u32,
    /// Model quantization mode (`SPARK_STAGE_PLAN_QUANTIZATION_*`).
    pub model_quantization_mode: u32,
    /// MoE pack root directory.
    pub moe_pack_root: Option<String>,
    /// Stagepack root directory.
    pub stagepack_root: Option<String>,
    /// Transport module shared-object path.
    pub transport_shared_object_path: Option<String>,
    /// Driver module shared-object path.
    pub driver_shared_object_path: Option<String>,
    /// Node-context-builder module shared-object path.
    pub node_context_builder_shared_object_path: Option<String>,
    /// Embedding pack path.
    pub embedding_pack_path: Option<String>,
    /// Driver program name.
    pub driver_program_name: Option<String>,
    /// Node target triple.
    pub node_target: Option<String>,
    /// Tokenizer file path.
    pub tokenizer_path: Option<String>,
    /// Final-event bind address.
    pub final_event_bind_address: Option<String>,
    /// Final-event return host.
    pub final_event_return_host: Option<String>,
    /// CUDA-resident socket path.
    pub cuda_resident_socket_path: Option<String>,
}

/// Backend view (`SparkServiceBackendView`), owned-data form.
///
/// The C view also carries `service`/`tokenizer` pointers; here the runtime
/// is reached through [`ServiceBackend::service`] instead, and the tokenizer
/// through that runtime's engine.
#[derive(Debug, Clone, Default)]
pub struct ServiceBackendView {
    /// Node runtime finished initializing.
    pub runtime_initialized: bool,
    /// Local control channel is ready.
    pub local_control_ready: bool,
    /// Configured KV context limit in tokens.
    pub configured_kv_context_limit_tokens: u32,
    /// Configured active-sequence cap.
    pub configured_max_active_sequences: u32,
    /// Transport capability flags.
    pub transport_capability_flags: u32,
    /// Speculation configuration flags.
    pub speculation_configuration_flags: u32,
    /// Request-API configuration flags.
    pub request_api_configuration_flags: u32,
    /// Adaptive decode batch width currently in effect.
    pub adaptive_decode_batch_width: u32,
    /// Decode batch capacity.
    pub decode_batch_capacity: u32,
    /// Prefill wave token count.
    pub prefill_wave_token_count: u32,
    /// Release generation.
    pub release_generation: u64,
    /// First blocker string, when not ready.
    pub first_blocker: Option<String>,
    /// Release identifier.
    pub release_id: Option<String>,
    /// Release git commit.
    pub release_git_commit: Option<String>,
    /// Transport shared-object path in use.
    pub transport_shared_object_path: Option<String>,
}

/// Poll descriptor (`SparkServiceBackendPollDescriptor`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ServiceBackendPollDescriptor {
    /// File descriptor to poll.
    pub fd: i32,
    /// [`POLL_READ`]/[`POLL_WRITE`] interest mask.
    pub events: u32,
}

/// The node-runtime backend seam (`SparkServiceBackendInterface`).
///
/// `initialize`/`destroy` from the C interface map to trait-object
/// construction and `Drop`.
pub trait ServiceBackend {
    /// `backend_interface->capability_flags`.
    fn capability_flags(&self) -> u32;
    /// `SparkServiceBackendGetViewFunction`.
    fn get_view(&mut self) -> Result<ServiceBackendView, ServingStatus>;
    /// `SparkServiceBackendPumpFunction` (fills `stats_out` as in C).
    fn pump(
        &mut self,
        max_dispatch_steps: u32,
        stats_out: &mut ServiceStats,
    ) -> Result<(), ServingStatus>;
    /// `SparkServiceBackendGetPollDescriptorsFunction`.
    ///
    /// Required when [`CAPABILITY_POLL_DESCRIPTORS`] is reported; backends
    /// without that capability may return `Err(ServingStatus::Unsupported)`.
    fn poll_descriptors(&mut self) -> Result<Vec<ServiceBackendPollDescriptor>, ServingStatus> {
        Err(ServingStatus::Unsupported)
    }
    /// Access to the embedded service runtime — the Rust form of the C
    /// view's `service` pointer. Backends reporting
    /// [`CAPABILITY_SERVICE_RUNTIME`] must return `Some`.
    fn service(&mut self) -> Option<&mut ServiceRuntime> {
        None
    }
}
