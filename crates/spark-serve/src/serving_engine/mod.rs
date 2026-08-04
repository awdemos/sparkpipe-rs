//! Serving engine orchestration — port of `api/serving_engine.c`,
//! `api/service.c`, `api/compat_api.c`, and the
//! `spark_service_backend.h` seam.
//!
//! - [`status`]: the shared `SparkStatus` code space
//! - [`bridge`]: the [`bridge::ServingRequestApi`] trait — the boundary the
//!   C draws as direct calls into the concrete `SparkRequestApi`
//!   (`api/request.c`, ported concurrently in `crate::request_api`). The
//!   request-API port should be adapted onto this trait; the FFI/dynamic
//!   side is unaffected.
//! - [`engine`]: the serving engine (request records, event ring,
//!   prefill/decode pump)
//! - [`service`]: the service runtime (clients, request mappings, frame
//!   protocol)
//! - [`backend`]: the node-runtime backend seam as a trait; the `dlopen`
//!   interface loader belongs to `spark-sys` (Phase 4)
//! - [`compat`]: the OpenAI/Anthropic-compatible JSON surface

pub mod backend;
pub mod bridge;
pub mod compat;
pub mod engine;
pub mod service;
pub mod status;

pub use backend::{
    validate_backend_capabilities, ServiceBackend, ServiceBackendConfiguration,
    ServiceBackendPollDescriptor, ServiceBackendView,
};
pub use bridge::{
    ApiSubmitRequest, DecodeDispatchLaneView, DecodeDispatchView, Dispatch, KvBlockTableView,
    PrefillDispatchLaneView, PrefillDispatchView, RequestApiCounters, RequestApiState,
    RequestCacheState, ServingRequestApi,
};
pub use compat::CompatTextRequest;
pub use engine::{
    DecodeFunction, PrefillFunction, ReleaseSequenceFunction, ServingDecodeDispatch,
    ServingDecodeResult, ServingEngine, ServingEngineConfiguration, ServingEvent, ServingEventKind,
    ServingPrefillDispatch, ServingRequestHandle, ServingStats, SubmitFailure, SubmitResult,
    SubmitTextRequest, SubmitTokenIdsRequest,
};
pub use service::{
    FrameHeader, ServiceClientId, ServiceEvent, ServiceEventKind, ServiceRequestId, ServiceRuntime,
    ServiceStats, ServiceSubmitResult, ServiceSubmitTextRequest, ServiceSubmitTokenIdsRequest,
};
pub use status::ServingStatus;
