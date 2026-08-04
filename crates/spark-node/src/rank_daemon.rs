//! Rank daemon port: `node/rank_daemon.c` + `node/rank_runtime.c`, plus the
//! pieces of `runtime/work_transaction.c`, `serving/spark_cuda_resident_ipc.c`,
//! `ring/transport/hidden_transport.c`, and
//! `model-families/glm52/src/spark_glm52_shape_config.c` that the daemon
//! calls.
//!
//! Module map:
//! - [`status`]: the shared `SparkStatus` code space (reuses
//!   `spark_serve::serving_engine::ServingStatus`).
//! - [`shape`]: shape-driven node-config derivation (FNV-1a configuration
//!   hash, bit-exact with the C).
//! - [`ring_runtime`]: rank plans, quantization modes, endpoint validation,
//!   MoE pack path/validation, final-event route/event builders.
//! - [`resident_ipc`]: the CUDA-resident Unix-socket IPC protocol codecs.
//! - [`wire`]: wire codecs for work packets, transaction acknowledgements,
//!   and final events.
//! - [`work_ledger`]: the work transaction ledger (kept here, not in
//!   `spark-sched`, because the daemon is its only consumer).
//! - [`net`]: the OS seam (links, monotonic clock, wake pipe); `unsafe` is
//!   confined to the documented libc calls there.
//! - [`daemon`]: the daemon state machine (work queue, dependency
//!   ordering, pump, inflight completion tracking, final events,
//!   CUDA-resident client, argument parsing).
//!
//! Cross-cutting port decisions:
//! - No GLM52 model constants in Rust (port deviation #1): every
//!   model-derived value arrives through config structs
//!   ([`ring_runtime::RingModelGeometry`], `spark_sched`'s
//!   `WorkControlConfig`, [`shape::ShapeModelInputs`] /
//!   [`shape::TpModelGeometry`]). The C reference values appear only in
//!   doc comments and test fixtures.
//! - The dlopen'd node-context builder / hidden transport / model driver
//!   modules stay behind `spark-sys`; the daemon talks to them through the
//!   [`daemon::SubmitEngine`] and [`daemon::WorkForwarder`] seams. The
//!   production poll loop (`main`) wiring is deferred; the state machine,
//!   IPC client, and socket pump are ported for real.
//! - Error-text side channels (`SparkReportError`) are dropped; the
//!   returned status codes are identical to the C.

pub mod daemon;
pub mod net;
pub mod resident_ipc;
pub mod ring_runtime;
pub mod shape;
pub mod status;
pub mod wire;
pub mod work_ledger;

pub use daemon::{
    distributed_initialize_acknowledgement, handle_resident_message, parse_arguments,
    pump_work_control, validate_resident_contract, CudaResidentEngine, DaemonCounters,
    DriverCompletionRecord, EngineCompletion, NoForwarder, RankDaemonConfig, RankDaemonCore,
    ResidentMessage, ResidentMessagePump, SocketForwarder, SubmitEngine, SubmitOutcome,
    WorkForwarder, WorkState, CONNECT_RETRY_NS, DRIVER_COMPLETION_QUEUE_CAPACITY,
    FINAL_EVENT_QUEUE_CAPACITY, INFLIGHT_COMPLETION_CAPACITY, INFLIGHT_TRANSACTION_CAPACITY,
    MAX_PIPELINE_SLOT_COUNT, RUNNER_PROGRESS_NS, TRANSACTION_LEDGER_CAPACITY, WORK_QUEUE_CAPACITY,
};
pub use net::{configure_low_latency_tcp, io_status, min_nonzero_ns, monotonic_ns, Link, WakePipe};
pub use ring_runtime::{
    build_final_event_route, build_fixed_stage_plan, build_moe_pack_path, build_rank_plan,
    build_shape_rank_plan, dsa_candidate_bucket, execution_row_capacity, rank_host_name,
    validate_endpoint, validate_final_event_route, validate_rank_plan,
    validate_stage_moe_pack_files, FinalEvent, FinalEventRoute, HiddenTransportEndpoint,
    PackFileSystem, QuantizationMode, RankPlan, RingModelGeometry, RingPackLayout,
    StdPackFileSystem, FINAL_EVENT_DESCRIPTOR_BYTES, FINAL_EVENT_MAGIC, RANK_FLAG_DENSE_PREFIX,
    RANK_FLAG_FINAL_STAGE, RANK_FLAG_HAS_NEXT, RANK_FLAG_HAS_PREVIOUS,
};
pub use shape::{
    derive_node_config, ShapeModelInputs, ShapeNodeConfig, TpModelGeometry, TpShapeDescriptor,
};
pub use status::{status_code, status_from_code, Result, SparkStatus};
pub use wire::{
    acknowledgement_from_wire, acknowledgement_to_wire, final_event_from_wire, final_event_to_wire,
    identity_fingerprint, initialize_acknowledgement, packet_dedup_hash32, packet_from_wire,
    packet_to_wire, validate_acknowledgement, WorkAcknowledgement, ACKNOWLEDGEMENT_WIRE_BYTES,
    FINAL_EVENT_WIRE_BYTES, IDENTITY_WIRE_BYTES,
};
pub use work_ledger::{
    state_is_terminal, LedgerEntry, Observation, TransactionLedger, STATE_ACCEPTED,
    STATE_CANCELLED, STATE_COMMITTED, STATE_EXECUTING, STATE_FAILED, STATE_PREPARED,
};
