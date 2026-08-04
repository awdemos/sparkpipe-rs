//! `spark-sched`: scheduling plane, ported from the C tree's `scheduler/`.
//!
//! - [`stage_plan`]: per-stage layer/geometry planning (port of
//!   `scheduler/stage_plan.c`)
//! - [`scheduler`]: admission, cohort formation, chunked prefill, prefix
//!   reuse, CUDA-graph padding buckets (port of `scheduler/scheduler.c`)
//! - [`work_control`]: ring work control, per-step batch re-formation
//!   (port of `scheduler/work_control.c`)
//! - [`long_context`]: long-context scheduling helpers (port of
//!   `scheduler/long_context.c`)
//! - [`topology_switch`]: topology switching policy (port of
//!   `scheduler/topology_switch.c`)
//!
//! Deliberate deviations from C (see docs/PORT_LEDGER.md):
//! - capacities come from configuration/model contract, not baked-in
//!   GLM52 constants (the C headers alias `SPARK_GLM52_MODEL_*` into
//!   nominally generic limits)

pub mod long_context;
pub mod scheduler;
pub mod stage_plan;
pub mod topology_switch;
pub mod work_control;
