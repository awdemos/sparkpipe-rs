//! `spark-node`: node runtime, ported from the C tree's `node/`.
//!
//! - [`backend`]: per-rank serving backend pump — sockets→driver event loop
//!   driving C stage modules through `spark-sys` (port of `node/backend.c`)
//! - [`rank_daemon`]: rank daemon lifecycle (port of `node/rank_daemon.c`)

pub mod backend;
pub mod rank_daemon;
