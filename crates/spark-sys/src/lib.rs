//! `spark-sys`: the only crate (besides `spark-abi`) where `unsafe` lives.
//!
//! Ports `src/spark_driver_loader.c` from the C tree: dlopen a
//! `model_driver.so`, resolve `SparkModelDriverGetInterface`, and run the same
//! fail-closed ABI validation the C loader performs before any function
//! pointer is callable. Safe wrapper types own the invariants; callers of this
//! crate never touch raw pointers.

pub mod driver;
pub mod error;

pub use driver::{ModelDriver, ProgramDescriptorInfo};
pub use error::DriverLoadError;
