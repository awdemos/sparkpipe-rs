//! Shared status space for the rank daemon port.
//!
//! The C entry points return `SparkStatus`; the port carries the same codes
//! as the `Err` side of `Result`, reusing the serving plane's port of the
//! `SparkStatus` code space (`spark_serve::serving_engine::ServingStatus`)
//! instead of growing a fourth copy of the enum.

pub use spark_serve::serving_engine::ServingStatus as SparkStatus;

use spark_sched::stage_plan::StagePlanError;
use spark_sched::work_control::WorkControlError;

/// Result alias used across the rank daemon module.
pub type Result<T> = std::result::Result<T, SparkStatus>;

/// `SparkStatus` code of `SPARK_STATUS_UNSUPPORTED` (the largest code).
pub const STATUS_UNSUPPORTED_CODE: u32 = SparkStatus::Unsupported as u32;

/// Numeric code of a status (C: `(uint32_t)status`).
pub fn status_code(status: SparkStatus) -> u32 {
    status as u32
}

/// Inverse of [`status_code`]; `None` when the code has no `SparkStatus`.
pub fn status_from_code(code: u32) -> Option<SparkStatus> {
    Some(match code {
        0 => SparkStatus::Ok,
        1 => SparkStatus::InvalidArgument,
        2 => SparkStatus::CapacityExceeded,
        3 => SparkStatus::NotFound,
        4 => SparkStatus::IoError,
        5 => SparkStatus::ParseError,
        6 => SparkStatus::SchemaError,
        7 => SparkStatus::HashMismatch,
        8 => SparkStatus::ModuleNotValidated,
        9 => SparkStatus::ValidationFailed,
        10 => SparkStatus::AbiMismatch,
        11 => SparkStatus::TargetMismatch,
        12 => SparkStatus::CompilerError,
        13 => SparkStatus::DriverLoadError,
        14 => SparkStatus::RouteNotFound,
        15 => SparkStatus::Busy,
        16 => SparkStatus::Duplicate,
        17 => SparkStatus::InternalError,
        18 => SparkStatus::Pending,
        19 => SparkStatus::Unsupported,
        _ => return None,
    })
}

/// Map a work-control error onto its `SparkStatus` code (same names).
pub fn from_work_control(error: WorkControlError) -> SparkStatus {
    match error {
        WorkControlError::InvalidArgument => SparkStatus::InvalidArgument,
        WorkControlError::CapacityExceeded => SparkStatus::CapacityExceeded,
        WorkControlError::NotFound => SparkStatus::NotFound,
        WorkControlError::ModuleNotValidated => SparkStatus::ModuleNotValidated,
        WorkControlError::ValidationFailed => SparkStatus::ValidationFailed,
        WorkControlError::AbiMismatch => SparkStatus::AbiMismatch,
        WorkControlError::Busy => SparkStatus::Busy,
        WorkControlError::InternalError => SparkStatus::InternalError,
    }
}

/// Map a stage-plan error onto its `SparkStatus` code.
pub fn from_stage_plan(error: &StagePlanError) -> SparkStatus {
    match error {
        StagePlanError::InvalidArgument(_) => SparkStatus::InvalidArgument,
        StagePlanError::CapacityExceeded(_) => SparkStatus::CapacityExceeded,
        StagePlanError::AbiMismatch(_) => SparkStatus::AbiMismatch,
        StagePlanError::InternalError(_) => SparkStatus::InternalError,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_code_round_trips() {
        for code in 0..=STATUS_UNSUPPORTED_CODE {
            let status = status_from_code(code).expect("known code");
            assert_eq!(status_code(status), code);
        }
        assert_eq!(status_from_code(20), None);
    }
}
