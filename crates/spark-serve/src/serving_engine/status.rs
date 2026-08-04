//! `ServingStatus`: the `SparkStatus` code space shared by the serving
//! engine, service runtime, service backend, and compat API.
//!
//! The C entry points return `SparkStatus` and use non-OK codes
//! (`PENDING`/`BUSY`/`NOT_FOUND`) as control flow; here those same codes are
//! carried as the `Err` side of `Result`, so a C `return SPARK_STATUS_BUSY;`
//! becomes `return Err(ServingStatus::Busy);`. `Ok` exists only for stored
//! `last_status` fields, never as an `Err` payload.

/// Port of `SparkStatus` (include/sparkpipe/spark_status.h).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, thiserror::Error)]
#[repr(u32)]
pub enum ServingStatus {
    /// `SPARK_STATUS_OK`.
    #[default]
    #[error("ok")]
    Ok = 0,
    /// `SPARK_STATUS_INVALID_ARGUMENT`.
    #[error("invalid argument")]
    InvalidArgument,
    /// `SPARK_STATUS_CAPACITY_EXCEEDED`.
    #[error("capacity exceeded")]
    CapacityExceeded,
    /// `SPARK_STATUS_NOT_FOUND`.
    #[error("not found")]
    NotFound,
    /// `SPARK_STATUS_IO_ERROR`.
    #[error("io error")]
    IoError,
    /// `SPARK_STATUS_PARSE_ERROR`.
    #[error("parse error")]
    ParseError,
    /// `SPARK_STATUS_SCHEMA_ERROR`.
    #[error("schema error")]
    SchemaError,
    /// `SPARK_STATUS_HASH_MISMATCH`.
    #[error("hash mismatch")]
    HashMismatch,
    /// `SPARK_STATUS_MODULE_NOT_VALIDATED`.
    #[error("module not validated")]
    ModuleNotValidated,
    /// `SPARK_STATUS_VALIDATION_FAILED`.
    #[error("validation failed")]
    ValidationFailed,
    /// `SPARK_STATUS_ABI_MISMATCH`.
    #[error("abi mismatch")]
    AbiMismatch,
    /// `SPARK_STATUS_TARGET_MISMATCH`.
    #[error("target mismatch")]
    TargetMismatch,
    /// `SPARK_STATUS_COMPILER_ERROR`.
    #[error("compiler error")]
    CompilerError,
    /// `SPARK_STATUS_DRIVER_LOAD_ERROR`.
    #[error("driver load error")]
    DriverLoadError,
    /// `SPARK_STATUS_ROUTE_NOT_FOUND`.
    #[error("route not found")]
    RouteNotFound,
    /// `SPARK_STATUS_BUSY`.
    #[error("busy")]
    Busy,
    /// `SPARK_STATUS_DUPLICATE`.
    #[error("duplicate")]
    Duplicate,
    /// `SPARK_STATUS_INTERNAL_ERROR`.
    #[error("internal error")]
    InternalError,
    /// `SPARK_STATUS_PENDING`.
    #[error("pending")]
    Pending,
    /// `SPARK_STATUS_UNSUPPORTED`.
    #[error("unsupported")]
    Unsupported,
}

impl ServingStatus {
    /// True for `SPARK_STATUS_OK`.
    pub fn is_ok(self) -> bool {
        self == ServingStatus::Ok
    }

    /// The C numeric status code (frame serialization).
    pub fn code(self) -> u32 {
        self as u32
    }
}

impl From<spark_text::tokenizer::TokenizerError> for ServingStatus {
    fn from(error: spark_text::tokenizer::TokenizerError) -> Self {
        use spark_text::tokenizer::TokenizerError as E;
        match error {
            E::InvalidArgument(_) => ServingStatus::InvalidArgument,
            E::CapacityExceeded => ServingStatus::CapacityExceeded,
            E::NotFound(_) => ServingStatus::NotFound,
            E::Io(_) => ServingStatus::IoError,
            E::Parse(_) => ServingStatus::ParseError,
            E::Schema(_) => ServingStatus::SchemaError,
            E::Duplicate(_) => ServingStatus::Duplicate,
            E::Internal(_) => ServingStatus::InternalError,
        }
    }
}

impl From<spark_text::chat_template::ChatTemplateError> for ServingStatus {
    fn from(error: spark_text::chat_template::ChatTemplateError) -> Self {
        use spark_text::chat_template::ChatTemplateError as E;
        match error {
            E::InvalidArgument => ServingStatus::InvalidArgument,
            E::CapacityExceeded => ServingStatus::CapacityExceeded,
        }
    }
}
