//! Error surface for model driver loading. Codes mirror the C `SparkStatus`
//! values the loader produces so logs stay comparable across the two trees.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DriverLoadError {
    #[error("invalid argument: {0}")]
    InvalidArgument(&'static str),

    #[error("cannot load model driver '{path}': {reason}")]
    Load { path: String, reason: String },

    #[error("driver '{path}' does not export SparkModelDriverGetInterface")]
    MissingInterfaceSymbol { path: String },

    #[error("model driver interface ABI is invalid")]
    InterfaceAbi,

    #[error("model driver descriptor ABI is invalid")]
    DescriptorAbi,

    #[error("driver program descriptor {0} is invalid")]
    ProgramAbi(u32),

    #[error("driver program profile {0} is inconsistent with its descriptor")]
    ProfileAbi(u32),

    #[error("driver program '{0}' claims no host staging but reports a nonzero staging ceiling")]
    HostStagingClaim(String),

    #[error("driver program '{0}' claims no device memcpy but reports a nonzero memcpy ceiling")]
    DeviceMemcpyClaim(String),

    #[error("driver program '{0}' claims validated latency without a validated latency value")]
    ValidatedLatencyClaim(String),

    #[error("driver program '{0}' claims private queue pressure without private queues")]
    PrivateQueueClaim(String),

    #[error("driver contains duplicate program descriptors")]
    DuplicateProgram,

    #[error("driver target '{driver}' does not match node target '{node}'")]
    TargetMismatch { driver: String, node: String },
}
