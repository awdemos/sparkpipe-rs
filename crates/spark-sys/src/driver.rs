//! Model driver loading — a 1:1 port of `SparkLoadModelDriver` /
//! `SparkValidateLoadedModelDriverInterface` from `src/spark_driver_loader.c`.
//!
//! A `ModelDriver` owns the dlopen handle; the interface pointer it hands out
//! borrows from the library's static storage and is valid exactly as long as
//! the `ModelDriver` itself. Dropping the `ModelDriver` dlcloses the library,
//! mirroring `SparkUnloadModelDriver`.

use std::ffi::CStr;
use std::path::Path;

use libloading::Library;
use spark_abi::*;

use crate::error::DriverLoadError;

/// Union of all flags the loader accepts (SPARK_MODEL_DRIVER_KNOWN_PROGRAM_FLAGS).
const KNOWN_PROGRAM_FLAGS: u32 = SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION
    | SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED
    | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE
    | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE
    | SPARK_MODEL_DRIVER_PROGRAM_FLAG_JIT_KV_CACHE
    | SPARK_MODEL_DRIVER_PROGRAM_FLAG_ZERO_COPY_NODE_CONTEXT
    | SPARK_MODEL_DRIVER_PROGRAM_FLAG_PRIVATE_QUEUE_PRESSURE
    | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_HOST_STAGING
    | SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE
    | SPARK_MODEL_DRIVER_PROGRAM_FLAG_VALIDATED_LATENCY
    | SPARK_MODEL_DRIVER_PROGRAM_FLAG_CAPTURED_CUDA_GRAPH
    | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_DEVICE_MEMCPY
    | SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_PRIVATE_EXPERT_QUEUES
    | SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_EVENT_DEPENDENCIES
    | SPARK_MODEL_DRIVER_PROGRAM_FLAG_RESIDENCY_AFFINITY_REQUIRED
    | SPARK_MODEL_DRIVER_PROGRAM_FLAG_BATCH_SHAPE_FIXED
    | SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT
    | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT
    | SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT
    | SPARK_MODEL_DRIVER_PROGRAM_FLAG_BULK_PREFILL;

/// A validated, loaded model driver (`model_driver.so`).
///
/// Construction is the only unsafe operation; a constructed value guarantees
/// the same invariants `SparkLoadModelDriver` establishes on success.
pub struct ModelDriver {
    // Field order matters: `library` must be dropped last. Rust drops fields
    // in declaration order, and the interface borrows the library's storage.
    interface: &'static SparkModelDriverInterface,
    /// Held purely for RAII — dropping it dlcloses the library, which must
    /// happen only after `interface` is no longer usable.
    #[allow(dead_code)]
    library: Library,
}

/// Read-only view of one program descriptor (name borrowed from the driver).
pub struct ProgramDescriptorInfo {
    pub program_id: u32,
    pub flags: u32,
    pub max_inflight: u32,
    pub name: String,
}

impl ModelDriver {
    /// Load and validate `driver_path`. If `expected_target` is `Some`, the
    /// driver descriptor's target must match it exactly (TARGET_MISMATCH).
    pub fn load(
        driver_path: &Path,
        expected_target: Option<&str>,
    ) -> Result<Self, DriverLoadError> {
        if driver_path.as_os_str().is_empty() {
            return Err(DriverLoadError::InvalidArgument("empty driver path"));
        }
        // SAFETY: dlopen of a path chosen by the operator. We resolve only the
        // documented interface symbol and validate everything before use.
        let library = unsafe {
            Library::new(driver_path).map_err(|e| DriverLoadError::Load {
                path: driver_path.display().to_string(),
                reason: e.to_string(),
            })?
        };

        // SAFETY: the symbol name is the fixed ABI entry point; the returned
        // function has no preconditions beyond being called.
        let get_interface: libloading::Symbol<
            unsafe extern "C" fn() -> *const SparkModelDriverInterface,
        > = unsafe {
            library.get(b"SparkModelDriverGetInterface\0").map_err(|_| {
                DriverLoadError::MissingInterfaceSymbol { path: driver_path.display().to_string() }
            })?
        };
        // SAFETY: ABI contract — returns a pointer to static storage owned by
        // the library, valid for the library's lifetime. Null is an ABI
        // violation; treat it as an invalid interface rather than deref.
        let interface_ptr = unsafe { get_interface() };
        if interface_ptr.is_null() {
            return Err(DriverLoadError::InterfaceAbi);
        }
        // SAFETY: validated non-null above; lifetime extended to 'static is
        // sound because `ModelDriver` owns `library` and drops it last, so
        // the referent outlives every use through `self.interface`.
        let interface: &'static SparkModelDriverInterface = unsafe { &*interface_ptr };

        validate_interface(interface, expected_target)?;

        Ok(Self { interface, library })
    }

    /// The validated driver descriptor (model id, target, program table...).
    pub fn descriptor(&self) -> &SparkModelDriverDescriptor {
        // SAFETY: validate_interface established descriptor non-null and
        // descriptor_bytes covers the full struct.
        unsafe { &*self.interface.descriptor }
    }

    /// Program descriptors as Rust-owned views (mirrors the C loop over
    /// `descriptor->programs`).
    pub fn programs(&self) -> Vec<ProgramDescriptorInfo> {
        let descriptor = self.descriptor();
        // SAFETY: validate_interface established `programs` non-null and
        // walked exactly `program_count` entries through the same pointer.
        let programs = unsafe {
            std::slice::from_raw_parts(descriptor.programs, descriptor.program_count as usize)
        };
        programs
            .iter()
            .map(|p| ProgramDescriptorInfo {
                program_id: p.program_id,
                flags: p.flags,
                max_inflight: p.max_inflight,
                // SAFETY: validation required name to be a valid C string.
                name: unsafe { CStr::from_ptr(p.name) }.to_string_lossy().into_owned(),
            })
            .collect()
    }

    /// Find a program by name (`SparkFindLoadedModelDriverProgram`).
    pub fn find_program(&self, name: &str) -> Option<ProgramDescriptorInfo> {
        self.programs().into_iter().find(|p| p.name == name)
    }

    /// Raw interface access for upcoming stage-control calls (create/admit/
    /// snapshot). Exposed crate-internally; safe wrappers will be built per
    /// call site as the node runtime lands.
    #[allow(dead_code)]
    pub(crate) fn interface(&self) -> &'static SparkModelDriverInterface {
        self.interface
    }
}

impl std::fmt::Debug for ModelDriver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let d = self.descriptor();
        // SAFETY: validation established these are valid C strings.
        let (model_id, target) = unsafe {
            (
                CStr::from_ptr(d.model_id).to_string_lossy(),
                CStr::from_ptr(d.target).to_string_lossy(),
            )
        };
        f.debug_struct("ModelDriver")
            .field("model_id", &model_id)
            .field("target", &target)
            .field("program_count", &d.program_count)
            .finish()
    }
}

/// C string field validity: non-null and non-empty (`SparkLoadedModelDriverTextIsValid`).
///
/// # Safety
/// `ptr` must point to a valid NUL-terminated C string when non-null.
unsafe fn text_is_valid(ptr: *const libc::c_char) -> bool {
    // SAFETY: caller guarantees a valid C string; we only read the first byte.
    !ptr.is_null() && unsafe { *ptr } != 0
}

/// SparkSha256HexIsValid: exactly 64 lowercase-or-uppercase hex chars.
///
/// # Safety
/// `ptr` must point to a valid NUL-terminated C string when non-null.
unsafe fn sha256_hex_is_valid(ptr: *const libc::c_char) -> bool {
    if ptr.is_null() {
        return false;
    }
    // SAFETY: caller guarantees a valid C string.
    let bytes = unsafe { CStr::from_ptr(ptr) }.to_bytes();
    bytes.len() == 64 && bytes.iter().all(|b| b.is_ascii_hexdigit())
}

/// # Safety
/// `ptr` must be null or point to a valid NUL-terminated C string.
unsafe fn cstr_to_string(ptr: *const libc::c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    // SAFETY: caller guarantees a valid C string.
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
}

fn validate_program(
    program: &SparkModelDriverProgramDescriptor,
    program_index: u32,
) -> Result<(), DriverLoadError> {
    // SAFETY: name/submit/profile null checks precede every dereference below.
    if program.program_id == 0
        || !unsafe { text_is_valid(program.name) }
        || program.submit.is_none()
        || program.max_inflight == 0
        || program.reserved != 0
        || program.profile.is_null()
        || (program.flags & !KNOWN_PROGRAM_FLAGS) != 0
    {
        return Err(DriverLoadError::ProgramAbi(program_index));
    }

    // SAFETY: null-checked above.
    let profile = unsafe { &*program.profile };
    let profile_program_flags =
        program.flags & !SPARK_MODEL_DRIVER_PROGRAM_FLAG_EXTERNAL_COMPLETION;
    if (profile.descriptor_bytes as usize) < std::mem::size_of::<SparkModelDriverProgramProfile>()
        || profile.reserved != 0
        || profile.max_inflight != program.max_inflight
        || profile.profile_flags != profile_program_flags
        || (profile.profile_flags & !KNOWN_PROGRAM_FLAGS) != 0
    {
        return Err(DriverLoadError::ProfileAbi(program_index));
    }

    // SAFETY: name validated non-null above.
    let name = unsafe { cstr_to_string(program.name) };
    if (program.flags & SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_HOST_STAGING) != 0
        && profile.host_staging_bytes_per_submit_ceiling != 0
    {
        return Err(DriverLoadError::HostStagingClaim(name));
    }
    if (program.flags & SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_DEVICE_MEMCPY) != 0
        && profile.device_memcpy_bytes_per_submit_ceiling != 0
    {
        return Err(DriverLoadError::DeviceMemcpyClaim(name));
    }
    if (program.flags & SPARK_MODEL_DRIVER_PROGRAM_FLAG_VALIDATED_LATENCY) != 0
        && profile.validated_latency_ns == 0
    {
        return Err(DriverLoadError::ValidatedLatencyClaim(name));
    }
    if (program.flags & SPARK_MODEL_DRIVER_PROGRAM_FLAG_PRIVATE_QUEUE_PRESSURE) != 0
        && profile.private_queue_count == 0
    {
        return Err(DriverLoadError::PrivateQueueClaim(name));
    }
    Ok(())
}

fn validate_interface(
    interface: &SparkModelDriverInterface,
    expected_target: Option<&str>,
) -> Result<(), DriverLoadError> {
    if interface.abi_version != SPARK_MODEL_DRIVER_ABI_VERSION
        || (interface.interface_bytes as usize) < std::mem::size_of::<SparkModelDriverInterface>()
        || interface.descriptor.is_null()
        || interface.create.is_none()
        || interface.destroy.is_none()
        || interface.admit.is_none()
        || interface.snapshot.is_none()
    {
        return Err(DriverLoadError::InterfaceAbi);
    }

    // SAFETY: null-checked above; descriptor_bytes is checked against the Rust
    // layout (itself pinned to C by spark-abi's layout gate) before we read
    // any field beyond it.
    let descriptor = unsafe { &*interface.descriptor };
    if descriptor.abi_version != SPARK_MODEL_DRIVER_ABI_VERSION
        || (descriptor.descriptor_bytes as usize)
            < std::mem::size_of::<SparkModelDriverDescriptor>()
        || !unsafe { text_is_valid(descriptor.model_id) }
        || !unsafe { text_is_valid(descriptor.model_revision) }
        || !unsafe { text_is_valid(descriptor.stage_name) }
        || !unsafe { text_is_valid(descriptor.target) }
        || !unsafe { sha256_hex_is_valid(descriptor.model_description_sha256) }
        || !unsafe { sha256_hex_is_valid(descriptor.compiled_program_sha256) }
        || descriptor.program_count == 0
        || descriptor.module_instance_count == 0
        || descriptor.programs.is_null()
    {
        return Err(DriverLoadError::DescriptorAbi);
    }

    if let Some(expected) = expected_target {
        if !expected.is_empty() {
            // SAFETY: validated non-null non-empty above.
            let target = unsafe { cstr_to_string(descriptor.target) };
            if target != expected {
                return Err(DriverLoadError::TargetMismatch {
                    driver: target,
                    node: expected.to_string(),
                });
            }
        }
    }

    // SAFETY: `programs` non-null with `program_count` entries per ABI; the C
    // loader walks the same range.
    let programs = unsafe {
        std::slice::from_raw_parts(descriptor.programs, descriptor.program_count as usize)
    };
    for (index, program) in programs.iter().enumerate() {
        validate_program(program, index as u32)?;
        for previous in &programs[..index] {
            // SAFETY: both names validated as C strings by validate_program.
            let (a, b) = unsafe { (CStr::from_ptr(previous.name), CStr::from_ptr(program.name)) };
            if previous.program_id == program.program_id || a == b {
                return Err(DriverLoadError::DuplicateProgram);
            }
        }
    }
    Ok(())
}
