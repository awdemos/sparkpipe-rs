//! Build script: generate Rust bindings for the sparkpipe C ABI headers.
//!
//! The C source tree is located via `SPARKPIPE_C_ROOT` (default: `../sparkpipe`
//! relative to the workspace root, i.e. the sibling repo). The C tree is read
//! strictly read-only.
//!
//! libclang is located via `LIBCLANG_PATH`; if unset we probe known locations,
//! including the user-local PyPI `libclang` wheel in `../sparkpipe-rs-tools`.

use std::env;
use std::path::{Path, PathBuf};

fn sparkpipe_c_root() -> PathBuf {
    if let Ok(root) = env::var("SPARKPIPE_C_ROOT") {
        return PathBuf::from(root);
    }
    // crates/spark-abi -> crates -> sparkpipe-rs -> code -> sparkpipe
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    manifest.join("..").join("..").join("..").join("sparkpipe")
}

fn ensure_libclang() {
    if env::var("LIBCLANG_PATH").is_ok() {
        return;
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let candidates = [
        // PyPI libclang wheel in the sibling tools venv.
        manifest.join("../../../sparkpipe-rs-tools/.venv/lib"),
        // Common system locations.
        PathBuf::from("/usr/lib64/llvm22/lib"),
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/usr/lib/llvm-18/lib"),
        PathBuf::from("/usr/lib"),
    ];
    for candidate in &candidates {
        let found = find_libclang(candidate);
        if let Some(dir) = found {
            env::set_var("LIBCLANG_PATH", &dir);
            return;
        }
    }
    panic!(
        "libclang not found; set LIBCLANG_PATH to a directory containing libclang.so \
         (hint: `uv pip install libclang` into ../sparkpipe-rs-tools/.venv)"
    );
}

fn find_libclang(root: &Path) -> Option<PathBuf> {
    if !root.is_dir() {
        return None;
    }
    // Direct hit.
    for name in ["libclang.so", "libclang.so.1", "libclang.so.18.1.1"] {
        if root.join(name).exists() {
            return Some(root.to_path_buf());
        }
    }
    // Shallow recursive search (site-packages layouts nest it).
    for entry in walkdir_shallow(root, 5) {
        if entry.file_name().is_some_and(|name| name.to_string_lossy().starts_with("libclang.so")) {
            return entry.parent().map(|p| p.to_path_buf());
        }
    }
    None
}

fn walkdir_shallow(root: &Path, max_depth: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        if depth > max_depth {
            continue;
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push((path, depth + 1));
                } else {
                    out.push(path);
                }
            }
        }
    }
    out
}

fn main() {
    ensure_libclang();

    let c_root = sparkpipe_c_root();
    let include = c_root.join("include");
    let glm52_include = c_root.join("model-families/glm52/include");
    assert!(
        include.is_dir(),
        "sparkpipe C tree not found at {} (set SPARKPIPE_C_ROOT)",
        c_root.display()
    );

    println!("cargo:rerun-if-env-changed=SPARKPIPE_C_ROOT");
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");
    for header_dir in [&include, &glm52_include] {
        println!("cargo:rerun-if-changed={}", header_dir.display());
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // The stage ABI header pulls in module ABI, KV cache, stage plan, hidden
    // transport, and the GLM52 model geometry headers. Status is standalone.
    let headers = [
        include.join("sparkpipe/spark_status.h"),
        include.join("sparkpipe/spark_resident_decode_stage.h"),
    ];

    // The PyPI libclang wheel does not ship clang's builtin freestanding
    // headers (stddef.h, stdarg.h, ...). GCC's freestanding include dir
    // provides compatible ones; probe it dynamically.
    let mut gcc_include_args = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/usr/lib/gcc") {
        for entry in entries.flatten() {
            if let Ok(versions) = std::fs::read_dir(entry.path()) {
                for version in versions.flatten() {
                    let inc = version.path().join("include");
                    if inc.join("stddef.h").exists() {
                        gcc_include_args.push(format!("-I{}", inc.display()));
                    }
                }
            }
        }
    }

    let mut builder = bindgen::Builder::default()
        .clang_arg(format!("-I{}", include.display()))
        .clang_arg(format!("-I{}", glm52_include.display()))
        .clang_args(&gcc_include_args)
        .clang_arg("-std=c11")
        .use_core()
        .ctypes_prefix("libc")
        .allowlist_type("Spark.*")
        .allowlist_function("Spark.*")
        .allowlist_var("SPARK.*")
        .layout_tests(true)
        .derive_default(true)
        .merge_extern_blocks(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    for header in &headers {
        builder = builder.header(header.display().to_string());
    }

    let bindings =
        builder.generate().unwrap_or_else(|e| panic!("bindgen failed on sparkpipe headers: {e}"));

    bindings.write_to_file(out_dir.join("bindings.rs")).expect("failed to write bindings.rs");

    emit_layout_gate(&include, &glm52_include, &out_dir);
}

/// Structs whose C layout is pinned by the gate below. Keep this list in sync
/// with the ABI surface `spark-sys` consumes; add new ABI structs here when
/// they start crossing the FFI boundary.
const GATED_STRUCTS: &[&str] = &[
    "SparkModelDriverBuffer",
    "SparkModelDriverResidencyToken",
    "SparkModelDriverCompletion",
    "SparkModelDriverFrame",
    "SparkModelDriverProgramProfile",
    "SparkModelDriverAdmissionRequest",
    "SparkModelDriverAdmissionDecision",
    "SparkModelDriverRuntimeSnapshot",
    "SparkModelDriverCreateRequest",
    "SparkModelDriverProgramDescriptor",
    "SparkModelDriverDescriptor",
    "SparkModelDriverInterface",
    "SparkFirmwareModuleConfiguration",
    "SparkFirmwareModuleHostServices",
    "SparkKvCacheCapacityRequest",
    "SparkKvCacheCapacityEstimate",
    "SparkKvJitStageBudgetRequest",
    "SparkKvJitStageBudget",
    "SparkKvCachePrefetchSourceBlock",
    "SparkKvCachePrefetchBlock",
    "SparkKvCachePrefetchPlan",
    "SparkKvCachePrefetchBackendSourceEntry",
    "SparkKvCacheAsyncPrefetchBackendConfiguration",
    "SparkKvCacheAsyncPrefetchRequest",
    "SparkKvCacheAsyncPrefetchBackend",
    "SparkKvCacheBlock",
    "SparkKvCacheConfiguration",
    "SparkKvCacheBlockView",
    "SparkKvBlockTableView",
    "SparkKvCacheArena",
    "SparkStagePlanGeometry",
    "SparkStagePlanStage",
    "SparkStagePlan",
    // Opaque pimpl handle (forward-declared only); sizeof is meaningless.
    // "SparkHiddenTransportSession",
    "SparkHiddenTransportEndpoint",
    "SparkHiddenTransportPacket",
    "SparkHiddenTransportCompletion",
    "SparkHiddenTransportCompletionQueue",
    "SparkHiddenTransportPersistentRingStatistics",
    "SparkHiddenTransportPollDescriptor",
    "SparkHiddenTransportInterface",
    "SparkHiddenTransportDynamicLibrary",
    "SparkResidentDecodeStagePrefillFrameView",
    "SparkResidentDecodeStageFrameContext",
    "SparkResidentDecodeStageQuantizedLinearView",
    "SparkResidentDecodeStageLinearPlan",
    "SparkResidentDecodeStageFp8KvCachePlan",
    "SparkResidentDecodeStageRestrictedLogitsPlan",
    "SparkResidentDecodeStageMtpDraftPlan",
    "SparkResidentDecodeStageFullStagePlan",
    "SparkResidentDecodeStageStageSlicePlan",
    "SparkResidentDecodeStageExactStageSlicePlan",
    "SparkResidentDecodeStagePagedPrefillPlan",
    "SparkResidentDecodeStageBulkPrefillPlan",
    "SparkResidentDecodeStageCudaPipelineSlotState",
    "SparkResidentDecodeStagePipelineSlot",
    "SparkResidentDecodeStageNodeContext",
    "SparkResidentDecodeStageSliceNodeContext",
];

/// Compile and run a tiny C probe that prints `sizeof`/`_Alignof` for every
/// gated struct as Rust constants. This replaces the `*_DESCRIPTOR_BYTES`
/// macros bindgen cannot evaluate and gives `spark-abi` a fail-closed
/// C-vs-Rust layout test (mirroring the C tree's gate philosophy).
fn emit_layout_gate(include: &Path, glm52_include: &Path, out_dir: &Path) {
    let mut probe = String::from(
        "#include <stdio.h>\n\
         #include <stddef.h>\n\
         #include \"sparkpipe/spark_status.h\"\n\
         #include \"sparkpipe/spark_resident_decode_stage.h\"\n\
         int main(void) {\n",
    );
    for name in GATED_STRUCTS {
        probe.push_str(&format!(
            "    printf(\"pub const {name}_SIZE: usize = %zu;\\n\", sizeof({name}));\n\
             \x20   printf(\"pub const {name}_ALIGN: usize = %zu;\\n\", _Alignof({name}));\n",
        ));
    }
    probe.push_str("    return 0;\n}\n");

    let probe_c = out_dir.join("layout_probe.c");
    let probe_bin = out_dir.join("layout_probe");
    std::fs::write(&probe_c, &probe).expect("failed to write layout probe");

    let compiler = env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let status = std::process::Command::new(&compiler)
        // Keep cc's intermediates off the (possibly quota-limited) /tmp tmpfs.
        .env("TMPDIR", out_dir)
        .arg("-std=c11")
        .arg(format!("-I{}", include.display()))
        .arg(format!("-I{}", glm52_include.display()))
        .arg(&probe_c)
        .arg("-o")
        .arg(&probe_bin)
        .status()
        .expect("failed to invoke C compiler for layout probe");
    assert!(status.success(), "layout probe failed to compile");

    let output =
        std::process::Command::new(&probe_bin).output().expect("failed to run layout probe");
    assert!(output.status.success(), "layout probe exited with {:?}", output.status);
    let stdout = String::from_utf8(output.stdout).expect("layout probe emitted non-UTF8");
    std::fs::write(out_dir.join("layout.rs"), stdout).expect("failed to write layout.rs");
}
