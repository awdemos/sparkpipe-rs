//! Integration test: compile the fixture model driver into a shared object
//! and exercise `ModelDriver::load` — happy path, target matching, and the
//! fail-closed rejection paths. GPU-free, mirrors the C tree's fixture-based
//! module ABI tests.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use spark_sys::{DriverLoadError, ModelDriver};

fn sparkpipe_c_root() -> PathBuf {
    if let Ok(root) = std::env::var("SPARKPIPE_C_ROOT") {
        return PathBuf::from(root);
    }
    // crates/spark-sys -> crates -> sparkpipe-rs -> code -> sparkpipe
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("..").join("sparkpipe")
}

/// Compile `fixtures/fake_model_driver.c` once per test binary run.
fn fixture_driver_path() -> &'static Path {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let c_root = sparkpipe_c_root();
        let include = c_root.join("include");
        assert!(include.is_dir(), "sparkpipe C tree not found at {}", c_root.display());

        let out_dir =
            std::env::var("CARGO_TARGET_TMPDIR").map(PathBuf::from).unwrap_or_else(|_| {
                // crates/spark-sys -> crates -> sparkpipe-rs -> target/...
                // (avoid std::env::temp_dir: the host /tmp tmpfs is per-user
                // quota-limited and cc intermediates must not land there)
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join("..")
                    .join("target")
                    .join("spark-sys-fixtures")
            });
        std::fs::create_dir_all(&out_dir).expect("cannot create fixture output dir");
        let so_path = out_dir.join("fake_model_driver.so");

        let source =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("fake_model_driver.c");
        // cc writes intermediate files to TMPDIR; the host's /tmp tmpfs can be
        // quota-exhausted, so redirect into the (already created) out dir.
        let tmp_dir = out_dir.join("tmp");
        std::fs::create_dir_all(&tmp_dir).expect("cannot create fixture tmp dir");
        let status = Command::new(std::env::var("CC").unwrap_or_else(|_| "cc".into()))
            .env("TMPDIR", &tmp_dir)
            .arg("-std=c11")
            .arg("-O1")
            .arg("-fPIC")
            .arg("-shared")
            .arg("-fvisibility=hidden")
            .arg(format!("-I{}", include.display()))
            .arg(&source)
            .arg("-o")
            .arg(&so_path)
            .status()
            .expect("failed to invoke C compiler for fixture driver");
        assert!(status.success(), "fixture driver failed to compile");
        so_path
    })
}

#[test]
fn loads_valid_fixture_driver() {
    let driver = ModelDriver::load(fixture_driver_path(), None).expect("fixture driver must load");

    let descriptor = driver.descriptor();
    assert_eq!(descriptor.abi_version, spark_abi::SPARK_MODEL_DRIVER_ABI_VERSION);
    assert_eq!(descriptor.program_count, 1);
    assert_eq!(descriptor.module_instance_count, 1);

    let programs = driver.programs();
    assert_eq!(programs.len(), 1);
    assert_eq!(programs[0].program_id, 1);
    assert_eq!(programs[0].name, "fixture_program");
    assert_eq!(programs[0].max_inflight, 1);

    assert!(driver.find_program("fixture_program").is_some());
    assert!(driver.find_program("nonexistent").is_none());

    let debug = format!("{driver:?}");
    assert!(debug.contains("fixture-model"), "debug output: {debug}");
    assert!(debug.contains("host.cpu"), "debug output: {debug}");
}

#[test]
fn matching_target_is_accepted() {
    ModelDriver::load(fixture_driver_path(), Some("host.cpu")).expect("matching target must load");
}

#[test]
fn mismatched_target_is_rejected() {
    let err = ModelDriver::load(fixture_driver_path(), Some("sm_121a.gb10"))
        .expect_err("mismatched target must be rejected");
    match err {
        DriverLoadError::TargetMismatch { driver, node } => {
            assert_eq!(driver, "host.cpu");
            assert_eq!(node, "sm_121a.gb10");
        }
        other => panic!("expected TargetMismatch, got {other:?}"),
    }
}

#[test]
fn missing_file_is_a_load_error() {
    let err = ModelDriver::load(Path::new("/nonexistent/model_driver.so"), None)
        .expect_err("missing file must fail");
    assert!(matches!(err, DriverLoadError::Load { .. }), "got {err:?}");
}

#[test]
fn library_without_interface_symbol_is_rejected() {
    // libc itself: a valid shared object with no SparkModelDriverGetInterface.
    let libc_path = "/usr/lib64/libc.so.6";
    if !Path::new(libc_path).exists() {
        eprintln!("skipping: {libc_path} not present");
        return;
    }
    let err = ModelDriver::load(Path::new(libc_path), None)
        .expect_err("libc must not pass driver validation");
    assert!(matches!(err, DriverLoadError::MissingInterfaceSymbol { .. }), "got {err:?}");
}
