# sparkpipe-rs

Rust core for the [sparkpipe](../sparkpipe) LLM serving engine (rings of up to
16× NVIDIA DGX Spark GB10 nodes, sm_121a). This is a **fork-in-progress**:
the orchestration plane moves to Rust; the CUDA kernels, per-model drivers,
and RDMA transport stay in C/CUDA behind a narrow FFI seam.

## What moves to Rust / what stays in C

| Subsystem | Status | Crate |
|---|---|---|
| Stage/driver ABI bindings (`#[repr(C)]`, bindgen + C layout gate) | ✅ Phase 0 | `spark-abi` |
| Model driver dlopen + ABI validation (port of `spark_driver_loader.c`) | ✅ Phase 0 | `spark-sys` |
| Model contract parsing (serde; replaces `runtime/json.c` usage) | ✅ Phase 0 | `spark-model` |
| Arenas, KV cache, prefix cache, state pool | ⬜ Phase 1 | `spark-core` |
| Scheduler, work control, stage plan | ⬜ Phase 2 | `spark-sched` |
| BPE tokenizer, chat template | ⬜ Phase 3 | `spark-text` |
| HTTP gateway + SSE, request lifecycle | ⬜ Phase 3 | `spark-serve` |
| Rank daemon, backend pump | ⬜ Phase 4 | `spark-node` |
| CUDA kernels (`inference/kernels/*.cuh`), per-model drivers, RDMA transport | **stays C** | linked via `spark-sys` |

Design rules:

- **`unsafe` lives only in `spark-abi` and `spark-sys`.** Everything else is
  safe Rust against safe wrapper types.
- **The C tree is read-only.** `SPARKPIPE_C_ROOT` (default `../sparkpipe`)
  points at it; bindgen and the layout probe read its headers, tests compile
  fixtures against them, nothing writes to it.
- **Fail-closed gates**, mirroring the C tree: `spark-abi`'s layout gate
  compiles and runs a C probe asserting `sizeof`/`_Alignof` parity for every
  ABI struct; driver loading runs the exact validation the C loader runs.
- **GPU-free CI is a first-class mode** (the C tree's cuda_stub philosophy).

## Build & test

```sh
# one-time: libclang for bindgen (PyPI wheel works fine)
uv venv ../sparkpipe-rs-tools/.venv && uv pip install --python ../sparkpipe-rs-tools/.venv/bin/python libclang

cargo build --workspace
cargo test --workspace
```

Environment variables: `SPARKPIPE_C_ROOT` (C tree location),
`LIBCLANG_PATH` (override libclang discovery), `CC` (layout probe compiler).

## Reference model

Kimi K2.7 Code (`moonshotai/Kimi-K2.7-Code`) — 1T MoE / 32B active, 61 layers,
MLA, 384 experts top-8, 256K context. Contract to be authored in Phase 5;
weights pulled from HuggingFace (disk check first: ~0.5–1 TB checkpoint).

See `docs/PORT_LEDGER.md` for the running record of what moved, what stayed,
and deliberate behavior changes.
