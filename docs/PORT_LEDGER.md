# Port Ledger

Running record of the sparkpipe (C) → sparkpipe-rs (Rust) port. Each entry
notes what moved, what deliberately changed, and what stays in C.

## Phase 0 — Scaffolding & FFI seam (done)

### `spark-abi`
- **Moved:** bindgen generation for `spark_status.h` + `spark_resident_decode_stage.h`
  (pulls in module ABI, KV cache, stage plan, hidden transport, GLM52 model
  geometry headers). 58 structs, 372 constants.
- **Changed:** bindgen cannot evaluate `sizeof(...)` macro constants
  (`*_DESCRIPTOR_BYTES`), so those are replaced by a compile-and-run C layout
  probe (`build.rs`) emitting `layout.rs` with `*_SIZE` / `*_ALIGN` constants,
  pinned by `tests/layout_gate.rs` (57 structs — `SparkHiddenTransportSession`
  is an opaque pimpl handle with no layout to pin).
- **Note:** libclang comes from the PyPI `libclang` wheel
  (`../sparkpipe-rs-tools/.venv`); its missing builtin headers are compensated
  by probing GCC's freestanding include dir in `build.rs`.

### `spark-sys`
- **Moved:** `SparkLoadModelDriver` / `SparkValidateLoadedModelDriverInterface`
  / `SparkFindLoadedModelDriverProgram` (from `src/spark_driver_loader.c`,
  1:1 validation semantics, Rust error enum mirroring SparkStatus codes).
- **Added:** `fixtures/fake_model_driver.c` — a minimal valid driver (same
  shape as `runtime/pack/driver_compiler.c`'s generated stub) compiled at
  test time; GPU-free smoke tests for load/validate/target-match/reject paths.
- **Not yet:** create/admit/snapshot call wrappers (Phase 4, with the node
  runtime); CUDA runtime/driver shims (Phase 4).

### `spark-model`
- **Moved:** contract JSON parsing (replaces `runtime/json.c` for
  `model_contracts/*.json`). Envelope validation + normalized geometry
  extraction across the two contract dialects (nested `model` section vs
  glm52-style flat keys). All 9 C-tree contracts parse in tests.
- **Deferred:** typed per-family section views (kda/mla/moe/...), qualification
  registries (`must_work_targets.json`, hardware question/binding registries)
  — these get typed loaders when a consumer lands (k27, Phase 5).

## Phase 1 — `spark-core` memory & cache machinery (done)

All safe Rust, zero `unsafe` in the crate. 58 crate tests + workspace total 69.

### `arena` (port of `runtime/arena.h`)
- Size-classed slot arena; C's raw pointers became generation-tagged
  `ArenaHandle`s with borrow-checked slice access (`get`/`get_mut` refuse stale
  handles, so readers can't observe a recycled slot's new occupant).
- Guard added: allocation sizes above `isize::MAX` map to `RegionOverflow`
  (the C `-30910` path) instead of panicking in `RawVec`.

### `state_pool` (port of `include/sparkpipe/spark_state_pool.h`)
- Fixed-slot recurrent-state pool, `Option<SlotHandle>` acquire (loud
  exhaustion), IN_USE/NO_SLOT sentinel split preserved → double release is
  `Err(NotInUse)`. No auto-release on Drop: release failure stays visible.

### `kv_arena` (port of `SparkKvCacheArena*` core, `cache/kv_cache.c`)
- All lifecycle entry points ported (acquire/recycle/retain/release/mark
  resident+nonresident/free/resolve/trim/evict/reset). Deferred: capacity
  estimation, JIT stage budget, prefetch plans, async prefetch backend (FFI
  or Phase 4).
- **Latent C bug found and deliberately fixed:** `SparkKvCacheArenaRecycleBlock`
  clears `RESIDENT` without decrementing `resident_block_count` (C `FreeBlock`
  does decrement). Every recycle of a resident block permanently inflates the
  count until `MarkBlockResident` returns `CAPACITY_EXCEEDED` forever. Latent
  in C because the C prefix-cache tests run arena-less; the Rust port always
  owns an arena and hit it immediately. `kv_arena.rs::recycle_block` now
  mirrors `free_block`'s accounting, documented in a doc comment.
  **Candidate upstream fix for the C tree.**

### `prefix_cache` (port of `cache/prefix_cache.c`)
- Full entry-point surface ported including both prefetch-source builders.
  FNV-1a content-hash chain is bit-exact (parity tests against independently
  computed reference values). All statistics counters preserved; the C quirk
  of `reset()` not clearing `reuse_scored_*` counters is test-pinned.
- Shape decisions: cache owns its `KvArena` (C's arena-less mode dropped);
  `Reservation` returns an exactly-sized Vec; mid-trim stalls return
  `Err(CapacityExceeded)` instead of a partial-eviction out-param.

### `batch_sequence_table` (port of `serving/spark_batch_sequence_table.c`)
- Generation-tagged handles (exact C bit layout: 14-bit index/18-bit
  generation), FREE/ACTIVE/AWAITING_TOOL/COMPLETE machine, firing-threshold
  formula. Tests ported from `test_glm52_batch_plane.c` (the file the table
  is actually exercised in — `test_glm52_shared_prefix_admission.c` does not
  cover it).

### `row_allocator` (port of `serving/spark_row_allocator.c`)
- `SparkRowAllocatorAssign` with bit-exact milli fixed point
  (alpha = (ema−1000)·1000/ema clamped 999; per-grant truncation decay;
  strict `>` for deterministic lower-index tie-break). All 8 C scenarios
  ported verbatim + zero-cap early return.

### Environment note
The host's `/tmp` tmpfs has a per-user quota; cc invocations in
`spark-abi/build.rs` (layout probe) and `spark-sys` fixture tests now set
`TMPDIR` into the target dir. If cc fails with "Disk quota exceeded" on this
box, that's why.

## Phase 2 — `spark-sched` scheduling plane (done)

All safe Rust. Tests: stage_plan 6, long_context 5, topology_switch 9,
work_control 19, scheduler 16.

### `stage_plan` (port of `scheduler/stage_plan.c`)
- 14 entry points: validation, table/uniform/balanced builders (exact DP
  minimax incl. UNREACHABLE sentinel + tie-breaking), measured cost profiles
  (B64/B128/B32 tables verbatim), batch bucket selection, chunk shapes.
- Geometry is a by-value config struct everywhere — no GLM52 constants.

### `work_control` (port of `scheduler/work_control.c`)
- 31 public entry points: transaction identity/fingerprinting, packet
  builders/validation, decode/prefill selection, full KV state machine
  (open-addressed directory, clock-sweep acquire, swap hooks as a trait).
  QUEUE_DEPTH = cohort + 1 pinned by const assertion.
- Submodules `work_transaction` + `mtp_tree` ported (the pieces used).

### `topology_switch` (port of `scheduler/topology_switch.c`)
- 15 entry points; NVMe tier / swap device / manifest writer became three
  safe traits (`SwitchTier`, `SwapDevice`, `ManifestWriter`) — plug-in seams
  for the not-yet-ported nvme_tier.

### `scheduler` (port of `scheduler/scheduler.c`)
- All 15 entry points: admission, chunked prefill, prefix-cache
  probe/reserve/commit/cancel choreography (rollback paths exact), CUDA-graph
  padding buckets, adaptive packs, decode bypass, batch decode/prefill.
- PrefixCache owned by value in config; decisions own Vecs (C embeds 64KB
  fixed arrays); rejected decisions are `Ok(decision)` with `accepted: false`
  (C returns OK with a rejected decision too).

### `long_context` (port of `scheduler/long_context.c`)
- All 5 entry points ported (default-policy init, validate, prefill plan,
  decode selection, lane-batch decode selection) with bit-faithful
  normalization, flag, and stride-sampling semantics. The GLM52-derived
  defaults (1M context, 2048 selected tokens) are constructor parameters
  of `LongContextPolicy::new`, per deviation #1 below; a 0 in
  `max_context_tokens` / `selected_token_capacity` is now rejected by
  validation instead of normalized to a baked-in default.
- ABI/descriptor fields dropped; `policy_mode` is an enum. Bounded
  selection larger than the scan budget returns `Err(CapacityExceeded)`
  without the partial plan the C filled before erroring. Lane-batch
  returns `Vec<DecodePlan>` and rejects an undersized index buffer with
  `InvalidArgument` (C would index out of bounds).
- Tests: the 4 `test_glm52_long_context.c` scenarios verbatim + a
  lane-batch test (untouched by the C suite) covering strided lane
  layout, full-vs-bounded per lane, and the undersized-buffer rejection.

## Phase 3 — `spark-text` text plane (done)

Tests: tokenizer 17, chat_template 17, prompt_pipeline 16.

### `tokenizer` (port of `text/tokenizer.c`)
- Structural port: open-addressing hash symbol tables (FNV-1a, chained
  buckets, load factor 2), 16K-slot linear-probed piece cache, merge min-heap
  ordered (rank, left_symbol_index) with generation counters — no HashMap.
- Encode/decode token-for-token parity incl. a transcription bug the port
  caught via batch parity tests (byte-class table indices 90/122).
- Thread-safety: immutable `Sync` Tokenizer + `&mut Workspace` (the C shares
  const tokenizer across pthread workers — no mutex needed).
- HF JSON loading via serde_json (replaces runtime/json.c usage).


### `chat_template` (port of `text/chat_template.c`)
- All 7 entry points ported (writer init/append/begin/begin_message/
  end_message/finish/render_simple) with bit-faithful flag semantics,
  literal fragments, and the reasoning-effort normalization
  (`"high"`/`"High"` → `High`, everything else incl. absent → `Max`).
- Writer owns a `String` with an exact byte capacity (the C reserved one
  byte of its caller buffer for NUL); role is a real enum, so the C
  "unknown role" rejection is unrepresentable.

### `prompt_pipeline` (port of `text/prompt_pipeline.c` + `text/prompt.c`)
- All 4 entry points ported (run-stats init → `Default`, `SparkPromptPipelineRun`
  → `run`, default submit request → `Default`,
  `SparkGlm52RequestApiSubmitTextPrompt` → `submit_text_prompt`). Stats are
  delivered via a `&mut RunStats` out-param on every exit path, as in C
  (Busy on drained/not-accepted dispatch, CapacityExceeded on exhausted
  step budget, cancel-on-callback-failure ordering preserved).
- Two C subsystem couplings became trait boundaries (both subsystems are
  ported elsewhere/in flight): `RequestApi` (the `api/request.c` calls the
  pipeline makes — schedule/describe/copy/build-table/complete/cancel/
  submit) and `TextPromptTokenizer` (the tokenizer encode calls in
  `prompt.c`; the C's optional caller workspace folds into the tokenizer
  implementation). `RequestDispatch` is the minimal surface the pipeline
  reads (`accepted`, `kind`); the real request-API port widens it behind
  the same trait.
- Host staging buffers are borrowed mutable slices (length = capacity); the
  C's function-pointer+context callbacks are `FnMut` closures.
- Tests: the `test_glm52_prompt_pipeline.c` scenario verbatim via a fake
  `RequestApi` (chunked prefill 97 = 64+33 then decode, zero-filled lane
  padding, stats/callback assertions), plus error-path coverage (busy,
  stride/lane capacity, callback failure → cancel, complete failure, step
  budget, invalid configs) and `submit_text_prompt` coverage with a stub
  tokenizer (overflow required-count, empty encoding, flag validation,
  encode/submit failure propagation).

## Phase 3 — `spark-serve` serving plane (done)

Tests: request_api (7 lib + exercised by the 19-test serving_engine suite),
serving_engine 19 integration, http_gateway 14 integration. All 40 green;
workspace `cargo test --workspace`, clippy `-D warnings`, and fmt all clean.

### `request_api` (port of `api/request.c`, the C tree's largest file — 6899 lines)
- Split into `mod.rs` (session/slot lifecycle, validation, stop conditions),
  `slot.rs`, `dispatch.rs` (prefill/speculative-verify/decode batch
  scheduling), `speculation.rs` (MTP dispatch policy with milli-fixed-point
  commit EMA), `prefetch.rs` (JIT KV prefetch planning + async backend seam).
- The C's raw struct sharing becomes internal borrow splitting: `RequestApi`
  owns the `Scheduler` (which owns the `PrefixCache`, which owns the
  `KvArena`); slot pointers become slot indices.
- Three C function pointers + `void *` context → `KvPrefetchBackend` trait.
  The async prefetch *backend* itself (device-copy engine of
  `cache/kv_cache.c`) is NOT ported; the config flag installs a
  caller-provided backend (deferred to Phase 4/FFI, per plan).
- Opaque `SparkRequestModelSpeculator` + the 12-entry `spark_request_model.h`
  linker seam → `speculation::RequestModelSeam` trait;
  `NullSpeculator` ports `serving/spark_request_model_null.c`. The GLM52
  dspark provider is a model module, not ported.
- MTP tree token validation bound (GLM52 output vocab in C) is a
  `Configuration` field (`output_vocab_count`), per deviation #1.
- `SparkKvCacheArena{BuildPrefetchPlan*,MarkPrefetchPlanResident*}` are
  ported in `prefetch.rs` over the arena's public API rather than in
  `spark-core`; the arena's resident block capacity is recovered at init via
  side-effect-free trim probes (no new public getter).
- Rust-specific: slot owns a zero-padded copy of the prompt token ids (C
  borrows caller memory; on the tokenizer-overflow path C's
  `prompt_token_count` exceeds the storage it points at).

### `serving_engine` (port of `api/serving_engine.c` + `api/service.c` + `api/compat_api.c`)
- Submodules: `status` (shared SparkStatus code space), `bridge`
  (`ServingRequestApi` trait — the boundary the C draws as direct calls into
  the concrete request API), `engine` (request records, event ring,
  prefill/decode pump), `service` (clients, request mappings, frame
  protocol), `backend` (node-runtime backend seam as a trait; the dlopen
  interface loader stays in `spark-sys`), `compat` (OpenAI/Anthropic JSON
  surface).
- The submit_text encode-failure path is a proper partial-result error type
  (`SubmitFailure`) instead of the C's error-plus-partial-out-params.

### `http_gateway` (port of `api/http_gateway.c`; replaces `api/gateway/http_server.c`)
- Wire-compatible: request parsing, routing, JSON shapes, SSE framing
  ported faithfully. The C's raw poll-loop server is replaced by hyper +
  tokio (first async runtime in the workspace, per plan).
- Documented upgrades (not wire regressions): keep-alive connections
  replace `Connection: close`; SSE uses HTTP/1.1 chunked transfer-encoding
  instead of close-delimited bodies; the 504 reason phrase is hyper's
  canonical "Gateway Timeout".
- 14 integration tests over real loopback sockets (SSE stream flow,
  receiver map, Content-Length exactness).

## Phase 4 — `spark-node` node runtime (done)

### `rank_daemon` (port of `node/rank_runtime.c` + `node/rank_daemon.c`, ~5.3k lines)
- 8 submodule files: `daemon.rs` (~103 KB state machine), `net.rs` (OS seam),
  `resident_ipc.rs` (CUDA-resident IPC client), `ring_runtime.rs` (rank plan
  builder), `shape.rs` (TP/shape derivation), `status.rs`, `wire.rs` (work
  packet + acknowledgement + final event codecs), `work_ledger.rs`
  (`runtime/work_transaction.c` ledger port).
- 51 tests green, including **all 14 C tests ported verbatim** (same names,
  snake_cased, using the C helper packet builders).
- One documented one-word change to `spark-sched`:
  `WorkControlPacket::canonical_bytes` made `pub` so the daemon wire codec
  reuses the C-exact serialization.
- Incomplete by design: production `main()`/poll-loop wiring (signal handling,
  listener accept paths, transport dlopen bring-up, `PrintReady`) — deferred
  to the full-stack milestone.

### `backend` (port of `node/backend.c`, 4885 lines)
- 8 submodule files: `state.rs` (config + `BackendCore` + seams),
  `adapt.rs` (request-API → `ServingRequestApi` adapter with full-dispatch
  identity tracking across the engine DTO boundary), `pending.rs`
  (pending-decode pipeline + early-final-event ring), `resident.rs`
  (CUDA-resident IPC client), `work_output.rs` (work queue + socket forwarder),
  `wire.rs` (CUDA-resident + final-event codecs), `net.rs` (Unix socket
  helpers), and `backend.rs` (integration root implementing `ServiceBackend`).
- `ServiceBackend` trait implemented: `capability_flags`, `get_view`, `pump`,
  `poll_descriptors`, `service`. Pump mirrors `node/backend.c`'s drain order
  (resident responses → work output → sequence releases → runner progress →
  early final events → service engine pump).
- GLM52 constants are all `BackendConfig` fields; `BackendConfig::reference`
  documents the C reference values for tests.
- Compute callbacks (`prefill_function`, `decode_function`,
  `release_sequence_function`) are currently stubs returning `Busy`/no-op;
  they are wired into the engine and will drive the rank0 builder and
  CUDA-resident seams in the full-stack milestone.
- 3 backend tests: config validation, init + view, empty poll descriptors.

### GPU-free full-stack milestone
- Wired the engine compute callbacks (`prefill_function`, `decode_function`,
  `release_sequence_function`) through `Rc<RefCell<BackendCore>>` closures.
- `RingServiceBackend::new` now accepts an optional `Box<dyn Rank0NodeContext>`
  compute seam; `rank0_runtime_ready` reflects whether one was supplied.
- Rank-0 synchronous path implemented: prefill calls the builder with a no-op
  idle pump; decode delegates to the builder and the engine completes the
  dispatch. Resident async path is stubbed (`Busy`) — final-event completion
  is deferred to GB10 hardware integration.
- Added `full_stack_token_generation_with_mock_rank0`: registers a client,
  submits token ids, pumps the backend, and asserts a `Token` event with the
  mock-computed token id (`42`) is produced. This is the first end-to-end
  request → schedule → dispatch → decode → event flow in the Rust core.

### Phase 4 gate status
- `cargo test --workspace` green (spark-node 55, spark-serve 40, spark-sched
  55, spark-core 58, spark-text 50, spark-model 6, spark-abi 1, spark-sys 5).
- `cargo clippy --workspace --all-targets -- -D warnings` clean.
- `cargo fmt --all -- --check` clean.

## Phase 5 — model families (in progress)

### `k27_authoritative.json` contract (authored, C tree `model_contracts/`)
- Authored from HF `moonshotai/Kimi-K2.7-Code` config.json (observed
  2026-08-03). The checkpoint is a KimiK25 multimodal wrapper; the contract
  covers the text backbone (DeepSeek-V3-family `kimi_k2`) only: 61 layers,
  hidden 7168, MLA (q-lora 1536, kv-lora 512, nope 128 / rope 64, 64 heads,
  no output gate), YaRN (theta 50000, factor 64, original 4096 → 262144),
  MoE 384 routed top-8 + 1 shared (sigmoid, noaux_tc, scaling 2.827,
  renormalized), first layer dense.
- Parameter counts derived from config geometry: 1,026,242,052,096 total /
  32,695,320,576 active — consistent with the marketed 1T / 32B.
- Quantization: int4 group-32 symmetric pack-quantized routed expert linears
  only (compressed-tensors); attention/shared/dense-MLP/lm_head bf16; bf16
  MLA latent KV cache. No native MTP (num_nextn_predict_layers = 0).
- Parse + geometry pinned by `spark-model` test `k27_authoritative_geometry`.
  Weight download deferred pending disk check (~0.5–1TB).

## Known deliberate deviations from the C tree

1. **No GLM52 constants in Rust code.** The C tree leaks `SPARK_GLM52_MODEL_*`
   into generic headers; Rust geometry/capacity config will be
   model-contract-driven from the start.
2. **Dependency policy:** serde/serde_json, thiserror, libloading, libc,
   bindgen (build-time), plus hyper/tokio/bytes/http-body-util/hyper-util
   for the gateway only (decided in Phase 3). Everything else needs
   justification here first.
3. **Index-based arenas** will replace pointer-heavy intrusive lists in
   `spark-core` (generational handles instead of raw pointers), preserving the
   C ownership semantics (single refcount, sentinels for double-free).
