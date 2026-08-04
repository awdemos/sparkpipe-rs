//! CUDA-resident IPC protocol (port of `serving/spark_cuda_resident_ipc.c`
//! and `include/sparkpipe/spark_cuda_resident_ipc.h`, the pieces the rank
//! daemon uses).
//!
//! The messages cross a Unix socket as raw C struct bytes; the codecs here
//! serialize field-by-field in little-endian order at the C struct offsets
//! (sizes pinned against a compile-and-run probe of the C headers:
//! header 32, hello 32, stats 592, submit-result 600, completion 264,
//! submit-work prefix 8).

use spark_sched::work_control::{WorkControlConfig, WorkControlPacket};

use super::status::{Result, SparkStatus};
use super::wire;

/// C: `SPARK_CUDA_RESIDENT_IPC_ABI_VERSION`.
pub const IPC_ABI_VERSION: u32 = 24;
/// C: `SPARK_CUDA_RESIDENT_IPC_MAGIC`.
pub const IPC_MAGIC: u32 = 0x5244_5543;

pub const IPC_HEADER_BYTES: u32 = 32;
pub const IPC_HELLO_BYTES: u32 = 32;
pub const IPC_STATS_BYTES: u32 = 592;
pub const IPC_SUBMIT_RESULT_BYTES: u32 = 600;
pub const IPC_COMPLETION_BYTES: u32 = 264;
pub const IPC_SUBMIT_WORK_PREFIX_BYTES: u32 = 8;

pub const IPC_KIND_HELLO: u32 = 1;
pub const IPC_KIND_HELLO_ACK: u32 = 2;
pub const IPC_KIND_SUBMIT_WORK: u32 = 3;
pub const IPC_KIND_SUBMIT_RESULT: u32 = 4;
pub const IPC_KIND_COMPLETION: u32 = 5;
pub const IPC_KIND_QUERY: u32 = 6;
pub const IPC_KIND_STATS: u32 = 7;
pub const IPC_KIND_SHUTDOWN: u32 = 8;
pub const IPC_KIND_ERROR: u32 = 9;
pub const IPC_KIND_SUBMIT_PREFILL: u32 = 10;
pub const IPC_KIND_SUBMIT_DECODE: u32 = 11;

/// C: `SPARK_CUDA_RESIDENT_IPC_SUBMIT_WORK_FLAG_EXPECT_RESULT`.
pub const SUBMIT_WORK_FLAG_EXPECT_RESULT: u32 = 0x0000_0001;
/// C: `SPARK_CUDA_RESIDENT_IPC_SUBMIT_WORK_KNOWN_FLAGS`.
pub const SUBMIT_WORK_KNOWN_FLAGS: u32 = SUBMIT_WORK_FLAG_EXPECT_RESULT;

pub const STATE_EMPTY: u32 = 0;
pub const STATE_LOADING: u32 = 1;
pub const STATE_READY: u32 = 2;
pub const STATE_DRAINING: u32 = 3;
pub const STATE_FAILED: u32 = 4;

/// C: `SPARK_CUDA_RESIDENT_IPC_COMPLETION_FLAG_DSPARK_DRAFT`.
pub const COMPLETION_FLAG_DSPARK_DRAFT: u32 = 0x0000_0001;
/// C: `SPARK_CUDA_RESIDENT_IPC_COMPLETION_KNOWN_FLAGS`.
pub const COMPLETION_KNOWN_FLAGS: u32 = COMPLETION_FLAG_DSPARK_DRAFT;

/// C: `SPARK_CUDA_RESIDENT_IPC_ERROR_TEXT_BYTES`.
pub const ERROR_TEXT_BYTES: usize = 160;

/// C: `SPARK_RING_NODE_CONTEXT_BUILDER_NVME_MODE_*` (protocol values the
/// resident contract validation accepts; the builder itself is not ported).
pub const NVME_MODE_DISABLED: u32 = 0;
pub const NVME_MODE_SYNCHRONOUS_FULL_HISTORY: u32 = 1;
pub const NVME_MODE_BATCHED_COHORT_JIT: u32 = 2;
pub const NVME_MODE_ASYNC_SELECTED_JIT: u32 = 3;

/// C: `SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY` (driver ABI).
pub const COMPLETION_TOKEN_CAPACITY: usize = 8;
/// C: `SPARK_MODEL_DRIVER_COMPLETION_DRAFT_TOKEN_CAPACITY` (driver ABI).
pub const COMPLETION_DRAFT_TOKEN_CAPACITY: usize = 8;
/// C: `SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS`.
pub const DRIVER_COMPLETION_FLAG_TOKEN_IDS: u32 = 0x0000_0001;
/// C: `SPARK_MODEL_DRIVER_COMPLETION_FLAG_DRAFT_TOKEN_IDS`.
pub const DRIVER_COMPLETION_FLAG_DRAFT_TOKEN_IDS: u32 = 0x0000_0002;

/// C: `SPARK_REQUEST_MODEL_ABI_VERSION` (`SPARK_GLM52_DSPARK_ABI_VERSION`
/// aliases it).
pub const DRAFT_ABI_VERSION: u32 = 2;
/// C: `SPARK_GLM52_DSPARK_DRAFT_RESULT_DESCRIPTOR_BYTES`.
pub const DRAFT_RESULT_BYTES: u32 = 72;
/// C: `SPARK_REQUEST_MODEL_MAX_SPECULATIVE_TOKENS` (request-model ABI).
pub const DRAFT_MAX_TOKENS: usize = 7;

/// C: `SparkGlm52DsparkDraftResult` (aliased from
/// `SparkRequestModelDraftResult`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsparkDraftResult {
    pub abi_version: u32,
    pub descriptor_bytes: u32,
    pub flags: u32,
    pub token_count: u32,
    pub confidence_milli: [u32; DRAFT_MAX_TOKENS],
    pub token_ids: [u32; DRAFT_MAX_TOKENS],
}

impl Default for DsparkDraftResult {
    fn default() -> Self {
        Self {
            abi_version: 0,
            descriptor_bytes: 0,
            flags: 0,
            token_count: 0,
            confidence_milli: [0; DRAFT_MAX_TOKENS],
            token_ids: [0; DRAFT_MAX_TOKENS],
        }
    }
}

impl DsparkDraftResult {
    /// C daemon check: draft carries the expected ABI tag.
    pub fn has_valid_descriptor(&self) -> bool {
        self.abi_version == DRAFT_ABI_VERSION && self.descriptor_bytes == DRAFT_RESULT_BYTES
    }

    fn encode(&self, out: &mut Vec<u8>) {
        let start = out.len();
        push_u32(out, self.abi_version);
        push_u32(out, self.descriptor_bytes);
        push_u32(out, self.flags);
        push_u32(out, self.token_count);
        for value in self.confidence_milli {
            push_u32(out, value);
        }
        for value in self.token_ids {
            push_u32(out, value);
        }
        debug_assert_eq!(out.len() - start, DRAFT_RESULT_BYTES as usize);
    }

    fn decode(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), DRAFT_RESULT_BYTES as usize);
        let mut result = DsparkDraftResult::default();
        let mut cursor = Reader::new(bytes);
        result.abi_version = cursor.u32();
        result.descriptor_bytes = cursor.u32();
        result.flags = cursor.u32();
        result.token_count = cursor.u32();
        for slot in result.confidence_milli.iter_mut() {
            *slot = cursor.u32();
        }
        for slot in result.token_ids.iter_mut() {
            *slot = cursor.u32();
        }
        result
    }
}

/// C: `SparkModelDriverResidencyToken`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResidencyToken {
    pub word0: u64,
    pub word1: u64,
    pub generation: u64,
    pub owner: u64,
}

/// C: `SparkModelDriverCompletion` (184 bytes on the wire).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DriverCompletion {
    pub request_id: u64,
    pub sequence_id: u64,
    pub sequence_position: u64,
    pub program_id: u32,
    pub driver_dispatch_slot: u32,
    pub accepted_token_count: u32,
    pub completion_flags: u32,
    pub token_count: u32,
    pub token_ids: [u32; COMPLETION_TOKEN_CAPACITY],
    pub draft_token_count: u32,
    pub draft_token_ids: [u32; COMPLETION_DRAFT_TOKEN_CAPACITY],
    /// C: `SparkStatus status` (carried as the raw code).
    pub status: u32,
    pub residency: ResidencyToken,
    pub queue_delay_ns: u64,
    pub service_time_ns: u64,
    pub device_memcpy_bytes: u64,
    pub host_staging_bytes: u64,
}

pub const DRIVER_COMPLETION_BYTES: usize = 184;

impl DriverCompletion {
    fn encode(&self, out: &mut Vec<u8>) {
        push_u64(out, self.request_id);
        push_u64(out, self.sequence_id);
        push_u64(out, self.sequence_position);
        push_u32(out, self.program_id);
        push_u32(out, self.driver_dispatch_slot);
        push_u32(out, self.accepted_token_count);
        push_u32(out, self.completion_flags);
        push_u32(out, self.token_count);
        for value in self.token_ids {
            push_u32(out, value);
        }
        push_u32(out, self.draft_token_count);
        for value in self.draft_token_ids {
            push_u32(out, value);
        }
        push_u32(out, self.status);
        push_u32(out, 0); // C tail padding before the 8-aligned residency token
        push_u64(out, self.residency.word0);
        push_u64(out, self.residency.word1);
        push_u64(out, self.residency.generation);
        push_u64(out, self.residency.owner);
        push_u64(out, self.queue_delay_ns);
        push_u64(out, self.service_time_ns);
        push_u64(out, self.device_memcpy_bytes);
        push_u64(out, self.host_staging_bytes);
    }

    fn decode(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), DRIVER_COMPLETION_BYTES);
        let mut completion = DriverCompletion::default();
        let mut cursor = Reader::new(bytes);
        completion.request_id = cursor.u64();
        completion.sequence_id = cursor.u64();
        completion.sequence_position = cursor.u64();
        completion.program_id = cursor.u32();
        completion.driver_dispatch_slot = cursor.u32();
        completion.accepted_token_count = cursor.u32();
        completion.completion_flags = cursor.u32();
        completion.token_count = cursor.u32();
        for slot in completion.token_ids.iter_mut() {
            *slot = cursor.u32();
        }
        completion.draft_token_count = cursor.u32();
        for slot in completion.draft_token_ids.iter_mut() {
            *slot = cursor.u32();
        }
        completion.status = cursor.u32();
        cursor.skip(4); // C tail padding
        completion.residency.word0 = cursor.u64();
        completion.residency.word1 = cursor.u64();
        completion.residency.generation = cursor.u64();
        completion.residency.owner = cursor.u64();
        completion.queue_delay_ns = cursor.u64();
        completion.service_time_ns = cursor.u64();
        completion.device_memcpy_bytes = cursor.u64();
        completion.host_staging_bytes = cursor.u64();
        completion
    }
}

/// C: `SparkCudaResidentIpcHeader`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IpcHeader {
    pub magic: u32,
    pub abi_version: u32,
    pub descriptor_bytes: u32,
    pub kind: u32,
    pub payload_bytes: u32,
    pub rank_index: u32,
    pub sequence_number: u64,
}

/// C: `SparkCudaResidentIpcInitializeHeader`.
pub fn initialize_header(
    kind: u32,
    rank_index: u32,
    sequence_number: u64,
    payload_bytes: u32,
) -> IpcHeader {
    IpcHeader {
        magic: IPC_MAGIC,
        abi_version: IPC_ABI_VERSION,
        descriptor_bytes: IPC_HEADER_BYTES,
        kind,
        payload_bytes,
        rank_index,
        sequence_number,
    }
}

/// C: `SparkCudaResidentIpcValidateHeader` (expected kind 0 means any).
pub fn validate_header(
    header: &IpcHeader,
    expected_kind: u32,
    maximum_payload_bytes: u32,
) -> Result<()> {
    if header.magic != IPC_MAGIC {
        return Err(SparkStatus::ParseError);
    }
    if header.abi_version != IPC_ABI_VERSION || header.descriptor_bytes != IPC_HEADER_BYTES {
        return Err(SparkStatus::AbiMismatch);
    }
    if expected_kind != 0 && header.kind != expected_kind {
        return Err(SparkStatus::SchemaError);
    }
    if header.payload_bytes > maximum_payload_bytes {
        return Err(SparkStatus::CapacityExceeded);
    }
    Ok(())
}

pub fn encode_header(header: &IpcHeader) -> [u8; IPC_HEADER_BYTES as usize] {
    let mut out = Vec::with_capacity(IPC_HEADER_BYTES as usize);
    push_u32(&mut out, header.magic);
    push_u32(&mut out, header.abi_version);
    push_u32(&mut out, header.descriptor_bytes);
    push_u32(&mut out, header.kind);
    push_u32(&mut out, header.payload_bytes);
    push_u32(&mut out, header.rank_index);
    push_u64(&mut out, header.sequence_number);
    out.try_into().expect("header layout is exactly 32 bytes")
}

pub fn decode_header(bytes: &[u8; IPC_HEADER_BYTES as usize]) -> IpcHeader {
    let mut cursor = Reader::new(bytes);
    IpcHeader {
        magic: cursor.u32(),
        abi_version: cursor.u32(),
        descriptor_bytes: cursor.u32(),
        kind: cursor.u32(),
        payload_bytes: cursor.u32(),
        rank_index: cursor.u32(),
        sequence_number: cursor.u64(),
    }
}

/// C: `SparkCudaResidentIpcHello`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IpcHello {
    pub descriptor_bytes: u32,
    pub rank_index: u32,
    pub rank_count: u32,
    pub expected_cuda_generation: u32,
    pub control_generation: u64,
    pub process_id: u64,
}

pub fn encode_hello(hello: &IpcHello) -> [u8; IPC_HELLO_BYTES as usize] {
    let mut out = Vec::with_capacity(IPC_HELLO_BYTES as usize);
    push_u32(&mut out, hello.descriptor_bytes);
    push_u32(&mut out, hello.rank_index);
    push_u32(&mut out, hello.rank_count);
    push_u32(&mut out, hello.expected_cuda_generation);
    push_u64(&mut out, hello.control_generation);
    push_u64(&mut out, hello.process_id);
    out.try_into().expect("hello layout is exactly 32 bytes")
}

/// C: `SparkCudaResidentIpcStats` (592 bytes on the wire).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IpcStats {
    pub descriptor_bytes: u32,
    pub state: u32,
    pub capability_flags: u32,
    pub rank_index: u32,
    pub max_active_sequence_count: u32,
    pub active_submission_count: u32,
    pub available_dispatch_slot_count: u32,
    pub private_queue_pressure: u32,
    pub submitted_count: u64,
    pub completed_count: u64,
    pub rejected_count: u64,
    pub resident_sequence_count: u64,
    pub resident_token_count: u64,
    pub kv_nvme_enabled: u32,
    pub kv_physical_block_capacity: u32,
    pub kv_logical_block_capacity: u32,
    pub kv_nvme_mode: u32,
    pub kv_logical_block_count: u64,
    pub kv_resident_block_count: u64,
    pub kv_swapped_block_count: u64,
    pub kv_nvme_record_bytes: u64,
    pub kv_nvme_store_count: u64,
    pub kv_nvme_load_count: u64,
    pub kv_nvme_write_bytes: u64,
    pub kv_nvme_read_bytes: u64,
    pub kv_nvme_synchronous_wait_count: u64,
    pub kv_nvme_batch_flush_count: u64,
    pub kv_nvme_maximum_batch_operation_count: u64,
    pub kv_resident_bytes_per_token: u64,
    pub kv_resident_pool_bytes: u64,
    pub kv_nvme_capacity_bytes: u64,
    pub kv_compact_selected_mla_working_set_bytes: u64,
    pub kv_nvme_batch_block_capacity: u32,
    pub kv_nvme_pending_store_count: u32,
    pub kv_nvme_pending_load_count: u32,
    pub kv_nvme_clean_evict_count: u32,
    pub work_queue_depth: u32,
    pub work_queue_capacity: u32,
    pub builder_pending_work: u32,
    pub resident_driver_inflight: u32,
    pub work_queue_accepted_count: u64,
    pub work_queue_submit_count: u64,
    pub work_queue_error_count: u64,
    pub asynchronous_submit_count: u64,
    pub asynchronous_completion_count: u64,
    pub asynchronous_failure_count: u64,
    pub logical_lane_capacity: u32,
    pub execution_row_capacity: u32,
    pub last_layer_major_logical_lane_count: u32,
    pub last_layer_major_rows_per_lane: u32,
    pub last_layer_major_execution_row_count: u32,
    pub moe_backend_kind: u32,
    pub moe_bound_layer_count: u32,
    pub moe_expected_layer_count: u32,
    pub fp8_scaled_gemm_bound_plan_count: u32,
    pub fp8_scaled_gemm_expected_plan_count: u32,
    pub model_quantization_mode: u32,
    pub layer_major_submit_count: u64,
    pub layer_major_completion_count: u64,
    pub layer_major_failure_count: u64,
    pub cuda_total_bytes: u64,
    pub cuda_initial_free_bytes: u64,
    pub cuda_current_free_bytes: u64,
    pub cuda_consumed_bytes: u64,
    pub cuda_builder_allocation_bytes: u64,
    pub cuda_largest_allocation_bytes: u64,
    pub host_mapped_allocation_bytes: u64,
    pub cuda_generation: u64,
    pub control_generation: u64,
    pub blocker: [u8; ERROR_TEXT_BYTES],
}

impl Default for IpcStats {
    fn default() -> Self {
        // `[u8; 160]` predates `Default` for large arrays; spell it out.
        Self {
            descriptor_bytes: 0,
            state: 0,
            capability_flags: 0,
            rank_index: 0,
            max_active_sequence_count: 0,
            active_submission_count: 0,
            available_dispatch_slot_count: 0,
            private_queue_pressure: 0,
            submitted_count: 0,
            completed_count: 0,
            rejected_count: 0,
            resident_sequence_count: 0,
            resident_token_count: 0,
            kv_nvme_enabled: 0,
            kv_physical_block_capacity: 0,
            kv_logical_block_capacity: 0,
            kv_nvme_mode: 0,
            kv_logical_block_count: 0,
            kv_resident_block_count: 0,
            kv_swapped_block_count: 0,
            kv_nvme_record_bytes: 0,
            kv_nvme_store_count: 0,
            kv_nvme_load_count: 0,
            kv_nvme_write_bytes: 0,
            kv_nvme_read_bytes: 0,
            kv_nvme_synchronous_wait_count: 0,
            kv_nvme_batch_flush_count: 0,
            kv_nvme_maximum_batch_operation_count: 0,
            kv_resident_bytes_per_token: 0,
            kv_resident_pool_bytes: 0,
            kv_nvme_capacity_bytes: 0,
            kv_compact_selected_mla_working_set_bytes: 0,
            kv_nvme_batch_block_capacity: 0,
            kv_nvme_pending_store_count: 0,
            kv_nvme_pending_load_count: 0,
            kv_nvme_clean_evict_count: 0,
            work_queue_depth: 0,
            work_queue_capacity: 0,
            builder_pending_work: 0,
            resident_driver_inflight: 0,
            work_queue_accepted_count: 0,
            work_queue_submit_count: 0,
            work_queue_error_count: 0,
            asynchronous_submit_count: 0,
            asynchronous_completion_count: 0,
            asynchronous_failure_count: 0,
            logical_lane_capacity: 0,
            execution_row_capacity: 0,
            last_layer_major_logical_lane_count: 0,
            last_layer_major_rows_per_lane: 0,
            last_layer_major_execution_row_count: 0,
            moe_backend_kind: 0,
            moe_bound_layer_count: 0,
            moe_expected_layer_count: 0,
            fp8_scaled_gemm_bound_plan_count: 0,
            fp8_scaled_gemm_expected_plan_count: 0,
            model_quantization_mode: 0,
            layer_major_submit_count: 0,
            layer_major_completion_count: 0,
            layer_major_failure_count: 0,
            cuda_total_bytes: 0,
            cuda_initial_free_bytes: 0,
            cuda_current_free_bytes: 0,
            cuda_consumed_bytes: 0,
            cuda_builder_allocation_bytes: 0,
            cuda_largest_allocation_bytes: 0,
            host_mapped_allocation_bytes: 0,
            cuda_generation: 0,
            control_generation: 0,
            blocker: [0; ERROR_TEXT_BYTES],
        }
    }
}

pub fn encode_stats(stats: &IpcStats) -> [u8; IPC_STATS_BYTES as usize] {
    let mut out = Vec::with_capacity(IPC_STATS_BYTES as usize);
    push_u32(&mut out, stats.descriptor_bytes);
    push_u32(&mut out, stats.state);
    push_u32(&mut out, stats.capability_flags);
    push_u32(&mut out, stats.rank_index);
    push_u32(&mut out, stats.max_active_sequence_count);
    push_u32(&mut out, stats.active_submission_count);
    push_u32(&mut out, stats.available_dispatch_slot_count);
    push_u32(&mut out, stats.private_queue_pressure);
    push_u64(&mut out, stats.submitted_count);
    push_u64(&mut out, stats.completed_count);
    push_u64(&mut out, stats.rejected_count);
    push_u64(&mut out, stats.resident_sequence_count);
    push_u64(&mut out, stats.resident_token_count);
    push_u32(&mut out, stats.kv_nvme_enabled);
    push_u32(&mut out, stats.kv_physical_block_capacity);
    push_u32(&mut out, stats.kv_logical_block_capacity);
    push_u32(&mut out, stats.kv_nvme_mode);
    push_u64(&mut out, stats.kv_logical_block_count);
    push_u64(&mut out, stats.kv_resident_block_count);
    push_u64(&mut out, stats.kv_swapped_block_count);
    push_u64(&mut out, stats.kv_nvme_record_bytes);
    push_u64(&mut out, stats.kv_nvme_store_count);
    push_u64(&mut out, stats.kv_nvme_load_count);
    push_u64(&mut out, stats.kv_nvme_write_bytes);
    push_u64(&mut out, stats.kv_nvme_read_bytes);
    push_u64(&mut out, stats.kv_nvme_synchronous_wait_count);
    push_u64(&mut out, stats.kv_nvme_batch_flush_count);
    push_u64(&mut out, stats.kv_nvme_maximum_batch_operation_count);
    push_u64(&mut out, stats.kv_resident_bytes_per_token);
    push_u64(&mut out, stats.kv_resident_pool_bytes);
    push_u64(&mut out, stats.kv_nvme_capacity_bytes);
    push_u64(&mut out, stats.kv_compact_selected_mla_working_set_bytes);
    push_u32(&mut out, stats.kv_nvme_batch_block_capacity);
    push_u32(&mut out, stats.kv_nvme_pending_store_count);
    push_u32(&mut out, stats.kv_nvme_pending_load_count);
    push_u32(&mut out, stats.kv_nvme_clean_evict_count);
    push_u32(&mut out, stats.work_queue_depth);
    push_u32(&mut out, stats.work_queue_capacity);
    push_u32(&mut out, stats.builder_pending_work);
    push_u32(&mut out, stats.resident_driver_inflight);
    push_u64(&mut out, stats.work_queue_accepted_count);
    push_u64(&mut out, stats.work_queue_submit_count);
    push_u64(&mut out, stats.work_queue_error_count);
    push_u64(&mut out, stats.asynchronous_submit_count);
    push_u64(&mut out, stats.asynchronous_completion_count);
    push_u64(&mut out, stats.asynchronous_failure_count);
    push_u32(&mut out, stats.logical_lane_capacity);
    push_u32(&mut out, stats.execution_row_capacity);
    push_u32(&mut out, stats.last_layer_major_logical_lane_count);
    push_u32(&mut out, stats.last_layer_major_rows_per_lane);
    push_u32(&mut out, stats.last_layer_major_execution_row_count);
    push_u32(&mut out, stats.moe_backend_kind);
    push_u32(&mut out, stats.moe_bound_layer_count);
    push_u32(&mut out, stats.moe_expected_layer_count);
    push_u32(&mut out, stats.fp8_scaled_gemm_bound_plan_count);
    push_u32(&mut out, stats.fp8_scaled_gemm_expected_plan_count);
    push_u32(&mut out, stats.model_quantization_mode);
    push_u32(&mut out, 0); // C struct padding to the 8-aligned u64 group
    push_u64(&mut out, stats.layer_major_submit_count);
    push_u64(&mut out, stats.layer_major_completion_count);
    push_u64(&mut out, stats.layer_major_failure_count);
    push_u64(&mut out, stats.cuda_total_bytes);
    push_u64(&mut out, stats.cuda_initial_free_bytes);
    push_u64(&mut out, stats.cuda_current_free_bytes);
    push_u64(&mut out, stats.cuda_consumed_bytes);
    push_u64(&mut out, stats.cuda_builder_allocation_bytes);
    push_u64(&mut out, stats.cuda_largest_allocation_bytes);
    push_u64(&mut out, stats.host_mapped_allocation_bytes);
    push_u64(&mut out, stats.cuda_generation);
    push_u64(&mut out, stats.control_generation);
    out.extend_from_slice(&stats.blocker);
    out.try_into().expect("stats layout is exactly 592 bytes")
}

pub fn decode_stats(bytes: &[u8; IPC_STATS_BYTES as usize]) -> IpcStats {
    let mut stats = IpcStats::default();
    let mut cursor = Reader::new(bytes);
    stats.descriptor_bytes = cursor.u32();
    stats.state = cursor.u32();
    stats.capability_flags = cursor.u32();
    stats.rank_index = cursor.u32();
    stats.max_active_sequence_count = cursor.u32();
    stats.active_submission_count = cursor.u32();
    stats.available_dispatch_slot_count = cursor.u32();
    stats.private_queue_pressure = cursor.u32();
    stats.submitted_count = cursor.u64();
    stats.completed_count = cursor.u64();
    stats.rejected_count = cursor.u64();
    stats.resident_sequence_count = cursor.u64();
    stats.resident_token_count = cursor.u64();
    stats.kv_nvme_enabled = cursor.u32();
    stats.kv_physical_block_capacity = cursor.u32();
    stats.kv_logical_block_capacity = cursor.u32();
    stats.kv_nvme_mode = cursor.u32();
    stats.kv_logical_block_count = cursor.u64();
    stats.kv_resident_block_count = cursor.u64();
    stats.kv_swapped_block_count = cursor.u64();
    stats.kv_nvme_record_bytes = cursor.u64();
    stats.kv_nvme_store_count = cursor.u64();
    stats.kv_nvme_load_count = cursor.u64();
    stats.kv_nvme_write_bytes = cursor.u64();
    stats.kv_nvme_read_bytes = cursor.u64();
    stats.kv_nvme_synchronous_wait_count = cursor.u64();
    stats.kv_nvme_batch_flush_count = cursor.u64();
    stats.kv_nvme_maximum_batch_operation_count = cursor.u64();
    stats.kv_resident_bytes_per_token = cursor.u64();
    stats.kv_resident_pool_bytes = cursor.u64();
    stats.kv_nvme_capacity_bytes = cursor.u64();
    stats.kv_compact_selected_mla_working_set_bytes = cursor.u64();
    stats.kv_nvme_batch_block_capacity = cursor.u32();
    stats.kv_nvme_pending_store_count = cursor.u32();
    stats.kv_nvme_pending_load_count = cursor.u32();
    stats.kv_nvme_clean_evict_count = cursor.u32();
    stats.work_queue_depth = cursor.u32();
    stats.work_queue_capacity = cursor.u32();
    stats.builder_pending_work = cursor.u32();
    stats.resident_driver_inflight = cursor.u32();
    stats.work_queue_accepted_count = cursor.u64();
    stats.work_queue_submit_count = cursor.u64();
    stats.work_queue_error_count = cursor.u64();
    stats.asynchronous_submit_count = cursor.u64();
    stats.asynchronous_completion_count = cursor.u64();
    stats.asynchronous_failure_count = cursor.u64();
    stats.logical_lane_capacity = cursor.u32();
    stats.execution_row_capacity = cursor.u32();
    stats.last_layer_major_logical_lane_count = cursor.u32();
    stats.last_layer_major_rows_per_lane = cursor.u32();
    stats.last_layer_major_execution_row_count = cursor.u32();
    stats.moe_backend_kind = cursor.u32();
    stats.moe_bound_layer_count = cursor.u32();
    stats.moe_expected_layer_count = cursor.u32();
    stats.fp8_scaled_gemm_bound_plan_count = cursor.u32();
    stats.fp8_scaled_gemm_expected_plan_count = cursor.u32();
    stats.model_quantization_mode = cursor.u32();
    cursor.skip(4); // C struct padding
    stats.layer_major_submit_count = cursor.u64();
    stats.layer_major_completion_count = cursor.u64();
    stats.layer_major_failure_count = cursor.u64();
    stats.cuda_total_bytes = cursor.u64();
    stats.cuda_initial_free_bytes = cursor.u64();
    stats.cuda_current_free_bytes = cursor.u64();
    stats.cuda_consumed_bytes = cursor.u64();
    stats.cuda_builder_allocation_bytes = cursor.u64();
    stats.cuda_largest_allocation_bytes = cursor.u64();
    stats.host_mapped_allocation_bytes = cursor.u64();
    stats.cuda_generation = cursor.u64();
    stats.control_generation = cursor.u64();
    stats.blocker.copy_from_slice(cursor.take(ERROR_TEXT_BYTES));
    stats
}

/// C: `SparkCudaResidentIpcSubmitResult`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IpcSubmitResult {
    pub descriptor_bytes: u32,
    pub status: u32,
    pub stats: IpcStats,
}

pub fn encode_submit_result(result: &IpcSubmitResult) -> [u8; IPC_SUBMIT_RESULT_BYTES as usize] {
    let mut out = Vec::with_capacity(IPC_SUBMIT_RESULT_BYTES as usize);
    push_u32(&mut out, result.descriptor_bytes);
    push_u32(&mut out, result.status);
    out.extend_from_slice(&encode_stats(&result.stats));
    out.try_into().expect("submit-result layout is exactly 600 bytes")
}

pub fn decode_submit_result(bytes: &[u8]) -> Option<IpcSubmitResult> {
    if bytes.len() != IPC_SUBMIT_RESULT_BYTES as usize {
        return None;
    }
    let mut cursor = Reader::new(bytes);
    let descriptor_bytes = cursor.u32();
    let status = cursor.u32();
    let stats_bytes: &[u8; IPC_STATS_BYTES as usize] =
        cursor.take(IPC_STATS_BYTES as usize).try_into().expect("slice length checked above");
    Some(IpcSubmitResult { descriptor_bytes, status, stats: decode_stats(stats_bytes) })
}

/// C: `SparkCudaResidentIpcCompletion`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IpcCompletion {
    pub descriptor_bytes: u32,
    pub flags: u32,
    pub completion: DriverCompletion,
    pub dspark_draft: DsparkDraftResult,
}

pub fn encode_completion(completion: &IpcCompletion) -> [u8; IPC_COMPLETION_BYTES as usize] {
    let mut out = Vec::with_capacity(IPC_COMPLETION_BYTES as usize);
    push_u32(&mut out, completion.descriptor_bytes);
    push_u32(&mut out, completion.flags);
    completion.completion.encode(&mut out);
    completion.dspark_draft.encode(&mut out);
    out.try_into().expect("completion layout is exactly 264 bytes")
}

pub fn decode_completion(bytes: &[u8]) -> Option<IpcCompletion> {
    if bytes.len() != IPC_COMPLETION_BYTES as usize {
        return None;
    }
    let mut cursor = Reader::new(bytes);
    let descriptor_bytes = cursor.u32();
    let flags = cursor.u32();
    let completion = DriverCompletion::decode(cursor.take(DRIVER_COMPLETION_BYTES));
    let dspark_draft = DsparkDraftResult::decode(cursor.take(DRAFT_RESULT_BYTES as usize));
    Some(IpcCompletion { descriptor_bytes, flags, completion, dspark_draft })
}

/// C: `SparkCudaResidentIpcSubmitWork` (prefix + the work packet bytes).
#[derive(Debug, Clone)]
pub struct IpcSubmitWork {
    pub descriptor_bytes: u32,
    pub flags: u32,
    pub work_packet: WorkControlPacket,
}

/// C: `SparkCudaResidentIpcCalculateSubmitWorkBytes` (0 when the packet
/// descriptor is out of range or the message would exceed the full struct).
pub fn calculate_submit_work_bytes(
    config: &WorkControlConfig,
    packet_descriptor_bytes: u32,
) -> u32 {
    let packet_prefix_bytes = config.packet_prefix_bytes();
    let maximum_packet_bytes =
        spark_sched::work_control::calculate_packet_bytes(config, config.max_lane_count);
    if packet_descriptor_bytes < packet_prefix_bytes
        || packet_descriptor_bytes > maximum_packet_bytes
    {
        return 0;
    }
    let message_bytes =
        u64::from(IPC_SUBMIT_WORK_PREFIX_BYTES) + u64::from(packet_descriptor_bytes);
    let maximum_message_bytes =
        u64::from(IPC_SUBMIT_WORK_PREFIX_BYTES) + u64::from(maximum_packet_bytes);
    if message_bytes <= maximum_message_bytes {
        message_bytes as u32
    } else {
        0
    }
}

/// C: `SparkCudaResidentIpcInitializeSubmitWork`.
pub fn initialize_submit_work(
    config: &WorkControlConfig,
    work_packet: &WorkControlPacket,
    flags: u32,
) -> Result<IpcSubmitWork> {
    if flags & !SUBMIT_WORK_KNOWN_FLAGS != 0 {
        return Err(SparkStatus::InvalidArgument);
    }
    let message = IpcSubmitWork { descriptor_bytes: 0, flags, work_packet: work_packet.clone() };
    let message_bytes = calculate_submit_work_bytes(config, work_packet.descriptor_bytes);
    if message_bytes == 0 {
        return Err(SparkStatus::InvalidArgument);
    }
    let mut message = message;
    message.descriptor_bytes = message_bytes;
    validate_submit_work(config, &message, message_bytes)?;
    Ok(message)
}

/// C: `SparkCudaResidentIpcValidateSubmitWork`.
pub fn validate_submit_work(
    config: &WorkControlConfig,
    message: &IpcSubmitWork,
    payload_bytes: u32,
) -> Result<()> {
    let expected_bytes = calculate_submit_work_bytes(config, message.work_packet.descriptor_bytes);
    if expected_bytes == 0 {
        return Err(SparkStatus::InvalidArgument);
    }
    if payload_bytes != expected_bytes || message.descriptor_bytes != expected_bytes {
        return Err(SparkStatus::AbiMismatch);
    }
    if message.flags & !SUBMIT_WORK_KNOWN_FLAGS != 0 {
        return Err(SparkStatus::InvalidArgument);
    }
    if message.work_packet.flags & spark_sched::work_control::FLAG_RELEASE_SEQUENCES != 0
        && message.flags & SUBMIT_WORK_FLAG_EXPECT_RESULT == 0
    {
        return Err(SparkStatus::InvalidArgument);
    }
    Ok(())
}

/// Serialize a submit-work message: prefix (descriptor + flags) then the
/// packet's canonical wire bytes (`[0, descriptor_bytes)`).
pub fn encode_submit_work(config: &WorkControlConfig, message: &IpcSubmitWork) -> Result<Vec<u8>> {
    let packet_bytes = wire::packet_to_wire(config, &message.work_packet)?;
    let mut out = Vec::with_capacity(IPC_SUBMIT_WORK_PREFIX_BYTES as usize + packet_bytes.len());
    push_u32(&mut out, message.descriptor_bytes);
    push_u32(&mut out, message.flags);
    out.extend_from_slice(&packet_bytes);
    debug_assert_eq!(out.len() as u32, message.descriptor_bytes);
    Ok(out)
}

/// C: `SPARK_CUDA_RESIDENT_IPC_MAX_CONTROL_PAYLOAD_BYTES`
/// (`SparkCudaResidentIpcSubmitDecode` header + lanes + block indices),
/// derived from the work-control capacities. GLM52 reference: 67,207,256.
pub fn max_control_payload_bytes(config: &WorkControlConfig) -> usize {
    // Submit-decode header through the transaction fields: 11 u32 + pad +
    // 3 u64 + 4 u32 = 88 bytes, then `max_lane_count` decode lanes of
    // align8(32 + 5*4 + 4 + 2*4 + 4*DRAFT_MAX_TOKENS) = 96 bytes, then
    // `max_lane_count * kv_block_capacity` u32 block indices.
    let header_bytes = 88usize;
    let decode_lane_bytes = 96usize;
    let lane_count = config.max_lane_count as usize;
    header_bytes
        + lane_count * decode_lane_bytes
        + lane_count * config.kv_block_capacity() as usize * 4
}

// ---------------------------------------------------------------------------
// Little-endian cursor helpers
// ---------------------------------------------------------------------------

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize) -> &'a [u8] {
        let slice = &self.bytes[self.offset..self.offset + count];
        self.offset += count;
        slice
    }

    fn skip(&mut self, count: usize) {
        self.offset += count;
    }

    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().expect("reader bounds"))
    }

    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().expect("reader bounds"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_sched::work_control::WorkControlConfig;

    /// GLM52-reference work-control capacities through config.
    pub(crate) fn glm52_work_config() -> WorkControlConfig {
        WorkControlConfig {
            mtp_draft_token_count: 6,
            dspark_max_speculative_token_count: 7,
            maximum_context_tokens: 1_048_576,
            kv_block_tokens: 64,
            max_prefill_tokens_per_packet: 256,
            output_vocab_count: 154_880,
            max_lane_count: 1024,
            max_active_sequence_count: 1024,
            cohort_capacity: 1023,
        }
    }

    #[test]
    fn max_control_payload_matches_c_probe() {
        assert_eq!(max_control_payload_bytes(&glm52_work_config()), 67_207_256);
    }

    #[test]
    fn header_round_trip_and_validate() {
        let header = initialize_header(IPC_KIND_SUBMIT_RESULT, 4, 9, IPC_SUBMIT_RESULT_BYTES);
        let bytes = encode_header(&header);
        let decoded = decode_header(&bytes);
        assert_eq!(decoded, header);
        assert_eq!(validate_header(&decoded, 0, 600), Ok(()));
        assert_eq!(
            validate_header(&decoded, IPC_KIND_COMPLETION, 600),
            Err(SparkStatus::SchemaError)
        );
        assert_eq!(validate_header(&decoded, 0, 100), Err(SparkStatus::CapacityExceeded));
        let mut bad = header;
        bad.magic = 0;
        assert_eq!(validate_header(&bad, 0, 600), Err(SparkStatus::ParseError));
        let mut bad = header;
        bad.abi_version = 1;
        assert_eq!(validate_header(&bad, 0, 600), Err(SparkStatus::AbiMismatch));
    }

    #[test]
    fn stats_round_trip() {
        let mut stats = IpcStats {
            descriptor_bytes: IPC_STATS_BYTES,
            state: STATE_READY,
            logical_lane_capacity: 64,
            execution_row_capacity: 512,
            kv_physical_block_capacity: 100,
            kv_logical_block_capacity: 200,
            model_quantization_mode: 2,
            moe_backend_kind: 1,
            moe_bound_layer_count: 75,
            moe_expected_layer_count: 75,
            ..IpcStats::default()
        };
        stats.blocker[0] = b'x';
        stats.cuda_generation = 42;
        let bytes = encode_stats(&stats);
        assert_eq!(decode_stats(&bytes), stats);
    }

    #[test]
    fn submit_result_round_trip() {
        let result = IpcSubmitResult {
            descriptor_bytes: IPC_SUBMIT_RESULT_BYTES,
            status: 0,
            stats: IpcStats {
                descriptor_bytes: IPC_STATS_BYTES,
                state: STATE_READY,
                ..IpcStats::default()
            },
        };
        let bytes = encode_submit_result(&result);
        assert_eq!(decode_submit_result(&bytes), Some(result));
        assert_eq!(decode_submit_result(&bytes[..100]), None);
    }

    #[test]
    fn completion_round_trip() {
        let completion = IpcCompletion {
            descriptor_bytes: IPC_COMPLETION_BYTES,
            flags: COMPLETION_FLAG_DSPARK_DRAFT,
            completion: DriverCompletion {
                request_id: 7,
                sequence_id: 8,
                sequence_position: 9,
                program_id: 3,
                token_count: 1,
                completion_flags: DRIVER_COMPLETION_FLAG_TOKEN_IDS,
                token_ids: [101, 0, 0, 0, 0, 0, 0, 0],
                status: 0,
                ..DriverCompletion::default()
            },
            dspark_draft: DsparkDraftResult {
                abi_version: DRAFT_ABI_VERSION,
                descriptor_bytes: DRAFT_RESULT_BYTES,
                token_count: 2,
                token_ids: [5, 6, 0, 0, 0, 0, 0],
                confidence_milli: [900, 800, 0, 0, 0, 0, 0],
                ..DsparkDraftResult::default()
            },
        };
        let bytes = encode_completion(&completion);
        assert_eq!(decode_completion(&bytes), Some(completion));
        assert_eq!(decode_completion(&bytes[..64]), None);
    }

    #[test]
    fn hello_layout() {
        let hello = IpcHello {
            descriptor_bytes: IPC_HELLO_BYTES,
            rank_index: 3,
            rank_count: 13,
            expected_cuda_generation: 0,
            control_generation: 0,
            process_id: 4242,
        };
        let bytes = encode_hello(&hello);
        assert_eq!(&bytes[4..8], &3u32.to_le_bytes());
        assert_eq!(&bytes[24..32], &4242u64.to_le_bytes());
    }
}
