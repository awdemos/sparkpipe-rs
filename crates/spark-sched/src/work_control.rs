//! Ring work control plane: per-step batch re-formation.
//!
//! Port of `scheduler/work_control.c` (C tree). Every decode/prefill/release
//! step re-forms its batch into a [`WorkControlPacket`], finalizes a
//! content-addressed transaction identity over the packet bytes, and drives
//! the shared KV block directory ([`KvState`]) that maps `(sequence, logical
//! block)` onto physical blocks with two-tier (GPU/NVMe) residency.
//!
//! Per-step batch re-formation is the core design property: nothing about a
//! cohort's composition survives a step boundary except KV directory state,
//! and the serving pipeline's queue depth invariant
//! (`QUEUE_DEPTH = cohort capacity + 1`, one in-flight slot per cohort lane
//! plus the prefill reserve — see `docs/SERVING_PIPELINE.md` in the C tree)
//! is preserved here as [`queue_depth`], a `const fn` with a compile-time
//! assertion, plus a `debug_assert` in [`WorkControlConfig::validate`].
//!
//! Deliberate deviations from C (see docs/PORT_LEDGER.md):
//! - No GLM52 constants are baked in. Every capacity the C code aliases from
//!   `SPARK_GLM52_*` (vocab, block tokens, context tokens, prefill dispatch
//!   cap, MTP/dSpark speculative caps) arrives via [`WorkControlConfig`].
//! - Owned `Vec`s replace caller-supplied fixed arrays and raw struct
//!   pointers. C capacities are noted on each field.
//! - The swap store/load callbacks become the [`KvSwapHooks`] trait instead
//!   of function pointers plus a `void *` context.
//! - The packet fingerprint hashes a canonical little-endian serialization
//!   laid out at the C struct offsets (so hashes are stable and the hash
//!   domain matches C's `[0, descriptor_bytes)` byte range), but the Rust
//!   port does not share wire format with the C struct itself.

pub mod work_transaction {
    //! Port of the pieces of `runtime/work_transaction.c` /
    //! `include/sparkpipe/spark_work_transaction.h` that the work control
    //! plane uses: transaction phases, identity validation, and the FNV-1a
    //! packet fingerprint. (`spark_distributed_work.h` aliases these same
    //! primitives; the ledger/credit machinery is not used by work control.)

    use super::{Result, WorkControlError};

    pub const ABI_VERSION: u32 = 2;
    pub const ACK_MAGIC: u32 = 0x4B41_5457;
    pub const INVALID_INDEX: u32 = u32::MAX;

    pub const PHASE_PREFILL: u32 = 1;
    pub const PHASE_DECODE: u32 = 2;
    pub const PHASE_VERIFY: u32 = 3;
    pub const PHASE_RELEASE: u32 = 4;
    pub const PHASE_CANCEL: u32 = 5;
    pub const PHASE_KNOWN_MAX: u32 = PHASE_CANCEL;

    pub const FINGERPRINT_OFFSET: u64 = 14695981039346656037;
    pub const FINGERPRINT_PRIME: u64 = 1099511628211;

    /// C: `SparkWorkTransactionPhaseIsValid`.
    pub fn phase_is_valid(phase: u32) -> bool {
        (PHASE_PREFILL..=PHASE_KNOWN_MAX).contains(&phase)
    }

    /// C: `SparkWorkTransactionFingerprintBytes` (FNV-1a over 64 bits; a zero
    /// result is remapped to 1 so 0 can mean "no fingerprint").
    pub fn fingerprint_bytes(data: &[u8]) -> u64 {
        if data.is_empty() {
            return 0;
        }
        let mut hash = FINGERPRINT_OFFSET;
        for &byte in data {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FINGERPRINT_PRIME);
        }
        if hash == 0 {
            1
        } else {
            hash
        }
    }

    /// C: `SparkWorkTransactionIdentity` / `SparkDistributedWorkIdentity`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct Identity {
        pub control_generation: u64,
        pub transaction_id: u64,
        pub dispatch_generation: u64,
        pub request_generation: u64,
        pub step_generation: u64,
        pub step_chunk_index: u32,
        pub step_chunk_count: u32,
        pub transaction_phase: u32,
    }

    /// C: `SparkWorkTransactionValidateIdentity` (exposed in C through
    /// `SparkDistributedWorkIdentityIsValid`).
    pub fn validate_identity(identity: &Identity) -> Result<()> {
        if identity.control_generation == 0
            || identity.transaction_id == 0
            || identity.dispatch_generation == 0
            || identity.request_generation == 0
            || identity.step_generation == 0
            || identity.step_chunk_count == 0
            || identity.step_chunk_index >= identity.step_chunk_count
            || !phase_is_valid(identity.transaction_phase)
        {
            return Err(WorkControlError::InvalidArgument);
        }
        Ok(())
    }
}

pub mod mtp_tree {
    //! Port of the pieces of `include/sparkpipe/spark_mtp_tree.h` that the
    //! work control plane uses: the fixed 5-candidate/6-row draft tree
    //! topology and resolution validation. These are model-topology (not
    //! GLM52-capacity) constants, so they stay as constants.

    /// C: `SPARK_MODEL_MTP_TREE_CANDIDATE_COUNT`.
    pub const CANDIDATE_COUNT: u32 = 5;
    /// C: `SPARK_MODEL_MTP_TREE_VERIFIER_ROW_COUNT`.
    pub const VERIFIER_ROW_COUNT: u32 = 6;
    /// C: `SPARK_MODEL_MTP_TREE_EXECUTION_STEP_COUNT`.
    pub const EXECUTION_STEP_COUNT: u32 = 3;
    /// C: `SPARK_MODEL_MTP_TREE_MAX_COMMITTED_TOKEN_COUNT`.
    pub const MAX_COMMITTED_TOKEN_COUNT: u32 = 4;
    /// C: `SPARK_MODEL_MTP_TREE_CONTEXT_EXTENSION`.
    pub const CONTEXT_EXTENSION: u32 = 3;
    /// C: `SPARK_MODEL_MTP_TREE_RESOLUTION_NONE`.
    pub const RESOLUTION_NONE: u32 = 0;
    /// C: `SPARK_MODEL_MTP_TREE_RESOLUTION_COUNT`.
    pub const RESOLUTION_COUNT: u32 = 6;

    /// C: `SparkMtpTreeNode`.
    #[derive(Debug, Clone, Copy)]
    pub struct Node {
        pub parent_row: u8,
        pub depth: u8,
        pub candidate_index: u8,
        pub child_row_base: u8,
        pub child_count: u8,
    }

    /// C: the static `Nodes` table inside `SparkMtpTreeNodeAt`.
    pub const NODES: [Node; VERIFIER_ROW_COUNT as usize] = [
        Node { parent_row: 0, depth: 0, candidate_index: 0, child_row_base: 1, child_count: 1 },
        Node { parent_row: 0, depth: 1, candidate_index: 0, child_row_base: 2, child_count: 2 },
        Node { parent_row: 1, depth: 2, candidate_index: 1, child_row_base: 4, child_count: 2 },
        Node { parent_row: 1, depth: 2, candidate_index: 2, child_row_base: 0, child_count: 0 },
        Node { parent_row: 2, depth: 3, candidate_index: 3, child_row_base: 0, child_count: 0 },
        Node { parent_row: 2, depth: 3, candidate_index: 4, child_row_base: 0, child_count: 0 },
    ];

    /// C: `SparkMtpTreeNodeAt`.
    pub fn node_at(row_index: u32) -> Option<&'static Node> {
        NODES.get(row_index as usize)
    }

    /// C: `SparkMtpTreeVerifierPositionOffset` / `SparkMtpTreeAcceptedTokenCount`.
    pub fn accepted_token_count(path_id: u32) -> u32 {
        node_at(path_id).map_or(0, |node| u32::from(node.depth))
    }

    /// C: `SparkMtpTreeFallbackRowIndex`.
    pub fn fallback_row_index(path_id: u32) -> u32 {
        if node_at(path_id).is_some() {
            path_id
        } else {
            0
        }
    }

    /// C: `SparkMtpTreeTailCandidateIndex`.
    pub fn tail_candidate_index(path_id: u32) -> u32 {
        node_at(path_id).map_or(0, |node| u32::from(node.candidate_index))
    }

    /// C: `SparkMtpTreeTailParentRowIndex`.
    pub fn tail_parent_row_index(path_id: u32) -> u32 {
        node_at(path_id).map_or(0, |node| u32::from(node.parent_row))
    }

    /// C: `SparkMtpTreeTailBasePositionOffset`.
    pub fn tail_base_position_offset(path_id: u32) -> u32 {
        let accepted = accepted_token_count(path_id);
        if accepted == 0 {
            0
        } else {
            accepted - 1
        }
    }

    /// C: `SparkMtpTreeResolutionIsValid`.
    pub fn resolution_is_valid(
        proposed_token_count: u32,
        accepted_token_count: u32,
        path_id: u32,
    ) -> bool {
        if accepted_token_count > proposed_token_count {
            return false;
        }
        if proposed_token_count == 0 {
            return accepted_token_count == 0 && path_id == RESOLUTION_NONE;
        }
        if proposed_token_count != CANDIDATE_COUNT {
            return path_id == RESOLUTION_NONE;
        }
        if path_id >= VERIFIER_ROW_COUNT {
            return false;
        }
        accepted_token_count == self::accepted_token_count(path_id)
    }

    /// C: `SparkMtpTreeTopologyIsValid`.
    pub fn topology_is_valid() -> bool {
        let root = NODES[0];
        if root.depth != 0 || root.parent_row != 0 {
            return false;
        }
        let mut candidate_seen_mask: u32 = 0;
        let mut max_depth: u32 = 0;
        for row_index in 1..VERIFIER_ROW_COUNT {
            let node = NODES[row_index as usize];
            let parent = NODES[node.parent_row as usize];
            if u32::from(node.parent_row) >= row_index
                || node.depth != parent.depth + 1
                || u32::from(node.candidate_index) >= CANDIDATE_COUNT
                || (candidate_seen_mask & (1 << node.candidate_index)) != 0
            {
                return false;
            }
            candidate_seen_mask |= 1 << node.candidate_index;
            max_depth = max_depth.max(u32::from(node.depth));
        }
        if candidate_seen_mask != (1 << CANDIDATE_COUNT) - 1
            || max_depth != CONTEXT_EXTENSION
            || max_depth + 1 != MAX_COMMITTED_TOKEN_COUNT
        {
            return false;
        }
        for (row_index, node) in NODES.iter().enumerate() {
            if node.child_count == 0 {
                continue;
            }
            if node.child_row_base as usize <= row_index
                || node.child_row_base as usize + node.child_count as usize > NODES.len()
            {
                return false;
            }
            for child_offset in 0..node.child_count {
                let child = NODES[(node.child_row_base + child_offset) as usize];
                if child.parent_row as usize != row_index {
                    return false;
                }
            }
        }
        true
    }
}

/// Status codes mirroring the `SparkStatus` values this module can return.
/// `SPARK_STATUS_OK` is `Ok(())`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum WorkControlError {
    /// C: `SPARK_STATUS_INVALID_ARGUMENT`.
    #[error("invalid argument")]
    InvalidArgument,
    /// C: `SPARK_STATUS_CAPACITY_EXCEEDED`.
    #[error("capacity exceeded")]
    CapacityExceeded,
    /// C: `SPARK_STATUS_NOT_FOUND`.
    #[error("not found")]
    NotFound,
    /// C: `SPARK_STATUS_MODULE_NOT_VALIDATED`.
    #[error("module not validated")]
    ModuleNotValidated,
    /// C: `SPARK_STATUS_VALIDATION_FAILED`.
    #[error("validation failed")]
    ValidationFailed,
    /// C: `SPARK_STATUS_ABI_MISMATCH`.
    #[error("ABI mismatch")]
    AbiMismatch,
    /// C: `SPARK_STATUS_BUSY`.
    #[error("busy")]
    Busy,
    /// C: `SPARK_STATUS_INTERNAL_ERROR`.
    #[error("internal error")]
    InternalError,
}

/// Result alias for this module.
pub type Result<T> = std::result::Result<T, WorkControlError>;

// ---------------------------------------------------------------------------
// Constants (C: spark_ring_work_control.h)
// ---------------------------------------------------------------------------

/// C: `SPARK_RING_WORK_CONTROL_ABI_VERSION`.
pub const ABI_VERSION: u32 = 16;
/// C: `SPARK_RING_WORK_CONTROL_PACKET_MAGIC`.
pub const PACKET_MAGIC: u32 = 0x3557_4350;
/// C: `SPARK_RING_WORK_CONTROL_STANDALONE_GENERATION`.
pub const STANDALONE_GENERATION: u64 = 1;
/// C: `SPARK_RING_WORK_CONTROL_KV_STATE_BYTES` (C `sizeof` of the state
/// struct, kept as ABI bookkeeping only; the Rust state is owned `Vec`s).
pub const KV_STATE_DESCRIPTOR_BYTES: u32 = 256;
/// C: `SPARK_KV_CACHE_ABI_VERSION` (for the block-table view).
pub const KV_CACHE_ABI_VERSION: u32 = 2;
/// C: `SPARK_KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES` (C `sizeof`, bookkeeping).
pub const KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES: u32 = 56;

/// C: `SPARK_RING_WORK_CONTROL_FLAG_PREFILL`.
pub const FLAG_PREFILL: u32 = 0x0000_0001;
/// C: `SPARK_RING_WORK_CONTROL_FLAG_MTP_DRAFT`.
pub const FLAG_MTP_DRAFT: u32 = 0x0000_0002;
/// C: `SPARK_RING_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE`.
pub const FLAG_DSPARK_TAP_CAPTURE: u32 = 0x0000_0004;
/// C: `SPARK_RING_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY`.
pub const FLAG_DSPARK_SPECULATIVE_VERIFY: u32 = 0x0000_0008;
/// C: `SPARK_RING_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY`.
pub const FLAG_MTP_SPECULATIVE_VERIFY: u32 = 0x0000_0010;
/// C: `SPARK_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES`.
pub const FLAG_RELEASE_SEQUENCES: u32 = 0x0000_0020;
/// C: `SPARK_RING_WORK_CONTROL_FLAG_MTP_RESOLVE`.
pub const FLAG_MTP_RESOLVE: u32 = 0x0000_0040;
/// C: `SPARK_RING_WORK_CONTROL_FLAG_MTP_TREE_VERIFY`.
pub const FLAG_MTP_TREE_VERIFY: u32 = 0x0000_0080;
/// C: `SPARK_RING_WORK_CONTROL_KNOWN_FLAGS`.
pub const KNOWN_FLAGS: u32 = FLAG_PREFILL
    | FLAG_MTP_DRAFT
    | FLAG_DSPARK_TAP_CAPTURE
    | FLAG_DSPARK_SPECULATIVE_VERIFY
    | FLAG_MTP_SPECULATIVE_VERIFY
    | FLAG_RELEASE_SEQUENCES
    | FLAG_MTP_RESOLVE
    | FLAG_MTP_TREE_VERIFY;

/// C: `SPARK_RING_WORK_CONTROL_INVALID_REQUEST_SLOT`.
pub const INVALID_REQUEST_SLOT: u32 = u32::MAX;

/// C: `SPARK_RING_KV_ENTRY_MISSING`.
pub const KV_ENTRY_MISSING: u8 = 0;
/// C: `SPARK_RING_KV_ENTRY_IN_FLIGHT`.
pub const KV_ENTRY_IN_FLIGHT: u8 = 1;
/// C: `SPARK_RING_KV_ENTRY_RESIDENT`.
pub const KV_ENTRY_RESIDENT: u8 = 2;
/// C: `SPARK_RING_KV_ENTRY_TRANSIENT`.
pub const KV_ENTRY_TRANSIENT: u8 = 3;

/// C: `SPARK_RING_KV_DIRECTORY_RESIDENCY_GPU`.
pub const KV_DIRECTORY_RESIDENCY_GPU: u8 = 1;
/// C: `SPARK_RING_KV_DIRECTORY_RESIDENCY_NVME`.
pub const KV_DIRECTORY_RESIDENCY_NVME: u8 = 2;
/// C: `SPARK_RING_KV_INVALID_BLOCK_INDEX`.
pub const KV_INVALID_BLOCK_INDEX: u32 = u32::MAX;

// ---------------------------------------------------------------------------
// Request API kinds/flags used by dispatch inputs
// (C: spark_request_api.h; only what work control consumes).
// ---------------------------------------------------------------------------

/// C: `SPARK_REQUEST_API_DISPATCH_KIND_PREFILL`.
pub const REQUEST_DISPATCH_KIND_PREFILL: u32 = 1;
/// C: `SPARK_REQUEST_API_DISPATCH_KIND_DECODE_BATCH`.
pub const REQUEST_DISPATCH_KIND_DECODE_BATCH: u32 = 2;
/// C: `SPARK_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH`.
pub const REQUEST_DISPATCH_KIND_PREFILL_BATCH: u32 = 3;
/// C: `SPARK_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH`.
pub const REQUEST_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH: u32 = 4;

/// C: `SPARK_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE`.
pub const REQUEST_DISPATCH_FLAG_DSPARK_TAP_CAPTURE: u32 = 0x0000_0040;
/// C: `SPARK_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY`.
pub const REQUEST_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY: u32 = 0x0000_0080;
/// C: `SPARK_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT`.
pub const REQUEST_DISPATCH_FLAG_MTP_COMMIT: u32 = 0x0000_0200;
/// C: `SPARK_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY`.
pub const REQUEST_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY: u32 = 0x0000_0400;
/// C: `SPARK_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY`.
pub const REQUEST_DISPATCH_FLAG_MTP_TREE_VERIFY: u32 = 0x0000_0800;

// ---------------------------------------------------------------------------
// Stage-plan batch buckets
// (local port of the `spark_stage_plan.h` pieces work_control.c uses; the
// crate's `stage_plan` module is ported separately).
// ---------------------------------------------------------------------------

/// C: `SPARK_STAGE_PLAN_BUCKET_B16`.
pub const STAGE_PLAN_BUCKET_B16: u32 = 16;
/// C: `SPARK_STAGE_PLAN_BUCKET_B32`.
pub const STAGE_PLAN_BUCKET_B32: u32 = 32;
/// C: `SPARK_STAGE_PLAN_BUCKET_B64`.
pub const STAGE_PLAN_BUCKET_B64: u32 = 64;
/// C: `SPARK_STAGE_PLAN_BUCKET_B128`.
pub const STAGE_PLAN_BUCKET_B128: u32 = 128;
/// C: `SPARK_STAGE_PLAN_BUCKET_B256`.
pub const STAGE_PLAN_BUCKET_B256: u32 = 256;
/// C: `SPARK_STAGE_PLAN_BUCKET_B512`.
pub const STAGE_PLAN_BUCKET_B512: u32 = 512;
/// C: `SPARK_STAGE_PLAN_BUCKET_B1024`.
pub const STAGE_PLAN_BUCKET_B1024: u32 = 1024;

/// C: `SPARK_STAGE_PLAN_BATCH_BUCKETS_INITIALIZER`.
pub const STAGE_PLAN_BATCH_BUCKETS: [u32; 7] = [
    STAGE_PLAN_BUCKET_B16,
    STAGE_PLAN_BUCKET_B32,
    STAGE_PLAN_BUCKET_B64,
    STAGE_PLAN_BUCKET_B128,
    STAGE_PLAN_BUCKET_B256,
    STAGE_PLAN_BUCKET_B512,
    STAGE_PLAN_BUCKET_B1024,
];

/// C: `SparkStagePlanBatchBucketIsSupported`.
pub fn batch_bucket_is_supported(batch_bucket: u32) -> bool {
    STAGE_PLAN_BATCH_BUCKETS.contains(&batch_bucket)
}

/// C: `SparkStagePlanSelectBatchBucket` (count over the maximum bucket maps
/// to `CAPACITY_EXCEEDED`).
pub fn select_batch_bucket(active_sequence_count: u32) -> Result<u32> {
    if active_sequence_count == 0 {
        return Err(WorkControlError::InvalidArgument);
    }
    STAGE_PLAN_BATCH_BUCKETS
        .iter()
        .copied()
        .find(|&bucket| active_sequence_count <= bucket)
        .ok_or(WorkControlError::CapacityExceeded)
}

/// C: `SparkStagePlanExecutionChunkShape`.
fn execution_chunk_shape(
    logical_sequence_count: u32,
    rows_per_sequence: u32,
    execution_row_capacity: u32,
    max_batch_bucket: u32,
) -> Result<(u32, u32)> {
    if logical_sequence_count == 0 || rows_per_sequence == 0 || execution_row_capacity == 0 {
        return Err(WorkControlError::InvalidArgument);
    }
    let execution_row_capacity = execution_row_capacity.min(max_batch_bucket);
    let mut maximum_sequences_per_chunk = execution_row_capacity / rows_per_sequence;
    if maximum_sequences_per_chunk == 0 {
        return Err(WorkControlError::CapacityExceeded);
    }
    if maximum_sequences_per_chunk > logical_sequence_count {
        maximum_sequences_per_chunk = logical_sequence_count;
    }
    let chunk_count = logical_sequence_count.div_ceil(maximum_sequences_per_chunk);
    Ok((maximum_sequences_per_chunk, chunk_count))
}

// ---------------------------------------------------------------------------
// Queue depth invariant (per-step batch re-formation)
// ---------------------------------------------------------------------------

/// `QUEUE_DEPTH = cohort capacity + 1`: one in-flight slot per cohort lane
/// plus the prefill reserve. Per-step batch re-formation — the design
/// property of this module — relies on the reserve so a newcomer can ride
/// the prefill slot while the cohort's step is re-formed.
pub const fn queue_depth(cohort_capacity: u32) -> u32 {
    cohort_capacity + 1
}

// Compile-time assertion that the invariant holds by construction.
const _: () = assert!(queue_depth(1023) == 1024);

// ---------------------------------------------------------------------------
// Configuration (deliberate deviation: capacities from config, not GLM52)
// ---------------------------------------------------------------------------

/// Model/service capacities the C code bakes in as `SPARK_GLM52_*` and
/// `SPARK_RING_WORK_CONTROL_MAX_*` constants. The equivalent GLM52 values
/// from the C tree are noted per field for reference only.
#[derive(Debug, Clone)]
pub struct WorkControlConfig {
    /// C: `SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT` (GLM52: 6).
    pub mtp_draft_token_count: u32,
    /// C: `SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT` (GLM52: 7).
    pub dspark_max_speculative_token_count: u32,
    /// C: `SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS` (GLM52: 1048576).
    pub maximum_context_tokens: u32,
    /// C: `SPARK_GLM52_KV_BLOCK_TOKENS` (GLM52: 64).
    pub kv_block_tokens: u32,
    /// C: `SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH` (GLM52: 256).
    pub max_prefill_tokens_per_packet: u32,
    /// C: `SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT` (GLM52: 154880).
    pub output_vocab_count: u32,
    /// C: `SPARK_RING_WORK_CONTROL_MAX_LANE_COUNT` (1024).
    pub max_lane_count: u32,
    /// C: `SPARK_RING_WORK_CONTROL_MAX_ACTIVE_SEQUENCE_COUNT`, aliased from
    /// `SPARK_STAGE_PLAN_MAX_BATCH_BUCKET` (1024). Also bounds prefill chunk
    /// row counts and packet prefill token capacity.
    pub max_active_sequence_count: u32,
    /// Cohort capacity of the serving pipeline; [`queue_depth`] is derived
    /// from it (`cohort + 1`).
    pub cohort_capacity: u32,
}

impl WorkControlConfig {
    /// C: `SPARK_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT`
    /// (`max(mtp_draft, dspark_max_speculative)`).
    pub fn max_speculative_token_count(&self) -> u32 {
        self.mtp_draft_token_count.max(self.dspark_max_speculative_token_count)
    }

    /// C: `SPARK_RING_WORK_CONTROL_KV_BLOCK_CAPACITY`
    /// (`maximum_context_tokens / kv_block_tokens`).
    pub fn kv_block_capacity(&self) -> u32 {
        self.maximum_context_tokens / self.kv_block_tokens
    }

    /// Validates cross-field consistency and the queue-depth invariant.
    pub fn validate(&self) -> Result<()> {
        if self.mtp_draft_token_count == 0
            || self.dspark_max_speculative_token_count == 0
            || self.maximum_context_tokens == 0
            || self.kv_block_tokens == 0
            || self.max_prefill_tokens_per_packet == 0
            || self.output_vocab_count == 0
            || self.max_lane_count == 0
            || self.max_active_sequence_count == 0
            || self.cohort_capacity == 0
            || self.maximum_context_tokens % self.kv_block_tokens != 0
        {
            return Err(WorkControlError::InvalidArgument);
        }
        // Per-step batch re-formation invariant: the pipeline queue depth is
        // always the cohort plus the prefill reserve.
        debug_assert!(queue_depth(self.cohort_capacity) == self.cohort_capacity + 1);
        Ok(())
    }

    /// C: `SPARK_RING_WORK_CONTROL_LANE_BYTES` — the C lane struct size,
    /// derived from the configured speculative capacity. Used only for
    /// `descriptor_bytes` bookkeeping.
    pub fn lane_bytes(&self) -> u32 {
        // 5 u64 + 5 u32 + u8 + u8 + u16 + spec ids, 8-aligned.
        align8(64 + 4 * self.max_speculative_token_count())
    }

    /// C: `SPARK_RING_WORK_CONTROL_PACKET_PREFIX_BYTES` — C `offsetof(lanes)`.
    pub fn packet_prefix_bytes(&self) -> u32 {
        // 4 u32 + 9 u64 + 15 u32 + spec ids + 4 u32 + prefill ids, 8-aligned.
        align8(164 + 4 * (self.max_speculative_token_count() + self.max_active_sequence_count))
    }

    /// C: `SparkRingWorkControlCalculatePacketBytes`. Returns 0 when the
    /// count is out of range or the size would not fit a u32.
    pub fn calculate_packet_bytes(&self, active_sequence_count: u32) -> u32 {
        if active_sequence_count == 0 || active_sequence_count > self.max_active_sequence_count {
            return 0;
        }
        let packet_bytes = u64::from(self.packet_prefix_bytes())
            + u64::from(active_sequence_count) * u64::from(self.lane_bytes());
        u32::try_from(packet_bytes).unwrap_or(0)
    }
}

const fn align8(value: u32) -> u32 {
    (value + 7) & !7
}

/// C: `SparkRingWorkControlCalculatePacketBytes` (free-function form).
pub fn calculate_packet_bytes(config: &WorkControlConfig, active_sequence_count: u32) -> u32 {
    config.calculate_packet_bytes(active_sequence_count)
}

// ---------------------------------------------------------------------------
// Packet and lane
// ---------------------------------------------------------------------------

/// C: `SparkRingWorkControlLane`.
#[derive(Debug, Clone, Default)]
pub struct WorkControlLane {
    pub request_id: u64,
    pub request_generation: u64,
    pub step_generation: u64,
    pub sequence_id: u64,
    pub sequence_position: u64,
    pub request_slot_index: u32,
    pub context_token_count: u32,
    pub input_token_id: u32,
    pub mtp_draft_token_count: u32,
    pub speculative_token_count: u32,
    pub mtp_resolution_proposed_token_count: u8,
    pub mtp_resolution_accepted_token_count: u8,
    pub mtp_resolution_path_id: u16,
    /// C capacity: `SPARK_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT`
    /// (config: [`WorkControlConfig::max_speculative_token_count`]).
    pub speculative_draft_token_ids: Vec<u32>,
}

/// C: `SparkRingWorkControlPacket`. Fixed arrays become `Vec`s whose lengths
/// are the C capacities (noted per field); `lanes.len() == lane_count`.
#[derive(Debug, Clone)]
pub struct WorkControlPacket {
    pub magic: u32,
    pub abi_version: u32,
    pub descriptor_bytes: u32,
    pub flags: u32,
    pub request_id: u64,
    pub sequence_id: u64,
    pub sequence_position: u64,
    pub deadline_time_ns: u64,
    pub control_generation: u64,
    pub transaction_id: u64,
    pub dispatch_generation: u64,
    pub request_generation: u64,
    pub step_generation: u64,
    pub step_chunk_index: u32,
    pub step_chunk_count: u32,
    pub transaction_phase: u32,
    pub reserved_transaction: u32,
    pub active_sequence_count: u32,
    pub new_token_count: u32,
    pub pipeline_slot: u32,
    pub priority: u32,
    pub block_token_count: u32,
    pub kv_block_table_token_count: u32,
    pub max_blocks_per_sequence: u32,
    pub mtp_draft_token_count: u32,
    pub input_token_id: u32,
    pub speculative_token_count: u32,
    pub speculative_token_index: u32,
    /// C capacity: `SPARK_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT`.
    pub speculative_draft_token_ids: Vec<u32>,
    pub lane_count: u32,
    pub rows_per_lane: u32,
    pub execution_row_count: u32,
    pub execution_batch_bucket: u32,
    /// C capacity: `SPARK_RING_WORK_CONTROL_MAX_ACTIVE_SEQUENCE_COUNT`.
    pub prefill_token_ids: Vec<u32>,
    /// C capacity: `SPARK_RING_WORK_CONTROL_MAX_LANE_COUNT` (1024). The
    /// packet is valid only for `lanes[..lane_count]`.
    pub lanes: Vec<WorkControlLane>,
}

impl WorkControlPacket {
    /// A zeroed packet with all `Vec`s allocated to their C capacities from
    /// `config` (mirrors C's `memset(packet, 0, packet_bytes)`).
    pub fn zeroed(config: &WorkControlConfig) -> Self {
        Self {
            magic: 0,
            abi_version: 0,
            descriptor_bytes: 0,
            flags: 0,
            request_id: 0,
            sequence_id: 0,
            sequence_position: 0,
            deadline_time_ns: 0,
            control_generation: 0,
            transaction_id: 0,
            dispatch_generation: 0,
            request_generation: 0,
            step_generation: 0,
            step_chunk_index: 0,
            step_chunk_count: 0,
            transaction_phase: 0,
            reserved_transaction: 0,
            active_sequence_count: 0,
            new_token_count: 0,
            pipeline_slot: 0,
            priority: 0,
            block_token_count: 0,
            kv_block_table_token_count: 0,
            max_blocks_per_sequence: 0,
            mtp_draft_token_count: 0,
            input_token_id: 0,
            speculative_token_count: 0,
            speculative_token_index: 0,
            speculative_draft_token_ids: vec![0; config.max_speculative_token_count() as usize],
            lane_count: 0,
            rows_per_lane: 0,
            execution_row_count: 0,
            execution_batch_bucket: 0,
            prefill_token_ids: vec![0; config.max_active_sequence_count as usize],
            lanes: Vec::new(),
        }
    }

    /// Sets `lane_count` and resizes `lanes` to match (new lanes zeroed),
    /// keeping the C invariant `lanes.len() == lane_count`.
    pub fn set_lane_count(&mut self, config: &WorkControlConfig, lane_count: u32) {
        self.lane_count = lane_count;
        self.lanes.resize_with(lane_count as usize, || WorkControlLane {
            speculative_draft_token_ids: vec![0; config.max_speculative_token_count() as usize],
            ..WorkControlLane::default()
        });
    }

    /// Canonical little-endian serialization laid out at the C struct
    /// offsets, exactly `descriptor_bytes` long. This is the hash domain of
    /// [`packet_fingerprint`] (C hashes the struct bytes
    /// `[0, descriptor_bytes)`) and the wire bytes the rank daemon
    /// transmits/receives.
    pub fn canonical_bytes(&self, config: &WorkControlConfig) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.descriptor_bytes as usize);
        let push_u32 = |bytes: &mut Vec<u8>, value: u32| {
            bytes.extend_from_slice(&value.to_le_bytes());
        };
        let push_u64 = |bytes: &mut Vec<u8>, value: u64| {
            bytes.extend_from_slice(&value.to_le_bytes());
        };
        push_u32(&mut bytes, self.magic);
        push_u32(&mut bytes, self.abi_version);
        push_u32(&mut bytes, self.descriptor_bytes);
        push_u32(&mut bytes, self.flags);
        push_u64(&mut bytes, self.request_id);
        push_u64(&mut bytes, self.sequence_id);
        push_u64(&mut bytes, self.sequence_position);
        push_u64(&mut bytes, self.deadline_time_ns);
        push_u64(&mut bytes, self.control_generation);
        push_u64(&mut bytes, self.transaction_id);
        push_u64(&mut bytes, self.dispatch_generation);
        push_u64(&mut bytes, self.request_generation);
        push_u64(&mut bytes, self.step_generation);
        push_u32(&mut bytes, self.step_chunk_index);
        push_u32(&mut bytes, self.step_chunk_count);
        push_u32(&mut bytes, self.transaction_phase);
        push_u32(&mut bytes, self.reserved_transaction);
        push_u32(&mut bytes, self.active_sequence_count);
        push_u32(&mut bytes, self.new_token_count);
        push_u32(&mut bytes, self.pipeline_slot);
        push_u32(&mut bytes, self.priority);
        push_u32(&mut bytes, self.block_token_count);
        push_u32(&mut bytes, self.kv_block_table_token_count);
        push_u32(&mut bytes, self.max_blocks_per_sequence);
        push_u32(&mut bytes, self.mtp_draft_token_count);
        push_u32(&mut bytes, self.input_token_id);
        push_u32(&mut bytes, self.speculative_token_count);
        push_u32(&mut bytes, self.speculative_token_index);
        for &token_id in &self.speculative_draft_token_ids {
            push_u32(&mut bytes, token_id);
        }
        push_u32(&mut bytes, self.lane_count);
        push_u32(&mut bytes, self.rows_per_lane);
        push_u32(&mut bytes, self.execution_row_count);
        push_u32(&mut bytes, self.execution_batch_bucket);
        for &token_id in &self.prefill_token_ids {
            push_u32(&mut bytes, token_id);
        }
        // Padding to the 8-aligned `lanes` offset (C struct layout).
        while bytes.len() % 8 != 0 {
            bytes.push(0);
        }
        debug_assert_eq!(bytes.len() as u32, config.packet_prefix_bytes());
        let lane_bytes = config.lane_bytes() as usize;
        for lane_index in 0..self.lane_count as usize {
            let lane_start = bytes.len();
            let lane = &self.lanes[lane_index];
            push_u64(&mut bytes, lane.request_id);
            push_u64(&mut bytes, lane.request_generation);
            push_u64(&mut bytes, lane.step_generation);
            push_u64(&mut bytes, lane.sequence_id);
            push_u64(&mut bytes, lane.sequence_position);
            push_u32(&mut bytes, lane.request_slot_index);
            push_u32(&mut bytes, lane.context_token_count);
            push_u32(&mut bytes, lane.input_token_id);
            push_u32(&mut bytes, lane.mtp_draft_token_count);
            push_u32(&mut bytes, lane.speculative_token_count);
            bytes.push(lane.mtp_resolution_proposed_token_count);
            bytes.push(lane.mtp_resolution_accepted_token_count);
            bytes.extend_from_slice(&lane.mtp_resolution_path_id.to_le_bytes());
            for &token_id in &lane.speculative_draft_token_ids {
                push_u32(&mut bytes, token_id);
            }
            bytes.resize(lane_start + lane_bytes, 0);
        }
        debug_assert_eq!(bytes.len() as u32, self.descriptor_bytes);
        bytes
    }
}

// ---------------------------------------------------------------------------
// Transaction identity (C: work_control.c, top section)
// ---------------------------------------------------------------------------

fn nonzero_identity_word(value: u64, fallback: u64) -> u64 {
    if value == 0 {
        fallback
    } else {
        value
    }
}

/// C: `SparkRingWorkControlTransactionPhase`.
pub fn transaction_phase(packet: &WorkControlPacket) -> u32 {
    if packet.flags & FLAG_RELEASE_SEQUENCES != 0 {
        return work_transaction::PHASE_RELEASE;
    }
    if packet.flags & FLAG_PREFILL != 0 {
        return work_transaction::PHASE_PREFILL;
    }
    if packet.flags & (FLAG_DSPARK_SPECULATIVE_VERIFY | FLAG_MTP_SPECULATIVE_VERIFY) != 0 {
        return work_transaction::PHASE_VERIFY;
    }
    work_transaction::PHASE_DECODE
}

/// C: `SparkRingWorkControlPacketFingerprint`.
pub fn packet_fingerprint(packet: &WorkControlPacket, config: &WorkControlConfig) -> u64 {
    if packet.descriptor_bytes < config.packet_prefix_bytes()
        || packet.descriptor_bytes > calculate_packet_bytes(config, config.max_lane_count)
    {
        return 0;
    }
    work_transaction::fingerprint_bytes(&packet.canonical_bytes(config))
}

/// C: `SparkRingWorkControlSetTransactionIdentity`.
pub fn set_transaction_identity(
    packet: &mut WorkControlPacket,
    config: &WorkControlConfig,
    control_generation: u64,
    transaction_id: u64,
    dispatch_generation: u64,
    step_generation: u64,
) -> Result<()> {
    if control_generation == 0
        || transaction_id == 0
        || dispatch_generation == 0
        || step_generation == 0
        || packet.lane_count == 0
        || packet.lane_count > config.max_lane_count
        || packet.step_chunk_count == 0
        || packet.step_chunk_index >= packet.step_chunk_count
        || !work_transaction::phase_is_valid(packet.transaction_phase)
    {
        return Err(WorkControlError::InvalidArgument);
    }
    for lane in packet.lanes[..packet.lane_count as usize].iter() {
        if lane.request_generation == 0 {
            return Err(WorkControlError::InvalidArgument);
        }
    }
    packet.control_generation = control_generation;
    packet.transaction_id = transaction_id;
    packet.dispatch_generation = dispatch_generation;
    packet.request_generation = packet.lanes[0].request_generation;
    packet.step_generation = step_generation;
    packet.reserved_transaction = 0;
    for lane in packet.lanes[..packet.lane_count as usize].iter_mut() {
        lane.step_generation = step_generation;
    }
    Ok(())
}

/// C: `SparkRingWorkControlFinalizeTransaction`.
pub fn finalize_transaction(
    packet: &mut WorkControlPacket,
    config: &WorkControlConfig,
    control_generation: u64,
    step_chunk_index: u32,
    step_chunk_count: u32,
) -> Result<()> {
    if control_generation == 0
        || step_chunk_count == 0
        || step_chunk_index >= step_chunk_count
        || packet.lane_count == 0
        || packet.lane_count > config.max_lane_count
        || packet.descriptor_bytes != calculate_packet_bytes(config, packet.lane_count)
    {
        return Err(WorkControlError::InvalidArgument);
    }
    for lane in packet.lanes[..packet.lane_count as usize].iter_mut() {
        if lane.request_generation == 0 {
            return Err(WorkControlError::InvalidArgument);
        }
        lane.step_generation = 0;
    }
    packet.control_generation = control_generation;
    packet.transaction_id = 0;
    packet.dispatch_generation = 0;
    packet.request_generation = packet.lanes[0].request_generation;
    packet.step_generation = 0;
    packet.step_chunk_index = step_chunk_index;
    packet.step_chunk_count = step_chunk_count;
    packet.transaction_phase = transaction_phase(packet);
    packet.reserved_transaction = 0;
    if !work_transaction::phase_is_valid(packet.transaction_phase) {
        return Err(WorkControlError::InvalidArgument);
    }
    let canonical_hash = work_transaction::fingerprint_bytes(&packet.canonical_bytes(config));
    if canonical_hash == 0 {
        return Err(WorkControlError::InternalError);
    }
    let dispatch_generation = nonzero_identity_word(canonical_hash ^ 0x9E37_79B9_7F4A_7C15, 1);
    let step_generation =
        nonzero_identity_word(canonical_hash.rotate_left(17) ^ 0xD1B5_4A32_D192_ED03, 1);
    set_transaction_identity(
        packet,
        config,
        control_generation,
        canonical_hash,
        dispatch_generation,
        step_generation,
    )
}

/// C: `SparkRingWorkControlGetTransactionIdentity`.
pub fn get_transaction_identity(packet: &WorkControlPacket) -> Result<work_transaction::Identity> {
    let identity = work_transaction::Identity {
        control_generation: packet.control_generation,
        transaction_id: packet.transaction_id,
        dispatch_generation: packet.dispatch_generation,
        request_generation: packet.request_generation,
        step_generation: packet.step_generation,
        step_chunk_index: packet.step_chunk_index,
        step_chunk_count: packet.step_chunk_count,
        transaction_phase: packet.transaction_phase,
    };
    work_transaction::validate_identity(&identity)?;
    Ok(identity)
}

/// C: `SparkRingWorkControlValidateTransactionFields` (static).
fn validate_transaction_fields(
    packet: &WorkControlPacket,
    config: &WorkControlConfig,
) -> Result<()> {
    if packet.lane_count == 0
        || packet.lane_count > config.max_lane_count
        || packet.descriptor_bytes != calculate_packet_bytes(config, packet.lane_count)
        || packet.request_generation == 0
        || packet.reserved_transaction != 0
    {
        return Err(WorkControlError::InvalidArgument);
    }
    get_transaction_identity(packet)?;
    let expected_phase = transaction_phase(packet);
    if packet.transaction_phase != expected_phase
        || packet.request_generation != packet.lanes[0].request_generation
    {
        return Err(WorkControlError::InvalidArgument);
    }
    for lane in packet.lanes[..packet.lane_count as usize].iter() {
        if lane.request_generation == 0 || lane.step_generation != packet.step_generation {
            return Err(WorkControlError::InvalidArgument);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Dispatch inputs (ports of the C request/serving/prompt-pipeline views,
// trimmed to the fields work control consumes)
// ---------------------------------------------------------------------------

/// C: `SparkRequestApiDispatch` (trimmed). The three decision structs of C
/// are flattened to their `batch_bucket` fields, which is all
/// [`select_execution_batch_bucket`] reads.
#[derive(Debug, Clone, Default)]
pub struct RequestApiDispatch {
    pub kind: u32,
    pub flags: u32,
    pub request_count: u32,
    pub highest_priority: u32,
    /// C capacity: `SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT`.
    pub request_handles: Vec<u64>,
    /// C capacity: `SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT`.
    pub request_ids: Vec<u64>,
    /// C capacity: `SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT`.
    pub sequence_ids: Vec<u64>,
    /// C: `prefill_decision.batch_bucket`.
    pub prefill_decision_batch_bucket: u32,
    /// C: `prefill_batch_decision.batch_bucket`.
    pub prefill_batch_decision_batch_bucket: u32,
    /// C: `decode_batch_decision.batch_bucket`.
    pub decode_batch_decision_batch_bucket: u32,
    pub speculative_verifier_token_count: u32,
    pub mtp_draft_token_budget: u32,
}

/// C: `SparkRequestApiDecodeDispatchLaneView`.
#[derive(Debug, Clone, Default)]
pub struct DecodeDispatchLaneView {
    pub request_index: u32,
    pub sequence_position: u32,
    pub context_token_count: u32,
    pub request_slot_index: u32,
    pub request_id: u64,
    pub sequence_id: u64,
    pub request_handle: u64,
    pub mtp_resolution_base_position: u64,
    pub mtp_resolution_proposed_token_count: u32,
    pub mtp_resolution_accepted_token_count: u32,
    pub mtp_resolution_committed_token_count: u32,
    pub mtp_resolution_path_id: u32,
}

/// C: `SparkRequestApiDecodeDispatchView` (trimmed).
#[derive(Debug, Clone, Default)]
pub struct DecodeDispatchView {
    pub lane_count: u32,
    /// C capacity: `SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT`.
    pub lanes: Vec<DecodeDispatchLaneView>,
}

/// The KV block table view attached to a dispatch. In C this is a borrowed
/// `const SparkKvBlockTableView *`; work control only checks presence
/// (decode) and `lane_count` (prefill).
#[derive(Debug, Clone, Copy, Default)]
pub struct DispatchKvBlockTableView {
    pub block_token_count: u32,
    pub lane_count: u32,
    pub lane_stride: u32,
    pub lane_capacity: u32,
}

/// C: `SparkServingDecodeDispatch` (trimmed).
#[derive(Debug, Clone, Default)]
pub struct ServingDecodeDispatch {
    pub dispatch_kind: u32,
    pub request_count: u32,
    pub active_sequence_count: u32,
    pub request_dispatch: RequestApiDispatch,
    /// C: `kv_block_table_view != 0` (contents unused by work control).
    pub kv_block_table_view: Option<DispatchKvBlockTableView>,
    pub decode_view: DecodeDispatchView,
    /// C capacity: `SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT`.
    pub input_token_ids: Vec<u32>,
    pub speculative_token_count: u32,
    /// C: `speculative_draft_token_ids[request][token]`; per-request rows
    /// have C capacity `SPARK_REQUEST_MODEL_MAX_SPECULATIVE_TOKENS`.
    pub speculative_draft_token_ids: Vec<Vec<u32>>,
}

/// C: `SparkRequestApiPrefillDispatchLaneView` (trimmed).
#[derive(Debug, Clone, Default)]
pub struct PrefillDispatchLaneView {
    pub request_index: u32,
    pub prompt_token_offset: u32,
    pub prompt_token_count: u32,
    pub request_slot_index: u32,
    pub request_id: u64,
    pub sequence_id: u64,
    pub request_handle: u64,
}

/// C: `SparkRequestApiPrefillDispatchView` (trimmed).
#[derive(Debug, Clone, Default)]
pub struct PrefillDispatchView {
    pub lane_count: u32,
    pub prompt_token_count: u32,
    pub prompt_token_stride: u32,
    /// C capacity: `SPARK_REQUEST_API_MAX_DISPATCH_REQUEST_COUNT`.
    pub lanes: Vec<PrefillDispatchLaneView>,
}

/// C: `SparkPromptPipelinePrefillDispatch` (trimmed). `host_token_ids`
/// replaces the raw `const uint32_t *` plus `host_token_stride`.
#[derive(Debug, Clone, Default)]
pub struct PromptPipelinePrefillDispatch {
    pub dispatch_kind: u32,
    pub active_sequence_count: u32,
    pub lane_count: u32,
    pub prompt_token_offset: u32,
    pub prompt_token_count: u32,
    pub prompt_token_stride: u32,
    pub host_token_stride: u32,
    pub request_dispatch: RequestApiDispatch,
    pub prefill_view: PrefillDispatchView,
    /// C: `host_token_ids`, indexed `lane * host_token_stride + token`.
    pub host_token_ids: Vec<u32>,
    /// C: `kv_block_table_view != 0` (only `lane_count` is read).
    pub kv_block_table_view: Option<DispatchKvBlockTableView>,
}

// ---------------------------------------------------------------------------
// Batch bucket / MTP budget selection
// ---------------------------------------------------------------------------

/// C: `SparkRingWorkControlSelectExecutionBatchBucket`.
pub fn select_execution_batch_bucket(
    request_dispatch: &RequestApiDispatch,
    batch_lane_or_row_count: u32,
) -> Result<u32> {
    if batch_lane_or_row_count == 0 {
        return Err(WorkControlError::InvalidArgument);
    }
    let mut batch_bucket = match request_dispatch.kind {
        REQUEST_DISPATCH_KIND_PREFILL => request_dispatch.prefill_decision_batch_bucket,
        REQUEST_DISPATCH_KIND_PREFILL_BATCH => request_dispatch.prefill_batch_decision_batch_bucket,
        REQUEST_DISPATCH_KIND_DECODE_BATCH | REQUEST_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH => {
            request_dispatch.decode_batch_decision_batch_bucket
        }
        _ => return Err(WorkControlError::InvalidArgument),
    };
    if !batch_bucket_is_supported(batch_bucket) {
        return Err(WorkControlError::InvalidArgument);
    }
    if batch_bucket < batch_lane_or_row_count {
        batch_bucket = select_batch_bucket(batch_lane_or_row_count)?;
    }
    Ok(batch_bucket)
}

/// C: `SparkRingWorkControlSelectMtpDraftBudget`.
pub fn select_mtp_draft_budget(
    dispatch_kind: u32,
    request_flags: u32,
    requested_budget: u32,
) -> Result<u32> {
    let producer = dispatch_kind == REQUEST_DISPATCH_KIND_DECODE_BATCH
        && (request_flags & REQUEST_DISPATCH_FLAG_MTP_COMMIT) != 0;
    let tree_verify = dispatch_kind == REQUEST_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH
        && (request_flags
            & (REQUEST_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY
                | REQUEST_DISPATCH_FLAG_MTP_TREE_VERIFY))
            == (REQUEST_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY
                | REQUEST_DISPATCH_FLAG_MTP_TREE_VERIFY);
    if !producer && !tree_verify {
        if requested_budget != 0 {
            return Err(WorkControlError::InvalidArgument);
        }
        return Ok(0);
    }
    if requested_budget != mtp_tree::CANDIDATE_COUNT {
        return Err(WorkControlError::ModuleNotValidated);
    }
    Ok(requested_budget)
}

// ---------------------------------------------------------------------------
// Decode packet construction
// ---------------------------------------------------------------------------

/// C: `SparkRingWorkControlWrittenPositionCount` (static).
fn written_position_count(packet: &WorkControlPacket) -> u32 {
    if packet.flags & FLAG_MTP_TREE_VERIFY != 0 {
        return mtp_tree::CONTEXT_EXTENSION + 1;
    }
    packet.new_token_count
}

/// C: `SparkRingWorkControlSetDecodeFlags` (static).
fn set_decode_flags(
    decode_dispatch: &ServingDecodeDispatch,
    mtp_budget: u32,
    packet: &mut WorkControlPacket,
) {
    if mtp_budget != 0 {
        packet.flags |= FLAG_MTP_DRAFT;
    }
    let request_flags = decode_dispatch.request_dispatch.flags;
    if request_flags & REQUEST_DISPATCH_FLAG_DSPARK_TAP_CAPTURE != 0 {
        packet.flags |= FLAG_DSPARK_TAP_CAPTURE;
    }
    if request_flags & REQUEST_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY != 0 {
        packet.flags |= FLAG_DSPARK_SPECULATIVE_VERIFY;
    }
    if request_flags & REQUEST_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY != 0 {
        packet.flags |= FLAG_MTP_SPECULATIVE_VERIFY;
    }
    if request_flags & REQUEST_DISPATCH_FLAG_MTP_TREE_VERIFY != 0 {
        packet.flags |= FLAG_MTP_TREE_VERIFY;
    }
}

/// C: `SparkRingWorkControlSetMtpResolutionFlag` (static).
fn set_mtp_resolution_flag(packet: &mut WorkControlPacket) {
    for lane in packet.lanes[..packet.lane_count as usize].iter() {
        if lane.mtp_resolution_proposed_token_count != 0 {
            packet.flags |= FLAG_MTP_RESOLVE;
            break;
        }
    }
}

/// C: `SparkRingWorkControlBuildDecodeLanes` (static).
fn build_decode_lanes(
    decode_dispatch: &ServingDecodeDispatch,
    lane_offset: u32,
    lane_count: u32,
    speculative_verify: bool,
    mtp_budget: u32,
    packet: &mut WorkControlPacket,
) -> Result<()> {
    let request_dispatch = &decode_dispatch.request_dispatch;
    let mtp_tree_verify = (request_dispatch.flags & REQUEST_DISPATCH_FLAG_MTP_TREE_VERIFY) != 0;
    for lane_index in 0..lane_count {
        let source_index = (lane_offset + lane_index) as usize;
        let source_lane = decode_dispatch
            .decode_view
            .lanes
            .get(source_index)
            .ok_or(WorkControlError::InvalidArgument)?;
        let request_id = *request_dispatch
            .request_ids
            .get(source_index)
            .ok_or(WorkControlError::InvalidArgument)?;
        let sequence_id = *request_dispatch
            .sequence_ids
            .get(source_index)
            .ok_or(WorkControlError::InvalidArgument)?;
        let request_handle = *request_dispatch
            .request_handles
            .get(source_index)
            .ok_or(WorkControlError::InvalidArgument)?;
        let input_token_id = *decode_dispatch
            .input_token_ids
            .get(source_index)
            .ok_or(WorkControlError::InvalidArgument)?;
        if source_lane.request_index != lane_offset + lane_index
            || source_lane.request_id != request_id
            || source_lane.sequence_id != sequence_id
        {
            return Err(WorkControlError::InvalidArgument);
        }
        if source_lane.request_handle != request_handle {
            return Err(WorkControlError::InvalidArgument);
        }
        let lane = &mut packet.lanes[lane_index as usize];
        lane.request_id = request_id;
        lane.request_generation = request_handle;
        lane.sequence_id = sequence_id;
        lane.sequence_position = u64::from(source_lane.sequence_position);
        lane.request_slot_index = source_lane.request_slot_index;
        lane.context_token_count = source_lane.context_token_count;
        lane.input_token_id = input_token_id;
        lane.mtp_draft_token_count = mtp_budget;
        lane.mtp_resolution_proposed_token_count =
            source_lane.mtp_resolution_proposed_token_count as u8;
        lane.mtp_resolution_accepted_token_count =
            source_lane.mtp_resolution_accepted_token_count as u8;
        lane.mtp_resolution_path_id = source_lane.mtp_resolution_path_id as u16;
        if speculative_verify {
            let context_extension = if mtp_tree_verify {
                mtp_tree::CONTEXT_EXTENSION
            } else {
                decode_dispatch.speculative_token_count
            };
            if lane.context_token_count > u32::MAX - context_extension {
                return Err(WorkControlError::CapacityExceeded);
            }
            lane.context_token_count += context_extension;
            lane.speculative_token_count = decode_dispatch.speculative_token_count;
            let source_tokens = decode_dispatch
                .speculative_draft_token_ids
                .get(source_index)
                .ok_or(WorkControlError::InvalidArgument)?;
            for token_index in 0..lane.speculative_token_count as usize {
                lane.speculative_draft_token_ids[token_index] =
                    *source_tokens.get(token_index).ok_or(WorkControlError::InvalidArgument)?;
            }
        } else if mtp_budget != 0 {
            if lane.context_token_count > u32::MAX - mtp_budget {
                return Err(WorkControlError::CapacityExceeded);
            }
            lane.context_token_count += mtp_budget;
        }
        if lane.context_token_count > packet.kv_block_table_token_count {
            packet.kv_block_table_token_count = lane.context_token_count;
        }
    }
    Ok(())
}

/// C: `SparkRingWorkControlBuildDecodePacketRange`.
pub fn build_decode_packet_range(
    config: &WorkControlConfig,
    decode_dispatch: &ServingDecodeDispatch,
    lane_offset: u32,
    lane_count: u32,
    speculative_token_index: u32,
) -> Result<WorkControlPacket> {
    if decode_dispatch.request_count == 0
        || decode_dispatch.request_count != decode_dispatch.active_sequence_count
        || decode_dispatch.decode_view.lane_count != decode_dispatch.active_sequence_count
        || decode_dispatch.active_sequence_count > config.max_active_sequence_count
        || decode_dispatch.kv_block_table_view.is_none()
        || lane_count == 0
        || lane_offset >= decode_dispatch.active_sequence_count
        || lane_count > decode_dispatch.active_sequence_count - lane_offset
    {
        return Err(WorkControlError::InvalidArgument);
    }
    let speculative_verify =
        decode_dispatch.dispatch_kind == REQUEST_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH;
    let mtp_tree_verify =
        (decode_dispatch.request_dispatch.flags & REQUEST_DISPATCH_FLAG_MTP_TREE_VERIFY) != 0;
    if (speculative_verify
        && (decode_dispatch.speculative_token_count == 0
            || decode_dispatch.speculative_token_count > config.max_speculative_token_count()
            || speculative_token_index != 0))
        || (!speculative_verify && speculative_token_index != 0)
    {
        return Err(WorkControlError::InvalidArgument);
    }
    let mtp_budget = select_mtp_draft_budget(
        decode_dispatch.dispatch_kind,
        decode_dispatch.request_dispatch.flags,
        decode_dispatch.request_dispatch.mtp_draft_token_budget,
    )?;
    if calculate_packet_bytes(config, lane_count) == 0 {
        return Err(WorkControlError::InvalidArgument);
    }
    let mut packet = WorkControlPacket::zeroed(config);
    packet.set_lane_count(config, lane_count);
    packet.magic = PACKET_MAGIC;
    packet.abi_version = ABI_VERSION;
    packet.control_generation = STANDALONE_GENERATION;
    packet.active_sequence_count = lane_count;
    packet.descriptor_bytes = calculate_packet_bytes(config, packet.active_sequence_count);
    packet.rows_per_lane = if speculative_verify {
        if mtp_tree_verify {
            decode_dispatch.request_dispatch.speculative_verifier_token_count
        } else {
            decode_dispatch.speculative_token_count + 1
        }
    } else {
        1
    };
    if mtp_tree_verify
        && (packet.rows_per_lane != mtp_tree::VERIFIER_ROW_COUNT
            || decode_dispatch.speculative_token_count != mtp_tree::CANDIDATE_COUNT)
    {
        return Err(WorkControlError::ModuleNotValidated);
    }
    if u64::from(packet.lane_count) * u64::from(packet.rows_per_lane) > u64::from(u32::MAX) {
        return Err(WorkControlError::CapacityExceeded);
    }
    packet.execution_row_count = packet.lane_count * packet.rows_per_lane;
    packet.execution_batch_bucket = select_execution_batch_bucket(
        &decode_dispatch.request_dispatch,
        packet.execution_row_count,
    )?;
    packet.new_token_count = if speculative_verify { packet.rows_per_lane } else { mtp_budget + 1 };
    packet.priority = decode_dispatch.request_dispatch.highest_priority;
    packet.block_token_count = config.kv_block_tokens;
    packet.max_blocks_per_sequence = config.kv_block_capacity();
    packet.mtp_draft_token_count = mtp_budget;
    set_decode_flags(decode_dispatch, mtp_budget, &mut packet);
    build_decode_lanes(
        decode_dispatch,
        lane_offset,
        lane_count,
        speculative_verify,
        mtp_budget,
        &mut packet,
    )?;
    set_mtp_resolution_flag(&mut packet);
    packet.request_id = packet.lanes[0].request_id;
    packet.sequence_id = packet.lanes[0].sequence_id;
    packet.sequence_position = packet.lanes[0].sequence_position;
    packet.input_token_id = packet.lanes[0].input_token_id;
    if speculative_verify {
        packet.speculative_token_count = decode_dispatch.speculative_token_count;
        packet.speculative_token_index = speculative_token_index;
        let source_tokens = decode_dispatch
            .speculative_draft_token_ids
            .get(lane_offset as usize)
            .ok_or(WorkControlError::InvalidArgument)?;
        for (index, destination) in packet.speculative_draft_token_ids.iter_mut().enumerate() {
            *destination = source_tokens.get(index).copied().unwrap_or(0);
        }
    }
    finalize_transaction(&mut packet, config, STANDALONE_GENERATION, 0, 1)?;
    Ok(packet)
}

/// C: `SparkRingWorkControlBuildDecodePacket`.
pub fn build_decode_packet(
    config: &WorkControlConfig,
    decode_dispatch: &ServingDecodeDispatch,
    speculative_token_index: u32,
) -> Result<WorkControlPacket> {
    build_decode_packet_range(
        config,
        decode_dispatch,
        0,
        decode_dispatch.active_sequence_count,
        speculative_token_index,
    )
}

// ---------------------------------------------------------------------------
// Prefill packet construction
// ---------------------------------------------------------------------------

/// C: `SparkRingWorkControlSelectPrefillChunk`.
pub fn select_prefill_chunk(
    config: &WorkControlConfig,
    prefill_dispatch: &PromptPipelinePrefillDispatch,
    token_offset: u32,
    maximum_execution_row_count: u32,
) -> Result<u32> {
    if prefill_dispatch.lane_count == 0
        || prefill_dispatch.lane_count != prefill_dispatch.active_sequence_count
        || prefill_dispatch.lane_count != prefill_dispatch.prefill_view.lane_count
        || token_offset >= prefill_dispatch.prompt_token_count
        || maximum_execution_row_count == 0
        || maximum_execution_row_count > config.max_active_sequence_count
    {
        return Err(WorkControlError::InvalidArgument);
    }
    let mut active_lane_count = 0u32;
    let mut minimum_remaining_token_count = u32::MAX;
    for lane_index in 0..prefill_dispatch.lane_count as usize {
        let lane = prefill_dispatch
            .prefill_view
            .lanes
            .get(lane_index)
            .ok_or(WorkControlError::InvalidArgument)?;
        if token_offset >= lane.prompt_token_count {
            continue;
        }
        let remaining_token_count = lane.prompt_token_count - token_offset;
        minimum_remaining_token_count = minimum_remaining_token_count.min(remaining_token_count);
        active_lane_count += 1;
    }
    if active_lane_count == 0 || active_lane_count > maximum_execution_row_count {
        return Err(WorkControlError::CapacityExceeded);
    }
    let mut token_count = maximum_execution_row_count / active_lane_count;
    token_count = token_count.min(minimum_remaining_token_count);
    token_count = token_count.min(config.max_prefill_tokens_per_packet);
    token_count = token_count.min(prefill_dispatch.prompt_token_count - token_offset);
    if token_count == 0 {
        return Err(WorkControlError::CapacityExceeded);
    }
    Ok(token_count)
}

/// C: `SparkRingWorkControlBuildPrefillPacket`.
pub fn build_prefill_packet(
    config: &WorkControlConfig,
    prefill_dispatch: &PromptPipelinePrefillDispatch,
    token_offset: u32,
    token_count: u32,
) -> Result<WorkControlPacket> {
    if prefill_dispatch.lane_count == 0
        || prefill_dispatch.lane_count != prefill_dispatch.active_sequence_count
        || prefill_dispatch.lane_count != prefill_dispatch.prefill_view.lane_count
        || prefill_dispatch.kv_block_table_view.is_none()
        || prefill_dispatch
            .kv_block_table_view
            .is_some_and(|view| view.lane_count != prefill_dispatch.lane_count)
        || prefill_dispatch.lane_count > config.max_active_sequence_count
        || prefill_dispatch.host_token_ids.is_empty()
        || prefill_dispatch.host_token_stride == 0
        || token_offset >= prefill_dispatch.prompt_token_count
        || token_count == 0
        || token_count > config.max_prefill_tokens_per_packet
        || token_count > prefill_dispatch.prompt_token_count - token_offset
    {
        return Err(WorkControlError::InvalidArgument);
    }
    // The C code reads `host_token_ids[lane * host_token_stride + token]`
    // from a raw pointer; the owned Vec must provably cover that range.
    if (prefill_dispatch.host_token_ids.len() as u64)
        < u64::from(prefill_dispatch.lane_count) * u64::from(prefill_dispatch.host_token_stride)
    {
        return Err(WorkControlError::InvalidArgument);
    }
    let mut active_lane_count = 0u32;
    for source_lane_index in 0..prefill_dispatch.lane_count as usize {
        let source_lane = prefill_dispatch
            .prefill_view
            .lanes
            .get(source_lane_index)
            .ok_or(WorkControlError::InvalidArgument)?;
        if token_offset < source_lane.prompt_token_count {
            if token_count > source_lane.prompt_token_count - token_offset {
                return Err(WorkControlError::InvalidArgument);
            }
            active_lane_count += 1;
        }
    }
    if u64::from(active_lane_count) * u64::from(token_count)
        > u64::from(config.max_active_sequence_count)
    {
        return Err(WorkControlError::CapacityExceeded);
    }
    if calculate_packet_bytes(config, active_lane_count) == 0 {
        return Err(WorkControlError::InvalidArgument);
    }
    let mut packet = WorkControlPacket::zeroed(config);
    packet.set_lane_count(config, active_lane_count);
    packet.magic = PACKET_MAGIC;
    packet.abi_version = ABI_VERSION;
    packet.control_generation = STANDALONE_GENERATION;
    packet.flags = FLAG_PREFILL;
    if (prefill_dispatch.request_dispatch.flags & REQUEST_DISPATCH_FLAG_DSPARK_TAP_CAPTURE) != 0 {
        packet.flags |= FLAG_DSPARK_TAP_CAPTURE;
    }
    packet.new_token_count = token_count;
    packet.priority = prefill_dispatch.request_dispatch.highest_priority;
    packet.block_token_count = config.kv_block_tokens;
    packet.max_blocks_per_sequence = config.kv_block_capacity();
    let mut destination_lane_count = 0u32;
    for source_lane_index in 0..prefill_dispatch.lane_count as usize {
        let source_lane = &prefill_dispatch.prefill_view.lanes[source_lane_index];
        if token_offset >= source_lane.prompt_token_count {
            continue;
        }
        if source_lane.prompt_token_count > prefill_dispatch.host_token_stride {
            return Err(WorkControlError::CapacityExceeded);
        }
        if source_lane.prompt_token_offset > u32::MAX - token_offset {
            return Err(WorkControlError::CapacityExceeded);
        }
        let position = source_lane.prompt_token_offset + token_offset;
        if position > u32::MAX - token_count {
            return Err(WorkControlError::CapacityExceeded);
        }
        let destination_lane = &mut packet.lanes[destination_lane_count as usize];
        destination_lane.request_id = source_lane.request_id;
        destination_lane.request_generation = source_lane.request_handle;
        destination_lane.sequence_id = source_lane.sequence_id;
        destination_lane.sequence_position = u64::from(position);
        destination_lane.request_slot_index = source_lane.request_slot_index;
        destination_lane.context_token_count = position + token_count;
        let execution_row_index = destination_lane_count * token_count;
        for row_offset in 0..token_count {
            packet.prefill_token_ids[(execution_row_index + row_offset) as usize] =
                prefill_dispatch.host_token_ids[source_lane_index
                    * prefill_dispatch.host_token_stride as usize
                    + (token_offset + row_offset) as usize];
        }
        destination_lane.input_token_id =
            packet.prefill_token_ids[(execution_row_index + token_count - 1) as usize];
        if destination_lane.context_token_count > packet.kv_block_table_token_count {
            packet.kv_block_table_token_count = destination_lane.context_token_count;
        }
        destination_lane_count += 1;
    }
    if destination_lane_count == 0 {
        return Err(WorkControlError::InvalidArgument);
    }
    packet.active_sequence_count = destination_lane_count;
    packet.lane_count = destination_lane_count;
    packet.rows_per_lane = token_count;
    packet.execution_row_count = destination_lane_count * token_count;
    packet.execution_batch_bucket = select_execution_batch_bucket(
        &prefill_dispatch.request_dispatch,
        packet.execution_row_count,
    )
    .map_err(|_| WorkControlError::InvalidArgument)?;
    packet.descriptor_bytes = calculate_packet_bytes(config, packet.active_sequence_count);
    packet.request_id = packet.lanes[0].request_id;
    packet.sequence_id = packet.lanes[0].sequence_id;
    packet.sequence_position = packet.lanes[0].sequence_position;
    packet.input_token_id = packet.lanes[0].input_token_id;
    finalize_transaction(&mut packet, config, STANDALONE_GENERATION, 0, 1)?;
    Ok(packet)
}

// ---------------------------------------------------------------------------
// Chunk planning / block counting
// ---------------------------------------------------------------------------

/// C: `SparkRingWorkControlBlockCount`.
pub fn block_count(token_count: u32, block_token_count: u32) -> u32 {
    if token_count == 0 || block_token_count == 0 {
        return 0;
    }
    token_count.div_ceil(block_token_count)
}

/// C: `SparkRingWorkControlPlanExecutionChunks`. Returns
/// `(maximum_lanes_per_chunk, chunk_count)`.
pub fn plan_execution_chunks(
    config: &WorkControlConfig,
    logical_lane_count: u32,
    rows_per_lane: u32,
    execution_row_capacity: u32,
) -> Result<(u32, u32)> {
    if logical_lane_count == 0
        || logical_lane_count > config.max_lane_count
        || rows_per_lane == 0
        || execution_row_capacity == 0
    {
        return Err(WorkControlError::InvalidArgument);
    }
    execution_chunk_shape(
        logical_lane_count,
        rows_per_lane,
        execution_row_capacity,
        config.max_active_sequence_count,
    )
}

// ---------------------------------------------------------------------------
// Packet validation (C: SparkRingWorkControlValidatePacket)
// ---------------------------------------------------------------------------

/// C: `SparkRingWorkControlValidatePacket`.
pub fn validate_packet(
    packet: &WorkControlPacket,
    config: &WorkControlConfig,
    max_active_sequence_count: u32,
    max_pipeline_slot_count: u32,
) -> Result<()> {
    if packet.magic != PACKET_MAGIC || packet.abi_version != ABI_VERSION {
        return Err(WorkControlError::AbiMismatch);
    }
    if (packet.flags & !KNOWN_FLAGS) != 0
        || packet.request_id == 0
        || packet.sequence_id == 0
        || packet.control_generation == 0
    {
        return Err(WorkControlError::InvalidArgument);
    }
    validate_transaction_fields(packet, config)?;
    let max_speculative_token_count = config.max_speculative_token_count() as usize;
    let release_sequences = (packet.flags & FLAG_RELEASE_SEQUENCES) != 0;
    if release_sequences {
        if packet.flags != FLAG_RELEASE_SEQUENCES
            || packet.descriptor_bytes != calculate_packet_bytes(config, packet.lane_count)
            || packet.active_sequence_count == 0
            || packet.active_sequence_count > config.max_lane_count
            || packet.lane_count != packet.active_sequence_count
            || packet.lane_count > config.max_lane_count
            || packet.new_token_count != 0
            || packet.pipeline_slot >= max_pipeline_slot_count
            || packet.block_token_count == 0
            || packet.kv_block_table_token_count == 0
            || packet.max_blocks_per_sequence == 0
            || packet.mtp_draft_token_count != 0
            || packet.input_token_id != 0
            || packet.speculative_token_count != 0
            || packet.speculative_token_index != 0
            || packet.rows_per_lane != 0
            || packet.execution_row_count != 0
            || packet.execution_batch_bucket != 0
        {
            return Err(WorkControlError::InvalidArgument);
        }
        if packet.speculative_draft_token_ids[..max_speculative_token_count]
            .iter()
            .any(|&token_id| token_id != 0)
        {
            return Err(WorkControlError::InvalidArgument);
        }
        for lane in packet.lanes[..packet.lane_count as usize].iter() {
            if lane.request_id == 0
                || lane.sequence_id == 0
                || lane.context_token_count == 0
                || lane.context_token_count > packet.kv_block_table_token_count
                || block_count(lane.context_token_count, packet.block_token_count)
                    > packet.max_blocks_per_sequence
                || lane.input_token_id != 0
                || lane.mtp_draft_token_count != 0
                || lane.speculative_token_count != 0
                || lane.mtp_resolution_proposed_token_count != 0
                || lane.mtp_resolution_accepted_token_count != 0
                || u32::from(lane.mtp_resolution_path_id) != mtp_tree::RESOLUTION_NONE
            {
                return Err(WorkControlError::InvalidArgument);
            }
            if lane.speculative_draft_token_ids[..max_speculative_token_count]
                .iter()
                .any(|&token_id| token_id != 0)
            {
                return Err(WorkControlError::InvalidArgument);
            }
        }
        if packet.request_id != packet.lanes[0].request_id
            || packet.sequence_id != packet.lanes[0].sequence_id
        {
            return Err(WorkControlError::InvalidArgument);
        }
        return Ok(());
    }
    if packet.active_sequence_count == 0
        || packet.active_sequence_count > config.max_lane_count
        || packet.lane_count == 0
        || packet.lane_count > config.max_lane_count
        || packet.lane_count != packet.active_sequence_count
        || packet.new_token_count == 0
        || packet.pipeline_slot >= max_pipeline_slot_count
        || packet.block_token_count == 0
        || packet.kv_block_table_token_count == 0
        || packet.max_blocks_per_sequence == 0
    {
        return Err(WorkControlError::InvalidArgument);
    }
    if packet.active_sequence_count > config.max_active_sequence_count
        || packet.descriptor_bytes != calculate_packet_bytes(config, packet.active_sequence_count)
        || packet.max_blocks_per_sequence > config.kv_block_capacity()
    {
        return Err(WorkControlError::AbiMismatch);
    }
    if (packet.flags & FLAG_PREFILL) == 0
        && packet.new_token_count > config.max_speculative_token_count() + 1
    {
        return Err(WorkControlError::InvalidArgument);
    }
    if (packet.flags & FLAG_PREFILL) != 0
        && (packet.new_token_count == 0
            || packet.new_token_count > config.max_prefill_tokens_per_packet)
    {
        return Err(WorkControlError::InvalidArgument);
    }
    let dspark_verify = (packet.flags & FLAG_DSPARK_SPECULATIVE_VERIFY) != 0;
    let mtp_verify = (packet.flags & FLAG_MTP_SPECULATIVE_VERIFY) != 0;
    let mtp_tree_verify = (packet.flags & FLAG_MTP_TREE_VERIFY) != 0;
    let speculative_verify = dspark_verify || mtp_verify;
    let expected_rows_per_lane = if (packet.flags & FLAG_PREFILL) != 0 {
        packet.new_token_count
    } else if speculative_verify {
        if mtp_tree_verify {
            mtp_tree::VERIFIER_ROW_COUNT
        } else {
            packet.speculative_token_count + 1
        }
    } else {
        1
    };
    let expected_execution_row_count =
        u64::from(packet.lane_count) * u64::from(expected_rows_per_lane);
    if packet.rows_per_lane != expected_rows_per_lane
        || expected_execution_row_count > u64::from(u32::MAX)
        || expected_execution_row_count > u64::from(max_active_sequence_count)
        || u64::from(packet.execution_row_count) != expected_execution_row_count
        || !batch_bucket_is_supported(packet.execution_batch_bucket)
        || (if (packet.flags & FLAG_PREFILL) != 0 {
            packet.execution_row_count
        } else {
            packet.lane_count
        }) > packet.execution_batch_bucket
    {
        return Err(WorkControlError::InvalidArgument);
    }
    if packet.input_token_id >= config.output_vocab_count {
        return Err(WorkControlError::InvalidArgument);
    }
    if (packet.flags & FLAG_PREFILL) != 0 {
        for execution_row_index in 0..packet.execution_row_count as usize {
            if packet.prefill_token_ids[execution_row_index] >= config.output_vocab_count {
                return Err(WorkControlError::InvalidArgument);
            }
        }
    }
    if dspark_verify && mtp_verify {
        return Err(WorkControlError::InvalidArgument);
    }
    if mtp_tree_verify
        && (!mtp_verify
            || packet.speculative_token_count != mtp_tree::CANDIDATE_COUNT
            || packet.rows_per_lane != mtp_tree::VERIFIER_ROW_COUNT)
    {
        return Err(WorkControlError::InvalidArgument);
    }
    if packet.mtp_draft_token_count > config.mtp_draft_token_count
        || ((packet.flags & FLAG_MTP_DRAFT) != 0) != (packet.mtp_draft_token_count != 0)
    {
        return Err(WorkControlError::InvalidArgument);
    }
    let mut mtp_resolution_lane_count = 0u32;
    if speculative_verify {
        if (packet.flags & FLAG_PREFILL) != 0
            || packet.new_token_count != packet.rows_per_lane
            || packet.speculative_token_count == 0
            || packet.speculative_token_count > config.max_speculative_token_count()
            || packet.speculative_token_index != 0
            || (dspark_verify && (packet.flags & FLAG_DSPARK_TAP_CAPTURE) == 0)
        {
            return Err(WorkControlError::InvalidArgument);
        }
        if mtp_tree_verify {
            if (packet.flags & FLAG_MTP_DRAFT) == 0
                || packet.mtp_draft_token_count != mtp_tree::CANDIDATE_COUNT
            {
                return Err(WorkControlError::ModuleNotValidated);
            }
        } else if (packet.flags & FLAG_MTP_DRAFT) != 0 || packet.mtp_draft_token_count != 0 {
            return Err(WorkControlError::InvalidArgument);
        }
        for token_index in 0..packet.speculative_token_count as usize {
            if packet.speculative_draft_token_ids[token_index] >= config.output_vocab_count {
                return Err(WorkControlError::InvalidArgument);
            }
        }
        if packet.speculative_draft_token_ids
            [packet.speculative_token_count as usize..max_speculative_token_count]
            .iter()
            .any(|&token_id| token_id != 0)
        {
            return Err(WorkControlError::InvalidArgument);
        }
    } else {
        if packet.speculative_token_count != 0 || packet.speculative_token_index != 0 {
            return Err(WorkControlError::InvalidArgument);
        }
        if packet.speculative_draft_token_ids[..max_speculative_token_count]
            .iter()
            .any(|&token_id| token_id != 0)
        {
            return Err(WorkControlError::InvalidArgument);
        }
    }
    for (lane_index, lane) in packet.lanes[..packet.lane_count as usize].iter().enumerate() {
        let lane_position_count = written_position_count(packet);
        if lane.request_id == 0
            || lane.sequence_id == 0
            || lane.request_slot_index == INVALID_REQUEST_SLOT
            || lane.context_token_count == 0
            || lane.sequence_position + u64::from(lane_position_count)
                > u64::from(lane.context_token_count)
            || lane.context_token_count > packet.kv_block_table_token_count
            || block_count(lane.context_token_count, packet.block_token_count)
                > packet.max_blocks_per_sequence
            || lane.input_token_id >= config.output_vocab_count
        {
            return Err(WorkControlError::InvalidArgument);
        }
        if (packet.flags & FLAG_PREFILL) != 0
            && (lane.mtp_draft_token_count != 0
                || lane.speculative_token_count != 0
                || u64::from(lane.context_token_count)
                    != lane.sequence_position + u64::from(packet.new_token_count)
                || lane.input_token_id
                    != packet.prefill_token_ids[lane_index * packet.rows_per_lane as usize
                        + packet.rows_per_lane as usize
                        - 1])
        {
            return Err(WorkControlError::InvalidArgument);
        }
        if lane.mtp_resolution_proposed_token_count == 0 {
            if lane.mtp_resolution_accepted_token_count != 0
                || u32::from(lane.mtp_resolution_path_id) != mtp_tree::RESOLUTION_NONE
            {
                return Err(WorkControlError::InvalidArgument);
            }
        } else {
            if (packet.flags & FLAG_MTP_RESOLVE) == 0
                || u32::from(lane.mtp_resolution_proposed_token_count)
                    > config.mtp_draft_token_count
                || !mtp_tree::resolution_is_valid(
                    u32::from(lane.mtp_resolution_proposed_token_count),
                    u32::from(lane.mtp_resolution_accepted_token_count),
                    u32::from(lane.mtp_resolution_path_id),
                )
                || lane.sequence_position < u64::from(lane.mtp_resolution_accepted_token_count) + 1
            {
                return Err(WorkControlError::InvalidArgument);
            }
            mtp_resolution_lane_count += 1;
        }
        if speculative_verify {
            if lane.mtp_draft_token_count != packet.mtp_draft_token_count
                || lane.speculative_token_count != packet.speculative_token_count
            {
                return Err(WorkControlError::InvalidArgument);
            }
            for token_index in 0..lane.speculative_token_count as usize {
                if lane.speculative_draft_token_ids[token_index] >= config.output_vocab_count {
                    return Err(WorkControlError::InvalidArgument);
                }
            }
            if lane.speculative_draft_token_ids
                [lane.speculative_token_count as usize..max_speculative_token_count]
                .iter()
                .any(|&token_id| token_id != 0)
            {
                return Err(WorkControlError::InvalidArgument);
            }
        } else {
            if lane.speculative_token_count != 0
                || lane.mtp_draft_token_count != packet.mtp_draft_token_count
            {
                return Err(WorkControlError::InvalidArgument);
            }
            if lane.speculative_draft_token_ids[..max_speculative_token_count]
                .iter()
                .any(|&token_id| token_id != 0)
            {
                return Err(WorkControlError::InvalidArgument);
            }
        }
    }
    if ((packet.flags & FLAG_MTP_RESOLVE) != 0) != (mtp_resolution_lane_count != 0) {
        return Err(WorkControlError::InvalidArgument);
    }
    if packet.request_id != packet.lanes[0].request_id
        || packet.sequence_id != packet.lanes[0].sequence_id
        || packet.sequence_position != packet.lanes[0].sequence_position
        || packet.input_token_id != packet.lanes[0].input_token_id
        || (speculative_verify
            && packet.speculative_draft_token_ids[..max_speculative_token_count]
                != packet.lanes[0].speculative_draft_token_ids[..max_speculative_token_count])
    {
        return Err(WorkControlError::InvalidArgument);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// KV keys and open-addressed index tables
// ---------------------------------------------------------------------------

/// C: `SparkRingKvKey` — 128-bit block identity. The low bit of `high` is
/// the domain tag: 1 marks a private key derived from
/// `(sequence_id, logical_block_index)`, 0 marks a content key derived from
/// the token prefix a block holds. The two domains can never alias, and an
/// all-zero key is reserved to mean "empty slot".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvKey {
    pub low: u64,
    pub high: u64,
}

impl KvKey {
    /// C: `SparkRingWorkControlKvKeyEmpty` (static).
    pub fn is_empty(&self) -> bool {
        self.low == 0 && self.high == 0
    }
}

// `SparkRingWorkControlKvKeyEqual` is replaced by `PartialEq` on `KvKey`.

/// C: `SparkRingWorkControlKvMix` (splitmix64 finalizer).
fn kv_mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// C: `SparkRingWorkControlPrivateKey`.
pub fn private_key(sequence_id: u64, logical_block_index: u32) -> KvKey {
    let low =
        kv_mix(sequence_id ^ (u64::from(logical_block_index)).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let high = kv_mix(low ^ 0xd6e8_feb8_6659_fd93) | 1;
    KvKey { low, high }
}

/// C: `SparkRingWorkControlContentKey`.
pub fn content_key(digest_low: u64, digest_high: u64) -> KvKey {
    KvKey { low: digest_low, high: (digest_high | 2) & !1 }
}

/// C: `SparkRingWorkControlKvCommittedFrontier`. Tokens of this lane that
/// are committed: everything below the speculative and MTP draft tail. A
/// block ending at or below this line may be shared; a block crossing it
/// holds a draft that may yet be rejected, so it stays private. An
/// outstanding count above the context length yields zero, which shares
/// nothing — the safe direction.
pub fn kv_committed_frontier(lane: &WorkControlLane) -> u64 {
    let outstanding_token_count =
        u64::from(lane.speculative_token_count) + u64::from(lane.mtp_draft_token_count);
    if outstanding_token_count < u64::from(lane.context_token_count) {
        u64::from(lane.context_token_count) - outstanding_token_count
    } else {
        0
    }
}

/// C: `SparkRingWorkControlKvDirectoryEntry`. One sequence slot; names the
/// block record it resolves to and owns no storage, so many sequence slots
/// may name one block.
#[derive(Debug, Clone, Copy, Default)]
pub struct KvDirectoryEntry {
    pub key: KvKey,
    pub block_key: KvKey,
}

/// C: `SparkRingWorkControlKvBlockEntry`. One block; sole owner of
/// residency, backing storage and the share refcount, so an eviction is
/// observed by every sequence that names this block. The C bitfields are
/// plain fields here.
#[derive(Debug, Clone, Copy, Default)]
pub struct KvBlockEntry {
    pub key: KvKey,
    pub physical_block_index: u32,
    pub backing_block_index: u32,
    pub reference_count: u32,
    /// C: 2-bit field (`SPARK_RING_KV_DIRECTORY_RESIDENCY_*`).
    pub residency_state: u8,
    /// C: 1-bit field.
    pub backing_valid: bool,
    /// C: 29-bit field.
    pub prefetch_mark: u32,
}

/// C: `SparkRingWorkControlKvPrefetchEntry`.
#[derive(Debug, Clone, Copy, Default)]
pub struct KvPrefetchEntry {
    pub key: KvKey,
    pub backing_block_index: u32,
}

/// Trait shared by the two open-addressed tables (C: the generic
/// `SparkRingWorkControlKvIndex*` helpers over "entries that begin with a
/// `SparkRingKvKey`").
trait KvIndexed: Copy + Default {
    fn key(&self) -> KvKey;
}

impl KvIndexed for KvDirectoryEntry {
    fn key(&self) -> KvKey {
        self.key
    }
}

impl KvIndexed for KvBlockEntry {
    fn key(&self) -> KvKey {
        self.key
    }
}

/// C: `SparkRingWorkControlKvIndexHome`. Both key domains carry an
/// avalanched low half — private keys are a splitmix output, content keys a
/// caller digest — so the home slot is a mask, not another mix.
fn kv_index_home(key: KvKey, mask: u32) -> u32 {
    (key.low & u64::from(mask)) as u32
}

/// C: `SparkRingWorkControlKvIndexProbe`. Linear probe over a power-of-two
/// open-addressed table. Returns `(slot, found)` — the slot to occupy when
/// the key is absent — so one implementation serves both the sequence
/// directory and the block table. `(u32::MAX, false)` when saturated.
fn kv_index_probe<E: KvIndexed>(entries: &[E], key: KvKey) -> (u32, bool) {
    let mask = entries.len() as u32 - 1;
    let mut slot = kv_index_home(key, mask);
    for _ in 0..=mask {
        let slot_key = entries[slot as usize].key();
        if slot_key.is_empty() {
            return (slot, false);
        }
        if slot_key == key {
            return (slot, true);
        }
        slot = (slot + 1) & mask;
    }
    (u32::MAX, false)
}

/// C: `SparkRingWorkControlKvIndexErase`. Backward-shift deletion keeps
/// every probe chain contiguous, so no tombstone state is needed and probe
/// length stays governed only by load factor. The scan is bounded by a full
/// revolution so a saturated table cannot spin.
fn kv_index_erase<E: KvIndexed>(entries: &mut [E], slot: u32) {
    let mask = entries.len() as u32 - 1;
    let mut hole = slot;
    entries[hole as usize] = E::default();
    let mut scan = (hole + 1) & mask;
    while scan != hole {
        let scan_key = entries[scan as usize].key();
        if scan_key.is_empty() {
            return;
        }
        let home = kv_index_home(scan_key, mask);
        let hole_distance = hole.wrapping_sub(home) & mask;
        let scan_distance = scan.wrapping_sub(home) & mask;
        if hole_distance >= scan_distance {
            scan = (scan + 1) & mask;
            continue;
        }
        entries[hole as usize] = entries[scan as usize];
        entries[scan as usize] = E::default();
        hole = scan;
        scan = (scan + 1) & mask;
    }
}

/// C: `SparkRingWorkControlKvIndexFind`. Slot of an existing key, or `None`.
/// Load factor is held at one half, so an exhausted probe and an absent key
/// are the same answer to every caller.
fn kv_index_find<E: KvIndexed>(entries: &[E], key: KvKey) -> Option<u32> {
    let (slot, found) = kv_index_probe(entries, key);
    if found {
        Some(slot)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// KV swap hooks (C: swap store/load function pointers + context)
// ---------------------------------------------------------------------------

/// C: `SparkRingWorkControlKvSwapStoreFunction` /
/// `SparkRingWorkControlKvSwapLoadFunction` plus the `void *swap_context`.
pub trait KvSwapHooks {
    /// C: swap store — write a resident block out to its backing slot.
    fn store(
        &mut self,
        key: KvKey,
        physical_block_index: u32,
        backing_block_index: u32,
    ) -> Result<()>;
    /// C: swap load — fault a block back from backing storage.
    fn load(
        &mut self,
        key: KvKey,
        physical_block_index: u32,
        backing_block_index: u32,
    ) -> Result<()>;
}

// ---------------------------------------------------------------------------
// KV state (C: SparkRingWorkControlKvState; owned Vecs replace the
// caller-supplied arrays)
// ---------------------------------------------------------------------------

/// C: `SparkRingWorkControlKvState`. All C pointer fields become owned
/// `Vec`s (or `Option<Vec>` for the opt-in pin/backing/sharing arrays);
/// counters and configuration stay as public fields so tests and callers
/// can observe them exactly as in C.
pub struct KvState {
    pub abi_version: u32,
    pub descriptor_bytes: u32,
    pub lane_capacity: u32,
    pub lane_stride: u32,
    pub block_token_count: u32,
    pub table_entry_capacity: u32,
    pub physical_block_capacity: u32,
    pub directory_capacity: u32,
    pub block_entry_capacity: u32,
    pub lane_block_key_stride: u32,
    pub next_physical_block_index: u32,
    pub backing_block_capacity: u32,
    pub free_backing_block_head: u32,
    pub directory_entry_count: u32,
    pub block_entry_count: u32,
    pub swapped_block_count: u32,
    pub clean_evict_count: u32,
    pub epoch: u64,
    pub control_generation: u64,
    pub control_generation_reset_count: u64,
    pub missing_block_count: u32,
    pub in_flight_block_count: u32,
    pub resident_block_count: u32,
    pub allocated_physical_block_count: u32,
    pub swap_store_count: u64,
    pub swap_load_count: u64,
    pub prefetch_generation: u32,
    pub share_hit_count: u64,
    pub share_admit_count: u64,
    /// C capacity: `lane_capacity * lane_stride`.
    pub physical_block_indices: Vec<u32>,
    /// C capacity: `lane_capacity`.
    pub lane_physical_block_counts: Vec<u32>,
    /// C capacity: `physical_block_capacity`.
    pub physical_block_states: Vec<u8>,
    /// C capacity: `physical_block_capacity`.
    pub physical_block_keys: Vec<KvKey>,
    /// C capacity: `physical_block_capacity`.
    pub physical_block_last_used_epochs: Vec<u64>,
    /// C capacity: `physical_block_capacity` (opt-in via
    /// [`KvState::configure_kv_pins`]).
    pub physical_block_pin_counts: Option<Vec<u32>>,
    /// C capacity: `directory_capacity` (power of two, half load factor).
    pub directory_entries: Vec<KvDirectoryEntry>,
    /// C capacity: `block_entry_capacity` (power of two).
    pub block_entries: Vec<KvBlockEntry>,
    /// C: borrowed `lane_block_keys` row table; owned here, supplied via
    /// [`KvState::configure_kv_sharing`].
    pub lane_block_keys: Option<Vec<KvKey>>,
    /// C capacity: `backing_block_capacity` (opt-in via
    /// [`KvState::configure_kv_swap`]).
    pub backing_block_free_next: Option<Vec<u32>>,
    swap_hooks: Option<Box<dyn KvSwapHooks>>,
}

/// C: `SparkKvBlockTableView` as produced by
/// `SparkRingWorkControlBuildHostKvBlockTable`. The C view carries raw
/// pointers into the state's storage; the Rust view borrows the same slices.
#[derive(Debug)]
pub struct KvBlockTableView<'a> {
    pub abi_version: u32,
    pub descriptor_bytes: u32,
    pub block_token_count: u32,
    pub lane_count: u32,
    pub lane_stride: u32,
    pub lane_capacity: u32,
    pub physical_block_indices: &'a [u32],
    pub lane_physical_block_counts: &'a [u32],
    pub host_physical_block_indices: &'a [u32],
    pub host_lane_physical_block_counts: &'a [u32],
}

impl KvState {
    /// C: `SparkRingWorkControlInitializeKvState`. Allocates the storage the
    /// C caller supplied as raw arrays.
    pub fn new(
        lane_capacity: u32,
        lane_stride: u32,
        block_token_count: u32,
        physical_block_capacity: u32,
        directory_capacity: u32,
        block_entry_capacity: u32,
    ) -> Result<Self> {
        if lane_capacity == 0
            || lane_stride == 0
            || block_token_count == 0
            || physical_block_capacity == 0
            || physical_block_capacity > u32::MAX / 2
            || directory_capacity < physical_block_capacity * 2
            || (directory_capacity & (directory_capacity - 1)) != 0
            || block_entry_capacity < physical_block_capacity * 2
            || (block_entry_capacity & (block_entry_capacity - 1)) != 0
        {
            return Err(WorkControlError::InvalidArgument);
        }
        let table_entry_capacity = u64::from(lane_capacity) * u64::from(lane_stride);
        if table_entry_capacity > u64::from(u32::MAX) {
            return Err(WorkControlError::CapacityExceeded);
        }
        Ok(Self {
            abi_version: ABI_VERSION,
            descriptor_bytes: KV_STATE_DESCRIPTOR_BYTES,
            lane_capacity,
            lane_stride,
            block_token_count,
            table_entry_capacity: table_entry_capacity as u32,
            physical_block_capacity,
            directory_capacity,
            block_entry_capacity,
            lane_block_key_stride: 0,
            next_physical_block_index: 0,
            backing_block_capacity: 0,
            free_backing_block_head: KV_INVALID_BLOCK_INDEX,
            directory_entry_count: 0,
            block_entry_count: 0,
            swapped_block_count: 0,
            clean_evict_count: 0,
            epoch: 0,
            control_generation: 0,
            control_generation_reset_count: 0,
            missing_block_count: 0,
            in_flight_block_count: 0,
            resident_block_count: 0,
            allocated_physical_block_count: 0,
            swap_store_count: 0,
            swap_load_count: 0,
            prefetch_generation: 0,
            share_hit_count: 0,
            share_admit_count: 0,
            physical_block_indices: vec![KV_INVALID_BLOCK_INDEX; table_entry_capacity as usize],
            lane_physical_block_counts: vec![0; lane_capacity as usize],
            physical_block_states: vec![KV_ENTRY_MISSING; physical_block_capacity as usize],
            physical_block_keys: vec![KvKey::default(); physical_block_capacity as usize],
            physical_block_last_used_epochs: vec![0; physical_block_capacity as usize],
            physical_block_pin_counts: None,
            directory_entries: vec![KvDirectoryEntry::default(); directory_capacity as usize],
            block_entries: vec![KvBlockEntry::default(); block_entry_capacity as usize],
            lane_block_keys: None,
            backing_block_free_next: None,
            swap_hooks: None,
        })
    }

    /// C: `SparkRingWorkControlConfigureKvSwap`.
    pub fn configure_kv_swap(
        &mut self,
        backing_block_capacity: u32,
        swap_hooks: Box<dyn KvSwapHooks>,
    ) -> Result<()> {
        if self.abi_version != ABI_VERSION
            || self.descriptor_bytes != KV_STATE_DESCRIPTOR_BYTES
            || backing_block_capacity == 0
            || backing_block_capacity > self.directory_capacity / 2
            || self.directory_entry_count != 0
            || self.allocated_physical_block_count != 0
            || self.backing_block_capacity != 0
        {
            return Err(WorkControlError::InvalidArgument);
        }
        let mut free_next = vec![KV_INVALID_BLOCK_INDEX; backing_block_capacity as usize];
        for (index, next) in free_next.iter_mut().enumerate() {
            *next = if index + 1 < backing_block_capacity as usize {
                index as u32 + 1
            } else {
                KV_INVALID_BLOCK_INDEX
            };
        }
        self.backing_block_capacity = backing_block_capacity;
        self.free_backing_block_head = 0;
        self.backing_block_free_next = Some(free_next);
        self.swap_hooks = Some(swap_hooks);
        Ok(())
    }

    /// C: `SparkRingWorkControlConfigureKvPins`. Allocates the pin count
    /// table the C caller supplied.
    pub fn configure_kv_pins(&mut self) -> Result<()> {
        if self.abi_version != ABI_VERSION
            || self.descriptor_bytes != KV_STATE_DESCRIPTOR_BYTES
            || self.physical_block_pin_counts.is_some()
            || self.directory_entry_count != 0
            || self.allocated_physical_block_count != 0
        {
            return Err(WorkControlError::InvalidArgument);
        }
        self.physical_block_pin_counts = Some(vec![0; self.physical_block_capacity as usize]);
        Ok(())
    }

    /// C: `SparkRingWorkControlConfigureKvSharing`. Supply per-lane content
    /// keys for the blocks the next packet will acquire. Leaving this unset,
    /// or leaving a key zero, keeps a block private, so sharing is opt-in
    /// per block and the default behaviour is unchanged. Takes an owned copy
    /// of the row table the C API borrowed.
    pub fn configure_kv_sharing(
        &mut self,
        lane_block_keys: Option<Vec<KvKey>>,
        lane_block_key_stride: u32,
    ) -> Result<()> {
        if self.abi_version != ABI_VERSION
            || lane_block_keys.is_none() != (lane_block_key_stride == 0)
            || lane_block_key_stride > self.lane_stride
        {
            return Err(WorkControlError::InvalidArgument);
        }
        self.lane_block_keys = lane_block_keys;
        self.lane_block_key_stride = lane_block_key_stride;
        Ok(())
    }

    /// C: `SparkRingWorkControlPinPhysicalBlock`.
    pub fn pin_physical_block(&mut self, physical_block_index: u32) -> Result<()> {
        if physical_block_index >= self.physical_block_capacity
            || self.physical_block_keys[physical_block_index as usize].is_empty()
        {
            return Err(WorkControlError::InvalidArgument);
        }
        let pin_counts =
            self.physical_block_pin_counts.as_mut().ok_or(WorkControlError::InvalidArgument)?;
        if pin_counts[physical_block_index as usize] == u32::MAX {
            return Err(WorkControlError::InvalidArgument);
        }
        pin_counts[physical_block_index as usize] += 1;
        Ok(())
    }

    /// C: `SparkRingWorkControlUnpinPhysicalBlock`.
    pub fn unpin_physical_block(&mut self, physical_block_index: u32) -> Result<()> {
        if physical_block_index >= self.physical_block_capacity {
            return Err(WorkControlError::InvalidArgument);
        }
        let pin_counts =
            self.physical_block_pin_counts.as_mut().ok_or(WorkControlError::InvalidArgument)?;
        if pin_counts[physical_block_index as usize] == 0 {
            return Err(WorkControlError::InvalidArgument);
        }
        pin_counts[physical_block_index as usize] -= 1;
        Ok(())
    }

    /// C: `SparkRingWorkControlAdvanceKvGeneration`.
    pub fn advance_kv_generation(&mut self, control_generation: u64) -> Result<()> {
        if control_generation == 0 {
            return Err(WorkControlError::InvalidArgument);
        }
        if control_generation < self.control_generation {
            return Err(WorkControlError::ValidationFailed);
        }
        if control_generation != self.control_generation {
            self.reset_kv_generation(control_generation);
        }
        Ok(())
    }

    /// C: `SparkRingWorkControlSelectKvGeneration` (static).
    fn select_kv_generation(&mut self, packet: &WorkControlPacket) -> Result<()> {
        match self.advance_kv_generation(packet.control_generation) {
            Err(WorkControlError::ValidationFailed) => Err(WorkControlError::NotFound),
            other => other,
        }
    }

    /// C: `SparkRingWorkControlResetBackingBlocks` (static).
    fn reset_backing_blocks(&mut self) {
        let Some(free_next) = self.backing_block_free_next.as_mut() else {
            self.free_backing_block_head = KV_INVALID_BLOCK_INDEX;
            return;
        };
        for (index, next) in free_next.iter_mut().enumerate() {
            *next = if index + 1 < self.backing_block_capacity as usize {
                index as u32 + 1
            } else {
                KV_INVALID_BLOCK_INDEX
            };
        }
        self.free_backing_block_head = 0;
    }

    /// C: `SparkRingWorkControlResetKvGeneration` (static).
    fn reset_kv_generation(&mut self, control_generation: u64) {
        self.physical_block_indices.fill(KV_INVALID_BLOCK_INDEX);
        self.lane_physical_block_counts.fill(0);
        self.physical_block_states.fill(KV_ENTRY_MISSING);
        self.physical_block_keys.fill(KvKey::default());
        self.physical_block_last_used_epochs.fill(0);
        if let Some(pin_counts) = self.physical_block_pin_counts.as_mut() {
            pin_counts.fill(0);
        }
        self.directory_entries.fill(KvDirectoryEntry::default());
        self.block_entries.fill(KvBlockEntry::default());
        self.reset_backing_blocks();
        self.next_physical_block_index = 0;
        self.directory_entry_count = 0;
        self.block_entry_count = 0;
        self.swapped_block_count = 0;
        self.epoch = 0;
        self.missing_block_count = 0;
        self.in_flight_block_count = 0;
        self.resident_block_count = 0;
        self.allocated_physical_block_count = 0;
        self.control_generation = control_generation;
        self.control_generation_reset_count += 1;
    }

    /// C: `SparkRingWorkControlResetReadinessCounts` (static).
    fn reset_readiness_counts(&mut self) {
        self.missing_block_count = 0;
        self.in_flight_block_count = 0;
        self.resident_block_count = 0;
    }

    /// C: `SparkRingWorkControlAccountReadiness` (static).
    fn account_readiness(&mut self, entry_state: u8) {
        if entry_state == KV_ENTRY_RESIDENT {
            self.resident_block_count += 1;
        } else if entry_state == KV_ENTRY_IN_FLIGHT {
            self.in_flight_block_count += 1;
        } else {
            self.missing_block_count += 1;
        }
    }

    /// C: `SPARK_KV_FIND_DIRECTORY`.
    fn find_directory(&self, sequence_id: u64, logical_block_index: u32) -> Option<u32> {
        kv_index_find(&self.directory_entries, private_key(sequence_id, logical_block_index))
    }

    /// C: `SPARK_KV_FIND_BLOCK`.
    fn find_block(&self, block_key: KvKey) -> Option<u32> {
        kv_index_find(&self.block_entries, block_key)
    }

    /// C: `SparkRingWorkControlKvSequenceBlock` (static). The block record a
    /// sequence slot names, or `None` when the slot is not held.
    fn kv_sequence_block(&self, sequence_id: u64, logical_block_index: u32) -> Option<u32> {
        let slot = self.find_directory(sequence_id, logical_block_index)?;
        self.find_block(self.directory_entries[slot as usize].block_key)
    }

    /// C: `SparkRingWorkControlKvClearPhysicalBlock` (static).
    fn kv_clear_physical_block(&mut self, physical_block_index: u32) {
        self.physical_block_keys[physical_block_index as usize] = KvKey::default();
        self.physical_block_last_used_epochs[physical_block_index as usize] = 0;
        self.physical_block_states[physical_block_index as usize] = KV_ENTRY_MISSING;
    }

    /// C: `SparkRingWorkControlKvAssignPhysicalBlock` (static).
    fn kv_assign_physical_block(&mut self, physical_block_index: u32, key: KvKey, state: u8) {
        self.physical_block_keys[physical_block_index as usize] = key;
        self.physical_block_last_used_epochs[physical_block_index as usize] = self.epoch;
        self.physical_block_states[physical_block_index as usize] = state;
        self.allocated_physical_block_count += 1;
    }

    /// C: `SparkRingWorkControlKvBackingAcquire` (static).
    fn kv_backing_acquire(&mut self) -> Result<u32> {
        if self.backing_block_capacity == 0 {
            return Ok(KV_INVALID_BLOCK_INDEX);
        }
        let backing_block_index = self.free_backing_block_head;
        if backing_block_index >= self.backing_block_capacity {
            return Err(WorkControlError::CapacityExceeded);
        }
        let free_next =
            self.backing_block_free_next.as_mut().ok_or(WorkControlError::InternalError)?;
        self.free_backing_block_head = free_next[backing_block_index as usize];
        free_next[backing_block_index as usize] = KV_INVALID_BLOCK_INDEX;
        Ok(backing_block_index)
    }

    /// C: `SparkRingWorkControlKvBackingRelease` (static).
    fn kv_backing_release(&mut self, backing_block_index: u32) -> Result<()> {
        if self.backing_block_capacity == 0 {
            return if backing_block_index == KV_INVALID_BLOCK_INDEX {
                Ok(())
            } else {
                Err(WorkControlError::InternalError)
            };
        }
        let free_next =
            self.backing_block_free_next.as_mut().ok_or(WorkControlError::InternalError)?;
        if backing_block_index >= self.backing_block_capacity
            || free_next[backing_block_index as usize] != KV_INVALID_BLOCK_INDEX
        {
            return Err(WorkControlError::InternalError);
        }
        free_next[backing_block_index as usize] = self.free_backing_block_head;
        self.free_backing_block_head = backing_block_index;
        Ok(())
    }

    /// C: `SparkRingWorkControlKvSpillBlock` (static). Write a resident
    /// block out to its backing slot and mark the record swapped.
    fn kv_spill_block(&mut self, slot: u32, physical_block_index: u32) -> Result<()> {
        let entry = self.block_entries[slot as usize];
        if !entry.backing_valid {
            let hooks = self.swap_hooks.as_mut().ok_or(WorkControlError::InternalError)?;
            hooks.store(entry.key, physical_block_index, entry.backing_block_index)?;
            self.block_entries[slot as usize].backing_valid = true;
            self.swap_store_count += 1;
        } else {
            self.clean_evict_count += 1;
        }
        let entry = &mut self.block_entries[slot as usize];
        entry.physical_block_index = KV_INVALID_BLOCK_INDEX;
        entry.residency_state = KV_DIRECTORY_RESIDENCY_NVME;
        self.kv_clear_physical_block(physical_block_index);
        self.allocated_physical_block_count -= 1;
        self.swapped_block_count += 1;
        Ok(())
    }

    /// C: `SparkRingWorkControlKvAcquirePhysicalBlock` (static). Clock
    /// sweep. A free slot wins outright, otherwise the first unpinned
    /// resident block outside the current epoch is spilled. The reverse map
    /// is the block key carried on the physical block, so the cost is one
    /// probe however many sequences share the block, and every sharer
    /// observes the spill because residency lives only on the block record.
    fn kv_acquire_physical_block(&mut self) -> Result<u32> {
        for _ in 0..self.physical_block_capacity {
            let physical_block_index = self.next_physical_block_index;
            self.next_physical_block_index =
                (physical_block_index + 1) % self.physical_block_capacity;
            if let Some(pin_counts) = self.physical_block_pin_counts.as_ref() {
                if pin_counts[physical_block_index as usize] != 0 {
                    continue;
                }
            }
            if self.physical_block_keys[physical_block_index as usize].is_empty() {
                return Ok(physical_block_index);
            }
            if self.backing_block_capacity == 0
                || self.physical_block_states[physical_block_index as usize] != KV_ENTRY_RESIDENT
                || self.physical_block_last_used_epochs[physical_block_index as usize] == self.epoch
            {
                continue;
            }
            let key = self.physical_block_keys[physical_block_index as usize];
            let slot = self.find_block(key).ok_or(WorkControlError::InternalError)?;
            self.kv_spill_block(slot, physical_block_index)?;
            return Ok(physical_block_index);
        }
        Err(WorkControlError::CapacityExceeded)
    }

    /// C: `SparkRingWorkControlAcquireTransientPhysicalBlock`.
    pub fn acquire_transient_physical_block(&mut self) -> Result<u32> {
        if self.physical_block_pin_counts.is_none() {
            return Err(WorkControlError::InvalidArgument);
        }
        let physical_block_index = self.kv_acquire_physical_block()?;
        self.kv_assign_physical_block(
            physical_block_index,
            private_key(u64::MAX, physical_block_index),
            KV_ENTRY_TRANSIENT,
        );
        let pin_counts =
            self.physical_block_pin_counts.as_mut().ok_or(WorkControlError::InternalError)?;
        pin_counts[physical_block_index as usize] = 1;
        Ok(physical_block_index)
    }

    /// C: `SparkRingWorkControlReleaseTransientPhysicalBlock`.
    pub fn release_transient_physical_block(&mut self, physical_block_index: u32) -> Result<()> {
        if physical_block_index >= self.physical_block_capacity
            || self.physical_block_states[physical_block_index as usize] != KV_ENTRY_TRANSIENT
            || self.physical_block_keys[physical_block_index as usize]
                != private_key(u64::MAX, physical_block_index)
            || self.allocated_physical_block_count == 0
        {
            return Err(WorkControlError::InvalidArgument);
        }
        let pin_counts =
            self.physical_block_pin_counts.as_mut().ok_or(WorkControlError::InvalidArgument)?;
        if pin_counts[physical_block_index as usize] != 1 {
            return Err(WorkControlError::InvalidArgument);
        }
        pin_counts[physical_block_index as usize] = 0;
        self.kv_clear_physical_block(physical_block_index);
        self.allocated_physical_block_count -= 1;
        Ok(())
    }

    /// C: `SparkRingWorkControlKvBlockResolve` (static). Take a reference on
    /// the block a key names, admitting a physical block and a backing slot
    /// on the first reference and sharing them on every later one.
    fn kv_block_resolve(&mut self, block_key: KvKey) -> Result<()> {
        let (slot, found) = kv_index_probe(&self.block_entries, block_key);
        if slot == u32::MAX {
            return Err(WorkControlError::CapacityExceeded);
        }
        if found {
            self.block_entries[slot as usize].reference_count += 1;
            self.share_hit_count += 1;
            return Ok(());
        }
        if self.block_entry_count
            >= if self.backing_block_capacity != 0 {
                self.backing_block_capacity
            } else {
                self.physical_block_capacity
            }
        {
            return Err(WorkControlError::CapacityExceeded);
        }
        let backing_block_index = self.kv_backing_acquire()?;
        let physical_block_index = match self.kv_acquire_physical_block() {
            Ok(index) => index,
            Err(error) => {
                let _ = self.kv_backing_release(backing_block_index);
                return Err(error);
            }
        };
        self.block_entries[slot as usize] = KvBlockEntry {
            key: block_key,
            physical_block_index,
            backing_block_index,
            reference_count: 1,
            residency_state: KV_DIRECTORY_RESIDENCY_GPU,
            backing_valid: false,
            prefetch_mark: 0,
        };
        self.block_entry_count += 1;
        self.share_admit_count += 1;
        self.kv_assign_physical_block(physical_block_index, block_key, KV_ENTRY_MISSING);
        Ok(())
    }

    /// C: `SparkRingWorkControlKvBlockDeref` (static). Drop a reference,
    /// freeing the physical and backing blocks at zero. The decrement is
    /// guarded so an unbalanced release reports rather than wrapping to
    /// `UINT32_MAX` and stranding the block forever.
    fn kv_block_deref(&mut self, block_key: KvKey) -> Result<()> {
        let slot = self.find_block(block_key).ok_or(WorkControlError::NotFound)?;
        let entry = self.block_entries[slot as usize];
        if entry.reference_count > 1 {
            self.block_entries[slot as usize].reference_count -= 1;
            return Ok(());
        }
        if entry.residency_state == KV_DIRECTORY_RESIDENCY_GPU {
            if let Some(pin_counts) = self.physical_block_pin_counts.as_ref() {
                if pin_counts[entry.physical_block_index as usize] != 0 {
                    return Err(WorkControlError::Busy);
                }
            }
            self.kv_clear_physical_block(entry.physical_block_index);
            self.allocated_physical_block_count -= 1;
        } else {
            self.swapped_block_count -= 1;
        }
        self.kv_backing_release(entry.backing_block_index)?;
        self.block_entries[slot as usize].reference_count = 0;
        self.block_entry_count -= 1;
        kv_index_erase(&mut self.block_entries, slot);
        Ok(())
    }

    /// C: `SparkRingWorkControlKvRekeyBlock` (static). Change a block
    /// record's identity in place. Used when a private block becomes the
    /// first block to publish its content, so the bytes already computed are
    /// kept and no other sequence has to recompute them.
    fn kv_rekey_block(&mut self, source_slot: u32, block_key: KvKey) {
        let mut record = self.block_entries[source_slot as usize];
        kv_index_erase(&mut self.block_entries, source_slot);
        record.key = block_key;
        let (slot, _found) = kv_index_probe(&self.block_entries, block_key);
        self.block_entries[slot as usize] = record;
        if record.residency_state == KV_DIRECTORY_RESIDENCY_GPU {
            self.physical_block_keys[record.physical_block_index as usize] = block_key;
        }
    }

    /// C: `SparkRingWorkControlKvPromoteBlock` (static). Move a sequence
    /// slot from its private block onto the shared block holding the same
    /// content, or publish the private block under that content key when no
    /// sharer exists yet. Refused while the private block carries an
    /// in-flight speculative write, so a rejected draft can never reach
    /// another sequence.
    fn kv_promote_block(&mut self, directory_slot: u32, block_key: KvKey) -> Result<()> {
        let source_slot = self
            .find_block(self.directory_entries[directory_slot as usize].block_key)
            .ok_or(WorkControlError::NotFound)?;
        let previous = self.block_entries[source_slot as usize];
        if previous.residency_state == KV_DIRECTORY_RESIDENCY_GPU {
            if let Some(pin_counts) = self.physical_block_pin_counts.as_ref() {
                if pin_counts[previous.physical_block_index as usize] != 0 {
                    return Ok(());
                }
            }
        }
        if self.find_block(block_key).is_none() && previous.reference_count == 1 {
            self.kv_rekey_block(source_slot, block_key);
            self.directory_entries[directory_slot as usize].block_key = block_key;
            return Ok(());
        }
        self.kv_block_resolve(block_key)?;
        let previous_key = self.directory_entries[directory_slot as usize].block_key;
        self.kv_block_deref(previous_key)?;
        self.directory_entries[directory_slot as usize].block_key = block_key;
        Ok(())
    }

    /// C: `SparkRingWorkControlKvBlockResident` (static). Return the
    /// physical block a key resides in, faulting it back from backing
    /// storage when the clock sweep has spilled it.
    fn kv_block_resident(&mut self, block_key: KvKey) -> Result<u32> {
        let slot = self.find_block(block_key).ok_or(WorkControlError::NotFound)?;
        let entry = self.block_entries[slot as usize];
        if entry.residency_state == KV_DIRECTORY_RESIDENCY_GPU {
            self.physical_block_last_used_epochs[entry.physical_block_index as usize] = self.epoch;
            return Ok(entry.physical_block_index);
        }
        let physical_block_index = self.kv_acquire_physical_block()?;
        let hooks = self.swap_hooks.as_mut().ok_or(WorkControlError::InternalError)?;
        hooks.load(block_key, physical_block_index, entry.backing_block_index)?;
        self.swapped_block_count -= 1;
        self.swap_load_count += 1;
        let entry = &mut self.block_entries[slot as usize];
        entry.physical_block_index = physical_block_index;
        entry.residency_state = KV_DIRECTORY_RESIDENCY_GPU;
        self.kv_assign_physical_block(physical_block_index, block_key, KV_ENTRY_RESIDENT);
        Ok(physical_block_index)
    }

    /// C: `SparkRingWorkControlKvDirectoryAcquire` (static).
    fn kv_directory_acquire(
        &mut self,
        sequence_id: u64,
        logical_block_index: u32,
        block_key: KvKey,
    ) -> Result<u32> {
        let directory_key = private_key(sequence_id, logical_block_index);
        let (slot, found) = kv_index_probe(&self.directory_entries, directory_key);
        if slot == u32::MAX {
            return Err(WorkControlError::CapacityExceeded);
        }
        if !found {
            if self.directory_entry_count >= self.directory_capacity / 2 {
                return Err(WorkControlError::CapacityExceeded);
            }
            self.kv_block_resolve(block_key)?;
            self.directory_entries[slot as usize] =
                KvDirectoryEntry { key: directory_key, block_key };
            self.directory_entry_count += 1;
        } else if self.directory_entries[slot as usize].block_key != block_key {
            self.kv_promote_block(slot, block_key)?;
        }
        let resolved_key = self.directory_entries[slot as usize].block_key;
        self.kv_block_resident(resolved_key)
    }

    /// C: `SparkRingWorkControlKvDirectoryRelease` (static).
    fn kv_directory_release(&mut self, sequence_id: u64, logical_block_index: u32) -> Result<()> {
        let Some(slot) = self.find_directory(sequence_id, logical_block_index) else {
            return Err(WorkControlError::NotFound);
        };
        let block_key = self.directory_entries[slot as usize].block_key;
        self.kv_block_deref(block_key)?;
        self.directory_entry_count -= 1;
        kv_index_erase(&mut self.directory_entries, slot);
        Ok(())
    }

    /// C: `SparkRingWorkControlValidateKvState` (static). C re-checks every
    /// pointer and capacity; the owned-`Vec` invariants are established by
    /// [`KvState::new`], so what remains is the capacity/counter consistency
    /// and the packet/state cross-checks.
    fn validate_kv_state(
        &self,
        packet: &WorkControlPacket,
        config: &WorkControlConfig,
    ) -> Result<()> {
        if self.abi_version != ABI_VERSION
            || self.descriptor_bytes != KV_STATE_DESCRIPTOR_BYTES
            || self.lane_capacity == 0
            || self.lane_stride == 0
            || self.block_token_count == 0
            || u64::from(self.table_entry_capacity)
                != u64::from(self.lane_capacity) * u64::from(self.lane_stride)
            || self.physical_block_capacity == 0
            || self.physical_block_capacity > u32::MAX / 2
            || self.directory_capacity < self.physical_block_capacity * 2
            || (self.directory_capacity & (self.directory_capacity - 1)) != 0
            || self.block_entry_capacity < self.physical_block_capacity * 2
            || (self.block_entry_capacity & (self.block_entry_capacity - 1)) != 0
        {
            return Err(WorkControlError::InvalidArgument);
        }
        if (self.backing_block_capacity == 0) != self.backing_block_free_next.is_none()
            || (self.backing_block_capacity == 0) != self.swap_hooks.is_none()
            || self.backing_block_capacity > self.block_entry_capacity / 2
            || self.block_entry_count
                > if self.backing_block_capacity != 0 {
                    self.backing_block_capacity
                } else {
                    self.physical_block_capacity
                }
            || self.directory_entry_count > self.directory_capacity / 2
            || self.allocated_physical_block_count > self.physical_block_capacity
            || self.swapped_block_count > self.block_entry_count
        {
            return Err(WorkControlError::InvalidArgument);
        }
        validate_packet(packet, config, self.lane_capacity, u32::MAX)
            .map_err(|_| WorkControlError::InvalidArgument)?;
        if packet.block_token_count != self.block_token_count
            || packet.max_blocks_per_sequence > self.lane_stride
            || packet.active_sequence_count > self.lane_capacity
        {
            return Err(WorkControlError::CapacityExceeded);
        }
        Ok(())
    }

    /// C: `SparkRingWorkControlCollectKvPrefetchEntries`. Deviation: on
    /// `CAPACITY_EXCEEDED` the partial entry list is discarded (C returns it
    /// through an out-parameter); callers that need partial results should
    /// size `entry_capacity` generously.
    pub fn collect_kv_prefetch_entries(
        &mut self,
        packets: &[WorkControlPacket],
        config: &WorkControlConfig,
        entry_capacity: u32,
    ) -> Result<Vec<KvPrefetchEntry>> {
        if packets.is_empty() || entry_capacity == 0 {
            return Err(WorkControlError::InvalidArgument);
        }
        let mut entries: Vec<KvPrefetchEntry> = Vec::new();
        self.prefetch_generation = self.prefetch_generation.wrapping_add(1);
        let mark = self.prefetch_generation & 0x1FFF_FFFF;
        for packet in packets {
            self.validate_kv_state(packet, config)?;
            if packet.control_generation != self.control_generation {
                continue;
            }
            if (packet.flags & FLAG_RELEASE_SEQUENCES) != 0 {
                continue;
            }
            for lane_index in 0..packet.active_sequence_count as usize {
                let lane = &packet.lanes[lane_index];
                let count = block_count(lane.context_token_count, packet.block_token_count);
                for block_index in 0..count {
                    let Some(block_slot) = self.kv_sequence_block(lane.sequence_id, block_index)
                    else {
                        continue;
                    };
                    // Sharers reach one record many times. Marking the record
                    // dedupes in constant time instead of rescanning what has
                    // been emitted.
                    let entry = &mut self.block_entries[block_slot as usize];
                    if entry.residency_state != KV_DIRECTORY_RESIDENCY_NVME
                        || !entry.backing_valid
                        || entry.prefetch_mark == mark
                    {
                        continue;
                    }
                    entry.prefetch_mark = mark;
                    if entries.len() as u32 >= entry_capacity {
                        return Err(WorkControlError::CapacityExceeded);
                    }
                    entries.push(KvPrefetchEntry {
                        key: entry.key,
                        backing_block_index: entry.backing_block_index,
                    });
                }
            }
        }
        Ok(entries)
    }

    /// C: `SparkRingWorkControlKvLaneKeys` (static). The lane's published
    /// key row, or `None` when this lane cannot share. Returns an owned copy
    /// so the row can live across the `&mut self` directory calls.
    fn kv_lane_keys(&self, lane_index: u32, count: u32) -> Option<Vec<KvKey>> {
        let keys = self.lane_block_keys.as_ref()?;
        if count > self.lane_block_key_stride {
            return None;
        }
        let start = lane_index as usize * self.lane_block_key_stride as usize;
        keys.get(start..start + count as usize).map(<[KvKey]>::to_vec)
    }

    /// C: `SparkRingWorkControlKvCountAdmission` (static). Count what this
    /// packet would add, before anything is mutated. Directory entries are
    /// per sequence slot; physical blocks are per distinct block key, so
    /// sequences presenting a prefix that is already admitted cost no
    /// physical block at all. Lanes presenting the same brand new key in one
    /// packet are counted once each, which over-states the need and can only
    /// reject conservatively.
    fn kv_count_admission(&self, packet: &WorkControlPacket) -> Result<(u32, u32)> {
        let mut new_entry_count = 0u32;
        let mut new_block_count = 0u32;
        for lane_index in 0..packet.active_sequence_count {
            let lane = &packet.lanes[lane_index as usize];
            let count = block_count(lane.context_token_count, packet.block_token_count);
            if count == 0
                || count > packet.max_blocks_per_sequence
                || count > self.physical_block_capacity
            {
                return Err(WorkControlError::CapacityExceeded);
            }
            let lane_keys = self.kv_lane_keys(lane_index, count);
            let committed_frontier = kv_committed_frontier(lane);
            let mut block_end_token = 0u64;
            for block_index in 0..count {
                block_end_token += u64::from(packet.block_token_count);
                if self.find_directory(lane.sequence_id, block_index).is_none() {
                    new_entry_count += 1;
                }
                if self
                    .find_block(kv_select_key(
                        lane_keys.as_deref(),
                        lane.sequence_id,
                        block_index,
                        block_end_token,
                        committed_frontier,
                    ))
                    .is_none()
                {
                    new_block_count += 1;
                }
            }
        }
        Ok((new_entry_count, new_block_count))
    }

    /// C: `SparkRingWorkControlKvAdmitPacket` (static). Reject a packet that
    /// cannot fit before anything is mutated. The cheap bound costs one
    /// multiply per lane and admits every packet that would fit even with no
    /// sharing at all; the probing count runs only when that bound is
    /// exceeded, which is exactly when sharing has to be measured to know
    /// whether it fits.
    fn kv_admit_packet(&self, packet: &WorkControlPacket) -> Result<()> {
        let mut total_block_count = 0u32;
        for lane_index in 0..packet.active_sequence_count {
            let lane = &packet.lanes[lane_index as usize];
            let count = block_count(lane.context_token_count, packet.block_token_count);
            if count == 0 || count > packet.max_blocks_per_sequence {
                return Err(WorkControlError::CapacityExceeded);
            }
            total_block_count += count;
        }
        if self.directory_entry_count + total_block_count <= self.directory_capacity / 2
            && total_block_count <= self.physical_block_capacity
        {
            return Ok(());
        }
        let (new_entry_count, new_block_count) = self.kv_count_admission(packet)?;
        if self.directory_entry_count > self.directory_capacity / 2
            || new_entry_count > (self.directory_capacity / 2) - self.directory_entry_count
            || new_block_count > self.physical_block_capacity
        {
            return Err(WorkControlError::CapacityExceeded);
        }
        Ok(())
    }

    /// C: `SparkRingWorkControlKvResolveEntryState` (static). Readiness
    /// state machine for one block of one lane. Prefill may start a missing
    /// block; decode requires every block but the last to be resident
    /// already.
    fn kv_resolve_entry_state(
        &mut self,
        packet: &WorkControlPacket,
        block_index: u32,
        count: u32,
        entry_state: &mut u8,
    ) -> Result<()> {
        if (packet.flags & FLAG_PREFILL) != 0 {
            if *entry_state == KV_ENTRY_MISSING {
                *entry_state = KV_ENTRY_IN_FLIGHT;
            }
            return Ok(());
        }
        if block_index + 1 < count && *entry_state != KV_ENTRY_RESIDENT {
            self.account_readiness(*entry_state);
            return Err(WorkControlError::Busy);
        }
        if block_index + 1 == count && *entry_state == KV_ENTRY_MISSING {
            *entry_state = KV_ENTRY_IN_FLIGHT;
        }
        Ok(())
    }

    /// C: `SparkRingWorkControlKvBuildLane` (static).
    fn kv_build_lane(&mut self, packet: &WorkControlPacket, lane_index: u32) -> Result<()> {
        let lane = &packet.lanes[lane_index as usize];
        let count = block_count(lane.context_token_count, packet.block_token_count);
        if count == 0 || count > packet.max_blocks_per_sequence {
            return Err(WorkControlError::CapacityExceeded);
        }
        let base_block_index = u64::from(lane_index) * u64::from(self.lane_stride);
        if base_block_index + u64::from(count) > u64::from(self.table_entry_capacity) {
            return Err(WorkControlError::CapacityExceeded);
        }
        let lane_keys = self.kv_lane_keys(lane_index, count);
        let committed_frontier = kv_committed_frontier(lane);
        self.lane_physical_block_counts[lane_index as usize] = count;
        let mut block_end_token = 0u64;
        for block_index in 0..count {
            block_end_token += u64::from(packet.block_token_count);
            let key = kv_select_key(
                lane_keys.as_deref(),
                lane.sequence_id,
                block_index,
                block_end_token,
                committed_frontier,
            );
            let physical_block_index =
                self.kv_directory_acquire(lane.sequence_id, block_index, key)?;
            let mut entry_state = self.physical_block_states[physical_block_index as usize];
            self.kv_resolve_entry_state(packet, block_index, count, &mut entry_state)?;
            self.physical_block_states[physical_block_index as usize] = entry_state;
            self.account_readiness(entry_state);
            self.physical_block_indices[(base_block_index + u64::from(block_index)) as usize] =
                physical_block_index;
        }
        Ok(())
    }

    /// C: `SparkRingWorkControlMarkTable` (static).
    fn mark_table(
        &mut self,
        packet: &WorkControlPacket,
        config: &WorkControlConfig,
        entry_state: u8,
    ) -> Result<()> {
        self.validate_kv_state(packet, config)?;
        if packet.control_generation != self.control_generation {
            return Err(WorkControlError::NotFound);
        }
        for lane_index in 0..packet.active_sequence_count as usize {
            let lane = &packet.lanes[lane_index];
            let count = block_count(lane.context_token_count, packet.block_token_count);
            if count == 0 || count > packet.max_blocks_per_sequence {
                return Err(WorkControlError::CapacityExceeded);
            }
            for block_index in 0..count {
                let block_slot = self
                    .kv_sequence_block(lane.sequence_id, block_index)
                    .ok_or(WorkControlError::NotFound)?;
                let physical_block_index =
                    self.block_entries[block_slot as usize].physical_block_index;
                // A block the table is asked to mark must be GPU-resident;
                // a swapped block here would mean a caller raced the mark
                // against eviction (an out-of-bounds write in C).
                if physical_block_index == KV_INVALID_BLOCK_INDEX {
                    return Err(WorkControlError::InternalError);
                }
                self.physical_block_states[physical_block_index as usize] = entry_state;
                self.physical_block_last_used_epochs[physical_block_index as usize] = self.epoch;
            }
        }
        Ok(())
    }

    /// C: `SparkRingWorkControlBuildHostKvBlockTable`.
    pub fn build_host_kv_block_table(
        &mut self,
        packet: &WorkControlPacket,
        config: &WorkControlConfig,
    ) -> Result<KvBlockTableView<'_>> {
        self.validate_kv_state(packet, config)?;
        self.select_kv_generation(packet)?;
        self.kv_admit_packet(packet)?;
        for count in
            self.lane_physical_block_counts[..packet.active_sequence_count as usize].iter_mut()
        {
            *count = 0;
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.epoch = 1;
        }
        self.reset_readiness_counts();
        for lane_index in 0..packet.active_sequence_count {
            self.kv_build_lane(packet, lane_index)?;
        }
        // C: `SparkRingWorkControlKvFillBlockTableView` (static).
        Ok(KvBlockTableView {
            abi_version: KV_CACHE_ABI_VERSION,
            descriptor_bytes: KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES,
            block_token_count: packet.block_token_count,
            lane_count: packet.active_sequence_count,
            lane_stride: self.lane_stride,
            lane_capacity: self.lane_stride,
            physical_block_indices: &self.physical_block_indices,
            lane_physical_block_counts: &self.lane_physical_block_counts,
            host_physical_block_indices: &self.physical_block_indices,
            host_lane_physical_block_counts: &self.lane_physical_block_counts,
        })
    }

    /// C: `SparkRingWorkControlCommitHostKvBlockTable`.
    pub fn commit_host_kv_block_table(
        &mut self,
        packet: &WorkControlPacket,
        config: &WorkControlConfig,
    ) -> Result<()> {
        self.mark_table(packet, config, KV_ENTRY_RESIDENT)?;
        let written_position_count = written_position_count(packet);
        if written_position_count == 0 {
            return Err(WorkControlError::InvalidArgument);
        }
        for lane_index in 0..packet.active_sequence_count as usize {
            let lane = &packet.lanes[lane_index];
            let write_end_token = lane.sequence_position + u64::from(written_position_count) - 1;
            if write_end_token >= u64::from(lane.context_token_count) {
                return Err(WorkControlError::InvalidArgument);
            }
            let first_written_block =
                (lane.sequence_position / u64::from(packet.block_token_count)) as u32;
            let last_written_block = (write_end_token / u64::from(packet.block_token_count)) as u32;
            for block_index in first_written_block..=last_written_block {
                let block_slot = self
                    .kv_sequence_block(lane.sequence_id, block_index)
                    .ok_or(WorkControlError::NotFound)?;
                self.block_entries[block_slot as usize].backing_valid = false;
            }
        }
        Ok(())
    }

    /// C: `SparkRingWorkControlCancelHostKvBlockTable`.
    pub fn cancel_host_kv_block_table(
        &mut self,
        packet: &WorkControlPacket,
        config: &WorkControlConfig,
    ) -> Result<()> {
        self.validate_kv_state(packet, config)?;
        if packet.control_generation != self.control_generation {
            return Err(WorkControlError::NotFound);
        }
        for lane_index in 0..packet.active_sequence_count as usize {
            let lane = &packet.lanes[lane_index];
            let count = block_count(lane.context_token_count, packet.block_token_count);
            if count == 0 || count > packet.max_blocks_per_sequence {
                return Err(WorkControlError::CapacityExceeded);
            }
            for block_index in 0..count {
                let Some(block_slot) = self.kv_sequence_block(lane.sequence_id, block_index) else {
                    continue;
                };
                let physical_block_index =
                    self.block_entries[block_slot as usize].physical_block_index;
                // Only in-flight GPU blocks are cancelled; a swapped block
                // has no physical slot to inspect (C reads the state array
                // unconditionally, which is only meaningful while resident).
                if physical_block_index != KV_INVALID_BLOCK_INDEX
                    && self.physical_block_states[physical_block_index as usize]
                        == KV_ENTRY_IN_FLIGHT
                {
                    self.kv_directory_release(lane.sequence_id, block_index)?;
                }
            }
        }
        Ok(())
    }

    /// C: `SparkRingWorkControlReleaseSequence`.
    pub fn release_sequence(&mut self, sequence_id: u64, logical_block_count: u32) -> Result<()> {
        if sequence_id == 0
            || logical_block_count == 0
            || self.abi_version != ABI_VERSION
            || self.descriptor_bytes != KV_STATE_DESCRIPTOR_BYTES
            || logical_block_count > self.lane_stride
        {
            return Err(WorkControlError::InvalidArgument);
        }
        for logical_block_index in 0..logical_block_count {
            match self.kv_directory_release(sequence_id, logical_block_index) {
                Ok(()) | Err(WorkControlError::NotFound) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    /// C: `SparkRingWorkControlReleasePacketSequences`.
    pub fn release_packet_sequences(
        &mut self,
        packet: &WorkControlPacket,
        config: &WorkControlConfig,
    ) -> Result<()> {
        self.validate_kv_state(packet, config)?;
        if (packet.flags & FLAG_RELEASE_SEQUENCES) == 0 {
            return Err(WorkControlError::InvalidArgument);
        }
        if packet.control_generation < self.control_generation {
            return Ok(());
        }
        self.select_kv_generation(packet)?;
        for lane_index in 0..packet.lane_count as usize {
            let lane = &packet.lanes[lane_index];
            let logical_block_count =
                block_count(lane.context_token_count, packet.block_token_count);
            self.release_sequence(lane.sequence_id, logical_block_count)?;
        }
        Ok(())
    }
}

/// C: `SparkRingWorkControlKvSelectKey` (static). Content key when the
/// caller published one for a committed block, private key otherwise. The
/// lane's key row and frontier are hoisted by the caller, so the per-block
/// cost here is one compare.
fn kv_select_key(
    lane_keys: Option<&[KvKey]>,
    sequence_id: u64,
    logical_block_index: u32,
    block_end_token: u64,
    committed_frontier: u64,
) -> KvKey {
    if let Some(keys) = lane_keys {
        if block_end_token <= committed_frontier && !keys[logical_block_index as usize].is_empty() {
            let key = keys[logical_block_index as usize];
            return content_key(key.low, key.high);
        }
    }
    private_key(sequence_id, logical_block_index)
}
