//! Batch-plane sequence table — port of `serving/spark_batch_sequence_table.c`
//! (API: `include/sparkpipe/spark_batch_sequence_table.h`).
//!
//! Admission and exchange lifecycle for longmem-style workloads: sequences of
//! B8 shared-prefix exchanges. This is the batch plane's own population
//! structure, separate from the 1024-slot frame API; the table is the authority
//! for the active count that drives the expert-queue firing threshold
//! (`active x lanes x topk / experts`), and pausing a sequence for a tool call
//! is the signal that its KV fragments become demotable in the JIT pool.
//!
//! Handles returned by [`BatchSequenceTable::admit`] tag the slot index (low
//! 14 bits) with a generation (high 18 bits) that increments when the slot is
//! recycled, so a stale handle held past [`BatchSequenceTable::complete`] is
//! refused instead of silently acting on the slot's next occupant.

/// Configuration ABI version (SPARK_BATCH_SEQUENCE_ABI_VERSION).
pub const ABI_VERSION: u32 = 2;
/// Hard ceiling on table capacity (SPARK_BATCH_SEQUENCE_MAX_SEQUENCES).
pub const MAX_SEQUENCES: u32 = 16384;
/// Handle low bits carrying the slot index
/// (SPARK_BATCH_SEQUENCE_HANDLE_INDEX_BITS).
pub const HANDLE_INDEX_BITS: u32 = 14;
/// Mask for the slot index inside a handle
/// (SPARK_BATCH_SEQUENCE_HANDLE_INDEX_MASK).
pub const HANDLE_INDEX_MASK: u32 = (1 << HANDLE_INDEX_BITS) - 1;

const FREE_HEAD_NONE: u32 = u32::MAX;

/// Sequence lifecycle states (SPARK_BATCH_SEQUENCE_STATE_*).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum SequenceState {
    #[default]
    Free = 0,
    Active = 1,
    AwaitingTool = 2,
    Complete = 3,
}

/// Generation-tagged handle to an admitted sequence. The raw encoding matches
/// the C exactly: slot index in the low [`HANDLE_INDEX_BITS`] bits, generation
/// above them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceHandle(u32);

impl SequenceHandle {
    /// Raw u32 encoding, identical to the C handle value.
    pub fn raw(self) -> u32 {
        self.0
    }

    /// Slot index carried by the handle (low [`HANDLE_INDEX_BITS`] bits).
    pub fn index(self) -> u32 {
        self.0 & HANDLE_INDEX_MASK
    }

    /// Generation carried by the handle (high bits).
    pub fn generation(self) -> u32 {
        self.0 >> HANDLE_INDEX_BITS
    }
}

/// One table slot (SparkBatchSequence). Fields are read-only to callers; the
/// table mutates them internally.
#[derive(Debug, Clone, Default)]
pub struct BatchSequence {
    pub sequence_id: u64,
    pub state: SequenceState,
    pub lane_count: u32,
    pub exchange_number: u32,
    pub context_tokens: u32,
    pub fragment_base: u32,
    pub fragment_count: u32,
    free_next: u32,
    generation: u32,
}

/// Errors mirroring the `SparkStatus` codes the C functions return.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BatchSequenceError {
    /// SPARK_STATUS_INVALID_ARGUMENT: bad configuration, zero fragment count,
    /// or a state transition the sequence's current state forbids.
    #[error("invalid argument")]
    InvalidArgument,
    /// SPARK_STATUS_NOT_FOUND: the handle does not resolve (out of range or
    /// stale generation).
    #[error("sequence not found (stale or out-of-range handle)")]
    NotFound,
    /// SPARK_STATUS_CAPACITY_EXCEEDED: every slot is in use.
    #[error("sequence capacity exceeded")]
    CapacityExceeded,
}

/// Table configuration (SparkBatchSequenceTableConfiguration).
#[derive(Debug, Clone, Copy)]
pub struct BatchSequenceTableConfiguration {
    pub abi_version: u32,
    pub sequence_capacity: u32,
    pub lane_count: u32,
}

/// Batch-plane sequence population (SparkBatchSequenceTable).
#[derive(Debug)]
pub struct BatchSequenceTable {
    abi_version: u32,
    sequence_capacity: u32,
    lane_count: u32,
    active_count: u32,
    awaiting_tool_count: u32,
    complete_count: u32,
    free_head: u32,
    free_high_water: u32,
    exchange_count: u64,
    sequences: Vec<BatchSequence>,
}

impl BatchSequenceTable {
    /// Initialize a table (SparkBatchSequenceTableInitialize). The slot
    /// storage is a `Vec` instead of C's fixed array, sized to the configured
    /// capacity (bounded by [`MAX_SEQUENCES`]).
    pub fn new(
        configuration: &BatchSequenceTableConfiguration,
    ) -> Result<Self, BatchSequenceError> {
        if configuration.abi_version != ABI_VERSION
            || configuration.sequence_capacity == 0
            || configuration.sequence_capacity > MAX_SEQUENCES
            || configuration.lane_count == 0
        {
            return Err(BatchSequenceError::InvalidArgument);
        }
        Ok(Self {
            abi_version: ABI_VERSION,
            sequence_capacity: configuration.sequence_capacity,
            lane_count: configuration.lane_count,
            active_count: 0,
            awaiting_tool_count: 0,
            complete_count: 0,
            free_head: FREE_HEAD_NONE,
            free_high_water: 0,
            exchange_count: 0,
            sequences: vec![BatchSequence::default(); configuration.sequence_capacity as usize],
        })
    }

    /// Resolve a handle to its slot index. Mirrors the C: the index must be in
    /// range and the slot's generation must match the handle's generation,
    /// otherwise the handle is stale (SPARK_STATUS_NOT_FOUND at call sites).
    fn resolve_handle(&self, handle: SequenceHandle) -> Result<u32, BatchSequenceError> {
        let index = handle.index();
        if index >= self.sequence_capacity
            || self.sequences[index as usize].generation != handle.generation()
        {
            return Err(BatchSequenceError::NotFound);
        }
        Ok(index)
    }

    /// Admit a sequence (SparkBatchSequenceTableAdmit). Recycled slots come
    /// off the free list (LIFO) before fresh slots are taken from the high
    /// water mark.
    pub fn admit(
        &mut self,
        sequence_id: u64,
        context_tokens: u32,
        fragment_base: u32,
        fragment_count: u32,
    ) -> Result<SequenceHandle, BatchSequenceError> {
        if fragment_count == 0 {
            return Err(BatchSequenceError::InvalidArgument);
        }
        let index = if self.free_head != FREE_HEAD_NONE {
            let index = self.free_head;
            self.free_head = self.sequences[index as usize].free_next;
            index
        } else if self.free_high_water < self.sequence_capacity {
            let index = self.free_high_water;
            self.free_high_water += 1;
            index
        } else {
            return Err(BatchSequenceError::CapacityExceeded);
        };
        let sequence = &mut self.sequences[index as usize];
        sequence.sequence_id = sequence_id;
        sequence.state = SequenceState::Active;
        sequence.lane_count = self.lane_count;
        sequence.exchange_number = 0;
        sequence.context_tokens = context_tokens;
        sequence.fragment_base = fragment_base;
        sequence.fragment_count = fragment_count;
        self.active_count += 1;
        self.exchange_count += 1;
        Ok(SequenceHandle(index | (sequence.generation << HANDLE_INDEX_BITS)))
    }

    /// Begin a new exchange on a sequence paused for a tool call
    /// (SparkBatchSequenceTableBeginExchange). Only valid from AWAITING_TOOL.
    pub fn begin_exchange(
        &mut self,
        handle: SequenceHandle,
        appended_context_tokens: u32,
    ) -> Result<(), BatchSequenceError> {
        let index = self.resolve_handle(handle)?;
        let sequence = &mut self.sequences[index as usize];
        if sequence.state != SequenceState::AwaitingTool {
            return Err(BatchSequenceError::InvalidArgument);
        }
        sequence.state = SequenceState::Active;
        sequence.exchange_number += 1;
        sequence.context_tokens = sequence.context_tokens.wrapping_add(appended_context_tokens);
        self.awaiting_tool_count -= 1;
        self.active_count += 1;
        self.exchange_count += 1;
        Ok(())
    }

    /// Pause a sequence for a tool call (SparkBatchSequenceTablePauseForTool).
    /// Only valid from ACTIVE.
    pub fn pause_for_tool(&mut self, handle: SequenceHandle) -> Result<(), BatchSequenceError> {
        let index = self.resolve_handle(handle)?;
        let sequence = &mut self.sequences[index as usize];
        if sequence.state != SequenceState::Active {
            return Err(BatchSequenceError::InvalidArgument);
        }
        sequence.state = SequenceState::AwaitingTool;
        self.active_count -= 1;
        self.awaiting_tool_count += 1;
        Ok(())
    }

    /// Complete a sequence and recycle its slot
    /// (SparkBatchSequenceTableComplete). Valid from ACTIVE or AWAITING_TOOL.
    /// The slot's generation increments, so every outstanding handle to it
    /// becomes stale.
    pub fn complete(&mut self, handle: SequenceHandle) -> Result<(), BatchSequenceError> {
        let index = self.resolve_handle(handle)?;
        let sequence = &mut self.sequences[index as usize];
        if sequence.state != SequenceState::Active && sequence.state != SequenceState::AwaitingTool
        {
            return Err(BatchSequenceError::InvalidArgument);
        }
        if sequence.state == SequenceState::Active {
            self.active_count -= 1;
        } else {
            self.awaiting_tool_count -= 1;
        }
        sequence.state = SequenceState::Complete;
        sequence.generation = sequence.generation.wrapping_add(1);
        sequence.free_next = self.free_head;
        self.free_head = index;
        self.complete_count += 1;
        Ok(())
    }

    /// Expert-queue firing threshold (SparkBatchSequenceTableFiringThreshold):
    /// `active x lanes x topk / experts`, floored at 1, capped at
    /// `threshold_cap`. Zero `expert_count` or `threshold_cap` yields 1, same
    /// as the C. Multiplication wraps like C uint32 arithmetic.
    pub fn firing_threshold(&self, topk: u32, expert_count: u32, threshold_cap: u32) -> u32 {
        if expert_count == 0 || threshold_cap == 0 {
            return 1;
        }
        let product = self.active_count.wrapping_mul(self.lane_count).wrapping_mul(topk);
        let mut threshold = product / expert_count;
        if threshold == 0 {
            threshold = 1;
        }
        threshold.min(threshold_cap)
    }

    /// Read-only view of a slot by handle. Fails on a stale handle, same as
    /// the mutating entry points.
    pub fn sequence(&self, handle: SequenceHandle) -> Result<&BatchSequence, BatchSequenceError> {
        let index = self.resolve_handle(handle)?;
        Ok(&self.sequences[index as usize])
    }

    pub fn abi_version(&self) -> u32 {
        self.abi_version
    }

    pub fn sequence_capacity(&self) -> u32 {
        self.sequence_capacity
    }

    pub fn lane_count(&self) -> u32 {
        self.lane_count
    }

    pub fn active_count(&self) -> u32 {
        self.active_count
    }

    pub fn awaiting_tool_count(&self) -> u32 {
        self.awaiting_tool_count
    }

    pub fn complete_count(&self) -> u32 {
        self.complete_count
    }

    pub fn exchange_count(&self) -> u64 {
        self.exchange_count
    }
}
