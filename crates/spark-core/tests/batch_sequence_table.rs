//! Port of the batch sequence table scenarios from the C tree's
//! `tests/test_glm52_batch_plane.c` (`SparkTestBatchSequenceTableLifecycleAndThreshold`),
//! plus validation and firing-threshold edge cases exercised against the C
//! entry points' documented status codes.

use spark_core::batch_sequence_table::{
    BatchSequenceError, BatchSequenceTable, BatchSequenceTableConfiguration, SequenceState,
    ABI_VERSION, HANDLE_INDEX_BITS, MAX_SEQUENCES,
};

fn configuration(sequence_capacity: u32, lane_count: u32) -> BatchSequenceTableConfiguration {
    BatchSequenceTableConfiguration { abi_version: ABI_VERSION, sequence_capacity, lane_count }
}

// Configuration validation mirrors SPARK_STATUS_INVALID_ARGUMENT from
// SparkBatchSequenceTableInitialize.
#[test]
fn initialize_rejects_invalid_configuration() {
    assert_eq!(
        BatchSequenceTable::new(&BatchSequenceTableConfiguration {
            abi_version: ABI_VERSION + 1,
            sequence_capacity: 4,
            lane_count: 8,
        })
        .unwrap_err(),
        BatchSequenceError::InvalidArgument
    );
    assert_eq!(
        BatchSequenceTable::new(&configuration(0, 8)).unwrap_err(),
        BatchSequenceError::InvalidArgument
    );
    assert_eq!(
        BatchSequenceTable::new(&configuration(MAX_SEQUENCES + 1, 8)).unwrap_err(),
        BatchSequenceError::InvalidArgument
    );
    assert_eq!(
        BatchSequenceTable::new(&configuration(4, 0)).unwrap_err(),
        BatchSequenceError::InvalidArgument
    );
    assert!(BatchSequenceTable::new(&configuration(MAX_SEQUENCES, 8)).is_ok());
}

// Admission tags the slot index with a generation; fragment_count == 0 is
// invalid. Mirrors the C Admit contract.
#[test]
fn admit_issues_generation_tagged_handles() {
    let mut table = BatchSequenceTable::new(&configuration(4, 8)).unwrap();
    assert_eq!(table.admit(900, 8192, 0, 0).unwrap_err(), BatchSequenceError::InvalidArgument);
    let first = table.admit(900, 8192, 0, 128).unwrap();
    let second = table.admit(901, 8192, 128, 128).unwrap();
    assert_eq!(first.index(), 0);
    assert_eq!(first.generation(), 0);
    assert_eq!(first.raw(), 0);
    assert_eq!(second.index(), 1);
    assert_eq!(table.active_count(), 2);
    assert_eq!(table.exchange_count(), 2);
    let sequence = table.sequence(first).unwrap();
    assert_eq!(sequence.sequence_id, 900);
    assert_eq!(sequence.state, SequenceState::Active);
    assert_eq!(sequence.lane_count, 8);
    assert_eq!(sequence.exchange_number, 0);
    assert_eq!(sequence.context_tokens, 8192);
    assert_eq!(sequence.fragment_base, 0);
    assert_eq!(sequence.fragment_count, 128);
}

// Direct port of SparkTestBatchSequenceTableLifecycleAndThreshold: capacity
// exhaustion, pause/begin-exchange/complete state transitions and counters,
// stale-handle refusal, and slot recycling under churn.
#[test]
fn lifecycle_and_threshold() {
    let mut table = BatchSequenceTable::new(&configuration(4, 8)).unwrap();
    let first_handle = table.admit(900, 8192, 0, 128).unwrap();
    let second_handle = table.admit(901, 8192, 128, 128).unwrap();
    assert_eq!(table.active_count(), 2);
    assert_eq!(table.firing_threshold(8, 256, 1024), 1);

    // Fill the table; the fifth admission exceeds capacity.
    for fill_index in 2..4u64 {
        table.admit(900 + fill_index, 8192, fill_index as u32 * 128, 128).unwrap();
    }
    assert_eq!(table.admit(999, 8192, 512, 128).unwrap_err(), BatchSequenceError::CapacityExceeded);
    assert_eq!(table.firing_threshold(8, 256, 1024), 1);

    // Pause: ACTIVE -> AWAITING_TOOL; a second pause is invalid.
    table.pause_for_tool(first_handle).unwrap();
    assert_eq!(table.active_count(), 3);
    assert_eq!(table.awaiting_tool_count(), 1);
    assert_eq!(
        table.pause_for_tool(first_handle).unwrap_err(),
        BatchSequenceError::InvalidArgument
    );

    // Begin exchange: AWAITING_TOOL -> ACTIVE, counting the exchange and
    // appending context tokens.
    table.begin_exchange(first_handle, 192).unwrap();
    let first = table.sequence(first_handle).unwrap();
    assert_eq!(first.exchange_number, 1);
    assert_eq!(first.context_tokens, 8384);
    assert_eq!(table.active_count(), 4);
    assert_eq!(table.exchange_count(), 5);

    // Begin exchange from ACTIVE is invalid (only AWAITING_TOOL resumes).
    assert_eq!(
        table.begin_exchange(first_handle, 64).unwrap_err(),
        BatchSequenceError::InvalidArgument
    );

    // Complete: ACTIVE -> COMPLETE; the slot's generation moves.
    table.complete(second_handle).unwrap();
    assert_eq!(table.active_count(), 3);
    assert_eq!(table.complete_count(), 1);

    // The stale handle now fails handle resolution, not just the state check:
    // the generation moved when the slot was freed, so a holdover handle can
    // never act on the slot's next occupant.
    assert_eq!(table.complete(second_handle).unwrap_err(), BatchSequenceError::NotFound);
    assert_eq!(table.pause_for_tool(second_handle).unwrap_err(), BatchSequenceError::NotFound);
    assert_eq!(table.sequence(second_handle).unwrap_err(), BatchSequenceError::NotFound);

    // The completed slot must be reclaimable: a capacity-4 table that has
    // seen completions keeps admitting under churn instead of leaking slots
    // forever. Each recycled handle shares the slot index but differs by
    // generation.
    let mut previous = second_handle;
    for churn_index in 0..64u64 {
        let recycled = table.admit(5000 + churn_index, 4089, 0, 64).unwrap();
        assert_eq!(recycled.index(), second_handle.index());
        assert_ne!(recycled, previous);
        previous = recycled;
        table.complete(recycled).unwrap();
    }
}

// Handle layout: index in the low 14 bits, generation above. After one
// recycle the handle is (index | 1 << 14).
#[test]
fn handle_layout_recycled_slot_bumps_generation() {
    let mut table = BatchSequenceTable::new(&configuration(2, 4)).unwrap();
    let handle = table.admit(7, 128, 0, 16).unwrap();
    table.complete(handle).unwrap();
    let recycled = table.admit(8, 128, 0, 16).unwrap();
    assert_eq!(recycled.index(), handle.index());
    assert_eq!(recycled.generation(), 1);
    assert_eq!(recycled.raw(), handle.index() | (1 << HANDLE_INDEX_BITS));
    // The original handle is stale even though its slot is occupied again.
    assert_eq!(table.pause_for_tool(handle).unwrap_err(), BatchSequenceError::NotFound);
}

// Complete is also legal from AWAITING_TOOL and debits the awaiting count.
#[test]
fn complete_from_awaiting_tool() {
    let mut table = BatchSequenceTable::new(&configuration(2, 4)).unwrap();
    let handle = table.admit(1, 64, 0, 8).unwrap();
    table.pause_for_tool(handle).unwrap();
    assert_eq!(table.awaiting_tool_count(), 1);
    table.complete(handle).unwrap();
    assert_eq!(table.awaiting_tool_count(), 0);
    assert_eq!(table.complete_count(), 1);
    assert_eq!(table.sequence(handle).unwrap_err(), BatchSequenceError::NotFound);
}

// Firing threshold formula: active x lanes x topk / experts, floored at 1,
// capped at threshold_cap; zero experts or zero cap yield 1.
#[test]
fn firing_threshold_formula() {
    let mut table = BatchSequenceTable::new(&configuration(64, 8)).unwrap();
    // No active sequences: product is zero, floored to 1.
    assert_eq!(table.firing_threshold(8, 256, 1024), 1);
    // Degenerate arguments yield 1, as in the C.
    assert_eq!(table.firing_threshold(8, 0, 1024), 1);
    assert_eq!(table.firing_threshold(8, 256, 0), 1);

    let mut handles = Vec::new();
    for index in 0..64u64 {
        handles.push(table.admit(100 + index, 4096, 0, 64).unwrap());
    }
    // 64 active x 8 lanes x 8 topk / 256 experts = 16.
    assert_eq!(table.firing_threshold(8, 256, 1024), 16);
    // Capped.
    assert_eq!(table.firing_threshold(8, 256, 10), 10);
    // Division truncates: 64 x 8 x 3 / 256 = 1536 / 256 = 6.
    assert_eq!(table.firing_threshold(3, 256, 1024), 6);
    // Paused sequences leave the active count, so the threshold drops.
    table.pause_for_tool(handles[0]).unwrap();
    // 63 x 8 x 8 / 256 = 4032 / 256 = 15 (truncated).
    assert_eq!(table.firing_threshold(8, 256, 1024), 15);
}
