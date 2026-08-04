//! Port of `tests/test_glm52_row_allocator.c` from the C tree, covering the
//! deterministic dispatch policy bit-exactly: base-row priority, probe grant
//! guarantee, marginal-value ordering, saturation convergence to all-real,
//! and lower-index tie breaking.

use spark_core::row_allocator::{row_allocator_assign, SlotInput};

fn slot(commit_ema_milli: u32, maximum_draft_depth: u32, probe: u32) -> SlotInput {
    SlotInput { commit_ema_milli, maximum_draft_depth, probe }
}

// A saturated wave converges to all-real: with as many slots as rows, every
// slot gets its base row and zero draft rows, regardless of how strong the
// acceptance signal is. Speculation never displaces a queued real row.
#[test]
fn saturated_wave_is_all_real() {
    let slots = [slot(2900, 5, 0); 8];
    let mut budgets = [0u32; 8];
    let total = row_allocator_assign(&slots, 8, 1000, &mut budgets);
    assert_eq!(total, 8);
    assert_eq!(budgets, [0; 8]);
}

// An undersubscribed wave fills with speculation, deepest on the strongest
// lane, and the total never exceeds the cap.
//
// Exact greedy trace (milli fixed point): slot0 alpha = (2900-1000)*1000/2900
// = 655, depth values 655, 429, 281, 184, 120; slot1 alpha = 300*1000/1300 =
// 230, values 230, 53. Grant order for the 6 spare rows: 655, 429, 281, 230,
// 184, 120 -> slot0 gets 5, slot1 gets 1.
#[test]
fn undersubscribed_fills_by_alpha() {
    let slots = [slot(2900, 5, 0), slot(1300, 5, 0)];
    let mut budgets = [0u32; 2];
    let total = row_allocator_assign(&slots, 8, 1000, &mut budgets);
    assert!(total <= 8);
    assert!(budgets[0] > budgets[1]);
    assert_eq!(budgets[0], 5);
    assert_eq!(budgets[1], 1);
    assert_eq!(total, 8);
}

// Suppressed slots (maximum depth zero) receive only their base row even when
// capacity is abundant.
#[test]
fn suppressed_slot_stays_plain() {
    let slots = [slot(900, 0, 0), slot(2900, 5, 0)];
    let mut budgets = [0u32; 2];
    let total = row_allocator_assign(&slots, 16, 1000, &mut budgets);
    assert_eq!(budgets[0], 0);
    assert_eq!(budgets[1], 5);
    assert_eq!(total, 7);
}

// A probing slot is guaranteed its two-row probe before value-ranked grants
// and competes no further; the probe is capacity-bounded.
#[test]
fn probe_grant() {
    let slots = [slot(1000, 5, 1), slot(2900, 5, 0)];
    let mut budgets = [0u32; 2];
    let total = row_allocator_assign(&slots, 16, 1000, &mut budgets);
    assert_eq!(budgets[0], 2);
    assert_eq!(budgets[1], 5);
    assert_eq!(total, 9);

    // Tight capacity: base rows consume 2 of 3; the single remaining row goes
    // to the probe, which precedes value grants.
    let total = row_allocator_assign(&slots, 3, 1000, &mut budgets);
    assert_eq!(budgets[0], 1);
    assert_eq!(budgets[1], 0);
    assert_eq!(total, 3);
}

// EMA at or below one committed token per cycle yields no speculative grants:
// drafting has demonstrated no value on that lane.
#[test]
fn no_value_no_spec() {
    let slots = [slot(1000, 5, 0)];
    let mut budgets = [0u32; 1];
    let total = row_allocator_assign(&slots, 16, 1000, &mut budgets);
    assert_eq!(budgets[0], 0);
    assert_eq!(total, 1);
}

// Determinism and tie breaking: equal-alpha slots receive grants in lower
// slot-index order, so repeated runs produce identical assignments.
#[test]
fn deterministic_ties() {
    let slots = [slot(2000, 5, 0); 3];
    let mut budgets = [0u32; 3];
    let total = row_allocator_assign(&slots, 5, 1000, &mut budgets);
    assert_eq!(total, 5);
    assert_eq!(budgets, [1, 1, 0]);
    let reference = budgets;
    for _ in 0..4 {
        let total = row_allocator_assign(&slots, 5, 1000, &mut budgets);
        assert_eq!(total, 5);
        assert_eq!(budgets, reference);
    }
}

// More active slots than the cap: the surplus receives nothing and the total
// equals the cap exactly; admission upstream owns the overflow.
#[test]
fn overflow_clamps_to_cap() {
    let slots = [slot(2900, 5, 0); 4];
    let mut budgets = [0u32; 4];
    let total = row_allocator_assign(&slots, 2, 1000, &mut budgets);
    assert_eq!(total, 2);
    assert_eq!(budgets, [0; 4]);
}

// Cost weighting: when measured routing correlation makes spec rows cheaper
// than real rows, depth-1 spec scores can exceed 1000; the ranking still
// fills deterministically and the saturated all-real property is unaffected
// because base rows are granted before any scored comparison.
//
// At 60 percent cost the scores scale by 1000/600: slot0's depth-1 score is
// 655*1000/600 = 1091 (above a real row's 1000), with the deeper scores
// descending from there; slot1's depth-1 score is 230*1000/600 = 383. The
// grant order for the 6 spare rows yields the identical split to equal cost
// here (slot0 5, slot1 1), but scores above 1000 demonstrate the regime
// where spec outranks a marginal real row; saturation still yields all-real.
#[test]
fn cost_weighted_scores() {
    let slots = [slot(2900, 5, 0), slot(1300, 5, 0)];
    let mut budgets = [0u32; 2];
    let total = row_allocator_assign(&slots, 8, 600, &mut budgets);
    assert_eq!(total, 8);
    assert_eq!(budgets, [5, 1]);
    let total = row_allocator_assign(&slots, 2, 600, &mut budgets);
    assert_eq!(total, 2);
    assert_eq!(budgets, [0, 0]);
}

// A zero firing cap returns zero rows, matching the C early return.
#[test]
fn zero_cap_assigns_nothing() {
    let slots = [slot(2900, 5, 0); 2];
    let mut budgets = [7u32; 2];
    let total = row_allocator_assign(&slots, 0, 1000, &mut budgets);
    assert_eq!(total, 0);
}
