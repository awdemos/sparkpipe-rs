//! Global firing-row allocator — port of `serving/spark_row_allocator.c`
//! (API: `include/sparkpipe/spark_row_allocator.h`).
//!
//! Divides a wave's row budget between real rows (one per active slot,
//! marginal expected commit exactly one token) and speculative draft rows
//! (marginal expected commit alpha^depth, the chain survival probability).
//! Real rows always dominate speculative rows of equal cost, so every active
//! slot receives its base row first; remaining capacity goes to draft rows in
//! descending marginal value, which makes the plane self-balancing: a
//! saturated wave converges to all-real and an undersubscribed wave fills
//! with deep speculation on the lanes whose measured acceptance justifies it.
//!
//! Per-slot alpha is derived from the commit EMA the request API already
//! maintains: for chain drafting the expected committed tokens per cycle is
//! the geometric sum (1-alpha^(k+1))/(1-alpha), and alpha = (ema-1000)/ema in
//! milli fixed point inverts it to within a few parts per thousand across the
//! operating range. All arithmetic is milli fixed point matching the C
//! bit-exactly (the assignment order is deterministic dispatch policy).

const MILLI: u32 = 1000;
const ALPHA_CEILING: u32 = 999;
const PROBE_ROWS: u32 = 2;

/// Per-slot allocator input (SparkRowAllocatorSlotInput).
#[derive(Debug, Clone, Copy, Default)]
pub struct SlotInput {
    /// EMA of committed tokens per verify cycle, times 1000 (the request
    /// API's mtp_commit_ema_milli). Values at or below 1000 imply no
    /// speculative value.
    pub commit_ema_milli: u32,
    /// Maximum draft rows this slot may receive. Zero for suppressed slots
    /// whose reprobe countdown has not elapsed; they receive only their base
    /// row.
    pub maximum_draft_depth: u32,
    /// Nonzero when the slot's reprobe countdown has elapsed: the slot is
    /// guaranteed a two-row probe grant (capacity permitting) so the EMA can
    /// observe beyond-first acceptance, and competes no further that wave.
    pub probe: u32,
}

/// alpha = (ema - 1000) / ema in milli fixed point, clamped to [0, 999]. An
/// EMA at or below one committed token per cycle means drafting has shown no
/// value, so alpha is zero and the slot's speculative rows never win a grant.
fn alpha_milli(commit_ema_milli: u32) -> u32 {
    if commit_ema_milli <= MILLI {
        return 0;
    }
    let alpha = ((u64::from(commit_ema_milli - MILLI) * u64::from(MILLI))
        / u64::from(commit_ema_milli)) as u32;
    alpha.min(ALPHA_CEILING)
}

/// Grant probing slots their fixed two-row probe, in slot-index order, so the
/// EMA can observe beyond-first acceptance at minimal cost. Probing slots do
/// not compete in the value-ranked phase. Returns the ungranted remainder.
fn grant_probes(slots: &[SlotInput], mut remaining: u32, draft_budgets_out: &mut [u32]) -> u32 {
    for (slot, budget) in slots.iter().zip(draft_budgets_out.iter_mut()) {
        if remaining == 0 {
            break;
        }
        if slot.probe == 0 || slot.maximum_draft_depth == 0 {
            continue;
        }
        let grant = PROBE_ROWS.min(slot.maximum_draft_depth).min(remaining);
        *budget = grant;
        remaining -= grant;
    }
    remaining
}

/// Exact greedy: repeatedly grant the single highest-marginal-value draft row
/// among non-probing slots, lower slot index winning ties, until capacity or
/// value is exhausted. The marginal value of a slot's next draft row at
/// depth d is alpha^d in milli fixed point, recomputed per scan from the
/// slot's current grant count to avoid scratch state, exactly as the C. The
/// scan is O(grants x slots); at wave cadence with bounded depth this is
/// negligible, and a threshold binary search is the drop-in replacement if it
/// ever profiles.
fn greedy_grant(
    slots: &[SlotInput],
    mut remaining: u32,
    spec_row_relative_cost_milli: u32,
    draft_budgets_out: &mut [u32],
) {
    let slot_count = slots.len();
    let cost = if spec_row_relative_cost_milli == 0 { MILLI } else { spec_row_relative_cost_milli };
    while remaining != 0 {
        let mut best_index = slot_count;
        let mut best_value = 0u32;
        for (slot_index, slot) in slots.iter().enumerate() {
            if slot.probe != 0 {
                continue;
            }
            let granted = draft_budgets_out[slot_index];
            if granted >= slot.maximum_draft_depth {
                continue;
            }
            let alpha = alpha_milli(slot.commit_ema_milli);
            if alpha == 0 {
                continue;
            }
            let mut value = (u64::from(alpha) * u64::from(MILLI) / u64::from(cost)) as u32;
            for _ in 0..granted {
                value = (u64::from(value) * u64::from(alpha) / u64::from(MILLI)) as u32;
            }
            // Strictly greater: the first (lowest-index) slot holding the
            // maximum value keeps it, which is the deterministic tie break.
            if value > best_value {
                best_value = value;
                best_index = slot_index;
            }
        }
        if best_index == slot_count || best_value == 0 {
            break;
        }
        draft_budgets_out[best_index] += 1;
        remaining -= 1;
    }
}

/// Fills `draft_budgets_out[slot]` with the draft rows granted to each slot
/// and returns the total rows assigned including base rows
/// (SparkRowAllocatorAssign). Base rows are granted in slot-index order up to
/// the cap; probe grants next; remaining capacity is granted greedily by
/// marginal value with deterministic lower-index tie breaking.
///
/// `spec_row_relative_cost_milli` scales speculative-row scores by marginal
/// cost: 1000 means a spec row costs the same marginal bytes as a real row
/// (scores are pure alpha^d and real rows always win contested slots);
/// measured routing correlation lowers it (a lane's own spec rows draw
/// correlated experts and are cheaper), letting cheap spec rows legitimately
/// outrank a marginal real row in the byte-bound regime:
/// `score = alpha^d * 1000 / cost`. Values below 1000 only.
///
/// `draft_budgets_out` must have the same length as `slots`. A zero
/// `firing_row_cap` returns 0 without touching the output, matching the C.
pub fn row_allocator_assign(
    slots: &[SlotInput],
    firing_row_cap: u32,
    spec_row_relative_cost_milli: u32,
    draft_budgets_out: &mut [u32],
) -> u32 {
    assert_eq!(slots.len(), draft_budgets_out.len(), "draft_budgets_out must mirror slots");
    if firing_row_cap == 0 {
        return 0;
    }
    draft_budgets_out.fill(0);
    // Base rows first: a real row's marginal expected commit is one token,
    // which no speculative row can match, so speculation never displaces a
    // queued real row. If more slots are active than the cap admits, the
    // surplus slots receive nothing this wave; admission control upstream
    // owns that case.
    let slot_count = slots.len() as u32;
    let base_rows = slot_count.min(firing_row_cap);
    let mut remaining = firing_row_cap - base_rows;
    remaining = grant_probes(slots, remaining, draft_budgets_out);
    greedy_grant(slots, remaining, spec_row_relative_cost_milli, draft_budgets_out);
    base_rows + draft_budgets_out.iter().sum::<u32>()
}
