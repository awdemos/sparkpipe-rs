//! Work transaction ledger (port of `runtime/work_transaction.c` plus the
//! `spark_distributed_work.h` wrapper semantics the rank daemon relies on).
//!
//! Kept inside `rank_daemon` on purpose: `spark-sched`'s work-control port
//! deliberately does not carry the ledger (see its module docs), and the
//! daemon is its only consumer in the port so far.
//!
//! Semantics are the C's, entry for entry: open addressing with linear
//! probing from `fingerprint(identity) % capacity`, `EMPTY` terminates a
//! probe, tombstones are recycled on insert, the oldest terminal entry is
//! recycled when full, and advancing the control generation resets the
//! ledger (lower generation → `ValidationFailed`, active entries → `Busy`).

use spark_sched::work_control::work_transaction::{validate_identity, Identity};

use super::status::{Result, SparkStatus};
use super::wire::identity_fingerprint;

/// C: `SPARK_WORK_TRANSACTION_STATE_*` (wire/descriptor values).
pub const STATE_EMPTY: u32 = 0;
pub const STATE_PREPARED: u32 = 1;
pub const STATE_ACCEPTED: u32 = 2;
pub const STATE_EXECUTING: u32 = 3;
pub const STATE_COMMITTED: u32 = 4;
pub const STATE_CANCELLED: u32 = 5;
pub const STATE_FAILED: u32 = 6;
pub const STATE_TOMBSTONE: u32 = 7;

const INVALID_INDEX: u32 = u32::MAX;

/// C: `SparkWorkTransactionStateIsTerminal`.
pub fn state_is_terminal(state: u32) -> bool {
    state == STATE_COMMITTED || state == STATE_CANCELLED || state == STATE_FAILED
}

/// C: `SparkWorkTransactionStateIsStored`.
fn state_is_stored(state: u32) -> bool {
    (STATE_PREPARED..=STATE_FAILED).contains(&state)
}

/// C: `SparkWorkTransactionTransitionIsLegal`.
fn transition_is_legal(current_state: u32, target_state: u32) -> bool {
    if current_state == target_state {
        return true;
    }
    match current_state {
        STATE_PREPARED => matches!(target_state, STATE_ACCEPTED | STATE_CANCELLED | STATE_FAILED),
        STATE_ACCEPTED => matches!(target_state, STATE_EXECUTING | STATE_CANCELLED | STATE_FAILED),
        STATE_EXECUTING => matches!(target_state, STATE_COMMITTED | STATE_CANCELLED | STATE_FAILED),
        _ => false,
    }
}

/// C: `SparkWorkTransactionEntry`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LedgerEntry {
    pub identity: Identity,
    pub payload_fingerprint: u64,
    pub last_observed_epoch: u64,
    pub state: u32,
    /// C: `SparkStatus terminal_status` as its code
    /// (`SPARK_STATUS_PENDING` while active).
    pub terminal_status: u32,
}

/// C: `SPARK_WORK_TRANSACTION_OBSERVATION_*` — what `observe` learned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Observation {
    /// C: `SPARK_WORK_TRANSACTION_OBSERVATION_NEW`.
    New,
    /// C: `SPARK_WORK_TRANSACTION_OBSERVATION_REPLAY_ACTIVE`.
    ReplayActive,
    /// C: `SPARK_WORK_TRANSACTION_OBSERVATION_REPLAY_TERMINAL`.
    ReplayTerminal,
    /// C: `SPARK_WORK_TRANSACTION_OBSERVATION_CONFLICT` (status
    /// `SPARK_STATUS_VALIDATION_FAILED`).
    Conflict,
}

/// C: `SparkWorkTransactionLedger` (the header fields live on the struct;
/// entries are a `Vec` instead of a caller buffer).
#[derive(Debug, Clone)]
pub struct TransactionLedger {
    entries: Vec<LedgerEntry>,
    pub entry_count: u32,
    pub active_entry_count: u32,
    pub terminal_entry_count: u32,
    pub tombstone_entry_count: u32,
    pub epoch: u64,
    pub active_control_generation: u64,
}

impl TransactionLedger {
    /// C: `SparkWorkTransactionInitializeLedger`.
    pub fn new(entry_capacity: u32) -> Result<Self> {
        if entry_capacity == 0 {
            return Err(SparkStatus::InvalidArgument);
        }
        Ok(Self {
            entries: vec![LedgerEntry::default(); entry_capacity as usize],
            entry_count: 0,
            active_entry_count: 0,
            terminal_entry_count: 0,
            tombstone_entry_count: 0,
            epoch: 0,
            active_control_generation: 0,
        })
    }

    pub fn entry_capacity(&self) -> u32 {
        self.entries.len() as u32
    }

    /// Invariant check (C: `SparkWorkTransactionValidateLedger`).
    pub fn validate(&self) -> Result<()> {
        let capacity = self.entry_capacity();
        if capacity == 0
            || self.entry_count > capacity
            || self.active_entry_count > self.entry_count
            || self.terminal_entry_count > self.entry_count
            || self.tombstone_entry_count > capacity
            || self.active_entry_count + self.terminal_entry_count != self.entry_count
        {
            return Err(SparkStatus::InvalidArgument);
        }
        Ok(())
    }

    fn start_index(&self, identity: &Identity) -> u32 {
        (identity_fingerprint(identity) % u64::from(self.entry_capacity())) as u32
    }

    /// C: `SparkWorkTransactionFindInternal`.
    fn find_internal(&self, identity: &Identity) -> std::result::Result<u32, SparkStatus> {
        let capacity = self.entry_capacity();
        let start_index = self.start_index(identity);
        for probe_index in 0..capacity {
            let entry_index = (start_index + probe_index) % capacity;
            let entry = &self.entries[entry_index as usize];
            if entry.state == STATE_EMPTY {
                return Err(SparkStatus::NotFound);
            }
            if entry.state != STATE_TOMBSTONE && entry.identity == *identity {
                return Ok(entry_index);
            }
        }
        Err(SparkStatus::NotFound)
    }

    /// C: `SparkWorkTransactionResetLedger`.
    fn reset(&mut self, control_generation: u64) {
        self.entries.fill(LedgerEntry::default());
        self.entry_count = 0;
        self.active_entry_count = 0;
        self.terminal_entry_count = 0;
        self.tombstone_entry_count = 0;
        self.epoch = 0;
        self.active_control_generation = control_generation;
    }

    /// C: `SparkWorkTransactionPrepareControlGeneration`.
    fn prepare_control_generation(&mut self, control_generation: u64) -> Result<()> {
        if self.active_control_generation == 0 {
            self.reset(control_generation);
            return Ok(());
        }
        if control_generation == self.active_control_generation {
            return Ok(());
        }
        if control_generation < self.active_control_generation {
            return Err(SparkStatus::ValidationFailed);
        }
        if self.active_entry_count != 0 {
            return Err(SparkStatus::Busy);
        }
        self.reset(control_generation);
        Ok(())
    }

    /// C: `SparkWorkTransactionRecycleOldestTerminalEntry`.
    fn recycle_oldest_terminal_entry(&mut self) -> Result<()> {
        let mut oldest_epoch = u64::MAX;
        let mut oldest_index = INVALID_INDEX;
        for (entry_index, entry) in self.entries.iter().enumerate() {
            if state_is_terminal(entry.state) && entry.last_observed_epoch < oldest_epoch {
                oldest_epoch = entry.last_observed_epoch;
                oldest_index = entry_index as u32;
            }
        }
        if oldest_index == INVALID_INDEX {
            return Err(SparkStatus::CapacityExceeded);
        }
        let entry = &mut self.entries[oldest_index as usize];
        *entry = LedgerEntry { state: STATE_TOMBSTONE, ..LedgerEntry::default() };
        self.entry_count -= 1;
        self.terminal_entry_count -= 1;
        self.tombstone_entry_count += 1;
        Ok(())
    }

    /// C: `SparkWorkTransactionSelectInsertionEntry`.
    fn select_insertion_entry(&mut self, identity: &Identity) -> Result<u32> {
        let capacity = self.entry_capacity();
        let start_index = self.start_index(identity);
        let mut first_tombstone_index = INVALID_INDEX;
        for probe_index in 0..capacity {
            let entry_index = (start_index + probe_index) % capacity;
            let entry = &self.entries[entry_index as usize];
            if entry.state == STATE_TOMBSTONE && first_tombstone_index == INVALID_INDEX {
                first_tombstone_index = entry_index;
                continue;
            }
            if entry.state == STATE_EMPTY {
                if first_tombstone_index != INVALID_INDEX {
                    self.tombstone_entry_count -= 1;
                    return Ok(first_tombstone_index);
                }
                return Ok(entry_index);
            }
        }
        if first_tombstone_index != INVALID_INDEX {
            self.tombstone_entry_count -= 1;
            return Ok(first_tombstone_index);
        }
        Err(SparkStatus::CapacityExceeded)
    }

    fn bump_epoch(&mut self) {
        self.epoch += 1;
        if self.epoch == 0 {
            self.epoch = 1;
        }
    }

    /// C: `SparkWorkTransactionObserve`. Returns the observation kind and,
    /// for `New`, the index of the freshly inserted `PREPARED` entry.
    pub fn observe(
        &mut self,
        identity: &Identity,
        payload_fingerprint: u64,
    ) -> Result<(Observation, Option<u32>)> {
        if payload_fingerprint == 0 {
            return Err(SparkStatus::InvalidArgument);
        }
        self.validate()?;
        validate_identity(identity).map_err(|_| SparkStatus::InvalidArgument)?;
        self.prepare_control_generation(identity.control_generation)?;
        self.bump_epoch();
        match self.find_internal(identity) {
            Ok(entry_index) => {
                let entry = &mut self.entries[entry_index as usize];
                entry.last_observed_epoch = self.epoch;
                if entry.payload_fingerprint != payload_fingerprint {
                    return Ok((Observation::Conflict, Some(entry_index)));
                }
                let observation = if state_is_terminal(entry.state) {
                    Observation::ReplayTerminal
                } else {
                    Observation::ReplayActive
                };
                Ok((observation, Some(entry_index)))
            }
            Err(SparkStatus::NotFound) => {
                let entry_index = match self.select_insertion_entry(identity) {
                    Ok(index) => index,
                    Err(SparkStatus::CapacityExceeded) => {
                        self.recycle_oldest_terminal_entry()?;
                        self.select_insertion_entry(identity)?
                    }
                    Err(status) => return Err(status),
                };
                let entry = &mut self.entries[entry_index as usize];
                *entry = LedgerEntry {
                    identity: *identity,
                    payload_fingerprint,
                    last_observed_epoch: self.epoch,
                    state: STATE_PREPARED,
                    terminal_status: SparkStatus::Pending as u32,
                };
                self.entry_count += 1;
                self.active_entry_count += 1;
                Ok((Observation::New, Some(entry_index)))
            }
            Err(status) => Err(status),
        }
    }

    /// C: `SparkWorkTransactionTransition`. `terminal_status` is carried as
    /// the `SparkStatus` code; it is recorded only for terminal targets.
    pub fn transition(
        &mut self,
        identity: &Identity,
        target_state: u32,
        terminal_status: SparkStatus,
    ) -> Result<()> {
        self.validate()?;
        validate_identity(identity).map_err(|_| SparkStatus::InvalidArgument)?;
        if !state_is_stored(target_state) {
            return Err(SparkStatus::InvalidArgument);
        }
        let entry_index = self.find_internal(identity)?;
        let entry = &self.entries[entry_index as usize];
        if state_is_terminal(entry.state) {
            if entry.state == target_state && entry.terminal_status == terminal_status as u32 {
                return Err(SparkStatus::Duplicate);
            }
            return Err(SparkStatus::InvalidArgument);
        }
        if !transition_is_legal(entry.state, target_state) {
            return Err(SparkStatus::InvalidArgument);
        }
        self.bump_epoch();
        let target_is_terminal = state_is_terminal(target_state);
        let entry = &mut self.entries[entry_index as usize];
        entry.last_observed_epoch = self.epoch;
        if target_is_terminal {
            self.active_entry_count -= 1;
            self.terminal_entry_count += 1;
        }
        entry.state = target_state;
        entry.terminal_status =
            if target_is_terminal { terminal_status as u32 } else { SparkStatus::Pending as u32 };
        Ok(())
    }

    /// C: `SparkWorkTransactionFind` (immutable view of the entry).
    pub fn find(&self, identity: &Identity) -> Result<&LedgerEntry> {
        self.validate()?;
        validate_identity(identity).map_err(|_| SparkStatus::InvalidArgument)?;
        self.find_internal(identity).map(|entry_index| &self.entries[entry_index as usize])
    }

    // -------------------------------------------------------------------
    // `spark_distributed_work.h` wrapper semantics used by the daemon
    // -------------------------------------------------------------------

    /// C: `SparkDistributedWorkObserveTransaction` — the daemon-facing
    /// wrapper. Replays (`REPLAY_ACTIVE` / `REPLAY_TERMINAL`) surface as
    /// `Err(Duplicate)`; a fingerprint conflict surfaces as
    /// `Err(ValidationFailed)`; `New` yields `Ok(entry_index)`.
    pub fn distributed_observe(
        &mut self,
        identity: &Identity,
        payload_fingerprint: u64,
    ) -> Result<u32> {
        match self.observe(identity, payload_fingerprint)? {
            (Observation::New, Some(entry_index)) => Ok(entry_index),
            (Observation::Conflict, _) => Err(SparkStatus::ValidationFailed),
            (Observation::ReplayActive | Observation::ReplayTerminal, _) => {
                Err(SparkStatus::Duplicate)
            }
            _ => Err(SparkStatus::InternalError),
        }
    }

    /// C: `SparkDistributedWorkTransitionTransaction` — verifies the
    /// recorded payload fingerprint first (mismatch → `ValidationFailed`)
    /// and maps `InvalidArgument` from the core to `ValidationFailed`.
    pub fn distributed_transition(
        &mut self,
        identity: &Identity,
        payload_fingerprint: u64,
        target_state: u32,
        terminal_status: SparkStatus,
    ) -> Result<()> {
        {
            let entry = self.find(identity)?;
            if entry.payload_fingerprint != payload_fingerprint {
                return Err(SparkStatus::ValidationFailed);
            }
        }
        match self.transition(identity, target_state, terminal_status) {
            Err(SparkStatus::InvalidArgument) => Err(SparkStatus::ValidationFailed),
            other => other,
        }
    }

    /// C: `SparkDistributedWorkFindTransaction` — find with fingerprint
    /// verification.
    pub fn distributed_find(
        &self,
        identity: &Identity,
        payload_fingerprint: u64,
    ) -> Result<&LedgerEntry> {
        let entry = self.find(identity)?;
        if entry.payload_fingerprint != payload_fingerprint {
            return Err(SparkStatus::ValidationFailed);
        }
        Ok(entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(transaction_id: u64) -> Identity {
        Identity {
            control_generation: 7,
            transaction_id,
            dispatch_generation: 1,
            request_generation: 1000 + transaction_id,
            step_generation: 2,
            step_chunk_index: 0,
            step_chunk_count: 1,
            transaction_phase: 1,
        }
    }

    #[test]
    fn observe_new_then_replay_then_conflict() {
        let mut ledger = TransactionLedger::new(16).unwrap();
        let id = identity(1);
        let (observation, entry_index) = ledger.observe(&id, 100).unwrap();
        assert_eq!(observation, Observation::New);
        let entry_index = entry_index.unwrap() as usize;
        assert_eq!(ledger.entries[entry_index].state, STATE_PREPARED);
        assert_eq!(ledger.entries[entry_index].terminal_status, SparkStatus::Pending as u32);
        assert_eq!(ledger.entry_count, 1);
        assert_eq!(ledger.active_entry_count, 1);
        // Replay with the same fingerprint.
        let (observation, _) = ledger.observe(&id, 100).unwrap();
        assert_eq!(observation, Observation::ReplayActive);
        // Same identity, different payload: conflict.
        let (observation, _) = ledger.observe(&id, 101).unwrap();
        assert_eq!(observation, Observation::Conflict);
        // Zero fingerprint is an argument error.
        assert_eq!(ledger.observe(&id, 0), Err(SparkStatus::InvalidArgument));
    }

    #[test]
    fn transition_matrix_and_terminal_replay() {
        let mut ledger = TransactionLedger::new(16).unwrap();
        let id = identity(2);
        ledger.observe(&id, 100).unwrap();
        // PREPARED -> EXECUTING is illegal.
        assert_eq!(
            ledger.transition(&id, STATE_EXECUTING, SparkStatus::Pending),
            Err(SparkStatus::InvalidArgument)
        );
        assert_eq!(ledger.transition(&id, STATE_ACCEPTED, SparkStatus::Pending), Ok(()));
        assert_eq!(ledger.transition(&id, STATE_EXECUTING, SparkStatus::Pending), Ok(()));
        assert_eq!(ledger.transition(&id, STATE_COMMITTED, SparkStatus::Ok), Ok(()));
        assert_eq!(ledger.terminal_entry_count, 1);
        assert_eq!(ledger.active_entry_count, 0);
        // Terminal replay with same state+status: DUPLICATE.
        assert_eq!(
            ledger.transition(&id, STATE_COMMITTED, SparkStatus::Ok),
            Err(SparkStatus::Duplicate)
        );
        // Terminal replay with a different status: INVALID_ARGUMENT.
        assert_eq!(
            ledger.transition(&id, STATE_COMMITTED, SparkStatus::InternalError),
            Err(SparkStatus::InvalidArgument)
        );
        // Unknown identity: NOT_FOUND.
        assert_eq!(
            ledger.transition(&identity(99), STATE_ACCEPTED, SparkStatus::Pending),
            Err(SparkStatus::NotFound)
        );
    }

    #[test]
    fn control_generation_advance_resets() {
        let mut ledger = TransactionLedger::new(16).unwrap();
        let id = identity(3);
        ledger.observe(&id, 100).unwrap();
        // Older generation: VALIDATION_FAILED.
        let mut older = id;
        older.control_generation = 6;
        assert_eq!(ledger.observe(&older, 100), Err(SparkStatus::ValidationFailed));
        // Newer generation with active entries: BUSY.
        let mut newer = id;
        newer.control_generation = 8;
        assert_eq!(ledger.observe(&newer, 100), Err(SparkStatus::Busy));
        // Commit, then the advance resets.
        ledger.transition(&id, STATE_ACCEPTED, SparkStatus::Pending).unwrap();
        ledger.transition(&id, STATE_CANCELLED, SparkStatus::Ok).unwrap();
        let (observation, _) = ledger.observe(&newer, 100).unwrap();
        assert_eq!(observation, Observation::New);
        assert_eq!(ledger.active_control_generation, 8);
        assert_eq!(ledger.entry_count, 1);
    }

    #[test]
    fn full_ledger_recycles_oldest_terminal() {
        let mut ledger = TransactionLedger::new(4).unwrap();
        for txn in 1..=4u64 {
            ledger.observe(&identity(txn), 100 + txn).unwrap();
        }
        // Commit txn 1, then observing a fifth identity recycles it.
        ledger.transition(&identity(1), STATE_ACCEPTED, SparkStatus::Pending).unwrap();
        ledger.transition(&identity(1), STATE_FAILED, SparkStatus::InternalError).unwrap();
        let (observation, _) = ledger.observe(&identity(5), 200).unwrap();
        assert_eq!(observation, Observation::New);
        assert_eq!(ledger.entry_count, 4);
        assert_eq!(ledger.find(&identity(1)).unwrap_err(), SparkStatus::NotFound);
        // Commit txn 2, then observing a sixth identity recycles it.
        ledger.transition(&identity(2), STATE_ACCEPTED, SparkStatus::Pending).unwrap();
        ledger.transition(&identity(2), STATE_CANCELLED, SparkStatus::Ok).unwrap();
        let (observation, _) = ledger.observe(&identity(6), 201).unwrap();
        assert_eq!(observation, Observation::New);
        assert_eq!(ledger.find(&identity(2)).unwrap_err(), SparkStatus::NotFound);
        // Full with no terminal entries: CAPACITY_EXCEEDED.
        let error = ledger.observe(&identity(7), 202).unwrap_err();
        assert_eq!(error, SparkStatus::CapacityExceeded);
    }

    #[test]
    fn distributed_wrappers_map_statuses() {
        let mut ledger = TransactionLedger::new(16).unwrap();
        let id = identity(4);
        let entry_index = ledger.distributed_observe(&id, 500).unwrap();
        assert_eq!(ledger.entries[entry_index as usize].state, STATE_PREPARED);
        // Replay: DUPLICATE.
        assert_eq!(ledger.distributed_observe(&id, 500), Err(SparkStatus::Duplicate));
        // Conflict: VALIDATION_FAILED.
        assert_eq!(ledger.distributed_observe(&id, 501), Err(SparkStatus::ValidationFailed));
        // Transition with wrong fingerprint: VALIDATION_FAILED.
        assert_eq!(
            ledger.distributed_transition(&id, 501, STATE_ACCEPTED, SparkStatus::Pending),
            Err(SparkStatus::ValidationFailed)
        );
        // Illegal transition maps INVALID_ARGUMENT to VALIDATION_FAILED.
        assert_eq!(
            ledger.distributed_transition(&id, 500, STATE_COMMITTED, SparkStatus::Ok),
            Err(SparkStatus::ValidationFailed)
        );
        assert_eq!(
            ledger.distributed_transition(&id, 500, STATE_ACCEPTED, SparkStatus::Pending),
            Ok(())
        );
        assert_eq!(ledger.distributed_find(&id, 500).unwrap().state, STATE_ACCEPTED);
        assert_eq!(ledger.distributed_find(&id, 501), Err(SparkStatus::ValidationFailed));
    }
}
