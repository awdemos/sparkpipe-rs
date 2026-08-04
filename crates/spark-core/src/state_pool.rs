//! Fixed-slot recurrent-state pool — port of
//! `include/sparkpipe/spark_state_pool.h`.
//!
//! One fixed-size slab per sequence (KDA decay-weighted matrix state +
//! convolution windows), O(1) acquire/release from caller-sized storage. The
//! C version takes raw `base`/`next_free` pointers; the Rust port owns its
//! backing and hands out [`SlotHandle`]s whose `Drop` returns the slot —
//! with explicit `release` for the failure-visible path the C API exposes.

const NO_SLOT: u32 = 0xffff_ffff;
const IN_USE: u32 = 0xffff_fffe;

/// Errors mirroring the C `-308xx` return codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum StatePoolError {
    #[error("invalid argument")]
    InvalidArgument,
    #[error("slot is not in use (double release or foreign slot)")]
    NotInUse,
}

/// Handle to an acquired slot. Does not auto-release on drop — the C API
/// makes release failure-visible, and silent drops would hide accounting
/// bugs; use [`StatePool::release`] explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotHandle(u32);

impl SlotHandle {
    pub fn index(self) -> u32 {
        self.0
    }
}

/// Fixed-slot pool for recurrent state.
#[derive(Debug)]
pub struct StatePool {
    backing: Vec<u8>,
    slot_bytes: u64,
    next_free: Vec<u32>,
    free_head: u32,
    free_count: u32,
}

impl StatePool {
    /// Create a pool of `slot_count` slots of `slot_bytes` each.
    pub fn new(slot_count: u32, slot_bytes: u64) -> Result<Self, StatePoolError> {
        if slot_count == 0 || slot_bytes == 0 {
            return Err(StatePoolError::InvalidArgument);
        }
        let backing_bytes = (slot_bytes * slot_count as u64)
            .try_into()
            .map_err(|_| StatePoolError::InvalidArgument)?;
        let mut next_free = Vec::with_capacity(slot_count as usize);
        for slot in 0..slot_count {
            next_free.push(slot + 1);
        }
        *next_free.last_mut().expect("slot_count > 0") = NO_SLOT;
        Ok(Self {
            backing: vec![0u8; backing_bytes],
            slot_bytes,
            next_free,
            free_head: 0,
            free_count: slot_count,
        })
    }

    pub fn free_count(&self) -> u32 {
        self.free_count
    }

    pub fn slot_count(&self) -> u32 {
        self.next_free.len() as u32
    }

    /// Acquire a slot, or `None` when the pool is exhausted (a loud admission
    /// failure, per the C header — never a quiet allocation).
    pub fn acquire(&mut self) -> Option<SlotHandle> {
        if self.free_count == 0 {
            return None;
        }
        let slot = self.free_head;
        self.free_head = self.next_free[slot as usize];
        self.next_free[slot as usize] = IN_USE;
        self.free_count -= 1;
        Some(SlotHandle(slot))
    }

    /// Release a slot. Releasing a slot that is not in use (double release)
    /// fails loudly — the distinct IN_USE/NO_SLOT sentinels make this
    /// unrepresentable in the C design and checked here.
    pub fn release(&mut self, handle: SlotHandle) -> Result<(), StatePoolError> {
        let slot = handle.0 as usize;
        if slot >= self.next_free.len() || self.next_free[slot] != IN_USE {
            return Err(StatePoolError::NotInUse);
        }
        self.next_free[slot] = self.free_head;
        self.free_head = handle.0;
        self.free_count += 1;
        Ok(())
    }

    /// Slot bytes for a live handle.
    pub fn slot(&self, handle: SlotHandle) -> Option<&[u8]> {
        let slot = handle.0 as usize;
        if slot >= self.next_free.len() || self.next_free[slot] != IN_USE {
            return None;
        }
        let start = slot * self.slot_bytes as usize;
        Some(&self.backing[start..start + self.slot_bytes as usize])
    }

    /// Mutable variant of [`StatePool::slot`].
    pub fn slot_mut(&mut self, handle: SlotHandle) -> Option<&mut [u8]> {
        let slot = handle.0 as usize;
        if slot >= self.next_free.len() || self.next_free[slot] != IN_USE {
            return None;
        }
        let start = slot * self.slot_bytes as usize;
        Some(&mut self.backing[start..start + self.slot_bytes as usize])
    }
}
