//! Size-classed slot arena — port of `runtime/arena.h`.
//!
//! One backing blob, up to 8 slot classes with strictly increasing slot
//! sizes, intrusive free lists, and per-slot generation counters so a stale
//! handle is refused at release instead of corrupting the free list. The C
//! version hands out raw pointers into the blob; the Rust port hands out
//! [`ArenaHandle`]s (class + slot + generation) and resolves them to slices
//! with borrow checking, which makes the "stale handle" failure mode
//! unrepresentable outside of explicit generation checks at release.

/// Maximum number of size classes (SPARK_ARENA_MAX_CLASS_COUNT).
pub const MAX_CLASS_COUNT: usize = 8;
/// Slot size alignment (SPARK_ARENA_ALIGNMENT).
pub const ALIGNMENT: usize = 16;

const NO_SLOT: u32 = 0xffff_ffff;
const IN_USE: u32 = 0xffff_fffe;

/// Descriptor for one size class: `slot_bytes` per slot, `slot_count` slots.
#[derive(Debug, Clone, Copy)]
pub struct ClassDescriptor {
    pub slot_bytes: u32,
    pub slot_count: u32,
}

/// Handle to a live allocation. Carries the slot generation; releasing with a
/// stale handle fails loudly (mirrors the C generation check).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaHandle {
    class_index: u32,
    slot_index: u32,
    generation: u64,
}

/// Errors mirroring the C `-309xx` return codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ArenaError {
    #[error("invalid argument")]
    InvalidArgument,
    #[error("size classes must be non-empty and strictly increasing in slot size")]
    InvalidClassLayout,
    #[error("region size overflow")]
    RegionOverflow,
    #[error("requested bytes exceed every size class")]
    NoFittingClass,
    #[error("size class exhausted")]
    ClassExhausted,
    #[error("stale or foreign handle")]
    StaleHandle,
}

#[derive(Debug)]
struct ArenaClass {
    slot_bytes: u32,
    slot_count: u32,
    /// Free-list links; IN_USE marks a live slot (distinct from NO_SLOT so a
    /// double-free is unrepresentable, per the C header's rationale).
    links: Vec<u32>,
    slot_generations: Vec<u64>,
    free_head: u32,
    free_count: u32,
    generation: u64,
}

/// Size-classed arena over one backing blob.
#[derive(Debug)]
pub struct Arena {
    backing: Vec<u8>,
    classes: Vec<ArenaClass>,
    /// Byte offset of each class's slot region inside `backing`.
    class_offsets: Vec<usize>,
}

fn round_slot_bytes(slot_bytes: u32) -> u64 {
    (slot_bytes as u64 + (ALIGNMENT as u64 - 1)) & !(ALIGNMENT as u64 - 1)
}

impl Arena {
    /// Create an arena from size-class descriptors (must be non-empty and
    /// strictly increasing in slot size; at most MAX_CLASS_COUNT classes).
    pub fn new(descriptors: &[ClassDescriptor]) -> Result<Self, ArenaError> {
        if descriptors.is_empty() || descriptors.len() > MAX_CLASS_COUNT {
            return Err(ArenaError::InvalidArgument);
        }

        let mut previous_slot_bytes = 0u64;
        let mut slot_region_bytes = 0u64;
        for descriptor in descriptors {
            let slot_bytes = round_slot_bytes(descriptor.slot_bytes);
            if descriptor.slot_bytes == 0 || descriptor.slot_count == 0 {
                return Err(ArenaError::InvalidClassLayout);
            }
            if slot_region_bytes > 0 && slot_bytes <= previous_slot_bytes {
                return Err(ArenaError::InvalidClassLayout);
            }
            slot_region_bytes = slot_region_bytes
                .checked_add(slot_bytes * descriptor.slot_count as u64)
                .ok_or(ArenaError::RegionOverflow)?;
            previous_slot_bytes = slot_bytes;
        }

        // Vec allocation caps at isize::MAX bytes; the C version fails at
        // total_bytes > SIZE_MAX (its -30910). Map both to RegionOverflow.
        if slot_region_bytes > isize::MAX as u64 {
            return Err(ArenaError::RegionOverflow);
        }
        let total_bytes: usize =
            slot_region_bytes.try_into().map_err(|_| ArenaError::RegionOverflow)?;
        let backing = vec![0u8; total_bytes];

        let mut classes = Vec::with_capacity(descriptors.len());
        let mut class_offsets = Vec::with_capacity(descriptors.len());
        let mut region_offset = 0usize;
        for descriptor in descriptors {
            let slot_bytes = round_slot_bytes(descriptor.slot_bytes) as u32;
            let slot_count = descriptor.slot_count;
            class_offsets.push(region_offset);
            region_offset += slot_bytes as usize * slot_count as usize;

            let mut links = Vec::with_capacity(slot_count as usize);
            let mut slot_generations = Vec::with_capacity(slot_count as usize);
            for slot in 0..slot_count {
                links.push(slot + 1);
                slot_generations.push(0u64);
            }
            *links.last_mut().expect("slot_count > 0") = NO_SLOT;

            classes.push(ArenaClass {
                slot_bytes,
                slot_count,
                links,
                slot_generations,
                free_head: 0,
                free_count: slot_count,
                generation: 0,
            });
        }

        Ok(Self { backing, classes, class_offsets })
    }

    /// Total reserved backing bytes (SPARK_ARENA_RESERVED_BYTES).
    pub fn reserved_bytes(&self) -> u64 {
        self.backing.len() as u64
    }

    /// Monotonic per-class generation counter (SPARK_ARENA_CLASS_GENERATION).
    pub fn class_generation(&self, class_index: u32) -> u64 {
        self.classes.get(class_index as usize).map_or(0, |class| class.generation)
    }

    /// Free slots in a class.
    pub fn class_free_count(&self, class_index: u32) -> u32 {
        self.classes.get(class_index as usize).map_or(0, |class| class.free_count)
    }

    /// Acquire the first fitting class's head slot. Mirrors C semantics: if
    /// the first fitting class is exhausted the call fails (it does not fall
    /// through to a larger class).
    pub fn acquire(&mut self, bytes: u32) -> Result<ArenaHandle, ArenaError> {
        if bytes == 0 {
            return Err(ArenaError::InvalidArgument);
        }
        for (class_index, class) in self.classes.iter_mut().enumerate() {
            if bytes > class.slot_bytes {
                continue;
            }
            if class.free_count == 0 {
                return Err(ArenaError::ClassExhausted);
            }
            let slot = class.free_head as usize;
            if class.slot_generations[slot] == u64::MAX {
                // C -30918: generation would wrap; the slot is retired.
                return Err(ArenaError::StaleHandle);
            }
            class.free_head = class.links[slot];
            class.links[slot] = IN_USE;
            class.free_count -= 1;
            class.generation = class.generation.saturating_add(1);
            let generation = class.slot_generations[slot] + 1;
            class.slot_generations[slot] = generation;
            return Ok(ArenaHandle {
                class_index: class_index as u32,
                slot_index: slot as u32,
                generation,
            });
        }
        Err(ArenaError::NoFittingClass)
    }

    /// Release a previously acquired handle. A stale generation, a foreign
    /// arena's handle, or a double release all fail loudly.
    pub fn release(&mut self, handle: ArenaHandle) -> Result<(), ArenaError> {
        let class =
            self.classes.get_mut(handle.class_index as usize).ok_or(ArenaError::StaleHandle)?;
        let slot = handle.slot_index as usize;
        if slot >= class.slot_count as usize {
            return Err(ArenaError::StaleHandle);
        }
        if class.links[slot] != IN_USE {
            return Err(ArenaError::StaleHandle);
        }
        if handle.generation == 0 || class.slot_generations[slot] != handle.generation {
            return Err(ArenaError::StaleHandle);
        }
        class.links[slot] = class.free_head;
        class.free_head = handle.slot_index;
        class.free_count += 1;
        Ok(())
    }

    /// Resolve a handle to its slot bytes. Fails on a stale handle, same as
    /// release — readers must not observe a recycled slot's new occupant.
    pub fn get(&self, handle: ArenaHandle) -> Result<&[u8], ArenaError> {
        let (class, start, end) = self.resolve(handle)?;
        let _ = class;
        Ok(&self.backing[start..end])
    }

    /// Mutable variant of [`Arena::get`].
    pub fn get_mut(&mut self, handle: ArenaHandle) -> Result<&mut [u8], ArenaError> {
        let (_, start, end) = self.resolve(handle)?;
        Ok(&mut self.backing[start..end])
    }

    fn resolve(&self, handle: ArenaHandle) -> Result<(&ArenaClass, usize, usize), ArenaError> {
        let class = self.classes.get(handle.class_index as usize).ok_or(ArenaError::StaleHandle)?;
        let slot = handle.slot_index as usize;
        if slot >= class.slot_count as usize {
            return Err(ArenaError::StaleHandle);
        }
        if class.links[slot] != IN_USE || class.slot_generations[slot] != handle.generation {
            return Err(ArenaError::StaleHandle);
        }
        let start =
            self.class_offsets[handle.class_index as usize] + slot * class.slot_bytes as usize;
        Ok((class, start, start + class.slot_bytes as usize))
    }
}
