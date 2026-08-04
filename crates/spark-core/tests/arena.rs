//! Port of tests/test_arena.c: generation handles, alignment, loud
//! exhaustion, geometry rejection. Byte-layout assertions from the C test
//! (backing formula with link/generation regions) do not apply — the Rust
//! arena keeps metadata in Vecs, not the backing blob.

use spark_core::arena::{Arena, ArenaError, ClassDescriptor};

#[test]
fn arena_cycles() {
    let mut arena = Arena::new(&[
        ClassDescriptor { slot_bytes: 100, slot_count: 3 },
        ClassDescriptor { slot_bytes: 4096, slot_count: 2 },
    ])
    .unwrap();

    // Fill class 0; handles resolve to writable, disjoint slots.
    let mut first = Vec::new();
    for index in 0..3u32 {
        let handle = arena.acquire(index + 1).unwrap();
        arena.get_mut(handle).unwrap()[..(index + 1) as usize].fill((index + 1) as u8);
        first.push(handle);
    }
    // First fitting class exhausted: loud failure, no fall-through to class 1.
    assert_eq!(arena.acquire(1), Err(ArenaError::ClassExhausted));
    assert_eq!(arena.class_generation(0), 3);

    // Class 1 still serves.
    let second = arena.acquire(4096).unwrap();
    let third = arena.acquire(2000).unwrap();
    assert_eq!(arena.acquire(4096), Err(ArenaError::ClassExhausted));
    // Bigger than every class: distinct failure.
    assert_eq!(arena.acquire(8192), Err(ArenaError::NoFittingClass));

    // Release + reacquire recycles the slot with a fresh generation; the
    // stale handle is refused.
    let stale = first[1];
    arena.release(first[1]).unwrap();
    let replacement = arena.acquire(64).unwrap();
    assert_ne!(replacement, stale);
    assert_eq!(arena.release(stale), Err(ArenaError::StaleHandle));
    arena.release(replacement).unwrap();
    assert_eq!(arena.class_generation(0), 4);
    assert_eq!(arena.get(first[0]).unwrap()[0], 1);

    // Double release refused.
    arena.release(second).unwrap();
    assert_eq!(arena.release(second), Err(ArenaError::StaleHandle));

    // Stale-generation read refused (recycled slot's new occupant protected).
    let recycle = arena.acquire(16).unwrap();
    assert_eq!(arena.get(stale), Err(ArenaError::StaleHandle));
    arena.release(recycle).unwrap();
    arena.release(third).unwrap();
    arena.release(first[0]).unwrap();
    arena.release(first[2]).unwrap();
}

#[test]
fn rejects_bad_geometry() {
    // Non-increasing slot sizes.
    let result = Arena::new(&[
        ClassDescriptor { slot_bytes: 4096, slot_count: 1 },
        ClassDescriptor { slot_bytes: 64, slot_count: 1 },
    ]);
    assert_eq!(result.unwrap_err(), ArenaError::InvalidClassLayout);
    // Equal after 16-byte rounding (17 -> 32, 31 -> 32).
    let result = Arena::new(&[
        ClassDescriptor { slot_bytes: 17, slot_count: 1 },
        ClassDescriptor { slot_bytes: 31, slot_count: 1 },
    ]);
    assert_eq!(result.unwrap_err(), ArenaError::InvalidClassLayout);
    // Zero slots.
    let result = Arena::new(&[ClassDescriptor { slot_bytes: 64, slot_count: 0 }]);
    assert_eq!(result.unwrap_err(), ArenaError::InvalidClassLayout);
    // Overflowing region.
    let result =
        Arena::new(&[ClassDescriptor { slot_bytes: 0xffff_ffff, slot_count: 0xffff_ffff }]);
    assert_eq!(result.unwrap_err(), ArenaError::RegionOverflow);
    // Empty class list.
    let result = Arena::new(&[]);
    assert_eq!(result.unwrap_err(), ArenaError::InvalidArgument);
}

#[test]
fn alignment_rounding() {
    // 100-byte slots round to 112 (16-byte alignment); the full rounded
    // stride is usable.
    let mut arena = Arena::new(&[ClassDescriptor { slot_bytes: 100, slot_count: 1 }]).unwrap();
    let handle = arena.acquire(112).unwrap();
    assert_eq!(arena.get(handle).unwrap().len(), 112);
    assert_eq!(arena.acquire(1), Err(ArenaError::ClassExhausted));
}
