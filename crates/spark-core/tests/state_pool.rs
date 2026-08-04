//! Port of tests/test_state_pool.c: every slot reachable exactly once,
//! double release refused, exhaustion loud, slots disjoint and stable.

use spark_core::state_pool::{StatePool, StatePoolError};

#[test]
fn state_pool_contract() {
    let mut pool = StatePool::new(8, 48).unwrap();

    let mut slots = Vec::new();
    for index in 0..8u32 {
        let handle = pool.acquire().expect("acquire ran dry early");
        pool.slot_mut(handle).unwrap().fill((index + 1) as u8);
        slots.push(handle);
    }

    // Exhaustion is loud.
    assert!(pool.acquire().is_none());
    assert_eq!(pool.free_count(), 0);

    // No aliasing: every slot kept its own bytes.
    for (index, handle) in slots.iter().enumerate() {
        assert_eq!(pool.slot(*handle).unwrap()[0], (index + 1) as u8);
    }

    // Double release refused; freed slot reissued exactly.
    pool.release(slots[3]).unwrap();
    assert_eq!(pool.release(slots[3]), Err(StatePoolError::NotInUse));
    assert_eq!(pool.free_count(), 1);
    let again = pool.acquire().unwrap();
    assert_eq!(again, slots[3]);
}

#[test]
fn released_slot_bytes_inaccessible() {
    let mut pool = StatePool::new(2, 16).unwrap();
    let handle = pool.acquire().unwrap();
    pool.release(handle).unwrap();
    // After release the handle resolves to nothing — reads must not observe
    // the slot's next occupant.
    assert!(pool.slot(handle).is_none());
    assert!(pool.slot_mut(handle).is_none());
    let next = pool.acquire().unwrap();
    assert_eq!(next, handle);
    assert!(pool.slot(next).is_some());
}

#[test]
fn invalid_geometry_rejected() {
    assert!(StatePool::new(0, 48).is_err());
    assert!(StatePool::new(8, 0).is_err());
}
