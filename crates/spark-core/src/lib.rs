//! `spark-core`: memory and cache machinery, ported from the C tree.
//!
//! - [`arena`]: size-classed slot arena with generational handles
//!   (port of `runtime/arena.h`)
//! - [`state_pool`]: fixed-slot recurrent-state pool (port of
//!   `include/sparkpipe/spark_state_pool.h`)
//! - [`kv_arena`]: paged KV block arena, two-tier (resident/pool) with
//!   refcount ownership (port of the `SparkKvCacheArena*` core in
//!   `cache/kv_cache.c`)
//! - [`prefix_cache`]: content-addressed prefix cache with LRU eviction,
//!   sequence bindings, reservations, and lookahead protection (port of
//!   `cache/prefix_cache.c`)
//! - [`batch_sequence_table`]: batch-plane sequence admission/exchange
//!   lifecycle (port of `serving/spark_batch_sequence_table.c`)
//! - [`row_allocator`]: speculative/real firing-row budget allocation
//!   (port of `serving/spark_row_allocator.c`)

pub mod arena;
pub mod batch_sequence_table;
pub mod kv_arena;
pub mod prefix_cache;
pub mod row_allocator;
pub mod state_pool;
