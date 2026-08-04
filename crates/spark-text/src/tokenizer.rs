//! Byte-level BPE tokenizer — structural port of the C tree's
//! `text/tokenizer.c` (API declared in `include/sparkpipe/spark_tokenizer.h`).
//!
//! This is a deliberate structural port, not a re-implementation:
//!
//! - The vocabulary/merge **string tables** keep the C design: an entry arena
//!   plus power-of-two bucket array with external chaining (`next_index`),
//!   hashed with FNV-1a-32. A `std::collections::HashMap` is not a substitute;
//!   the tuned table layout is the point.
//! - The pretoken **piece cache** keeps the C open-addressing design: 16384
//!   linear-probed slots, inline 32-byte pieces, and a bounded token pool,
//!   hashed with FNV-1a-64. It lives in the [`Workspace`] so it stays warm
//!   across encode calls and across symbol-buffer growth.
//! - The merge loop keeps the C **min-heap of merge candidates** ordered by
//!   `(rank, left_symbol_index)` with generation counters to invalidate stale
//!   entries, so BPE tie-breaking is bit-for-bit the C behavior and encode
//!   output is token-for-token identical with the C tokenizer.
//! - GPT-2-style regex pretokenization keeps the C 256-entry byte-class table
//!   and counted run scans (portable scalar code the compiler can
//!   auto-vectorize), not a regex engine.
//!
//! UTF-8 handling: exactly like the C, all encode input is treated as raw
//! bytes (`&[u8]`). Byte-level BPE maps every byte to a token through the
//! GPT-2 byte-to-unicode table, so invalid UTF-8 in the input is not an
//! error; it is encoded byte by byte. Token text stored in the vocabulary is
//! UTF-8 over byte-to-unicode code points; decode reverses the mapping.
//!
//! Thread-safety: the C tree does **not** put a mutex around encode. Batch
//! encoding spawns pthread workers that share the `const SparkTokenizer`
//! read-only and give each worker its own `SparkTokenizerWorkspace`
//! (see `SparkTokenizerBatchWorkerMain`); single-request serving
//! (`text/prompt.c`) either uses a caller-owned workspace or a temporary one.
//! The Rust port mirrors that call pattern: [`Tokenizer`] is immutable after
//! loading (`Send + Sync` automatically) and every encode entry point takes
//! `&self` plus `&mut Workspace`. Sharing one workspace across threads is a
//! compile-time error, which is stronger than the C contract with identical
//! runtime cost. [`Tokenizer::encode_batch`] uses scoped threads with one
//! workspace per worker, exactly like the C.
//!
//! Error reporting mirrors the `SparkStatus` codes the C tokenizer returns
//! via [`TokenizerError`].

use std::path::Path;

/// Errors mirroring the `SparkStatus` codes returned by `text/tokenizer.c`.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    /// `SPARK_STATUS_INVALID_ARGUMENT`.
    #[error("invalid argument: {0}")]
    InvalidArgument(&'static str),
    /// `SPARK_STATUS_CAPACITY_EXCEEDED`.
    #[error("capacity exceeded")]
    CapacityExceeded,
    /// `SPARK_STATUS_NOT_FOUND`.
    #[error("not found: {0}")]
    NotFound(&'static str),
    /// `SPARK_STATUS_IO_ERROR`.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// `SPARK_STATUS_PARSE_ERROR`.
    #[error("parse error: {0}")]
    Parse(String),
    /// `SPARK_STATUS_SCHEMA_ERROR`.
    #[error("schema error: {0}")]
    Schema(String),
    /// `SPARK_STATUS_DUPLICATE`.
    #[error("duplicate entry: {0}")]
    Duplicate(&'static str),
    /// `SPARK_STATUS_INTERNAL_ERROR`.
    #[error("internal error: {0}")]
    Internal(&'static str),
}

/// `SPARK_TOKENIZER_BPE_MODEL_KIND_BYTE_LEVEL`.
pub const BPE_MODEL_KIND_BYTE_LEVEL: u32 = 1;

/// `SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH`.
pub const ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH: u32 = 0x0000_0001;
/// `SPARK_TOKENIZER_ENCODE_FLAG_ADD_PREFIX_SPACE`.
pub const ENCODE_FLAG_ADD_PREFIX_SPACE: u32 = 0x0000_0002;
/// `SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_REGEX_PRETOKENIZATION`.
pub const ENCODE_FLAG_DISABLE_REGEX_PRETOKENIZATION: u32 = 0x0000_0004;
/// `SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_PIECE_CACHE`.
pub const ENCODE_FLAG_DISABLE_PIECE_CACHE: u32 = 0x0000_0008;
/// `SPARK_TOKENIZER_ENCODE_KNOWN_FLAGS`.
pub const ENCODE_KNOWN_FLAGS: u32 = ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH
    | ENCODE_FLAG_ADD_PREFIX_SPACE
    | ENCODE_FLAG_DISABLE_REGEX_PRETOKENIZATION
    | ENCODE_FLAG_DISABLE_PIECE_CACHE;

/// `SPARK_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS`.
pub const DECODE_FLAG_SKIP_SPECIAL_TOKENS: u32 = 0x0000_0001;
/// `SPARK_TOKENIZER_DECODE_KNOWN_FLAGS`.
pub const DECODE_KNOWN_FLAGS: u32 = DECODE_FLAG_SKIP_SPECIAL_TOKENS;

/// `SPARK_TOKENIZER_PIECE_CACHE_INLINE_BYTES`.
pub const PIECE_CACHE_INLINE_BYTES: usize = 32;
/// `SPARK_TOKENIZER_PIECE_CACHE_SLOT_COUNT`.
pub const PIECE_CACHE_SLOT_COUNT: usize = 16384;
/// `SPARK_TOKENIZER_PIECE_CACHE_TOKEN_CAPACITY`.
pub const PIECE_CACHE_TOKEN_CAPACITY: usize = 262144;
const PIECE_CACHE_EMPTY_HASH: u64 = 0;

/// `SPARK_TOKENIZER_NO_TOKEN_ID`.
pub const NO_TOKEN_ID: u32 = u32::MAX;
/// `SPARK_TOKENIZER_MAX_PARALLEL_WORKER_COUNT`.
pub const MAX_PARALLEL_WORKER_COUNT: usize = 16;

const COMPILED_FILE_MAGIC: u64 = 0x0031_4b4f_5453_5053;
const COMPILED_FILE_VERSION: u32 = 1;

const EMPTY_BUCKET: u32 = u32::MAX;
const HASH_LOAD_FACTOR: u32 = 2;
const MERGE_HEAP_CAPACITY_FACTOR: usize = 4;
const MERGE_HEAP_CAPACITY_SLACK: usize = 16;
const SYMBOL_NONE: u32 = u32::MAX;

/// `SparkTokenizerNextPowerOfTwo`.
fn next_power_of_two(value: u32) -> u32 {
    let mut power_of_two = 1u32;
    while power_of_two < value && power_of_two <= u32::MAX / 2 {
        power_of_two <<= 1;
    }
    power_of_two
}

/// `SparkTokenizerHashBytes` (FNV-1a-32).
fn hash_bytes(text: &[u8]) -> u32 {
    let mut hash_value = 2_166_136_261u32;
    for &byte in text {
        hash_value ^= u32::from(byte);
        hash_value = hash_value.wrapping_mul(16_777_619);
    }
    hash_value
}

/// `SparkTokenizerHashTokenPair` (FNV-1a-32 over the two ids).
fn hash_token_pair(left_token_id: u32, right_token_id: u32) -> u32 {
    let mut hash_value = 2_166_136_261u32;
    hash_value ^= left_token_id;
    hash_value = hash_value.wrapping_mul(16_777_619);
    hash_value ^= right_token_id;
    hash_value = hash_value.wrapping_mul(16_777_619);
    hash_value
}

/// `SparkTokenizerByteToUnicodeCodePoint` (GPT-2 bytes_to_unicode).
fn byte_to_unicode_code_point(byte_value: u32) -> u32 {
    let printable = |candidate: u32| {
        (33..=126).contains(&candidate)
            || (161..=172).contains(&candidate)
            || (174..=255).contains(&candidate)
    };
    if printable(byte_value) {
        return byte_value;
    }
    let mut next_code_point = 256u32;
    for candidate in 0..=byte_value {
        if !printable(candidate) {
            if candidate == byte_value {
                return next_code_point;
            }
            next_code_point += 1;
        }
    }
    byte_value
}

/// `SparkTokenizerUnicodeCodePointToByte`.
fn unicode_code_point_to_byte(code_point: u32) -> Option<u8> {
    let printable = |candidate: u32| {
        (33..=126).contains(&candidate)
            || (161..=172).contains(&candidate)
            || (174..=255).contains(&candidate)
    };
    if printable(code_point) {
        return Some(code_point as u8);
    }
    let mut next_code_point = 256u32;
    for candidate in 0..=255u32 {
        if !printable(candidate) {
            if code_point == next_code_point {
                return Some(candidate as u8);
            }
            next_code_point += 1;
        }
    }
    None
}

/// `SparkTokenizerReadUtf8CodePoint`: reads one UTF-8 code point starting at
/// `*position`, advancing it past the consumed bytes. Returns `None` on
/// truncated/invalid sequences, exactly like the C (which then fails decode
/// with `SPARK_STATUS_PARSE_ERROR`).
fn read_utf8_code_point(text: &[u8], position: &mut usize) -> Option<u32> {
    if *position >= text.len() {
        return None;
    }
    let first_byte = text[*position];
    let remaining_bytes = text.len() - *position;
    let continuation = |offset: usize| text[*position + offset] & 0xc0 == 0x80;
    if first_byte < 0x80 {
        *position += 1;
        return Some(u32::from(first_byte));
    }
    if (first_byte & 0xe0) == 0xc0 && remaining_bytes >= 2 && continuation(1) {
        let code_point =
            (u32::from(first_byte & 0x1f) << 6) | u32::from(text[*position + 1] & 0x3f);
        *position += 2;
        return Some(code_point);
    }
    if (first_byte & 0xf0) == 0xe0 && remaining_bytes >= 3 && continuation(1) && continuation(2) {
        let code_point = (u32::from(first_byte & 0x0f) << 12)
            | (u32::from(text[*position + 1] & 0x3f) << 6)
            | u32::from(text[*position + 2] & 0x3f);
        *position += 3;
        return Some(code_point);
    }
    if (first_byte & 0xf8) == 0xf0
        && remaining_bytes >= 4
        && continuation(1)
        && continuation(2)
        && continuation(3)
    {
        let code_point = (u32::from(first_byte & 0x07) << 18)
            | (u32::from(text[*position + 1] & 0x3f) << 12)
            | (u32::from(text[*position + 2] & 0x3f) << 6)
            | u32::from(text[*position + 3] & 0x3f);
        *position += 4;
        return Some(code_point);
    }
    None
}

/// `SparkTokenizerAppendUtf8CodePoint` (1-3 byte forms; the byte-to-unicode
/// table never produces code points above 0x10F + 255 = 323).
fn append_utf8_code_point(destination: &mut Vec<u8>, code_point: u32) {
    if code_point <= 0x7f {
        destination.push(code_point as u8);
    } else if code_point <= 0x7ff {
        destination.push(0xc0 | (code_point >> 6) as u8);
        destination.push(0x80 | (code_point & 0x3f) as u8);
    } else {
        destination.push(0xe0 | (code_point >> 12) as u8);
        destination.push(0x80 | ((code_point >> 6) & 0x3f) as u8);
        destination.push(0x80 | (code_point & 0x3f) as u8);
    }
}

/// `SparkTokenizerStringEntry`. `text` is `None` for the empty arena slots
/// the C leaves behind when a duplicate merge string is skipped.
struct StringEntry {
    text: Option<Vec<u8>>,
    value: u32,
    next_index: u32,
}

/// The C string table: entry arena + power-of-two bucket array with external
/// chaining (`SparkTokenizerAllocateStringTable`,
/// `SparkTokenizerFindStringEntryInTable`, `SparkTokenizerInsertStringEntry`).
struct StringTable {
    entries: Vec<StringEntry>,
    buckets: Vec<u32>,
}

impl StringTable {
    fn with_entry_count(entry_count: usize) -> Self {
        if entry_count == 0 {
            return StringTable { entries: Vec::new(), buckets: Vec::new() };
        }
        let bucket_count = next_power_of_two(entry_count as u32 * HASH_LOAD_FACTOR + 1) as usize;
        StringTable {
            entries: Vec::with_capacity(entry_count),
            buckets: vec![EMPTY_BUCKET; bucket_count],
        }
    }

    fn find(&self, text: &[u8]) -> Option<&StringEntry> {
        if self.buckets.is_empty() {
            return None;
        }
        let mut entry_index = self.buckets[(hash_bytes(text) as usize) & (self.buckets.len() - 1)];
        while entry_index != EMPTY_BUCKET {
            let entry = &self.entries[entry_index as usize];
            if let Some(entry_text) = &entry.text {
                if entry_text.as_slice() == text {
                    return Some(entry);
                }
            }
            entry_index = entry.next_index;
        }
        None
    }

    /// Pushes the next arena entry and links it into its bucket. Returns
    /// `Err(Duplicate)` without inserting when the text already exists; the
    /// caller then pushes a placeholder empty slot to mirror the C arena
    /// layout (used by merge parsing).
    fn insert(&mut self, text: &[u8], value: u32) -> Result<(), TokenizerError> {
        if self.buckets.is_empty() {
            return Err(TokenizerError::InvalidArgument("string table not allocated"));
        }
        if self.find(text).is_some() {
            return Err(TokenizerError::Duplicate("string table entry"));
        }
        let bucket_index = (hash_bytes(text) as usize) & (self.buckets.len() - 1);
        let entry_index = self.entries.len() as u32;
        self.entries.push(StringEntry {
            text: Some(text.to_vec()),
            value,
            next_index: self.buckets[bucket_index],
        });
        self.buckets[bucket_index] = entry_index;
        Ok(())
    }

    /// Pushes an empty arena slot (never linked into a bucket), mirroring the
    /// zeroed `SparkTokenizerStringEntry` the C leaves on duplicate merges.
    fn push_empty_slot(&mut self) {
        self.entries.push(StringEntry { text: None, value: 0, next_index: EMPTY_BUCKET });
    }
}

/// `SparkTokenizerFastMergePair`.
#[derive(Clone, Copy)]
struct FastMergePair {
    left_token_id: u32,
    right_token_id: u32,
    merged_token_id: u32,
    rank: u32,
}

/// The C merge-pair table (`SparkTokenizerAllocateMergePairTable` and
/// friends): same arena + chained-bucket layout as [`StringTable`], keyed by
/// the `(left, right)` token-id pair.
struct MergePairTable {
    pairs: Vec<FastMergePair>,
    next_indices: Vec<u32>,
    buckets: Vec<u32>,
}

impl MergePairTable {
    fn with_entry_count(entry_count: usize) -> Self {
        if entry_count == 0 {
            return MergePairTable {
                pairs: Vec::new(),
                next_indices: Vec::new(),
                buckets: Vec::new(),
            };
        }
        let bucket_count = next_power_of_two(entry_count as u32 * HASH_LOAD_FACTOR + 1) as usize;
        MergePairTable {
            pairs: Vec::with_capacity(entry_count),
            next_indices: Vec::with_capacity(entry_count),
            buckets: vec![EMPTY_BUCKET; bucket_count],
        }
    }

    fn find(&self, left_token_id: u32, right_token_id: u32) -> Option<&FastMergePair> {
        if self.buckets.is_empty() {
            return None;
        }
        let mut entry_index = self.buckets
            [(hash_token_pair(left_token_id, right_token_id) as usize) & (self.buckets.len() - 1)];
        while entry_index != EMPTY_BUCKET {
            let entry = &self.pairs[entry_index as usize];
            if entry.left_token_id == left_token_id && entry.right_token_id == right_token_id {
                return Some(entry);
            }
            entry_index = self.next_indices[entry_index as usize];
        }
        None
    }

    fn insert(&mut self, pair: FastMergePair) -> Result<(), TokenizerError> {
        if self.buckets.is_empty() {
            return Err(TokenizerError::InvalidArgument("merge pair table not allocated"));
        }
        if self.find(pair.left_token_id, pair.right_token_id).is_some() {
            return Err(TokenizerError::Duplicate("merge pair"));
        }
        let bucket_index = (hash_token_pair(pair.left_token_id, pair.right_token_id) as usize)
            & (self.buckets.len() - 1);
        let entry_index = self.pairs.len() as u32;
        self.pairs.push(pair);
        self.next_indices.push(self.buckets[bucket_index]);
        self.buckets[bucket_index] = entry_index;
        Ok(())
    }
}

/// `SparkTokenizerSpecialToken` (public read-only view).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialToken {
    text: Vec<u8>,
    token_id: u32,
}

impl SpecialToken {
    /// The literal text matched during encode.
    pub fn text(&self) -> &[u8] {
        &self.text
    }

    /// The token id emitted when the text matches.
    pub fn token_id(&self) -> u32 {
        self.token_id
    }
}

/// `SparkTokenizerMergeCandidate`.
#[derive(Clone, Copy, PartialEq, Eq)]
struct MergeCandidate {
    rank: u32,
    left_symbol_index: u32,
    left_generation: u32,
    right_generation: u32,
}

/// `SparkTokenizerCompareMergeCandidates`: rank first, then left symbol
/// index. This is the BPE tie-breaking rule; it must match the C exactly.
fn compare_merge_candidates(left: &MergeCandidate, right: &MergeCandidate) -> std::cmp::Ordering {
    left.rank.cmp(&right.rank).then(left.left_symbol_index.cmp(&right.left_symbol_index))
}

/// `SparkTokenizerPieceCacheEntry`.
#[derive(Clone)]
struct PieceCacheEntry {
    hash: u64,
    piece_bytes: u32,
    token_offset: u32,
    token_count: u32,
    piece: [u8; PIECE_CACHE_INLINE_BYTES],
}

impl PieceCacheEntry {
    const EMPTY: Self = PieceCacheEntry {
        hash: PIECE_CACHE_EMPTY_HASH,
        piece_bytes: 0,
        token_offset: 0,
        token_count: 0,
        piece: [0u8; PIECE_CACHE_INLINE_BYTES],
    };
}

/// `SparkTokenizerWorkspace`: all mutable encode state. One workspace may be
/// reused across encode calls (its piece cache stays warm, including across
/// symbol-buffer growth) but never concurrently — enforced at compile time by
/// the `&mut Workspace` parameters.
pub struct Workspace {
    maximum_symbol_count: usize,
    symbol_token_ids: Vec<u32>,
    previous_symbol_indices: Vec<u32>,
    next_symbol_indices: Vec<u32>,
    symbol_generations: Vec<u32>,
    merge_heap: Vec<MergeCandidate>,
    heap_capacity: usize,
    piece_cache_entries: Vec<PieceCacheEntry>,
    piece_cache_token_pool: Vec<u32>,
    piece_cache_token_used: usize,
}

impl Workspace {
    /// `SparkTokenizerWorkspaceInitialize`: allocates symbol buffers for
    /// `maximum_symbol_count` symbols and the piece cache.
    pub fn new(maximum_symbol_count: usize) -> Result<Self, TokenizerError> {
        if maximum_symbol_count == 0 {
            return Err(TokenizerError::InvalidArgument("maximum_symbol_count is zero"));
        }
        let mut workspace = Workspace {
            maximum_symbol_count: 0,
            symbol_token_ids: Vec::new(),
            previous_symbol_indices: Vec::new(),
            next_symbol_indices: Vec::new(),
            symbol_generations: Vec::new(),
            merge_heap: Vec::new(),
            heap_capacity: 0,
            piece_cache_entries: Vec::new(),
            piece_cache_token_pool: Vec::new(),
            piece_cache_token_used: 0,
        };
        workspace.ensure_symbol_buffers(maximum_symbol_count)?;
        workspace.ensure_piece_cache();
        Ok(workspace)
    }

    /// `SparkTokenizerWorkspaceEnsureSymbolBuffers`: grows only the
    /// input-sized buffers, leaving the piece cache untouched so it stays
    /// warm.
    fn ensure_symbol_buffers(&mut self, maximum_symbol_count: usize) -> Result<(), TokenizerError> {
        if !self.symbol_token_ids.is_empty() && self.maximum_symbol_count >= maximum_symbol_count {
            return Ok(());
        }
        if maximum_symbol_count
            > (u32::MAX as usize - MERGE_HEAP_CAPACITY_SLACK) / MERGE_HEAP_CAPACITY_FACTOR
        {
            return Err(TokenizerError::CapacityExceeded);
        }
        self.maximum_symbol_count = maximum_symbol_count;
        self.heap_capacity =
            maximum_symbol_count * MERGE_HEAP_CAPACITY_FACTOR + MERGE_HEAP_CAPACITY_SLACK;
        self.symbol_token_ids = vec![0u32; maximum_symbol_count];
        self.previous_symbol_indices = vec![0u32; maximum_symbol_count];
        self.next_symbol_indices = vec![0u32; maximum_symbol_count];
        self.symbol_generations = vec![0u32; maximum_symbol_count];
        self.merge_heap = Vec::with_capacity(self.heap_capacity);
        Ok(())
    }

    /// `SparkTokenizerWorkspaceEnsurePieceCache`: allocates the cache once;
    /// never wipes a warm cache.
    fn ensure_piece_cache(&mut self) {
        if !self.piece_cache_entries.is_empty() && !self.piece_cache_token_pool.is_empty() {
            return;
        }
        self.piece_cache_token_used = 0;
        self.piece_cache_entries = vec![PieceCacheEntry::EMPTY; PIECE_CACHE_SLOT_COUNT];
        self.piece_cache_token_pool = vec![0u32; PIECE_CACHE_TOKEN_CAPACITY];
    }

    /// `SparkTokenizerHeapPush`.
    fn heap_push(&mut self, candidate: MergeCandidate) -> Result<(), TokenizerError> {
        if self.merge_heap.len() >= self.heap_capacity {
            return Err(TokenizerError::CapacityExceeded);
        }
        let mut heap_index = self.merge_heap.len();
        self.merge_heap.push(candidate);
        while heap_index > 0 {
            let parent_index = (heap_index - 1) / 2;
            if compare_merge_candidates(
                &self.merge_heap[heap_index],
                &self.merge_heap[parent_index],
            ) != std::cmp::Ordering::Less
            {
                break;
            }
            self.merge_heap.swap(heap_index, parent_index);
            heap_index = parent_index;
        }
        Ok(())
    }

    /// `SparkTokenizerHeapPop`.
    fn heap_pop(&mut self) -> Option<MergeCandidate> {
        if self.merge_heap.is_empty() {
            return None;
        }
        let candidate = self.merge_heap[0];
        let heap_count = self.merge_heap.len() - 1;
        if heap_count == 0 {
            self.merge_heap.pop();
            return Some(candidate);
        }
        self.merge_heap[0] = self.merge_heap[heap_count];
        self.merge_heap.pop();
        let mut heap_index = 0usize;
        loop {
            let left_child_index = heap_index * 2 + 1;
            let right_child_index = left_child_index + 1;
            if left_child_index >= heap_count {
                break;
            }
            let mut best_child_index = left_child_index;
            if right_child_index < heap_count
                && compare_merge_candidates(
                    &self.merge_heap[right_child_index],
                    &self.merge_heap[left_child_index],
                ) == std::cmp::Ordering::Less
            {
                best_child_index = right_child_index;
            }
            if compare_merge_candidates(
                &self.merge_heap[heap_index],
                &self.merge_heap[best_child_index],
            ) != std::cmp::Ordering::Greater
            {
                break;
            }
            self.merge_heap.swap(heap_index, best_child_index);
            heap_index = best_child_index;
        }
        Some(candidate)
    }
}

/// `SparkTokenizerPieceHash` (FNV-1a-64, remapping the empty sentinel).
fn piece_hash(text: &[u8]) -> u64 {
    let mut hash = 1_469_598_103_934_665_603u64;
    for &byte in text {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    if hash == PIECE_CACHE_EMPTY_HASH {
        hash = 1;
    }
    hash
}

/// `SparkTokenizerEncoding`: encode output. `token_ids` holds the first
/// `token_capacity` tokens; further tokens are counted in
/// `overflow_token_count`, exactly like the C caller-buffer contract.
#[derive(Debug, Clone, Default)]
pub struct Encoding {
    token_ids: Vec<u32>,
    token_capacity: usize,
    overflow_token_count: usize,
    invalid_segment_count: usize,
}

impl Encoding {
    /// An encoding that never overflows (capacity is effectively unbounded);
    /// used by the convenience entry points.
    pub fn unbounded() -> Self {
        Encoding {
            token_ids: Vec::new(),
            token_capacity: usize::MAX,
            overflow_token_count: 0,
            invalid_segment_count: 0,
        }
    }

    /// An encoding capped at `token_capacity` tokens, mirroring the C
    /// caller-provided `token_ids` buffer. Tokens beyond the capacity are
    /// counted in `overflow_token_count` and the final encode status is
    /// [`TokenizerError::CapacityExceeded`], while `token_ids` still holds
    /// the leading `token_capacity` tokens.
    pub fn with_capacity(token_capacity: usize) -> Self {
        Encoding {
            token_ids: Vec::with_capacity(token_capacity),
            token_capacity,
            overflow_token_count: 0,
            invalid_segment_count: 0,
        }
    }

    /// The encoded token ids (at most `token_capacity` of them).
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    /// `encoding.token_count`.
    pub fn token_count(&self) -> usize {
        self.token_ids.len()
    }

    /// `encoding.overflow_token_count`.
    pub fn overflow_token_count(&self) -> usize {
        self.overflow_token_count
    }

    /// `encoding.invalid_segment_count`.
    pub fn invalid_segment_count(&self) -> usize {
        self.invalid_segment_count
    }

    fn reset(&mut self) {
        self.token_ids.clear();
        self.overflow_token_count = 0;
        self.invalid_segment_count = 0;
    }

    /// `SparkTokenizerAppendTokenToEncoding`.
    fn append(&mut self, token_id: u32) {
        if self.token_ids.len() < self.token_capacity {
            self.token_ids.push(token_id);
        } else {
            self.overflow_token_count += 1;
        }
    }
}

/// `SparkTokenizer`: a byte-level BPE model. Immutable after loading; `Send +
/// Sync` automatically, so it can be shared across threads while each thread
/// uses its own [`Workspace`] (the C batch-worker call pattern).
pub struct Tokenizer {
    model_kind: u32,
    add_prefix_space: bool,
    byte_fallback: bool,
    has_unk_token: bool,
    unk_token_id: u32,
    maximum_token_id: u32,
    byte_level_use_regex: bool,
    token_text_by_id: Vec<Option<Vec<u8>>>,
    vocabulary: StringTable,
    merge_count: usize,
    merges: StringTable,
    special_tokens: Vec<SpecialToken>,
    byte_token_ids: [u32; 256],
    merge_pairs: MergePairTable,
}

impl Default for Tokenizer {
    /// `SparkTokenizerReset`: an empty byte-level tokenizer.
    fn default() -> Self {
        Tokenizer {
            model_kind: BPE_MODEL_KIND_BYTE_LEVEL,
            add_prefix_space: false,
            byte_fallback: false,
            has_unk_token: false,
            unk_token_id: 0,
            maximum_token_id: 0,
            byte_level_use_regex: true,
            token_text_by_id: Vec::new(),
            vocabulary: StringTable::with_entry_count(0),
            merge_count: 0,
            merges: StringTable::with_entry_count(0),
            special_tokens: Vec::new(),
            byte_token_ids: [NO_TOKEN_ID; 256],
            merge_pairs: MergePairTable::with_entry_count(0),
        }
    }
}

/// `SparkTokenizerMakeMergeKey`: `left + " " + right`.
fn make_merge_key(left_text: &[u8], right_text: &[u8]) -> Vec<u8> {
    let mut merge_key = Vec::with_capacity(left_text.len() + 1 + right_text.len());
    merge_key.extend_from_slice(left_text);
    merge_key.push(b' ');
    merge_key.extend_from_slice(right_text);
    merge_key
}

/// `SparkTokenizerParseMergeString`: split at the first space.
fn parse_merge_string(merge_text: &[u8]) -> Result<Vec<u8>, TokenizerError> {
    for (byte_index, &byte) in merge_text.iter().enumerate() {
        if byte == b' ' {
            return Ok(make_merge_key(&merge_text[..byte_index], &merge_text[byte_index + 1..]));
        }
    }
    Err(TokenizerError::Parse("merge string has no separator".into()))
}

/// `SparkTokenizerJsonSearchBooleanMemberRecursive`: depth-first search for a
/// boolean member, checking an object's own member before recursing into its
/// children in order, and arrays element by element.
fn json_search_boolean_member(value: &serde_json::Value, member_name: &str) -> Option<bool> {
    match value {
        serde_json::Value::Object(map) => {
            if let Some(member) = map.get(member_name) {
                if let Some(boolean) = member.as_bool() {
                    return Some(boolean);
                }
            }
            for member_value in map.values() {
                if let Some(found) = json_search_boolean_member(member_value, member_name) {
                    return Some(found);
                }
            }
            None
        }
        serde_json::Value::Array(elements) => {
            for element in elements {
                if let Some(found) = json_search_boolean_member(element, member_name) {
                    return Some(found);
                }
            }
            None
        }
        _ => None,
    }
}

/// Reads a JSON string member as raw UTF-8 bytes (`SparkJsonCopyString`).
fn json_string_bytes<'a>(
    value: &'a serde_json::Value,
    what: &str,
) -> Result<&'a [u8], TokenizerError> {
    value
        .as_str()
        .map(|text| text.as_bytes())
        .ok_or_else(|| TokenizerError::Parse(format!("expected string for {what}")))
}

/// Reads a JSON unsigned 32-bit integer member (`SparkJsonGetUInt32`).
fn json_u32(value: &serde_json::Value, what: &str) -> Result<u32, TokenizerError> {
    value
        .as_u64()
        .and_then(|number| u32::try_from(number).ok())
        .ok_or_else(|| TokenizerError::Parse(format!("expected u32 for {what}")))
}

impl Tokenizer {
    /// `SparkTokenizerLoadHuggingFaceJson`: loads a HuggingFace
    /// `tokenizer.json` (BPE model with `vocab`, `merges`, optional
    /// `unk_token`/`byte_fallback`, `pre_tokenizer`, `added_tokens`).
    ///
    /// The C tree walks its own JSON DOM; here `serde_json` supplies the DOM.
    /// Two inconsequential parser differences: duplicate object keys are
    /// collapsed by `serde_json` (the C reports `SPARK_STATUS_DUPLICATE` when
    /// inserting into the vocabulary table), and the recursive boolean-member
    /// search visits object members in `serde_json`'s map order rather than
    /// document order (only observable if a pre_tokenizer subtree declares
    /// the same flag twice with different values).
    #[allow(clippy::field_reassign_with_default)] // fields are populated by staged parse steps
    pub fn load_huggingface_json(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let bytes = std::fs::read(path.as_ref())?;
        let document: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| TokenizerError::Parse(error.to_string()))?;

        let model = document
            .get("model")
            .ok_or_else(|| TokenizerError::Parse("missing \"model\" member".into()))?;
        let vocabulary_object = model
            .get("vocab")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| TokenizerError::Parse("missing \"model.vocab\" object".into()))?;
        let merges_array = model
            .get("merges")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| TokenizerError::Parse("missing \"model.merges\" array".into()))?;

        let mut tokenizer = Tokenizer::default();

        // SparkTokenizerParseVocabulary.
        tokenizer.vocabulary = StringTable::with_entry_count(vocabulary_object.len());
        for (token_text, token_id_value) in vocabulary_object {
            let token_id = json_u32(token_id_value, "vocab value")?;
            tokenizer.vocabulary.insert(token_text.as_bytes(), token_id)?;
            if token_id > tokenizer.maximum_token_id {
                tokenizer.maximum_token_id = token_id;
            }
        }

        // SparkTokenizerBuildReverseVocabulary.
        if !tokenizer.vocabulary.entries.is_empty() {
            tokenizer.token_text_by_id = vec![None; tokenizer.maximum_token_id as usize + 1];
            for entry in &tokenizer.vocabulary.entries {
                if let Some(text) = &entry.text {
                    if entry.value <= tokenizer.maximum_token_id {
                        tokenizer.token_text_by_id[entry.value as usize] = Some(text.clone());
                    }
                }
            }
        }

        // unk_token lookup.
        if let Some(unknown_token) = model.get("unk_token").and_then(serde_json::Value::as_str) {
            if let Some(entry) = tokenizer.vocabulary.find(unknown_token.as_bytes()) {
                tokenizer.has_unk_token = true;
                tokenizer.unk_token_id = entry.value;
            }
        }

        // SparkTokenizerBuildByteTokenTable.
        tokenizer.build_byte_token_table();

        // SparkTokenizerParseMerges.
        tokenizer.merge_count = merges_array.len();
        tokenizer.merges = StringTable::with_entry_count(merges_array.len());
        for (merge_index, merge_value) in merges_array.iter().enumerate() {
            let merge_key = match merge_value {
                serde_json::Value::String(merge_text) => parse_merge_string(merge_text.as_bytes())?,
                serde_json::Value::Array(elements) => {
                    // SparkTokenizerParseMergeArray.
                    if elements.len() != 2 {
                        return Err(TokenizerError::Parse(
                            "merge array must have two elements".into(),
                        ));
                    }
                    let left_text = json_string_bytes(&elements[0], "merge left")?;
                    let right_text = json_string_bytes(&elements[1], "merge right")?;
                    make_merge_key(left_text, right_text)
                }
                _ => {
                    return Err(TokenizerError::Parse("merge must be string or pair".into()));
                }
            };
            match tokenizer.merges.insert(&merge_key, merge_index as u32) {
                Ok(()) => {}
                Err(TokenizerError::Duplicate(_)) => tokenizer.merges.push_empty_slot(),
                Err(error) => return Err(error),
            }
        }

        // SparkTokenizerBuildMergePairTableFromMergeKeys.
        tokenizer.build_merge_pair_table()?;

        // SparkTokenizerParseAddedTokens (optional member).
        if let Some(added_tokens) =
            document.get("added_tokens").and_then(serde_json::Value::as_array)
        {
            for token_object in added_tokens {
                let serde_json::Value::Object(_) = token_object else {
                    continue;
                };
                let is_special = token_object
                    .get("special")
                    .map(|value| {
                        value
                            .as_bool()
                            .ok_or_else(|| TokenizerError::Parse("\"special\" not boolean".into()))
                    })
                    .transpose()?
                    .unwrap_or(false);
                if !is_special {
                    continue;
                }
                let (Some(id_value), Some(content_value)) =
                    (token_object.get("id"), token_object.get("content"))
                else {
                    continue;
                };
                let token_id = json_u32(id_value, "added token id")?;
                let content = json_string_bytes(content_value, "added token content")?;
                tokenizer.special_tokens.push(SpecialToken { text: content.to_vec(), token_id });
            }
        }

        // byte_fallback (optional boolean, default false).
        tokenizer.byte_fallback =
            model.get("byte_fallback").and_then(serde_json::Value::as_bool).unwrap_or(false);

        // pre_tokenizer flags (recursive search; defaults false / true).
        if let Some(pre_tokenizer) = document.get("pre_tokenizer") {
            if let Some(add_prefix_space) =
                json_search_boolean_member(pre_tokenizer, "add_prefix_space")
            {
                tokenizer.add_prefix_space = add_prefix_space;
            }
            if let Some(use_regex) = json_search_boolean_member(pre_tokenizer, "use_regex") {
                tokenizer.byte_level_use_regex = use_regex;
            }
        }

        tokenizer.sort_special_tokens();
        Ok(tokenizer)
    }

    /// `SparkTokenizerBuildByteTokenTable`.
    fn build_byte_token_table(&mut self) {
        for byte_value in 0..256u32 {
            self.byte_token_ids[byte_value as usize] = NO_TOKEN_ID;
            let mut encoded_text = Vec::with_capacity(3);
            append_utf8_code_point(&mut encoded_text, byte_to_unicode_code_point(byte_value));
            if let Some(entry) = self.vocabulary.find(&encoded_text) {
                self.byte_token_ids[byte_value as usize] = entry.value;
            } else if self.has_unk_token {
                self.byte_token_ids[byte_value as usize] = self.unk_token_id;
            }
        }
    }

    /// `SparkTokenizerBuildMergePairTableFromMergeKeys`.
    fn build_merge_pair_table(&mut self) -> Result<(), TokenizerError> {
        self.merge_pairs = MergePairTable::with_entry_count(self.merge_count);
        for merge_entry in &self.merges.entries {
            let Some(merge_text) = &merge_entry.text else {
                continue;
            };
            let mut separator_index = 0usize;
            while separator_index < merge_text.len() && merge_text[separator_index] != b' ' {
                separator_index += 1;
            }
            if separator_index == 0 || separator_index >= merge_text.len() {
                continue;
            }
            let Some(left_entry) = self.vocabulary.find(&merge_text[..separator_index]) else {
                continue;
            };
            let left_token_id = left_entry.value;
            let Some(right_entry) = self.vocabulary.find(&merge_text[separator_index + 1..]) else {
                continue;
            };
            let right_token_id = right_entry.value;
            let mut merged_text = Vec::with_capacity(merge_text.len() - 1);
            merged_text.extend_from_slice(&merge_text[..separator_index]);
            merged_text.extend_from_slice(&merge_text[separator_index + 1..]);
            let Some(merged_entry) = self.vocabulary.find(&merged_text) else {
                continue;
            };
            let pair = FastMergePair {
                left_token_id,
                right_token_id,
                merged_token_id: merged_entry.value,
                rank: merge_entry.value,
            };
            match self.merge_pairs.insert(pair) {
                Ok(()) | Err(TokenizerError::Duplicate(_)) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    /// `SparkTokenizerSortSpecialTokens`: longest text first, ties broken by
    /// ascending token id.
    fn sort_special_tokens(&mut self) {
        self.special_tokens.sort_by(|left, right| {
            right.text.len().cmp(&left.text.len()).then(left.token_id.cmp(&right.token_id))
        });
    }

    /// `SparkTokenizerFindTokenId`.
    pub fn find_token_id(&self, text: &[u8]) -> Result<u32, TokenizerError> {
        self.vocabulary
            .find(text)
            .map(|entry| entry.value)
            .ok_or(TokenizerError::NotFound("token text"))
    }

    /// `tokenizer.vocabulary_count`.
    pub fn vocabulary_count(&self) -> usize {
        self.vocabulary.entries.len()
    }

    /// `tokenizer.merge_count` (the number of merge strings in the model,
    /// including skipped duplicates).
    pub fn merge_count(&self) -> usize {
        self.merge_count
    }

    /// `tokenizer.maximum_token_id`.
    pub fn maximum_token_id(&self) -> u32 {
        self.maximum_token_id
    }

    /// `tokenizer.byte_fallback`.
    pub fn byte_fallback(&self) -> bool {
        self.byte_fallback
    }

    /// `tokenizer.add_prefix_space`.
    pub fn add_prefix_space(&self) -> bool {
        self.add_prefix_space
    }

    /// The UTF-8 token text for `token_id` (`tokenizer.token_text_by_id`),
    /// or `None` when the id has no vocabulary entry.
    pub fn token_text(&self, token_id: u32) -> Option<&[u8]> {
        self.token_text_by_id.get(token_id as usize)?.as_deref()
    }

    /// The special tokens, in match order (longest first).
    pub fn special_tokens(&self) -> &[SpecialToken] {
        &self.special_tokens
    }
}

// ---------------------------------------------------------------------------
// Compiled file format (`SparkTokenizerSaveCompiledFile` /
// `SparkTokenizerLoadCompiledFile`). Integers are written little-endian,
// matching the C `fwrite` layout on the little-endian targets the C tree
// supports, so files are interchangeable with the C tools there.
// ---------------------------------------------------------------------------

fn write_u64(buffer: &mut Vec<u8>, value: u64) {
    buffer.extend_from_slice(&value.to_le_bytes());
}

fn write_u32(buffer: &mut Vec<u8>, value: u32) {
    buffer.extend_from_slice(&value.to_le_bytes());
}

struct CompiledReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> CompiledReader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], TokenizerError> {
        let end = self
            .position
            .checked_add(count)
            .filter(|end| *end <= self.bytes.len())
            .ok_or(TokenizerError::Schema("truncated compiled file".into()))?;
        let slice = &self.bytes[self.position..end];
        self.position = end;
        Ok(slice)
    }

    fn read_u64(&mut self) -> Result<u64, TokenizerError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().map_err(|_| TokenizerError::Internal("u64"))?))
    }

    fn read_u32(&mut self) -> Result<u32, TokenizerError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().map_err(|_| TokenizerError::Internal("u32"))?))
    }
}

impl Tokenizer {
    /// `SparkTokenizerSaveCompiledFile`.
    pub fn save_compiled_file(&self, path: impl AsRef<Path>) -> Result<(), TokenizerError> {
        let mut buffer = Vec::new();
        write_u64(&mut buffer, COMPILED_FILE_MAGIC);
        write_u32(&mut buffer, COMPILED_FILE_VERSION);
        write_u32(&mut buffer, self.model_kind);
        write_u32(&mut buffer, u32::from(self.add_prefix_space));
        write_u32(&mut buffer, u32::from(self.byte_fallback));
        write_u32(&mut buffer, u32::from(self.has_unk_token));
        write_u32(&mut buffer, self.unk_token_id);
        write_u32(&mut buffer, self.maximum_token_id);
        write_u32(&mut buffer, u32::from(self.byte_level_use_regex));
        write_u32(&mut buffer, self.vocabulary.entries.len() as u32);
        write_u32(&mut buffer, self.merge_pairs.pairs.len() as u32);
        write_u32(&mut buffer, self.special_tokens.len() as u32);
        for entry in &self.vocabulary.entries {
            let text = entry.text.as_deref().unwrap_or(&[]);
            write_u32(&mut buffer, entry.value);
            write_u32(&mut buffer, text.len() as u32);
            buffer.extend_from_slice(text);
        }
        for pair in &self.merge_pairs.pairs {
            write_u32(&mut buffer, pair.left_token_id);
            write_u32(&mut buffer, pair.right_token_id);
            write_u32(&mut buffer, pair.merged_token_id);
            write_u32(&mut buffer, pair.rank);
        }
        for special_token in &self.special_tokens {
            write_u32(&mut buffer, special_token.token_id);
            write_u32(&mut buffer, special_token.text.len() as u32);
            buffer.extend_from_slice(&special_token.text);
        }
        std::fs::write(path, buffer)?;
        Ok(())
    }

    /// `SparkTokenizerLoadCompiledFile`.
    #[allow(clippy::field_reassign_with_default)] // header fields are read sequentially from the file
    pub fn load_compiled_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let bytes = std::fs::read(path.as_ref())?;
        let mut reader = CompiledReader { bytes: &bytes, position: 0 };

        let magic = reader.read_u64()?;
        let version = reader.read_u32()?;
        if magic != COMPILED_FILE_MAGIC || version != COMPILED_FILE_VERSION {
            return Err(TokenizerError::Schema("bad compiled file magic or version".into()));
        }

        let mut tokenizer = Tokenizer::default();
        tokenizer.model_kind = reader.read_u32()?;
        tokenizer.add_prefix_space = reader.read_u32()? != 0;
        tokenizer.byte_fallback = reader.read_u32()? != 0;
        tokenizer.has_unk_token = reader.read_u32()? != 0;
        tokenizer.unk_token_id = reader.read_u32()?;
        tokenizer.maximum_token_id = reader.read_u32()?;
        tokenizer.byte_level_use_regex = reader.read_u32()? != 0;
        let vocabulary_count = reader.read_u32()? as usize;
        let fast_merge_pair_count = reader.read_u32()? as usize;
        let special_token_count = reader.read_u32()? as usize;

        tokenizer.vocabulary = StringTable::with_entry_count(vocabulary_count);
        for _ in 0..vocabulary_count {
            let token_id = reader.read_u32()?;
            let text_bytes = reader.read_u32()? as usize;
            let text = reader.take(text_bytes)?;
            tokenizer.vocabulary.insert(text, token_id)?;
        }

        // SparkTokenizerBuildReverseVocabulary.
        if vocabulary_count != 0 {
            tokenizer.token_text_by_id = vec![None; tokenizer.maximum_token_id as usize + 1];
            for entry in &tokenizer.vocabulary.entries {
                if let Some(text) = &entry.text {
                    if entry.value <= tokenizer.maximum_token_id {
                        tokenizer.token_text_by_id[entry.value as usize] = Some(text.clone());
                    }
                }
            }
        }

        tokenizer.build_byte_token_table();

        tokenizer.merge_pairs = MergePairTable::with_entry_count(fast_merge_pair_count);
        for _ in 0..fast_merge_pair_count {
            let pair = FastMergePair {
                left_token_id: reader.read_u32()?,
                right_token_id: reader.read_u32()?,
                merged_token_id: reader.read_u32()?,
                rank: reader.read_u32()?,
            };
            tokenizer.merge_pairs.insert(pair)?;
        }

        for _ in 0..special_token_count {
            let token_id = reader.read_u32()?;
            let text_bytes = reader.read_u32()? as usize;
            let text = reader.take(text_bytes)?;
            tokenizer.special_tokens.push(SpecialToken { text: text.to_vec(), token_id });
        }

        tokenizer.sort_special_tokens();
        Ok(tokenizer)
    }
}

// ---------------------------------------------------------------------------
// Encode path.
// ---------------------------------------------------------------------------

/// `g_spark_tokenizer_byte_class`: 1 whitespace, 2 letter or any byte >=
/// 0x80, 3 digit, 4 everything else. Copied verbatim from the C table.
#[rustfmt::skip]
const BYTE_CLASS: [u8; 256] = [
    4,4,4,4,4,4,4,4,4,1,1,1,1,1,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,
    4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,
    4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
];

/// `SparkTokenizerIsAsciiWhitespace`.
fn is_ascii_whitespace(value: u8) -> bool {
    matches!(value, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

/// `SparkTokenizerScanClassRun`: advance from `start` while the byte class
/// equals `target_class`, returning the exclusive end. A plain counted loop
/// over a byte array, kept auto-vectorizable like the C.
fn scan_class_run(text: &[u8], mut scan_position: usize, target_class: u8) -> usize {
    while scan_position < text.len() && BYTE_CLASS[text[scan_position] as usize] == target_class {
        scan_position += 1;
    }
    scan_position
}

/// `SparkTokenizerMatchesContraction`.
fn matches_contraction(text: &[u8], position: usize) -> Option<usize> {
    if position + 1 >= text.len() || text[position] != b'\'' {
        return None;
    }
    let mut first = text[position + 1];
    if first.is_ascii_uppercase() {
        first = first - b'A' + b'a';
    }
    if matches!(first, b's' | b't' | b'm' | b'd') {
        return Some(2);
    }
    if position + 2 >= text.len() {
        return None;
    }
    let mut second = text[position + 2];
    if second.is_ascii_uppercase() {
        second = second - b'A' + b'a';
    }
    let third = first;
    if (third == b'r' && second == b'e')
        || (third == b'v' && second == b'e')
        || (third == b'l' && second == b'l')
    {
        return Some(3);
    }
    None
}

/// `SparkTokenizerFindNextRegexPiece`: returns `(piece_start, piece_bytes)`.
fn find_next_regex_piece(text: &[u8], position: usize) -> Option<(usize, usize)> {
    if position >= text.len() {
        return None;
    }
    let piece_start = position;

    if let Some(piece_bytes) = matches_contraction(text, position) {
        return Some((piece_start, piece_bytes));
    }

    if text[position] == b' '
        && position + 1 < text.len()
        && !is_ascii_whitespace(text[position + 1])
    {
        let mut scan_position = position + 1;
        let class_id = BYTE_CLASS[text[scan_position] as usize];
        // A leading-space run of letters or digits has no interior boundary
        // and scans in one sweep. An "other" (class 4) run can be cut short
        // by a contraction apostrophe, so it is scanned byte by byte with the
        // contraction check.
        if class_id == 2 || class_id == 3 {
            scan_position = scan_class_run(text, scan_position, class_id);
        } else {
            while scan_position < text.len() && BYTE_CLASS[text[scan_position] as usize] == class_id
            {
                if matches_contraction(text, scan_position).is_some() {
                    break;
                }
                scan_position += 1;
            }
        }
        return Some((piece_start, scan_position - position));
    }

    let class_id = BYTE_CLASS[text[position] as usize];
    let scan_position = if class_id == 2 || class_id == 3 || class_id == 1 {
        // Letter, digit, and whitespace runs have no interior contraction
        // break, so they scan in a single sweep.
        scan_class_run(text, position, class_id)
    } else {
        let mut scan_position = position;
        while scan_position < text.len() && BYTE_CLASS[text[scan_position] as usize] == class_id {
            if matches_contraction(text, scan_position).is_some() {
                break;
            }
            scan_position += 1;
        }
        scan_position
    };
    let piece_bytes = scan_position - position;
    if piece_bytes == 0 {
        return None;
    }
    Some((piece_start, piece_bytes))
}

impl Tokenizer {
    /// `SparkTokenizerFindSpecialTokenAt`: returns `(token_id, matched_bytes)`.
    fn find_special_token_at(&self, text: &[u8]) -> Option<(u32, usize)> {
        for special_token in &self.special_tokens {
            if special_token.text.len() <= text.len() && text.starts_with(&special_token.text) {
                return Some((special_token.token_id, special_token.text.len()));
            }
        }
        None
    }

    /// `SparkTokenizerTokenIdIsSpecial`.
    fn token_id_is_special(&self, token_id: u32) -> bool {
        self.special_tokens.iter().any(|special_token| special_token.token_id == token_id)
    }

    /// `SparkTokenizerPushPairCandidate`.
    fn push_pair_candidate(
        &self,
        workspace: &mut Workspace,
        left_symbol_index: u32,
    ) -> Result<(), TokenizerError> {
        if left_symbol_index == SYMBOL_NONE {
            return Ok(());
        }
        let right_symbol_index = workspace.next_symbol_indices[left_symbol_index as usize];
        if right_symbol_index == SYMBOL_NONE {
            return Ok(());
        }
        let Some(merge_pair) = self.merge_pairs.find(
            workspace.symbol_token_ids[left_symbol_index as usize],
            workspace.symbol_token_ids[right_symbol_index as usize],
        ) else {
            return Ok(());
        };
        let candidate = MergeCandidate {
            rank: merge_pair.rank,
            left_symbol_index,
            left_generation: workspace.symbol_generations[left_symbol_index as usize],
            right_generation: workspace.symbol_generations[right_symbol_index as usize],
        };
        workspace.heap_push(candidate)
    }

    /// `SparkTokenizerEncodeByteLevelPieceUncached`: the merge loop. On
    /// success returns the emitted token ids; on an unmapped byte returns
    /// `Err(NotFound)` (the C sets `invalid_out` and the wrapper counts an
    /// invalid segment).
    fn encode_byte_level_piece_uncached(
        &self,
        text: &[u8],
        workspace: &mut Workspace,
    ) -> Result<Vec<u32>, TokenizerError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        if text.len() > workspace.maximum_symbol_count {
            return Err(TokenizerError::CapacityExceeded);
        }

        workspace.merge_heap.clear();
        let text_bytes = text.len();
        for (symbol_index, &byte) in text.iter().enumerate() {
            let token_id = self.byte_token_ids[byte as usize];
            if token_id == NO_TOKEN_ID {
                return Err(TokenizerError::NotFound("byte token"));
            }
            workspace.symbol_token_ids[symbol_index] = token_id;
            workspace.previous_symbol_indices[symbol_index] =
                if symbol_index == 0 { SYMBOL_NONE } else { symbol_index as u32 - 1 };
            workspace.next_symbol_indices[symbol_index] =
                if symbol_index + 1 < text_bytes { symbol_index as u32 + 1 } else { SYMBOL_NONE };
            workspace.symbol_generations[symbol_index] = 1;
        }

        for symbol_index in 0..text_bytes.saturating_sub(1) {
            self.push_pair_candidate(workspace, symbol_index as u32)?;
        }

        let head_symbol_index = 0u32;
        let mut live_symbol_count = text_bytes;
        while live_symbol_count > 1 {
            let Some(candidate) = workspace.heap_pop() else {
                break;
            };
            let left_symbol_index = candidate.left_symbol_index;
            if left_symbol_index as usize >= text_bytes
                || workspace.symbol_generations[left_symbol_index as usize]
                    != candidate.left_generation
            {
                continue;
            }
            let right_symbol_index = workspace.next_symbol_indices[left_symbol_index as usize];
            if right_symbol_index == SYMBOL_NONE
                || workspace.symbol_generations[right_symbol_index as usize]
                    != candidate.right_generation
            {
                continue;
            }
            let Some(merge_pair) = self.merge_pairs.find(
                workspace.symbol_token_ids[left_symbol_index as usize],
                workspace.symbol_token_ids[right_symbol_index as usize],
            ) else {
                continue;
            };
            if merge_pair.rank != candidate.rank {
                continue;
            }
            let merged_token_id = merge_pair.merged_token_id;

            let previous_symbol_index =
                workspace.previous_symbol_indices[left_symbol_index as usize];
            let next_symbol_index = workspace.next_symbol_indices[right_symbol_index as usize];
            workspace.symbol_token_ids[left_symbol_index as usize] = merged_token_id;
            workspace.next_symbol_indices[left_symbol_index as usize] = next_symbol_index;
            if next_symbol_index != SYMBOL_NONE {
                workspace.previous_symbol_indices[next_symbol_index as usize] = left_symbol_index;
            }
            workspace.symbol_generations[left_symbol_index as usize] += 1;
            workspace.symbol_generations[right_symbol_index as usize] += 1;
            live_symbol_count -= 1;

            self.push_pair_candidate(workspace, previous_symbol_index)?;
            self.push_pair_candidate(workspace, left_symbol_index)?;
        }

        let mut token_ids = Vec::with_capacity(live_symbol_count);
        let mut symbol_index = head_symbol_index;
        while symbol_index != SYMBOL_NONE {
            token_ids.push(workspace.symbol_token_ids[symbol_index as usize]);
            symbol_index = workspace.next_symbol_indices[symbol_index as usize];
        }
        Ok(token_ids)
    }

    /// `SparkTokenizerPieceCacheLookup`: returns the probing stop slot —
    /// either the matching entry or the first empty slot (for insert), or
    /// `None` when the table is full.
    fn piece_cache_lookup(workspace: &Workspace, text: &[u8], hash: u64) -> Option<usize> {
        let slot_count = workspace.piece_cache_entries.len();
        if slot_count == 0 {
            return None;
        }
        let mut slot = (hash as usize) & (slot_count - 1);
        for _ in 0..slot_count {
            let entry = &workspace.piece_cache_entries[slot];
            if entry.hash == PIECE_CACHE_EMPTY_HASH {
                return Some(slot);
            }
            if entry.hash == hash
                && entry.piece_bytes as usize == text.len()
                && &entry.piece[..text.len()] == text
            {
                return Some(slot);
            }
            slot = (slot + 1) & (slot_count - 1);
        }
        None
    }

    /// `SparkTokenizerEncodeByteLevelPiece`.
    fn encode_byte_level_piece(
        &self,
        text: &[u8],
        encode_flags: u32,
        workspace: &mut Workspace,
        encoding: &mut Encoding,
    ) -> Result<(), TokenizerError> {
        if text.is_empty() {
            return Ok(());
        }
        // Pieces too long to cache inline, when the cache is unavailable, or
        // when the caller disables the cache, encode directly through the
        // merge loop.
        if text.len() > PIECE_CACHE_INLINE_BYTES
            || (encode_flags & ENCODE_FLAG_DISABLE_PIECE_CACHE) != 0
            || workspace.piece_cache_entries.is_empty()
            || workspace.piece_cache_token_pool.is_empty()
        {
            let produced = match self.encode_byte_level_piece_uncached(text, workspace) {
                Ok(token_ids) => token_ids,
                Err(error) => {
                    if matches!(error, TokenizerError::NotFound(_)) {
                        encoding.invalid_segment_count += 1;
                    }
                    return Err(error);
                }
            };
            for token_id in produced {
                encoding.append(token_id);
            }
            return Ok(());
        }

        let hash = piece_hash(text);
        let slot = Self::piece_cache_lookup(workspace, text, hash);
        if let Some(slot_index) = slot {
            let entry = &workspace.piece_cache_entries[slot_index];
            if entry.hash != PIECE_CACHE_EMPTY_HASH {
                // Hit: replay the memoized token sequence from the pool.
                let token_offset = entry.token_offset as usize;
                let token_count = entry.token_count as usize;
                for token_index in 0..token_count {
                    encoding.append(workspace.piece_cache_token_pool[token_offset + token_index]);
                }
                return Ok(());
            }
        }
        // Miss: encode once, then, if it fits and a free slot exists, commit
        // the entry so future occurrences hit.
        let produced = match self.encode_byte_level_piece_uncached(text, workspace) {
            Ok(token_ids) => token_ids,
            Err(error) => {
                if matches!(error, TokenizerError::NotFound(_)) {
                    encoding.invalid_segment_count += 1;
                }
                return Err(error);
            }
        };
        for &token_id in &produced {
            encoding.append(token_id);
        }
        if let Some(slot_index) = slot {
            let pool_start = workspace.piece_cache_token_used;
            let pool_room = workspace.piece_cache_token_pool.len() - pool_start;
            if produced.len() <= pool_room {
                workspace.piece_cache_token_pool[pool_start..pool_start + produced.len()]
                    .copy_from_slice(&produced);
                let entry = &mut workspace.piece_cache_entries[slot_index];
                entry.hash = hash;
                entry.piece_bytes = text.len() as u32;
                entry.token_offset = pool_start as u32;
                entry.token_count = produced.len() as u32;
                entry.piece[..text.len()].copy_from_slice(text);
                workspace.piece_cache_token_used = pool_start + produced.len();
            }
        }
        Ok(())
    }

    /// `SparkTokenizerEncodeRegularSegmentWithWorkspace`.
    fn encode_regular_segment(
        &self,
        text: &[u8],
        encode_flags: u32,
        workspace: &mut Workspace,
        encoding: &mut Encoding,
    ) -> Result<(), TokenizerError> {
        if text.is_empty() {
            return Ok(());
        }
        if !self.byte_level_use_regex
            || (encode_flags & ENCODE_FLAG_DISABLE_REGEX_PRETOKENIZATION) != 0
        {
            return self.encode_byte_level_piece(text, encode_flags, workspace, encoding);
        }

        let mut position = 0usize;
        while position < text.len() {
            let Some((piece_start, piece_bytes)) = find_next_regex_piece(text, position) else {
                return Err(TokenizerError::Parse("regex pretokenization stalled".into()));
            };
            if let Err(error) = self.encode_byte_level_piece(
                &text[piece_start..piece_start + piece_bytes],
                encode_flags,
                workspace,
                encoding,
            ) {
                // Matches the C: any failing piece bumps the invalid-segment
                // counter here (in addition to the bump inside
                // encode_byte_level_piece on the unmapped-byte path).
                encoding.invalid_segment_count += 1;
                return Err(error);
            }
            position = piece_start + piece_bytes;
        }
        Ok(())
    }

    /// `SparkTokenizerEncodeUtf8WithWorkspace`.
    ///
    /// On success the encoding holds the token ids; when tokens overflowed
    /// the encoding capacity this returns [`TokenizerError::CapacityExceeded`]
    /// with the encoding still populated (the C contract).
    pub fn encode_with_workspace(
        &self,
        text: &[u8],
        encode_flags: u32,
        workspace: &mut Workspace,
        encoding: &mut Encoding,
    ) -> Result<(), TokenizerError> {
        if (encode_flags & !ENCODE_KNOWN_FLAGS) != 0 {
            return Err(TokenizerError::InvalidArgument("unknown encode flags"));
        }
        workspace.ensure_symbol_buffers(text.len() + 1)?;
        workspace.ensure_piece_cache();

        encoding.reset();
        let add_prefix_space = ((encode_flags & ENCODE_FLAG_ADD_PREFIX_SPACE) != 0
            || self.add_prefix_space)
            && (text.is_empty() || text[0] != b' ');
        if add_prefix_space {
            self.encode_regular_segment(b" ", encode_flags, workspace, encoding)?;
        }

        let mut position = 0usize;
        let mut segment_start = 0usize;
        while position < text.len() {
            if (encode_flags & ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH) == 0 {
                if let Some((special_token_id, matched_text_bytes)) =
                    self.find_special_token_at(&text[position..])
                {
                    self.encode_regular_segment(
                        &text[segment_start..position],
                        encode_flags,
                        workspace,
                        encoding,
                    )?;
                    encoding.append(special_token_id);
                    position += matched_text_bytes;
                    segment_start = position;
                    continue;
                }
            }
            position += 1;
        }
        self.encode_regular_segment(&text[segment_start..], encode_flags, workspace, encoding)?;
        if encoding.overflow_token_count != 0 {
            return Err(TokenizerError::CapacityExceeded);
        }
        Ok(())
    }

    /// `SparkTokenizerEncodeUtf8`: encodes with a temporary workspace.
    /// Equivalent to [`Tokenizer::encode_with_workspace`] with a fresh
    /// [`Workspace`]; callers encoding many requests should keep a workspace
    /// to preserve the warm piece cache.
    pub fn encode(&self, text: &[u8], encode_flags: u32) -> Result<Encoding, TokenizerError> {
        let maximum_symbol_count = text.len() + 1;
        let mut workspace = Workspace::new(maximum_symbol_count)?;
        // The C caller supplies the encoding capacity; this convenience
        // wrapper uses an unbounded encoding so overflow never occurs here.
        let mut encoding = Encoding::unbounded();
        self.encode_with_workspace(text, encode_flags, &mut workspace, &mut encoding)?;
        Ok(encoding)
    }
}

// ---------------------------------------------------------------------------
// Decode path.
// ---------------------------------------------------------------------------

impl Tokenizer {
    /// `SparkTokenizerDecodeOneTokenText` applied to one token id, appending
    /// to `text` with the C per-byte capacity rule (`text_bytes + 1 >=
    /// text_capacity` fails, leaving room for the C NUL terminator).
    fn decode_one_token(
        &self,
        token_id: u32,
        text: &mut Vec<u8>,
        text_capacity: usize,
    ) -> Result<(), TokenizerError> {
        let Some(token_text) = self.token_text(token_id) else {
            return Err(TokenizerError::InvalidArgument("token id has no vocabulary text"));
        };
        let mut position = 0usize;
        while position < token_text.len() {
            let code_point = read_utf8_code_point(token_text, &mut position)
                .ok_or_else(|| TokenizerError::Parse("invalid UTF-8 in token text".into()))?;
            let decoded_byte = unicode_code_point_to_byte(code_point).ok_or_else(|| {
                TokenizerError::Parse("unmappable code point in token text".into())
            })?;
            if text.len() + 1 >= text_capacity {
                return Err(TokenizerError::CapacityExceeded);
            }
            text.push(decoded_byte);
        }
        Ok(())
    }

    /// `SparkTokenizerDecodeTokenIds` with the C caller-buffer capacity
    /// contract: `text_capacity` must be nonzero and decoding fails with
    /// [`TokenizerError::CapacityExceeded`] when the decoded bytes would not
    /// fit (including the C NUL terminator's byte). Unknown token ids fail
    /// with [`TokenizerError::InvalidArgument`].
    pub fn decode_with_capacity(
        &self,
        token_ids: &[u32],
        decode_flags: u32,
        text_capacity: usize,
    ) -> Result<Vec<u8>, TokenizerError> {
        if (decode_flags & !DECODE_KNOWN_FLAGS) != 0 {
            return Err(TokenizerError::InvalidArgument("unknown decode flags"));
        }
        if text_capacity == 0 {
            return Err(TokenizerError::InvalidArgument("text capacity is zero"));
        }
        let mut text = Vec::new();
        for &token_id in token_ids {
            if (decode_flags & DECODE_FLAG_SKIP_SPECIAL_TOKENS) != 0
                && self.token_id_is_special(token_id)
            {
                continue;
            }
            if token_id > self.maximum_token_id {
                return Err(TokenizerError::InvalidArgument("token id above maximum"));
            }
            self.decode_one_token(token_id, &mut text, text_capacity)?;
        }
        Ok(text)
    }

    /// Decode with no capacity limit (the common case).
    pub fn decode(&self, token_ids: &[u32], decode_flags: u32) -> Result<Vec<u8>, TokenizerError> {
        self.decode_with_capacity(token_ids, decode_flags, usize::MAX)
    }
}

// ---------------------------------------------------------------------------
// Batch encode (`SparkTokenizerEncodeBatchUtf8` /
// `SparkTokenizerEncodeBatchUtf8Configured`).
// ---------------------------------------------------------------------------

/// Batch encode output, mirroring the C caller-buffer layout: `token_ids` is
/// `text_count * token_stride` ids (text `i` starts at `i * token_stride`,
/// with `token_counts[i]` valid ids), plus per-text overflow and invalid
/// segment counts.
#[derive(Debug, Clone, Default)]
pub struct BatchEncoding {
    token_stride: usize,
    token_ids: Vec<u32>,
    token_counts: Vec<usize>,
    overflow_token_counts: Vec<usize>,
    invalid_segment_counts: Vec<usize>,
}

impl BatchEncoding {
    /// The stride each text was given in `token_ids`.
    pub fn token_stride(&self) -> usize {
        self.token_stride
    }

    /// All token ids, `token_stride` per text.
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    /// The valid token ids for text `text_index`.
    pub fn tokens_of(&self, text_index: usize) -> &[u32] {
        let start = text_index * self.token_stride;
        &self.token_ids[start..start + self.token_counts[text_index]]
    }

    /// Per-text token counts.
    pub fn token_counts(&self) -> &[usize] {
        &self.token_counts
    }

    /// Per-text overflow token counts.
    pub fn overflow_token_counts(&self) -> &[usize] {
        &self.overflow_token_counts
    }

    /// Per-text invalid segment counts.
    pub fn invalid_segment_counts(&self) -> &[usize] {
        &self.invalid_segment_counts
    }
}

impl Tokenizer {
    /// `SparkTokenizerBatchWorkerMain`: encodes `texts[first..first + count]`
    /// with one workspace, stopping at the first encode error. Returns the
    /// per-text encodings produced before any error, plus the error status.
    fn batch_worker(
        &self,
        texts: &[&[u8]],
        first_text_index: usize,
        text_count: usize,
        encode_flags: u32,
        token_stride: usize,
    ) -> (Vec<Encoding>, Result<(), TokenizerError>) {
        let mut maximum_text_bytes = 1usize;
        for local_index in 0..text_count {
            let text_bytes = texts[first_text_index + local_index].len() + 1;
            if text_bytes > maximum_text_bytes {
                maximum_text_bytes = text_bytes;
            }
        }
        let mut encodings = Vec::with_capacity(text_count);
        let mut status = Ok(());
        let mut workspace = match Workspace::new(maximum_text_bytes) {
            Ok(workspace) => workspace,
            Err(error) => return (encodings, Err(error)),
        };
        for local_index in 0..text_count {
            let text = texts[first_text_index + local_index];
            let mut encoding = Encoding::with_capacity(token_stride);
            if let Err(error) =
                self.encode_with_workspace(text, encode_flags, &mut workspace, &mut encoding)
            {
                encodings.push(encoding);
                status = Err(error);
                break;
            }
            encodings.push(encoding);
        }
        (encodings, status)
    }

    /// `SparkTokenizerEncodeBatchUtf8ConfiguredInternal`.
    ///
    /// Worker distribution matches the C exactly: `worker_count == 0` means
    /// [`MAX_PARALLEL_WORKER_COUNT`], capped to the text count; workers get
    /// `count / workers` texts with the remainder going to the leading
    /// workers; each worker runs its own workspace (scoped threads, the C
    /// pthread-worker pattern). Unlike the C, partial per-text results are
    /// discarded on error — the `Err` return carries the first worker's
    /// status in worker order.
    pub fn encode_batch_with_workers(
        &self,
        texts: &[&[u8]],
        encode_flags: u32,
        token_stride: usize,
        worker_count: usize,
    ) -> Result<BatchEncoding, TokenizerError> {
        if !texts.is_empty() && token_stride == 0 {
            return Err(TokenizerError::InvalidArgument("token stride is zero"));
        }
        if texts.is_empty() {
            return Ok(BatchEncoding::default());
        }
        let mut worker_count =
            if worker_count == 0 { MAX_PARALLEL_WORKER_COUNT } else { worker_count };
        if worker_count > texts.len() {
            worker_count = texts.len();
        }

        let base_count = texts.len() / worker_count;
        let remainder_count = texts.len() % worker_count;

        // Run the workers (in-thread when there is only one, like the C).
        let mut worker_results: Vec<(usize, Vec<Encoding>, Result<(), TokenizerError>)> =
            Vec::with_capacity(worker_count);
        if worker_count <= 1 {
            let (encodings, status) =
                self.batch_worker(texts, 0, texts.len(), encode_flags, token_stride);
            worker_results.push((0, encodings, status));
        } else {
            let mut assignments = Vec::with_capacity(worker_count);
            let mut next_text_index = 0usize;
            for worker_index in 0..worker_count {
                let assigned_count = base_count + usize::from(worker_index < remainder_count);
                assignments.push((next_text_index, assigned_count));
                next_text_index += assigned_count;
            }
            std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(worker_count);
                for (first_text_index, assigned_count) in assignments {
                    handles.push(scope.spawn(move || {
                        let (encodings, status) = self.batch_worker(
                            texts,
                            first_text_index,
                            assigned_count,
                            encode_flags,
                            token_stride,
                        );
                        (first_text_index, encodings, status)
                    }));
                }
                for handle in handles {
                    worker_results.push(handle.join().expect("batch worker panicked"));
                }
            });
            worker_results.sort_by_key(|(first_text_index, _, _)| *first_text_index);
        }

        // First failing worker in worker order decides the status, as in the
        // C join loop.
        let mut final_status = Ok(());
        for (_, _, status) in &mut worker_results {
            if status.is_err() && final_status.is_ok() {
                final_status = std::mem::replace(status, Ok(()));
            }
        }

        let mut batch = BatchEncoding {
            token_stride,
            token_ids: vec![0u32; texts.len() * token_stride],
            token_counts: vec![0; texts.len()],
            overflow_token_counts: vec![0; texts.len()],
            invalid_segment_counts: vec![0; texts.len()],
        };
        for (first_text_index, encodings, _) in worker_results {
            for (local_index, encoding) in encodings.into_iter().enumerate() {
                let text_index = first_text_index + local_index;
                let start = text_index * token_stride;
                batch.token_ids[start..start + encoding.token_ids.len()]
                    .copy_from_slice(&encoding.token_ids);
                batch.token_counts[text_index] = encoding.token_ids.len();
                batch.overflow_token_counts[text_index] = encoding.overflow_token_count;
                batch.invalid_segment_counts[text_index] = encoding.invalid_segment_count;
            }
        }
        final_status?;
        Ok(batch)
    }

    /// `SparkTokenizerEncodeBatchUtf8`: batch encode with the default worker
    /// count ([`MAX_PARALLEL_WORKER_COUNT`] for two or more texts, one
    /// otherwise).
    pub fn encode_batch(
        &self,
        texts: &[&[u8]],
        encode_flags: u32,
        token_stride: usize,
    ) -> Result<BatchEncoding, TokenizerError> {
        self.encode_batch_with_workers(
            texts,
            encode_flags,
            token_stride,
            if texts.len() >= 2 { MAX_PARALLEL_WORKER_COUNT } else { 1 },
        )
    }
}
