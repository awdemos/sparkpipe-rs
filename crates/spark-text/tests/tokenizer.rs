//! Port of the C tree's `tests/test_tokenizer.c`, plus edge-case and
//! differential round-trip fuzz tests.

use spark_text::tokenizer::{
    Encoding, Tokenizer, TokenizerError, Workspace, DECODE_FLAG_SKIP_SPECIAL_TOKENS,
    ENCODE_FLAG_ADD_PREFIX_SPACE, ENCODE_FLAG_DISABLE_PIECE_CACHE,
    ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH,
};

const TOKEN_A: u32 = 1;
const TOKEN_B: u32 = 2;
const TOKEN_C: u32 = 3;
const TOKEN_AB: u32 = 4;
const TOKEN_ABC: u32 = 5;
const TOKEN_UNKNOWN: u32 = 6;
const TOKEN_STOP: u32 = 7;
const TOKEN_SPACE: u32 = 8;
const TOKEN_X: u32 = 9;
const TOKEN_Y: u32 = 10;
const TOKEN_Z: u32 = 11;
const TOKEN_XY: u32 = 12;
const TOKEN_XYZ: u32 = 13;

fn fixture_path(name: &str) -> std::path::PathBuf {
    // Tests run on threads within one process; an atomic counter keeps each
    // fixture file unique so concurrent tests never race on the same path.
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "spark_text_tokenizer_{}_{}_{}.json",
        std::process::id(),
        unique,
        name
    ))
}

/// The C `SparkTestTokenizerWriteFixtureJson` fixture. The `Ġ` (U+0120)
/// token is the byte-to-unicode form of byte 0x20 (space), matching the C
/// fixture's `"\304\240"` entry.
fn write_fixture_json() -> std::path::PathBuf {
    let path = fixture_path("byte_bpe");
    let json = format!(
        "{{\n\
        \x20 \"model\": {{\n\
        \x20   \"type\": \"BPE\",\n\
        \x20   \"unk_token\": \"<unk>\",\n\
        \x20   \"byte_fallback\": false,\n\
        \x20   \"vocab\": {{\n\
        \x20     \"a\": {TOKEN_A},\n\
        \x20     \"b\": {TOKEN_B},\n\
        \x20     \"c\": {TOKEN_C},\n\
        \x20     \"ab\": {TOKEN_AB},\n\
        \x20     \"abc\": {TOKEN_ABC},\n\
        \x20     \"<unk>\": {TOKEN_UNKNOWN},\n\
        \x20     \"<|stop|>\": {TOKEN_STOP},\n\
        \x20     \"\u{0120}\": {TOKEN_SPACE},\n\
        \x20     \"x\": {TOKEN_X},\n\
        \x20     \"y\": {TOKEN_Y},\n\
        \x20     \"z\": {TOKEN_Z},\n\
        \x20     \"xy\": {TOKEN_XY},\n\
        \x20     \"xyz\": {TOKEN_XYZ}\n\
        \x20   }},\n\
        \x20   \"merges\": [\n\
        \x20     \"a b\",\n\
        \x20     \"ab c\",\n\
        \x20     \"x y\",\n\
        \x20     \"xy z\"\n\
        \x20   ]\n\
        \x20 }},\n\
        \x20 \"pre_tokenizer\": {{\n\
        \x20   \"type\": \"ByteLevel\",\n\
        \x20   \"add_prefix_space\": false\n\
        \x20 }},\n\
        \x20 \"added_tokens\": [\n\
        \x20   {{\"id\": {TOKEN_STOP}, \"content\": \"<|stop|>\", \"special\": true}}\n\
        \x20 ]\n\
        }}\n"
    );
    std::fs::write(&path, json).unwrap();
    path
}

/// The C `SparkTestTokenizerWriteLargeMergeFixtureJson` fixture.
fn write_large_merge_fixture_json(merge_count: usize) -> std::path::PathBuf {
    let path = fixture_path("large_merge");
    let mut json = format!(
        "{{\n\
        \x20 \"model\": {{\n\
        \x20   \"type\": \"BPE\",\n\
        \x20   \"unk_token\": \"<unk>\",\n\
        \x20   \"byte_fallback\": false,\n\
        \x20   \"vocab\": {{\n\
        \x20     \"a\": {TOKEN_A},\n\
        \x20     \"b\": {TOKEN_B},\n\
        \x20     \"ab\": {TOKEN_AB},\n\
        \x20     \"<unk>\": {TOKEN_UNKNOWN}\n\
        \x20   }},\n\
        \x20   \"merges\": [\n\
        \x20     \"a b\""
    );
    for merge_index in 1..merge_count {
        json.push_str(&format!(
            ",\n      \"unused_left_{merge_index:06} unused_right_{merge_index:06}\""
        ));
    }
    json.push_str(
        "\n    ]\n  },\n  \"pre_tokenizer\": {\n    \"type\": \"ByteLevel\",\n    \
         \"add_prefix_space\": false\n  }\n}\n",
    );
    std::fs::write(&path, json).unwrap();
    path
}

fn load_fixture() -> Tokenizer {
    Tokenizer::load_huggingface_json(write_fixture_json()).unwrap()
}

/// Port of `SparkTestTokenizerEncodesByteBpeAndSpecialTokens`.
#[test]
fn encodes_byte_bpe_and_special_tokens() {
    let tokenizer = load_fixture();

    let encoding = tokenizer.encode(b"abc", 0).unwrap();
    assert_eq!(encoding.token_count(), 1);
    assert_eq!(encoding.overflow_token_count(), 0);
    assert_eq!(encoding.invalid_segment_count(), 0);
    assert_eq!(encoding.token_ids(), &[TOKEN_ABC]);

    let encoding = tokenizer.encode(b"a<|stop|>bc", 0).unwrap();
    assert_eq!(encoding.token_count(), 4);
    assert_eq!(encoding.token_ids(), &[TOKEN_A, TOKEN_STOP, TOKEN_B, TOKEN_C]);

    let encoding = tokenizer.encode(b"abc", ENCODE_FLAG_ADD_PREFIX_SPACE).unwrap();
    assert_eq!(encoding.token_count(), 2);
    assert_eq!(encoding.token_ids(), &[TOKEN_SPACE, TOKEN_ABC]);

    // Caller-capped encoding: overflow mirrors the C capacity contract.
    let mut workspace = Workspace::new(16).unwrap();
    let mut encoding = Encoding::with_capacity(2);
    let result = tokenizer.encode_with_workspace(b"a<|stop|>bc", 0, &mut workspace, &mut encoding);
    assert!(matches!(result, Err(TokenizerError::CapacityExceeded)));
    assert_eq!(encoding.token_count(), 2);
    assert_eq!(encoding.overflow_token_count(), 2);
    assert_eq!(encoding.token_ids(), &[TOKEN_A, TOKEN_STOP]);
}

/// Port of `SparkTestTokenizerDecodesByteLevelTokens`.
#[test]
fn decodes_byte_level_tokens() {
    let tokenizer = load_fixture();

    let token_ids = [TOKEN_SPACE, TOKEN_ABC, TOKEN_STOP];
    let text = tokenizer.decode(&token_ids, DECODE_FLAG_SKIP_SPECIAL_TOKENS).unwrap();
    assert_eq!(text, b" abc");

    let result = tokenizer.decode_with_capacity(&token_ids[..2], 0, 3);
    assert!(matches!(result, Err(TokenizerError::CapacityExceeded)));
}

/// Port of `SparkTestTokenizerEncodesBatch`.
#[test]
fn encodes_batch() {
    let tokenizer = load_fixture();

    let batch = tokenizer.encode_batch(&[b"abc", b"xyz"], 0, 4).unwrap();
    assert_eq!(batch.token_counts(), &[1, 1]);
    assert_eq!(batch.overflow_token_counts(), &[0, 0]);
    assert_eq!(batch.tokens_of(0), &[TOKEN_ABC]);
    assert_eq!(batch.tokens_of(1), &[TOKEN_XYZ]);
}

/// Port of `SparkTestTokenizerCompiledFileAndConfiguredBatch`.
#[test]
fn compiled_file_and_configured_batch() {
    let tokenizer = load_fixture();
    let compiled_path = fixture_path("compiled");
    tokenizer.save_compiled_file(&compiled_path).unwrap();
    let loaded_tokenizer = Tokenizer::load_compiled_file(&compiled_path).unwrap();

    let texts: &[&[u8]] = &[b"abc", b"xyz", b"a<|stop|>bc"];
    let batch = loaded_tokenizer.encode_batch_with_workers(texts, 0, 4, 2).unwrap();
    assert_eq!(batch.token_counts(), &[1, 1, 4]);
    assert_eq!(batch.tokens_of(0), &[TOKEN_ABC]);
    assert_eq!(batch.tokens_of(1), &[TOKEN_XYZ]);
    assert_eq!(batch.tokens_of(2), &[TOKEN_A, TOKEN_STOP, TOKEN_B, TOKEN_C]);
    assert_eq!(batch.invalid_segment_counts(), &[0, 0, 0]);

    // The compiled round trip must also encode identically one at a time.
    let encoding = loaded_tokenizer.encode(b"a<|stop|>bc", 0).unwrap();
    assert_eq!(encoding.token_ids(), &[TOKEN_A, TOKEN_STOP, TOKEN_B, TOKEN_C]);

    let _ = std::fs::remove_file(&compiled_path);
}

/// Port of `SparkTestTokenizerLoadsLargeMergeArrayWithoutIndexedArrayWalk`.
#[test]
fn loads_large_merge_array_without_indexed_array_walk() {
    let path = write_large_merge_fixture_json(8192);
    let tokenizer = Tokenizer::load_huggingface_json(&path).unwrap();
    assert_eq!(tokenizer.merge_count(), 8192);
    assert_eq!(tokenizer.find_token_id(b"ab").unwrap(), TOKEN_AB);

    let encoding = tokenizer.encode(b"ab", 0).unwrap();
    assert_eq!(encoding.token_count(), 1);
    assert_eq!(encoding.token_ids(), &[TOKEN_AB]);

    let _ = std::fs::remove_file(&path);
}

/// Port of `SparkTestTokenizerWarmCacheSurvivesGrowthAndMatches`: a persistent
/// workspace keeps its piece cache warm across encode calls and across
/// symbol-buffer growth, and warm-cache output stays identical to a fresh
/// cold encode.
#[test]
fn warm_cache_survives_growth_and_matches() {
    const BASE: &[u8] = b"the quick brown fox jumps over the lazy dog and the cat sat on the mat \
        she said don't and he said can't while they walked to the market today \
        numbers 123 456 and symbols !!! ... mixed with words words words again";
    let path = write_large_merge_fixture_json(8192);
    let tokenizer = Tokenizer::load_huggingface_json(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    let mut workspace = Workspace::new(16).unwrap();
    let sizes = [20usize, 20, BASE.len(), 20, BASE.len(), 30];
    for request_bytes in sizes {
        let request_bytes = request_bytes.min(BASE.len());
        let text = &BASE[..request_bytes];
        let mut warm_encoding = Encoding::with_capacity(512);
        tokenizer.encode_with_workspace(text, 0, &mut workspace, &mut warm_encoding).unwrap();
        let cold_encoding = tokenizer.encode(text, 0).unwrap();
        assert_eq!(warm_encoding.token_count(), cold_encoding.token_count());
        assert_eq!(warm_encoding.token_ids(), cold_encoding.token_ids());
    }
}

/// Port of `SparkTestTokenizerPieceCacheMatchesUncached`: the piece cache
/// must produce byte-identical output to the uncached merge path.
#[test]
fn piece_cache_matches_uncached() {
    const CORPUS: &[u8] = b"the quick brown fox the quick brown fox jumps over the lazy dog \
        the the the quick quick brown fox jumps jumps over over the lazy lazy \
        hello world hello world foo bar foo bar baz qux the quick brown fox";
    let path = write_large_merge_fixture_json(8192);
    let tokenizer = Tokenizer::load_huggingface_json(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    let cached_encoding = tokenizer.encode(CORPUS, 0).unwrap();
    let uncached_encoding = tokenizer.encode(CORPUS, ENCODE_FLAG_DISABLE_PIECE_CACHE).unwrap();
    assert_eq!(cached_encoding.token_count(), uncached_encoding.token_count());
    assert_eq!(cached_encoding.token_ids(), uncached_encoding.token_ids());
}

// ---------------------------------------------------------------------------
// Edge cases and additional parity tests beyond the C test file.
// ---------------------------------------------------------------------------

/// Merge rank ordering must decide the outcome, not just the final vocabulary
/// set: with merges `a b` (rank 0), `b c` (rank 1), `ab c` (rank 2) and no
/// `a bc` merge, the rank-0 `a b` merge must win over the adjacent rank-1
/// `b c` merge, so "abc" becomes a single `abc` token. If `b c` merged first
/// the result would be `[a, bc]`.
#[test]
fn merge_rank_ordering_beats_adjacent_higher_rank() {
    let path = fixture_path("rank_order");
    let json = "{\n  \"model\": {\n    \"type\": \"BPE\",\n    \"vocab\": {\n      \
        \"a\": 1,\n      \"b\": 2,\n      \"c\": 3,\n      \"ab\": 4,\n      \
        \"bc\": 5,\n      \"abc\": 6\n    },\n    \"merges\": [\n      \"a b\",\n      \
        \"b c\",\n      \"ab c\"\n    ]\n  },\n  \"pre_tokenizer\": {\n    \
        \"type\": \"ByteLevel\"\n  }\n}\n";
    std::fs::write(&path, json).unwrap();
    let tokenizer = Tokenizer::load_huggingface_json(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    let encoding = tokenizer.encode(b"abc", 0).unwrap();
    assert_eq!(encoding.token_ids(), &[6]);

    // "aabb" has a single applicable merge (`a b`) at one position only;
    // applying it yields [a, ab, b] (the C produces the same).
    let encoding = tokenizer.encode(b"aabb", 0).unwrap();
    assert_eq!(encoding.token_ids(), &[TOKEN_A, TOKEN_AB, TOKEN_B]);
}

/// Special tokens are matched longest-first after the C length-descending
/// sort, and `ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH` turns matching off.
#[test]
fn special_tokens_longest_first_and_disable_flag() {
    let path = fixture_path("specials");
    let json = "{\n  \"model\": {\n    \"type\": \"BPE\",\n    \"unk_token\": \"<unk>\",\n    \
        \"vocab\": {\n      \"a\": 1,\n      \"b\": 2,\n      \"c\": 3,\n      \
        \"<unk>\": 4,\n      \"ab\": 5,\n      \"abc\": 6\n    },\n    \"merges\": []\n  },\n  \
        \"added_tokens\": [\n    {\"id\": 10, \"content\": \"ab\", \"special\": true},\n    \
        {\"id\": 11, \"content\": \"abc\", \"special\": true}\n  ]\n}\n";
    std::fs::write(&path, json).unwrap();
    let tokenizer = Tokenizer::load_huggingface_json(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    // "abc" must match the longer special token, not "ab" + c.
    let encoding = tokenizer.encode(b"abc", 0).unwrap();
    assert_eq!(encoding.token_ids(), &[11]);

    // With matching disabled the bytes are encoded normally; merges is empty
    // so each byte stands alone: a, b, c.
    let encoding = tokenizer.encode(b"abc", ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH).unwrap();
    assert_eq!(encoding.token_ids(), &[1, 2, 3]);
}

/// Empty input, prefix-space edge cases.
#[test]
fn empty_input_and_prefix_space_edges() {
    let tokenizer = load_fixture();

    let encoding = tokenizer.encode(b"", 0).unwrap();
    assert_eq!(encoding.token_count(), 0);
    assert_eq!(tokenizer.decode(&[], 0).unwrap(), b"");

    // The C adds the prefix space even for empty input when the flag is set
    // (the condition includes `text_bytes == 0`).
    let encoding = tokenizer.encode(b"", ENCODE_FLAG_ADD_PREFIX_SPACE).unwrap();
    assert_eq!(encoding.token_ids(), &[TOKEN_SPACE]);

    // Input already starting with a space does not get a second one.
    let encoding = tokenizer.encode(b" abc", ENCODE_FLAG_ADD_PREFIX_SPACE).unwrap();
    assert_eq!(encoding.token_ids(), &[TOKEN_SPACE, TOKEN_ABC]);
}

/// Unknown bytes fall back to the unk token when the model declares one, and
/// fail with `NotFound` (counting invalid segments like the C) otherwise.
#[test]
fn unknown_bytes_unk_fallback_and_not_found() {
    let tokenizer = load_fixture();

    // 'q' has no vocab entry; the fixture declares "<unk>": byte fallback to
    // the unk id.
    let encoding = tokenizer.encode(b"q", 0).unwrap();
    assert_eq!(encoding.token_ids(), &[TOKEN_UNKNOWN]);
    assert_eq!(tokenizer.decode(&[TOKEN_UNKNOWN], 0).unwrap(), b"<unk>");

    // A model without unk_token cannot map 'q' at all.
    let path = fixture_path("no_unk");
    let json = "{\n  \"model\": {\n    \"type\": \"BPE\",\n    \"vocab\": {\n      \
        \"a\": 1,\n      \"b\": 2,\n      \"ab\": 3\n    },\n    \"merges\": [\"a b\"]\n  }\n}\n";
    std::fs::write(&path, json).unwrap();
    let no_unk = Tokenizer::load_huggingface_json(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    let mut workspace = Workspace::new(16).unwrap();
    let mut encoding = Encoding::unbounded();
    let result = no_unk.encode_with_workspace(b"q", 0, &mut workspace, &mut encoding);
    assert!(matches!(result, Err(TokenizerError::NotFound(_))));
    // The C bumps the counter once in the piece encode and once in the
    // segment encode on the error path.
    assert_eq!(encoding.invalid_segment_count(), 2);
    assert!(no_unk.encode(b"q", 0).is_err());
    assert!(no_unk.encode(b"ab", 0).is_ok());
}

/// Long words bypass the inline piece cache (> 32 bytes) and must grow the
/// workspace symbol buffers; output must match a fresh cold encode and the
/// cache-disabled path.
#[test]
fn long_words_grow_buffers_and_match() {
    let tokenizer = load_fixture();

    let mut long = Vec::new();
    long.extend(std::iter::repeat_n(b'a', 1000));
    long.extend_from_slice(b"abc abc abc");
    long.extend(std::iter::repeat_n(b'x', 700));

    let expected = tokenizer.encode(&long, 0).unwrap();
    // 1000 x 'a' (no `a a` merge), "abc" -> 1, " abc" -> 2 each, 700 x 'x'.
    assert_eq!(expected.token_count(), 1000 + 1 + 2 + 2 + 700);

    let mut workspace = Workspace::new(16).unwrap();
    let mut warm = Encoding::unbounded();
    tokenizer.encode_with_workspace(&long, 0, &mut workspace, &mut warm).unwrap();
    assert_eq!(warm.token_ids(), expected.token_ids());

    let uncached = tokenizer.encode(&long, ENCODE_FLAG_DISABLE_PIECE_CACHE).unwrap();
    assert_eq!(uncached.token_ids(), expected.token_ids());
}

/// Vocabulary / merge inspection helpers.
#[test]
fn inspection_helpers() {
    let tokenizer = load_fixture();

    assert_eq!(tokenizer.find_token_id(b"ab").unwrap(), TOKEN_AB);
    assert!(matches!(tokenizer.find_token_id(b"nope"), Err(TokenizerError::NotFound(_))));
    assert_eq!(tokenizer.merge_count(), 4);
    assert_eq!(tokenizer.vocabulary_count(), 13);
    assert_eq!(tokenizer.maximum_token_id(), TOKEN_XYZ);
    assert!(!tokenizer.byte_fallback());
    assert!(!tokenizer.add_prefix_space());
    assert_eq!(tokenizer.token_text(TOKEN_ABC), Some(b"abc".as_slice()));
    assert_eq!(tokenizer.token_text(TOKEN_XYZ + 100), None);
    assert_eq!(tokenizer.special_tokens().len(), 1);
    assert_eq!(tokenizer.special_tokens()[0].text(), b"<|stop|>");
    assert_eq!(tokenizer.special_tokens()[0].token_id(), TOKEN_STOP);
}

/// Invalid flags and ids are rejected like the C argument validation.
#[test]
fn argument_validation() {
    let tokenizer = load_fixture();

    assert!(matches!(
        tokenizer.encode(b"abc", 0x8000_0000),
        Err(TokenizerError::InvalidArgument(_))
    ));
    assert!(matches!(
        tokenizer.decode(&[TOKEN_A], 0x8000_0000),
        Err(TokenizerError::InvalidArgument(_))
    ));
    // Id above maximum_token_id has no reverse-vocab entry.
    assert!(matches!(
        tokenizer.decode(&[TOKEN_XYZ + 1], 0),
        Err(TokenizerError::InvalidArgument(_))
    ));
    assert!(matches!(
        tokenizer.decode_with_capacity(&[TOKEN_A], 0, 0),
        Err(TokenizerError::InvalidArgument(_))
    ));
    // A stride of zero with non-empty input is invalid, as in the C.
    assert!(matches!(
        tokenizer.encode_batch(&[b"abc"], 0, 0),
        Err(TokenizerError::InvalidArgument(_))
    ));
    // Empty batch is a no-op.
    let batch = tokenizer.encode_batch(&[], 0, 4).unwrap();
    assert_eq!(batch.token_counts(), &[] as &[usize]);
}

/// Loading failures: missing file, malformed JSON, missing members.
#[test]
fn load_errors() {
    assert!(matches!(
        Tokenizer::load_huggingface_json(fixture_path("does_not_exist")),
        Err(TokenizerError::Io(_))
    ));

    let path = fixture_path("malformed");
    std::fs::write(&path, "{ not json").unwrap();
    assert!(matches!(Tokenizer::load_huggingface_json(&path), Err(TokenizerError::Parse(_))));

    std::fs::write(&path, "{\"model\": {\"vocab\": {}}}").unwrap();
    assert!(matches!(Tokenizer::load_huggingface_json(&path), Err(TokenizerError::Parse(_))));
    let _ = std::fs::remove_file(&path);
}

/// Array-form merges (`["l", "r"]` instead of `"l r"`) parse identically.
#[test]
fn array_form_merges() {
    let path = fixture_path("array_merges");
    let json = "{\n  \"model\": {\n    \"type\": \"BPE\",\n    \"vocab\": {\n      \
        \"a\": 1,\n      \"b\": 2,\n      \"ab\": 3\n    },\n    \"merges\": [[\"a\", \"b\"]]\n  }\n}\n";
    std::fs::write(&path, json).unwrap();
    let tokenizer = Tokenizer::load_huggingface_json(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(tokenizer.merge_count(), 1);
    assert_eq!(tokenizer.encode(b"ab", 0).unwrap().token_ids(), &[3]);
}

/// Differential round-trip fuzz: for random byte strings over the fixture's
/// fully-decodable alphabet (`a b c x y z` and space, plus the `<|stop|>`
/// special token), `decode(encode(x)) == x` must hold, and a warm persistent
/// workspace must match a fresh cold encode. Deterministic xorshift PRNG; no
/// external crates.
#[test]
fn fuzz_round_trip_decode_of_encode() {
    let tokenizer = load_fixture();
    const ALPHABET: &[u8] = b"abcxyz ";

    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next_random = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let mut workspace = Workspace::new(16).unwrap();
    for iteration in 0..500 {
        let length = (next_random() % 129) as usize;
        let mut text = Vec::with_capacity(length + 9);
        for _ in 0..length {
            text.push(ALPHABET[(next_random() % ALPHABET.len() as u64) as usize]);
            if next_random() % 10 == 0 {
                text.extend_from_slice(b"<|stop|>");
            }
        }

        let encoding = tokenizer.encode(&text, 0).unwrap();
        let decoded = tokenizer.decode(encoding.token_ids(), 0).unwrap();
        assert_eq!(decoded, text, "round trip mismatch at iteration {iteration}");

        // Warm workspace (piece cache populated by all prior iterations) must
        // agree with the fresh cold encode above.
        let mut warm = Encoding::unbounded();
        tokenizer.encode_with_workspace(&text, 0, &mut workspace, &mut warm).unwrap();
        assert_eq!(warm.token_ids(), encoding.token_ids());

        // So must the cache-disabled path.
        let uncached = tokenizer.encode(&text, ENCODE_FLAG_DISABLE_PIECE_CACHE).unwrap();
        assert_eq!(uncached.token_ids(), encoding.token_ids());
    }
}
