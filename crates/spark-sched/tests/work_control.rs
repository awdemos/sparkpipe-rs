//! Port of `tests/test_glm52_ring_work_control.c` (GPU-free scenarios).
//!
//! The C test wraps the public entry points to re-finalize each packet's
//! transaction identity after direct field mutation
//! (`SparkTestRefreshWorkTransaction`); the helpers here mirror those
//! wrappers exactly. GLM52 capacities arrive through [`WorkControlConfig`]
//! (the deliberate deviation — nothing is baked into the module).

use std::cell::RefCell;
use std::rc::Rc;

use spark_sched::work_control::mtp_tree;
use spark_sched::work_control::*;

/// GLM52-equivalent capacities, supplied via config instead of being baked
/// in (C: `SPARK_GLM52_*` / `SPARK_RING_WORK_CONTROL_MAX_*`).
fn test_config() -> WorkControlConfig {
    let config = WorkControlConfig {
        mtp_draft_token_count: 6, // SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT
        dspark_max_speculative_token_count: 7, // SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT
        maximum_context_tokens: 1_048_576, // SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS
        kv_block_tokens: 64,      // SPARK_GLM52_KV_BLOCK_TOKENS
        max_prefill_tokens_per_packet: 256, // SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH
        output_vocab_count: 154_880, // SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT
        max_lane_count: 1024,     // SPARK_RING_WORK_CONTROL_MAX_LANE_COUNT
        max_active_sequence_count: 1024, // SPARK_STAGE_PLAN_MAX_BATCH_BUCKET
        cohort_capacity: 1023,
    };
    config.validate().unwrap();
    config
}

// C: SPARK_TEST_KV_LANE_STRIDE / SPARK_TEST_KV_PHYSICAL_BLOCK_CAPACITY /
// SPARK_TEST_KV_DIRECTORY_CAPACITY.
const TEST_KV_LANE_STRIDE: u32 = 4096;
const TEST_KV_PHYSICAL_BLOCK_CAPACITY: u32 = 1024;
const TEST_KV_DIRECTORY_CAPACITY: u32 = 2048;

/// C: `SparkTestRefreshWorkTransaction`.
fn refresh_work_transaction(packet: &mut WorkControlPacket, config: &WorkControlConfig) {
    if packet.lane_count == 0 || packet.lane_count > config.max_lane_count {
        return;
    }
    for lane in packet.lanes[..packet.lane_count as usize].iter_mut() {
        if lane.request_generation == 0 && lane.request_id != 0 {
            lane.request_generation = lane.request_id + 1_000_000;
        }
    }
    let control_generation = if packet.control_generation != 0 {
        packet.control_generation
    } else {
        STANDALONE_GENERATION
    };
    let chunk_count = if packet.step_chunk_count != 0 { packet.step_chunk_count } else { 1 };
    let chunk_index =
        if packet.step_chunk_index < chunk_count { packet.step_chunk_index } else { 0 };
    let _ = finalize_transaction(packet, config, control_generation, chunk_index, chunk_count);
}

/// C: `SparkTestValidatePacket` (validates a refreshed copy).
fn validate(
    packet: &WorkControlPacket,
    config: &WorkControlConfig,
    max_active_sequence_count: u32,
    max_pipeline_slot_count: u32,
) -> Result<()> {
    let mut copy = packet.clone();
    refresh_work_transaction(&mut copy, config);
    validate_packet(&copy, config, max_active_sequence_count, max_pipeline_slot_count)
}

/// C: `SparkTestBuildHostKvBlockTable`.
fn build<'a>(
    state: &'a mut KvState,
    packet: &mut WorkControlPacket,
    config: &WorkControlConfig,
) -> Result<KvBlockTableView<'a>> {
    refresh_work_transaction(packet, config);
    state.build_host_kv_block_table(packet, config)
}

/// C: `SparkTestCommitHostKvBlockTable`.
fn commit(
    state: &mut KvState,
    packet: &mut WorkControlPacket,
    config: &WorkControlConfig,
) -> Result<()> {
    refresh_work_transaction(packet, config);
    state.commit_host_kv_block_table(packet, config)
}

/// C: `SparkTestReleasePacketSequences`.
fn release_packets(
    state: &mut KvState,
    packet: &mut WorkControlPacket,
    config: &WorkControlConfig,
) -> Result<()> {
    refresh_work_transaction(packet, config);
    state.release_packet_sequences(packet, config)
}

/// C: `SparkTestInitializeKvState` (lane stride 4096, block tokens 256,
/// physical 1024, directory 2048/2048, pins configured).
fn init_kv_state(lane_capacity: u32) -> KvState {
    let mut state = KvState::new(
        lane_capacity,
        TEST_KV_LANE_STRIDE,
        256,
        TEST_KV_PHYSICAL_BLOCK_CAPACITY,
        TEST_KV_DIRECTORY_CAPACITY,
        TEST_KV_DIRECTORY_CAPACITY,
    )
    .unwrap();
    state.configure_kv_pins().unwrap();
    state
}

/// C: `SparkTestInitializeWorkPacket`.
fn init_work_packet(config: &WorkControlConfig) -> WorkControlPacket {
    let mut packet = WorkControlPacket::zeroed(config);
    packet.set_lane_count(config, 4);
    packet.magic = PACKET_MAGIC;
    packet.abi_version = ABI_VERSION;
    packet.control_generation = STANDALONE_GENERATION;
    packet.active_sequence_count = 4;
    packet.descriptor_bytes = calculate_packet_bytes(config, packet.active_sequence_count);
    packet.request_id = 7;
    packet.sequence_id = 11;
    packet.sequence_position = 128;
    packet.new_token_count = 1;
    packet.pipeline_slot = 0;
    packet.block_token_count = 256;
    packet.kv_block_table_token_count = 32769;
    packet.max_blocks_per_sequence = 4096;
    packet.lane_count = packet.active_sequence_count;
    packet.rows_per_lane = 1;
    packet.execution_row_count = packet.lane_count;
    packet.execution_batch_bucket = STAGE_PLAN_BUCKET_B16;
    let request_id = packet.request_id;
    let sequence_id = packet.sequence_id;
    let sequence_position = packet.sequence_position;
    let kv_block_table_token_count = packet.kv_block_table_token_count;
    let input_token_id = packet.input_token_id;
    for lane_index in 0..packet.lane_count as usize {
        let lane = &mut packet.lanes[lane_index];
        lane.request_id = request_id + lane_index as u64;
        lane.request_generation = lane.request_id + 1_000_000;
        lane.sequence_id = sequence_id + lane_index as u64;
        lane.sequence_position = sequence_position + lane_index as u64;
        lane.request_slot_index = lane_index as u32;
        lane.context_token_count = kv_block_table_token_count - lane_index as u32;
        lane.input_token_id = input_token_id;
    }
    refresh_work_transaction(&mut packet, config);
    packet
}

/// C: `SparkTestPrefillPacketLanes` (builds + commits a 1-lane prefill
/// packet per source lane, making every block resident).
fn prefill_packet_lanes(
    source_packet: &WorkControlPacket,
    state: &mut KvState,
    config: &WorkControlConfig,
) {
    for lane_index in 0..source_packet.lane_count as usize {
        let mut packet = source_packet.clone();
        packet.flags = FLAG_PREFILL;
        packet.active_sequence_count = 1;
        packet.lane_count = 1;
        packet.rows_per_lane = 1;
        packet.execution_row_count = 1;
        packet.descriptor_bytes = calculate_packet_bytes(config, 1);
        packet.lanes[0] = source_packet.lanes[lane_index].clone();
        packet.lanes[0].sequence_position = u64::from(packet.lanes[0].context_token_count) - 1;
        packet.request_id = packet.lanes[0].request_id;
        packet.sequence_id = packet.lanes[0].sequence_id;
        packet.sequence_position = packet.lanes[0].sequence_position;
        packet.input_token_id = packet.lanes[0].input_token_id;
        packet.prefill_token_ids[0] = packet.input_token_id;
        packet.kv_block_table_token_count = packet.lanes[0].context_token_count;
        build(state, &mut packet, config).unwrap();
        commit(state, &mut packet, config).unwrap();
    }
}

/// C: `SparkTestWorkControlSwapStorage` + the store/load callbacks.
#[derive(Default)]
struct SwapStorage {
    physical_values: [u32; 2],
    backing_values: [u32; 8],
    backing_sequence_ids: [u64; 8],
    backing_logical_indices: [u32; 8],
    store_count: u32,
    load_count: u32,
}

struct SwapProbe(Rc<RefCell<SwapStorage>>);

impl KvSwapHooks for SwapProbe {
    fn store(
        &mut self,
        key: KvKey,
        physical_block_index: u32,
        backing_block_index: u32,
    ) -> Result<()> {
        let mut storage = self.0.borrow_mut();
        if physical_block_index >= 2 || backing_block_index >= 8 {
            return Err(WorkControlError::InvalidArgument);
        }
        storage.backing_values[backing_block_index as usize] =
            storage.physical_values[physical_block_index as usize];
        storage.backing_sequence_ids[backing_block_index as usize] = key.low;
        storage.backing_logical_indices[backing_block_index as usize] = key.high as u32;
        storage.store_count += 1;
        Ok(())
    }

    fn load(
        &mut self,
        key: KvKey,
        physical_block_index: u32,
        backing_block_index: u32,
    ) -> Result<()> {
        let mut storage = self.0.borrow_mut();
        if physical_block_index >= 2
            || backing_block_index >= 8
            || storage.backing_sequence_ids[backing_block_index as usize] != key.low
            || storage.backing_logical_indices[backing_block_index as usize] != key.high as u32
        {
            return Err(WorkControlError::ValidationFailed);
        }
        storage.physical_values[physical_block_index as usize] =
            storage.backing_values[backing_block_index as usize];
        storage.load_count += 1;
        Ok(())
    }
}

/// C: `SparkTestGlm52RingWorkControlPacket`.
#[test]
fn packet_validation() {
    let config = test_config();
    let mut packet = init_work_packet(&config);
    assert_eq!(validate(&packet, &config, 1024, 4), Ok(()));
    packet.execution_batch_bucket = 0;
    assert_eq!(validate(&packet, &config, 1024, 4), Err(WorkControlError::InvalidArgument));
    packet.execution_batch_bucket = STAGE_PLAN_BUCKET_B16;
    packet.new_token_count = 9;
    assert_eq!(validate(&packet, &config, 1024, 4), Err(WorkControlError::InvalidArgument));
    let mut packet = init_work_packet(&config);
    packet.flags = FLAG_PREFILL;
    packet.active_sequence_count = 1;
    packet.lane_count = 1;
    packet.execution_row_count = 1;
    packet.descriptor_bytes = calculate_packet_bytes(&config, 1);
    packet.kv_block_table_token_count = packet.sequence_position as u32 + 1;
    packet.lanes[0].context_token_count = packet.kv_block_table_token_count;
    packet.prefill_token_ids[0] = packet.lanes[0].input_token_id;
    assert_eq!(validate(&packet, &config, 1024, 4), Ok(()));
    packet.new_token_count = config.max_prefill_tokens_per_packet + 1;
    packet.rows_per_lane = packet.new_token_count;
    packet.execution_row_count = packet.new_token_count;
    assert_eq!(validate(&packet, &config, 1024, 4), Err(WorkControlError::InvalidArgument));
    packet.new_token_count = config.max_prefill_tokens_per_packet;
    packet.rows_per_lane = packet.new_token_count;
    packet.active_sequence_count = 1025;
    packet.lane_count = 1025;
    packet.execution_row_count = 1025;
    assert_eq!(validate(&packet, &config, 1024, 4), Err(WorkControlError::InvalidArgument));
}

/// C: `SparkTestGlm52RingWorkControlExecutionChunks`.
#[test]
fn execution_chunks() {
    let config = test_config();
    assert_eq!(plan_execution_chunks(&config, 1024, 1, 1024), Ok((1024, 1)));
    assert_eq!(plan_execution_chunks(&config, 1024, 7, 1024), Ok((146, 8)));
    assert_eq!(plan_execution_chunks(&config, 1024, 7, 7168), Ok((146, 8)));
    assert_eq!(
        plan_execution_chunks(
            &config,
            STAGE_PLAN_BUCKET_B1024,
            mtp_tree::VERIFIER_ROW_COUNT,
            STAGE_PLAN_BUCKET_B1024 * mtp_tree::VERIFIER_ROW_COUNT,
        ),
        Ok((170, 7))
    );
    assert_eq!(plan_execution_chunks(&config, 146, 7, 1024), Ok((146, 1)));
    assert_eq!(plan_execution_chunks(&config, 1, 7, 6), Err(WorkControlError::CapacityExceeded));
}

/// C: `SparkTestGlm52RingWorkControlHostBlockTable`.
#[test]
fn host_block_table() {
    let config = test_config();
    let mut packet = init_work_packet(&config);
    let mut state = init_kv_state(4);
    prefill_packet_lanes(&packet, &mut state, &config);
    let indices_ptr = state.physical_block_indices.as_ptr();
    {
        let view = build(&mut state, &mut packet, &config).unwrap();
        assert_eq!(view.abi_version, KV_CACHE_ABI_VERSION);
        assert_eq!(view.block_token_count, packet.block_token_count);
        assert_eq!(view.lane_count, 4);
        assert_eq!(view.lane_stride, 4096);
        assert_eq!(view.lane_capacity, view.lane_stride);
        // C: view pointers alias the state's storage arrays.
        assert!(std::ptr::eq(view.physical_block_indices.as_ptr(), indices_ptr));
        assert!(std::ptr::eq(view.host_physical_block_indices.as_ptr(), indices_ptr));
        assert_eq!(view.host_lane_physical_block_counts[0], 129);
        assert_eq!(view.host_lane_physical_block_counts[3], 128);
        assert_eq!(view.host_physical_block_indices[0], 0);
        assert_eq!(view.host_physical_block_indices[128], 128);
        assert_eq!(view.host_physical_block_indices[4096], 129);
        assert_eq!(view.host_physical_block_indices[(3 * 4096) + 127], 512);
    }
    assert_eq!(state.allocated_physical_block_count, 513);
}

/// C: `SparkTestGlm52RingWorkControlTracksKvReadiness`.
#[test]
fn tracks_kv_readiness() {
    let config = test_config();
    let mut packet = init_work_packet(&config);
    packet.active_sequence_count = 2;
    packet.lane_count = 2;
    packet.execution_row_count = 2;
    packet.descriptor_bytes = calculate_packet_bytes(&config, 2);
    let mut state = init_kv_state(2);
    assert_eq!(build(&mut state, &mut packet, &config).map(|_| ()), Err(WorkControlError::Busy));
    assert_eq!(state.missing_block_count, 1);
    prefill_packet_lanes(&packet, &mut state, &config);
    assert_eq!(state.physical_block_states[0], KV_ENTRY_RESIDENT);
    {
        let view = build(&mut state, &mut packet, &config).unwrap();
        assert_eq!(view.lane_count, 2);
    }
    assert_ne!(state.resident_block_count, 0);
    assert_eq!(state.cancel_host_kv_block_table(&packet, &config), Ok(()));
    assert_eq!(state.physical_block_states[0], KV_ENTRY_RESIDENT);
}

/// C: `SparkTestGlm52RingWorkControlKeepsStableBlocksAcrossLaneReorder`.
#[test]
fn keeps_stable_blocks_across_lane_reorder() {
    let config = test_config();
    let mut packet = init_work_packet(&config);
    packet.active_sequence_count = 2;
    packet.lane_count = 2;
    packet.execution_row_count = 2;
    packet.descriptor_bytes = calculate_packet_bytes(&config, 2);
    let mut state = init_kv_state(2);
    prefill_packet_lanes(&packet, &mut state, &config);
    let (sequence11_block0, sequence12_block0) = {
        let view = build(&mut state, &mut packet, &config).unwrap();
        (
            view.host_physical_block_indices[0],
            view.host_physical_block_indices[TEST_KV_LANE_STRIDE as usize],
        )
    };
    assert_ne!(sequence11_block0, sequence12_block0);

    packet.lanes.swap(0, 1);
    packet.request_id = packet.lanes[0].request_id;
    packet.sequence_id = packet.lanes[0].sequence_id;
    packet.sequence_position = packet.lanes[0].sequence_position;
    packet.input_token_id = packet.lanes[0].input_token_id;
    packet.flags = 0;
    {
        let view = build(&mut state, &mut packet, &config).unwrap();
        assert_eq!(view.host_physical_block_indices[0], sequence12_block0);
        assert_eq!(
            view.host_physical_block_indices[TEST_KV_LANE_STRIDE as usize],
            sequence11_block0
        );
    }
}

/// C: `SparkTestGlm52RingWorkControlDsparkVerify`.
#[test]
fn dspark_verify() {
    let config = test_config();
    let mut packet = init_work_packet(&config);
    packet.active_sequence_count = 1;
    packet.lane_count = 1;
    packet.descriptor_bytes = calculate_packet_bytes(&config, 1);
    packet.flags = FLAG_DSPARK_TAP_CAPTURE | FLAG_DSPARK_SPECULATIVE_VERIFY;
    packet.speculative_token_count = config.dspark_max_speculative_token_count;
    packet.speculative_token_index = 0;
    packet.rows_per_lane = packet.speculative_token_count + 1;
    packet.execution_row_count = packet.rows_per_lane;
    packet.new_token_count = packet.rows_per_lane;
    packet.input_token_id = 101;
    packet.lanes[0].input_token_id = packet.input_token_id;
    for token_index in 0..config.dspark_max_speculative_token_count as usize {
        packet.speculative_draft_token_ids[token_index] = 200 + token_index as u32;
    }
    packet.lanes[0].speculative_token_count = packet.speculative_token_count;
    packet.lanes[0]
        .speculative_draft_token_ids
        .copy_from_slice(&packet.speculative_draft_token_ids);
    assert_eq!(validate(&packet, &config, 1024, 4), Ok(()));
    packet.flags &= !FLAG_DSPARK_TAP_CAPTURE;
    assert_eq!(validate(&packet, &config, 1024, 4), Err(WorkControlError::InvalidArgument));
    packet.flags |= FLAG_DSPARK_TAP_CAPTURE;
    packet.speculative_token_index = 1;
    assert_eq!(validate(&packet, &config, 1024, 4), Err(WorkControlError::InvalidArgument));
}

/// C: `SparkTestGlm52RingWorkControlMtpVerify`.
#[test]
fn mtp_verify() {
    let config = test_config();
    let mut packet = init_work_packet(&config);
    packet.active_sequence_count = 1;
    packet.lane_count = 1;
    packet.descriptor_bytes = calculate_packet_bytes(&config, 1);
    packet.flags = FLAG_MTP_SPECULATIVE_VERIFY;
    packet.speculative_token_count = config.mtp_draft_token_count;
    packet.speculative_token_index = 0;
    packet.rows_per_lane = packet.speculative_token_count + 1;
    packet.execution_row_count = packet.rows_per_lane;
    packet.new_token_count = packet.rows_per_lane;
    packet.input_token_id = 101;
    packet.lanes[0].input_token_id = packet.input_token_id;
    for token_index in 0..config.mtp_draft_token_count as usize {
        packet.speculative_draft_token_ids[token_index] = 300 + token_index as u32;
    }
    packet.lanes[0].speculative_token_count = packet.speculative_token_count;
    packet.lanes[0]
        .speculative_draft_token_ids
        .copy_from_slice(&packet.speculative_draft_token_ids);
    assert_eq!(validate(&packet, &config, 1024, 4), Ok(()));
    packet.flags |= FLAG_DSPARK_TAP_CAPTURE;
    assert_eq!(validate(&packet, &config, 1024, 4), Ok(()));
    packet.flags = FLAG_MTP_SPECULATIVE_VERIFY | FLAG_DSPARK_SPECULATIVE_VERIFY;
    assert_eq!(validate(&packet, &config, 1024, 4), Err(WorkControlError::InvalidArgument));
    packet.flags = FLAG_MTP_SPECULATIVE_VERIFY;
    packet.rows_per_lane = 1;
    packet.execution_row_count = 1;
    packet.new_token_count = 1;
    packet.speculative_token_index = 3;
    assert_eq!(validate(&packet, &config, 1024, 4), Err(WorkControlError::InvalidArgument));
}

/// C: `SparkTestGlm52RingWorkControlB1024MtpBatch`.
#[test]
fn b1024_mtp_batch() {
    let config = test_config();
    let mut packet = init_work_packet(&config);
    packet.active_sequence_count = config.max_lane_count;
    packet.set_lane_count(&config, packet.active_sequence_count);
    packet.execution_batch_bucket = STAGE_PLAN_BUCKET_B1024;
    packet.descriptor_bytes = calculate_packet_bytes(&config, packet.lane_count);
    packet.execution_row_count = packet.lane_count;
    packet.flags = FLAG_MTP_DRAFT;
    packet.mtp_draft_token_count = mtp_tree::CANDIDATE_COUNT;
    packet.new_token_count = packet.mtp_draft_token_count + 1;
    for lane_index in 0..packet.lane_count as usize {
        let lane = &mut packet.lanes[lane_index];
        lane.request_id = 1000 + lane_index as u64;
        lane.sequence_id = 2000 + lane_index as u64;
        lane.sequence_position = 4096;
        lane.request_slot_index = lane_index as u32;
        lane.context_token_count = 4097 + mtp_tree::CANDIDATE_COUNT;
        lane.input_token_id = 100 + (lane_index as u32 % 100);
        lane.mtp_draft_token_count = packet.mtp_draft_token_count;
    }
    packet.request_id = packet.lanes[0].request_id;
    packet.sequence_id = packet.lanes[0].sequence_id;
    packet.sequence_position = packet.lanes[0].sequence_position;
    packet.input_token_id = packet.lanes[0].input_token_id;
    packet.kv_block_table_token_count = 4097 + mtp_tree::CANDIDATE_COUNT;
    assert_eq!(validate(&packet, &config, 1024, 4), Ok(()));
    packet.execution_row_count -= 1;
    assert_eq!(validate(&packet, &config, 1024, 4), Err(WorkControlError::InvalidArgument));
}

/// C: `SparkTestGlm52RingWorkControlB1024LayerMajorMtpVerify`.
#[test]
fn b1024_layer_major_mtp_verify() {
    let config = test_config();
    let mut packet = init_work_packet(&config);
    packet.flags = FLAG_MTP_DRAFT | FLAG_MTP_SPECULATIVE_VERIFY | FLAG_MTP_TREE_VERIFY;
    packet.active_sequence_count = 170;
    packet.set_lane_count(&config, packet.active_sequence_count);
    packet.execution_batch_bucket = STAGE_PLAN_BUCKET_B1024;
    packet.descriptor_bytes = calculate_packet_bytes(&config, packet.lane_count);
    packet.speculative_token_count = mtp_tree::CANDIDATE_COUNT;
    packet.rows_per_lane = mtp_tree::VERIFIER_ROW_COUNT;
    packet.execution_row_count = packet.lane_count * packet.rows_per_lane;
    packet.new_token_count = packet.rows_per_lane;
    packet.kv_block_table_token_count = 4103;
    packet.mtp_draft_token_count = mtp_tree::CANDIDATE_COUNT;
    for token_index in 0..packet.speculative_token_count as usize {
        packet.speculative_draft_token_ids[token_index] = 300 + token_index as u32;
    }
    for lane_index in 0..packet.lane_count as usize {
        let lane = &mut packet.lanes[lane_index];
        lane.request_id = 1000 + lane_index as u64;
        lane.sequence_id = 2000 + lane_index as u64;
        lane.sequence_position = 4096;
        lane.request_slot_index = lane_index as u32;
        lane.context_token_count = 4103;
        lane.input_token_id = 100 + (lane_index as u32 % 100);
        lane.mtp_draft_token_count = packet.mtp_draft_token_count;
        lane.speculative_token_count = packet.speculative_token_count;
        lane.speculative_draft_token_ids.copy_from_slice(&packet.speculative_draft_token_ids);
    }
    packet.request_id = packet.lanes[0].request_id;
    packet.sequence_id = packet.lanes[0].sequence_id;
    packet.sequence_position = packet.lanes[0].sequence_position;
    packet.input_token_id = packet.lanes[0].input_token_id;
    assert_eq!(validate(&packet, &config, 1020, 4), Ok(()));
    assert_eq!(validate(&packet, &config, 1019, 4), Err(WorkControlError::InvalidArgument));
}

/// C: `SparkTestGlm52RingWorkControlB1024LayerMajorDsparkVerify`.
#[test]
fn b1024_layer_major_dspark_verify() {
    let config = test_config();
    let mut packet = init_work_packet(&config);
    packet.flags = FLAG_DSPARK_TAP_CAPTURE | FLAG_DSPARK_SPECULATIVE_VERIFY;
    packet.active_sequence_count = config.max_lane_count;
    packet.set_lane_count(&config, packet.active_sequence_count);
    packet.execution_batch_bucket = STAGE_PLAN_BUCKET_B1024;
    packet.descriptor_bytes = calculate_packet_bytes(&config, packet.lane_count);
    packet.speculative_token_count = config.dspark_max_speculative_token_count;
    packet.rows_per_lane = packet.speculative_token_count + 1;
    packet.execution_row_count = packet.lane_count * packet.rows_per_lane;
    packet.new_token_count = packet.rows_per_lane;
    packet.kv_block_table_token_count = 4104;
    for token_index in 0..packet.speculative_token_count as usize {
        packet.speculative_draft_token_ids[token_index] = 400 + token_index as u32;
    }
    for lane_index in 0..packet.lane_count as usize {
        let lane = &mut packet.lanes[lane_index];
        lane.request_id = 3000 + lane_index as u64;
        lane.sequence_id = 4000 + lane_index as u64;
        lane.sequence_position = 4096;
        lane.request_slot_index = lane_index as u32;
        lane.context_token_count = 4104;
        lane.input_token_id = 200 + (lane_index as u32 % 100);
        lane.speculative_token_count = packet.speculative_token_count;
        lane.speculative_draft_token_ids.copy_from_slice(&packet.speculative_draft_token_ids);
    }
    packet.request_id = packet.lanes[0].request_id;
    packet.sequence_id = packet.lanes[0].sequence_id;
    packet.sequence_position = packet.lanes[0].sequence_position;
    packet.input_token_id = packet.lanes[0].input_token_id;
    assert_eq!(validate(&packet, &config, 8192, 4), Ok(()));
    assert_eq!(validate(&packet, &config, 8191, 4), Err(WorkControlError::InvalidArgument));
}

/// C: `SparkTestGlm52RingWorkControlCommitsTreePositions`.
#[test]
fn commits_tree_positions() {
    let config = test_config();
    let mut state = init_kv_state(16);
    let mut packet = init_work_packet(&config);
    packet.flags = FLAG_MTP_DRAFT | FLAG_MTP_SPECULATIVE_VERIFY | FLAG_MTP_TREE_VERIFY;
    packet.active_sequence_count = 1;
    packet.lane_count = 1;
    packet.descriptor_bytes = calculate_packet_bytes(&config, 1);
    packet.execution_batch_bucket = STAGE_PLAN_BUCKET_B16;
    packet.speculative_token_count = mtp_tree::CANDIDATE_COUNT;
    packet.rows_per_lane = mtp_tree::VERIFIER_ROW_COUNT;
    packet.execution_row_count = packet.rows_per_lane;
    packet.new_token_count = packet.rows_per_lane;
    packet.mtp_draft_token_count = mtp_tree::CANDIDATE_COUNT;
    packet.kv_block_table_token_count = mtp_tree::CONTEXT_EXTENSION + 1;
    packet.sequence_position = 0;
    packet.lanes[0].request_id = packet.request_id;
    packet.lanes[0].sequence_id = packet.sequence_id;
    packet.lanes[0].sequence_position = packet.sequence_position;
    packet.lanes[0].request_slot_index = 0;
    packet.lanes[0].context_token_count = packet.kv_block_table_token_count;
    packet.lanes[0].input_token_id = 101;
    packet.lanes[0].mtp_draft_token_count = packet.mtp_draft_token_count;
    packet.lanes[0].speculative_token_count = packet.speculative_token_count;
    packet.input_token_id = packet.lanes[0].input_token_id;
    for token_index in 0..packet.speculative_token_count as usize {
        packet.speculative_draft_token_ids[token_index] = 300 + token_index as u32;
        packet.lanes[0].speculative_draft_token_ids[token_index] =
            packet.speculative_draft_token_ids[token_index];
    }
    assert_eq!(validate(&packet, &config, STAGE_PLAN_BUCKET_B16, 1), Ok(()));
    let physical_block_index = {
        let view = build(&mut state, &mut packet, &config).unwrap();
        view.host_physical_block_indices[0]
    };
    assert_eq!(commit(&mut state, &mut packet, &config), Ok(()));
    assert_eq!(state.physical_block_states[physical_block_index as usize], KV_ENTRY_RESIDENT);
}

/// C: `SparkTestGlm52RingWorkControlB1024PhysicalDirectory`.
#[test]
fn b1024_physical_directory() {
    let config = test_config();
    // C initializes this state without pins.
    let mut state = KvState::new(1024, 1, 256, 1024, 2048, 2048).unwrap();
    let mut packet = init_work_packet(&config);
    packet.active_sequence_count = 1024;
    packet.set_lane_count(&config, 1024);
    packet.execution_batch_bucket = STAGE_PLAN_BUCKET_B1024;
    packet.execution_row_count = 1024;
    packet.descriptor_bytes = calculate_packet_bytes(&config, packet.lane_count);
    packet.block_token_count = 256;
    packet.kv_block_table_token_count = 1;
    packet.max_blocks_per_sequence = 1;
    for lane_index in 0..packet.lane_count as usize {
        let lane = &mut packet.lanes[lane_index];
        lane.request_id = 10000 + lane_index as u64;
        lane.sequence_id = 20000 + lane_index as u64;
        lane.sequence_position = 0;
        lane.request_slot_index = lane_index as u32;
        lane.context_token_count = 1;
        lane.input_token_id = lane_index as u32;
    }
    packet.request_id = packet.lanes[0].request_id;
    packet.sequence_id = packet.lanes[0].sequence_id;
    packet.sequence_position = packet.lanes[0].sequence_position;
    packet.input_token_id = packet.lanes[0].input_token_id;
    prefill_packet_lanes(&packet, &mut state, &config);
    {
        let view = build(&mut state, &mut packet, &config).unwrap();
        assert_eq!(view.lane_count, 1024);
        for lane_index in 0..packet.lane_count as usize {
            assert_eq!(view.host_lane_physical_block_counts[lane_index], 1);
            assert!(view.host_physical_block_indices[lane_index] < 1024);
        }
    }
    assert_eq!(state.allocated_physical_block_count, 1024);
    assert_eq!(commit(&mut state, &mut packet, &config), Ok(()));
    let first_physical_block = state.physical_block_indices[0];
    let last_physical_block = state.physical_block_indices[1023];
    packet.lanes.swap(0, 1023);
    packet.request_id = packet.lanes[0].request_id;
    packet.sequence_id = packet.lanes[0].sequence_id;
    packet.sequence_position = packet.lanes[0].sequence_position;
    packet.input_token_id = packet.lanes[0].input_token_id;
    packet.flags = 0;
    {
        let view = build(&mut state, &mut packet, &config).unwrap();
        assert_eq!(view.host_physical_block_indices[0], last_physical_block);
        assert_eq!(view.host_physical_block_indices[1023], first_physical_block);
    }
}

/// C: `SparkTestGlm52RingWorkControlNvmeSwapAndRelease`.
#[test]
fn nvme_swap_and_release() {
    let config = test_config();
    let swap_storage = Rc::new(RefCell::new(SwapStorage::default()));
    let mut state = KvState::new(2, 4, 64, 2, 16, 16).unwrap();
    state.configure_kv_swap(8, Box::new(SwapProbe(Rc::clone(&swap_storage)))).unwrap();

    let mut packet = init_work_packet(&config);
    packet.flags = 0;
    packet.active_sequence_count = 2;
    packet.lane_count = 2;
    packet.execution_row_count = 2;
    packet.descriptor_bytes = calculate_packet_bytes(&config, 2);
    packet.block_token_count = 64;
    packet.kv_block_table_token_count = 65;
    packet.max_blocks_per_sequence = 4;
    for sequence_index in 0..2usize {
        packet.lanes[sequence_index].request_id = 50 + sequence_index as u64;
        packet.lanes[sequence_index].sequence_id = 60 + sequence_index as u64;
        packet.lanes[sequence_index].sequence_position = 64;
        packet.lanes[sequence_index].request_slot_index = sequence_index as u32;
        packet.lanes[sequence_index].context_token_count = 65;
    }
    packet.request_id = packet.lanes[0].request_id;
    packet.sequence_id = packet.lanes[0].sequence_id;
    packet.sequence_position = packet.lanes[0].sequence_position;
    assert_eq!(
        build(&mut state, &mut packet, &config).map(|_| ()),
        Err(WorkControlError::CapacityExceeded)
    );
    assert_eq!(state.directory_entry_count, 0);
    assert_eq!(swap_storage.borrow().store_count, 0);
    assert_eq!(swap_storage.borrow().load_count, 0);

    for sequence_index in 0..3u32 {
        let mut packet = init_work_packet(&config);
        packet.flags = FLAG_PREFILL;
        packet.active_sequence_count = 1;
        packet.lane_count = 1;
        packet.execution_row_count = 1;
        packet.descriptor_bytes = calculate_packet_bytes(&config, 1);
        packet.block_token_count = 64;
        packet.kv_block_table_token_count = 1;
        packet.max_blocks_per_sequence = 4;
        packet.request_id = 100 + u64::from(sequence_index);
        packet.sequence_id = 200 + u64::from(sequence_index);
        packet.sequence_position = 0;
        packet.lanes[0].request_id = packet.request_id;
        packet.lanes[0].sequence_id = packet.sequence_id;
        packet.lanes[0].sequence_position = 0;
        packet.lanes[0].request_slot_index = sequence_index;
        packet.lanes[0].context_token_count = 1;
        let physical_block_index = {
            let view = build(&mut state, &mut packet, &config).unwrap();
            let index = view.host_physical_block_indices[0];
            assert!(index < 2);
            index
        };
        swap_storage.borrow_mut().physical_values[physical_block_index as usize] =
            1000 + sequence_index;
        assert_eq!(commit(&mut state, &mut packet, &config), Ok(()));
    }
    assert_eq!(state.directory_entry_count, 3);
    assert_eq!(state.allocated_physical_block_count, 2);
    assert_eq!(state.swapped_block_count, 1);
    assert_eq!(state.swap_store_count, 1);
    assert_eq!(swap_storage.borrow().store_count, 1);

    let mut packet = init_work_packet(&config);
    packet.active_sequence_count = 1;
    packet.lane_count = 1;
    packet.execution_row_count = 1;
    packet.descriptor_bytes = calculate_packet_bytes(&config, 1);
    packet.block_token_count = 64;
    packet.kv_block_table_token_count = 1;
    packet.max_blocks_per_sequence = 4;
    packet.request_id = 100;
    packet.sequence_id = 200;
    packet.sequence_position = 0;
    packet.lanes[0].request_id = packet.request_id;
    packet.lanes[0].sequence_id = packet.sequence_id;
    packet.lanes[0].sequence_position = 0;
    packet.lanes[0].request_slot_index = 0;
    packet.lanes[0].context_token_count = 1;
    let mut prefetch_packets = [packet.clone(), packet.clone()];
    for prefetch_packet in prefetch_packets.iter_mut() {
        refresh_work_transaction(prefetch_packet, &config);
    }
    let prefetch_entries =
        state.collect_kv_prefetch_entries(&prefetch_packets, &config, 2).unwrap();
    assert_eq!(prefetch_entries.len(), 1);
    assert_eq!(prefetch_entries[0].key, private_key(200, 0));
    assert!(prefetch_entries[0].backing_block_index < 8);
    let physical_block_index = {
        let view = build(&mut state, &mut packet, &config).unwrap();
        view.host_physical_block_indices[0]
    };
    assert_eq!(swap_storage.borrow().physical_values[physical_block_index as usize], 1000);
    assert_eq!(state.swap_load_count, 1);
    assert_eq!(swap_storage.borrow().load_count, 1);
    assert_eq!(state.swap_store_count, 2);

    packet.request_id = 101;
    packet.sequence_id = 201;
    packet.lanes[0].request_id = packet.request_id;
    packet.lanes[0].sequence_id = packet.sequence_id;
    build(&mut state, &mut packet, &config).unwrap();
    packet.request_id = 102;
    packet.sequence_id = 202;
    packet.lanes[0].request_id = packet.request_id;
    packet.lanes[0].sequence_id = packet.sequence_id;
    build(&mut state, &mut packet, &config).unwrap();
    assert_eq!(state.swap_load_count, 3);
    assert_eq!(swap_storage.borrow().load_count, 3);
    assert_eq!(state.swap_store_count, 3);
    assert_eq!(swap_storage.borrow().store_count, 3);
    assert_eq!(state.clean_evict_count, 1);

    assert_eq!(state.release_sequence(200, 1), Ok(()));
    assert_eq!(state.directory_entry_count, 2);
    assert_eq!(state.release_sequence(200, 1), Ok(()));

    let mut packet = WorkControlPacket::zeroed(&config);
    packet.set_lane_count(&config, 2);
    packet.magic = PACKET_MAGIC;
    packet.abi_version = ABI_VERSION;
    packet.control_generation = STANDALONE_GENERATION;
    packet.descriptor_bytes = calculate_packet_bytes(&config, 2);
    packet.flags = FLAG_RELEASE_SEQUENCES;
    packet.request_id = 101;
    packet.sequence_id = 201;
    packet.active_sequence_count = 2;
    packet.lane_count = 2;
    packet.block_token_count = 64;
    packet.kv_block_table_token_count = 1;
    packet.max_blocks_per_sequence = 4;
    for sequence_index in 0..2usize {
        packet.lanes[sequence_index].request_id = 101 + sequence_index as u64;
        packet.lanes[sequence_index].sequence_id = 201 + sequence_index as u64;
        packet.lanes[sequence_index].context_token_count = 1;
    }
    assert_eq!(validate(&packet, &config, 1024, 4), Ok(()));
    assert_eq!(release_packets(&mut state, &mut packet, &config), Ok(()));
    assert_eq!(state.directory_entry_count, 0);
    assert_eq!(state.allocated_physical_block_count, 0);
    assert_eq!(state.swapped_block_count, 0);
}

/// C: `SparkTestGlm52RingWorkControlBuildDecodeBatch`.
#[test]
fn build_decode_batch() {
    let config = test_config();
    let request_dispatch = RequestApiDispatch {
        kind: REQUEST_DISPATCH_KIND_DECODE_BATCH,
        request_count: 4,
        highest_priority: 77,
        decode_batch_decision_batch_bucket: STAGE_PLAN_BUCKET_B64,
        request_ids: vec![100, 101, 102, 103],
        request_handles: vec![1000, 1001, 1002, 1003],
        sequence_ids: vec![200, 201, 202, 203],
        ..RequestApiDispatch::default()
    };
    let mut decode_view = DecodeDispatchView {
        lane_count: 4,
        lanes: (0..4)
            .map(|lane_index| DecodeDispatchLaneView {
                request_index: lane_index,
                sequence_position: 31,
                context_token_count: 32,
                request_slot_index: lane_index,
                request_id: 100 + u64::from(lane_index),
                sequence_id: 200 + u64::from(lane_index),
                request_handle: 1000 + u64::from(lane_index),
                ..DecodeDispatchLaneView::default()
            })
            .collect(),
    };
    decode_view.lanes[0].mtp_resolution_base_position = 29;
    decode_view.lanes[0].mtp_resolution_proposed_token_count = 3;
    decode_view.lanes[0].mtp_resolution_accepted_token_count = 1;
    decode_view.lanes[0].mtp_resolution_committed_token_count = 2;
    let decode_dispatch = ServingDecodeDispatch {
        dispatch_kind: REQUEST_DISPATCH_KIND_DECODE_BATCH,
        request_count: 4,
        active_sequence_count: 4,
        request_dispatch,
        kv_block_table_view: Some(DispatchKvBlockTableView {
            block_token_count: 64,
            lane_count: 4,
            lane_stride: 2,
            lane_capacity: 4,
        }),
        decode_view,
        input_token_ids: vec![300, 301, 302, 303],
        speculative_token_count: 0,
        speculative_draft_token_ids: vec![
            vec![0; config.max_speculative_token_count() as usize];
            4
        ],
    };
    let packet = build_decode_packet(&config, &decode_dispatch, 0).unwrap();
    assert_eq!(packet.active_sequence_count, 4);
    assert_eq!(packet.execution_batch_bucket, STAGE_PLAN_BUCKET_B64);
    assert_eq!(packet.descriptor_bytes, calculate_packet_bytes(&config, 4));
    assert_eq!(packet.lanes[3].request_id, 103);
    assert_eq!(packet.lanes[3].input_token_id, 303);
    assert_eq!(packet.lanes[3].request_slot_index, 3);
    assert_ne!(packet.flags & FLAG_MTP_RESOLVE, 0);
    assert_eq!(packet.lanes[0].mtp_resolution_proposed_token_count, 3);
    assert_eq!(packet.lanes[0].mtp_resolution_accepted_token_count, 1);
    assert_eq!(validate(&packet, &config, 4, 1), Ok(()));
    let mut packet = packet;
    packet.flags &= !FLAG_MTP_RESOLVE;
    assert_eq!(validate(&packet, &config, 4, 1), Err(WorkControlError::InvalidArgument));
    let packet = build_decode_packet_range(&config, &decode_dispatch, 2, 2, 0).unwrap();
    assert_eq!(packet.active_sequence_count, 2);
    assert_eq!(packet.lanes[0].request_id, 102);
    assert_eq!(packet.lanes[1].request_id, 103);
    assert_eq!(packet.descriptor_bytes, calculate_packet_bytes(&config, 2));
    assert_eq!(validate(&packet, &config, 4, 1), Ok(()));
}

/// C: `SparkTestGlm52RingWorkControlBuildPackedMtpVerify`.
#[test]
fn build_packed_mtp_verify() {
    let config = test_config();
    let request_dispatch = RequestApiDispatch {
        kind: REQUEST_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH,
        flags: REQUEST_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY,
        request_count: 1,
        decode_batch_decision_batch_bucket: STAGE_PLAN_BUCKET_B16,
        request_ids: vec![100],
        request_handles: vec![1000],
        sequence_ids: vec![200],
        ..RequestApiDispatch::default()
    };
    let decode_view = DecodeDispatchView {
        lane_count: 1,
        lanes: vec![DecodeDispatchLaneView {
            request_index: 0,
            sequence_position: 32,
            context_token_count: 33,
            request_slot_index: 7,
            request_id: 100,
            sequence_id: 200,
            request_handle: 1000,
            mtp_resolution_proposed_token_count: 2,
            mtp_resolution_accepted_token_count: 1,
            ..DecodeDispatchLaneView::default()
        }],
    };
    let decode_dispatch = ServingDecodeDispatch {
        dispatch_kind: REQUEST_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH,
        request_count: 1,
        active_sequence_count: 1,
        request_dispatch,
        kv_block_table_view: Some(DispatchKvBlockTableView::default()),
        decode_view,
        input_token_ids: vec![300],
        speculative_token_count: 3,
        speculative_draft_token_ids: vec![vec![400, 401, 402, 0, 0, 0, 0]],
    };
    let packet = build_decode_packet(&config, &decode_dispatch, 0).unwrap();
    assert_eq!(packet.rows_per_lane, 4);
    assert_eq!(packet.execution_row_count, 4);
    assert_eq!(packet.execution_batch_bucket, STAGE_PLAN_BUCKET_B16);
    assert_eq!(packet.new_token_count, 4);
    assert_eq!(packet.sequence_position, 32);
    assert_eq!(packet.input_token_id, 300);
    assert_ne!(packet.flags & FLAG_MTP_RESOLVE, 0);
    assert_eq!(validate(&packet, &config, 7, 1), Ok(()));
    assert_eq!(
        build_decode_packet(&config, &decode_dispatch, 1).map(|_| ()),
        Err(WorkControlError::InvalidArgument)
    );
}

/// C: `SparkTestGlm52RingWorkControlBuildPrefillBatch`.
#[test]
fn build_prefill_batch() {
    let config = test_config();
    let request_dispatch = RequestApiDispatch {
        kind: REQUEST_DISPATCH_KIND_PREFILL_BATCH,
        request_count: 4,
        highest_priority: 81,
        prefill_batch_decision_batch_bucket: STAGE_PLAN_BUCKET_B32,
        request_ids: vec![100, 101, 102, 103],
        request_handles: vec![2000, 2001, 2002, 2003],
        sequence_ids: vec![200, 201, 202, 203],
        ..RequestApiDispatch::default()
    };
    let prefill_view = PrefillDispatchView {
        lane_count: 4,
        prompt_token_count: 2,
        prompt_token_stride: 2,
        lanes: (0..4)
            .map(|lane_index| PrefillDispatchLaneView {
                request_index: lane_index,
                prompt_token_offset: 4,
                prompt_token_count: if lane_index == 1 { 1 } else { 2 },
                request_slot_index: lane_index,
                request_id: 100 + u64::from(lane_index),
                sequence_id: 200 + u64::from(lane_index),
                request_handle: 2000 + u64::from(lane_index),
            })
            .collect(),
    };
    let mut prefill_dispatch = PromptPipelinePrefillDispatch {
        dispatch_kind: REQUEST_DISPATCH_KIND_PREFILL_BATCH,
        active_sequence_count: 4,
        lane_count: 4,
        prompt_token_count: 2,
        prompt_token_stride: 2,
        host_token_stride: 2,
        request_dispatch,
        prefill_view,
        // token_ids[lane][token] = {300+i, 400+i}, flattened by stride 2.
        host_token_ids: vec![300, 400, 301, 401, 302, 402, 303, 403],
        kv_block_table_view: Some(DispatchKvBlockTableView {
            block_token_count: 64,
            lane_count: 4,
            lane_stride: 1,
            lane_capacity: 4,
        }),
        ..PromptPipelinePrefillDispatch::default()
    };
    assert_eq!(select_prefill_chunk(&config, &prefill_dispatch, 0, 8), Ok(1));
    let packet = build_prefill_packet(&config, &prefill_dispatch, 0, 1).unwrap();
    assert_eq!(packet.active_sequence_count, 4);
    assert_eq!(packet.rows_per_lane, 1);
    assert_eq!(packet.new_token_count, 1);
    assert_eq!(packet.execution_row_count, 4);
    assert_eq!(packet.lanes[0].input_token_id, 300);
    assert_eq!(packet.lanes[1].input_token_id, 301);
    assert_eq!(packet.lanes[1].context_token_count, 5);
    assert_eq!(validate(&packet, &config, 4, 1), Ok(()));
    let mut packet = packet;
    packet.lanes[0].input_token_id = config.output_vocab_count;
    assert_eq!(validate(&packet, &config, 4, 1), Err(WorkControlError::InvalidArgument));
    packet.lanes[0].input_token_id = 300;
    assert_eq!(
        build_prefill_packet(&config, &prefill_dispatch, 0, 2).map(|_| ()),
        Err(WorkControlError::InvalidArgument)
    );
    prefill_dispatch.prefill_view.lanes[1].prompt_token_count = 2;
    assert_eq!(select_prefill_chunk(&config, &prefill_dispatch, 0, 8), Ok(2));
    let packet = build_prefill_packet(&config, &prefill_dispatch, 0, 2).unwrap();
    assert_eq!(packet.active_sequence_count, 4);
    assert_eq!(packet.rows_per_lane, 2);
    assert_eq!(packet.new_token_count, 2);
    assert_eq!(packet.execution_row_count, 8);
    assert_eq!(packet.prefill_token_ids[0], 300);
    assert_eq!(packet.prefill_token_ids[1], 400);
    assert_eq!(packet.prefill_token_ids[2], 301);
    assert_eq!(packet.prefill_token_ids[7], 403);
    assert_eq!(packet.lanes[0].input_token_id, 400);
    assert_eq!(packet.lanes[0].context_token_count, 6);
    assert_eq!(validate(&packet, &config, 8, 1), Ok(()));
    let mut packet = packet;
    packet.prefill_token_ids[7] = config.output_vocab_count;
    assert_eq!(validate(&packet, &config, 8, 1), Err(WorkControlError::InvalidArgument));
    prefill_dispatch.prefill_view.lanes[1].prompt_token_count = 1;
    assert_eq!(select_prefill_chunk(&config, &prefill_dispatch, 1, 8), Ok(1));
    let packet = build_prefill_packet(&config, &prefill_dispatch, 1, 1).unwrap();
    assert_eq!(packet.active_sequence_count, 3);
    assert_eq!(packet.execution_batch_bucket, STAGE_PLAN_BUCKET_B32);
    assert_eq!(packet.descriptor_bytes, calculate_packet_bytes(&config, 3));
    // C also asserts `SparkCudaResidentIpcCalculateSubmitPrefillBytes` here;
    // that belongs to the resident-IPC module, not work control.
    assert_eq!(packet.lanes[0].request_id, 100);
    assert_eq!(packet.lanes[1].request_id, 102);
    assert_eq!(packet.lanes[1].input_token_id, 402);
    assert_eq!(packet.lanes[1].sequence_position, 5);
    assert_eq!(packet.lanes[1].request_slot_index, 2);
    assert_eq!(validate(&packet, &config, 4, 1), Ok(()));
}

/// C: `SparkTestGlm52RingWorkControlResetsOlderGeneration`.
#[test]
fn resets_older_generation() {
    let config = test_config();
    let mut state = init_kv_state(1);
    let mut packet = init_work_packet(&config);
    packet.flags = FLAG_PREFILL;
    packet.control_generation = 100;
    packet.active_sequence_count = 1;
    packet.lane_count = 1;
    packet.execution_row_count = 1;
    packet.descriptor_bytes = calculate_packet_bytes(&config, 1);
    packet.request_id = 101;
    packet.sequence_id = 201;
    packet.sequence_position = 0;
    packet.kv_block_table_token_count = 1;
    packet.lanes[0].request_id = packet.request_id;
    packet.lanes[0].sequence_id = packet.sequence_id;
    packet.lanes[0].sequence_position = 0;
    packet.lanes[0].context_token_count = 1;
    packet.prefill_token_ids[0] = packet.lanes[0].input_token_id;
    build(&mut state, &mut packet, &config).unwrap();
    assert_eq!(commit(&mut state, &mut packet, &config), Ok(()));
    assert_eq!(state.directory_entry_count, 1);
    assert_eq!(state.control_generation, 100);
    assert_eq!(state.control_generation_reset_count, 1);
    assert_eq!(state.advance_kv_generation(200), Ok(()));
    assert_eq!(state.directory_entry_count, 0);
    assert_eq!(state.allocated_physical_block_count, 0);
    assert_eq!(state.control_generation, 200);
    assert_eq!(state.control_generation_reset_count, 2);
    packet.control_generation = 200;
    packet.request_id = 102;
    packet.sequence_id = 202;
    packet.lanes[0].request_id = packet.request_id;
    packet.lanes[0].sequence_id = packet.sequence_id;
    build(&mut state, &mut packet, &config).unwrap();
    assert_eq!(state.directory_entry_count, 1);
    assert_eq!(state.allocated_physical_block_count, 1);
    assert_eq!(state.control_generation, 200);
    assert_eq!(state.control_generation_reset_count, 2);
    packet.control_generation = 100;
    assert_eq!(state.advance_kv_generation(100), Err(WorkControlError::ValidationFailed));
    assert_eq!(
        build(&mut state, &mut packet, &config).map(|_| ()),
        Err(WorkControlError::NotFound)
    );
    assert_eq!(state.directory_entry_count, 1);
    assert_eq!(state.control_generation, 200);
}

/// C: `SparkTestGlm52RingWorkControlPinsSpeculativeBlocks`.
#[test]
fn pins_speculative_blocks() {
    let config = test_config();
    let mut state = init_kv_state(1);
    let mut packet = init_work_packet(&config);
    packet.flags = FLAG_PREFILL;
    packet.active_sequence_count = 1;
    packet.lane_count = 1;
    packet.execution_row_count = 1;
    packet.descriptor_bytes = calculate_packet_bytes(&config, 1);
    packet.sequence_position = 0;
    packet.kv_block_table_token_count = 1;
    packet.lanes[0].request_id = packet.request_id;
    packet.lanes[0].sequence_id = packet.sequence_id;
    packet.lanes[0].sequence_position = 0;
    packet.lanes[0].request_slot_index = 0;
    packet.lanes[0].context_token_count = 1;
    packet.prefill_token_ids[0] = packet.lanes[0].input_token_id;
    let physical_block_index = {
        let view = build(&mut state, &mut packet, &config).unwrap();
        view.host_physical_block_indices[0]
    };
    assert_eq!(commit(&mut state, &mut packet, &config), Ok(()));
    assert_eq!(state.pin_physical_block(physical_block_index), Ok(()));
    assert_eq!(state.physical_block_pin_counts.as_ref().unwrap()[physical_block_index as usize], 1);
    assert_eq!(state.release_sequence(packet.sequence_id, 1), Err(WorkControlError::Busy));
    assert_eq!(state.directory_entry_count, 1);
    assert_eq!(state.unpin_physical_block(physical_block_index), Ok(()));
    assert_eq!(state.release_sequence(packet.sequence_id, 1), Ok(()));
    assert_eq!(state.directory_entry_count, 0);
}

/// Invariants that cross-cut the C scenarios: the queue-depth invariant and
/// the MTP tree topology the resolution validation relies on.
#[test]
fn design_invariants() {
    let config = test_config();
    // QUEUE_DEPTH = cohort capacity + 1 (per-step batch re-formation).
    assert_eq!(queue_depth(config.cohort_capacity), config.cohort_capacity + 1);
    assert!(mtp_tree::topology_is_valid());
    assert!(work_transaction::phase_is_valid(work_transaction::PHASE_PREFILL));
    assert!(work_transaction::phase_is_valid(work_transaction::PHASE_CANCEL));
    assert!(!work_transaction::phase_is_valid(0));
}
