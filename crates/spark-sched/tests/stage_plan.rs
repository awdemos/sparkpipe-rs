//! Port of `tests/test_glm52_stage_plan.c`. GPU-free: the geometry that the
//! C test compiles in from `SPARK_GLM52_MODEL_*` (78 layers, first routed
//! layer 3) is supplied here as an explicit [`StagePlanGeometry`], matching
//! the Rust port's runtime-geometry deviation.

use spark_sched::stage_plan::{
    self, QuantizationMode, StagePlan, StagePlanError, StagePlanGeometry, StagePlanStage,
    ABI_VERSION, BUCKET_B1024, BUCKET_B128, BUCKET_B16, BUCKET_B256, BUCKET_B32, BUCKET_B512,
    BUCKET_B64, CURRENT_SPARK_COUNT, DESCRIPTOR_BYTES, MEASURED_PROFILE_20260701,
    PIPELINE_INFLIGHT_REQUEST_CAPACITY, STAGE_FLAG_DENSE_PREFIX, STAGE_FLAG_FINAL_TOKEN,
    STAGE_FLAG_INPUT_HIDDEN, STAGE_FLAG_OUTPUT_HIDDEN,
};

const GLM52_GEOMETRY: StagePlanGeometry =
    StagePlanGeometry { layer_count: 78, first_routed_layer: 3 };

/// `SparkTestGlm52StagePlanMaximumStageCostNs`.
fn maximum_stage_cost_ns(
    stage_plan: &StagePlan,
    layer_cost_ns: &[u64],
    final_stage_extra_cost_ns: u64,
) -> u64 {
    let mut maximum_stage_cost_ns = 0u64;
    for stage in &stage_plan.stages[..stage_plan.stage_count as usize] {
        let mut stage_cost_ns = 0u64;
        for layer_offset in 0..stage.layer_count {
            stage_cost_ns += layer_cost_ns[(stage.first_layer_index + layer_offset) as usize];
        }
        if stage.flags & STAGE_FLAG_FINAL_TOKEN != 0 {
            stage_cost_ns += final_stage_extra_cost_ns;
        }
        if stage_cost_ns > maximum_stage_cost_ns {
            maximum_stage_cost_ns = stage_cost_ns;
        }
    }
    maximum_stage_cost_ns
}

/// `SparkTestGlm52StagePlanValidRing`.
#[test]
fn valid_ring() {
    assert_eq!(PIPELINE_INFLIGHT_REQUEST_CAPACITY, 13 * 1024);

    let mut stage_plan = StagePlan {
        abi_version: ABI_VERSION,
        descriptor_bytes: DESCRIPTOR_BYTES,
        stage_count: 13,
        reserved: 0,
        stages: Vec::new(),
    };
    for stage_index in 0..13u32 {
        stage_plan.stages.push(StagePlanStage {
            first_layer_index: stage_index * 6,
            layer_count: 6,
            flags: STAGE_FLAG_INPUT_HIDDEN | STAGE_FLAG_OUTPUT_HIDDEN,
            reserved: 0,
        });
    }
    stage_plan.stages[0].flags |= STAGE_FLAG_DENSE_PREFIX;
    stage_plan.stages[12].flags = STAGE_FLAG_INPUT_HIDDEN | STAGE_FLAG_FINAL_TOKEN;

    assert!(stage_plan::validate(&GLM52_GEOMETRY, &stage_plan).is_ok());
}

/// `SparkTestGlm52StagePlanLayerCountTable`.
#[test]
fn layer_count_table() {
    let layer_counts = [6u32, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6];

    let stage_plan = stage_plan::build_from_layer_counts(&GLM52_GEOMETRY, &layer_counts).unwrap();
    assert_eq!(stage_plan.stage_count, 12);
    assert_eq!(stage_plan.stages[0].first_layer_index, 0);
    assert_eq!(stage_plan.stages[0].layer_count, 6);
    assert_eq!(stage_plan.stages[1].first_layer_index, 6);
    assert_eq!(stage_plan.stages[1].layer_count, 7);
    assert_eq!(stage_plan.stages[11].first_layer_index, 72);
    assert_eq!(stage_plan.stages[11].layer_count, 6);
    assert!(stage_plan.stages[11].flags & STAGE_FLAG_FINAL_TOKEN != 0);

    let mut invalid_layer_counts = layer_counts;
    invalid_layer_counts[11] = 5;
    assert_eq!(
        stage_plan::build_from_layer_counts(&GLM52_GEOMETRY, &invalid_layer_counts),
        Err(StagePlanError::InvalidArgument("stage plan does not cover all GLM-5.2 layers"))
    );
}

/// `SparkTestGlm52StagePlanBuilderAndBuckets`.
#[test]
fn builder_and_buckets() {
    let layer_cost_ns: Vec<u64> =
        (0..GLM52_GEOMETRY.layer_count as u64).map(|layer_index| 1000 + layer_index).collect();
    let stage_plan = stage_plan::build_balanced(&GLM52_GEOMETRY, &layer_cost_ns, 13).unwrap();
    assert_eq!(stage_plan.stage_count, 13);
    assert_eq!(stage_plan.stages[0].first_layer_index, 0);
    assert_eq!(
        stage_plan.stages[12].first_layer_index + stage_plan.stages[12].layer_count,
        GLM52_GEOMETRY.layer_count
    );
    assert!(stage_plan.stages[12].flags & STAGE_FLAG_FINAL_TOKEN != 0);

    assert_eq!(stage_plan::select_batch_bucket(1), Ok(BUCKET_B16));
    assert_eq!(stage_plan::select_batch_bucket(17), Ok(BUCKET_B32));
    assert_eq!(stage_plan::select_batch_bucket(33), Ok(BUCKET_B64));
    assert_eq!(stage_plan::select_batch_bucket(65), Ok(BUCKET_B128));
    assert_eq!(stage_plan::select_batch_bucket(129), Ok(BUCKET_B256));
    assert_eq!(stage_plan::select_batch_bucket(257), Ok(BUCKET_B512));
    assert_eq!(stage_plan::select_batch_bucket(513), Ok(BUCKET_B1024));
    assert!(matches!(
        stage_plan::select_batch_bucket(1025),
        Err(StagePlanError::CapacityExceeded(_))
    ));
}

/// `SparkTestGlm52StagePlanMeasuredBalanced`.
#[test]
fn measured_balanced() {
    let profile = stage_plan::load_measured_cost_profile(
        &GLM52_GEOMETRY,
        MEASURED_PROFILE_20260701,
        BUCKET_B64,
    )
    .unwrap();
    assert_eq!(profile.final_stage_extra_cost_ns, 0);
    let stage_plan = stage_plan::build_current_spark_measured_balanced(
        &GLM52_GEOMETRY,
        MEASURED_PROFILE_20260701,
        BUCKET_B64,
    )
    .unwrap();
    assert_eq!(stage_plan.stage_count, 13);
    assert_eq!(stage_plan.stages[0].first_layer_index, 0);
    assert_eq!(stage_plan.stages[0].layer_count, 6);
    assert_eq!(stage_plan.stages[1].first_layer_index, 6);
    assert_eq!(stage_plan.stages[1].layer_count, 6);
    assert_eq!(stage_plan.stages[2].first_layer_index, 12);
    assert_eq!(stage_plan.stages[2].layer_count, 6);
    assert_eq!(stage_plan.stages[3].first_layer_index, 18);
    assert_eq!(stage_plan.stages[3].layer_count, 6);
    assert_eq!(stage_plan.stages[4].first_layer_index, 24);
    assert_eq!(stage_plan.stages[4].layer_count, 6);
    assert_eq!(stage_plan.stages[12].first_layer_index, 72);
    assert_eq!(stage_plan.stages[12].layer_count, 6);
    assert!(stage_plan.stages[12].flags & STAGE_FLAG_FINAL_TOKEN != 0);
    assert_eq!(
        maximum_stage_cost_ns(
            &stage_plan,
            &profile.layer_cost_ns,
            profile.final_stage_extra_cost_ns
        ),
        50660288
    );

    let profile = stage_plan::load_measured_cost_profile(
        &GLM52_GEOMETRY,
        MEASURED_PROFILE_20260701,
        BUCKET_B128,
    )
    .unwrap();
    assert_eq!(profile.final_stage_extra_cost_ns, 0);
    let stage_plan = stage_plan::build_current_spark_measured_balanced(
        &GLM52_GEOMETRY,
        MEASURED_PROFILE_20260701,
        BUCKET_B128,
    )
    .unwrap();
    assert_eq!(stage_plan.stage_count, 13);
    assert_eq!(stage_plan.stages[0].first_layer_index, 0);
    assert_eq!(stage_plan.stages[0].layer_count, 6);
    assert_eq!(stage_plan.stages[12].first_layer_index, 72);
    assert_eq!(stage_plan.stages[12].layer_count, 6);
    assert_eq!(
        maximum_stage_cost_ns(
            &stage_plan,
            &profile.layer_cost_ns,
            profile.final_stage_extra_cost_ns
        ),
        197361562
    );

    let profile = stage_plan::load_measured_cost_profile(
        &GLM52_GEOMETRY,
        MEASURED_PROFILE_20260701,
        BUCKET_B256,
    )
    .unwrap();
    assert_eq!(profile.final_stage_extra_cost_ns, 0);
    let stage_plan = stage_plan::build_current_spark_measured_balanced(
        &GLM52_GEOMETRY,
        MEASURED_PROFILE_20260701,
        BUCKET_B256,
    )
    .unwrap();
    assert_eq!(stage_plan.stage_count, 13);
    assert_eq!(stage_plan.stages[0].first_layer_index, 0);
    assert_eq!(stage_plan.stages[0].layer_count, 6);
    assert_eq!(stage_plan.stages[12].first_layer_index, 72);
    assert_eq!(stage_plan.stages[12].layer_count, 6);

    let profile = stage_plan::load_measured_cost_profile(
        &GLM52_GEOMETRY,
        MEASURED_PROFILE_20260701,
        BUCKET_B32,
    )
    .unwrap();
    assert_eq!(profile.final_stage_extra_cost_ns, 39342000);
    let stage_plan = stage_plan::build_current_spark_measured_balanced(
        &GLM52_GEOMETRY,
        MEASURED_PROFILE_20260701,
        BUCKET_B32,
    )
    .unwrap();
    assert_eq!(stage_plan.stage_count, CURRENT_SPARK_COUNT);
    assert_eq!(stage_plan.stages[0].first_layer_index, 0);
    assert_eq!(stage_plan.stages[0].layer_count, 10);
    assert_eq!(stage_plan.stages[12].first_layer_index, 77);
    assert_eq!(stage_plan.stages[12].layer_count, 1);
    assert!(stage_plan.stages[12].flags & STAGE_FLAG_FINAL_TOKEN != 0);
    assert_eq!(
        maximum_stage_cost_ns(
            &stage_plan,
            &profile.layer_cost_ns,
            profile.final_stage_extra_cost_ns
        ),
        63119000
    );

    assert!(matches!(
        stage_plan::build_measured_balanced(
            &GLM52_GEOMETRY,
            MEASURED_PROFILE_20260701,
            BUCKET_B32,
            CURRENT_SPARK_COUNT + 1,
        ),
        Err(StagePlanError::InvalidArgument(_))
    ));
}

/// `SparkTestGlm52StagePlanMeasuredBalancedQuantizationModes`.
#[test]
fn measured_balanced_quantization_modes() {
    let profile_4bit = stage_plan::load_measured_cost_profile_for_quantization(
        &GLM52_GEOMETRY,
        MEASURED_PROFILE_20260701,
        BUCKET_B64,
        QuantizationMode::Nvfp4_4Bit,
    )
    .unwrap();
    let profile_8bit = stage_plan::load_measured_cost_profile_for_quantization(
        &GLM52_GEOMETRY,
        MEASURED_PROFILE_20260701,
        BUCKET_B64,
        QuantizationMode::Fp8E4m3_8Bit,
    )
    .unwrap();
    let profile_auto = stage_plan::load_measured_cost_profile(
        &GLM52_GEOMETRY,
        MEASURED_PROFILE_20260701,
        BUCKET_B64,
    )
    .unwrap();
    assert_eq!(profile_4bit.final_stage_extra_cost_ns, 0);
    assert_eq!(profile_8bit.final_stage_extra_cost_ns, 0);
    assert_eq!(profile_auto.final_stage_extra_cost_ns, 0);
    assert_eq!(profile_4bit.layer_cost_ns, profile_8bit.layer_cost_ns);
    assert_eq!(profile_8bit.layer_cost_ns, profile_auto.layer_cost_ns);

    let stage_plan_4bit = stage_plan::build_current_spark_measured_balanced_for_quantization(
        &GLM52_GEOMETRY,
        MEASURED_PROFILE_20260701,
        BUCKET_B64,
        QuantizationMode::Nvfp4_4Bit,
    )
    .unwrap();
    let stage_plan_8bit = stage_plan::build_current_spark_measured_balanced_for_quantization(
        &GLM52_GEOMETRY,
        MEASURED_PROFILE_20260701,
        BUCKET_B64,
        QuantizationMode::Fp8E4m3_8Bit,
    )
    .unwrap();
    let stage_plan_auto = stage_plan::build_current_spark_measured_balanced(
        &GLM52_GEOMETRY,
        MEASURED_PROFILE_20260701,
        BUCKET_B64,
    )
    .unwrap();
    assert_eq!(stage_plan_4bit, stage_plan_8bit);
    assert_eq!(stage_plan_8bit, stage_plan_auto);

    // The C test passes the raw mode value 99, which normalization rejects;
    // the Rust API makes unknown modes unrepresentable, so exercise the
    // equivalent invalid-profile rejection path instead.
    assert!(matches!(
        stage_plan::build_current_spark_measured_balanced_for_quantization(
            &GLM52_GEOMETRY,
            99,
            BUCKET_B64,
            QuantizationMode::Nvfp4_4Bit,
        ),
        Err(StagePlanError::InvalidArgument(_))
    ));
}

/// `SparkTestGlm52StagePlanInvalidCuts`.
#[test]
fn invalid_cuts() {
    let mut stage_plan = stage_plan::build_uniform(&GLM52_GEOMETRY, 13).unwrap();
    stage_plan.stages[1].layer_count = 9;
    stage_plan.stages[2].first_layer_index = 15;
    assert!(matches!(
        stage_plan::validate(&GLM52_GEOMETRY, &stage_plan),
        Err(StagePlanError::InvalidArgument(_))
    ));

    let mut stage_plan = stage_plan::build_uniform(&GLM52_GEOMETRY, 13).unwrap();
    stage_plan.stages[0].flags |= STAGE_FLAG_FINAL_TOKEN;
    assert!(matches!(
        stage_plan::validate(&GLM52_GEOMETRY, &stage_plan),
        Err(StagePlanError::InvalidArgument(_))
    ));
}
