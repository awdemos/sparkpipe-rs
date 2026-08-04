//! Fail-closed layout gate: every ABI struct's Rust `size_of`/`align_of` must
//! equal the C compiler's `sizeof`/`_Alignof`. Mirrors the C tree's
//! `descriptor_bytes` discipline — if a C header changes layout, this test
//! fails before any FFI call can corrupt memory.

use spark_abi::layout;
use spark_abi::*;

macro_rules! check {
    ($ty:ident, $size:ident, $align:ident) => {
        assert_eq!(
            core::mem::size_of::<$ty>(),
            layout::$size,
            concat!(stringify!($ty), ": Rust size != C sizeof")
        );
        assert_eq!(
            core::mem::align_of::<$ty>(),
            layout::$align,
            concat!(stringify!($ty), ": Rust align != C _Alignof")
        );
    };
}

#[test]
fn c_layout_matches_rust_layout() {
    check!(SparkModelDriverBuffer, SparkModelDriverBuffer_SIZE, SparkModelDriverBuffer_ALIGN);
    check!(
        SparkModelDriverResidencyToken,
        SparkModelDriverResidencyToken_SIZE,
        SparkModelDriverResidencyToken_ALIGN
    );
    check!(
        SparkModelDriverCompletion,
        SparkModelDriverCompletion_SIZE,
        SparkModelDriverCompletion_ALIGN
    );
    check!(SparkModelDriverFrame, SparkModelDriverFrame_SIZE, SparkModelDriverFrame_ALIGN);
    check!(
        SparkModelDriverProgramProfile,
        SparkModelDriverProgramProfile_SIZE,
        SparkModelDriverProgramProfile_ALIGN
    );
    check!(
        SparkModelDriverAdmissionRequest,
        SparkModelDriverAdmissionRequest_SIZE,
        SparkModelDriverAdmissionRequest_ALIGN
    );
    check!(
        SparkModelDriverAdmissionDecision,
        SparkModelDriverAdmissionDecision_SIZE,
        SparkModelDriverAdmissionDecision_ALIGN
    );
    check!(
        SparkModelDriverRuntimeSnapshot,
        SparkModelDriverRuntimeSnapshot_SIZE,
        SparkModelDriverRuntimeSnapshot_ALIGN
    );
    check!(
        SparkModelDriverCreateRequest,
        SparkModelDriverCreateRequest_SIZE,
        SparkModelDriverCreateRequest_ALIGN
    );
    check!(
        SparkModelDriverProgramDescriptor,
        SparkModelDriverProgramDescriptor_SIZE,
        SparkModelDriverProgramDescriptor_ALIGN
    );
    check!(
        SparkModelDriverDescriptor,
        SparkModelDriverDescriptor_SIZE,
        SparkModelDriverDescriptor_ALIGN
    );
    check!(
        SparkModelDriverInterface,
        SparkModelDriverInterface_SIZE,
        SparkModelDriverInterface_ALIGN
    );
    check!(
        SparkFirmwareModuleConfiguration,
        SparkFirmwareModuleConfiguration_SIZE,
        SparkFirmwareModuleConfiguration_ALIGN
    );
    check!(
        SparkFirmwareModuleHostServices,
        SparkFirmwareModuleHostServices_SIZE,
        SparkFirmwareModuleHostServices_ALIGN
    );
    check!(
        SparkKvCacheCapacityRequest,
        SparkKvCacheCapacityRequest_SIZE,
        SparkKvCacheCapacityRequest_ALIGN
    );
    check!(
        SparkKvCacheCapacityEstimate,
        SparkKvCacheCapacityEstimate_SIZE,
        SparkKvCacheCapacityEstimate_ALIGN
    );
    check!(
        SparkKvJitStageBudgetRequest,
        SparkKvJitStageBudgetRequest_SIZE,
        SparkKvJitStageBudgetRequest_ALIGN
    );
    check!(SparkKvJitStageBudget, SparkKvJitStageBudget_SIZE, SparkKvJitStageBudget_ALIGN);
    check!(
        SparkKvCachePrefetchSourceBlock,
        SparkKvCachePrefetchSourceBlock_SIZE,
        SparkKvCachePrefetchSourceBlock_ALIGN
    );
    check!(
        SparkKvCachePrefetchBlock,
        SparkKvCachePrefetchBlock_SIZE,
        SparkKvCachePrefetchBlock_ALIGN
    );
    check!(SparkKvCachePrefetchPlan, SparkKvCachePrefetchPlan_SIZE, SparkKvCachePrefetchPlan_ALIGN);
    check!(
        SparkKvCachePrefetchBackendSourceEntry,
        SparkKvCachePrefetchBackendSourceEntry_SIZE,
        SparkKvCachePrefetchBackendSourceEntry_ALIGN
    );
    check!(
        SparkKvCacheAsyncPrefetchBackendConfiguration,
        SparkKvCacheAsyncPrefetchBackendConfiguration_SIZE,
        SparkKvCacheAsyncPrefetchBackendConfiguration_ALIGN
    );
    check!(
        SparkKvCacheAsyncPrefetchRequest,
        SparkKvCacheAsyncPrefetchRequest_SIZE,
        SparkKvCacheAsyncPrefetchRequest_ALIGN
    );
    check!(
        SparkKvCacheAsyncPrefetchBackend,
        SparkKvCacheAsyncPrefetchBackend_SIZE,
        SparkKvCacheAsyncPrefetchBackend_ALIGN
    );
    check!(SparkKvCacheBlock, SparkKvCacheBlock_SIZE, SparkKvCacheBlock_ALIGN);
    check!(
        SparkKvCacheConfiguration,
        SparkKvCacheConfiguration_SIZE,
        SparkKvCacheConfiguration_ALIGN
    );
    check!(SparkKvCacheBlockView, SparkKvCacheBlockView_SIZE, SparkKvCacheBlockView_ALIGN);
    check!(SparkKvBlockTableView, SparkKvBlockTableView_SIZE, SparkKvBlockTableView_ALIGN);
    check!(SparkKvCacheArena, SparkKvCacheArena_SIZE, SparkKvCacheArena_ALIGN);
    check!(SparkStagePlanGeometry, SparkStagePlanGeometry_SIZE, SparkStagePlanGeometry_ALIGN);
    check!(SparkStagePlanStage, SparkStagePlanStage_SIZE, SparkStagePlanStage_ALIGN);
    check!(SparkStagePlan, SparkStagePlan_SIZE, SparkStagePlan_ALIGN);
    // SparkHiddenTransportSession is an opaque pimpl handle (forward-declared
    // only in the C headers) — no layout to gate.
    check!(
        SparkHiddenTransportEndpoint,
        SparkHiddenTransportEndpoint_SIZE,
        SparkHiddenTransportEndpoint_ALIGN
    );
    check!(
        SparkHiddenTransportPacket,
        SparkHiddenTransportPacket_SIZE,
        SparkHiddenTransportPacket_ALIGN
    );
    check!(
        SparkHiddenTransportCompletion,
        SparkHiddenTransportCompletion_SIZE,
        SparkHiddenTransportCompletion_ALIGN
    );
    check!(
        SparkHiddenTransportCompletionQueue,
        SparkHiddenTransportCompletionQueue_SIZE,
        SparkHiddenTransportCompletionQueue_ALIGN
    );
    check!(
        SparkHiddenTransportPersistentRingStatistics,
        SparkHiddenTransportPersistentRingStatistics_SIZE,
        SparkHiddenTransportPersistentRingStatistics_ALIGN
    );
    check!(
        SparkHiddenTransportPollDescriptor,
        SparkHiddenTransportPollDescriptor_SIZE,
        SparkHiddenTransportPollDescriptor_ALIGN
    );
    check!(
        SparkHiddenTransportInterface,
        SparkHiddenTransportInterface_SIZE,
        SparkHiddenTransportInterface_ALIGN
    );
    check!(
        SparkHiddenTransportDynamicLibrary,
        SparkHiddenTransportDynamicLibrary_SIZE,
        SparkHiddenTransportDynamicLibrary_ALIGN
    );
    check!(
        SparkResidentDecodeStagePrefillFrameView,
        SparkResidentDecodeStagePrefillFrameView_SIZE,
        SparkResidentDecodeStagePrefillFrameView_ALIGN
    );
    check!(
        SparkResidentDecodeStageFrameContext,
        SparkResidentDecodeStageFrameContext_SIZE,
        SparkResidentDecodeStageFrameContext_ALIGN
    );
    check!(
        SparkResidentDecodeStageQuantizedLinearView,
        SparkResidentDecodeStageQuantizedLinearView_SIZE,
        SparkResidentDecodeStageQuantizedLinearView_ALIGN
    );
    check!(
        SparkResidentDecodeStageLinearPlan,
        SparkResidentDecodeStageLinearPlan_SIZE,
        SparkResidentDecodeStageLinearPlan_ALIGN
    );
    check!(
        SparkResidentDecodeStageFp8KvCachePlan,
        SparkResidentDecodeStageFp8KvCachePlan_SIZE,
        SparkResidentDecodeStageFp8KvCachePlan_ALIGN
    );
    check!(
        SparkResidentDecodeStageRestrictedLogitsPlan,
        SparkResidentDecodeStageRestrictedLogitsPlan_SIZE,
        SparkResidentDecodeStageRestrictedLogitsPlan_ALIGN
    );
    check!(
        SparkResidentDecodeStageMtpDraftPlan,
        SparkResidentDecodeStageMtpDraftPlan_SIZE,
        SparkResidentDecodeStageMtpDraftPlan_ALIGN
    );
    check!(
        SparkResidentDecodeStageFullStagePlan,
        SparkResidentDecodeStageFullStagePlan_SIZE,
        SparkResidentDecodeStageFullStagePlan_ALIGN
    );
    check!(
        SparkResidentDecodeStageStageSlicePlan,
        SparkResidentDecodeStageStageSlicePlan_SIZE,
        SparkResidentDecodeStageStageSlicePlan_ALIGN
    );
    check!(
        SparkResidentDecodeStageExactStageSlicePlan,
        SparkResidentDecodeStageExactStageSlicePlan_SIZE,
        SparkResidentDecodeStageExactStageSlicePlan_ALIGN
    );
    check!(
        SparkResidentDecodeStagePagedPrefillPlan,
        SparkResidentDecodeStagePagedPrefillPlan_SIZE,
        SparkResidentDecodeStagePagedPrefillPlan_ALIGN
    );
    check!(
        SparkResidentDecodeStageBulkPrefillPlan,
        SparkResidentDecodeStageBulkPrefillPlan_SIZE,
        SparkResidentDecodeStageBulkPrefillPlan_ALIGN
    );
    check!(
        SparkResidentDecodeStageCudaPipelineSlotState,
        SparkResidentDecodeStageCudaPipelineSlotState_SIZE,
        SparkResidentDecodeStageCudaPipelineSlotState_ALIGN
    );
    check!(
        SparkResidentDecodeStagePipelineSlot,
        SparkResidentDecodeStagePipelineSlot_SIZE,
        SparkResidentDecodeStagePipelineSlot_ALIGN
    );
    check!(
        SparkResidentDecodeStageNodeContext,
        SparkResidentDecodeStageNodeContext_SIZE,
        SparkResidentDecodeStageNodeContext_ALIGN
    );
    check!(
        SparkResidentDecodeStageSliceNodeContext,
        SparkResidentDecodeStageSliceNodeContext_SIZE,
        SparkResidentDecodeStageSliceNodeContext_ALIGN
    );
}
