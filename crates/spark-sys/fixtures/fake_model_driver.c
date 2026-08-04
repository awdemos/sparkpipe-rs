/* Minimal valid model driver used by spark-sys integration tests.
 *
 * Mirrors the stub shape that runtime/pack/driver_compiler.c generates:
 * a single program with all-zero flags, static descriptor storage, and
 * no-op create/destroy/admit/snapshot implementations. Compiled into a
 * shared object by the test harness (tests/driver_load.rs).
 */

#include <string.h>

#include "sparkpipe/spark_model_driver.h"

static uint64_t g_fixture_create_count;
static uint64_t g_fixture_destroy_count;

static SparkStatus SparkFixtureSubmit(
    void *driver_instance,
    SparkModelDriverFrame *frame)
{
    (void)driver_instance;
    (void)frame;
    return SPARK_STATUS_UNSUPPORTED;
}

static SparkStatus SparkFixtureCreate(
    const SparkModelDriverCreateRequest *request,
    void **driver_instance)
{
    if (request == 0 || driver_instance == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    g_fixture_create_count += 1u;
    *driver_instance = &g_fixture_create_count;
    return SPARK_STATUS_OK;
}

static void SparkFixtureDestroy(void *driver_instance)
{
    if (driver_instance != 0)
    {
        g_fixture_destroy_count += 1u;
    }
}

static SparkStatus SparkFixtureAdmit(
    void *driver_instance,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision)
{
    (void)driver_instance;
    if (request == 0 || decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(decision, 0, sizeof(*decision));
    decision->descriptor_bytes = (uint32_t)sizeof(*decision);
    decision->accepted = 1u;
    decision->driver_dispatch_slot = 0u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkFixtureSnapshot(
    void *driver_instance,
    uint32_t program_id,
    SparkModelDriverRuntimeSnapshot *snapshot)
{
    (void)driver_instance;
    if (snapshot == 0 || program_id != 1u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->descriptor_bytes = (uint32_t)sizeof(*snapshot);
    snapshot->program_id = program_id;
    return SPARK_STATUS_OK;
}

static const SparkModelDriverProgramProfile SparkFixtureProfile = {
    .descriptor_bytes = (uint32_t)sizeof(SparkModelDriverProgramProfile),
    .profile_flags = 0u,
    .max_inflight = 1u,
    .max_active_slots = 8u,
    .max_new_tokens = 4u,
    .max_resident_sequences = 8u,
    .max_sequence_tokens = 4096u,
    .target_latency_ns = 0u,
    .validated_latency_ns = 0u,
    .resident_weight_bytes = 0u,
    .resident_kv_bytes = 0u,
    .static_workspace_bytes = 0u,
    .device_memcpy_bytes_per_submit_ceiling = 0u,
    .host_staging_bytes_per_submit_ceiling = 0u,
    .private_queue_count = 0u,
    .reserved = 0u,
};

static const SparkModelDriverProgramDescriptor SparkFixturePrograms[] = {
    {
        1u,                 /* program_id */
        0u,                 /* flags */
        1u,                 /* max_inflight */
        0u,                 /* reserved */
        "fixture_program",  /* name */
        &SparkFixtureProfile,
        SparkFixtureSubmit,
    },
};

static const SparkModelDriverDescriptor SparkFixtureDescriptor = {
    SPARK_MODEL_DRIVER_ABI_VERSION,
    (uint32_t)sizeof(SparkModelDriverDescriptor),
    "fixture-model",
    "rev-0001",
    "fixture_stage",
    "host.cpu",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    1u, /* program_count */
    1u, /* module_instance_count */
    SparkFixturePrograms,
};

static const SparkModelDriverInterface SparkFixtureInterface = {
    SPARK_MODEL_DRIVER_ABI_VERSION,
    (uint32_t)sizeof(SparkModelDriverInterface),
    &SparkFixtureDescriptor,
    SparkFixtureCreate,
    SparkFixtureDestroy,
    SparkFixtureAdmit,
    SparkFixtureSnapshot,
};

SPARK_MODEL_DRIVER_EXPORT const SparkModelDriverInterface *SparkModelDriverGetInterface(void)
{
    return &SparkFixtureInterface;
}
