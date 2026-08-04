//! Service runtime — port of `api/service.c`.
//!
//! Sits in front of the [`ServingEngine`]: client sessions, the
//! client-request-id ↔ serving-handle mapping tables, event forwarding, and
//! the binary frame protocol (`SparkServiceFrameHeader` + frame bodies).
//!
//! Rust-port deviations (behavior otherwise faithful):
//! - Caller-provided session/map/event buffers become runtime-owned `Vec`s
//!   sized from configuration capacities; the runtime owns its engine.
//! - ABI/descriptor words and per-call runtime revalidation disappear;
//!   construction-time validation remains.
//! - Frame bodies are parsed from byte slices with explicit little-endian
//!   reads at the C struct offsets (the C casts native memory; on the
//!   little-endian targets this project supports the layouts coincide).

use super::engine::{
    ServingEngine, ServingEvent, ServingEventKind, ServingStats, SubmitTextRequest,
    SubmitTokenIdsRequest,
};
use super::status::ServingStatus;

/// `SPARK_SERVICE_ABI_VERSION`.
pub const SERVICE_ABI_VERSION: u32 = 1;
/// `SPARK_SERVICE_DEFAULT_PUMP_DISPATCH_STEPS`.
pub const DEFAULT_PUMP_DISPATCH_STEPS: u32 = 256;
/// `SPARK_SERVICE_DEFAULT_REQUEST_ID_BASE`.
pub const DEFAULT_REQUEST_ID_BASE: u64 = 5_000_000_000;
/// `SPARK_SERVICE_FRAME_MAGIC`.
pub const FRAME_MAGIC: u32 = 0x3550_4B53;
/// `SPARK_SERVICE_MAX_FRAME_BODY_BYTES`.
pub const MAX_FRAME_BODY_BYTES: u32 = 128 * 1024 * 1024;
/// `SPARK_SERVICE_MAX_TEXT_BYTES`.
pub const MAX_TEXT_BYTES: u32 = 64 * 1024 * 1024;
/// `SPARK_SERVICE_MAX_TOKEN_FRAME_COUNT`
/// (`SPARK_SCHEDULER_MAX_CONTEXT_TOKENS`).
pub const MAX_TOKEN_FRAME_COUNT: u32 = 1_048_576;
/// `SPARK_SERVICE_CLIENT_HASH_SLOTS`.
pub const CLIENT_HASH_SLOTS: usize = 1024;
/// `SPARK_SERVICE_REQUEST_MAP_HASH_SLOTS`.
pub const REQUEST_MAP_HASH_SLOTS: usize = 4096;
/// `SPARK_SERVICE_NO_HASH_SLOT`.
pub const NO_HASH_SLOT: u32 = u32::MAX;

/// `SPARK_SERVICE_CONFIGURATION_FLAG_AUTO_RELEASE_COMPLETED_MAPPINGS`.
pub const CONFIGURATION_FLAG_AUTO_RELEASE_COMPLETED_MAPPINGS: u32 = 0x0000_0001;
/// `SPARK_SERVICE_CONFIGURATION_FLAG_DRAIN_ENGINE_EVENTS_BEFORE_PUMP`.
pub const CONFIGURATION_FLAG_DRAIN_ENGINE_EVENTS_BEFORE_PUMP: u32 = 0x0000_0002;
/// `SPARK_SERVICE_CONFIGURATION_DEFAULT_FLAGS`.
pub const CONFIGURATION_DEFAULT_FLAGS: u32 = CONFIGURATION_FLAG_AUTO_RELEASE_COMPLETED_MAPPINGS
    | CONFIGURATION_FLAG_DRAIN_ENGINE_EVENTS_BEFORE_PUMP;
/// `SPARK_SERVICE_CONFIGURATION_KNOWN_FLAGS`.
pub const CONFIGURATION_KNOWN_FLAGS: u32 = CONFIGURATION_DEFAULT_FLAGS;

/// `SPARK_SERVICE_FRAME_KIND_SUBMIT_TEXT`.
pub const FRAME_KIND_SUBMIT_TEXT: u32 = 1;
/// `SPARK_SERVICE_FRAME_KIND_SUBMIT_TOKEN_IDS`.
pub const FRAME_KIND_SUBMIT_TOKEN_IDS: u32 = 2;
/// `SPARK_SERVICE_FRAME_KIND_CANCEL_REQUEST`.
pub const FRAME_KIND_CANCEL_REQUEST: u32 = 3;
/// `SPARK_SERVICE_FRAME_KIND_PING`.
pub const FRAME_KIND_PING: u32 = 4;
/// `SPARK_SERVICE_FRAME_KIND_EVENT`.
pub const FRAME_KIND_EVENT: u32 = 100;
/// `SPARK_SERVICE_FRAME_KIND_SUBMIT_ACK`.
pub const FRAME_KIND_SUBMIT_ACK: u32 = 101;
/// `SPARK_SERVICE_FRAME_KIND_ERROR`.
pub const FRAME_KIND_ERROR: u32 = 102;
/// `SPARK_SERVICE_FRAME_KIND_PONG`.
pub const FRAME_KIND_PONG: u32 = 103;

/// `SPARK_SERVICE_FRAME_FLAG_REALTIME`.
pub const FRAME_FLAG_REALTIME: u32 = super::engine::SUBMIT_FLAG_REALTIME;
/// `SPARK_SERVICE_FRAME_FLAG_DISABLE_SPECULATION`.
pub const FRAME_FLAG_DISABLE_SPECULATION: u32 = super::engine::SUBMIT_FLAG_DISABLE_SPECULATION;
/// `SPARK_SERVICE_FRAME_KNOWN_SUBMIT_FLAGS`.
pub const FRAME_KNOWN_SUBMIT_FLAGS: u32 = super::engine::SUBMIT_KNOWN_FLAGS;

/// On-wire size of `SparkServiceFrameHeader`.
pub const FRAME_HEADER_BYTES: usize = 40;
/// On-wire size of `SparkServiceSubmitTextFrameBody`.
pub const FRAME_SUBMIT_TEXT_BODY_BYTES: usize = 40;
/// On-wire size of `SparkServiceSubmitTokenIdsFrameBody`.
pub const FRAME_SUBMIT_TOKENS_BODY_BYTES: usize = 40;
/// On-wire size of `SparkServiceCancelFrameBody`.
pub const FRAME_CANCEL_BODY_BYTES: usize = 24;
/// On-wire size of `SparkServiceEvent` (frame body for event frames).
pub const SERVICE_EVENT_FRAME_BYTES: usize = 88;

/// `SparkServiceClientId`.
pub type ServiceClientId = u64;
/// `SparkServiceRequestId`.
pub type ServiceRequestId = u64;

/// Client session state (`SPARK_SERVICE_CLIENT_STATE_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClientState {
    /// `SPARK_SERVICE_CLIENT_STATE_FREE`.
    #[default]
    Free,
    /// `SPARK_SERVICE_CLIENT_STATE_CONNECTED`.
    Connected,
    /// `SPARK_SERVICE_CLIENT_STATE_DRAINING`.
    Draining,
}

/// Request mapping state (`SPARK_SERVICE_REQUEST_STATE_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum RequestMapState {
    #[default]
    Free,
    Live,
    Completed,
    Cancelled,
}

/// `SparkServiceClientSession`.
#[derive(Debug, Clone, Default)]
struct ClientSession {
    state: ClientState,
    /// Carried for C struct-layout fidelity; unread upstream as well.
    #[allow(dead_code)]
    flags: u32,
    client_id: ServiceClientId,
    user_cookie: u64,
    accepted_request_count: u64,
    completed_request_count: u64,
    client_hash_next: u32,
}

impl ClientSession {
    /// `SparkServiceInitializeClientSession`.
    fn reset(&mut self) {
        *self = ClientSession { client_hash_next: NO_HASH_SLOT, ..ClientSession::default() };
    }
}

/// `SparkServiceRequestMap`.
#[derive(Debug, Clone, Default)]
struct RequestMap {
    state: RequestMapState,
    /// Carried for C struct-layout fidelity; unread upstream as well.
    #[allow(dead_code)]
    flags: u32,
    client_id: ServiceClientId,
    client_request_id: ServiceRequestId,
    serving_request_id: u64,
    sequence_id: u64,
    serving_request_handle: u64,
    client_request_hash_next: u32,
    serving_handle_hash_next: u32,
}

impl RequestMap {
    /// `SparkServiceInitializeRequestMap`.
    fn reset(&mut self) {
        *self = RequestMap {
            client_request_hash_next: NO_HASH_SLOT,
            serving_handle_hash_next: NO_HASH_SLOT,
            ..RequestMap::default()
        };
    }
}

/// Service event kind (`SPARK_SERVICE_EVENT_KIND_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum ServiceEventKind {
    /// `SPARK_SERVICE_EVENT_KIND_NONE`.
    #[default]
    None = 0,
    /// `SPARK_SERVICE_EVENT_KIND_REQUEST_ACCEPTED`.
    RequestAccepted = 1,
    /// `SPARK_SERVICE_EVENT_KIND_PREFILL_PROGRESS`.
    PrefillProgress = 2,
    /// `SPARK_SERVICE_EVENT_KIND_TOKEN`.
    Token = 3,
    /// `SPARK_SERVICE_EVENT_KIND_REQUEST_COMPLETED`.
    RequestCompleted = 4,
    /// `SPARK_SERVICE_EVENT_KIND_REQUEST_CANCELLED`.
    RequestCancelled = 5,
    /// `SPARK_SERVICE_EVENT_KIND_ERROR`.
    Error = 6,
    /// `SPARK_SERVICE_EVENT_KIND_BACKPRESSURE`.
    Backpressure = 7,
    /// `SPARK_SERVICE_EVENT_KIND_CLIENT_CONNECTED`.
    ClientConnected = 8,
    /// `SPARK_SERVICE_EVENT_KIND_CLIENT_DISCONNECTED`.
    ClientDisconnected = 9,
    /// `SPARK_SERVICE_EVENT_KIND_STATS`.
    Stats = 10,
}

/// Service event (`SparkServiceEvent`): a serving event plus the
/// client-side routing identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ServiceEvent {
    /// Event kind.
    pub kind: ServiceEventKind,
    /// Event flags (forwarded from the serving event).
    pub flags: u32,
    /// Status attached to the event.
    pub status: ServingStatus,
    /// Token id for `Token` events.
    pub token_id: u32,
    /// Absolute token index within the request's sequence.
    pub token_index: u32,
    /// Tokens in the decode step that produced this event.
    pub token_count: u32,
    /// Prefill progress: first prompt token covered.
    pub prompt_token_offset: u32,
    /// Prompt tokens on the request (or covered by a prefill step).
    pub prompt_token_count: u32,
    /// Dispatch kind value that produced this event (`dispatch_kind` in C;
    /// serialized as the `SPARK_REQUEST_API_DISPATCH_KIND_*` number).
    pub dispatch_kind: u32,
    /// Dispatch flags that produced this event.
    pub dispatch_flags: u32,
    /// Client this event routes to.
    pub client_id: ServiceClientId,
    /// Client-scoped request id.
    pub client_request_id: ServiceRequestId,
    /// Serving-side request id.
    pub serving_request_id: u64,
    /// Assigned sequence id.
    pub sequence_id: u64,
    /// Serving-side request handle.
    pub serving_request_handle: u64,
}

/// `SparkServiceSubmitTextRequest`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ServiceSubmitTextRequest<'a> {
    /// `SPARK_SERVICE_FRAME_FLAG_*` submit bits.
    pub flags: u32,
    /// Scheduling priority.
    pub priority: u32,
    /// Thinking-phase token budget (0 = engine default).
    pub thinking_token_budget: u32,
    /// Output token budget (0 = engine default).
    pub output_token_budget: u32,
    /// Maximum prefill tokens per step (0 = engine default).
    pub max_prefill_tokens_per_step: u32,
    /// `SPARK_TOKENIZER_ENCODE_FLAG_*` bits.
    pub tokenizer_encode_flags: u32,
    /// Submitting client.
    pub client_id: ServiceClientId,
    /// Client-scoped request id (must be non-zero and unique per client).
    pub client_request_id: ServiceRequestId,
    /// Caller sequence id (0 = request API assigns one).
    pub sequence_id: u64,
    /// UTF-8 prompt text (`text_bytes` in C is this slice's length).
    pub text: &'a [u8],
}

/// `SparkServiceSubmitTokenIdsRequest`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ServiceSubmitTokenIdsRequest<'a> {
    /// `SPARK_SERVICE_FRAME_FLAG_*` submit bits.
    pub flags: u32,
    /// Scheduling priority.
    pub priority: u32,
    /// Thinking-phase token budget (0 = engine default).
    pub thinking_token_budget: u32,
    /// Output token budget (0 = engine default).
    pub output_token_budget: u32,
    /// Maximum prefill tokens per step (0 = engine default).
    pub max_prefill_tokens_per_step: u32,
    /// Submitting client.
    pub client_id: ServiceClientId,
    /// Client-scoped request id (must be non-zero and unique per client).
    pub client_request_id: ServiceRequestId,
    /// Caller sequence id (0 = request API assigns one).
    pub sequence_id: u64,
    /// Prompt token ids (`token_count` in C is this slice's length).
    pub token_ids: &'a [u32],
}

/// Submit result (`SparkServiceSubmitResult`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ServiceSubmitResult {
    /// Prompt tokens accepted.
    pub prompt_token_count: u32,
    /// Token capacity the request requires.
    pub required_token_capacity: u32,
    /// Thinking budget after clamping.
    pub thinking_token_budget: u32,
    /// Output budget after clamping.
    pub output_token_budget: u32,
    /// Submitting client.
    pub client_id: ServiceClientId,
    /// Client-scoped request id.
    pub client_request_id: ServiceRequestId,
    /// Serving-side request id.
    pub serving_request_id: u64,
    /// Assigned sequence id.
    pub sequence_id: u64,
    /// Serving-side request handle.
    pub serving_request_handle: u64,
}

/// Service stats (`SparkServiceStats`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ServiceStats {
    /// Sessions not in `Free` state.
    pub connected_client_count: u32,
    /// Mappings in `Live` state.
    pub live_request_count: u32,
    /// Mappings in `Completed` state.
    pub completed_request_mapping_count: u32,
    /// Events currently in the ring.
    pub event_count: u32,
    /// Event ring capacity.
    pub event_capacity: u32,
    /// Events dropped because the ring was full.
    pub dropped_event_count: u32,
    /// Status of the last pump-level operation.
    pub last_status: ServingStatus,
    /// Submissions attempted.
    pub submitted_request_count: u64,
    /// Submissions accepted.
    pub accepted_request_count: u64,
    /// Serving events forwarded into the service ring.
    pub forwarded_event_count: u64,
    /// Engine pumps issued.
    pub engine_pump_count: u64,
    /// Engine stats snapshot.
    pub serving_stats: ServingStats,
}

/// Service configuration (`SparkServiceConfiguration`).
pub struct ServiceConfiguration {
    /// `SPARK_SERVICE_CONFIGURATION_FLAG_*` bits (0 = defaults).
    pub flags: u32,
    /// Default pump step bound (0 = [`DEFAULT_PUMP_DISPATCH_STEPS`]).
    pub default_pump_dispatch_steps: u32,
    /// Generated request-id base (0 = [`DEFAULT_REQUEST_ID_BASE`]).
    pub request_id_base: u64,
    /// The serving engine this service fronts.
    pub serving_engine: ServingEngine,
    /// Client session slots.
    pub client_session_capacity: u32,
    /// Request mapping slots.
    pub request_map_capacity: u32,
    /// Event ring slots.
    pub event_ring_capacity: u32,
}

/// Frame header (`SparkServiceFrameHeader`), 40 bytes on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameHeader {
    /// `SPARK_SERVICE_FRAME_KIND_*`.
    pub kind: u32,
    /// `SPARK_SERVICE_FRAME_FLAG_*` bits.
    pub flags: u32,
    /// Frame body size in bytes.
    pub body_bytes: u32,
    /// Sending client.
    pub client_id: ServiceClientId,
    /// Client-scoped request id.
    pub client_request_id: ServiceRequestId,
}

/// `SparkServiceInitializeFrameHeader`.
pub fn initialize_frame_header(frame_kind: u32) -> FrameHeader {
    FrameHeader { kind: frame_kind, ..FrameHeader::default() }
}

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("slice bounds checked"))
}

fn read_u64_le(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("slice bounds checked"))
}

fn write_u32_le(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64_le(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

/// Parses a wire-format frame header, checking the magic, ABI version, and
/// descriptor words the C checks in `SparkServiceValidateFrameHeader`.
pub fn parse_frame_header(bytes: &[u8]) -> Result<FrameHeader, ServingStatus> {
    if bytes.len() < FRAME_HEADER_BYTES
        || read_u32_le(bytes, 0) != FRAME_MAGIC
        || read_u32_le(bytes, 4) != SERVICE_ABI_VERSION
        || read_u32_le(bytes, 8) != FRAME_HEADER_BYTES as u32
    {
        return Err(ServingStatus::InvalidArgument);
    }
    Ok(FrameHeader {
        kind: read_u32_le(bytes, 12),
        flags: read_u32_le(bytes, 16),
        body_bytes: read_u32_le(bytes, 20),
        client_id: read_u64_le(bytes, 24),
        client_request_id: read_u64_le(bytes, 32),
    })
}

/// Serializes a frame header to its 40-byte wire format.
pub fn serialize_frame_header(frame_header: &FrameHeader) -> [u8; FRAME_HEADER_BYTES] {
    let mut bytes = [0u8; FRAME_HEADER_BYTES];
    write_u32_le(&mut bytes, 0, FRAME_MAGIC);
    write_u32_le(&mut bytes, 4, SERVICE_ABI_VERSION);
    write_u32_le(&mut bytes, 8, FRAME_HEADER_BYTES as u32);
    write_u32_le(&mut bytes, 12, frame_header.kind);
    write_u32_le(&mut bytes, 16, frame_header.flags);
    write_u32_le(&mut bytes, 20, frame_header.body_bytes);
    write_u64_le(&mut bytes, 24, frame_header.client_id);
    write_u64_le(&mut bytes, 32, frame_header.client_request_id);
    bytes
}

/// `SparkServiceValidateFrameHeader`.
pub fn validate_frame_header(
    frame_header: &FrameHeader,
    maximum_body_bytes: u32,
) -> Result<(), ServingStatus> {
    let body_limit =
        if maximum_body_bytes != 0 { maximum_body_bytes } else { MAX_FRAME_BODY_BYTES };
    if frame_header.body_bytes > body_limit {
        return Err(ServingStatus::CapacityExceeded);
    }
    match frame_header.kind {
        FRAME_KIND_SUBMIT_TEXT
        | FRAME_KIND_SUBMIT_TOKEN_IDS
        | FRAME_KIND_CANCEL_REQUEST
        | FRAME_KIND_PING
        | FRAME_KIND_EVENT
        | FRAME_KIND_SUBMIT_ACK
        | FRAME_KIND_ERROR
        | FRAME_KIND_PONG => Ok(()),
        _ => Err(ServingStatus::InvalidArgument),
    }
}

/// `SparkServiceNormalizeFlags`.
fn normalize_flags(flags: u32) -> u32 {
    if flags == 0 {
        CONFIGURATION_DEFAULT_FLAGS
    } else {
        flags
    }
}

/// `SparkServiceNormalizePumpDispatchSteps`.
fn normalize_pump_dispatch_steps(default_pump_dispatch_steps: u32) -> u32 {
    if default_pump_dispatch_steps == 0 {
        DEFAULT_PUMP_DISPATCH_STEPS
    } else {
        default_pump_dispatch_steps
    }
}

/// `SparkServiceHash64`.
fn hash64(value: u64, slot_count: usize) -> usize {
    let mut hash = value;
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    (hash % slot_count as u64) as usize
}

/// `SparkServiceHashClientRequest`.
fn hash_client_request(client_id: ServiceClientId, client_request_id: ServiceRequestId) -> usize {
    let hash = client_id
        ^ (client_request_id
            .wrapping_add(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(client_id << 6)
            .wrapping_add(client_id >> 2));
    hash64(hash, REQUEST_MAP_HASH_SLOTS)
}

/// `SparkServiceMapServingEventKind`.
fn map_serving_event_kind(serving_event_kind: ServingEventKind) -> ServiceEventKind {
    match serving_event_kind {
        ServingEventKind::RequestAccepted => ServiceEventKind::RequestAccepted,
        ServingEventKind::PrefillProgress => ServiceEventKind::PrefillProgress,
        ServingEventKind::Token => ServiceEventKind::Token,
        ServingEventKind::RequestCompleted => ServiceEventKind::RequestCompleted,
        ServingEventKind::RequestCancelled => ServiceEventKind::RequestCancelled,
        ServingEventKind::Error => ServiceEventKind::Error,
        ServingEventKind::Backpressure => ServiceEventKind::Backpressure,
        ServingEventKind::None => ServiceEventKind::Error,
    }
}

/// Numeric dispatch kind for the serialized event (C stores the raw
/// `SPARK_REQUEST_API_DISPATCH_KIND_*` value).
fn dispatch_kind_code(kind: spark_text::prompt_pipeline::DispatchKind) -> u32 {
    use spark_text::prompt_pipeline::DispatchKind;
    match kind {
        DispatchKind::None => 0,
        DispatchKind::Prefill => 1,
        DispatchKind::DecodeBatch => 2,
        DispatchKind::PrefillBatch => 3,
        DispatchKind::SpeculativeVerifyBatch => 4,
    }
}

/// Service runtime (`SparkServiceRuntime`).
pub struct ServiceRuntime {
    flags: u32,
    default_pump_dispatch_steps: u32,
    next_generated_request_id: u64,
    next_generated_client_id: u64,
    serving_engine: ServingEngine,
    client_sessions: Vec<ClientSession>,
    request_maps: Vec<RequestMap>,
    events: Vec<ServiceEvent>,
    event_read_index: usize,
    event_write_index: usize,
    event_count: u32,
    dropped_event_count: u32,
    client_hash_heads: Vec<u32>,
    client_request_hash_heads: Vec<u32>,
    serving_handle_hash_heads: Vec<u32>,
    stats: ServiceStats,
}

impl ServiceRuntime {
    /// `SparkServiceInitialize`.
    pub fn new(configuration: ServiceConfiguration) -> Result<Self, ServingStatus> {
        let flags = normalize_flags(configuration.flags);
        if (flags & !CONFIGURATION_KNOWN_FLAGS) != 0
            || configuration.client_session_capacity == 0
            || configuration.request_map_capacity == 0
            || configuration.event_ring_capacity == 0
        {
            return Err(ServingStatus::InvalidArgument);
        }

        let mut client_sessions =
            Vec::with_capacity(configuration.client_session_capacity as usize);
        client_sessions.resize_with(configuration.client_session_capacity as usize, || {
            let mut session = ClientSession::default();
            session.reset();
            session
        });
        let mut request_maps = Vec::with_capacity(configuration.request_map_capacity as usize);
        request_maps.resize_with(configuration.request_map_capacity as usize, || {
            let mut map = RequestMap::default();
            map.reset();
            map
        });

        let mut service = ServiceRuntime {
            flags,
            default_pump_dispatch_steps: normalize_pump_dispatch_steps(
                configuration.default_pump_dispatch_steps,
            ),
            next_generated_request_id: if configuration.request_id_base == 0 {
                DEFAULT_REQUEST_ID_BASE
            } else {
                configuration.request_id_base
            },
            next_generated_client_id: 1,
            serving_engine: configuration.serving_engine,
            client_sessions,
            request_maps,
            events: vec![ServiceEvent::default(); configuration.event_ring_capacity as usize],
            event_read_index: 0,
            event_write_index: 0,
            event_count: 0,
            dropped_event_count: 0,
            client_hash_heads: vec![NO_HASH_SLOT; CLIENT_HASH_SLOTS],
            client_request_hash_heads: vec![NO_HASH_SLOT; REQUEST_MAP_HASH_SLOTS],
            serving_handle_hash_heads: vec![NO_HASH_SLOT; REQUEST_MAP_HASH_SLOTS],
            stats: ServiceStats::default(),
        };
        service.refresh_stats();
        Ok(service)
    }

    /// The fronted serving engine.
    pub fn serving_engine(&self) -> &ServingEngine {
        &self.serving_engine
    }

    /// The fronted serving engine, mutably.
    pub fn serving_engine_mut(&mut self) -> &mut ServingEngine {
        &mut self.serving_engine
    }

    /// `SparkServiceEventRingFreeCount`.
    fn event_ring_free_count(&self) -> u32 {
        let capacity = self.events.len() as u32;
        capacity.saturating_sub(self.event_count)
    }

    /// `SparkServicePushEvent`.
    fn push_event(&mut self, event: &ServiceEvent) -> Result<(), ServingStatus> {
        if self.event_count == self.events.len() as u32 {
            self.dropped_event_count += 1;
            return Err(ServingStatus::CapacityExceeded);
        }
        self.events[self.event_write_index] = *event;
        self.event_write_index += 1;
        if self.event_write_index == self.events.len() {
            self.event_write_index = 0;
        }
        self.event_count += 1;
        Ok(())
    }

    /// `SparkServicePushClientEvent`.
    fn push_client_event(
        &mut self,
        client_id: ServiceClientId,
        event_kind: ServiceEventKind,
        status: ServingStatus,
    ) -> Result<(), ServingStatus> {
        let event = ServiceEvent { kind: event_kind, status, client_id, ..ServiceEvent::default() };
        self.push_event(&event)
    }

    /// `SparkServiceInsertClientHash`.
    fn insert_client_hash(&mut self, client_index: usize) {
        let client_id = self.client_sessions[client_index].client_id;
        if client_id == 0 {
            return;
        }
        let hash_slot = hash64(client_id, CLIENT_HASH_SLOTS);
        self.client_sessions[client_index].client_hash_next = self.client_hash_heads[hash_slot];
        self.client_hash_heads[hash_slot] = client_index as u32;
    }

    /// `SparkServiceRemoveClientHash`.
    fn remove_client_hash(&mut self, client_index: usize) {
        let client_id = self.client_sessions[client_index].client_id;
        if client_id == 0 {
            return;
        }
        let hash_slot = hash64(client_id, CLIENT_HASH_SLOTS);
        let mut current_index = self.client_hash_heads[hash_slot];
        let mut previous_index = NO_HASH_SLOT;
        while current_index != NO_HASH_SLOT {
            if current_index as usize == client_index {
                if previous_index == NO_HASH_SLOT {
                    self.client_hash_heads[hash_slot] =
                        self.client_sessions[current_index as usize].client_hash_next;
                } else {
                    self.client_sessions[previous_index as usize].client_hash_next =
                        self.client_sessions[current_index as usize].client_hash_next;
                }
                self.client_sessions[current_index as usize].client_hash_next = NO_HASH_SLOT;
                return;
            }
            previous_index = current_index;
            current_index = self.client_sessions[current_index as usize].client_hash_next;
        }
    }

    /// `SparkServiceFindClient`.
    fn find_client(&self, client_id: ServiceClientId) -> Option<usize> {
        if client_id == 0 {
            return None;
        }
        let mut client_index = self.client_hash_heads[hash64(client_id, CLIENT_HASH_SLOTS)];
        while client_index != NO_HASH_SLOT && (client_index as usize) < self.client_sessions.len() {
            let session = &self.client_sessions[client_index as usize];
            if session.state != ClientState::Free && session.client_id == client_id {
                return Some(client_index as usize);
            }
            client_index = session.client_hash_next;
        }
        None
    }

    /// `SparkServiceFindFreeClient`.
    fn find_free_client(&self) -> Option<usize> {
        self.client_sessions.iter().position(|session| session.state == ClientState::Free)
    }

    /// `SparkServiceFindFreeRequestMap`.
    fn find_free_request_map(&self) -> Option<usize> {
        self.request_maps.iter().position(|map| map.state == RequestMapState::Free)
    }

    /// `SparkServiceInsertRequestMapHash`.
    fn insert_request_map_hash(&mut self, request_index: usize) {
        let map = &self.request_maps[request_index];
        let client_request_slot = hash_client_request(map.client_id, map.client_request_id);
        let serving_handle_slot = hash64(map.serving_request_handle, REQUEST_MAP_HASH_SLOTS);
        let map = &mut self.request_maps[request_index];
        map.client_request_hash_next = self.client_request_hash_heads[client_request_slot];
        map.serving_handle_hash_next = self.serving_handle_hash_heads[serving_handle_slot];
        self.client_request_hash_heads[client_request_slot] = request_index as u32;
        self.serving_handle_hash_heads[serving_handle_slot] = request_index as u32;
    }

    /// `SparkServiceRemoveRequestMapHash`.
    fn remove_request_map_hash(&mut self, request_index: usize) {
        let (client_id, client_request_id, serving_request_handle) = {
            let map = &self.request_maps[request_index];
            (map.client_id, map.client_request_id, map.serving_request_handle)
        };
        let mut current_index =
            self.client_request_hash_heads[hash_client_request(client_id, client_request_id)];
        let mut previous_index = NO_HASH_SLOT;
        while current_index != NO_HASH_SLOT {
            if current_index as usize == request_index {
                if previous_index == NO_HASH_SLOT {
                    self.client_request_hash_heads
                        [hash_client_request(client_id, client_request_id)] =
                        self.request_maps[current_index as usize].client_request_hash_next;
                } else {
                    self.request_maps[previous_index as usize].client_request_hash_next =
                        self.request_maps[current_index as usize].client_request_hash_next;
                }
                break;
            }
            previous_index = current_index;
            current_index = self.request_maps[current_index as usize].client_request_hash_next;
        }
        let serving_slot = hash64(serving_request_handle, REQUEST_MAP_HASH_SLOTS);
        let mut current_index = self.serving_handle_hash_heads[serving_slot];
        let mut previous_index = NO_HASH_SLOT;
        while current_index != NO_HASH_SLOT {
            if current_index as usize == request_index {
                if previous_index == NO_HASH_SLOT {
                    self.serving_handle_hash_heads[serving_slot] =
                        self.request_maps[current_index as usize].serving_handle_hash_next;
                } else {
                    self.request_maps[previous_index as usize].serving_handle_hash_next =
                        self.request_maps[current_index as usize].serving_handle_hash_next;
                }
                break;
            }
            previous_index = current_index;
            current_index = self.request_maps[current_index as usize].serving_handle_hash_next;
        }
        let map = &mut self.request_maps[request_index];
        map.client_request_hash_next = NO_HASH_SLOT;
        map.serving_handle_hash_next = NO_HASH_SLOT;
    }

    /// `SparkServiceFindRequestMapByClientRequest`.
    fn find_request_map_by_client_request(
        &self,
        client_id: ServiceClientId,
        client_request_id: ServiceRequestId,
    ) -> Option<usize> {
        if client_id == 0 || client_request_id == 0 {
            return None;
        }
        let mut request_index =
            self.client_request_hash_heads[hash_client_request(client_id, client_request_id)];
        while request_index != NO_HASH_SLOT && (request_index as usize) < self.request_maps.len() {
            let map = &self.request_maps[request_index as usize];
            if map.state != RequestMapState::Free
                && map.client_id == client_id
                && map.client_request_id == client_request_id
            {
                return Some(request_index as usize);
            }
            request_index = map.client_request_hash_next;
        }
        None
    }

    /// `SparkServiceFindRequestMapByServingHandle`.
    fn find_request_map_by_serving_handle(&self, serving_request_handle: u64) -> Option<usize> {
        if serving_request_handle == 0 {
            return None;
        }
        let mut request_index =
            self.serving_handle_hash_heads[hash64(serving_request_handle, REQUEST_MAP_HASH_SLOTS)];
        while request_index != NO_HASH_SLOT && (request_index as usize) < self.request_maps.len() {
            let map = &self.request_maps[request_index as usize];
            if map.state != RequestMapState::Free
                && map.serving_request_handle == serving_request_handle
            {
                return Some(request_index as usize);
            }
            request_index = map.serving_handle_hash_next;
        }
        None
    }

    /// `SparkServiceRefreshStats`.
    fn refresh_stats(&mut self) {
        self.stats.connected_client_count = self
            .client_sessions
            .iter()
            .filter(|session| session.state != ClientState::Free)
            .count() as u32;
        self.stats.live_request_count =
            self.request_maps.iter().filter(|map| map.state == RequestMapState::Live).count()
                as u32;
        self.stats.completed_request_mapping_count =
            self.request_maps.iter().filter(|map| map.state == RequestMapState::Completed).count()
                as u32;
        self.stats.event_count = self.event_count;
        self.stats.event_capacity = self.events.len() as u32;
        self.stats.dropped_event_count = self.dropped_event_count;
        self.stats.serving_stats = self.serving_engine.get_stats();
    }

    /// `SparkServiceRegisterClient`.
    pub fn register_client(&mut self, user_cookie: u64) -> Result<ServiceClientId, ServingStatus> {
        let client_index = self.find_free_client().ok_or(ServingStatus::CapacityExceeded)?;
        let client_id = self.next_generated_client_id;
        self.next_generated_client_id += 1;
        if self.next_generated_client_id == 0 {
            self.next_generated_client_id = 1;
        }
        {
            let session = &mut self.client_sessions[client_index];
            session.state = ClientState::Connected;
            session.client_id = client_id;
            session.user_cookie = user_cookie;
        }
        self.insert_client_hash(client_index);
        let status =
            self.push_client_event(client_id, ServiceEventKind::ClientConnected, ServingStatus::Ok);
        self.refresh_stats();
        status?;
        Ok(client_id)
    }

    /// `SparkServiceDisconnectClient`.
    pub fn disconnect_client(&mut self, client_id: ServiceClientId) -> Result<(), ServingStatus> {
        let client_index = self.find_client(client_id).ok_or(ServingStatus::NotFound)?;
        for request_index in 0..self.request_maps.len() {
            if self.request_maps[request_index].state == RequestMapState::Live
                && self.request_maps[request_index].client_id == client_id
            {
                let _ = self
                    .serving_engine
                    .cancel_request(self.request_maps[request_index].serving_request_handle);
                self.request_maps[request_index].state = RequestMapState::Cancelled;
            }
        }
        self.remove_client_hash(client_index);
        self.client_sessions[client_index].reset();
        let status = self.push_client_event(
            client_id,
            ServiceEventKind::ClientDisconnected,
            ServingStatus::Ok,
        );
        self.refresh_stats();
        status
    }

    /// `SparkServiceReserveRequestMap`.
    fn reserve_request_map(
        &self,
        client_id: ServiceClientId,
        client_request_id: ServiceRequestId,
    ) -> Result<usize, ServingStatus> {
        if self.find_client(client_id).is_none() {
            return Err(ServingStatus::NotFound);
        }
        if client_request_id == 0 {
            return Err(ServingStatus::InvalidArgument);
        }
        if self.find_request_map_by_client_request(client_id, client_request_id).is_some() {
            return Err(ServingStatus::Duplicate);
        }
        self.find_free_request_map().ok_or(ServingStatus::CapacityExceeded)
    }

    /// `SparkServiceFillSubmitResult`.
    fn fill_submit_result(
        &self,
        request_index: usize,
        serving_result: &super::engine::SubmitResult,
    ) -> ServiceSubmitResult {
        let map = &self.request_maps[request_index];
        ServiceSubmitResult {
            prompt_token_count: serving_result.prompt_token_count,
            required_token_capacity: serving_result.required_token_capacity,
            thinking_token_budget: serving_result.thinking_token_budget,
            output_token_budget: serving_result.output_token_budget,
            client_id: map.client_id,
            client_request_id: map.client_request_id,
            serving_request_id: map.serving_request_id,
            sequence_id: map.sequence_id,
            serving_request_handle: map.serving_request_handle,
        }
    }

    /// Shared tail of `SparkServiceSubmitTokenIds` / `SparkServiceSubmitText`.
    fn finish_submit(
        &mut self,
        request_index: usize,
        client_id: ServiceClientId,
        client_request_id: ServiceRequestId,
        serving_result: &super::engine::SubmitResult,
    ) -> ServiceSubmitResult {
        {
            let map = &mut self.request_maps[request_index];
            map.state = RequestMapState::Live;
            map.client_id = client_id;
            map.client_request_id = client_request_id;
            map.serving_request_id = serving_result.request_id;
            map.sequence_id = serving_result.sequence_id;
            map.serving_request_handle = serving_result.request_handle;
        }
        self.insert_request_map_hash(request_index);
        if let Some(client_index) = self.find_client(client_id) {
            self.client_sessions[client_index].accepted_request_count += 1;
        }
        self.stats.submitted_request_count += 1;
        self.stats.accepted_request_count += 1;
        let result = self.fill_submit_result(request_index, serving_result);
        self.refresh_stats();
        result
    }

    /// `SparkServiceSubmitTokenIds`.
    pub fn submit_token_ids(
        &mut self,
        request: &ServiceSubmitTokenIdsRequest,
    ) -> Result<ServiceSubmitResult, ServingStatus> {
        if (request.flags & !FRAME_KNOWN_SUBMIT_FLAGS) != 0 || request.token_ids.is_empty() {
            return Err(ServingStatus::InvalidArgument);
        }
        let request_index =
            self.reserve_request_map(request.client_id, request.client_request_id)?;

        let request_id = self.next_generated_request_id;
        self.next_generated_request_id += 1;
        let serving_request = SubmitTokenIdsRequest {
            flags: request.flags,
            priority: request.priority,
            thinking_token_budget: request.thinking_token_budget,
            output_token_budget: request.output_token_budget,
            max_prefill_tokens_per_step: request.max_prefill_tokens_per_step,
            request_id,
            sequence_id: request.sequence_id,
            token_ids: request.token_ids,
        };
        let serving_result = self.serving_engine.submit_token_ids(&serving_request)?;
        Ok(self.finish_submit(
            request_index,
            request.client_id,
            request.client_request_id,
            &serving_result,
        ))
    }

    /// `SparkServiceSubmitText`.
    pub fn submit_text(
        &mut self,
        request: &ServiceSubmitTextRequest,
    ) -> Result<ServiceSubmitResult, ServingStatus> {
        if (request.flags & !FRAME_KNOWN_SUBMIT_FLAGS) != 0
            || request.text.is_empty()
            || request.text.len() > MAX_TEXT_BYTES as usize
        {
            return Err(ServingStatus::InvalidArgument);
        }
        let request_index =
            self.reserve_request_map(request.client_id, request.client_request_id)?;

        let request_id = self.next_generated_request_id;
        self.next_generated_request_id += 1;
        let serving_request = SubmitTextRequest {
            flags: request.flags,
            priority: request.priority,
            thinking_token_budget: request.thinking_token_budget,
            output_token_budget: request.output_token_budget,
            max_prefill_tokens_per_step: request.max_prefill_tokens_per_step,
            tokenizer_encode_flags: request.tokenizer_encode_flags,
            request_id,
            sequence_id: request.sequence_id,
            text: request.text,
        };
        let serving_result = self
            .serving_engine
            .submit_text(&serving_request)
            .map_err(|failure| failure.status())?;
        Ok(self.finish_submit(
            request_index,
            request.client_id,
            request.client_request_id,
            &serving_result,
        ))
    }

    /// `SparkServiceHandleSubmitTokenIdsFrame`.
    pub fn handle_submit_token_ids_frame(
        &mut self,
        client_id: ServiceClientId,
        frame_header: &FrameHeader,
        body: &[u8],
    ) -> Result<ServiceSubmitResult, ServingStatus> {
        validate_frame_header(frame_header, MAX_FRAME_BODY_BYTES)?;
        if frame_header.kind != FRAME_KIND_SUBMIT_TOKEN_IDS
            || body.len() < FRAME_SUBMIT_TOKENS_BODY_BYTES
        {
            return Err(ServingStatus::InvalidArgument);
        }
        if read_u32_le(body, 0) != SERVICE_ABI_VERSION
            || read_u32_le(body, 4) != FRAME_SUBMIT_TOKENS_BODY_BYTES as u32
        {
            return Err(ServingStatus::InvalidArgument);
        }
        let token_count = read_u32_le(body, 24);
        if token_count == 0 || token_count > MAX_TOKEN_FRAME_COUNT {
            return Err(ServingStatus::InvalidArgument);
        }
        if token_count as u64 > (u32::MAX as u64 - FRAME_SUBMIT_TOKENS_BODY_BYTES as u64) / 4 {
            return Err(ServingStatus::CapacityExceeded);
        }
        let required_body_bytes = FRAME_SUBMIT_TOKENS_BODY_BYTES + token_count as usize * 4;
        if body.len() != required_body_bytes || frame_header.body_bytes as usize != body.len() {
            return Err(ServingStatus::InvalidArgument);
        }
        let token_ids: Vec<u32> = (0..token_count as usize)
            .map(|index| read_u32_le(body, FRAME_SUBMIT_TOKENS_BODY_BYTES + index * 4))
            .collect();

        let request = ServiceSubmitTokenIdsRequest {
            flags: frame_header.flags,
            priority: read_u32_le(body, 8),
            thinking_token_budget: read_u32_le(body, 12),
            output_token_budget: read_u32_le(body, 16),
            max_prefill_tokens_per_step: read_u32_le(body, 20),
            client_id,
            client_request_id: frame_header.client_request_id,
            sequence_id: read_u64_le(body, 32),
            token_ids: &token_ids,
        };
        self.submit_token_ids(&request)
    }

    /// `SparkServiceHandleSubmitTextFrame`.
    pub fn handle_submit_text_frame(
        &mut self,
        client_id: ServiceClientId,
        frame_header: &FrameHeader,
        body: &[u8],
    ) -> Result<ServiceSubmitResult, ServingStatus> {
        validate_frame_header(frame_header, MAX_FRAME_BODY_BYTES)?;
        if frame_header.kind != FRAME_KIND_SUBMIT_TEXT || body.len() < FRAME_SUBMIT_TEXT_BODY_BYTES
        {
            return Err(ServingStatus::InvalidArgument);
        }
        if read_u32_le(body, 0) != SERVICE_ABI_VERSION
            || read_u32_le(body, 4) != FRAME_SUBMIT_TEXT_BODY_BYTES as u32
        {
            return Err(ServingStatus::InvalidArgument);
        }
        let text_bytes = read_u32_le(body, 28);
        if text_bytes == 0 || text_bytes > MAX_TEXT_BYTES {
            return Err(ServingStatus::InvalidArgument);
        }
        if text_bytes as u64 > u32::MAX as u64 - FRAME_SUBMIT_TEXT_BODY_BYTES as u64 {
            return Err(ServingStatus::CapacityExceeded);
        }
        let required_body_bytes = FRAME_SUBMIT_TEXT_BODY_BYTES + text_bytes as usize;
        if body.len() != required_body_bytes || frame_header.body_bytes as usize != body.len() {
            return Err(ServingStatus::InvalidArgument);
        }
        let text = &body[FRAME_SUBMIT_TEXT_BODY_BYTES..];

        let request = ServiceSubmitTextRequest {
            flags: frame_header.flags,
            priority: read_u32_le(body, 8),
            thinking_token_budget: read_u32_le(body, 12),
            output_token_budget: read_u32_le(body, 16),
            max_prefill_tokens_per_step: read_u32_le(body, 20),
            tokenizer_encode_flags: read_u32_le(body, 24),
            client_id,
            client_request_id: frame_header.client_request_id,
            sequence_id: read_u64_le(body, 32),
            text,
        };
        self.submit_text(&request)
    }

    /// `SparkServiceHandleCancelFrame`.
    pub fn handle_cancel_frame(
        &mut self,
        client_id: ServiceClientId,
        frame_header: &FrameHeader,
        body: &[u8],
    ) -> Result<(), ServingStatus> {
        validate_frame_header(frame_header, MAX_FRAME_BODY_BYTES)?;
        if frame_header.kind != FRAME_KIND_CANCEL_REQUEST
            || body.len() != FRAME_CANCEL_BODY_BYTES
            || frame_header.body_bytes as usize != body.len()
        {
            return Err(ServingStatus::InvalidArgument);
        }
        if read_u32_le(body, 0) != SERVICE_ABI_VERSION
            || read_u32_le(body, 4) != FRAME_CANCEL_BODY_BYTES as u32
        {
            return Err(ServingStatus::InvalidArgument);
        }
        let body_client_request_id = read_u64_le(body, 16);
        let client_request_id = if body_client_request_id != 0 {
            body_client_request_id
        } else {
            frame_header.client_request_id
        };
        if client_request_id == 0 {
            return Err(ServingStatus::InvalidArgument);
        }
        self.cancel_request(client_id, client_request_id)
    }

    /// `SparkServiceCancelRequest`.
    pub fn cancel_request(
        &mut self,
        client_id: ServiceClientId,
        client_request_id: ServiceRequestId,
    ) -> Result<(), ServingStatus> {
        let request_index = self
            .find_request_map_by_client_request(client_id, client_request_id)
            .ok_or(ServingStatus::NotFound)?;
        self.serving_engine
            .cancel_request(self.request_maps[request_index].serving_request_handle)?;
        self.request_maps[request_index].state = RequestMapState::Cancelled;
        self.refresh_stats();
        Ok(())
    }

    /// `SparkServiceForwardServingEvent`.
    fn forward_serving_event(&mut self, serving_event: &ServingEvent) -> Result<(), ServingStatus> {
        let request_index = self
            .find_request_map_by_serving_handle(serving_event.request_handle)
            .ok_or(ServingStatus::NotFound)?;

        let service_event = {
            let map = &self.request_maps[request_index];
            ServiceEvent {
                kind: map_serving_event_kind(serving_event.kind),
                flags: serving_event.flags,
                status: serving_event.status,
                token_id: serving_event.token_id,
                token_index: serving_event.token_index,
                token_count: serving_event.token_count,
                prompt_token_offset: serving_event.prompt_token_offset,
                prompt_token_count: serving_event.prompt_token_count,
                dispatch_kind: dispatch_kind_code(serving_event.dispatch_kind),
                dispatch_flags: serving_event.dispatch_flags,
                client_id: map.client_id,
                client_request_id: map.client_request_id,
                serving_request_id: map.serving_request_id,
                sequence_id: map.sequence_id,
                serving_request_handle: map.serving_request_handle,
            }
        };

        if service_event.kind == ServiceEventKind::RequestCompleted {
            self.request_maps[request_index].state = RequestMapState::Completed;
            let client_id = self.request_maps[request_index].client_id;
            if let Some(client_index) = self.find_client(client_id) {
                self.client_sessions[client_index].completed_request_count += 1;
            }
        } else if service_event.kind == ServiceEventKind::RequestCancelled {
            self.request_maps[request_index].state = RequestMapState::Cancelled;
        }

        self.push_event(&service_event)
    }

    /// `SparkServiceReleaseCompletedMappingsIfRequested`.
    fn release_completed_mappings_if_requested(&mut self) {
        if (self.flags & CONFIGURATION_FLAG_AUTO_RELEASE_COMPLETED_MAPPINGS) == 0 {
            return;
        }
        for request_index in 0..self.request_maps.len() {
            if matches!(
                self.request_maps[request_index].state,
                RequestMapState::Completed | RequestMapState::Cancelled
            ) {
                self.remove_request_map_hash(request_index);
                self.request_maps[request_index].reset();
            }
        }
    }

    /// `SparkServiceDrainServingEvents`.
    fn drain_serving_events(&mut self) -> Result<(), ServingStatus> {
        let mut final_status = ServingStatus::Ok;
        loop {
            if self.event_ring_free_count() == 0 {
                final_status = ServingStatus::Busy;
                break;
            }
            let serving_event = match self.serving_engine.pop_event() {
                Ok(serving_event) => serving_event,
                Err(ServingStatus::NotFound) => break,
                Err(status) => {
                    final_status = status;
                    break;
                }
            };
            match self.forward_serving_event(&serving_event) {
                Ok(()) => {
                    self.stats.forwarded_event_count += 1;
                }
                Err(ServingStatus::NotFound) => continue,
                Err(status) => {
                    final_status = status;
                    break;
                }
            }
        }
        self.release_completed_mappings_if_requested();
        if final_status.is_ok() {
            Ok(())
        } else {
            Err(final_status)
        }
    }

    /// `SparkServicePump`.
    ///
    /// As in C, the engine's `NotFound`/`Busy`/`Pending` pump outcomes are
    /// folded to `Ok(())` here; only hard failures propagate.
    pub fn pump(
        &mut self,
        max_dispatch_steps: u32,
        mut stats_out: Option<&mut ServiceStats>,
    ) -> Result<(), ServingStatus> {
        if (self.flags & CONFIGURATION_FLAG_DRAIN_ENGINE_EVENTS_BEFORE_PUMP) != 0 {
            match self.drain_serving_events() {
                Ok(()) | Err(ServingStatus::Busy) => {}
                Err(drain_status) => {
                    self.stats.last_status = drain_status;
                    self.refresh_stats();
                    if let Some(stats_out) = stats_out.as_deref_mut() {
                        *stats_out = self.stats.clone();
                    }
                    return Err(drain_status);
                }
            }
        }

        let dispatch_steps = if max_dispatch_steps != 0 {
            max_dispatch_steps
        } else {
            self.default_pump_dispatch_steps
        };
        let mut engine_stats = ServingStats::default();
        let mut status = self
            .serving_engine
            .pump(0, dispatch_steps, Some(&mut engine_stats))
            .err()
            .unwrap_or(ServingStatus::Ok);
        self.stats.engine_pump_count += 1;
        match self.drain_serving_events() {
            Ok(()) | Err(ServingStatus::Busy) => {}
            Err(drain_status) => {
                status = drain_status;
            }
        }
        if matches!(status, ServingStatus::NotFound | ServingStatus::Busy | ServingStatus::Pending)
        {
            status = ServingStatus::Ok;
        }
        self.stats.last_status = status;
        self.refresh_stats();
        if let Some(stats_out) = stats_out {
            *stats_out = self.stats.clone();
        }
        if status.is_ok() {
            Ok(())
        } else {
            Err(status)
        }
    }

    /// `SparkServicePopEvent`; `Err(NotFound)` when the ring is empty.
    pub fn pop_event(&mut self) -> Result<ServiceEvent, ServingStatus> {
        if self.event_count == 0 {
            return Err(ServingStatus::NotFound);
        }
        let event = self.events[self.event_read_index];
        self.events[self.event_read_index] = ServiceEvent::default();
        self.event_read_index += 1;
        if self.event_read_index == self.events.len() {
            self.event_read_index = 0;
        }
        self.event_count -= 1;
        self.refresh_stats();
        Ok(event)
    }

    /// `SparkServiceGetStats`.
    pub fn get_stats(&mut self) -> ServiceStats {
        self.refresh_stats();
        self.stats.clone()
    }
}

/// `SparkServiceBuildEventFrame`: builds the event frame header plus the
/// serialized event body ([`SERVICE_EVENT_FRAME_BYTES`] bytes).
///
/// Infallible: the C's null/ABI checks are unrepresentable for a
/// `ServiceEvent` value.
pub fn build_event_frame(event: &ServiceEvent) -> (FrameHeader, [u8; SERVICE_EVENT_FRAME_BYTES]) {
    let frame_header = FrameHeader {
        kind: FRAME_KIND_EVENT,
        body_bytes: SERVICE_EVENT_FRAME_BYTES as u32,
        client_id: event.client_id,
        client_request_id: event.client_request_id,
        ..FrameHeader::default()
    };
    let mut body = [0u8; SERVICE_EVENT_FRAME_BYTES];
    write_u32_le(&mut body, 0, SERVICE_ABI_VERSION);
    write_u32_le(&mut body, 4, SERVICE_EVENT_FRAME_BYTES as u32);
    write_u32_le(&mut body, 8, event.kind as u32);
    write_u32_le(&mut body, 12, event.flags);
    write_u32_le(&mut body, 16, event.status as u32);
    write_u32_le(&mut body, 20, event.token_id);
    write_u32_le(&mut body, 24, event.token_index);
    write_u32_le(&mut body, 28, event.token_count);
    write_u32_le(&mut body, 32, event.prompt_token_offset);
    write_u32_le(&mut body, 36, event.prompt_token_count);
    write_u32_le(&mut body, 40, event.dispatch_kind);
    write_u32_le(&mut body, 44, event.dispatch_flags);
    write_u64_le(&mut body, 48, event.client_id);
    write_u64_le(&mut body, 56, event.client_request_id);
    write_u64_le(&mut body, 64, event.serving_request_id);
    write_u64_le(&mut body, 72, event.sequence_id);
    write_u64_le(&mut body, 80, event.serving_request_handle);
    (frame_header, body)
}
