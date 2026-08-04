//! HTTP/1.1 gateway + SSE surface.
//!
//! Wire-compatible port of `api/http_gateway.c` (request parsing, routing, JSON
//! shapes, SSE framing) with the raw poll-loop server in
//! `api/gateway/http_server.c` replaced by hyper + tokio.
//!
//! Deliberate deviations from the C server (documented upgrades, not wire
//! regressions):
//! - Keep-alive connections replace `Connection: close`.
//! - SSE responses use HTTP/1.1 chunked transfer-encoding instead of
//!   close-delimited bodies.
//! - The 504 reason phrase is hyper's canonical "Gateway Timeout" (the C server
//!   fell through to "Bad Request"); reason phrases carry no semantics.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::fmt::Write as _;
use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::{Frame, Incoming};
use hyper::header::HeaderValue;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinSet;

/// Boxed response body used by every gateway response.
pub type GatewayBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

// ---------------------------------------------------------------------------
// Constants ported from the C tree.
// ---------------------------------------------------------------------------

/// `SPARK_SERVICE_EVENT_KIND_*` (`include/sparkpipe/spark_service.h`).
pub mod event_kind {
    pub const NONE: u32 = 0;
    pub const REQUEST_ACCEPTED: u32 = 1;
    pub const PREFILL_PROGRESS: u32 = 2;
    pub const TOKEN: u32 = 3;
    pub const REQUEST_COMPLETED: u32 = 4;
    pub const REQUEST_CANCELLED: u32 = 5;
    pub const ERROR: u32 = 6;
    pub const BACKPRESSURE: u32 = 7;
    pub const CLIENT_CONNECTED: u32 = 8;
    pub const CLIENT_DISCONNECTED: u32 = 9;
    pub const STATS: u32 = 10;
}

/// `SPARK_SERVICE_BACKEND_CONFIGURATION_FLAG_DSPARK`.
pub const BACKEND_CONFIGURATION_FLAG_DSPARK: u32 = 0x0000_0001;
/// `SPARK_SERVICE_BACKEND_CONFIGURATION_FLAG_MTP`.
pub const BACKEND_CONFIGURATION_FLAG_MTP: u32 = 0x0000_0002;
/// `SPARK_REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH`.
pub const REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH: u32 = 0x0000_0001;

/// `SPARK_GATEWAY_NONSTREAM_REQUEST_BIT`: tags fire-and-forget submissions.
pub const NONSTREAM_REQUEST_BIT: u64 = 1u64 << 63;

/// `SPARK_SERVICE_MAX_TEXT_BYTES`: default JSON upload cap.
pub const DEFAULT_MAX_UPLOAD_BYTES: usize = 64 * 1024 * 1024;
/// `SPARK_GATEWAY_REQUEST_BYTES`: upload cap plus header slack.
pub const DEFAULT_MAX_REQUEST_BYTES: usize = DEFAULT_MAX_UPLOAD_BYTES + 1024 * 1024;

const STREAM_MEMBER: &[u8] = b"\"stream\"";
const STREAM_TRUE: &[u8] = b"true";

const BODY_UNAUTHORIZED: &str =
    "{\"error\":{\"type\":\"unauthorized\",\"message\":\"missing or invalid bearer token\"}}\n";
const BODY_NOT_FOUND: &str =
    "{\"error\":{\"type\":\"not_found\",\"message\":\"unknown endpoint\"}}\n";
const BODY_BAD_REQUEST: &str =
    "{\"error\":{\"type\":\"bad_request\",\"message\":\"invalid or oversized request\"}}\n";
const BODY_BACKEND_UNAVAILABLE_JSON: &str = "{\"error\":{\"type\":\"backend_unavailable\",\"message\":\"GLM52 RING backend is not attached\"}}\n";
const BODY_BACKEND_UNAVAILABLE_SSE: &str = "event: error\ndata: {\"error\":{\"type\":\"backend_unavailable\",\"message\":\"GLM52 RING backend is not attached\"}}\n\n";
const BODY_REQUEST_TIMEOUT_JSON: &str = "{\"error\":{\"type\":\"request_timeout\",\"message\":\"GLM52 RING request produced no terminal event before the stream poll budget\"}}\n";
const BODY_REQUEST_TIMEOUT_SSE: &str = "event: error\ndata: {\"error\":{\"type\":\"request_timeout\",\"message\":\"GLM52 RING request produced no terminal event before the stream poll budget\"}}\n\n";

/// Byte-exact port of the demo UI from `SparkHttpGatewayBuildDemoUi`.
const DEMO_UI_HTML: &str = r#"<!doctype html><html><head><meta charset="utf-8"><title>SparkPipe GLM 5.2</title><style>body{font-family:system-ui;margin:2rem;max-width:960px}textarea{width:100%;height:16rem}pre{white-space:pre-wrap;background:#111;color:#eee;padding:1rem;min-height:10rem}input,button,textarea{font:inherit}button{padding:.55rem 1rem}.row{margin:.75rem 0}.small{color:#555}</style></head><body><h1>SparkPipe GLM 5.2</h1><p>Public demo path: paste a prompt, attach text files, and stream the response from <code>/v1/chat/completions</code>.</p><div class="row"><input id="key" placeholder="Bearer token, if required" type="password" style="width:100%"></div><div class="row"><textarea id="prompt">Summarize the attached file and point out anything surprising.</textarea></div><div class="row"><input id="files" type="file" multiple></div><div class="row small">Text files are folded into the model prompt inside SparkPipe; clients do not manage context windows.</div><div class="row"><button id="run">Send</button></div><pre id="out"></pre><script>document.getElementById('run').onclick=async()=>{const out=document.getElementById('out');out.textContent='';const key=document.getElementById('key').value;const prompt=document.getElementById('prompt').value;const files=await Promise.all([...document.getElementById('files').files].map(async f=>({filename:f.name,content:await f.text()})));const body={model:'glm-5.2',stream:true,max_tokens:1024,messages:[{role:'user',content:prompt}],files};const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'content-type':'application/json','authorization':key?('Bearer '+key):''},body:JSON.stringify(body)});const reader=r.body.getReader();const dec=new TextDecoder();for(;;){const x=await reader.read();if(x.done)break;out.textContent+=dec.decode(x.value);}};}</script></body></html>"#;

// ---------------------------------------------------------------------------
// Route surface (`SparkHttpGatewayRoute`).
// ---------------------------------------------------------------------------

/// Route identifiers (`SPARK_HTTP_GATEWAY_ROUTE_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    None,
    DemoUi,
    Health,
    OpenAiChat,
    OpenAiCompletions,
    AnthropicMessages,
    CorsPreflight,
}

/// Port of `SparkHttpGatewayRoute`. Exact method + path string equality.
pub fn route_request(method: &Method, path: &str) -> Route {
    if method == Method::OPTIONS {
        return Route::CorsPreflight;
    }
    if method == Method::GET && path == "/" {
        return Route::DemoUi;
    }
    if method == Method::GET && path == "/health" {
        return Route::Health;
    }
    if method == Method::POST && path == "/v1/chat/completions" {
        return Route::OpenAiChat;
    }
    if method == Method::POST && path == "/v1/completions" {
        return Route::OpenAiCompletions;
    }
    if method == Method::POST && path == "/v1/messages" {
        return Route::AnthropicMessages;
    }
    Route::None
}

// ---------------------------------------------------------------------------
// Stream-flag scan (`SparkHttpGatewayBodyRequestsStream`).
// ---------------------------------------------------------------------------

/// Byte-faithful port of `SparkHttpGatewayBodyRequestsStream`: scans the raw
/// body for the literal `"stream"` member, optional whitespace, a colon,
/// optional whitespace, then the literal `true`. Deliberately not a JSON
/// parse — matches C behavior on malformed bodies.
pub fn body_requests_stream(body: &[u8]) -> bool {
    if body.len() < STREAM_MEMBER.len() + STREAM_TRUE.len() {
        return false;
    }
    let mut index = 0usize;
    while index + STREAM_MEMBER.len() <= body.len() {
        if &body[index..index + STREAM_MEMBER.len()] != STREAM_MEMBER {
            index += 1;
            continue;
        }
        let mut cursor = index + STREAM_MEMBER.len();
        while cursor < body.len() && matches!(body[cursor], b' ' | b'\t' | b'\r' | b'\n') {
            cursor += 1;
        }
        if cursor >= body.len() || body[cursor] != b':' {
            index += 1;
            continue;
        }
        cursor += 1;
        while cursor < body.len() && matches!(body[cursor], b' ' | b'\t' | b'\r' | b'\n') {
            cursor += 1;
        }
        if cursor + STREAM_TRUE.len() <= body.len()
            && &body[cursor..cursor + STREAM_TRUE.len()] == STREAM_TRUE
        {
            return true;
        }
        index += 1;
    }
    false
}

// ---------------------------------------------------------------------------
// Authorization (`SparkHttpGatewayAuthorizationMatches`).
// ---------------------------------------------------------------------------

/// Port of `SparkHttpGatewayAuthorizationMatches`. With no configured key (or
/// an empty key) every request passes; otherwise the `Authorization` header
/// value must byte-match `Bearer <key>`. Keys that would overflow the C
/// server's 512-byte buffer never match.
pub fn authorization_matches(authorization: Option<&[u8]>, api_key: Option<&str>) -> bool {
    let key = match api_key {
        None | Some("") => return true,
        Some(key) => key,
    };
    let expected = format!("Bearer {key}");
    if expected.len() >= 512 {
        return false;
    }
    matches!(authorization, Some(value) if value == expected.as_bytes())
}

// ---------------------------------------------------------------------------
// JSON escaping (`SparkGlm52HttpAppendJsonEscapedBytes`).
// ---------------------------------------------------------------------------

/// Byte-faithful port of the C JSON escaper: `"`, `\`, `\n`, `\r`, `\t` get
/// backslash escapes, other control bytes `< 0x20` become `\u00xx`, and every
/// other byte (including raw UTF-8) is passed through untouched.
///
/// Used for health-document fields, which are Rust strings and therefore
/// valid UTF-8; passthrough runs stay valid UTF-8 because escapes only split
/// at ASCII boundaries. SSE token text is escaped into a byte buffer instead
/// (see `build_service_event_frame`) so non-UTF-8 decoder output survives.
pub fn append_json_escaped(out: &mut String, bytes: &[u8]) {
    let mut run_start = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        let byte = bytes[index];
        let escape: Option<&str> = match byte {
            b'"' => Some("\\\""),
            b'\\' => Some("\\\\"),
            b'\n' => Some("\\n"),
            b'\r' => Some("\\r"),
            b'\t' => Some("\\t"),
            _ => None,
        };
        if let Some(escape) = escape {
            out.push_str(std::str::from_utf8(&bytes[run_start..index]).unwrap_or(""));
            out.push_str(escape);
            index += 1;
            run_start = index;
        } else if byte < 0x20 {
            out.push_str(std::str::from_utf8(&bytes[run_start..index]).unwrap_or(""));
            let _ = write!(out, "\\u{byte:04x}");
            index += 1;
            run_start = index;
        } else {
            index += 1;
        }
    }
    out.push_str(std::str::from_utf8(&bytes[run_start..]).unwrap_or(""));
}

// ---------------------------------------------------------------------------
// Backend-facing data types (boundary with `serving_engine.rs`).
// ---------------------------------------------------------------------------

/// One service event, mirrors `SparkServiceEvent` fields used on the wire.
#[derive(Debug, Clone, Default)]
pub struct ServiceEvent {
    pub kind: u32,
    pub status: u32,
    pub client_id: u64,
    pub client_request_id: u64,
    pub serving_request_id: u64,
    pub sequence_id: u64,
    pub token_id: u32,
    pub token_index: u32,
}

impl ServiceEvent {
    /// `SparkGlm52HttpServiceEventIsTerminal`.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.kind,
            event_kind::REQUEST_COMPLETED | event_kind::REQUEST_CANCELLED | event_kind::ERROR
        )
    }

    /// SSE `event:` field name for this kind.
    pub fn sse_event_name(&self) -> &'static str {
        if self.is_terminal() {
            "done"
        } else if self.kind == event_kind::TOKEN {
            "token"
        } else {
            "event"
        }
    }
}

/// Result of an accepted submission (`SparkServiceSubmitResult`).
#[derive(Debug, Clone, Copy, Default)]
pub struct SubmitResult {
    pub client_request_id: u64,
    pub serving_request_id: u64,
    pub sequence_id: u64,
    pub prompt_token_count: u32,
    pub output_token_budget: u32,
}

/// Which completion surface a submission arrived on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitRoute {
    OpenAiChat,
    OpenAiCompletions,
    AnthropicMessages,
}

/// One prompt submission handed to the backend.
#[derive(Debug, Clone)]
pub struct Submission {
    pub route: SubmitRoute,
    pub client_request_id: u64,
    pub body: Vec<u8>,
}

/// Backend identity/configuration view (`SparkServiceBackendView` subset that
/// reaches the wire). When `attached` is false the gateway behaves exactly
/// like the C server without a service backend: zeroed health view and 503 on
/// submissions.
#[derive(Debug, Clone, Default)]
pub struct BackendView {
    pub attached: bool,
    pub runtime_initialized: bool,
    pub local_control_ready: bool,
    pub release_id: String,
    pub release_git_commit: String,
    pub release_generation: u64,
    pub configured_kv_context_limit_tokens: u32,
    pub configured_max_active_sequences: u32,
    pub adaptive_decode_batch_width: u32,
    pub decode_batch_capacity: u32,
    pub prefill_wave_token_count: u32,
    pub transport_shared_object_path: String,
    pub transport_capability_flags: u32,
    pub request_api_configuration_flags: u32,
    pub speculation_configuration_flags: u32,
    pub first_blocker: String,
}

/// Serving counters nested inside [`ServiceStats`].
#[derive(Debug, Clone, Default)]
pub struct ServingStats {
    pub queued_request_count: u32,
    pub completed_stream_count: u64,
    pub prefill_dispatch_count: u64,
    pub prefill_batch_dispatch_count: u64,
    pub prefill_token_count: u64,
    pub decode_dispatch_count: u64,
    pub decoded_token_count: u64,
    pub maximum_prefill_active_sequence_count: u32,
    pub maximum_prefill_lane_count: u32,
    pub maximum_decode_active_sequence_count: u32,
    pub maximum_decode_lane_count: u32,
    pub jit_prefetch_dispatch_count: u64,
    pub jit_prefetch_block_count: u64,
    pub async_jit_prefetch_start_count: u64,
    pub async_jit_prefetch_completion_count: u64,
    pub mtp_draft_token_count: u64,
    pub mtp_verify_dispatch_count: u64,
    pub mtp_draft_ready_count: u64,
    pub mtp_accepted_draft_token_count: u64,
    pub mtp_committed_token_count: u64,
    pub mtp_rejected_token_count: u64,
}

/// Service-level counters (`SparkServiceStats` subset that reaches the wire).
#[derive(Debug, Clone, Default)]
pub struct ServiceStats {
    pub connected_client_count: u32,
    pub live_request_count: u32,
    pub event_count: u32,
    pub dropped_event_count: u32,
    pub serving: ServingStats,
}

/// Why a submission failed; mapped to the C status-code surface.
#[derive(Debug, thiserror::Error)]
pub enum SubmitError {
    /// `SPARK_STATUS_BUSY` → 504 request timeout payload.
    #[error("request produced no terminal event before the stream poll budget")]
    Busy,
    /// Any other submission failure → 503 backend unavailable payload.
    #[error("backend is not attached")]
    BackendUnavailable,
}

/// Boundary the serving engine implements (`src/serving_engine.rs`). The C
/// gateway calls into `SparkServiceRuntime` directly; here the surface is
/// narrowed to submit / poll events / cancel plus health snapshots.
pub trait GatewayBackend: Send + Sync + 'static {
    /// Snapshot for `/health` and the submission availability gate.
    fn backend_view(&self) -> BackendView;
    /// Counters for `/health`.
    fn service_stats(&self) -> ServiceStats;
    /// Submit a prompt body. Mirrors `SparkHttpGatewaySubmitJsonToService`.
    fn submit(&self, submission: &Submission) -> Result<SubmitResult, SubmitError>;
    /// Take the event channel for an accepted streaming submission. Events
    /// arrive in order; the gateway stops after a terminal event.
    fn event_stream(&self, client_request_id: u64) -> Option<mpsc::Receiver<ServiceEvent>>;
    /// Cancel a live request (`SparkServiceCancelRequest`) — client
    /// disconnect or gateway shutdown.
    fn cancel(&self, client_request_id: u64);
    /// Decode one token id for the SSE `text` field (tokenizer with
    /// `SKIP_SPECIAL_TOKENS`); empty output on failure, as in C.
    fn decode_token(&self, token_id: u32) -> Vec<u8>;
}

// ---------------------------------------------------------------------------
// Response builders (wire-compat, byte-exact with the C bodies).
// ---------------------------------------------------------------------------

/// Port of `SparkHttpGatewayBuildHealth` (two-flag observation ping). Kept for
/// parity with the C API; `/health` uses [`build_service_health_body`].
pub fn build_health_body(runtime_initialized: bool, local_control_ready: bool) -> String {
    format!(
        "{{\"schema\":\"sparkpipe.runtime_observation.v1\",\
\"runtime_initialized\":{},\"local_control_ready\":{},\
\"end_to_end_observation_status\":\"NOT_MEASURED\",\
\"accuracy_status\":\"NOT_MEASURED\",\
\"performance_status\":\"NOT_MEASURED\"}}\n",
        u32::from(runtime_initialized),
        u32::from(local_control_ready),
    )
}

/// Port of `SparkHttpGatewayBuildServiceHealth`: full observation document.
/// When `view.attached` is false the C server handed in zeroed stats/view;
/// replicate by clearing both.
pub fn build_service_health_body(stats: &ServiceStats, view: &BackendView) -> String {
    let empty_stats = ServiceStats::default();
    let empty_view = BackendView::default();
    let (stats, view) = if view.attached { (stats, view) } else { (&empty_stats, &empty_view) };

    let identity_status = if !view.release_id.is_empty()
        && !view.release_git_commit.is_empty()
        && view.release_generation != 0
    {
        "OBSERVED"
    } else {
        "NOT_MEASURED"
    };
    let end_to_end_status =
        if stats.serving.completed_stream_count != 0 && stats.serving.decoded_token_count != 0 {
            "OBSERVED"
        } else {
            "NOT_MEASURED"
        };
    let batching_status = if view.configured_max_active_sequences <= 1 {
        "NOT_WORKING"
    } else if stats.serving.maximum_decode_lane_count > 1
        || stats.serving.maximum_prefill_lane_count > 1
    {
        "OBSERVED"
    } else {
        "NOT_MEASURED"
    };
    let jit_kv_status = if (view.request_api_configuration_flags
        & REQUEST_API_CONFIGURATION_FLAG_JIT_KV_PREFETCH)
        == 0
    {
        "NOT_WORKING"
    } else if stats.serving.jit_prefetch_dispatch_count != 0 {
        "OBSERVED"
    } else {
        "NOT_MEASURED"
    };
    let mtp_status = if (view.speculation_configuration_flags & BACKEND_CONFIGURATION_FLAG_MTP) == 0
    {
        "NOT_WORKING"
    } else if stats.serving.mtp_verify_dispatch_count != 0 {
        "OBSERVED"
    } else {
        "NOT_MEASURED"
    };
    let dspark_status =
        if (view.speculation_configuration_flags & BACKEND_CONFIGURATION_FLAG_DSPARK) == 0 {
            "NOT_WORKING"
        } else {
            "NOT_MEASURED"
        };
    let transport_status =
        if view.transport_capability_flags != 0 { "OBSERVED" } else { "NOT_MEASURED" };

    let mut escaped_blocker = String::new();
    let mut escaped_commit = String::new();
    let mut escaped_release = String::new();
    let mut escaped_transport = String::new();
    append_json_escaped(&mut escaped_blocker, view.first_blocker.as_bytes());
    append_json_escaped(&mut escaped_commit, view.release_git_commit.as_bytes());
    append_json_escaped(&mut escaped_release, view.release_id.as_bytes());
    append_json_escaped(&mut escaped_transport, view.transport_shared_object_path.as_bytes());

    let serving = &stats.serving;
    format!(
        "{{\
\"schema\":\"sparkpipe.runtime_observation.v1\",\
\"release_identity_status\":\"{identity_status}\",\
\"release_id\":\"{escaped_release}\",\
\"release_git_commit\":\"{escaped_commit}\",\
\"release_generation\":{release_generation},\
\"runtime_initialized\":{runtime_initialized},\
\"local_control_ready\":{local_control_ready},\
\"configured_kv_context_limit_tokens\":{kv_limit},\
\"configured_max_active_sequences\":{max_active},\
\"adaptive_decode_batch_width\":{adaptive_width},\
\"decode_batch_capacity\":{decode_capacity},\
\"prefill_wave_token_count\":{prefill_wave},\
\"transport_shared_object\":\"{escaped_transport}\",\
\"bound_transport_interface_flags\":{transport_flags},\
\"transport_binding_status\":\"{transport_status}\",\
\"end_to_end_observation_status\":\"{end_to_end_status}\",\
\"accuracy_status\":\"NOT_MEASURED\",\
\"performance_status\":\"NOT_MEASURED\",\
\"multi_sequence_batching_status\":\"{batching_status}\",\
\"long_context_status\":\"NOT_MEASURED\",\
\"jit_kv_status\":\"{jit_kv_status}\",\
\"dspark_status\":\"{dspark_status}\",\
\"mtp_status\":\"{mtp_status}\",\
\"connected_clients\":{connected_clients},\
\"live_requests\":{live_requests},\
\"queued_requests\":{queued_requests},\
\"completed_streams\":{completed_streams},\
\"prefill_dispatches\":{prefill_dispatches},\
\"prefill_batch_dispatches\":{prefill_batch_dispatches},\
\"prefill_tokens\":{prefill_tokens},\
\"decode_dispatches\":{decode_dispatches},\
\"decoded_tokens\":{decoded_tokens},\
\"maximum_prefill_active_sequences\":{max_prefill_active},\
\"maximum_prefill_lanes\":{max_prefill_lanes},\
\"maximum_decode_active_sequences\":{max_decode_active},\
\"maximum_decode_lanes\":{max_decode_lanes},\
\"event_backlog\":{event_backlog},\
\"dropped_events\":{dropped_events},\
\"jit_prefetch_dispatches\":{jit_prefetch_dispatches},\
\"jit_prefetch_blocks\":{jit_prefetch_blocks},\
\"async_jit_prefetch_starts\":{async_jit_starts},\
\"async_jit_prefetch_completions\":{async_jit_completions},\
\"mtp_draft_tokens\":{mtp_draft_tokens},\
\"mtp_verify_dispatches\":{mtp_verify_dispatches},\
\"mtp_draft_ready\":{mtp_draft_ready},\
\"mtp_accepted_draft_tokens\":{mtp_accepted_draft_tokens},\
\"mtp_committed_tokens\":{mtp_committed_tokens},\
\"mtp_rejected_tokens\":{mtp_rejected_tokens},\
\"first_blocker\":\"{escaped_blocker}\"\
}}\n",
        release_generation = view.release_generation,
        runtime_initialized = u32::from(view.runtime_initialized),
        local_control_ready = u32::from(view.local_control_ready),
        kv_limit = view.configured_kv_context_limit_tokens,
        max_active = view.configured_max_active_sequences,
        adaptive_width = view.adaptive_decode_batch_width,
        decode_capacity = view.decode_batch_capacity,
        prefill_wave = view.prefill_wave_token_count,
        transport_flags = view.transport_capability_flags,
        connected_clients = stats.connected_client_count,
        live_requests = stats.live_request_count,
        queued_requests = serving.queued_request_count,
        completed_streams = serving.completed_stream_count,
        prefill_dispatches = serving.prefill_dispatch_count,
        prefill_batch_dispatches = serving.prefill_batch_dispatch_count,
        prefill_tokens = serving.prefill_token_count,
        decode_dispatches = serving.decode_dispatch_count,
        decoded_tokens = serving.decoded_token_count,
        max_prefill_active = serving.maximum_prefill_active_sequence_count,
        max_prefill_lanes = serving.maximum_prefill_lane_count,
        max_decode_active = serving.maximum_decode_active_sequence_count,
        max_decode_lanes = serving.maximum_decode_lane_count,
        event_backlog = stats.event_count,
        dropped_events = stats.dropped_event_count,
        jit_prefetch_dispatches = serving.jit_prefetch_dispatch_count,
        jit_prefetch_blocks = serving.jit_prefetch_block_count,
        async_jit_starts = serving.async_jit_prefetch_start_count,
        async_jit_completions = serving.async_jit_prefetch_completion_count,
        mtp_draft_tokens = serving.mtp_draft_token_count,
        mtp_verify_dispatches = serving.mtp_verify_dispatch_count,
        mtp_draft_ready = serving.mtp_draft_ready_count,
        mtp_accepted_draft_tokens = serving.mtp_accepted_draft_token_count,
        mtp_committed_tokens = serving.mtp_committed_token_count,
        mtp_rejected_tokens = serving.mtp_rejected_token_count,
    )
}

/// Port of `SparkHttpGatewayBuildSubmitAccepted`, non-stream variant.
pub fn build_submit_accepted_json(result: &SubmitResult) -> String {
    format!(
        "{{\"id\":\"spreq-{id}\",\"object\":\"sparkpipe.request\",\
\"client_request_id\":{client_request_id},\"serving_request_id\":{serving_request_id},\
\"sequence_id\":{sequence_id},\"prompt_tokens\":{prompt_tokens},\
\"output_token_budget\":{output_token_budget},\"status\":\"queued\"}}\n",
        id = result.serving_request_id,
        client_request_id = result.client_request_id,
        serving_request_id = result.serving_request_id,
        sequence_id = result.sequence_id,
        prompt_tokens = result.prompt_token_count,
        output_token_budget = result.output_token_budget,
    )
}

/// Port of `SparkHttpGatewayBuildSubmitAccepted`, stream variant (first SSE
/// frame of an accepted streaming request).
pub fn build_submit_accepted_sse_frame(result: &SubmitResult) -> Vec<u8> {
    format!(
        "event: accepted\ndata: {{\"client_request_id\":{client_request_id},\
\"serving_request_id\":{serving_request_id},\"sequence_id\":{sequence_id},\
\"prompt_tokens\":{prompt_tokens},\"output_token_budget\":{output_token_budget}}}\n\n",
        client_request_id = result.client_request_id,
        serving_request_id = result.serving_request_id,
        sequence_id = result.sequence_id,
        prompt_tokens = result.prompt_token_count,
        output_token_budget = result.output_token_budget,
    )
    .into_bytes()
}

/// Port of `SparkHttpGatewayBuildServiceEventStream`: one SSE frame for a
/// service event. `token_text` is the already-decoded token bytes (empty when
/// the kind is not TOKEN or decoding failed, as in C).
pub fn build_service_event_frame(event: &ServiceEvent, token_text: &[u8]) -> Vec<u8> {
    let mut frame = format!(
        "event: {name}\ndata: {{\"kind\":{kind},\"status\":{status},\"client_id\":{client_id},\
\"client_request_id\":{client_request_id},\"serving_request_id\":{serving_request_id},\
\"sequence_id\":{sequence_id},\"token_id\":{token_id},\"token_index\":{token_index},\
\"text\":\"",
        name = event.sse_event_name(),
        kind = event.kind,
        status = event.status,
        client_id = event.client_id,
        client_request_id = event.client_request_id,
        serving_request_id = event.serving_request_id,
        sequence_id = event.sequence_id,
        token_id = event.token_id,
        token_index = event.token_index,
    )
    .into_bytes();
    // Escape into a byte buffer (not String) so raw non-UTF-8 token bytes pass
    // through untouched, exactly like the C escaper.
    for &byte in token_text {
        match byte {
            b'"' => frame.extend_from_slice(b"\\\""),
            b'\\' => frame.extend_from_slice(b"\\\\"),
            b'\n' => frame.extend_from_slice(b"\\n"),
            b'\r' => frame.extend_from_slice(b"\\r"),
            b'\t' => frame.extend_from_slice(b"\\t"),
            byte if byte < 0x20 => {
                frame.extend_from_slice(format!("\\u{byte:04x}").as_bytes());
            }
            byte => frame.push(byte),
        }
    }
    frame.extend_from_slice(b"\"}\n\n");
    frame
}

// ---------------------------------------------------------------------------
// HTTP response assembly.
// ---------------------------------------------------------------------------

fn boxed_full(body: impl Into<Bytes>) -> GatewayBody {
    Full::new(body.into()).boxed()
}

/// Every response carries the C server's CORS + cache headers.
fn base_response(status: StatusCode, content_type: &'static str) -> Response<GatewayBody> {
    let mut response = Response::new(boxed_full(Bytes::new()));
    *response.status_mut() = status;
    let headers = response.headers_mut();
    headers.insert(hyper::header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    headers.insert(
        hyper::header::HeaderName::from_static("access-control-allow-origin"),
        HeaderValue::from_static("*"),
    );
    headers.insert(
        hyper::header::HeaderName::from_static("access-control-allow-headers"),
        HeaderValue::from_static("authorization, content-type"),
    );
    headers.insert(
        hyper::header::HeaderName::from_static("access-control-allow-methods"),
        HeaderValue::from_static("GET, POST, OPTIONS"),
    );
    headers.insert(hyper::header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response
}

fn full_response(
    status: StatusCode,
    content_type: &'static str,
    body: impl Into<Bytes>,
) -> Response<GatewayBody> {
    let mut response = base_response(status, content_type);
    *response.body_mut() = boxed_full(body);
    response
}

fn json_error(status: StatusCode, body: &'static str) -> Response<GatewayBody> {
    full_response(status, "application/json", body)
}

/// 503 / 504 payload selection by stream flag (`SparkHttpGatewayBuildBackendUnavailable`
/// / `SparkHttpGatewayBuildRequestTimeout`).
fn unavailable_or_timeout(timeout: bool, stream: bool) -> Response<GatewayBody> {
    match (timeout, stream) {
        (false, false) => {
            json_error(StatusCode::SERVICE_UNAVAILABLE, BODY_BACKEND_UNAVAILABLE_JSON)
        }
        (false, true) => sse_error(StatusCode::SERVICE_UNAVAILABLE, BODY_BACKEND_UNAVAILABLE_SSE),
        (true, false) => json_error(StatusCode::GATEWAY_TIMEOUT, BODY_REQUEST_TIMEOUT_JSON),
        (true, true) => sse_error(StatusCode::GATEWAY_TIMEOUT, BODY_REQUEST_TIMEOUT_SSE),
    }
}

/// Stream-flavored error: the whole SSE body is delivered as one frame and
/// the response ends (C kept the connection in the pending pool; the body
/// bytes are identical).
fn sse_error(status: StatusCode, body: &'static str) -> Response<GatewayBody> {
    let mut response = base_response(status, "text/event-stream");
    *response.body_mut() = boxed_full(Bytes::from_static(body.as_bytes()));
    response
}

// ---------------------------------------------------------------------------
// SSE streaming body.
// ---------------------------------------------------------------------------

/// Streaming body for an accepted SSE request: yields the `accepted` frame,
/// then one frame per backend event until a terminal event, channel closure,
/// or client disconnect (which cancels the request, mirroring
/// `SparkGatewayCancelPendingStream`).
pub struct SseBody<B: GatewayBackend> {
    backend: Arc<B>,
    client_request_id: u64,
    pending: VecDeque<Bytes>,
    receiver: Option<mpsc::Receiver<ServiceEvent>>,
    terminal_seen: bool,
}

impl<B: GatewayBackend> SseBody<B> {
    fn new(
        backend: Arc<B>,
        client_request_id: u64,
        accepted_frame: Bytes,
        receiver: mpsc::Receiver<ServiceEvent>,
    ) -> Self {
        let mut pending = VecDeque::new();
        pending.push_back(accepted_frame);
        Self { backend, client_request_id, pending, receiver: Some(receiver), terminal_seen: false }
    }
}

impl<B: GatewayBackend> hyper::body::Body for SseBody<B> {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        if let Some(bytes) = this.pending.pop_front() {
            return Poll::Ready(Some(Ok(Frame::data(bytes))));
        }
        if this.terminal_seen {
            return Poll::Ready(None);
        }
        let receiver = match this.receiver.as_mut() {
            Some(receiver) => receiver,
            None => return Poll::Ready(None),
        };
        loop {
            match receiver.poll_recv(cx) {
                Poll::Ready(Some(event)) => {
                    // `SparkGatewayDispatchServiceEvents` drops REQUEST_ACCEPTED.
                    if event.kind == event_kind::REQUEST_ACCEPTED {
                        continue;
                    }
                    let token_text = if event.kind == event_kind::TOKEN {
                        this.backend.decode_token(event.token_id)
                    } else {
                        Vec::new()
                    };
                    if event.is_terminal() {
                        this.terminal_seen = true;
                        this.receiver = None;
                    }
                    let frame = build_service_event_frame(&event, &token_text);
                    return Poll::Ready(Some(Ok(Frame::data(Bytes::from(frame)))));
                }
                Poll::Ready(None) => {
                    this.terminal_seen = true;
                    this.receiver = None;
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl<B: GatewayBackend> Drop for SseBody<B> {
    fn drop(&mut self) {
        if !self.terminal_seen {
            self.backend.cancel(self.client_request_id);
        }
    }
}

// ---------------------------------------------------------------------------
// Gateway state + request handler.
// ---------------------------------------------------------------------------

/// Gateway configuration (subset of `SparkGatewayConfig` that survives the
/// hyper port).
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// Bearer key; `None`/empty disables auth (`--api-key` / `--api-key-file`).
    pub api_key: Option<String>,
    /// Request-size cap (`SPARK_GATEWAY_REQUEST_BYTES`); larger bodies get the
    /// 400 bad-request payload.
    pub max_request_bytes: usize,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self { api_key: None, max_request_bytes: DEFAULT_MAX_REQUEST_BYTES }
    }
}

struct GatewayState<B: GatewayBackend> {
    backend: Arc<B>,
    config: GatewayConfig,
    next_stream_request_id: AtomicU64,
    next_nonstream_request_id: AtomicU64,
}

impl<B: GatewayBackend> GatewayState<B> {
    fn next_stream_request_id(&self) -> u64 {
        let id = self.next_stream_request_id.fetch_add(1, Ordering::Relaxed);
        if id == 0 || (id & NONSTREAM_REQUEST_BIT) != 0 {
            self.next_stream_request_id.store(2, Ordering::Relaxed);
            return 1;
        }
        id
    }

    fn next_nonstream_request_id(&self) -> u64 {
        let id = self.next_nonstream_request_id.fetch_add(1, Ordering::Relaxed);
        if id == 0 || (id & NONSTREAM_REQUEST_BIT) != 0 {
            self.next_nonstream_request_id.store(2, Ordering::Relaxed);
            return NONSTREAM_REQUEST_BIT | 1;
        }
        NONSTREAM_REQUEST_BIT | id
    }
}

async fn handle_request<B: GatewayBackend>(
    state: Arc<GatewayState<B>>,
    request: Request<Incoming>,
) -> Response<GatewayBody> {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let authorization =
        request.headers().get(hyper::header::AUTHORIZATION).map(|value| value.as_bytes().to_vec());

    // Request-size cap (`SparkGatewayReadRequest` -6 → bad_request payload).
    let oversized = request
        .headers()
        .get(hyper::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<usize>().ok())
        .is_some_and(|length| length > state.config.max_request_bytes);
    if oversized {
        return json_error(StatusCode::BAD_REQUEST, BODY_BAD_REQUEST);
    }
    let body = match request.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => return json_error(StatusCode::BAD_REQUEST, BODY_BAD_REQUEST),
    };
    if body.len() > state.config.max_request_bytes {
        return json_error(StatusCode::BAD_REQUEST, BODY_BAD_REQUEST);
    }

    match route_request(&method, &path) {
        Route::CorsPreflight => full_response(StatusCode::OK, "text/plain", Bytes::new()),
        Route::DemoUi => full_response(StatusCode::OK, "text/html; charset=utf-8", DEMO_UI_HTML),
        Route::Health => {
            let view = state.backend.backend_view();
            let stats = state.backend.service_stats();
            full_response(
                StatusCode::OK,
                "application/json",
                build_service_health_body(&stats, &view),
            )
        }
        Route::None => json_error(StatusCode::NOT_FOUND, BODY_NOT_FOUND),
        Route::OpenAiChat | Route::OpenAiCompletions | Route::AnthropicMessages => {
            handle_submission(state, &route_request(&method, &path), authorization, body)
        }
    }
}

fn handle_submission<B: GatewayBackend>(
    state: Arc<GatewayState<B>>,
    route: &Route,
    authorization: Option<Vec<u8>>,
    body: Bytes,
) -> Response<GatewayBody> {
    if !authorization_matches(authorization.as_deref(), state.config.api_key.as_deref()) {
        return json_error(StatusCode::UNAUTHORIZED, BODY_UNAUTHORIZED);
    }
    let stream = body_requests_stream(&body);
    let view = state.backend.backend_view();
    let available = view.attached && view.runtime_initialized && view.local_control_ready;
    if !available {
        return unavailable_or_timeout(false, stream);
    }
    let submit_route = match route {
        Route::OpenAiChat => SubmitRoute::OpenAiChat,
        Route::OpenAiCompletions => SubmitRoute::OpenAiCompletions,
        _ => SubmitRoute::AnthropicMessages,
    };
    let client_request_id =
        if stream { state.next_stream_request_id() } else { state.next_nonstream_request_id() };
    let submission = Submission { route: submit_route, client_request_id, body: body.to_vec() };
    let result = match state.backend.submit(&submission) {
        Ok(result) => result,
        Err(SubmitError::Busy) => return unavailable_or_timeout(true, stream),
        Err(SubmitError::BackendUnavailable) => return unavailable_or_timeout(false, stream),
    };
    if !stream {
        return full_response(
            StatusCode::ACCEPTED,
            "application/json",
            build_submit_accepted_json(&result),
        );
    }
    let receiver = match state.backend.event_stream(result.client_request_id) {
        Some(receiver) => receiver,
        None => return unavailable_or_timeout(false, true),
    };
    let body = SseBody::new(
        state.backend.clone(),
        result.client_request_id,
        Bytes::from(build_submit_accepted_sse_frame(&result)),
        receiver,
    );
    let mut response = base_response(StatusCode::ACCEPTED, "text/event-stream");
    *response.body_mut() = body.boxed();
    response
}

// ---------------------------------------------------------------------------
// Server: hyper accept loop with graceful shutdown.
// ---------------------------------------------------------------------------

/// Outcome of [`select2`].
enum Either<A, B> {
    Left(A),
    Right(B),
}

/// Biased two-way select. The crate pins tokio without the `macros` feature,
/// so `tokio::select!` is unavailable; this polls `first` before `second`.
async fn select2<F1, F2, T1, T2>(first: F1, second: F2) -> Either<T1, T2>
where
    F1: Future<Output = T1>,
    F2: Future<Output = T2>,
{
    tokio::pin!(first);
    tokio::pin!(second);
    std::future::poll_fn(|cx| {
        if let Poll::Ready(value) = first.as_mut().poll(cx) {
            return Poll::Ready(Either::Left(value));
        }
        if let Poll::Ready(value) = second.as_mut().poll(cx) {
            return Poll::Ready(Either::Right(value));
        }
        Poll::Pending
    })
    .await
}

/// The running gateway. Binds a [`TcpListener`] and serves until the shutdown
/// signal fires, then gracefully drains in-flight connections.
pub struct HttpGateway<B: GatewayBackend> {
    state: Arc<GatewayState<B>>,
}

impl<B: GatewayBackend> HttpGateway<B> {
    pub fn new(backend: Arc<B>, config: GatewayConfig) -> Self {
        Self {
            state: Arc::new(GatewayState {
                backend,
                config,
                next_stream_request_id: AtomicU64::new(1),
                next_nonstream_request_id: AtomicU64::new(1),
            }),
        }
    }

    /// Serve on `listener` until `shutdown` flips to `true`. In-flight
    /// connections receive hyper's graceful shutdown; SSE bodies dropped by
    /// shutdown cancel their backend request via [`SseBody::drop`].
    pub async fn serve(
        self,
        listener: TcpListener,
        mut shutdown: watch::Receiver<bool>,
    ) -> io::Result<()> {
        let mut connections = JoinSet::new();
        loop {
            match select2(listener.accept(), shutdown.changed()).await {
                Either::Left(accepted) => {
                    let (stream, _) = accepted?;
                    let state = self.state.clone();
                    let conn_shutdown = shutdown.clone();
                    connections.spawn(async move {
                        serve_connection(state, stream, conn_shutdown).await;
                    });
                }
                Either::Right(changed) => {
                    if changed.is_err() || *shutdown.borrow() {
                        break;
                    }
                }
            }
        }
        while connections.join_next().await.is_some() {}
        Ok(())
    }
}

/// Serve one accepted connection; on shutdown, gracefully drain in-flight
/// requests. SSE bodies dropped by shutdown cancel their backend request via
/// [`SseBody::drop`].
async fn serve_connection<B: GatewayBackend>(
    state: Arc<GatewayState<B>>,
    stream: tokio::net::TcpStream,
    mut shutdown: watch::Receiver<bool>,
) {
    let io = TokioIo::new(stream);
    let service = service_fn(move |request| {
        let state = state.clone();
        async move { Ok::<_, Infallible>(handle_request(state, request).await) }
    });
    let mut connection = Box::pin(http1::Builder::new().serve_connection(io, service));
    match select2(shutdown.changed(), &mut connection).await {
        Either::Left(changed) => {
            if changed.is_ok() {
                connection.as_mut().graceful_shutdown();
                if let Err(error) = connection.await {
                    eprintln!("gateway connection error: {error}");
                }
            }
        }
        Either::Right(result) => {
            if let Err(error) = result {
                eprintln!("gateway connection error: {error}");
            }
        }
    }
}

/// Handle to a gateway running on a background task.
pub struct GatewayHandle {
    pub local_addr: SocketAddr,
    shutdown_tx: watch::Sender<bool>,
    join: tokio::task::JoinHandle<io::Result<()>>,
}

impl GatewayHandle {
    /// Signal shutdown and wait for the accept loop + connections to drain.
    pub async fn shutdown(self) {
        let _ = self.shutdown_tx.send(true);
        let _ = self.join.await;
    }
}

/// Bind `bind`, spawn the accept loop on the current runtime, and return a
/// handle. Convenience wrapper used by tests and the binary entrypoint.
pub async fn spawn_gateway<B: GatewayBackend>(
    backend: Arc<B>,
    config: GatewayConfig,
    bind: SocketAddr,
) -> io::Result<GatewayHandle> {
    let listener = TcpListener::bind(bind).await?;
    let local_addr = listener.local_addr()?;
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let gateway = HttpGateway::new(backend, config);
    let join = tokio::spawn(async move { gateway.serve(listener, shutdown_rx).await });
    Ok(GatewayHandle { local_addr, shutdown_tx, join })
}

/// Run a future on the calling thread's shared current-thread runtime.
///
/// Used by integration tests (the crate pins `tokio` without the `macros`
/// feature, so `#[tokio::test]` is unavailable). The runtime is reused across
/// calls on the same thread so tasks spawned by [`spawn_gateway`] outlive any
/// single `block_on` call.
pub fn block_on<F: Future>(future: F) -> F::Output {
    use std::cell::RefCell;
    thread_local! {
        static RUNTIME: RefCell<Option<tokio::runtime::Runtime>> = const { RefCell::new(None) };
    }
    RUNTIME.with(|cell| {
        let mut slot = cell.borrow_mut();
        let runtime = slot.get_or_insert_with(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("gateway test runtime")
        });
        runtime.block_on(future)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_match_c_surface() {
        assert_eq!(route_request(&Method::OPTIONS, "/v1/chat/completions"), Route::CorsPreflight);
        assert_eq!(route_request(&Method::GET, "/"), Route::DemoUi);
        assert_eq!(route_request(&Method::GET, "/health"), Route::Health);
        assert_eq!(route_request(&Method::POST, "/v1/chat/completions"), Route::OpenAiChat);
        assert_eq!(route_request(&Method::POST, "/v1/completions"), Route::OpenAiCompletions);
        assert_eq!(route_request(&Method::POST, "/v1/messages"), Route::AnthropicMessages);
        assert_eq!(route_request(&Method::POST, "/bad"), Route::None);
        assert_eq!(route_request(&Method::GET, "/v1/chat/completions"), Route::None);
    }

    #[test]
    fn stream_flag_scan_matches_c() {
        assert!(body_requests_stream(b"{\"model\":\"glm-5.2\",\"stream\" : true}"));
        assert!(!body_requests_stream(b"{\"model\":\"glm-5.2\"}"));
        assert!(body_requests_stream(b"{\"stream\":true}"));
        assert!(body_requests_stream(b"{\"stream\"\t:\r\ntrue}"));
        assert!(!body_requests_stream(b"{\"stream\":false}"));
        assert!(!body_requests_stream(b"{\"streamed\":true}"));
        // C prefix-matches `true` without checking the following byte.
        assert!(body_requests_stream(b"{\"stream\":truex}"));
        assert!(!body_requests_stream(b"\"stream\""));
        assert!(body_requests_stream(b"prefix \"stream\" : true suffix"));
    }

    #[test]
    fn authorization_matches_c_rules() {
        assert!(authorization_matches(None, None));
        assert!(authorization_matches(None, Some("")));
        assert!(authorization_matches(Some(b"Bearer secret"), Some("secret")));
        assert!(!authorization_matches(Some(b"Bearer bad"), Some("secret")));
        assert!(!authorization_matches(None, Some("secret")));
        let long_key = "k".repeat(600);
        assert!(!authorization_matches(Some(b"Bearer kkk"), Some(&long_key)));
    }

    #[test]
    fn event_names_and_terminal_match_c() {
        let mut event = ServiceEvent { kind: event_kind::TOKEN, ..Default::default() };
        assert_eq!(event.sse_event_name(), "token");
        assert!(!event.is_terminal());
        event.kind = event_kind::PREFILL_PROGRESS;
        assert_eq!(event.sse_event_name(), "event");
        for kind in
            [event_kind::REQUEST_COMPLETED, event_kind::REQUEST_CANCELLED, event_kind::ERROR]
        {
            event.kind = kind;
            assert_eq!(event.sse_event_name(), "done");
            assert!(event.is_terminal());
        }
    }

    #[test]
    fn token_frame_is_byte_exact() {
        let event = ServiceEvent {
            kind: event_kind::TOKEN,
            client_id: 11,
            client_request_id: 22,
            token_id: 333,
            token_index: 4,
            ..Default::default()
        };
        let frame = build_service_event_frame(&event, b"");
        let text = String::from_utf8(frame).unwrap();
        assert!(text.starts_with("event: token\n"));
        assert!(text.contains("\"token_id\":333"));
        assert!(text.contains("\"client_request_id\":22"));
        assert!(text.ends_with("\"text\":\"\"}\n\n"));
    }

    #[test]
    fn token_frame_escapes_like_c() {
        let event = ServiceEvent { kind: event_kind::TOKEN, ..Default::default() };
        let frame = build_service_event_frame(&event, b"a\"b\\c\nd\re\tf\x01g");
        let text = String::from_utf8(frame).unwrap();
        assert!(text.contains("\"text\":\"a\\\"b\\\\c\\nd\\re\\tf\\u0001g\""));
    }

    #[test]
    fn service_health_matches_c_fixture() {
        // Mirrors SparkTestHttpGatewayBuildsServiceHealth.
        let stats = ServiceStats {
            connected_client_count: 2,
            live_request_count: 3,
            serving: ServingStats {
                queued_request_count: 4,
                jit_prefetch_dispatch_count: 5,
                mtp_draft_token_count: 6,
                mtp_verify_dispatch_count: 7,
                mtp_accepted_draft_token_count: 8,
                mtp_committed_token_count: 8,
                completed_stream_count: 1,
                decoded_token_count: 9,
                maximum_decode_lane_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let view = BackendView {
            attached: true,
            runtime_initialized: true,
            local_control_ready: true,
            configured_kv_context_limit_tokens: 1_048_576,
            configured_max_active_sequences: 1,
            adaptive_decode_batch_width: 1,
            decode_batch_capacity: 256,
            prefill_wave_token_count: 16,
            speculation_configuration_flags: BACKEND_CONFIGURATION_FLAG_MTP,
            release_id: "release-1".into(),
            release_git_commit: "abcdef".into(),
            release_generation: 42,
            transport_shared_object_path: "tcp-host-staged.so".into(),
            first_blocker: "none".into(),
            ..Default::default()
        };
        let body = build_service_health_body(&stats, &view);
        assert!(body.contains("\"configured_kv_context_limit_tokens\":1048576"));
        assert!(body.contains("\"adaptive_decode_batch_width\":1"));
        assert!(body.contains("\"decode_batch_capacity\":256"));
        assert!(body.contains("\"prefill_wave_token_count\":16"));
        assert!(body.contains("\"release_git_commit\":\"abcdef\""));
        assert!(body.contains("\"end_to_end_observation_status\":\"OBSERVED\""));
        assert!(body.contains("\"performance_status\":\"NOT_MEASURED\""));
        assert!(body.contains("\"multi_sequence_batching_status\":\"NOT_WORKING\""));
        assert!(!body.contains("production_contract_flags"));
        assert!(body.contains("\"connected_clients\":2"));
        assert!(body.contains("\"jit_prefetch_dispatches\":5"));
        assert!(body.contains("\"mtp_draft_tokens\":6"));
        assert!(body.contains("\"mtp_verify_dispatches\":7"));
        assert!(body.contains("\"mtp_accepted_draft_tokens\":8"));
        assert!(body.contains("\"first_blocker\":\"none\""));
        assert!(body.ends_with("}\n"));
    }
}
