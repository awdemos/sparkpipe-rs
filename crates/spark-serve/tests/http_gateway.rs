//! Integration tests for the hyper-based HTTP gateway: route surface, JSON
//! shapes, SSE framing (byte-level), auth, and shutdown behavior. Ports the
//! expectations of `tests/test_glm52_http_gateway.c` from the C tree.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use spark_serve::http_gateway::{
    block_on, event_kind, spawn_gateway, BackendView, GatewayBackend, GatewayConfig, ServiceEvent,
    ServiceStats, Submission, SubmitError, SubmitResult, NONSTREAM_REQUEST_BIT,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// Fake backend.
// ---------------------------------------------------------------------------

struct FakeSubmission {
    submission: Submission,
}

struct FakeBackend {
    view: Mutex<BackendView>,
    stats: Mutex<ServiceStats>,
    submit_error: Mutex<Option<SubmitError>>,
    submissions: Mutex<Vec<FakeSubmission>>,
    senders: Mutex<HashMap<u64, mpsc::Sender<ServiceEvent>>>,
    receivers: Mutex<HashMap<u64, mpsc::Receiver<ServiceEvent>>>,
    cancelled: Mutex<Vec<u64>>,
    next_serving_request_id: Mutex<u64>,
}

impl FakeBackend {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            view: Mutex::new(BackendView {
                attached: true,
                runtime_initialized: true,
                local_control_ready: true,
                ..Default::default()
            }),
            stats: Mutex::new(ServiceStats::default()),
            submit_error: Mutex::new(None),
            submissions: Mutex::new(Vec::new()),
            senders: Mutex::new(HashMap::new()),
            receivers: Mutex::new(HashMap::new()),
            cancelled: Mutex::new(Vec::new()),
            next_serving_request_id: Mutex::new(9000),
        })
    }

    fn set_ready(&self, ready: bool) {
        let mut view = self.view.lock().unwrap();
        view.attached = ready;
        view.runtime_initialized = ready;
        view.local_control_ready = ready;
    }

    fn set_submit_error(&self, error: SubmitError) {
        *self.submit_error.lock().unwrap() = Some(error);
    }

    /// Queue events for a streaming submission (sender registered at submit).
    fn push_events(&self, client_request_id: u64, events: Vec<ServiceEvent>) {
        let sender = self
            .senders
            .lock()
            .unwrap()
            .get(&client_request_id)
            .cloned()
            .expect("stream sender registered");
        for event in events {
            sender.try_send(event).unwrap();
        }
    }

    fn cancelled_ids(&self) -> Vec<u64> {
        self.cancelled.lock().unwrap().clone()
    }

    fn submissions(&self) -> Vec<(u64, Vec<u8>)> {
        self.submissions
            .lock()
            .unwrap()
            .iter()
            .map(|entry| (entry.submission.client_request_id, entry.submission.body.clone()))
            .collect()
    }
}

impl GatewayBackend for FakeBackend {
    fn backend_view(&self) -> BackendView {
        self.view.lock().unwrap().clone()
    }

    fn service_stats(&self) -> ServiceStats {
        self.stats.lock().unwrap().clone()
    }

    fn submit(&self, submission: &Submission) -> Result<SubmitResult, SubmitError> {
        if let Some(error) = self.submit_error.lock().unwrap().take() {
            return Err(error);
        }
        let mut serving_id = self.next_serving_request_id.lock().unwrap();
        *serving_id += 1;
        let result = SubmitResult {
            client_request_id: submission.client_request_id,
            serving_request_id: *serving_id,
            sequence_id: 77,
            prompt_token_count: 12,
            output_token_budget: 256,
        };
        let (sender, receiver) = mpsc::channel(64);
        self.senders.lock().unwrap().insert(submission.client_request_id, sender);
        self.receivers.lock().unwrap().insert(submission.client_request_id, receiver);
        self.submissions.lock().unwrap().push(FakeSubmission { submission: submission.clone() });
        Ok(result)
    }

    fn event_stream(&self, client_request_id: u64) -> Option<mpsc::Receiver<ServiceEvent>> {
        self.receivers.lock().unwrap().remove(&client_request_id)
    }

    fn cancel(&self, client_request_id: u64) {
        self.cancelled.lock().unwrap().push(client_request_id);
    }

    fn decode_token(&self, token_id: u32) -> Vec<u8> {
        match token_id {
            333 => b"hello".to_vec(),
            444 => b"a\"b\nc".to_vec(),
            _ => Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Minimal raw HTTP client (byte-level control, no extra deps).
// ---------------------------------------------------------------------------

struct RawResponse {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl RawResponse {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|window| window == needle)
}

async fn read_more(stream: &mut TcpStream, buffer: &mut Vec<u8>) {
    let mut chunk = [0u8; 4096];
    let read = stream.read(&mut chunk).await.expect("read from socket");
    assert!(read > 0, "connection closed mid-response");
    buffer.extend_from_slice(&chunk[..read]);
}

/// Read status line + headers; leftover body bytes stay in `buffer`.
async fn read_head(stream: &mut TcpStream, buffer: &mut Vec<u8>) -> (u16, Vec<(String, String)>) {
    let header_end = loop {
        if let Some(pos) = find_subslice(buffer, b"\r\n\r\n") {
            break pos;
        }
        read_more(stream, buffer).await;
    };
    let head = String::from_utf8(buffer[..header_end].to_vec()).unwrap();
    *buffer = buffer.split_off(header_end + 4);
    let mut lines = head.split("\r\n");
    let status: u16 = lines.next().unwrap().split_whitespace().nth(1).unwrap().parse().unwrap();
    let mut headers = Vec::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_string(), value.trim().to_string()));
        }
    }
    (status, headers)
}

/// Read one chunked-transfer chunk; `None` on the terminal zero chunk.
async fn read_chunk(stream: &mut TcpStream, buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    while find_subslice(buffer, b"\r\n").is_none() {
        read_more(stream, buffer).await;
    }
    let line_end = find_subslice(buffer, b"\r\n").unwrap();
    let size_text = String::from_utf8(buffer[..line_end].to_vec()).unwrap();
    let size = usize::from_str_radix(size_text.trim(), 16).unwrap();
    if size == 0 {
        return None;
    }
    while buffer.len() < line_end + 2 + size + 2 {
        read_more(stream, buffer).await;
    }
    let data = buffer[line_end + 2..line_end + 2 + size].to_vec();
    *buffer = buffer.split_off(line_end + 2 + size + 2);
    Some(data)
}

/// Read a full response with a Content-Length or chunked body.
async fn read_response(stream: &mut TcpStream, buffer: &mut Vec<u8>) -> RawResponse {
    let (status, headers) = read_head(stream, buffer).await;
    let chunked = headers.iter().any(|(key, value)| {
        key.eq_ignore_ascii_case("transfer-encoding") && value.contains("chunked")
    });
    let body = if chunked {
        let mut decoded = Vec::new();
        while let Some(chunk) = read_chunk(stream, buffer).await {
            decoded.extend_from_slice(&chunk);
        }
        decoded
    } else {
        let length: usize = headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case("content-length"))
            .and_then(|(_, value)| value.parse().ok())
            .unwrap_or(0);
        while buffer.len() < length {
            read_more(stream, buffer).await;
        }
        let body = buffer[..length].to_vec();
        *buffer = buffer.split_off(length);
        body
    };
    RawResponse { status, headers, body }
}

// ---------------------------------------------------------------------------
// Server fixture.
// ---------------------------------------------------------------------------

struct TestServer {
    addr: SocketAddr,
    backend: Arc<FakeBackend>,
    handle: spark_serve::http_gateway::GatewayHandle,
}

fn start_server(config: GatewayConfig) -> TestServer {
    let backend = FakeBackend::new();
    let backend_clone = backend.clone();
    let handle = block_on(async move {
        spawn_gateway(backend_clone, config, "127.0.0.1:0".parse().unwrap()).await.unwrap()
    });
    TestServer { addr: handle.local_addr, backend, handle }
}

impl TestServer {
    fn request(&self, raw: &str) -> RawResponse {
        block_on(async {
            let mut stream = TcpStream::connect(self.addr).await.unwrap();
            stream.write_all(raw.as_bytes()).await.unwrap();
            let mut buffer = Vec::new();
            read_response(&mut stream, &mut buffer).await
        })
    }

    fn shutdown(self) {
        block_on(self.handle.shutdown());
    }
}

// ---------------------------------------------------------------------------
// Route surface + static responses.
// ---------------------------------------------------------------------------

#[test]
fn cors_preflight_matches_c() {
    let server = start_server(GatewayConfig::default());
    let response = server.request("OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: x\r\n\r\n");
    assert_eq!(response.status, 200);
    assert_eq!(response.header("content-type"), Some("text/plain"));
    assert_eq!(response.header("access-control-allow-origin"), Some("*"));
    assert_eq!(
        response.header("access-control-allow-headers"),
        Some("authorization, content-type")
    );
    assert_eq!(response.header("access-control-allow-methods"), Some("GET, POST, OPTIONS"));
    assert_eq!(response.header("cache-control"), Some("no-cache"));
    assert!(response.body.is_empty());
    server.shutdown();
}

#[test]
fn demo_ui_serves_html() {
    let server = start_server(GatewayConfig::default());
    let response = server.request("GET / HTTP/1.1\r\nHost: x\r\n\r\n");
    assert_eq!(response.status, 200);
    assert_eq!(response.header("content-type"), Some("text/html; charset=utf-8"));
    assert!(response.body.starts_with(b"<!doctype html>"));
    assert!(find_subslice(&response.body, b"SparkPipe GLM 5.2").is_some());
    assert!(response.body.ends_with(b"</html>"));
    server.shutdown();
}

#[test]
fn unknown_route_is_404_byte_exact() {
    let server = start_server(GatewayConfig::default());
    for raw in [
        "GET /bad HTTP/1.1\r\nHost: x\r\n\r\n",
        "POST /bad HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}",
    ] {
        let response = server.request(raw);
        assert_eq!(response.status, 404);
        assert_eq!(response.header("content-type"), Some("application/json"));
        assert_eq!(
            response.body,
            b"{\"error\":{\"type\":\"not_found\",\"message\":\"unknown endpoint\"}}\n"
        );
    }
    server.shutdown();
}

#[test]
fn health_reports_unattached_backend_as_zeros() {
    let server = start_server(GatewayConfig::default());
    server.backend.set_ready(false);
    let response = server.request("GET /health HTTP/1.1\r\nHost: x\r\n\r\n");
    assert_eq!(response.status, 200);
    assert_eq!(response.header("content-type"), Some("application/json"));
    let body = String::from_utf8(response.body).unwrap();
    assert!(body.starts_with("{\"schema\":\"sparkpipe.runtime_observation.v1\","));
    assert!(body.contains("\"runtime_initialized\":0"));
    assert!(body.contains("\"local_control_ready\":0"));
    assert!(body.contains("\"release_identity_status\":\"NOT_MEASURED\""));
    assert!(body.contains("\"multi_sequence_batching_status\":\"NOT_WORKING\""));
    assert!(body.contains("\"mtp_status\":\"NOT_WORKING\""));
    assert!(body.contains("\"first_blocker\":\"\""));
    assert!(body.ends_with("}\n"));
    server.shutdown();
}

#[test]
fn health_full_body_is_byte_exact() {
    let server = start_server(GatewayConfig::default());
    {
        let mut view = server.backend.view.lock().unwrap();
        view.release_id = "release-1".into();
        view.release_git_commit = "abcdef".into();
        view.release_generation = 42;
        view.configured_kv_context_limit_tokens = 1_048_576;
        view.configured_max_active_sequences = 1;
        view.adaptive_decode_batch_width = 1;
        view.decode_batch_capacity = 256;
        view.prefill_wave_token_count = 16;
        view.speculation_configuration_flags =
            spark_serve::http_gateway::BACKEND_CONFIGURATION_FLAG_MTP;
        view.transport_shared_object_path = "tcp-host-staged.so".into();
        view.first_blocker = "none".into();
        let mut stats = server.backend.stats.lock().unwrap();
        stats.connected_client_count = 2;
        stats.live_request_count = 3;
        stats.serving.queued_request_count = 4;
        stats.serving.jit_prefetch_dispatch_count = 5;
        stats.serving.mtp_draft_token_count = 6;
        stats.serving.mtp_verify_dispatch_count = 7;
        stats.serving.mtp_accepted_draft_token_count = 8;
        stats.serving.mtp_committed_token_count = 8;
        stats.serving.completed_stream_count = 1;
        stats.serving.decoded_token_count = 9;
        stats.serving.maximum_decode_lane_count = 1;
    }
    let response = server.request("GET /health HTTP/1.1\r\nHost: x\r\n\r\n");
    assert_eq!(response.status, 200);
    let expected = concat!(
        "{\"schema\":\"sparkpipe.runtime_observation.v1\",",
        "\"release_identity_status\":\"OBSERVED\",",
        "\"release_id\":\"release-1\",",
        "\"release_git_commit\":\"abcdef\",",
        "\"release_generation\":42,",
        "\"runtime_initialized\":1,",
        "\"local_control_ready\":1,",
        "\"configured_kv_context_limit_tokens\":1048576,",
        "\"configured_max_active_sequences\":1,",
        "\"adaptive_decode_batch_width\":1,",
        "\"decode_batch_capacity\":256,",
        "\"prefill_wave_token_count\":16,",
        "\"transport_shared_object\":\"tcp-host-staged.so\",",
        "\"bound_transport_interface_flags\":0,",
        "\"transport_binding_status\":\"NOT_MEASURED\",",
        "\"end_to_end_observation_status\":\"OBSERVED\",",
        "\"accuracy_status\":\"NOT_MEASURED\",",
        "\"performance_status\":\"NOT_MEASURED\",",
        "\"multi_sequence_batching_status\":\"NOT_WORKING\",",
        "\"long_context_status\":\"NOT_MEASURED\",",
        "\"jit_kv_status\":\"NOT_WORKING\",",
        "\"dspark_status\":\"NOT_WORKING\",",
        "\"mtp_status\":\"OBSERVED\",",
        "\"connected_clients\":2,",
        "\"live_requests\":3,",
        "\"queued_requests\":4,",
        "\"completed_streams\":1,",
        "\"prefill_dispatches\":0,",
        "\"prefill_batch_dispatches\":0,",
        "\"prefill_tokens\":0,",
        "\"decode_dispatches\":0,",
        "\"decoded_tokens\":9,",
        "\"maximum_prefill_active_sequences\":0,",
        "\"maximum_prefill_lanes\":0,",
        "\"maximum_decode_active_sequences\":0,",
        "\"maximum_decode_lanes\":1,",
        "\"event_backlog\":0,",
        "\"dropped_events\":0,",
        "\"jit_prefetch_dispatches\":5,",
        "\"jit_prefetch_blocks\":0,",
        "\"async_jit_prefetch_starts\":0,",
        "\"async_jit_prefetch_completions\":0,",
        "\"mtp_draft_tokens\":6,",
        "\"mtp_verify_dispatches\":7,",
        "\"mtp_draft_ready\":0,",
        "\"mtp_accepted_draft_tokens\":8,",
        "\"mtp_committed_tokens\":8,",
        "\"mtp_rejected_tokens\":0,",
        "\"first_blocker\":\"none\"}\n",
    );
    assert_eq!(String::from_utf8(response.body).unwrap(), expected);
    server.shutdown();
}

// ---------------------------------------------------------------------------
// Auth.
// ---------------------------------------------------------------------------

#[test]
fn auth_enforced_on_completion_routes_only() {
    let config = GatewayConfig { api_key: Some("secret".into()), ..Default::default() };
    let server = start_server(config);
    let post = "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}";

    // Missing header → 401 byte-exact payload.
    let response = server.request(post);
    assert_eq!(response.status, 401);
    assert_eq!(
        response.body,
        b"{\"error\":{\"type\":\"unauthorized\",\"message\":\"missing or invalid bearer token\"}}\n"
    );

    // Wrong key → 401.
    let response =
        server.request(&post.replace("\r\n\r\n", "\r\nAuthorization: Bearer bad\r\n\r\n"));
    assert_eq!(response.status, 401);

    // Correct key → submission accepted.
    let response =
        server.request(&post.replace("\r\n\r\n", "\r\nAuthorization: Bearer secret\r\n\r\n"));
    assert_eq!(response.status, 202);

    // Health and demo UI are not auth-gated (matches C routing order).
    let response = server.request("GET /health HTTP/1.1\r\nHost: x\r\n\r\n");
    assert_eq!(response.status, 200);
    let response = server.request("GET / HTTP/1.1\r\nHost: x\r\n\r\n");
    assert_eq!(response.status, 200);
    server.shutdown();
}

// ---------------------------------------------------------------------------
// Submission: unavailable / timeout / accepted / oversized.
// ---------------------------------------------------------------------------

#[test]
fn backend_unavailable_json_and_sse() {
    let server = start_server(GatewayConfig::default());
    server.backend.set_ready(false);

    let response = server
        .request("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}");
    assert_eq!(response.status, 503);
    assert_eq!(response.header("content-type"), Some("application/json"));
    assert_eq!(
        response.body,
        b"{\"error\":{\"type\":\"backend_unavailable\",\"message\":\"GLM52 RING backend is not attached\"}}\n"
    );

    // Same failure, streaming request → SSE error frame, byte-exact.
    let response = server.request(
        "POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 15\r\n\r\n{\"stream\":true}",
    );
    assert_eq!(response.status, 503);
    assert_eq!(response.header("content-type"), Some("text/event-stream"));
    assert_eq!(
        response.body,
        b"event: error\ndata: {\"error\":{\"type\":\"backend_unavailable\",\"message\":\"GLM52 RING backend is not attached\"}}\n\n"
    );
    server.shutdown();
}

#[test]
fn busy_submit_maps_to_504_json_and_sse() {
    let server = start_server(GatewayConfig::default());
    server.backend.set_submit_error(SubmitError::Busy);
    let response =
        server.request("POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}");
    assert_eq!(response.status, 504);
    assert_eq!(response.header("content-type"), Some("application/json"));
    assert_eq!(
        response.body,
        b"{\"error\":{\"type\":\"request_timeout\",\"message\":\"GLM52 RING request produced no terminal event before the stream poll budget\"}}\n"
    );

    server.backend.set_submit_error(SubmitError::Busy);
    let response = server.request(
        "POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Length: 15\r\n\r\n{\"stream\":true}",
    );
    assert_eq!(response.status, 504);
    assert_eq!(response.header("content-type"), Some("text/event-stream"));
    assert_eq!(
        response.body,
        b"event: error\ndata: {\"error\":{\"type\":\"request_timeout\",\"message\":\"GLM52 RING request produced no terminal event before the stream poll budget\"}}\n\n"
    );
    server.shutdown();
}

#[test]
fn nonstream_submit_accepted_is_byte_exact() {
    let server = start_server(GatewayConfig::default());
    let response = server
        .request("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}");
    assert_eq!(response.status, 202);
    assert_eq!(response.header("content-type"), Some("application/json"));
    let submissions = server.backend.submissions();
    assert_eq!(submissions.len(), 1);
    let (client_request_id, body) = &submissions[0];
    assert_ne!(*client_request_id & NONSTREAM_REQUEST_BIT, 0);
    assert_eq!(body, b"{}");
    let expected = format!(
        "{{\"id\":\"spreq-9001\",\"object\":\"sparkpipe.request\",\
\"client_request_id\":{client_request_id},\"serving_request_id\":9001,\
\"sequence_id\":77,\"prompt_tokens\":12,\"output_token_budget\":256,\
\"status\":\"queued\"}}\n"
    );
    assert_eq!(String::from_utf8(response.body).unwrap(), expected);
    server.shutdown();
}

#[test]
fn oversized_request_is_400_byte_exact() {
    let config = GatewayConfig { max_request_bytes: 8, ..Default::default() };
    let server = start_server(config);
    let response = server.request(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 15\r\n\r\n{\"stream\":true}",
    );
    assert_eq!(response.status, 400);
    assert_eq!(response.header("content-type"), Some("application/json"));
    assert_eq!(
        response.body,
        b"{\"error\":{\"type\":\"bad_request\",\"message\":\"invalid or oversized request\"}}\n"
    );
    server.shutdown();
}

// ---------------------------------------------------------------------------
// SSE streaming.
// ---------------------------------------------------------------------------

#[test]
fn sse_stream_frames_are_byte_exact() {
    let server = start_server(GatewayConfig::default());
    block_on(async {
        let mut stream = TcpStream::connect(server.addr).await.unwrap();
        stream
            .write_all(
                b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 33\r\n\r\n{\"model\":\"glm-5.2\",\"stream\":true}",
            )
            .await
            .unwrap();
        let mut buffer = Vec::new();
        let (status, headers) = read_head(&mut stream, &mut buffer).await;
        assert_eq!(status, 202);
        let content_type = headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case("content-type"))
            .map(|(_, value)| value.as_str());
        assert_eq!(content_type, Some("text/event-stream"));
        let transfer = headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case("transfer-encoding"))
            .map(|(_, value)| value.as_str());
        assert_eq!(transfer, Some("chunked"), "SSE body must be streamed, not a fixed-length body");

        let submissions = server.backend.submissions();
        assert_eq!(submissions.len(), 1);
        let client_request_id = submissions[0].0;
        assert_eq!(client_request_id & NONSTREAM_REQUEST_BIT, 0);
        assert_eq!(submissions[0].1, b"{\"model\":\"glm-5.2\",\"stream\":true}");

        // First chunk: the accepted frame, byte-exact.
        let first = read_chunk(&mut stream, &mut buffer).await.expect("accepted frame chunk");
        let accepted = format!(
            "event: accepted\ndata: {{\"client_request_id\":{client_request_id},\
\"serving_request_id\":9001,\"sequence_id\":77,\"prompt_tokens\":12,\
\"output_token_budget\":256}}\n\n"
        );
        assert_eq!(String::from_utf8(first).unwrap(), accepted);

        server.backend.push_events(
            client_request_id,
            vec![
                ServiceEvent {
                    kind: event_kind::TOKEN,
                    client_request_id,
                    serving_request_id: 9001,
                    sequence_id: 77,
                    token_id: 333,
                    token_index: 4,
                    ..Default::default()
                },
                // Non-token, non-terminal kind → `event: event`.
                ServiceEvent {
                    kind: event_kind::PREFILL_PROGRESS,
                    client_request_id,
                    serving_request_id: 9001,
                    sequence_id: 77,
                    ..Default::default()
                },
                // REQUEST_ACCEPTED is filtered out by the gateway.
                ServiceEvent {
                    kind: event_kind::REQUEST_ACCEPTED,
                    client_request_id,
                    serving_request_id: 9001,
                    sequence_id: 77,
                    ..Default::default()
                },
                ServiceEvent {
                    kind: event_kind::REQUEST_COMPLETED,
                    client_request_id,
                    serving_request_id: 9001,
                    sequence_id: 77,
                    ..Default::default()
                },
            ],
        );

        let mut rest = Vec::new();
        while let Some(chunk) = read_chunk(&mut stream, &mut buffer).await {
            rest.extend_from_slice(&chunk);
        }
        let token_frame = format!(
            "event: token\ndata: {{\"kind\":3,\"status\":0,\"client_id\":0,\
\"client_request_id\":{client_request_id},\"serving_request_id\":9001,\
\"sequence_id\":77,\"token_id\":333,\"token_index\":4,\"text\":\"hello\"}}\n\n"
        );
        let progress_frame = format!(
            "event: event\ndata: {{\"kind\":2,\"status\":0,\"client_id\":0,\
\"client_request_id\":{client_request_id},\"serving_request_id\":9001,\
\"sequence_id\":77,\"token_id\":0,\"token_index\":0,\"text\":\"\"}}\n\n"
        );
        let done_frame = format!(
            "event: done\ndata: {{\"kind\":4,\"status\":0,\"client_id\":0,\
\"client_request_id\":{client_request_id},\"serving_request_id\":9001,\
\"sequence_id\":77,\"token_id\":0,\"token_index\":0,\"text\":\"\"}}\n\n"
        );
        let expected = format!("{token_frame}{progress_frame}{done_frame}");
        assert_eq!(String::from_utf8(rest).unwrap(), expected);

        // Terminal event ends the stream without a cancel.
        assert!(server.backend.cancelled_ids().is_empty());
    });
    server.shutdown();
}

#[test]
fn sse_token_text_is_json_escaped_like_c() {
    let server = start_server(GatewayConfig::default());
    block_on(async {
        let mut stream = TcpStream::connect(server.addr).await.unwrap();
        stream
            .write_all(
                b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 15\r\n\r\n{\"stream\":true}",
            )
            .await
            .unwrap();
        let mut buffer = Vec::new();
        let _ = read_head(&mut stream, &mut buffer).await;
        let _accepted = read_chunk(&mut stream, &mut buffer).await;
        let submissions = server.backend.submissions();
        let client_request_id = submissions[0].0;

        server.backend.push_events(
            client_request_id,
            vec![
                ServiceEvent {
                    kind: event_kind::TOKEN,
                    client_request_id,
                    serving_request_id: 9001,
                    token_id: 444,
                    token_index: 0,
                    ..Default::default()
                },
                ServiceEvent {
                    kind: event_kind::REQUEST_CANCELLED,
                    client_request_id,
                    serving_request_id: 9001,
                    ..Default::default()
                },
            ],
        );
        let mut rest = Vec::new();
        while let Some(chunk) = read_chunk(&mut stream, &mut buffer).await {
            rest.extend_from_slice(&chunk);
        }
        let body = String::from_utf8(rest).unwrap();
        // decode_token(444) → a"b\nc → escaped per the C escaper.
        assert!(body.contains("\"text\":\"a\\\"b\\nc\""), "body: {body}");
        // REQUEST_CANCELLED is terminal → `event: done`.
        assert!(body.contains("event: done\n"), "body: {body}");
        assert!(server.backend.cancelled_ids().is_empty());
    });
    server.shutdown();
}

#[test]
fn sse_client_disconnect_cancels_request() {
    let server = start_server(GatewayConfig::default());
    block_on(async {
        let mut stream = TcpStream::connect(server.addr).await.unwrap();
        stream
            .write_all(
                b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 15\r\n\r\n{\"stream\":true}",
            )
            .await
            .unwrap();
        let mut buffer = Vec::new();
        let _ = read_head(&mut stream, &mut buffer).await;
        let _accepted = read_chunk(&mut stream, &mut buffer).await;
        let submissions = server.backend.submissions();
        let client_request_id = submissions[0].0;

        // Drop the client connection mid-stream (no terminal event).
        drop(stream);
        for _ in 0..100 {
            if server.backend.cancelled_ids().contains(&client_request_id) {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("gateway did not cancel the disconnected stream");
    });
    server.shutdown();
}

// ---------------------------------------------------------------------------
// Keep-alive (deliberate upgrade over the C `Connection: close`).
// ---------------------------------------------------------------------------

#[test]
fn keep_alive_serves_multiple_requests_on_one_connection() {
    let server = start_server(GatewayConfig::default());
    block_on(async {
        let mut stream = TcpStream::connect(server.addr).await.unwrap();
        let mut buffer = Vec::new();
        stream.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n").await.unwrap();
        let first = read_response(&mut stream, &mut buffer).await;
        assert_eq!(first.status, 200);
        stream.write_all(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n").await.unwrap();
        let second = read_response(&mut stream, &mut buffer).await;
        assert_eq!(second.status, 200);
        assert!(second.body.starts_with(b"<!doctype html>"));
    });
    server.shutdown();
}
