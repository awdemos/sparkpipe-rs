//! `spark-serve`: serving plane, ported from the C tree's `api/`.
//!
//! - [`request_api`]: request/session slot lifecycle, dispatch, speculative
//!   (MTP) policy (port of `api/request.c` — the C tree's largest file)
//! - [`serving_engine`]: engine orchestration (port of `api/serving_engine.c`
//!   + `api/service.c`)
//! - [`http_gateway`]: HTTP/1.1 gateway + SSE on hyper (replaces
//!   `api/gateway/http_server.c`'s raw poll loop; API surface ported from
//!   `api/http_gateway.c` + OpenAI-compat `api/compat_api.c`)

pub mod http_gateway;
pub mod request_api;
pub mod serving_engine;
