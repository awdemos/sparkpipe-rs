//! OpenAI/Anthropic-compatible API surface — port of `api/compat_api.c`
//! (`spark_glm52_compat_api.h`).
//!
//! Shapes chat-completions JSON into GLM-5.2 chat-template text plus submit
//! budgets, ready for [`ServiceRuntime::submit_text`]. The C parses with the
//! in-tree `SparkJsonDocument`; here `serde_json::Value` plays that role
//! (member lookup, type tests, string decoding are 1:1), and the GLM-5.2
//! chat template fragments come from `spark-text`'s `ChatTemplateWriter`.
//!
//! Rust-port deviations (behavior otherwise faithful):
//! - The caller-provided `text`/`text_capacity` buffer becomes an owned
//!   writer inside [`CompatTextRequest`]; `capacity` is the exact byte
//!   budget (the C reserves one byte for the NUL terminator).
//! - On a failed `prepare_*` the C leaves partially written template text
//!   in the buffer; here the buffer contents after an error are likewise
//!   unspecified (only the status is part of the contract).

use spark_text::chat_template::{
    ChatTemplateWriter, Role, FLAG_ADD_GENERATION_PROMPT, FLAG_ENABLE_THINKING,
};

use super::service::{ServiceRuntime, ServiceSubmitResult, ServiceSubmitTextRequest};
use super::status::ServingStatus;

/// `SPARK_GLM52_COMPAT_API_ABI_VERSION`.
pub const COMPAT_API_ABI_VERSION: u32 = 2;

/// `SPARK_GLM52_COMPAT_DEFAULT_CHAT_FLAGS`.
pub const COMPAT_DEFAULT_CHAT_FLAGS: u32 = FLAG_ADD_GENERATION_PROMPT | FLAG_ENABLE_THINKING;

/// `SparkGlm52CompatTextRequest`.
///
/// Built by [`CompatTextRequest::new`] with a byte capacity; the prepared
/// template text is readable via [`CompatTextRequest::text`].
pub struct CompatTextRequest {
    /// `SPARK_SERVICE_FRAME_FLAG_*` submit bits, passed through to submit.
    pub flags: u32,
    /// `SPARK_GLM52_CHAT_TEMPLATE_FLAG_*` bits resolved from the request.
    pub chat_template_flags: u32,
    /// Scheduling priority (`"priority"` member).
    pub priority: u32,
    /// Thinking-phase budget (`"thinking_budget_tokens"` /
    /// `"thinking_token_budget"`).
    pub thinking_token_budget: u32,
    /// Output budget (`"max_tokens"` / `"max_completion_tokens"`).
    pub output_token_budget: u32,
    /// Maximum prefill tokens per step, passed through to submit.
    pub max_prefill_tokens_per_step: u32,
    /// `SPARK_TOKENIZER_ENCODE_FLAG_*` bits, passed through to submit.
    pub tokenizer_encode_flags: u32,
    /// Submitting client (caller-assigned before submit).
    pub client_id: u64,
    /// Client-scoped request id (caller-assigned before submit).
    pub client_request_id: u64,
    /// Caller sequence id.
    pub sequence_id: u64,
    writer: ChatTemplateWriter,
}

impl CompatTextRequest {
    /// `SparkGlm52CompatInitializeTextRequest`.
    pub fn new(text_capacity: usize) -> Self {
        CompatTextRequest {
            flags: 0,
            chat_template_flags: COMPAT_DEFAULT_CHAT_FLAGS,
            priority: 0,
            thinking_token_budget: 0,
            output_token_budget: 0,
            max_prefill_tokens_per_step: 0,
            tokenizer_encode_flags: 0,
            client_id: 0,
            client_request_id: 0,
            sequence_id: 0,
            writer: ChatTemplateWriter::new(text_capacity),
        }
    }

    /// The prepared text (`text` in C).
    pub fn text(&self) -> &str {
        self.writer.as_str()
    }

    /// Bytes prepared so far (`text_bytes` in C).
    pub fn text_bytes(&self) -> usize {
        self.writer.len()
    }

    /// Byte budget (`text_capacity` in C).
    pub fn text_capacity(&self) -> usize {
        self.writer.capacity()
    }

    /// `SparkGlm52CompatAppendBytes`.
    fn append(&mut self, text: &str) -> Result<(), ServingStatus> {
        self.writer.append(text).map_err(ServingStatus::from)
    }

    /// `SparkGlm52CompatBeginTemplate` (reasoning effort is always the C's
    /// hardcoded `"Max"`; the writer normalizes it identically).
    fn begin_template(&mut self) -> Result<(), ServingStatus> {
        self.writer.begin(Some("Max"), self.chat_template_flags).map_err(ServingStatus::from)
    }

    /// `SparkGlm52CompatBeginMessage`.
    fn begin_message(&mut self, role: Role) -> Result<(), ServingStatus> {
        self.writer.begin_message(role).map_err(ServingStatus::from)
    }

    /// `SparkGlm52CompatEndMessage`.
    fn end_message(&mut self, role: Role) -> Result<(), ServingStatus> {
        self.writer.end_message(role).map_err(ServingStatus::from)
    }

    /// `SparkGlm52CompatFinishTemplate`.
    fn finish_template(&mut self) -> Result<(), ServingStatus> {
        self.writer.finish(self.chat_template_flags).map_err(ServingStatus::from)
    }
}

/// `SparkCompatReadOptionalUInt32`.
fn read_optional_u32(
    root: &serde_json::Value,
    member_name: &str,
) -> Result<Option<u32>, ServingStatus> {
    let Some(value) = root.get(member_name) else {
        return Ok(None);
    };
    let Some(value) = value.as_u64() else {
        return Err(ServingStatus::InvalidArgument);
    };
    if value > u32::MAX as u64 {
        return Err(ServingStatus::InvalidArgument);
    }
    Ok(Some(value as u32))
}

/// `SparkGlm52CompatAppendOptionalJsonString`; returns whether a string was
/// appended.
fn append_optional_json_string(
    object: &serde_json::Value,
    member_name: &str,
    request: &mut CompatTextRequest,
) -> Result<bool, ServingStatus> {
    let Some(value) = object.get(member_name) else {
        return Ok(false);
    };
    let Some(text) = value.as_str() else {
        return Err(ServingStatus::InvalidArgument);
    };
    request.append(text)?;
    Ok(true)
}

/// `SparkGlm52CompatAppendFileName`.
fn append_file_name(
    object: &serde_json::Value,
    request: &mut CompatTextRequest,
) -> Result<(), ServingStatus> {
    if append_optional_json_string(object, "filename", request)? {
        return Ok(());
    }
    if append_optional_json_string(object, "file_name", request)? {
        return Ok(());
    }
    append_optional_json_string(object, "name", request)?;
    Ok(())
}

/// `SparkGlm52CompatAppendFileContentField`.
fn append_file_content_field(
    object: &serde_json::Value,
    request: &mut CompatTextRequest,
) -> Result<bool, ServingStatus> {
    const FIELD_NAMES: [&str; 5] = ["content", "file_content", "file_text", "text", "data"];
    for field_name in FIELD_NAMES {
        if append_optional_json_string(object, field_name, request)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `SparkGlm52CompatAppendFileObject`.
fn append_file_object(
    object: &serde_json::Value,
    request: &mut CompatTextRequest,
) -> Result<bool, ServingStatus> {
    request.append("\n[uploaded file")?;
    request.append(": ")?;
    append_file_name(object, request)?;
    request.append("]\n")?;
    let appended = append_file_content_field(object, request)?;
    if !appended {
        return Err(ServingStatus::InvalidArgument);
    }
    request.append("\n[/uploaded file]\n")?;
    Ok(true)
}

/// `SparkGlm52CompatAppendContentObject`.
fn append_content_object(
    element: &serde_json::Value,
    request: &mut CompatTextRequest,
) -> Result<(), ServingStatus> {
    if let Some(text) = element.get("text") {
        let Some(text) = text.as_str() else {
            return Err(ServingStatus::InvalidArgument);
        };
        return request.append(text);
    }
    if element.get("content").is_some()
        || element.get("file_content").is_some()
        || element.get("file_text").is_some()
        || element.get("data").is_some()
        || element.get("filename").is_some()
        || element.get("file_name").is_some()
        || element.get("name").is_some()
    {
        append_file_object(element, request)?;
        return Ok(());
    }
    if let Some(source) = element.get("source") {
        if source.is_object() && append_file_content_field(source, request)? {
            return Ok(());
        }
    }
    Ok(())
}

/// `SparkGlm52CompatAppendContentToken`.
fn append_content_token(
    content: &serde_json::Value,
    request: &mut CompatTextRequest,
) -> Result<(), ServingStatus> {
    if let Some(text) = content.as_str() {
        return request.append(text);
    }
    if !content.is_array() {
        return Err(ServingStatus::InvalidArgument);
    }
    for element in content.as_array().expect("checked above") {
        if let Some(text) = element.as_str() {
            request.append(text)?;
            continue;
        }
        if !element.is_object() {
            return Err(ServingStatus::InvalidArgument);
        }
        append_content_object(element, request)?;
    }
    Ok(())
}

/// `SparkGlm52CompatGetMessageRole`.
fn message_role(message: &serde_json::Value) -> Result<Role, ServingStatus> {
    match message.get("role").and_then(serde_json::Value::as_str) {
        None | Some("user") => Ok(Role::User),
        Some("system") => Ok(Role::System),
        Some("assistant") => Ok(Role::Assistant),
        Some("tool") => Ok(Role::Tool),
        _ => Err(ServingStatus::InvalidArgument),
    }
}

/// `SparkGlm52CompatAppendMessages`.
fn append_messages(
    messages: &serde_json::Value,
    request: &mut CompatTextRequest,
) -> Result<(), ServingStatus> {
    if !messages.is_array() {
        return Err(ServingStatus::InvalidArgument);
    }
    let messages = messages.as_array().expect("checked above");
    if messages.is_empty() {
        return Err(ServingStatus::InvalidArgument);
    }
    for message in messages {
        if !message.is_object() {
            return Err(ServingStatus::InvalidArgument);
        }
        let Some(content) = message.get("content") else {
            return Err(ServingStatus::InvalidArgument);
        };
        let role = message_role(message)?;
        request.begin_message(role)?;
        append_content_token(content, request)?;
        request.end_message(role)?;
    }
    Ok(())
}

/// `SparkGlm52CompatAppendFilesArray`.
fn append_files_array(
    files: Option<&serde_json::Value>,
    request: &mut CompatTextRequest,
) -> Result<(), ServingStatus> {
    let Some(files) = files else {
        return Ok(());
    };
    if !files.is_array() {
        return Err(ServingStatus::InvalidArgument);
    }
    for file in files.as_array().expect("checked above") {
        if !file.is_object() {
            return Err(ServingStatus::InvalidArgument);
        }
        append_file_object(file, request)?;
    }
    Ok(())
}

/// `SparkGlm52CompatAppendRequestFiles`.
fn append_request_files(
    root: &serde_json::Value,
    request: &mut CompatTextRequest,
) -> Result<(), ServingStatus> {
    append_files_array(root.get("files"), request)?;
    append_files_array(root.get("attachments"), request)
}

/// `SparkGlm52CompatPrepareCommon`.
fn prepare_common(
    root: &serde_json::Value,
    request: &mut CompatTextRequest,
) -> Result<(), ServingStatus> {
    if request.text_capacity() == 0 || !root.is_object() {
        return Err(ServingStatus::InvalidArgument);
    }
    request.writer = ChatTemplateWriter::new(request.text_capacity());
    request.thinking_token_budget = 0;
    request.chat_template_flags = COMPAT_DEFAULT_CHAT_FLAGS;
    if let Some(priority) = read_optional_u32(root, "priority")? {
        request.priority = priority;
    }
    if let Some(output_budget) = read_optional_u32(root, "max_tokens")? {
        request.output_token_budget = output_budget;
    }
    if let Some(output_budget) = read_optional_u32(root, "max_completion_tokens")? {
        request.output_token_budget = output_budget;
    }
    let thinking_budget = read_optional_u32(root, "thinking_budget_tokens")?;
    let thinking_budget_alias = read_optional_u32(root, "thinking_token_budget")?;
    if let (Some(budget), Some(alias)) = (thinking_budget, thinking_budget_alias) {
        if budget != alias {
            return Err(ServingStatus::InvalidArgument);
        }
    }
    let Some(thinking_budget) = thinking_budget.or(thinking_budget_alias) else {
        return Ok(());
    };
    request.thinking_token_budget = thinking_budget;
    request.chat_template_flags = FLAG_ADD_GENERATION_PROMPT;
    if thinking_budget != 0 {
        request.chat_template_flags |= FLAG_ENABLE_THINKING;
    }
    Ok(())
}

/// `SparkGlm52CompatPrepareOpenAiJson`.
pub fn prepare_openai_json(
    json_text: &str,
    request: &mut CompatTextRequest,
) -> Result<(), ServingStatus> {
    let root: serde_json::Value =
        serde_json::from_str(json_text).map_err(|_| ServingStatus::ParseError)?;
    prepare_common(&root, request)?;
    let messages = root.get("messages");
    let prompt = root.get("prompt");
    if let Some(messages) = messages {
        request.begin_template()?;
        append_messages(messages, request)?;
    } else if let Some(prompt) = prompt {
        let Some(text) = prompt.as_str() else {
            return Err(ServingStatus::InvalidArgument);
        };
        request.append(text)?;
    } else {
        return Err(ServingStatus::InvalidArgument);
    }
    append_request_files(&root, request)?;
    if messages.is_some() {
        request.finish_template()?;
    }
    Ok(())
}

/// `SparkGlm52CompatPrepareAnthropicJson`.
pub fn prepare_anthropic_json(
    json_text: &str,
    request: &mut CompatTextRequest,
) -> Result<(), ServingStatus> {
    let root: serde_json::Value =
        serde_json::from_str(json_text).map_err(|_| ServingStatus::ParseError)?;
    prepare_common(&root, request)?;
    request.begin_template()?;
    if let Some(system) = root.get("system") {
        request.begin_message(Role::System)?;
        let Some(text) = system.as_str() else {
            return Err(ServingStatus::InvalidArgument);
        };
        request.append(text)?;
        request.end_message(Role::System)?;
    }
    let Some(messages) = root.get("messages") else {
        return Err(ServingStatus::InvalidArgument);
    };
    append_messages(messages, request)?;
    append_request_files(&root, request)?;
    request.finish_template()?;
    Ok(())
}

/// `SparkGlm52CompatSubmitPrepared`.
fn submit_prepared(
    service: &mut ServiceRuntime,
    compat_request: &CompatTextRequest,
) -> Result<ServiceSubmitResult, ServingStatus> {
    let request = ServiceSubmitTextRequest {
        flags: compat_request.flags,
        priority: compat_request.priority,
        thinking_token_budget: compat_request.thinking_token_budget,
        output_token_budget: compat_request.output_token_budget,
        max_prefill_tokens_per_step: compat_request.max_prefill_tokens_per_step,
        tokenizer_encode_flags: compat_request.tokenizer_encode_flags,
        client_id: compat_request.client_id,
        client_request_id: compat_request.client_request_id,
        sequence_id: compat_request.sequence_id,
        text: compat_request.text().as_bytes(),
    };
    service.submit_text(&request)
}

/// `SparkGlm52CompatSubmitOpenAiJson`.
pub fn submit_openai_json(
    service: &mut ServiceRuntime,
    json_text: &str,
    request: &mut CompatTextRequest,
) -> Result<ServiceSubmitResult, ServingStatus> {
    prepare_openai_json(json_text, request)?;
    submit_prepared(service, request)
}

/// `SparkGlm52CompatSubmitAnthropicJson`.
pub fn submit_anthropic_json(
    service: &mut ServiceRuntime,
    json_text: &str,
    request: &mut CompatTextRequest,
) -> Result<ServiceSubmitResult, ServingStatus> {
    prepare_anthropic_json(json_text, request)?;
    submit_prepared(service, request)
}
