//! GLM-5.2 chat template rendering — port of `text/chat_template.c`
//! (API declared in
//! `model-families/glm52/include/sparkpipe/spark_glm52_chat_template.h`).
//!
//! All literal fragments, flag semantics, and the reasoning-effort
//! normalization (`"high"`/`"High"` → `High`, everything else including
//! absent → `Max`) mirror the C exactly.
//!
//! Rust-port deviations from the C surface (behavior is otherwise
//! signature-faithful):
//!   - The writer owns a `String` with an explicit byte capacity instead of
//!     writing into a caller buffer with NUL termination. The C reserved one
//!     byte of capacity for the NUL (`next_bytes >= capacity` fails); here
//!     `capacity` is the exact byte budget. The C's `text == NULL`
//!     measure-only mode is just a writer whose capacity you read back via
//!     [`ChatTemplateWriter::len`] after a failed/simulated pass.
//!   - The role is a real enum, so the C "unknown role" rejection in
//!     `SparkGlm52ChatTemplateBeginMessage`/`EndMessage` is unrepresentable
//!     rather than validated.
//!   - `InitializeWriter`'s pre-fill (`text_bytes`) and null-argument checks
//!     have no Rust counterpart; [`ChatTemplateWriter::new`] starts empty.

/// Template flag: append the assistant generation prompt on `finish`
/// (SPARK_GLM52_CHAT_TEMPLATE_FLAG_ADD_GENERATION_PROMPT).
pub const FLAG_ADD_GENERATION_PROMPT: u32 = 0x0000_0001;
/// Template flag: emit the reasoning-effort prefix and open `<think>`
/// (SPARK_GLM52_CHAT_TEMPLATE_FLAG_ENABLE_THINKING).
pub const FLAG_ENABLE_THINKING: u32 = 0x0000_0002;
/// All recognized template flags (SPARK_GLM52_CHAT_TEMPLATE_KNOWN_FLAGS).
pub const KNOWN_FLAGS: u32 = FLAG_ADD_GENERATION_PROMPT | FLAG_ENABLE_THINKING;

const TEMPLATE_PREFIX: &str = "[gMASK]<sop>";
const TEMPLATE_REASONING: &str = "<|system|>Reasoning Effort: ";
const TEMPLATE_SYSTEM: &str = "<|system|>";
const TEMPLATE_USER: &str = "<|user|>";
const TEMPLATE_ASSISTANT: &str = "<|assistant|>";
const TEMPLATE_OBSERVATION: &str = "<|observation|>";
const TEMPLATE_THINK: &str = "<think>";
const TEMPLATE_EMPTY_THINK: &str = "<think></think>";
const TEMPLATE_TOOL_RESPONSE: &str = "<tool_response>";
const TEMPLATE_TOOL_RESPONSE_END: &str = "</tool_response>";
const TEMPLATE_REASONING_HIGH: &str = "High";
const TEMPLATE_REASONING_MAX: &str = "Max";

/// Errors mirroring the `SparkStatus` codes the C entry points return.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ChatTemplateError {
    /// SPARK_STATUS_INVALID_ARGUMENT.
    #[error("invalid argument")]
    InvalidArgument,
    /// SPARK_STATUS_CAPACITY_EXCEEDED.
    #[error("capacity exceeded")]
    CapacityExceeded,
}

/// Message role (SparkGlm52ChatTemplateRole).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// SPARK_GLM52_CHAT_TEMPLATE_ROLE_SYSTEM.
    System,
    /// SPARK_GLM52_CHAT_TEMPLATE_ROLE_USER.
    User,
    /// SPARK_GLM52_CHAT_TEMPLATE_ROLE_ASSISTANT.
    Assistant,
    /// SPARK_GLM52_CHAT_TEMPLATE_ROLE_TOOL.
    Tool,
}

/// Bounded output writer (SparkGlm52ChatTemplateWriter).
///
/// Append-only: every template entry point returns
/// [`ChatTemplateError::CapacityExceeded`] once a fragment would push the
/// rendered text past `capacity`, leaving the already-written prefix intact
/// (matching the C, which keeps `text_bytes` unchanged on failure).
#[derive(Debug, Clone)]
pub struct ChatTemplateWriter {
    text: String,
    capacity: usize,
}

impl ChatTemplateWriter {
    /// Port of `SparkGlm52ChatTemplateInitializeWriter` with an empty buffer.
    pub fn new(capacity: usize) -> Self {
        ChatTemplateWriter { text: String::new(), capacity }
    }

    /// Bytes written so far (`text_bytes` in C).
    pub fn len(&self) -> usize {
        self.text.len()
    }

    /// True when nothing has been written yet.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Byte budget (`text_capacity` in C).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Rendered text so far.
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// Consume the writer, returning the rendered text.
    pub fn into_string(self) -> String {
        self.text
    }

    /// Port of `SparkGlm52ChatTemplateAppend`.
    pub fn append(&mut self, text: &str) -> Result<(), ChatTemplateError> {
        let next_bytes = match self.text.len().checked_add(text.len()) {
            Some(next_bytes) => next_bytes,
            // Mirrors the C UINT32_MAX overflow guard.
            None => return Err(ChatTemplateError::CapacityExceeded),
        };
        if next_bytes > self.capacity {
            return Err(ChatTemplateError::CapacityExceeded);
        }
        self.text.push_str(text);
        Ok(())
    }

    /// Port of `SparkGlm52ChatTemplateBegin`.
    ///
    /// `reasoning_effort` follows the C normalization: `"high"` or `"High"`
    /// renders `High`; any other value (including `None`) renders `Max`.
    pub fn begin(
        &mut self,
        reasoning_effort: Option<&str>,
        flags: u32,
    ) -> Result<(), ChatTemplateError> {
        if (flags & !KNOWN_FLAGS) != 0 {
            return Err(ChatTemplateError::InvalidArgument);
        }
        let effort = match reasoning_effort {
            Some("high") | Some("High") => TEMPLATE_REASONING_HIGH,
            _ => TEMPLATE_REASONING_MAX,
        };
        self.append(TEMPLATE_PREFIX)?;
        if (flags & FLAG_ENABLE_THINKING) == 0 {
            return Ok(());
        }
        self.append(TEMPLATE_REASONING)?;
        self.append(effort)
    }

    /// Port of `SparkGlm52ChatTemplateBeginMessage`.
    pub fn begin_message(&mut self, role: Role) -> Result<(), ChatTemplateError> {
        match role {
            Role::System => self.append(TEMPLATE_SYSTEM),
            Role::User => self.append(TEMPLATE_USER),
            Role::Assistant => {
                self.append(TEMPLATE_ASSISTANT)?;
                self.append(TEMPLATE_EMPTY_THINK)
            }
            Role::Tool => {
                self.append(TEMPLATE_OBSERVATION)?;
                self.append(TEMPLATE_TOOL_RESPONSE)
            }
        }
    }

    /// Port of `SparkGlm52ChatTemplateEndMessage`.
    ///
    /// Only tool messages have a closing fragment; every other role is a
    /// no-op, exactly as in C.
    pub fn end_message(&mut self, role: Role) -> Result<(), ChatTemplateError> {
        match role {
            Role::Tool => self.append(TEMPLATE_TOOL_RESPONSE_END),
            _ => Ok(()),
        }
    }

    /// Port of `SparkGlm52ChatTemplateFinish`.
    pub fn finish(&mut self, flags: u32) -> Result<(), ChatTemplateError> {
        if (flags & !KNOWN_FLAGS) != 0 {
            return Err(ChatTemplateError::InvalidArgument);
        }
        if (flags & FLAG_ADD_GENERATION_PROMPT) == 0 {
            return Ok(());
        }
        self.append(TEMPLATE_ASSISTANT)?;
        if (flags & FLAG_ENABLE_THINKING) != 0 {
            self.append(TEMPLATE_THINK)
        } else {
            self.append(TEMPLATE_EMPTY_THINK)
        }
    }

    /// Port of `SparkGlm52ChatTemplateRenderSimple`.
    ///
    /// Renders prefix, optional system message, user message, and the finish
    /// fragment for a single-turn prompt.
    pub fn render_simple(
        &mut self,
        prompt: &str,
        system_prompt: Option<&str>,
        reasoning_effort: Option<&str>,
        flags: u32,
    ) -> Result<(), ChatTemplateError> {
        self.begin(reasoning_effort, flags)?;
        if let Some(system_prompt) = system_prompt {
            if !system_prompt.is_empty() {
                self.begin_message(Role::System)?;
                self.append(system_prompt)?;
            }
        }
        self.begin_message(Role::User)?;
        self.append(prompt)?;
        self.finish(flags)
    }
}
