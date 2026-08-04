//! Tests for the chat template port (`text/chat_template.c`).
//!
//! Expected strings are taken from the C tree's assertions in
//! `tests/test_glm52_compat_api.c` (SPARK_TEST_GLM52_CHAT_BEGIN /
//! SPARK_TEST_GLM52_CHAT_END and the no-think variants).

use spark_text::chat_template::{
    ChatTemplateError, ChatTemplateWriter, Role, FLAG_ADD_GENERATION_PROMPT, FLAG_ENABLE_THINKING,
};

const CAPACITY: usize = 4096;

const CHAT_BEGIN: &str = "[gMASK]<sop><|system|>Reasoning Effort: Max";
const CHAT_END: &str = "<|assistant|><think>";
const NO_THINK_CHAT_BEGIN: &str = "[gMASK]<sop>";
const NO_THINK_CHAT_END: &str = "<|assistant|><think></think>";

#[test]
fn begin_without_flags_emits_prefix_only() {
    let mut writer = ChatTemplateWriter::new(CAPACITY);
    writer.begin(None, 0).unwrap();
    assert_eq!(writer.as_str(), NO_THINK_CHAT_BEGIN);
}

#[test]
fn begin_with_thinking_defaults_to_max_reasoning() {
    let mut writer = ChatTemplateWriter::new(CAPACITY);
    writer.begin(None, FLAG_ENABLE_THINKING).unwrap();
    assert_eq!(writer.as_str(), CHAT_BEGIN);
}

#[test]
fn begin_normalizes_high_reasoning_effort() {
    // C: "high" and "High" both render "High" (tests/test_glm52_compat_api.c
    // asserts the literal "[gMASK]<sop><|system|>Reasoning Effort: High").
    for effort in ["high", "High"] {
        let mut writer = ChatTemplateWriter::new(CAPACITY);
        writer.begin(Some(effort), FLAG_ENABLE_THINKING).unwrap();
        assert_eq!(writer.as_str(), "[gMASK]<sop><|system|>Reasoning Effort: High");
    }
}

#[test]
fn begin_maps_any_other_effort_to_max() {
    for effort in [Some("low"), Some("medium"), Some(""), Some("MAX")] {
        let mut writer = ChatTemplateWriter::new(CAPACITY);
        writer.begin(effort, FLAG_ENABLE_THINKING).unwrap();
        assert_eq!(writer.as_str(), CHAT_BEGIN);
    }
}

#[test]
fn begin_rejects_unknown_flags() {
    let mut writer = ChatTemplateWriter::new(CAPACITY);
    assert_eq!(writer.begin(None, 0x8000_0000), Err(ChatTemplateError::InvalidArgument));
    assert!(writer.is_empty());
}

#[test]
fn begin_message_emits_role_fragments() {
    let cases: [(Role, &str); 4] = [
        (Role::System, "<|system|>"),
        (Role::User, "<|user|>"),
        (Role::Assistant, "<|assistant|><think></think>"),
        (Role::Tool, "<|observation|><tool_response>"),
    ];
    for (role, expected) in cases {
        let mut writer = ChatTemplateWriter::new(CAPACITY);
        writer.begin_message(role).unwrap();
        assert_eq!(writer.as_str(), expected);
    }
}

#[test]
fn end_message_closes_only_tool_messages() {
    // Tool role closes the tool response.
    let mut writer = ChatTemplateWriter::new(CAPACITY);
    writer.begin_message(Role::Tool).unwrap();
    writer.append("result").unwrap();
    writer.end_message(Role::Tool).unwrap();
    assert_eq!(writer.as_str(), "<|observation|><tool_response>result</tool_response>");

    // Every other role is a no-op (C returns SPARK_STATUS_OK unchanged).
    for role in [Role::System, Role::User, Role::Assistant] {
        let mut writer = ChatTemplateWriter::new(CAPACITY);
        writer.end_message(role).unwrap();
        assert_eq!(writer.as_str(), "");
    }
}

#[test]
fn finish_without_generation_prompt_is_noop() {
    let mut writer = ChatTemplateWriter::new(CAPACITY);
    writer.finish(0).unwrap();
    writer.finish(FLAG_ENABLE_THINKING).unwrap();
    assert!(writer.is_empty());
}

#[test]
fn finish_appends_generation_prompt_variants() {
    let mut writer = ChatTemplateWriter::new(CAPACITY);
    writer.finish(FLAG_ADD_GENERATION_PROMPT | FLAG_ENABLE_THINKING).unwrap();
    assert_eq!(writer.as_str(), CHAT_END);

    let mut writer = ChatTemplateWriter::new(CAPACITY);
    writer.finish(FLAG_ADD_GENERATION_PROMPT).unwrap();
    assert_eq!(writer.as_str(), NO_THINK_CHAT_END);
}

#[test]
fn finish_rejects_unknown_flags() {
    let mut writer = ChatTemplateWriter::new(CAPACITY);
    assert_eq!(
        writer.finish(FLAG_ADD_GENERATION_PROMPT | 0x0000_0100),
        Err(ChatTemplateError::InvalidArgument)
    );
    assert!(writer.is_empty());
}

#[test]
fn render_simple_without_system_prompt_or_thinking() {
    // Expected layout from tests/test_glm52_compat_api.c
    // (SPARK_TEST_GLM52_NO_THINK_CHAT_BEGIN .. NO_THINK_CHAT_END).
    let mut writer = ChatTemplateWriter::new(CAPACITY);
    writer.render_simple("Answer.", None, None, FLAG_ADD_GENERATION_PROMPT).unwrap();
    assert_eq!(writer.as_str(), format!("{NO_THINK_CHAT_BEGIN}<|user|>Answer.{NO_THINK_CHAT_END}"));
}

#[test]
fn render_simple_with_system_prompt_and_thinking() {
    // Expected layout from tests/test_glm52_compat_api.c (OpenAI chat case).
    let mut writer = ChatTemplateWriter::new(CAPACITY);
    writer
        .render_simple(
            "Read this C code.",
            Some("You are terse."),
            None,
            FLAG_ADD_GENERATION_PROMPT | FLAG_ENABLE_THINKING,
        )
        .unwrap();
    assert_eq!(
        writer.as_str(),
        format!("{CHAT_BEGIN}<|system|>You are terse.<|user|>Read this C code.{CHAT_END}")
    );
}

#[test]
fn render_simple_skips_empty_system_prompt() {
    // C guards on system_prompt_bytes != 0; Some("") must render like None.
    let flags = FLAG_ADD_GENERATION_PROMPT | FLAG_ENABLE_THINKING;
    let mut with_empty = ChatTemplateWriter::new(CAPACITY);
    with_empty.render_simple("Hi", Some(""), None, flags).unwrap();
    let mut with_none = ChatTemplateWriter::new(CAPACITY);
    with_none.render_simple("Hi", None, None, flags).unwrap();
    assert_eq!(with_empty.as_str(), with_none.as_str());
}

#[test]
fn render_simple_uses_high_reasoning_effort() {
    let mut writer = ChatTemplateWriter::new(CAPACITY);
    writer.render_simple("Hi", None, Some("high"), FLAG_ENABLE_THINKING).unwrap();
    assert_eq!(writer.as_str(), "[gMASK]<sop><|system|>Reasoning Effort: High<|user|>Hi");
}

#[test]
fn append_reports_capacity_exceeded_and_keeps_prefix() {
    let mut writer = ChatTemplateWriter::new(5);
    writer.append("hello").unwrap();
    assert_eq!(writer.append("!"), Err(ChatTemplateError::CapacityExceeded));
    // Failed append leaves the written prefix intact (C keeps text_bytes).
    assert_eq!(writer.as_str(), "hello");
}

#[test]
fn writer_capacity_is_exact_byte_budget() {
    // Rust capacity is exact (no NUL reservation as in the C writer).
    let mut writer = ChatTemplateWriter::new(5);
    writer.append("hello").unwrap();
    assert_eq!(writer.len(), 5);
    assert_eq!(writer.capacity(), 5);
}

#[test]
fn render_simple_reports_capacity_exceeded() {
    let mut writer = ChatTemplateWriter::new(8);
    assert_eq!(
        writer.render_simple("Hello world", None, None, 0),
        Err(ChatTemplateError::CapacityExceeded)
    );
    // The 12-byte prefix fragment exceeds the budget; nothing is written.
    assert!(writer.is_empty());
}
