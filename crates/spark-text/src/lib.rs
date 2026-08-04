//! `spark-text`: tokenizer and prompt pipeline, ported from the C tree's
//! `text/`.
//!
//! - [`tokenizer`]: BPE tokenizer (port of `text/tokenizer.c`) — open-hash
//!   symbol table + merge min-heap, ported structurally (the tuned table is
//!   the point; a std HashMap would not beat it)
//! - [`chat_template`]: chat template rendering (port of
//!   `text/chat_template.c`)
//! - [`prompt_pipeline`]: prompt assembly (port of `text/prompt_pipeline.c`
//!   and `text/prompt.c`)

pub mod chat_template;
pub mod prompt_pipeline;
pub mod tokenizer;
