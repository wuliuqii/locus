//! Example binary: vector text bring-up (ASCII) + baseline/ascender/descender guides.
//!
//! This is intentionally an `examples/` target so the main library surface stays clean.
//!
//! Run:
//! - `cargo run --example vector_text_ascii`
//!
//! This file is a thin wrapper that uses the shared state-only module:
//! - `examples/vector_text_ascii_state.rs`

fn main() -> anyhow::Result<()> {
    // Keep logging setup in the example binary.
    env_logger::init();

    locus::render::app::run_with_builder(
        locus::render::app::AppConfig {
            title: "locus: vector_text_ascii".to_string(),
            ..Default::default()
        },
        |window| async move {
            // Reuse the state-only implementation (no duplicated winit/wgpu boilerplate).
            vector_text_ascii_state::State::new(window).await
        },
    )
}

// Pull in the shared state-only module from a sibling file.
// This keeps the demo logic in one place while letting this example remain a tiny wrapper.
#[path = "vector_text_ascii_state.rs"]
mod vector_text_ascii_state;
