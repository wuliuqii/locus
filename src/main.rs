//! Thin binary wrapper for local development.
//!
//! Project direction: `locus` is primarily a **library**.
//! This binary exists only to preserve the convenience of `cargo run`.
//!
//! Default behavior:
//! - Run the vector text ASCII demo via the library runner (`run_with_builder`).
//! - The actual demo state lives in `examples/vector_text_ascii_state.rs` and is included
//!   here to avoid duplicating event loop/renderer boilerplate in `main.rs`.
//!
//! Run:
//! - `cargo run`

fn main() -> anyhow::Result<()> {
    // Keep logging setup in the binary so the library remains unopinionated.
    env_logger::init();

    locus::render::app::run_with_builder(
        locus::render::app::AppConfig {
            title: "locus: vector_text_ascii (default)".to_string(),
            ..Default::default()
        },
        |window| async move { vector_text_ascii_state::State::new(window).await },
    )
}

// Include the shared state-only demo implementation.
// This file must not define `fn main()`.
#[path = "../examples/vector_text_ascii_state.rs"]
mod vector_text_ascii_state;
