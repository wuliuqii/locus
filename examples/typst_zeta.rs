//! Example binary: windowed Typst ζ demo.
//!
//! This example is intentionally thin:
//! - It uses the library app runner (`run_with_builder`) so it doesn't duplicate the winit loop.
//! - It reuses the shared state-only implementation in `examples/typst_zeta_state.rs`.
//!
//! Run:
//! - `cargo run --example typst_zeta`
//!
//! Notes:
//! - The current state renders only extracted line shapes (e.g. fraction bars) as thin quads.
//! - Text/glyph extraction is the next step.

fn main() -> anyhow::Result<()> {
    // Keep logging setup in the example binary (library stays unopinionated).
    env_logger::init();

    locus::render::app::run_with_builder(
        locus::render::app::AppConfig {
            title: "locus: typst_zeta".to_string(),
            ..Default::default()
        },
        |window| async move { typst_zeta_state::State::new(window).await },
    )
}

// Pull in the shared state-only module from a sibling file.
// This keeps the demo logic in one place while letting this example remain a tiny wrapper.
#[path = "typst_zeta_state.rs"]
mod typst_zeta_state;
