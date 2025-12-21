//! `locus` library crate root.
//!
//! This crate is intended to be used primarily as a **library**. The binary target (if any)
//! should stay thin and call into these exported entrypoints.
//!
//! Public API philosophy (for now):
//! - Keep modules public so downstream apps can assemble their own pipelines.
//! - Provide a couple of stable entrypoints (`run_app`, `compile_math_only`) that mirror
//!   the current demos and are useful for integration tests / bring-up.

pub mod anim;
pub mod font;
pub mod render;
pub mod scene;
pub mod typst;

/// Run the current winit/wgpu demo application (default config).
///
/// This is the same entrypoint previously used by `main.rs`, but exposed from the library
/// so downstream binaries can stay minimal.
///
/// Note: This function does **not** initialize logging; callers can decide their own
/// logging/tracing setup.
pub fn run_app() -> anyhow::Result<()> {
    render::app::run()
}

/// Compile a math-only Typst snippet (Phase A helper).
///
/// This is a convenience wrapper around `typst::compile_math_only(...)` to make it easy
/// for callers to just depend on the `locus` library.
///
/// If you want more control (fonts/world/etc.), use the APIs under `typst::engine`.
pub fn compile_math_only(snippet: &str) -> anyhow::Result<typst::MathScene> {
    typst::compile_math_only(snippet)
}
