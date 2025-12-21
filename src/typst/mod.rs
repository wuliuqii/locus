//! Typst integration layer.
//!
//! Goal (Phase A): "math-only" rendering.
//! - Compile a snippet containing a single math formula into a Typst frame-like output.
//! - Extract positioned glyphs/shapes from the result.
//! - Feed them into the renderer as vector outlines (tessellated).
//!
//! This module keeps a stable public API (`compile_math_only` returning `MathScene`),
//! while delegating the actual compilation plumbing to `typst::engine`.

pub mod engine;
pub mod math;

// Reusable Typst → Scene extraction/tessellation pipeline.
pub mod render;

// Small ergonomic wrappers intended for demos/teaching scenarios.
// (Not a stable public API yet.)
pub mod demo;

/// A tiny, renderer-friendly output for "math-only" mode.
///
/// In Phase A we only need 2D geometry, no images/gradients/links.
/// Ultimately this should be a list of draw calls (meshes) with transforms.
#[derive(Debug, Default, Clone)]
pub struct MathScene {
    /// A human-readable debug summary of what was compiled.
    ///
    /// For now this is a bridge artifact while we wire up real Typst compilation and
    /// frame extraction. Later this will be replaced/augmented with actual draw items.
    pub debug_summary: String,
}

/// Compile a math-only Typst snippet into a `MathScene`.
///
/// Phase A target formula:
/// `$\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}$`
///
/// Current behavior:
/// - Delegates to `typst::engine::compile_math_only` (currently still a stub compile).
///
/// Next steps (to be implemented in `typst::engine`):
/// 1. Build a Typst `World` (virtual filesystem + font resolver).
/// 2. Compile a minimal document containing only the math expression.
/// 3. Obtain the `Frame` (or equivalent) for the resulting layout.
/// 4. Recursively extract items:
///    - `Group` / `Transform` nesting (matrix multiply order must be correct)
///    - `Text` items (glyph runs) -> outlines via `ttf-parser`
///    - `Shape` items (rules, radicals, fraction bars) -> lyon tessellation
/// 5. Return meshes ready for GPU upload.
pub fn compile_math_only(snippet: &str) -> anyhow::Result<MathScene> {
    let compiled = engine::compile_math_only(snippet)?;
    Ok(MathScene {
        debug_summary: compiled.debug_summary.to_string(),
    })
}
