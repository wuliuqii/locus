//! Math-only (Phase A) Typst integration.
//!
//! Phase A goal:
//! - Compile a single math expression (no full page / no paragraph layout) into a
//!   renderer-friendly representation.
//! - Ultimately: a set of positioned glyph outlines + simple shapes (fraction bars,
//!   radicals, etc.) with correct nested transforms.
//!
//! Current status:
//! - This is a placeholder module to support modularization.
//! - It provides stable APIs and data structures that the renderer can depend on,
//!   while we incrementally implement the real Typst compilation + extraction.
//!
//! Next implementation steps (planned):
//! 1. Implement a minimal Typst `World`:
//!    - virtual filesystem for the source
//!    - font resolver (using `fontdb` + `ttf-parser`)
//! 2. Compile a minimal document that contains only the requested math.
//! 3. Extract the resulting `Frame`/display list into `MathDrawItem`s.
//! 4. Ensure transform accumulation is correct (parent * local, respecting Typst conventions).
//! 5. Convert glyph outlines to `lyon::path::Path`, tessellate to meshes, and return them.

use crate::typst::MathScene;

/// The canonical validation target for Phase A.
pub const ZETA_VALIDATION_FORMULA: &str = r#"$\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}$"#;

/// A minimal description of a "math-only request".
///
/// This is intentionally tiny: just the formula source.
/// Later you can extend it with:
/// - font size / style presets
/// - DPI / pixel ratio
/// - color theme
/// - alignment/anchoring options
#[derive(Debug, Clone)]
pub struct MathRequest {
    pub formula: String,
}

impl MathRequest {
    /// Create a request for the Phase A validation formula.
    pub fn zeta_validation() -> Self {
        Self {
            formula: ZETA_VALIDATION_FORMULA.to_owned(),
        }
    }
}

/// Compile a single math formula into a `MathScene`.
///
/// Placeholder behavior:
/// - Returns a `MathScene` with a debug summary.
///
/// Intended future behavior:
/// - Returns meshes (vector outlines) and/or draw items to be uploaded to GPU.
///
/// Notes:
/// - You can call this from the render layer to wire the app "end-to-end" early.
/// - The real Typst compilation will be filled in once the font resolver and Typst
///   world plumbing are implemented.
pub fn compile_math_formula(req: &MathRequest) -> anyhow::Result<MathScene> {
    // Keep this string format stable so downstream logs/tests can depend on it
    // while we replace the internals with real Typst compilation.
    let snippet = &req.formula;

    Ok(MathScene {
        debug_summary: format!(
            "typst::math::compile_math_formula stub. formula_len={}, formula={:?}",
            snippet.len(),
            snippet
        ),
    })
}
