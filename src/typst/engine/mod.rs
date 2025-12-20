//! Typst engine integration (Phase A: math-only).
//!
//! This module is responsible for:
//! - creating a minimal Typst `World` (sources + fonts + packages)
//! - compiling an in-memory Typst snippet into a Typst document/frame tree
//! - exposing stable entrypoints for the renderer layer
//!
//! Current status:
//! - We build a `TypstWorld` backed by in-memory source + system fonts.
//! - We compile a real `typst::layout::PagedDocument`.
//!
//! Next steps:
//! - Extract frames into renderer-friendly draw items (group/transform recursion + glyph outlines).
//! - Add caching/incremental recompilation.
//! - Add package/file support as needed for larger projects.

pub mod extract;
pub mod world;

use ecow::EcoString;

use crate::typst::math::ZETA_VALIDATION_FORMULA;

/// A small "compiled math" artifact for Phase A.
///
/// For now, we keep both:
/// - a debug summary (useful for logging/bring-up)
/// - the real compiled `PagedDocument` (for frame extraction next)
#[derive(Debug, Clone)]
pub struct CompiledMath {
    pub debug_summary: EcoString,
    pub document: typst::layout::PagedDocument,
}

/// Compile a single math-only snippet into a real Typst `PagedDocument`.
///
/// Notes:
/// - This uses our custom `TypstWorld` (system fonts + in-memory source).
/// - The snippet should already include math mode `$...$` if desired.
pub fn compile_math_only(snippet: &str) -> anyhow::Result<CompiledMath> {
    let document = world::compile_math_paged(snippet)?;

    Ok(CompiledMath {
        debug_summary: EcoString::from(format!(
            "typst compiled ok: pages={}",
            document.pages.len()
        )),
        document,
    })
}

/// Convenience: compile the Phase A validation formula.
///
/// This is the canonical test we want to pass once real compilation + extraction exists:
/// `$\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}$`
pub fn compile_zeta_validation() -> anyhow::Result<CompiledMath> {
    compile_math_only(ZETA_VALIDATION_FORMULA)
}
