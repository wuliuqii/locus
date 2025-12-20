//! Font face resolution (placeholder).
//!
//! This module will be responsible for selecting a concrete font face given a
//! `FontQuery` (families/weight/italic), usually by consulting a `fontdb::Database`
//! plus project-specific fallback rules.
//!
//! Current status:
//! - Stubbed out to satisfy the module tree during modularization.
//!
//! Planned responsibilities:
//! - Resolve preferred families in order, falling back to generic families.
//! - Match weight/style as closely as possible.
//! - Provide a last-resort embedded fallback (e.g. Noto) if system fonts are missing.
//! - Produce a `ResolvedFace` containing bytes + face index for `ttf-parser`.
//!
//! Notes on design:
//! - Keep resolution policy here, not in `outline` extraction.
//! - Keep caching hooks here (e.g. memoize queries -> face ids).

use crate::font::{FontError, FontQuery, ResolvedFace};

/// Resolve a font face for the given query.
///
/// Placeholder implementation:
/// - Always returns `FontError::ResolveFailed`.
///
/// Intended real implementation:
/// - Consult a `fontdb::Database` (owned by `FontSystem` / `FontDatabase`).
/// - Return `ResolvedFace { bytes, index, face_id }`.
pub fn resolve_face(_query: &FontQuery) -> Result<ResolvedFace, FontError> {
    Err(FontError::ResolveFailed(_query.clone()))
}

/// Pick a reasonable default query for math rendering.
///
/// This is a helper for Phase A (math-only) Typst integration. Once Typst is
/// wired up properly, Typst itself will drive font selection; this becomes a
/// fallback.
pub fn default_math_query() -> FontQuery {
    FontQuery {
        families: vec![
            // Prefer high-quality serif math-friendly fonts if present.
            "STIX Two Text".to_string(),
            "STIX Two Math".to_string(),
            "Latin Modern Roman".to_string(),
            "Linux Libertine".to_string(),
            // Generic fallback family (policy handled in real resolver).
            "serif".to_string(),
        ],
        weight: 400,
        italic: false,
    }
}
