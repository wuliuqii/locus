//! Glyph outline extraction (placeholder).
//!
//! This module will contain the core "vector text rendering" plumbing:
//! - Parse a font face (TTF/OTF/TTC/OTC) using `ttf-parser`
//! - Extract glyph outlines via the `OutlineBuilder` callbacks
//! - Convert those outlines into a `lyon::path::Path`
//! - Optionally normalize/flip coordinate systems as needed (font units vs screen units)
//!
//! Current status:
//! - Stubbed out to satisfy the module tree during the modularization phase.
//!
//! Intended API direction:
//! - Provide a function that, given `ResolvedFace` (bytes + face index) and a glyph id,
//!   returns a `lyon::path::Path` in *font units*.
//! - The caller applies scaling (font size) and positioning transforms.

use crate::font::{FontError, ResolvedFace};

/// A small container for metrics you often need when placing glyph outlines.
#[derive(Debug, Copy, Clone, Default)]
pub struct GlyphMetrics {
    /// Advance width in font units.
    pub advance_width: f32,
    /// Left side bearing in font units (optional; depends on font data).
    pub left_side_bearing: f32,
    /// Bounding box in font units (min_x, min_y, max_x, max_y).
    pub bbox: Option<[f32; 4]>,
}

/// Extract the outline for a glyph as a lyon path.
///
/// Returned path is in font coordinate units (typically "font units per em").
/// The Y axis direction depends on the font coordinate system; most font outlines
/// are defined with Y+ up. You may choose to flip Y later to match your screen
/// coordinate convention.
///
/// Placeholder implementation:
/// - Always returns `FontError::Other`.
pub fn glyph_outline_path(
    _face: &ResolvedFace,
    _glyph_id: u16,
) -> Result<lyon::path::Path, FontError> {
    Err(FontError::Other(
        "font::outline::glyph_outline_path is not implemented yet".to_string(),
    ))
}

/// Extract basic glyph metrics useful for layout and positioning.
///
/// Placeholder implementation:
/// - Always returns default metrics.
pub fn glyph_metrics(_face: &ResolvedFace, _glyph_id: u16) -> Result<GlyphMetrics, FontError> {
    Ok(GlyphMetrics::default())
}

/// A helper to apply a uniform scale to a lyon path.
///
/// This is a convenience for callers that want to scale from font units to world units.
/// In a real implementation, you might instead apply transforms at tessellation time,
/// or bake transforms into vertex generation.
///
/// Placeholder implementation:
/// - Returns the input path unchanged.
pub fn scale_path(path: &lyon::path::Path, _scale: f32) -> lyon::path::Path {
    // TODO: implement path transformation. Lyon supports transforming paths via iterators.
    path.clone()
}

/// A helper to apply a translation to a lyon path.
///
/// Placeholder implementation:
/// - Returns the input path unchanged.
pub fn translate_path(path: &lyon::path::Path, _dx: f32, _dy: f32) -> lyon::path::Path {
    // TODO: implement path transformation.
    path.clone()
}

/// A helper to flip Y coordinates in a lyon path (common when converting between
/// font coordinate space and screen space).
///
/// Placeholder implementation:
/// - Returns the input path unchanged.
pub fn flip_y_path(path: &lyon::path::Path) -> lyon::path::Path {
    // TODO: implement path transformation.
    path.clone()
}

/// Notes for the real implementation (design sketch):
///
/// - Use `ttf_parser::Face::parse(&bytes, index)`
/// - Implement `ttf_parser::OutlineBuilder`:
///   - `move_to(x, y)`
///   - `line_to(x, y)`
///   - `quad_to(x1, y1, x, y)`
///   - `curve_to(x1, y1, x2, y2, x, y)`
///   - `close()`
/// - Convert these events into a `lyon::path::Builder`
///   - `begin(point)`
///   - `line_to(point)`
///   - `quadratic_bezier_to(ctrl, to)`
///   - `cubic_bezier_to(ctrl1, ctrl2, to)`
///   - `close()`
/// - Return the built `Path`.
///
/// Pitfalls:
/// - Correctly handle multiple contours per glyph (multiple `begin/close` segments).
/// - Be careful about coordinate orientation (Y-up vs Y-down).
/// - Handle missing outlines (e.g. space) gracefully.
/// - Cache parsed faces and glyph paths aggressively for performance.
#[allow(dead_code)]
const _OUTLINE_IMPL_NOTES: () = ();
