//! Simple text layout helpers (Phase A bring-up).
//!
//! This module provides a *very* small text-to-mesh pipeline:
//! - Input: an ASCII string
//! - Output: a single merged `scene::Mesh2D`
//!
//! The layout model is intentionally naive. Kerning plumbing exists but is currently disabled:
//! - One baseline at y = 0 (pt-space)
//! - One font size (pt)
//! - Pen advances using `glyph_h_metrics.advance_width`
//! - Kerning: disabled for the current `ttf-parser` version (we'll re-enable later)
//! - No shaping (ligatures, RTL, complex scripts) (out of scope for this helper)
//!
//! This is meant to validate the critical plumbing for vector text rendering:
//! - font discovery + face resolution
//! - glyph id lookup
//! - glyph outline extraction (ttf-parser)
//! - outline tessellation (lyon)
//! - merging into a single mesh (u16 indices)
//!
//! Once this works, the next step is to replace this with Typst-driven glyph runs.

use crate::font::tessellate::{Affine2x3, TessellateOptions, append_mesh, tessellate_path_to_mesh};
use crate::font::{FontError, FontQuery, FontSystem};
use crate::scene::Mesh2D;

/// Layout options for `layout_ascii_text_to_mesh`.
#[derive(Debug, Clone)]
pub struct TextLayoutOptions {
    /// Font selection query (family list, weight, italic).
    pub font: FontQuery,

    /// Font size in pt.
    pub font_size_pt: f32,

    /// Tessellation tolerance (smaller -> smoother curves, more triangles).
    pub tolerance: f32,

    /// Extra spacing between glyphs in pt (added after each advance).
    pub letter_spacing_pt: f32,

    /// Kerning scale factor.
    ///
    /// - `1.0`: use the font's kerning as-is
    /// - `0.0`: disable kerning without changing code paths
    pub kerning_strength: f32,

    /// If true, replace non-ASCII characters with '?'.
    /// If false, return an error on non-ASCII.
    pub replace_non_ascii: bool,
}

impl Default for TextLayoutOptions {
    fn default() -> Self {
        Self {
            font: FontQuery {
                families: vec![
                    "STIX Two Text".to_string(),
                    "Latin Modern Roman".to_string(),
                    "Linux Libertine".to_string(),
                    "serif".to_string(),
                ],
                weight: 400,
                italic: false,
            },
            font_size_pt: 64.0,
            tolerance: 0.02,
            letter_spacing_pt: 0.0,
            kerning_strength: 1.0,
            replace_non_ascii: true,
        }
    }
}

/// Result of laying out a string into a mesh.
///
/// - `mesh`: merged triangle mesh (filled outlines)
/// - `advance_width_pt`: total advance in pt (useful for centering)
///
/// Note:
/// - This is *advance-based*, not tight-bounds. You still need to compute bounds
///   from the mesh if you want camera framing.
#[derive(Debug, Clone)]
pub struct TextMesh {
    pub mesh: Mesh2D,
    pub advance_width_pt: f32,
}

/// Layout an ASCII string into a single merged mesh (pt-space).
///
/// Coordinate conventions:
/// - Returned mesh positions are in pt, baseline at y=0.
/// - Y axis is *not* flipped here. If your renderer treats Y-up, this matches fonts.
///   If your renderer treats Y-down, flip in camera or transform.
///
/// Behavior:
/// - Each character becomes one glyph.
/// - Pen starts at x=0.
/// - For each glyph:
///   - apply kerning from previous glyph (if available)
///   - extract outline path in font units
///   - tessellate with transform = scale(font_units->pt) + translate(pen_x, 0)
///   - append into merged mesh
///   - pen_x += advance_width_pt + letter_spacing_pt
///
/// Notes:
/// - Space characters may have no outline; we still advance by their metrics.
/// - Kerning is currently disabled for the current `ttf-parser` version.
///   The option plumbing remains so we can re-enable kerning later.
/// - Typst/shaping will handle advanced positioning later.
pub fn layout_ascii_text_to_mesh(
    font_system: &FontSystem,
    text: &str,
    opts: &TextLayoutOptions,
) -> Result<TextMesh, FontError> {
    if opts.font_size_pt <= 0.0 {
        return Err(FontError::Other("font_size_pt must be > 0".to_string()));
    }

    let face = font_system.resolve(&opts.font)?;
    let scale = face.font_units_to_pt_scale(opts.font_size_pt);

    let tess_opts = TessellateOptions {
        tolerance: opts.tolerance.max(0.0005),
        ..Default::default()
    };

    // Kerning plumbing exists via `TextLayoutOptions::kerning_strength`, but kerning is
    // currently disabled for the current `ttf-parser` version.

    let mut merged = Mesh2D::default();
    let mut pen_x_pt = 0.0f32;

    // Previous glyph for kerning (currently unused; kept for future kerning support).
    let mut _prev_gid: Option<ttf_parser::GlyphId> = None;

    for ch in text.chars() {
        let ch = if ch.is_ascii() {
            ch
        } else if opts.replace_non_ascii {
            '?'
        } else {
            return Err(FontError::Other(format!(
                "non-ascii character not supported: {:?}",
                ch
            )));
        };

        // Newlines: minimal line break support. We keep it simple for now:
        // - reset pen
        // - clear prev glyph (no kerning across lines)
        if ch == '\n' {
            pen_x_pt = 0.0;
            _prev_gid = None;
            continue;
        }

        // Resolve glyph + metrics.
        let glyph_id = font_system.glyph_id_for_char(&face, ch)?;
        let _gid = ttf_parser::GlyphId(glyph_id);

        // Kerning is currently disabled for the current `ttf-parser` version.
        // We keep the option plumbing in `TextLayoutOptions` so we can re-enable later:
        // if opts.kerning_strength != 0.0 { ... }

        let hm = font_system.glyph_h_metrics(&face, glyph_id)?;
        let adv_pt = hm.advance_width * scale;

        // Try to extract outline + tessellate.
        // Some glyphs (e.g., space) can have no outline; we still advance.
        match font_system.glyph_outline_path(&face, glyph_id) {
            Ok(outline) => {
                // Transform from font units -> pt and translate by current pen.
                let xf = Affine2x3::scale_translate(scale, pen_x_pt, 0.0);
                let glyph_mesh =
                    tessellate_path_to_mesh(&outline, xf, tess_opts).map_err(FontError::Other)?;

                // Merge into the big mesh.
                append_mesh(&mut merged, &glyph_mesh);
            }
            Err(FontError::MissingGlyph { .. }) => {
                // No outline; skip geometry but still advance.
            }
            Err(e) => return Err(e),
        }

        pen_x_pt += adv_pt + opts.letter_spacing_pt;
        _prev_gid = Some(_gid);
    }

    Ok(TextMesh {
        mesh: merged,
        advance_width_pt: pen_x_pt,
    })
}

/// Build simple horizontal guide meshes for baseline visualization.
///
/// Returns `(baseline_mesh, ascender_mesh, descender_mesh)`.
///
/// - `width_pt`: total width to draw
/// - `thickness_pt`: thickness of each guide line
/// - `asc_pt`: y position of ascender line in pt (typically positive)
/// - `desc_pt`: y position of descender line in pt (typically negative)
///
/// Meshes are centered on x=0 and placed at the specified y.
pub fn baseline_guides_meshes(
    width_pt: f32,
    thickness_pt: f32,
    asc_pt: f32,
    desc_pt: f32,
) -> (Mesh2D, Mesh2D, Mesh2D) {
    fn line_mesh(width: f32, thickness: f32, y: f32) -> Mesh2D {
        let w = width.max(1.0);
        let t = thickness.max(0.5);
        let hw = w * 0.5;
        let ht = t * 0.5;

        Mesh2D {
            positions: vec![[-hw, y - ht], [hw, y - ht], [hw, y + ht], [-hw, y + ht]],
            indices: vec![0, 1, 2, 0, 2, 3],
        }
    }

    let baseline = line_mesh(width_pt, thickness_pt, 0.0);
    let ascender = line_mesh(width_pt, thickness_pt, asc_pt);
    let descender = line_mesh(width_pt, thickness_pt, desc_pt);

    (baseline, ascender, descender)
}
