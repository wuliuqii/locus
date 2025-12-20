//! Font module root.
//!
//! This project renders text via **vector glyph outlines**:
//! - Discover fonts (system + embedded fallbacks).
//! - Resolve a requested family/weight/style to a concrete font face.
//! - Extract glyph outlines (move_to/line_to/quad_to/curve_to) from TTF/OTF.
//! - Convert outlines to `lyon::path::Path` and tessellate into triangles.
//!
//! This file contains a working baseline implementation for:
//! - loading system fonts via `fontdb`
//! - resolving a face (family/weight/italic) with fallback
//! - extracting a single glyph outline as a `lyon::path::Path` via `ttf-parser`
//! - extracting minimal glyph metrics (advance width) for simple string layout
//!
//! Next steps:
//! - Cache file bytes per face source (avoid re-reading).
//! - Cache parsed `ttf_parser::Face`.
//! - Add embedded fallback fonts for portability.

pub mod db;
pub mod outline;
pub mod resolve;
pub mod tessellate;
pub mod text;

use std::{fs, path::PathBuf, sync::Arc};

use fontdb::{Database, Family, ID, Query, Source, Style, Weight};
use lyon::math::point;
use lyon::path::Path;

/// A stable identifier for a selected font face within our font system.
///
/// Internally we keep `fontdb::ID` directly (it's Copy and hashable).
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct FontFaceId(pub ID);

/// Simplified font style selection.
///
/// Note:
/// - Typst has its own model for font selection (family/variant/weight/stretch/style).
/// - We'll translate from Typst's requests into this struct.
#[derive(Debug, Clone, Default)]
pub struct FontQuery {
    /// Preferred font family names, in priority order.
    /// Example: `["Linux Libertine", "Times New Roman", "serif"]`
    pub families: Vec<String>,

    /// Weight in CSS-ish terms (100..900). Common:
    /// - 400 = regular
    /// - 700 = bold
    pub weight: u16,

    /// Italic / oblique.
    pub italic: bool,
}

/// Basic vertical metrics needed for consistent baseline alignment.
///
/// Values are in **font units** (units-per-em).
#[derive(Debug, Copy, Clone)]
pub struct FontVMetrics {
    pub units_per_em: f32,
    pub ascender: f32,
    pub descender: f32,
    pub line_gap: f32,
}

impl FontVMetrics {
    pub fn default_for_upm(units_per_em: f32) -> Self {
        // Conservative defaults when the face doesn't provide metrics.
        // asc/desc roughly match many fonts (but real values should come from OS/2/hhea).
        Self {
            units_per_em,
            ascender: 0.8 * units_per_em,
            descender: -0.2 * units_per_em,
            line_gap: 0.0,
        }
    }
}

/// Minimal per-glyph metrics needed for naive string layout.
///
/// All values are in **font units** (units-per-em). Convert to pt by multiplying
/// by `ResolvedFace::font_units_to_pt_scale(font_size_pt)`.
#[derive(Debug, Copy, Clone, Default)]
pub struct GlyphHMetrics {
    /// Advance width in font units.
    pub advance_width: f32,
}

/// A resolved face plus enough information to access its bytes.
///
/// - `bytes` contains the full font file/collection.
/// - `index` selects the face within the collection.
/// - `v_metrics` provides units-per-em and baseline-related metrics (font units).
#[derive(Debug, Clone)]
pub struct ResolvedFace {
    pub face_id: FontFaceId,
    pub bytes: Arc<[u8]>,
    pub index: u32,
    pub v_metrics: FontVMetrics,
}

impl ResolvedFace {
    /// Returns units-per-em for this face.
    #[inline]
    pub fn units_per_em(&self) -> f32 {
        self.v_metrics.units_per_em
    }

    /// Convert a font-size in pt into a scale factor that maps font units -> pt.
    #[inline]
    pub fn font_units_to_pt_scale(&self, font_size_pt: f32) -> f32 {
        font_size_pt / self.v_metrics.units_per_em
    }
}

/// Errors produced by the font subsystem.
#[derive(thiserror::Error, Debug)]
pub enum FontError {
    #[error("no fonts found on this system and no embedded fallback configured")]
    NoFontsAvailable,

    #[error("failed to resolve a font face for query: {0:?}")]
    ResolveFailed(FontQuery),

    #[error("font face has no file-backed source (embedded sources not implemented yet)")]
    NonFileBackedSource,

    #[error("failed to read font file from disk: {0}")]
    ReadFailed(String),

    #[error("failed to parse font face")]
    ParseFailed,

    #[error("glyph outline not found for glyph id {glyph_id}")]
    MissingGlyph { glyph_id: u16 },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("other: {0}")]
    Other(String),
}

/// The primary entrypoint to the font system.
///
/// Responsibilities (current):
/// - Own a `fontdb::Database` and load system fonts.
/// - Resolve faces based on `FontQuery`.
/// - Extract a glyph outline as a lyon `Path` via `ttf-parser`.
pub struct FontSystem {
    db: Database,
}

impl FontSystem {
    /// Create a new font system and load system fonts.
    pub fn new() -> Result<Self, FontError> {
        let mut db = Database::new();
        db.load_system_fonts();

        // `faces()` is an iterator; check emptiness by `next()`.
        if db.faces().next().is_none() {
            return Err(FontError::NoFontsAvailable);
        }

        Ok(Self { db })
    }

    /// Resolve a `FontQuery` to a concrete face.
    ///
    /// Resolution strategy (minimal, but works):
    /// - Try each named family in order with the requested weight/style.
    /// - Fall back to generic `serif`/`sans-serif`/`monospace` if present in the list.
    /// - If still not found, fall back to "first face in database".
    pub fn resolve(&self, query: &FontQuery) -> Result<ResolvedFace, FontError> {
        if self.db.faces().next().is_none() {
            return Err(FontError::NoFontsAvailable);
        }

        let style = if query.italic {
            Style::Italic
        } else {
            Style::Normal
        };

        // Map CSS-ish weight (100..900) into fontdb's Weight.
        let weight = Weight(query.weight.clamp(1, 1000));

        // Build family list for fontdb query.
        let mut families: Vec<Family<'_>> = Vec::new();
        for f in &query.families {
            let s = f.trim();
            if s.eq_ignore_ascii_case("serif") {
                families.push(Family::Serif);
            } else if s.eq_ignore_ascii_case("sans-serif") || s.eq_ignore_ascii_case("sans") {
                families.push(Family::SansSerif);
            } else if s.eq_ignore_ascii_case("monospace") || s.eq_ignore_ascii_case("mono") {
                families.push(Family::Monospace);
            } else if !s.is_empty() {
                families.push(Family::Name(s));
            }
        }

        let q = Query {
            families: &families,
            weight,
            style,
            stretch: fontdb::Stretch::Normal,
        };

        let id = self
            .db
            .query(&q)
            .or_else(|| {
                // If we didn't provide any family names, or none matched:
                // try a sane default: serif.
                let fallback_families = [Family::Serif];
                self.db.query(&Query {
                    families: &fallback_families,
                    weight,
                    style,
                    stretch: fontdb::Stretch::Normal,
                })
            })
            .unwrap_or_else(|| self.db.faces().next().unwrap().id);

        let face = self
            .db
            .face(id)
            .ok_or_else(|| FontError::ResolveFailed(query.clone()))?;

        let (path, index) = match &face.source {
            Source::File(p) => (p.to_path_buf(), face.index),
            _ => return Err(FontError::NonFileBackedSource),
        };

        let bytes = read_font_bytes(&path)?;

        // Parse metrics once here (cheap) so callers can scale correctly.
        let parsed = ttf_parser::Face::parse(&bytes, index).map_err(|_| FontError::ParseFailed)?;

        // `ttf-parser` returns `u16` (not Option) for units-per-em in this version.
        let units_per_em = parsed.units_per_em() as f32;

        // Vertical metrics:
        // Prefer `typographic_*` accessors when available (OS/2), otherwise fall back to
        // the classic ascender/descender/line_gap accessors (hhea).
        //
        // Note: These are in font units and are typically i16-ish.
        let asc = parsed
            .typographic_ascender()
            .unwrap_or_else(|| parsed.ascender()) as f32;
        let desc = parsed
            .typographic_descender()
            .unwrap_or_else(|| parsed.descender()) as f32;
        let gap = parsed
            .typographic_line_gap()
            .unwrap_or_else(|| parsed.line_gap()) as f32;

        let v_metrics = FontVMetrics {
            units_per_em,
            ascender: asc,
            descender: desc,
            line_gap: gap,
        };

        Ok(ResolvedFace {
            face_id: FontFaceId(id),
            bytes,
            index,
            v_metrics,
        })
    }

    /// Extract a glyph outline as a lyon `Path`.
    ///
    /// The returned path is in **font units** (units-per-em). Caller should scale it
    /// to pt using: `scale = font_size_pt / units_per_em`.
    pub fn glyph_outline_path(
        &self,
        face: &ResolvedFace,
        glyph_id: u16,
    ) -> Result<Path, FontError> {
        let parsed =
            ttf_parser::Face::parse(&face.bytes, face.index).map_err(|_| FontError::ParseFailed)?;
        let gid = ttf_parser::GlyphId(glyph_id);

        let mut builder = LyonOutlineBuilder::new();
        // `outline_glyph` returns an Option<bbox>. None means "no outline".
        let bbox = parsed.outline_glyph(gid, &mut builder);

        if bbox.is_none() {
            return Err(FontError::MissingGlyph { glyph_id });
        }

        Ok(builder.build())
    }

    /// Convenience: find the glyph id for a Unicode codepoint.
    pub fn glyph_id_for_char(&self, face: &ResolvedFace, ch: char) -> Result<u16, FontError> {
        let parsed =
            ttf_parser::Face::parse(&face.bytes, face.index).map_err(|_| FontError::ParseFailed)?;
        let gid = parsed.glyph_index(ch).ok_or(FontError::Other(format!(
            "glyph not found for char {:?}",
            ch
        )))?;
        Ok(gid.0)
    }

    /// Get horizontal glyph metrics (advance width) for a glyph id.
    ///
    /// This enables simple string layout:
    /// - pen_x starts at 0 on the baseline
    /// - for each glyph:
    ///   - place its outline translated by (pen_x, 0)
    ///   - advance pen_x by `advance_width`
    ///
    /// The returned advance width is in **font units**.
    pub fn glyph_h_metrics(
        &self,
        face: &ResolvedFace,
        glyph_id: u16,
    ) -> Result<GlyphHMetrics, FontError> {
        let parsed =
            ttf_parser::Face::parse(&face.bytes, face.index).map_err(|_| FontError::ParseFailed)?;
        let gid = ttf_parser::GlyphId(glyph_id);

        // `glyph_hor_advance` returns Option<u16> (None for missing metrics).
        let adv = parsed.glyph_hor_advance(gid).ok_or_else(|| {
            FontError::Other(format!("missing hor advance for glyph id {}", glyph_id))
        })? as f32;

        Ok(GlyphHMetrics { advance_width: adv })
    }
}

fn read_font_bytes(path: &PathBuf) -> Result<Arc<[u8]>, FontError> {
    let data = fs::read(path).map_err(|_| FontError::ReadFailed(path.display().to_string()))?;
    Ok(Arc::<[u8]>::from(data))
}

/// Convert `ttf-parser` outline callbacks into a `lyon::path::Path`.
///
/// Important:
/// - A glyph may contain multiple contours. `move_to` starts a new contour.
/// - `close` ends the current contour.
struct LyonOutlineBuilder {
    builder: lyon::path::Builder,
    contour_open: bool,
}

impl LyonOutlineBuilder {
    fn new() -> Self {
        Self {
            builder: Path::builder(),
            contour_open: false,
        }
    }

    fn build(mut self) -> Path {
        // If a contour was left open, close it (defensive).
        if self.contour_open {
            self.builder.close();
            self.contour_open = false;
        }
        self.builder.build()
    }
}

impl ttf_parser::OutlineBuilder for LyonOutlineBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        // Close previous contour if it was open.
        if self.contour_open {
            self.builder.close();
            self.contour_open = false;
        }
        self.builder.begin(point(x, y));
        self.contour_open = true;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.builder.line_to(point(x, y));
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.builder.quadratic_bezier_to(point(x1, y1), point(x, y));
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.builder
            .cubic_bezier_to(point(x1, y1), point(x2, y2), point(x, y));
    }

    fn close(&mut self) {
        if self.contour_open {
            self.builder.close();
            self.contour_open = false;
        }
    }
}
