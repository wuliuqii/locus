//! Minimal Typst `World` implementation (first pass).
//!
//! Goal (Phase A, Strategy B):
//! - Compile an in-memory Typst source (math-only) with a **controlled** font provisioning strategy.
//! - Provide:
//!   - `library()`
//!   - `book()` (FontBook)
//!   - `font(index)` (Font)
//!   - `main()` and `source(main)`
//! - Keep everything deterministic and self-contained (no package resolution yet).
//!
//! Design notes (updated):
//! - Phase A default is **typst-assets only** (no system scan), which is more stable and reduces
//!   runtime pressure from enumerating / reading many system fonts.
//! - Optionally, you can enable a constrained system-font load by specifying a small set of
//!   candidate family names (e.g. STIX / Libertine / Latin Modern) plus fallback.
//!
//! Diagnostics note:
//! - When compilation fails, prefer rich diagnostics (with spans) over dumping raw sources.
//!   We attempt to format `SourceDiagnostic`s into a readable string for debugging.
//!
//! Implementation constraints:
//! - `typst` wants:
//!   - `library: LazyHash<Library>`
//!   - `book: LazyHash<FontBook>`
//!   - `font(index) -> Option<Font>` with stable indexing consistent with `FontBook`
//! - `typst` compilation uses `World` as the access layer.

use std::{collections::HashMap, fs, path::PathBuf, sync::Arc};

use ecow::EcoString;
use fontdb::{Database, Source as FontSource};
use typst::{
    Library, LibraryExt,
    diag::{FileError, FileResult, SourceDiagnostic},
    foundations::{Bytes, Datetime},
    syntax::{FileId, Source as TypstSource, VirtualPath},
    text::{Font, FontBook},
    utils::LazyHash,
};

/// In-memory main source document.
#[derive(Debug, Clone)]
pub struct InMemoryDoc {
    /// Virtual file path for diagnostics.
    pub main_path: EcoString,
    /// Typst source contents.
    pub main_source: EcoString,
}

impl InMemoryDoc {
    pub fn new(main_path: impl Into<EcoString>, main_source: impl Into<EcoString>) -> Self {
        Self {
            main_path: main_path.into(),
            main_source: main_source.into(),
        }
    }
}

/// A first-pass Typst world.
///
/// - Single in-memory main file.
/// - No filesystem reads for sources beyond the main file.
/// - Fonts are provisioned according to `FontLoadingStrategy`.
pub struct TypstWorld {
    doc: Arc<InMemoryDoc>,
    main: FileId,

    /// Typst standard library.
    library: LazyHash<Library>,

    /// Typst font book (built from loaded fonts).
    book: LazyHash<FontBook>,

    /// Loaded fonts in the exact order corresponding to indices used by `font(index)`.
    ///
    /// Note: `typst::text::Font` internally holds `Bytes` that must be `'static`.
    /// We therefore keep the underlying font data alive in `font_data` for the
    /// lifetime of the world.
    fonts: Vec<Font>,

    /// Owned font bytes backing `fonts`.
    ///
    /// We store owned font data as `Arc<[u8]>` to satisfy Typst's `'static` requirement
    /// for `Bytes::new(...)` (it stores owned, `'static` data).
    font_data: Vec<Arc<[u8]>>,

    /// (Optional) cache for binary file access; currently unused because we don't support `file()`.
    #[allow(dead_code)]
    bin_cache: HashMap<FileId, Bytes>,
}

impl TypstWorld {
    /// Create a world from a single in-memory Typst source.
    ///
    /// Default behavior (Phase A):
    /// - Use `typst-assets` fonts only (no system scan).
    ///
    /// This is usually sufficient for math (e.g. ζ) and avoids loading a huge number
    /// of system fonts at startup.
    pub fn new(doc: InMemoryDoc) -> anyhow::Result<Self> {
        Self::new_with_fonts(doc, FontLoadingStrategy::TypstAssetsOnly)
    }

    /// Create a world with an explicit font loading strategy.
    pub fn new_with_fonts(doc: InMemoryDoc, strategy: FontLoadingStrategy) -> anyhow::Result<Self> {
        // Create a deterministic FileId for our in-memory main file.
        //
        // typst-syntax 0.14 expects:
        //   FileId::new(Option<PackageSpec>, VirtualPath)
        let main = FileId::new(None, VirtualPath::new(doc.main_path.as_str()));

        // Library: use the built-in Typst standard library.
        let library = LazyHash::new(Library::default());

        let (fonts, font_data, book) = load_fonts_into_typst(strategy)?;

        Ok(Self {
            doc: Arc::new(doc),
            main,
            library,
            book: LazyHash::new(book),
            fonts,
            font_data,
            bin_cache: HashMap::new(),
        })
    }

    /// Access to the loaded fonts (for debugging/metrics).
    pub fn font_count(&self) -> usize {
        self.fonts.len()
    }
}

/// Controls how fonts are provisioned to Typst.
///
/// Phase A default is `TypstAssetsOnly` to keep runtime steady and deterministic.
#[derive(Debug, Clone)]
pub enum FontLoadingStrategy {
    /// Load only embedded fonts from `typst-assets` (no system scan).
    TypstAssetsOnly,

    /// Load only a small set of system font families (by name), plus `typst-assets` fallback.
    ///
    /// This is useful if you want to prefer a known math/text family (e.g. STIX, Libertine,
    /// Latin Modern) but avoid loading every system font.
    LimitedSystemFamilies {
        /// Candidate family names in priority order (case-insensitive match on `fontdb` families).
        families: Vec<String>,
    },

    /// Load all system fonts (previous behavior), with `typst-assets` fallback.
    LoadAllSystemFonts,
}

fn load_fonts_into_typst(
    strategy: FontLoadingStrategy,
) -> anyhow::Result<(Vec<Font>, Vec<Arc<[u8]>>, FontBook)> {
    let mut fonts: Vec<Font> = Vec::new();
    let mut font_data: Vec<Arc<[u8]>> = Vec::new();

    match strategy {
        FontLoadingStrategy::TypstAssetsOnly => {
            load_typst_assets_fonts(&mut fonts, &mut font_data);
        }
        FontLoadingStrategy::LimitedSystemFamilies { families } => {
            load_system_fonts_filtered_into_typst(&families, &mut fonts, &mut font_data);
            if fonts.is_empty() {
                load_typst_assets_fonts(&mut fonts, &mut font_data);
            }
        }
        FontLoadingStrategy::LoadAllSystemFonts => {
            load_all_system_fonts_into_typst(&mut fonts, &mut font_data);
            if fonts.is_empty() {
                load_typst_assets_fonts(&mut fonts, &mut font_data);
            }
        }
    }

    if fonts.is_empty() {
        anyhow::bail!("no fonts could be loaded (strategy produced zero usable fonts)");
    }

    let book = FontBook::from_fonts(fonts.iter());
    Ok((fonts, font_data, book))
}

fn load_typst_assets_fonts(fonts: &mut Vec<Font>, font_data: &mut Vec<Arc<[u8]>>) {
    // `typst_assets::fonts()` yields borrowed font bytes. We must *own* the bytes to satisfy
    // Typst's `'static` requirement for `Bytes::new(...)`.
    //
    // Important: We must copy the bytes out of the iterator item (e.g. `&'static [u8]`)
    // into an owned buffer, then keep that buffer alive in `font_data`.
    for data in typst_assets::fonts() {
        // Be explicit about the slice type to help inference.
        let data: &[u8] = data;

        let owned: Arc<[u8]> = Arc::from(data.to_vec());
        let bytes = Bytes::new(owned.clone());

        if let Some(font) = Font::new(bytes, 0) {
            font_data.push(owned);
            fonts.push(font);
        }
    }
}

fn load_all_system_fonts_into_typst(fonts: &mut Vec<Font>, font_data: &mut Vec<Arc<[u8]>>) {
    let mut db = Database::new();
    db.load_system_fonts();

    for face in db.faces() {
        let (path, index) = match &face.source {
            FontSource::File(p) => (p.clone(), face.index),
            _ => continue,
        };

        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(_) => continue,
        };

        let owned: Arc<[u8]> = Arc::from(bytes);
        let bytes = Bytes::new(owned.clone());
        let font = Font::new(bytes, index);

        if let Some(font) = font {
            font_data.push(owned);
            fonts.push(font);
        }
    }
}

fn load_system_fonts_filtered_into_typst(
    families: &[String],
    fonts: &mut Vec<Font>,
    font_data: &mut Vec<Arc<[u8]>>,
) {
    let mut db = Database::new();
    db.load_system_fonts();

    if families.is_empty() {
        return;
    }

    // Build a list of wanted family names (trimmed). We match case-insensitively.
    let mut wanted: Vec<String> = Vec::new();
    for fam in families {
        let s = fam.trim();
        if !s.is_empty() {
            wanted.push(s.to_string());
        }
    }

    if wanted.is_empty() {
        return;
    }

    // Iterate all discovered faces, but only load bytes for faces whose family matches.
    // This avoids `Database::retain_faces` (not available in fontdb 0.21) while still
    // preventing the expensive part: reading/parsing tons of font files.
    for face in db.faces() {
        // In fontdb 0.21, a face can have multiple family names (e.g. localized / aliases).
        // We accept the face if *any* of its family names matches our wanted list.
        let family_matches = face
            .families
            .iter()
            .any(|(name, _)| wanted.iter().any(|w| name.eq_ignore_ascii_case(w)));

        if !family_matches {
            continue;
        }

        let (path, index) = match &face.source {
            FontSource::File(p) => (p.clone(), face.index),
            _ => continue,
        };

        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(_) => continue,
        };

        let owned: Arc<[u8]> = Arc::from(bytes);
        let bytes = Bytes::new(owned.clone());
        let font = Font::new(bytes, index);

        if let Some(font) = font {
            font_data.push(owned);
            fonts.push(font);
        }
    }
}

impl typst::World for TypstWorld {
    fn library(&self) -> &LazyHash<Library> {
        &self.library
    }

    fn book(&self) -> &LazyHash<FontBook> {
        &self.book
    }

    fn main(&self) -> FileId {
        self.main
    }

    fn source(&self, id: FileId) -> FileResult<TypstSource> {
        if id == self.main {
            Ok(TypstSource::new(id, self.doc.main_source.to_string()))
        } else {
            Err(FileError::NotFound(PathBuf::from("<memory>")))
        }
    }

    fn file(&self, _id: FileId) -> FileResult<Bytes> {
        // Phase A: No external assets/files.
        Err(FileError::NotFound(PathBuf::from("<memory>")))
    }

    fn font(&self, index: usize) -> Option<Font> {
        self.fonts.get(index).cloned()
    }

    fn today(&self, _offset: Option<i64>) -> Option<Datetime> {
        // `Datetime::today` requires an engine context in typst 0.14, so we return None here.
        // This is fine for deterministic math-only compilation.
        None
    }
}

/// Helper to compile a math-only snippet into a PagedDocument.
///
/// This is not used by the renderer yet, but it provides the "Typst compile closed loop"
/// for Phase A:
/// - build world
/// - `typst::compile::<PagedDocument>(&world)`
///
/// The caller can then inspect document pages and frames for extraction.
pub fn compile_math_paged(snippet: &str) -> anyhow::Result<typst::layout::PagedDocument> {
    let doc = InMemoryDoc::new("<math-only>", snippet);
    let world = TypstWorld::new(doc)?;
    let warned = typst::compile::<typst::layout::PagedDocument>(&world);

    warned.output.map_err(|errs| {
        let formatted = format_source_diagnostics_with_excerpt(snippet, &errs);
        anyhow::anyhow!(
            "typst compile failed with {} error(s):\n{}",
            errs.len(),
            formatted
        )
    })
}

fn format_source_diagnostics(diags: &[SourceDiagnostic]) -> String {
    format_source_diagnostics_with_excerpt("", diags)
}

fn format_source_diagnostics_with_excerpt(source: &str, diags: &[SourceDiagnostic]) -> String {
    // `SourceDiagnostic` carries spans/hints/traces, but formatting helpers differ between
    // Typst versions. We provide a best-effort formatter that is stable for debugging.
    //
    // Additionally, we try to print a short excerpt around the diagnostic span. In typst 0.14
    // we do not have a stable public API to resolve `Span` into `file:line:col` here, so this
    // excerpt is heuristic:
    // - It prints the first N characters of the source, and
    // - It always prints the raw span debug representation.
    //
    // Once we wire in Typst's diagnostic rendering utilities, we should replace this with a
    // true span-to-source mapping and a precise caret highlight.
    let mut out = String::new();

    // Keep the excerpt short so logs remain readable.
    const EXCERPT_CHARS: usize = 280;

    for (i, d) in diags.iter().enumerate() {
        use std::fmt::Write as _;
        let _ = writeln!(out, "[{}] {:?}: {}", i + 1, d.severity, d.message);

        if !d.hints.is_empty() {
            let _ = writeln!(out, "    hints:");
            for h in &d.hints {
                let _ = writeln!(out, "      - {}", h);
            }
        }

        if !d.trace.is_empty() {
            let _ = writeln!(out, "    trace:");
            for t in &d.trace {
                let _ = writeln!(out, "      - {:?}", t);
            }
        }

        // Best-effort excerpt (heuristic).
        if !source.is_empty() {
            let excerpt: String = source.chars().take(EXCERPT_CHARS).collect();
            let truncated = source.chars().count() > EXCERPT_CHARS;
            if truncated {
                let _ = writeln!(
                    out,
                    "    source_excerpt(first {} chars): {:?}…",
                    EXCERPT_CHARS, excerpt
                );
            } else {
                let _ = writeln!(out, "    source_excerpt: {:?}", excerpt);
            }
        }

        // We include the raw span identifier as a last resort. A richer "file:line:col"
        // mapping can be added once we wire in Typst's diagnostic rendering utilities.
        let _ = writeln!(out, "    span: {:?}", d.span);
    }

    out
}
