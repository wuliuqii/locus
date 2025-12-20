//! Minimal Typst `World` implementation (first pass).
//!
//! Goal (Phase A, Strategy B):
//! - Compile an in-memory Typst source (math-only) using **our own** system font discovery.
//! - Provide:
//!   - `library()`
//!   - `book()` (FontBook)
//!   - `font(index)` (Font)
//!   - `main()` and `source(main)`
//! - Keep everything deterministic and self-contained (no package resolution yet).
//!
//! Design notes:
//! - Fonts are discovered through `fontdb` (system fonts).
//! - Font data is loaded from disk into memory once and exposed to Typst via `font(index)`.
//! - `FontBook` is built from the loaded `Font`s so Typst can select faces.
//! - This is intentionally a *first-pass* implementation: no file imports, no packages,
//!   no binary assets, no caching across multiple documents (yet).
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
    diag::{FileError, FileResult},
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
/// - System fonts discovered via fontdb and loaded from disk.
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
    /// This will:
    /// - create a `FileId` for the virtual main file
    /// - load system fonts via `fontdb`
    /// - read font bytes and parse into `typst::text::Font`
    /// - build a `FontBook`
    pub fn new(doc: InMemoryDoc) -> anyhow::Result<Self> {
        // Create a deterministic FileId for our in-memory main file.
        //
        // typst-syntax 0.14 expects:
        //   FileId::new(Option<PackageSpec>, VirtualPath)
        let main = FileId::new(None, VirtualPath::new(doc.main_path.as_str()));

        // Library: use the built-in Typst standard library.
        let library = LazyHash::new(Library::default());

        // Fonts: discover and load system fonts.
        let (fonts, font_data, book) = load_system_fonts_into_typst()?;

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

/// Load system fonts via `fontdb`, parse them into `typst::text::Font`, and build a `FontBook`.
///
/// Notes:
/// - We only support file-backed fonts (`Source::File`) here.
/// - We attempt to load as many fonts as possible; parse failures are skipped.
/// - We *do not* do any custom fallback logic here; Typst will use the FontBook.
fn load_system_fonts_into_typst() -> anyhow::Result<(Vec<Font>, Vec<Arc<[u8]>>, FontBook)> {
    let mut db = Database::new();
    db.load_system_fonts();

    let mut fonts: Vec<Font> = Vec::new();
    let mut font_data: Vec<Arc<[u8]>> = Vec::new();

    // Load file-backed fonts.
    //
    // `fontdb` can yield faces that come from:
    // - Source::File(PathBuf)
    // - Source::Binary(Arc<[u8]>)
    //
    // For first pass, we only handle Source::File.
    for face in db.faces() {
        let (path, index) = match &face.source {
            FontSource::File(p) => (p.clone(), face.index),
            _ => continue,
        };

        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(_) => continue,
        };

        // Make font bytes `'static` by owning them in an Arc.
        let owned: Arc<[u8]> = Arc::from(bytes);

        // Parse font for this face index inside file/collection.
        //
        // In typst 0.14, `Bytes::new(...)` requires owned `'static` backing storage.
        // By passing an `Arc<[u8]>`, we satisfy that requirement.
        let bytes = Bytes::new(owned.clone());
        let font = Font::new(bytes, index);

        if let Some(font) = font {
            font_data.push(owned);
            fonts.push(font);
        }
    }

    // As a fallback, ensure we have at least some fonts by loading Typst's built-in font set.
    // `typst-assets` provides embedded fonts used by Typst itself.
    //
    // This is still consistent with "Strategy B" (we control font provisioning) and makes the
    // project robust on minimal systems/containers.
    if fonts.is_empty() {
        for data in typst_assets::fonts() {
            // Make font bytes `'static` by owning them in an Arc.
            let owned: Arc<[u8]> = Arc::from(data);

            let bytes = Bytes::new(owned.clone());
            if let Some(font) = Font::new(bytes, 0) {
                font_data.push(owned);
                fonts.push(font);
            }
        }
    }

    if fonts.is_empty() {
        anyhow::bail!("no fonts could be loaded (system fonts + typst-assets fallback failed)");
    }

    // Build the font book.
    //
    // `FontBook::from_fonts` expects an iterator over `Font`s.
    // We keep the resulting book aligned with the vector order, so `font(index)` returns
    // the corresponding `Font`.
    let book = FontBook::from_fonts(fonts.iter());

    Ok((fonts, font_data, book))
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
        anyhow::anyhow!(
            "typst compile failed with {} error(s): {:?}",
            errs.len(),
            errs
        )
    })
}
