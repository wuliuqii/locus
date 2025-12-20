//! Typst frame extraction (Phase A): structure walk + logging.
//!
//! Goal:
//! - Walk a compiled `typst::layout::PagedDocument` and log the nested structure
//!   of frames/items (including groups/transforms) to validate recursion.
//!
//! Non-goals (for now):
//! - No geometry extraction yet (no glyph outlines, no shape tessellation).
//! - No strict stability of the printed format; it's a debugging tool.
//!
//! Why this exists:
//! - Math in Typst results in deeply nested frame items (groups + transforms).
//! - If transform composition order is wrong, formulas "explode".
//! - Before implementing mesh extraction, we want a reliable traversal that can
//!   show us the exact nesting Typst produced.
//!
//! Notes:
//! - Typst's internal item types can evolve across versions.
//! - This module aims to be tolerant to such changes by logging what we can and
//!   skipping details that are hard to access without private APIs.
//!
//! Usage example (Phase A bring-up):
//! ```ignore
//! let compiled = crate::typst::engine::compile_zeta_validation()?;
//! crate::typst::engine::extract::log_paged_document_tree(&compiled.document, 3);
//! ```

use std::fmt::Write as _;

/// Log the structure of a paged document.
///
/// - `max_depth`: limits recursion depth to avoid flooding logs.
pub fn log_paged_document_tree(document: &typst::layout::PagedDocument, max_depth: usize) {
    let mut out = String::new();

    let _ = writeln!(
        out,
        "Typst PagedDocument: pages={}, max_depth={}",
        document.pages.len(),
        max_depth
    );

    for (page_idx, page) in document.pages.iter().enumerate() {
        let _ = writeln!(out, "page[{page_idx}]:");

        // Most Typst page types include a `frame`. The exact field names can vary.
        // We'll use a helper that is version-tolerant via pattern matching where possible.
        log_page_frame(&mut out, page, 1, max_depth);
    }

    log::info!("{out}");
}

fn log_page_frame(out: &mut String, page: &typst::layout::Page, depth: usize, max_depth: usize) {
    if depth > max_depth {
        let _ = writeln!(out, "{}<depth limit reached>", indent(depth));
        return;
    }

    // In typst 0.14, `Page` has a `frame` field (public).
    // If that changes in the future, this module will need an update.
    let frame = &page.frame;
    let _ = writeln!(out, "{}frame: size={:?}", indent(depth), frame.size());

    log_frame(out, frame, depth + 1, max_depth);
}

fn log_frame(out: &mut String, frame: &typst::layout::Frame, depth: usize, max_depth: usize) {
    if depth > max_depth {
        let _ = writeln!(out, "{}<depth limit reached>", indent(depth));
        return;
    }

    // Frame items are usually stored as positioned items.
    // In typst 0.14, a frame exposes an iterator via `items()`.
    let mut count = 0usize;

    for (pos, item) in frame.items() {
        count += 1;

        if depth > max_depth {
            break;
        }

        // `frame.items()` yields `(&Point, &FrameItem)` in typst 0.14.
        log_frame_item(out, *pos, item, depth, max_depth);
    }

    if count == 0 {
        let _ = writeln!(out, "{}<empty frame>", indent(depth));
    }
}

/// Log one frame item with its position.
fn log_frame_item(
    out: &mut String,
    pos: typst::layout::Point,
    item: &typst::layout::FrameItem,
    depth: usize,
    max_depth: usize,
) {
    if depth > max_depth {
        let _ = writeln!(out, "{}<depth limit reached>", indent(depth));
        return;
    }

    // Print a compact header line for the item.
    // We avoid trying to print huge internal structs.
    match item {
        typst::layout::FrameItem::Group(group) => {
            let _ = writeln!(
                out,
                "{}Group @pos={:?} (items={})",
                indent(depth),
                pos,
                group.frame.items().count()
            );

            // Group has its own transform and an inner frame.
            // The exact fields on Group can differ; in 0.14 it includes:
            // - `transform`
            // - `frame`
            let _ = writeln!(out, "{}transform={:?}", indent(depth + 1), group.transform);

            if depth + 1 <= max_depth {
                log_frame(out, &group.frame, depth + 1, max_depth);
            }
        }

        typst::layout::FrameItem::Text(text) => {
            // Text is usually a shaped glyph run with font and size.
            // We log only high-level info.
            let _ = writeln!(out, "{}Text @pos={:?}", indent(depth), pos);

            // Try to log some fields if they are public in this version.
            // We keep this conservative to avoid breaking on minor API changes.
            #[allow(unused_variables)]
            {
                // Some versions expose: text.size, text.font, text.glyphs/cluster data, etc.
                // We log what we can without deep inspection.
                let _ = writeln!(out, "{}(text item details omitted)", indent(depth + 1));
            }
        }

        typst::layout::FrameItem::Shape(shape, _) => {
            let _ = writeln!(out, "{}Shape @pos={:?}", indent(depth), pos);
            let _ = writeln!(out, "{}shape={:?}", indent(depth + 1), shape);
        }

        typst::layout::FrameItem::Image(image, _, _) => {
            let _ = writeln!(out, "{}Image @pos={:?}", indent(depth), pos);
            let _ = writeln!(out, "{}image={:?}", indent(depth + 1), image);
        }

        typst::layout::FrameItem::Link(dest, size) => {
            let _ = writeln!(out, "{}Link @pos={:?}", indent(depth), pos);
            let _ = writeln!(out, "{}dest={:?}", indent(depth + 1), dest);
            let _ = writeln!(out, "{}size={:?}", indent(depth + 1), size);
        }

        // Future-proof: if Typst adds new variants, we at least print Debug.
        other => {
            let _ = writeln!(
                out,
                "{}Item @pos={:?} kind={}",
                indent(depth),
                pos,
                frame_item_kind_name(other)
            );
            let _ = writeln!(out, "{}{:?}", indent(depth + 1), other);
        }
    }
}

fn frame_item_kind_name(item: &typst::layout::FrameItem) -> &'static str {
    match item {
        typst::layout::FrameItem::Group(_) => "Group",
        typst::layout::FrameItem::Text(_) => "Text",
        typst::layout::FrameItem::Shape(..) => "Shape",
        typst::layout::FrameItem::Image(..) => "Image",
        typst::layout::FrameItem::Link(..) => "Link",
        _ => "Other",
    }
}

fn indent(depth: usize) -> &'static str {
    // Keep indentation small and deterministic.
    // We return a static string for the first few depths and clamp afterwards.
    const INDENTS: [&str; 9] = [
        "",
        "  ",
        "    ",
        "      ",
        "        ",
        "          ",
        "            ",
        "              ",
        "                ",
    ];
    INDENTS[depth.min(INDENTS.len() - 1)]
}
