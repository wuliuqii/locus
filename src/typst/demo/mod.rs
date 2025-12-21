//! Helpers for demo/teaching scenarios built on top of Typst + the renderer.
//!
//! This module intentionally provides **small, ergonomic wrappers** around the lower-level
//! Typst pipeline:
//! - compile a snippet (in-memory)
//! - extract meshes via `typst::render`
//! - package them into a single `Scene2D` object (`Mobject2D`) that can be animated/placed
//!
//! It is not a stable public API yet. Treat this as "demo glue" to accelerate teaching demos.
//!
//! Typical usage in a teaching example:
//! - compile a short label like `$a$` or `$c^2$`
//! - insert as a `Mobject2D` into an existing `Scene2D`
//! - animate its `local_from_parent` and/or `fill.a` via `anim::Timeline`
//!
//! New in this iteration:
//! - `compile_snippet_to_group_mobject_baseline(...)` produces a **group** mobject that
//!   is anchored at the Typst baseline origin (x=0, y=0 in Typst item space).
//!   This makes it easy to place/animate text with a baseline anchor.
//!
//! Bounds note:
//! - The child mesh's `local_bounds` is intentionally left as `None` here.
//!   This avoids accidentally relying on cached/partial bounds when framing the camera.
//!   If you need framing, compute bounds via `compute_local_bounds()` on the group/root.
//!
//! Note: This still produces a **single merged mesh** per snippet. If you need per-glyph animation
//! later, you'll want a different output that preserves substructure.

use anyhow::Context as _;

use crate::scene::{Affine2, Mesh2D, Mobject2D, Rgba};
use crate::typst;

/// Options for compiling a snippet into a single mesh object.
#[derive(Debug, Clone)]
pub struct TypstMeshOptions {
    /// Name assigned to the created `Mobject2D`.
    pub name: String,

    /// Fill color used for the merged mesh.
    ///
    /// This is currently applied uniformly to the output mesh, independent of Typst styling.
    pub fill: Rgba,

    /// Local transform to apply to the `Mobject2D` (e.g. translate to a position).
    pub local_from_parent: Affine2,

    /// Typst render options (tessellation tolerance, enabling glyphs/shapes/lines).
    pub render: typst::render::RenderOptions,

    /// If true, also include Typst shape geometry (rules/rects/curves) alongside glyphs.
    ///
    /// This simply toggles `render.enable_shapes` and `render.enable_lines` for convenience.
    pub include_shapes: bool,

    /// If true, include the text debug overlay mesh.
    ///
    /// This is generally only useful while developing layout logic.
    pub include_text_debug: bool,
}

/// Options for compiling a snippet into a baseline-anchored group mobject.
///
/// This exists to make Timeline animation ergonomic:
/// - you animate the group's `local_from_parent` (translate/scale/rotate, alpha)
/// - the actual glyph mesh is a child node, kept at the baseline origin
///
/// Baseline anchoring:
/// - Typst text items are positioned relative to a baseline.
//  - In our extraction code, glyph outlines are emitted in "item space" with baseline at y=0.
#[derive(Debug, Clone)]
pub struct TypstGroupOptions {
    /// Name assigned to the created **group** `Mobject2D`.
    pub name: String,

    /// Name assigned to the child mesh node.
    pub child_name: String,

    /// Fill color used for the merged mesh child.
    pub fill: Rgba,

    /// Local transform to apply to the group (use this to position the baseline in world space).
    pub local_from_parent: Affine2,

    /// Typst render options (tessellation tolerance, enabling glyphs/shapes/lines).
    pub render: typst::render::RenderOptions,

    /// If true, also include Typst shape geometry (rules/rects/curves) alongside glyphs.
    pub include_shapes: bool,

    /// If true, include the text debug overlay mesh.
    pub include_text_debug: bool,
}

impl Default for TypstGroupOptions {
    fn default() -> Self {
        Self {
            name: "typst_group".to_string(),
            child_name: "mesh".to_string(),
            fill: Rgba::WHITE,
            local_from_parent: Affine2::IDENTITY,
            render: typst::render::RenderOptions::default(),
            include_shapes: false,
            include_text_debug: false,
        }
    }
}

impl Default for TypstMeshOptions {
    fn default() -> Self {
        Self {
            name: "typst_mesh".to_string(),
            fill: Rgba::WHITE,
            local_from_parent: Affine2::IDENTITY,
            render: typst::render::RenderOptions::default(),
            include_shapes: false,
            include_text_debug: false,
        }
    }
}

/// Compile a Typst snippet into a single merged `Mobject2D`.
///
/// What you get:
/// - A single `Mobject2D` with one `Mesh2D` (triangles).
/// - The mesh is built by compiling the snippet with `typst::engine` and extracting
///   glyph outlines (and optionally shapes) via `typst::render`.
///
/// What you do NOT get (yet):
/// - per-glyph substructure
/// - per-item colors from Typst
///
/// The snippet should be valid Typst. For math, use Typst math mode, e.g.:
/// - `"$a$"`
/// - `"$c^2 = a^2 + b^2$"`
pub fn compile_snippet_to_mobject(
    snippet: &str,
    mut opts: TypstMeshOptions,
) -> anyhow::Result<Mobject2D> {
    // Convenience flags.
    if opts.include_shapes {
        opts.render.enable_shapes = true;
        opts.render.enable_lines = true;
    }

    let compiled = typst::engine::compile_math_only(snippet)
        .with_context(|| format!("typst demo: failed to compile snippet: {snippet:?}"))?;

    let (meshes, stats) =
        typst::render::build_meshes_from_paged_document(&compiled.document, &opts.render);

    // Merge meshes in a deterministic order.
    //
    // If you later want independent z-ordering or different colors, use
    // `typst::render::build_draw_items_from_paged_document` instead.
    let mut merged = Mesh2D::default();

    // Shapes first (behind text) if requested.
    if opts.include_shapes {
        append_mesh(&mut merged, &meshes.shapes);
        append_mesh(&mut merged, &meshes.lines);
    }

    // Glyphs on top.
    append_mesh(&mut merged, &meshes.glyphs);

    // Optional debug overlay (usually on top).
    if opts.include_text_debug {
        append_mesh(&mut merged, &meshes.text_debug);
    }

    // If everything was empty, provide a helpful error message.
    if merged.positions.is_empty() || merged.indices.is_empty() {
        anyhow::bail!(
            "typst demo: snippet produced no renderable triangles (pages={}, groups={}, texts_seen={}, shapes_seen={}, glyph_tris={})",
            stats.pages,
            stats.groups,
            stats.texts_seen,
            stats.shapes_seen,
            stats.glyph_triangles
        );
    }

    Ok(Mobject2D::new(opts.name)
        .with_mesh(merged)
        .with_fill(opts.fill)
        .with_transform(opts.local_from_parent))
}

/// Compile a Typst snippet into a **baseline-anchored group** `Mobject2D` suitable for Timeline animation.
///
/// The returned object is a parent "group" node with:
/// - `local_from_parent` applied on the group (animate this)
/// - a single child mesh node holding the compiled snippet triangles
///
/// Baseline anchoring:
/// - the child is placed at identity transform in the group
/// - the group's origin corresponds to the Typst baseline origin (x=0, y=0)
///
/// This makes placement predictable: you translate the group to where you want the baseline start.
pub fn compile_snippet_to_group_mobject_baseline(
    snippet: &str,
    mut opts: TypstGroupOptions,
) -> anyhow::Result<Mobject2D> {
    // Convenience flags.
    if opts.include_shapes {
        opts.render.enable_shapes = true;
        opts.render.enable_lines = true;
    }

    let compiled = typst::engine::compile_math_only(snippet)
        .with_context(|| format!("typst demo: failed to compile snippet: {snippet:?}"))?;

    let (meshes, stats) =
        typst::render::build_meshes_from_paged_document(&compiled.document, &opts.render);

    let mut merged = Mesh2D::default();

    if opts.include_shapes {
        append_mesh(&mut merged, &meshes.shapes);
        append_mesh(&mut merged, &meshes.lines);
    }

    append_mesh(&mut merged, &meshes.glyphs);

    if opts.include_text_debug {
        append_mesh(&mut merged, &meshes.text_debug);
    }

    if merged.positions.is_empty() || merged.indices.is_empty() {
        anyhow::bail!(
            "typst demo: snippet produced no renderable triangles (pages={}, groups={}, texts_seen={}, shapes_seen={}, glyph_tris={})",
            stats.pages,
            stats.groups,
            stats.texts_seen,
            stats.shapes_seen,
            stats.glyph_triangles
        );
    }

    let mut group = Mobject2D::new(opts.name).with_transform(opts.local_from_parent);
    let mut child = Mobject2D::new(opts.child_name)
        .with_mesh(merged)
        .with_fill(opts.fill)
        .with_transform(Affine2::IDENTITY);

    // Intentionally do not set `child.local_bounds` here.
    // If you later add bounds caching, ensure it reflects the full mesh content.
    child.local_bounds = None;

    group.add_child(child);
    Ok(group)
}

/// Append `src` mesh into `dst`, offsetting indices.
fn append_mesh(dst: &mut Mesh2D, src: &Mesh2D) {
    if src.positions.is_empty() || src.indices.is_empty() {
        return;
    }

    let base = dst.positions.len();
    // Keep parity with other mesh utilities: u16 indices.
    assert!(
        base + src.positions.len() <= u16::MAX as usize,
        "typst demo: mesh vertex count overflow for u16 indices"
    );

    dst.positions.extend_from_slice(&src.positions);
    let base_u16 = base as u16;
    dst.indices
        .extend(src.indices.iter().copied().map(|i| base_u16 + i));
}
