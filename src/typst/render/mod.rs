//! Typst → Scene extraction and mesh building.
//!
//! This module turns a compiled `typst::layout::PagedDocument` into renderer-friendly meshes.
//! It is intentionally **pure** (no winit/wgpu) and focuses on extraction + tessellation.
//!
//! Current capabilities (Phase A):
//! - Traverse nested frames (`FrameItem::Group`) while accumulating transforms.
//! - Extract and tessellate:
//!   - `Shape::Line` as a thin quad (stroke approximation).
//!   - `Shape::Rect` as a filled quad via lyon path tessellation.
//!   - `Shape::Curve` (Move/Line/Cubic/Close) via lyon path tessellation.
//! - Extract and tessellate text glyph outlines:
//!   - `FrameItem::Text(TextItem)` → iterate `TextItem.glyphs`
//!   - For each glyph: outline via Typst's `Font` TTF face + `ttf-parser`
//!   - Tessellate outline via lyon into `scene::Mesh2D`
//! - Basic glyph mesh caching keyed by (glyph id, scale bits).
//!
//! New in this iteration:
//! - Produce **colored draw items** rather than a few monolithic meshes.
//! - Map basic Typst paints (solid colors) for fills/strokes.
//!
//! Notes / limitations:
//! - Paint mapping is best-effort: we currently handle solid colors (`Paint::Solid`).
//! - Gradient/image paints are ignored for now.
//! - `Shape::Line` is still approximated as a quad using stroke thickness.
//! - Performance tuning (shared tessellator, better cache keys, atlas-like caches) is future work.

use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
};

use lyon::math::point as lyon_point;
use lyon::path::Path;

use crate::scene::{Affine2, DrawItem2D, Mesh2D, Rgba};
use typst::{
    layout::{Frame, FrameItem, PagedDocument, Transform},
    text::TextItem,
    visualize::{CurveItem, Geometry, Paint, Shape},
};

/// Options controlling extraction behavior.
#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// Tessellation tolerance. Smaller => more triangles.
    pub tolerance: f32,

    /// Enable glyph extraction (text → outlines).
    pub enable_glyphs: bool,

    /// Enable shape extraction (rect/curve).
    pub enable_shapes: bool,

    /// Enable line extraction (`Geometry::Line`).
    pub enable_lines: bool,

    /// Approximation thickness for `Shape::Line` when stroke is absent.
    pub default_line_thickness_pt: f32,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            tolerance: 0.02,
            enable_glyphs: true,
            enable_shapes: true,
            enable_lines: true,
            default_line_thickness_pt: 0.75,
        }
    }
}

/// Output meshes from extraction (legacy grouping).
///
/// Prefer [`build_draw_items_from_paged_document`] for color-aware draw items.
#[derive(Debug, Default)]
pub struct ExtractedMeshes {
    /// Filled glyph outlines (tessellated triangles).
    pub glyphs: Mesh2D,
    /// Filled shapes (rect/curve tessellated triangles).
    pub shapes: Mesh2D,
    /// Lines/rules (quad approximation).
    pub lines: Mesh2D,
    /// Optional debug overlay for text item extents (baseline-relative boxes).
    pub text_debug: Mesh2D,
}

/// A color-aware extraction output: a list of draw items (mesh + transform + fill + z).
///
/// This is the recommended output for rendering because it keeps per-item colors intact.
#[derive(Debug, Default)]
pub struct ExtractedDrawItems {
    pub items: Vec<DrawItem2D>,
}

/// Basic extraction stats for logging / profiling.
#[derive(Debug, Default, Clone, Copy)]
pub struct ExtractStats {
    pub pages: usize,
    pub groups: usize,
    pub shapes_seen: usize,
    pub texts_seen: usize,

    pub lines_emitted: usize,
    pub filled_shapes_emitted: usize,

    pub glyph_calls: usize,
    pub glyph_tess_calls: usize,
    pub glyph_triangles: usize,
}

/// Build meshes from a compiled Typst `PagedDocument` (legacy grouping output).
///
/// Prefer [`build_draw_items_from_paged_document`] to keep per-item colors.
pub fn build_meshes_from_paged_document(
    doc: &PagedDocument,
    opts: &RenderOptions,
) -> (ExtractedMeshes, ExtractStats) {
    let mut out = ExtractedMeshes::default();
    let mut stats = ExtractStats::default();

    stats.pages = doc.pages.len();

    let mut cache = GlyphMeshCache::default();
    let ctx = ExtractCtx::default();

    for page in &doc.pages {
        walk_frame(
            &page.frame,
            Affine2::IDENTITY,
            ctx,
            opts,
            &mut cache,
            &mut out,
            &mut stats,
        );
    }

    (out, stats)
}

/// Build draw items (mesh + fill color + transform) from a compiled Typst `PagedDocument`.
///
/// This preserves basic per-shape colors (`Paint::Solid`) for fills/strokes.
///
/// - For fills, we emit filled meshes with `fill` color.
/// - For strokes, we currently only support `Geometry::Line` as a quad; other strokes are ignored.
pub fn build_draw_items_from_paged_document(
    doc: &PagedDocument,
    opts: &RenderOptions,
) -> (ExtractedDrawItems, ExtractStats) {
    let mut out = ExtractedDrawItems::default();
    let mut stats = ExtractStats::default();

    stats.pages = doc.pages.len();

    let mut cache = GlyphMeshCache::default();
    let ctx = ExtractCtx::default();

    for page in &doc.pages {
        walk_frame_draw_items(
            &page.frame,
            Affine2::IDENTITY,
            ctx,
            opts,
            &mut cache,
            &mut out.items,
            &mut stats,
        );
    }

    (out, stats)
}

/// Extraction context carried through recursion.
#[derive(Debug, Default, Clone, Copy)]
struct ExtractCtx {
    /// 0 = normal, 1+ = likely sub/superscript nesting.
    script_level: u8,
}

impl ExtractCtx {
    fn with_group_transform(self, t: Transform) -> Self {
        let sx = t.sx.get() as f32;
        let sy = t.sy.get() as f32;
        let s = (sx.abs() + sy.abs()) * 0.5;
        if s < 0.86 {
            Self {
                script_level: self.script_level.saturating_add(1),
            }
        } else {
            self
        }
    }

    fn script_scale(self) -> f32 {
        match self.script_level {
            0 => 1.0,
            1 => 0.72,
            _ => 0.62,
        }
    }
}

fn walk_frame(
    frame: &Frame,
    world_from_frame: Affine2,
    ctx: ExtractCtx,
    opts: &RenderOptions,
    cache: &mut GlyphMeshCache,
    out: &mut ExtractedMeshes,
    stats: &mut ExtractStats,
) {
    for (pos, item) in frame.items() {
        match item {
            FrameItem::Group(group) => {
                stats.groups += 1;
                let world_from_group =
                    world_from_frame.mul(affine2_from_typst_transform(group.transform));
                let child_ctx = ctx.with_group_transform(group.transform);
                walk_frame(
                    &group.frame,
                    world_from_group,
                    child_ctx,
                    opts,
                    cache,
                    out,
                    stats,
                );
            }

            FrameItem::Shape(shape, _span) => {
                stats.shapes_seen += 1;
                let world_from_item = world_from_frame.mul(Affine2::translate(
                    pos.x.to_pt() as f32,
                    pos.y.to_pt() as f32,
                ));
                extract_shape(world_from_item, shape, opts, out, stats);
            }

            FrameItem::Text(text) => {
                stats.texts_seen += 1;
                let world_from_item = world_from_frame.mul(Affine2::translate(
                    pos.x.to_pt() as f32,
                    pos.y.to_pt() as f32,
                ));

                // Optional: debug text extents
                let bb = derive_text_debug_box(text);
                append_baseline_rect_transformed(&mut out.text_debug, world_from_item, bb);

                if opts.enable_glyphs {
                    append_text_glyph_outlines(
                        &mut out.glyphs,
                        cache,
                        world_from_item,
                        text,
                        ctx.script_scale(),
                        opts,
                        stats,
                    );
                }
            }

            _ => {}
        }
    }
}

/// Walk frames and emit color-aware draw items.
fn walk_frame_draw_items(
    frame: &Frame,
    world_from_frame: Affine2,
    ctx: ExtractCtx,
    opts: &RenderOptions,
    cache: &mut GlyphMeshCache,
    out: &mut Vec<DrawItem2D>,
    stats: &mut ExtractStats,
) {
    for (pos, item) in frame.items() {
        match item {
            FrameItem::Group(group) => {
                stats.groups += 1;
                let world_from_group =
                    world_from_frame.mul(affine2_from_typst_transform(group.transform));
                let child_ctx = ctx.with_group_transform(group.transform);
                walk_frame_draw_items(
                    &group.frame,
                    world_from_group,
                    child_ctx,
                    opts,
                    cache,
                    out,
                    stats,
                );
            }

            FrameItem::Shape(shape, _span) => {
                stats.shapes_seen += 1;
                let world_from_item = world_from_frame.mul(Affine2::translate(
                    pos.x.to_pt() as f32,
                    pos.y.to_pt() as f32,
                ));
                extract_shape_draw_items(world_from_item, shape, opts, out, stats);
            }

            FrameItem::Text(text) => {
                stats.texts_seen += 1;
                let world_from_item = world_from_frame.mul(Affine2::translate(
                    pos.x.to_pt() as f32,
                    pos.y.to_pt() as f32,
                ));

                if opts.enable_glyphs {
                    // Tessellate glyphs directly into one mesh, then emit a draw item.
                    let mut glyph_mesh = Mesh2D::default();
                    let before = glyph_mesh.indices.len();
                    append_text_glyph_outlines(
                        &mut glyph_mesh,
                        cache,
                        world_from_item,
                        text,
                        ctx.script_scale(),
                        opts,
                        stats,
                    );
                    if glyph_mesh.indices.len() > before {
                        out.push(DrawItem2D {
                            mesh: glyph_mesh,
                            fill: Rgba::WHITE,
                            world_from_local: Affine2::IDENTITY,
                            z: 0,
                        });
                    }
                }
            }

            _ => {}
        }
    }
}

fn extract_shape(
    world_from_item: Affine2,
    shape: &Shape,
    opts: &RenderOptions,
    out: &mut ExtractedMeshes,
    stats: &mut ExtractStats,
) {
    match &shape.geometry {
        Geometry::Line(delta) if opts.enable_lines => {
            let (x0, y0) = world_from_item.transform_point(0.0, 0.0);
            let (x1, y1) =
                world_from_item.transform_point(delta.x.to_pt() as f32, delta.y.to_pt() as f32);

            let thickness_pt = shape
                .stroke
                .as_ref()
                .map(|s| s.thickness.to_pt() as f32)
                .unwrap_or(opts.default_line_thickness_pt)
                .max(0.25);

            append_line_as_rect(&mut out.lines, [x0, y0], [x1, y1], thickness_pt);
            stats.lines_emitted += 1;
        }

        Geometry::Rect(size) if opts.enable_shapes => {
            let w = size.x.to_pt() as f32;
            let h = size.y.to_pt() as f32;

            let mut b = Path::builder();
            b.begin(lyon_point(0.0, 0.0));
            b.line_to(lyon_point(w, 0.0));
            b.line_to(lyon_point(w, h));
            b.line_to(lyon_point(0.0, h));
            b.close();
            let path = b.build();

            let xf = affine2x3_from_scene(world_from_item);

            let before = out.shapes.indices.len();
            let _ = crate::font::tessellate::append_tessellated_path(
                &mut out.shapes,
                &path,
                xf,
                crate::font::tessellate::TessellateOptions {
                    tolerance: opts.tolerance,
                    ..Default::default()
                },
            );
            if out.shapes.indices.len() > before {
                stats.filled_shapes_emitted += 1;
            }
        }

        Geometry::Curve(curve) if opts.enable_shapes => {
            let mut b = Path::builder();
            let mut started = false;

            for item in curve.0.iter() {
                match item {
                    CurveItem::Move(p) => {
                        if started {
                            b.close();
                        }
                        b.begin(lyon_point(p.x.to_pt() as f32, p.y.to_pt() as f32));
                        started = true;
                    }
                    CurveItem::Line(p) => {
                        if !started {
                            b.begin(lyon_point(0.0, 0.0));
                            started = true;
                        }
                        b.line_to(lyon_point(p.x.to_pt() as f32, p.y.to_pt() as f32));
                    }
                    CurveItem::Cubic(p1, p2, p) => {
                        if !started {
                            b.begin(lyon_point(0.0, 0.0));
                            started = true;
                        }
                        b.cubic_bezier_to(
                            lyon_point(p1.x.to_pt() as f32, p1.y.to_pt() as f32),
                            lyon_point(p2.x.to_pt() as f32, p2.y.to_pt() as f32),
                            lyon_point(p.x.to_pt() as f32, p.y.to_pt() as f32),
                        );
                    }
                    CurveItem::Close => {
                        if started {
                            b.close();
                            started = false;
                        }
                    }
                }
            }

            if started {
                b.close();
            }

            let path = b.build();
            let xf = affine2x3_from_scene(world_from_item);

            let before = out.shapes.indices.len();
            let _ = crate::font::tessellate::append_tessellated_path(
                &mut out.shapes,
                &path,
                xf,
                crate::font::tessellate::TessellateOptions {
                    tolerance: opts.tolerance,
                    ..Default::default()
                },
            );
            if out.shapes.indices.len() > before {
                stats.filled_shapes_emitted += 1;
            }
        }

        _ => {}
    }
}

/// Map a Typst `Paint` to a scene color (best-effort).
fn paint_to_rgba(paint: &Paint) -> Option<Rgba> {
    match paint {
        Paint::Solid(c) => {
            // typst 0.14 uses palette types. `to_rgb()` returns an `Alpha<Rgb, f32>`-like type
            // with `red/green/blue` fields and a separate `alpha`.
            let rgba = c.to_rgb();
            Some(Rgba {
                r: rgba.red as f32,
                g: rgba.green as f32,
                b: rgba.blue as f32,
                a: rgba.alpha as f32,
            })
        }
        _ => None,
    }
}

/// Extract a shape into colored draw items.
fn extract_shape_draw_items(
    world_from_item: Affine2,
    shape: &Shape,
    opts: &RenderOptions,
    out: &mut Vec<DrawItem2D>,
    stats: &mut ExtractStats,
) {
    // Fill
    if opts.enable_shapes {
        if let Some(fill) = shape.fill.as_ref().and_then(paint_to_rgba) {
            // Tessellate geometry into a mesh and emit.
            let mut mesh = Mesh2D::default();
            let xf = affine2x3_from_scene(world_from_item);

            let path = match &shape.geometry {
                Geometry::Rect(size) => {
                    let w = size.x.to_pt() as f32;
                    let h = size.y.to_pt() as f32;
                    let mut b = Path::builder();
                    b.begin(lyon_point(0.0, 0.0));
                    b.line_to(lyon_point(w, 0.0));
                    b.line_to(lyon_point(w, h));
                    b.line_to(lyon_point(0.0, h));
                    b.close();
                    b.build()
                }
                Geometry::Curve(curve) => {
                    let mut b = Path::builder();
                    let mut started = false;
                    for item in curve.0.iter() {
                        match item {
                            CurveItem::Move(p) => {
                                if started {
                                    b.close();
                                }
                                b.begin(lyon_point(p.x.to_pt() as f32, p.y.to_pt() as f32));
                                started = true;
                            }
                            CurveItem::Line(p) => {
                                if !started {
                                    b.begin(lyon_point(0.0, 0.0));
                                    started = true;
                                }
                                b.line_to(lyon_point(p.x.to_pt() as f32, p.y.to_pt() as f32));
                            }
                            CurveItem::Cubic(p1, p2, p) => {
                                if !started {
                                    b.begin(lyon_point(0.0, 0.0));
                                    started = true;
                                }
                                b.cubic_bezier_to(
                                    lyon_point(p1.x.to_pt() as f32, p1.y.to_pt() as f32),
                                    lyon_point(p2.x.to_pt() as f32, p2.y.to_pt() as f32),
                                    lyon_point(p.x.to_pt() as f32, p.y.to_pt() as f32),
                                );
                            }
                            CurveItem::Close => {
                                if started {
                                    b.close();
                                    started = false;
                                }
                            }
                        }
                    }
                    if started {
                        b.close();
                    }
                    b.build()
                }
                _ => return,
            };

            let before = mesh.indices.len();
            let _ = crate::font::tessellate::append_tessellated_path(
                &mut mesh,
                &path,
                xf,
                crate::font::tessellate::TessellateOptions {
                    tolerance: opts.tolerance,
                    ..Default::default()
                },
            );
            if mesh.indices.len() > before {
                out.push(DrawItem2D {
                    mesh,
                    fill,
                    world_from_local: Affine2::IDENTITY,
                    z: 0,
                });
                stats.filled_shapes_emitted += 1;
            }
        }
    }

    // Stroke-only lines (best-effort)
    if opts.enable_lines {
        if let Geometry::Line(delta) = &shape.geometry {
            if let Some(stroke) = shape.stroke.as_ref() {
                if let Some(color) = paint_to_rgba(&stroke.paint) {
                    let (x0, y0) = world_from_item.transform_point(0.0, 0.0);
                    let (x1, y1) = world_from_item
                        .transform_point(delta.x.to_pt() as f32, delta.y.to_pt() as f32);
                    let thickness_pt = stroke.thickness.to_pt() as f32;
                    let mut mesh = Mesh2D::default();
                    append_line_as_rect(&mut mesh, [x0, y0], [x1, y1], thickness_pt.max(0.25));
                    out.push(DrawItem2D {
                        mesh,
                        fill: color,
                        world_from_local: Affine2::IDENTITY,
                        z: 0,
                    });
                    stats.lines_emitted += 1;
                }
            }
        }
    }
}

/// A baseline-relative debug rectangle description.
#[derive(Debug, Clone, Copy)]
struct BaselineBox {
    w_pt: f32,
    above_baseline_pt: f32,
    below_baseline_pt: f32,
    scale: f32,
}

/// Baseline-relative debug box for a text item.
/// Uses `TextItem.width()` and font bbox edges.
fn derive_text_debug_box(text: &TextItem) -> BaselineBox {
    use typst::text::{BottomEdge, BottomEdgeMetric, TextEdgeBounds, TopEdge, TopEdgeMetric};

    let w_pt = (text.width().to_pt() as f32).max(3.0).clamp(2.0, 200.0);

    let bounds = if let Some(g) = text.glyphs.first() {
        TextEdgeBounds::Glyph(g.id)
    } else {
        TextEdgeBounds::Zero
    };

    let (t, b) = text.font.edges(
        TopEdge::Metric(TopEdgeMetric::Bounds),
        BottomEdge::Metric(BottomEdgeMetric::Bounds),
        text.size,
        bounds,
    );

    let above = (t.to_pt() as f32).clamp(2.0, 80.0);
    let below = ((-b.to_pt() as f32).max(0.0)).clamp(0.0, 80.0);

    BaselineBox {
        w_pt,
        above_baseline_pt: above,
        below_baseline_pt: below,
        scale: 1.0,
    }
}

/// Append a baseline-relative rectangle transformed by `world_from_item`.
fn append_baseline_rect_transformed(mesh: &mut Mesh2D, world_from_item: Affine2, bb: BaselineBox) {
    let w = bb.w_pt * bb.scale;
    let above = bb.above_baseline_pt * bb.scale;
    let below = bb.below_baseline_pt * bb.scale;
    let hw = 0.5 * w;

    let (x0, y0) = world_from_item.transform_point(-hw, -below);
    let (x1, y1) = world_from_item.transform_point(hw, -below);
    let (x2, y2) = world_from_item.transform_point(hw, above);
    let (x3, y3) = world_from_item.transform_point(-hw, above);

    let base = mesh.positions.len() as u16;
    mesh.positions
        .extend_from_slice(&[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]);
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// Extract and append tessellated glyph outlines for a shaped Typst `TextItem`.
fn append_text_glyph_outlines(
    dst: &mut Mesh2D,
    cache: &mut GlyphMeshCache,
    world_from_item: Affine2,
    text: &TextItem,
    script_scale: f32,
    opts: &RenderOptions,
    stats: &mut ExtractStats,
) {
    use typst::text::{BottomEdge, BottomEdgeMetric, TextEdgeBounds, TopEdge, TopEdgeMetric};

    let face = text.font.ttf();
    let upm = face.units_per_em() as f32;
    if upm <= 0.0 {
        return;
    }

    let font_units_to_pt = (text.size.to_pt() as f32) / upm;
    let sx = font_units_to_pt * script_scale;
    let sy = font_units_to_pt * script_scale;

    let world_from_item_2x3 = affine2x3_from_scene(world_from_item);

    let mut pen_x_pt = 0.0f32;
    let pen_y_pt = 0.0f32;

    for g in text.glyphs.iter() {
        stats.glyph_calls += 1;

        let adv_pt = g.x_advance.at(text.size).to_pt() as f32;
        let x_off_pt = g.x_offset.at(text.size).to_pt() as f32;
        let y_off_pt = g.y_offset.at(text.size).to_pt() as f32;

        let tx = pen_x_pt + x_off_pt;
        let ty = pen_y_pt + y_off_pt;

        let key = GlyphCacheKey {
            glyph_id: g.id,
            sx_bits: sx.to_bits(),
            sy_bits: sy.to_bits(),
        };

        let cached = cache.get_or_insert_with(key, || {
            let gid = ttf_parser::GlyphId(g.id);
            let mut builder = LyonOutlineBuilder::new();
            let bbox = face.outline_glyph(gid, &mut builder);
            if bbox.is_none() {
                return None;
            }

            let path = builder.build();
            let local_no_translate = crate::font::tessellate::Affine2x3 {
                a: sx,
                b: 0.0,
                c: 0.0,
                d: sy,
                tx: 0.0,
                ty: 0.0,
            };

            let mut tmp = Mesh2D::default();
            if crate::font::tessellate::append_tessellated_path(
                &mut tmp,
                &path,
                local_no_translate,
                crate::font::tessellate::TessellateOptions {
                    tolerance: opts.tolerance,
                    ..Default::default()
                },
            )
            .is_err()
            {
                return None;
            }

            Some(tmp)
        });

        if let Some(src) = cached {
            let translate_only = crate::font::tessellate::Affine2x3 {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 1.0,
                tx,
                ty,
            };
            let xf = mul_affine2x3(world_from_item_2x3, translate_only);

            let before = dst.indices.len();
            append_mesh_with_transform(dst, src, xf);
            let added = dst.indices.len().saturating_sub(before);

            stats.glyph_triangles += added / 3;
            stats.glyph_tess_calls += 1;
        }

        // Keep pen moving forward.
        pen_x_pt += adv_pt;

        // Touch font edges for completeness (future: per-glyph vertical metrics).
        let _ = text.font.edges(
            TopEdge::Metric(TopEdgeMetric::Bounds),
            BottomEdge::Metric(BottomEdgeMetric::Bounds),
            text.size,
            TextEdgeBounds::Glyph(g.id),
        );
    }
}

/// Convert `scene::Affine2` into tessellator `Affine2x3`.
fn affine2x3_from_scene(xf: Affine2) -> crate::font::tessellate::Affine2x3 {
    crate::font::tessellate::Affine2x3 {
        a: xf.m[0][0],
        b: xf.m[0][1],
        c: xf.m[1][0],
        d: xf.m[1][1],
        tx: xf.m[2][0],
        ty: xf.m[2][1],
    }
}

/// Multiply two `Affine2x3` transforms: out = a * b.
fn mul_affine2x3(
    a: crate::font::tessellate::Affine2x3,
    b: crate::font::tessellate::Affine2x3,
) -> crate::font::tessellate::Affine2x3 {
    crate::font::tessellate::Affine2x3 {
        a: a.a * b.a + a.c * b.b,
        b: a.b * b.a + a.d * b.b,
        c: a.a * b.c + a.c * b.d,
        d: a.b * b.c + a.d * b.d,
        tx: a.a * b.tx + a.c * b.ty + a.tx,
        ty: a.b * b.tx + a.d * b.ty + a.ty,
    }
}

/// Append `src` into `dst` after transforming src positions by `xf`.
fn append_mesh_with_transform(
    dst: &mut Mesh2D,
    src: &Mesh2D,
    xf: crate::font::tessellate::Affine2x3,
) {
    let base = dst.positions.len();
    assert!(
        base + src.positions.len() <= u16::MAX as usize,
        "append_mesh_with_transform: vertex count overflow for u16 indices"
    );

    dst.positions.extend(src.positions.iter().map(|p| {
        let (x, y) = xf.transform_point(p[0], p[1]);
        [x, y]
    }));

    let base_u16 = base as u16;
    dst.indices
        .extend(src.indices.iter().copied().map(|i| base_u16 + i));
}

/// Glyph cache key: glyph ID + scale bits.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct GlyphCacheKey {
    glyph_id: u16,
    sx_bits: u32,
    sy_bits: u32,
}
impl Hash for GlyphCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.glyph_id.hash(state);
        self.sx_bits.hash(state);
        self.sy_bits.hash(state);
    }
}

/// Cache for tessellated glyph meshes (scaled to pt, no translation).
#[derive(Debug, Default)]
struct GlyphMeshCache {
    inner: HashMap<GlyphCacheKey, Option<Mesh2D>>,
}
impl GlyphMeshCache {
    fn get_or_insert_with(
        &mut self,
        key: GlyphCacheKey,
        f: impl FnOnce() -> Option<Mesh2D>,
    ) -> Option<&Mesh2D> {
        let entry = self.inner.entry(key).or_insert_with(f);
        entry.as_ref()
    }
}

/// Convert `ttf-parser` outline callbacks into a `lyon::path::Path`.
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
        if self.contour_open {
            self.builder.close();
            self.contour_open = false;
        }
        self.builder.build()
    }
}
impl ttf_parser::OutlineBuilder for LyonOutlineBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        if self.contour_open {
            self.builder.close();
            self.contour_open = false;
        }
        self.builder.begin(lyon_point(x, y));
        self.contour_open = true;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.builder.line_to(lyon_point(x, y));
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.builder
            .quadratic_bezier_to(lyon_point(x1, y1), lyon_point(x, y));
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.builder
            .cubic_bezier_to(lyon_point(x1, y1), lyon_point(x2, y2), lyon_point(x, y));
    }

    fn close(&mut self) {
        if self.contour_open {
            self.builder.close();
            self.contour_open = false;
        }
    }
}

/// Convert Typst `Transform` into our `scene::Affine2`.
fn affine2_from_typst_transform(t: Transform) -> Affine2 {
    let sx = t.sx.get() as f32;
    let sy = t.sy.get() as f32;
    let kx = t.kx.get() as f32;
    let ky = t.ky.get() as f32;
    let tx = t.tx.to_pt() as f32;
    let ty = t.ty.to_pt() as f32;

    Affine2 {
        m: [[sx, ky, 0.0], [kx, sy, 0.0], [tx, ty, 1.0]],
    }
}

/// Append a line segment as a thin rectangle mesh (two triangles).
fn append_line_as_rect(mesh: &mut Mesh2D, a: [f32; 2], b: [f32; 2], thickness_pt: f32) {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let len = (dx * dx + dy * dy).sqrt();

    if len < 1e-6 {
        return;
    }

    let nx = -dy / len;
    let ny = dx / len;
    let half = 0.5 * thickness_pt;

    let p0 = [a[0] + nx * half, a[1] + ny * half];
    let p1 = [a[0] - nx * half, a[1] - ny * half];
    let p2 = [b[0] - nx * half, b[1] - ny * half];
    let p3 = [b[0] + nx * half, b[1] + ny * half];

    let base = mesh.positions.len() as u16;
    mesh.positions.extend_from_slice(&[p0, p1, p2, p3]);
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}
