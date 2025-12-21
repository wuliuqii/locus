//! Typst ζ demo state: compile the canonical validation formula and render extracted primitives.
//!
//! This is a **state-only** module (no `fn main()`), meant to be used by an example wrapper:
//! - `examples/typst_zeta.rs` (thin wrapper) and/or by `src/main.rs` via `#[path = ...]`.
//!
//! Current scope (Phase A, incremental):
//! - Compile the canonical math formula: `locus::typst::math::ZETA_VALIDATION_FORMULA`.
//! - Traverse the resulting `PagedDocument` frames.
//! - Extract:
//!   - `Shape::Line(...)` items and render them as stroked rectangles (very thin quads)
//!   - `Text` items into:
//!     - **debug rectangles** (baseline-relative), so we can validate layout without glyph outlines
//!     - **placeholder "glyph outline" rectangles** (one box per glyph), as a bridge towards real outlines
//!
//! Non-goals (for now):
//! - Real glyph outline rendering (needs robust font mapping + ttf parsing pipeline wired to Typst fonts).
//! - Full shape geometry support beyond simple lines.
//!
//! Notes:
//! - Typst's coordinate system here is page-local in `pt` units.
//! - We accumulate `Group.transform` and item positions so rules render in correct locations.
//! - Text item internals are version-sensitive; we log a debug representation once (rate-limited)
//!   and derive better debug boxes when possible.
//!
//! Refinement:
//! - Debug text boxes are baseline-relative and are scaled by an *inferred*
//!   script level derived from group transforms (sub/superscripts tend to be downscaled by Typst).
//!
//! Placeholder glyph outline pipeline (this patch):
//! - We iterate `TextItem.glyphs` and emit a small baseline-relative rectangle for each glyph.
//! - This verifies the **glyph-level placement** pipeline before wiring real outline extraction.
//! - The actual outline extraction will replace the glyph rectangles with lyon-tessellated paths.

use std::{sync::Arc, time::Instant};

use anyhow::Context as _;
use winit::window::Window;

use locus::{
    render::{app::AppState, gpu::Gpu, mesh_renderer::MeshRenderer},
    scene::{Affine2, Mesh2D, Mobject2D, Rgba, Scene2D},
};

// In this file we need Typst's public types (`layout`, `visualize`).
// The `locus::typst` module is our integration layer, but it doesn't re-export these namespaces.
use typst::{layout, text::TextItem, visualize};

// `Abs` is Typst's absolute length type; we convert it to pt for debug visualization.
use typst::layout::Abs;
use typst::text::{BottomEdge, BottomEdgeMetric, TopEdge, TopEdgeMetric};

/// Demo state for "Typst ζ → extract primitives → render".
pub struct State {
    pub window: Arc<Window>,
    pub gpu: Gpu,

    pub scene: Scene2D,
    pub renderer: MeshRenderer,

    // Simple camera animation (optional).
    start_time: Instant,
    base_zoom: f32,

    // Debug: log the first few Text items once so we can learn the public surface of typst 0.14's
    // `layout::TextItem` on this build and decide what we can use for better debug boxes.
    text_debug_logged: usize,

    // Debug: log the first few "glyph bridge" details.
    glyph_bridge_logged: usize,
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let gpu = Gpu::new(window.clone()).await?;

        let mut scene = Scene2D::new();
        scene
            .camera
            .set_viewport_px(gpu.size.width.max(1), gpu.size.height.max(1));

        // Compile the canonical formula.
        let compiled = locus::typst::engine::compile_zeta_validation()
            .context("typst: failed to compile ζ validation formula")?;

        // Extract primitives from the compiled document.
        //
        // - `line_mesh`: fraction bars / rule strokes (as thin quads)
        // - `text_dbg_mesh`: baseline-relative text boxes (debug overlay)
        // - `glyph_dbg_mesh`: per-glyph boxes (placeholder "glyph outline" pipeline)
        let mut line_mesh = Mesh2D::default();
        let mut text_dbg_mesh = Mesh2D::default();
        let mut glyph_dbg_mesh = Mesh2D::default();

        let mut stats = ExtractStats::default();

        // Extract once with logging enabled for a small number of Text items.
        // This is intentionally bounded so `timeout ... cargo run --example typst_zeta` remains usable.
        let mut text_debug_logged = 0usize;
        let mut glyph_bridge_logged = 0usize;
        extract_primitives_into_meshes(
            &compiled.document,
            &mut line_mesh,
            &mut text_dbg_mesh,
            &mut glyph_dbg_mesh,
            &mut stats,
            &mut text_debug_logged,
            &mut glyph_bridge_logged,
        );

        // Log extraction stats once at startup so `timeout ... cargo run --example typst_zeta`
        // still gives useful feedback even if the window is killed quickly.
        log::info!(
            "typst_zeta: extracted lines={} text_dbg_rects={} glyph_dbg_rects={} groups={} shapes={} texts={} pages={}",
            stats.lines,
            stats.text_debug_rects,
            stats.glyph_debug_rects,
            stats.groups,
            stats.shapes,
            stats.texts,
            compiled.document.pages.len()
        );

        // Add to scene (world/local are in pt).
        scene.add_root(
            Mobject2D::new("typst_lines")
                .with_mesh(line_mesh)
                .with_fill(Rgba {
                    r: 0.90,
                    g: 0.90,
                    b: 0.95,
                    a: 1.0,
                }),
        );

        // Debug overlay: baseline-relative text boxes.
        scene.add_root(
            Mobject2D::new("typst_text_dbg")
                .with_mesh(text_dbg_mesh)
                .with_fill(Rgba {
                    r: 0.30,
                    g: 0.85,
                    b: 0.95,
                    a: 0.40,
                }),
        );

        // Placeholder glyph overlay: per-glyph boxes.
        scene.add_root(
            Mobject2D::new("typst_glyph_dbg")
                .with_mesh(glyph_dbg_mesh)
                .with_fill(Rgba {
                    r: 0.95,
                    g: 0.45,
                    b: 0.25,
                    a: 0.55,
                }),
        );

        // Frame the camera around extracted geometry (prefer glyph overlay if present).
        if stats.glyph_debug_rects > 0 {
            if let Some(root) = scene.get("typst_glyph_dbg") {
                let bounds = root.compute_local_bounds();
                scene.camera.frame_bounds(bounds, 60.0, 0.85);
            }
        } else if stats.text_debug_rects > 0 {
            if let Some(root) = scene.get("typst_text_dbg") {
                let bounds = root.compute_local_bounds();
                scene.camera.frame_bounds(bounds, 60.0, 0.85);
            }
        } else if stats.lines > 0 {
            if let Some(root) = scene.get("typst_lines") {
                let bounds = root.compute_local_bounds();
                // Conservative padding; rules can be thin so give it some breathing room.
                scene.camera.frame_bounds(bounds, 60.0, 0.85);
            }
        }

        let renderer = MeshRenderer::new(&gpu)?;
        let base_zoom = scene.camera.zoom;

        Ok(Self {
            window,
            gpu,
            scene,
            renderer,
            start_time: Instant::now(),
            base_zoom,
            text_debug_logged,
            glyph_bridge_logged,
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.gpu.resize(new_size);
        self.scene
            .camera
            .set_viewport_px(self.gpu.size.width.max(1), self.gpu.size.height.max(1));
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        // Gentle camera zoom "breathing" to validate scaling stability.
        let t = self.start_time.elapsed().as_secs_f32();
        self.scene.camera.zoom = self.base_zoom * (1.0 + 0.03 * (t * 0.8).sin());

        let (surface_texture, view) = match self.gpu.acquire_frame() {
            Ok(v) => v,
            Err(wgpu::SurfaceError::Outdated) | Err(wgpu::SurfaceError::Lost) => {
                self.gpu.resize(self.gpu.size);
                self.window.request_redraw();
                return Ok(());
            }
            Err(wgpu::SurfaceError::Timeout) => {
                self.window.request_redraw();
                return Ok(());
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                return Err(anyhow::anyhow!("wgpu SurfaceError::OutOfMemory"));
            }
            Err(wgpu::SurfaceError::Other) => {
                self.gpu.resize(self.gpu.size);
                self.window.request_redraw();
                return Ok(());
            }
        };

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Typst Zeta Encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Typst Zeta Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.06,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let mut items = self.scene.flatten();
            items.sort_by_key(|it| it.z);

            self.renderer
                .draw_items(&self.gpu, &mut pass, &self.scene.camera, &items)?;
        }

        self.gpu.queue.submit(Some(encoder.finish()));
        self.window.pre_present_notify();
        surface_texture.present();

        // Continuous animation.
        self.window.request_redraw();

        Ok(())
    }
}

impl AppState for State {
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        Self::resize(self, new_size)
    }

    fn render(&mut self) -> anyhow::Result<()> {
        Self::render(self)
    }

    fn request_redraw(&self) {
        self.window.request_redraw();
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct ExtractStats {
    pages: usize,
    groups: usize,
    shapes: usize,
    texts: usize,

    lines: usize,
    text_debug_rects: usize,
    glyph_debug_rects: usize,

    // Debug-only: how many text items we classified as scripts (sub/sup) vs. normal.
    text_script: usize,
    text_normal: usize,

    // Debug-only: how many text items we attempted to bridge into glyph data.
    glyph_bridge_attempts: usize,
}

/// Extraction context carried through recursion.
///
/// Goal:
/// - Provide a cheap, deterministic heuristic for detecting "script contexts"
///   (sub/superscripts tend to be downscaled by Typst).
///
/// This is a stepping stone towards real glyph metrics extraction.
#[derive(Debug, Default, Clone, Copy)]
struct ExtractCtx {
    /// 0 = normal, 1+ = likely sub/superscript nesting.
    script_level: u8,
}

impl ExtractCtx {
    fn with_group_transform(self, t: layout::Transform) -> Self {
        // Heuristic: sub/superscripts are often scaled down (e.g. ~70%).
        // We treat any significant downscale as "entering a script level".
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
        // Apply a gentle downscale per inferred level.
        // This is for debug overlays only.
        match self.script_level {
            0 => 1.0,
            1 => 0.72,
            _ => 0.62,
        }
    }
}

/// A baseline-relative debug rectangle description.
///
/// This makes text overlay easier to interpret:
/// - `w_pt`: width
/// - `above_baseline_pt`: height above baseline
/// - `below_baseline_pt`: depth below baseline
/// - `scale`: extra scale (e.g. script scaling)
#[derive(Debug, Clone, Copy)]
struct BaselineBox {
    w_pt: f32,
    above_baseline_pt: f32,
    below_baseline_pt: f32,
    scale: f32,
}

impl BaselineBox {
    fn height_pt(self) -> f32 {
        self.above_baseline_pt + self.below_baseline_pt
    }
}

/// Extract primitives from the document and append them to two meshes:
/// - `line_out`: stroked rules (currently only `Geometry::Line`)
/// - `text_dbg_out`: approximate debug boxes for text items
///
/// Implementation notes:
/// - We walk each page frame recursively (FrameItem::Group, FrameItem::Shape, FrameItem::Text).
/// - We **accumulate group transforms and item positions** into a single affine transform.
/// - Debug text boxes are approximate and intentionally conservative (to visualize layout).
fn extract_primitives_into_meshes(
    doc: &layout::PagedDocument,
    line_out: &mut Mesh2D,
    text_dbg_out: &mut Mesh2D,
    glyph_dbg_out: &mut Mesh2D,
    stats: &mut ExtractStats,
    text_debug_logged: &mut usize,
    glyph_bridge_logged: &mut usize,
) {
    stats.pages = doc.pages.len();

    // Track script scaling based on transforms. We start at "normal" (level 0).
    let ctx = ExtractCtx::default();

    for page in &doc.pages {
        let frame = &page.frame;
        walk_frame_for_primitives(
            frame,
            Affine2::IDENTITY,
            ctx,
            line_out,
            text_dbg_out,
            glyph_dbg_out,
            stats,
            text_debug_logged,
            glyph_bridge_logged,
        );
    }
}

fn walk_frame_for_primitives(
    frame: &layout::Frame,
    world_from_frame: Affine2,
    ctx: ExtractCtx,
    line_out: &mut Mesh2D,
    text_dbg_out: &mut Mesh2D,
    glyph_dbg_out: &mut Mesh2D,
    stats: &mut ExtractStats,
    text_debug_logged: &mut usize,
    glyph_bridge_logged: &mut usize,
) {
    for (pos, item) in frame.items() {
        match item {
            layout::FrameItem::Group(group) => {
                stats.groups += 1;

                // Compose: parent(frame) * group.transform.
                let t = affine2_from_typst_transform(group.transform);
                let world_from_group = world_from_frame.mul(t);

                // Infer "script level" based on group scale.
                let child_ctx = ctx.with_group_transform(group.transform);

                walk_frame_for_primitives(
                    &group.frame,
                    world_from_group,
                    child_ctx,
                    line_out,
                    text_dbg_out,
                    glyph_dbg_out,
                    stats,
                    text_debug_logged,
                    glyph_bridge_logged,
                );
            }
            layout::FrameItem::Shape(shape, _span) => {
                stats.shapes += 1;
                let world_from_item = world_from_frame.mul(Affine2::translate(
                    pos.x.to_pt() as f32,
                    pos.y.to_pt() as f32,
                ));
                extract_shape_lines(world_from_item, shape, line_out, stats);
            }
            layout::FrameItem::Text(text) => {
                // For Phase A layout debugging, we draw:
                // - one baseline-relative debug rectangle per TextItem (run-level)
                // - one baseline-relative debug rectangle per glyph (glyph-level placeholder)
                //
                // This gets us very close to the eventual outline pipeline, but without requiring
                // font mapping yet.
                stats.texts += 1;

                if *text_debug_logged < 6 {
                    log::info!("typst_zeta: Text item debug: {:?}", text);
                    *text_debug_logged += 1;
                }

                // --- Glyph bridge (Phase A bring-up) ---
                // We log glyph counts and some basic font sizing information to prepare for
                // outline rendering.
                stats.glyph_bridge_attempts += 1;
                if *glyph_bridge_logged < 6 {
                    let glyph_count = text.glyphs.len();
                    log::info!(
                        "typst_zeta: glyph bridge: script_level={} glyphs={} size={:?}",
                        ctx.script_level,
                        glyph_count,
                        text.size
                    );
                    *glyph_bridge_logged += 1;
                }

                let world_from_item = world_from_frame.mul(Affine2::translate(
                    pos.x.to_pt() as f32,
                    pos.y.to_pt() as f32,
                ));

                // Derive a baseline-relative debug box from real metrics, then apply script scaling.
                let mut bb = derive_text_debug_box(text);
                bb.scale *= ctx.script_scale();

                if ctx.script_level > 0 {
                    stats.text_script += 1;
                } else {
                    stats.text_normal += 1;
                }

                append_baseline_rect_transformed(text_dbg_out, world_from_item, bb);
                stats.text_debug_rects += 1;

                // Placeholder glyph mesh: one box per glyph.
                //
                // Strategy:
                // - Use each glyph's advance to place successive boxes along the baseline.
                // - Use font edges (bounds) for vertical extents.
                //
                // This approximates the outline placement pipeline enough to validate the glue.
                append_glyph_boxes(glyph_dbg_out, world_from_item, text, bb.scale, stats);
            }
            _ => {}
        }
    }
}

fn extract_shape_lines(
    world_from_item: Affine2,
    shape: &visualize::Shape,
    out: &mut Mesh2D,
    stats: &mut ExtractStats,
) {
    // We only support `geometry: Line(...)` for now.
    if let visualize::Geometry::Line(delta) = &shape.geometry {
        let (x0, y0) = world_from_item.transform_point(0.0, 0.0);
        let (x1, y1) =
            world_from_item.transform_point(delta.x.to_pt() as f32, delta.y.to_pt() as f32);

        let thickness_pt = shape
            .stroke
            .as_ref()
            .map(|s| s.thickness.to_pt() as f32)
            .unwrap_or(0.75)
            .max(0.25);

        append_line_as_rect(out, [x0, y0], [x1, y1], thickness_pt);
        stats.lines += 1;
    }
}

/// Emit a per-glyph placeholder mesh for a shaped Typst `TextItem`.
///
/// This is a bridge step towards real glyph outline rendering:
/// - It uses glyph advances to place boxes.
/// - It uses font edges (bounds) to size boxes vertically.
///
/// The boxes are filled quads and use whatever fill color the scene assigns.
fn append_glyph_boxes(
    mesh: &mut Mesh2D,
    world_from_item: Affine2,
    text: &TextItem,
    scale: f32,
    stats: &mut ExtractStats,
) {
    use typst::text::TextEdgeBounds;

    let mut pen_x_pt = 0.0f32;

    for g in text.glyphs.iter() {
        // Horizontal advance for this glyph (in pt).
        //
        // `x_advance` is in `Em` (relative to font size). Convert it to `Abs` at the
        // TextItem's font size, then to pt.
        let adv_pt = g.x_advance.at(text.size).to_pt() as f32;

        // Vertical extents: bounds-based edges for this specific glyph.
        let (t, b) = text.font.edges(
            TopEdge::Metric(TopEdgeMetric::Bounds),
            BottomEdge::Metric(BottomEdgeMetric::Bounds),
            text.size,
            TextEdgeBounds::Glyph(g.id),
        );

        let above = (t.to_pt() as f32).clamp(2.0, 80.0);
        let below = ((-b.to_pt() as f32).max(0.0)).clamp(0.0, 80.0);

        // Use the glyph advance as width, but ensure it's visible.
        let w = (adv_pt.max(3.0)).clamp(2.0, 200.0);

        let bb = BaselineBox {
            w_pt: w,
            above_baseline_pt: above,
            below_baseline_pt: below,
            scale,
        };

        // Apply an extra translation by pen position within the item.
        let world_from_glyph = world_from_item.mul(Affine2::translate(pen_x_pt, 0.0));
        append_baseline_rect_transformed(mesh, world_from_glyph, bb);

        stats.glyph_debug_rects += 1;
        pen_x_pt += adv_pt;
    }
}

/// Append a line segment as a thin rectangle mesh (two triangles).
///
/// - `a`, `b` are endpoints in pt-space.
/// - `thickness_pt` is the rectangle thickness in pt.
///
/// This produces a *filled* quad. We currently ignore caps/joins/dashes.
fn append_line_as_rect(mesh: &mut Mesh2D, a: [f32; 2], b: [f32; 2], thickness_pt: f32) {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let len = (dx * dx + dy * dy).sqrt();

    // Skip degenerate lines.
    if len < 1e-6 {
        return;
    }

    // Perpendicular unit vector.
    let nx = -dy / len;
    let ny = dx / len;

    let half = 0.5 * thickness_pt;

    // Quad vertices around the segment.
    let p0 = [a[0] + nx * half, a[1] + ny * half];
    let p1 = [a[0] - nx * half, a[1] - ny * half];
    let p2 = [b[0] - nx * half, b[1] - ny * half];
    let p3 = [b[0] + nx * half, b[1] + ny * half];

    let base = mesh.positions.len() as u16;
    mesh.positions.extend_from_slice(&[p0, p1, p2, p3]);

    // Two triangles: (0,1,2) and (0,2,3)
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// Append a baseline-relative rectangle transformed by `world_from_item`.
///
/// Convention:
/// - In item-local space, the baseline is at y=0.
/// - We draw the rectangle from:
///   - x in [-w/2, +w/2]
///   - y in [-below, +above]
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

/// Baseline-relative debug box for a text item.
///
/// This uses only stable-ish information we have today:
/// - `Debug` output of `TextItem` (e.g. `Text("∑")`)
///
/// It intentionally does not try to be exact; it is meant to:
/// - show relative size differences
/// - show baseline alignment (above vs below)
fn derive_text_debug_box(text: &TextItem) -> BaselineBox {
    // Prefer real Typst metrics when available.
    //
    // TextItem fields in typst 0.14 (as constructed by typst-layout) include:
    // - font, size, text, glyphs
    // and methods like:
    // - width(): Abs
    //
    // We use:
    // - width as `w_pt`
    // - a conservative edge estimate from the font at this size:
    //   - top/bottom edges based on the first glyph (or zero bounds)
    //
    // If anything is missing or yields degenerate sizes, we fall back to heuristics.
    let width_abs: Abs = text.width();
    let mut w_pt = width_abs.to_pt() as f32;

    // Conservative defaults for vertical metrics when we can't infer edges.
    let mut above = 8.0f32;
    let mut below = 2.0f32;

    // Try to infer better vertical bounds using the font's edge metrics.
    // We use TextEdgeBounds::Zero as a safe fallback. If glyphs exist, we try the first glyph id.
    //
    // NOTE: We intentionally keep this robust and avoid deep coupling to private APIs.
    {
        use typst::text::TextEdgeBounds;

        let bounds = if let Some(g) = text.glyphs.first() {
            TextEdgeBounds::Glyph(g.id)
        } else {
            TextEdgeBounds::Zero
        };

        // The exact signature in typst 0.14 is: font.edges(top_edge, bottom_edge, size, bounds).
        // We don't have access to the full style chain here, so we pick common defaults:
        // - top_edge / bottom_edge are resolved by Typst's text system internally; for debug,
        //   we approximate using `TextEdgeBounds`.
        //
        // As a best-effort, if edges aren't accessible, we just keep defaults.
        let (t, b) = text.font.edges(
            TopEdge::Metric(TopEdgeMetric::Bounds),
            BottomEdge::Metric(BottomEdgeMetric::Bounds),
            text.size,
            bounds,
        );
        above = t.to_pt() as f32;
        below = (-b.to_pt() as f32).max(0.0);
    }

    // Guard against nonsense.
    if !w_pt.is_finite() || w_pt <= 0.0 {
        w_pt = 6.0;
    }
    if !above.is_finite() || above <= 0.0 {
        above = 8.0;
    }
    if !below.is_finite() || below < 0.0 {
        below = 2.0;
    }

    // Clamp so one weird glyph doesn't blow up framing.
    w_pt = w_pt.clamp(2.0, 200.0);
    above = above.clamp(2.0, 80.0);
    below = below.clamp(0.0, 80.0);

    // If the run is very narrow, use a minimum width so punctuation remains visible.
    w_pt = w_pt.max(3.0);

    BaselineBox {
        w_pt,
        above_baseline_pt: above,
        below_baseline_pt: below,
        scale: 1.0,
    }
}

/// Convert a Typst `Transform` into our `scene::Affine2`.
///
/// Typst transforms are affine 2D transforms with:
/// - scale/shear components
/// - translation components
///
/// We map them into our column-major 3x3 `Affine2` with column-vector convention.
///
/// Note:
/// - The exact field types on `typst::layout::Transform` are stable in typst 0.14 (as seen in logs).
/// - If Typst changes this struct in the future, this function will need an update.
fn affine2_from_typst_transform(t: layout::Transform) -> Affine2 {
    // The fields are percentages/lengths. We convert:
    // - sx/sy/ kx/ky as dimensionless factors
    // - tx/ty as pt
    let sx = t.sx.get() as f32;
    let sy = t.sy.get() as f32;
    let kx = t.kx.get() as f32;
    let ky = t.ky.get() as f32;
    let tx = t.tx.to_pt() as f32;
    let ty = t.ty.to_pt() as f32;

    // With column vectors, our matrix columns represent the transformed basis vectors plus translation:
    // [ a00 a01 0 ]
    // [ a10 a11 0 ]
    // [ tx  ty  1 ]
    //
    // Where:
    // - a00/a10 is X basis after transform
    // - a01/a11 is Y basis after transform
    //
    // We treat Typst's fields as:
    // - sx, sy scale
    // - kx, ky shear terms
    //
    // This matches the common 2D affine parameterization:
    // x' = sx*x + kx*y + tx
    // y' = ky*x + sy*y + ty
    Affine2 {
        m: [[sx, ky, 0.0], [kx, sy, 0.0], [tx, ty, 1.0]],
    }
}
