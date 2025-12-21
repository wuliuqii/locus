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
//!   - `Shape::Rect(...)` items and render them as filled quads (top-left origin)
//!   - `Shape::Curve(...)` items and render them as filled curves (Move/Line/Cubic/Close)
//!   - `Text` items into:
//!     - baseline-relative debug rectangles (run-level; optional overlay)
//!     - REAL glyph outlines (per-glyph), tessellated into triangles (optional)
//!
//! Debug toggles (environment variables):
//! - `LOCUS_TY_ZETA_SHOW_TEXT_DBG=0|1` (default: 0)
//! - `LOCUS_TY_ZETA_SHOW_GLYPHS=0|1`   (default: 1)
//! - `LOCUS_TY_ZETA_SHOW_LINES=0|1`    (default: 1)
//! - `LOCUS_TY_ZETA_SHOW_SHAPES=0|1`   (default: 1)
//!
//! Notes:
//! - Typst's coordinate system here is page-local in `pt` units.
//! - We accumulate `Group.transform` and item positions so rules render in correct locations.
//!
//! Glyph outline rendering strategy:
//! - For each `TextItem`, iterate `text.glyphs`
//! - For each glyph:
//!   - Extract outline from Typst's `Font` (TTF face) using the glyph ID
//!   - Tessellate outline via lyon into `scene::Mesh2D`
//!   - Place it using pen position + glyph offsets, then apply `world_from_item`
//!
//! Shape rendering strategy:
//! - We tessellate `Curve` geometry into a lyon `Path` and fill it.
//! - For `Rect`, we generate a rectangle `Path` and fill it.
//!
//! Performance note:
//! - This demo can tessellate many glyph outlines at startup.
//! - We cache per-glyph tessellations (by glyph ID and sizing parameters) to avoid repeated work.

use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::Arc,
    time::Instant,
};

fn env_flag(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "y" | "on" => true,
            "0" | "false" | "no" | "n" | "off" => false,
            _ => default,
        },
        Err(_) => default,
    }
}

use anyhow::Context as _;
use winit::window::Window;

use locus::{
    render::{app::AppState, gpu::Gpu, mesh_renderer::MeshRenderer},
    scene::{Affine2, Mesh2D, Mobject2D, Rgba, Scene2D},
};

use lyon::math::point as lyon_point;
use lyon::path::Path;

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
        // - `shape_mesh`: filled shapes (rect/curve) tessellated into triangles
        // - `text_dbg_mesh`: baseline-relative text boxes (debug overlay)
        // - `glyph_mesh`: real glyph outlines tessellated into triangles
        let mut line_mesh = Mesh2D::default();
        let mut shape_mesh = Mesh2D::default();
        let mut text_dbg_mesh = Mesh2D::default();
        let mut glyph_mesh = Mesh2D::default();

        let mut stats = ExtractStats::default();

        // Cache to avoid re-tessellating identical glyph outlines at the same size.
        let mut glyph_cache = GlyphMeshCache::default();

        // Extract once with logging enabled for a small number of Text items.
        // This is intentionally bounded so `timeout ... cargo run --example typst_zeta` remains usable.
        let mut text_debug_logged = 0usize;
        let mut glyph_bridge_logged = 0usize;
        extract_primitives_into_meshes(
            &compiled.document,
            &mut line_mesh,
            &mut shape_mesh,
            &mut text_dbg_mesh,
            &mut glyph_mesh,
            &mut glyph_cache,
            &mut stats,
            &mut text_debug_logged,
            &mut glyph_bridge_logged,
        );

        // Log extraction stats once at startup so `timeout ... cargo run --example typst_zeta`
        // still gives useful feedback even if the window is killed quickly.
        log::info!(
            "typst_zeta: extracted lines={} filled_shapes={} text_dbg_rects={} glyph_tris={} glyph_tess_calls={} glyph_calls={} groups={} shapes={} texts={} pages={}",
            stats.lines,
            stats.filled_shapes,
            stats.text_debug_rects,
            stats.glyph_triangles,
            stats.glyph_tess_calls,
            stats.glyph_calls,
            stats.groups,
            stats.shapes,
            stats.texts,
            compiled.document.pages.len()
        );

        // Environment toggles (see module docs).
        let show_lines = env_flag("LOCUS_TY_ZETA_SHOW_LINES", true);
        let show_text_dbg = env_flag("LOCUS_TY_ZETA_SHOW_TEXT_DBG", false);
        let show_glyphs = env_flag("LOCUS_TY_ZETA_SHOW_GLYPHS", true);
        let show_shapes = env_flag("LOCUS_TY_ZETA_SHOW_SHAPES", true);

        // Add to scene (world/local are in pt).
        if show_lines {
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
        }

        // Optional: filled shapes (rect/curve) extracted from Typst.
        if show_shapes {
            // `shape_mesh` is appended during extraction; if empty, this is a no-op at render time.
            scene.add_root(
                Mobject2D::new("typst_shapes")
                    .with_mesh(shape_mesh)
                    .with_fill(Rgba {
                        r: 0.95,
                        g: 0.95,
                        b: 0.95,
                        a: 0.90,
                    }),
            );
        }

        if show_text_dbg {
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
        }

        if show_glyphs {
            // Glyph outlines (tessellated fill).
            scene.add_root(
                Mobject2D::new("typst_glyphs")
                    .with_mesh(glyph_mesh)
                    .with_fill(Rgba {
                        r: 0.95,
                        g: 0.95,
                        b: 0.95,
                        a: 1.0,
                    }),
            );
        }

        // Frame the camera around extracted geometry (prefer glyph outlines if present).
        if show_glyphs && stats.glyph_triangles > 0 {
            if let Some(root) = scene.get("typst_glyphs") {
                let bounds = root.compute_local_bounds();
                scene.camera.frame_bounds(bounds, 60.0, 0.85);
            }
        } else if show_text_dbg && stats.text_debug_rects > 0 {
            if let Some(root) = scene.get("typst_text_dbg") {
                let bounds = root.compute_local_bounds();
                scene.camera.frame_bounds(bounds, 60.0, 0.85);
            }
        } else if show_lines && stats.lines > 0 {
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
    filled_shapes: usize,
    text_debug_rects: usize,
    glyph_triangles: usize,
    glyph_tess_calls: usize,
    glyph_calls: usize,

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
    shape_out: &mut Mesh2D,
    text_dbg_out: &mut Mesh2D,
    glyph_dbg_out: &mut Mesh2D,
    glyph_cache: &mut GlyphMeshCache,
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
            shape_out,
            text_dbg_out,
            glyph_dbg_out,
            glyph_cache,
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
    shape_out: &mut Mesh2D,
    text_dbg_out: &mut Mesh2D,
    glyph_dbg_out: &mut Mesh2D,
    glyph_cache: &mut GlyphMeshCache,
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
                    shape_out,
                    text_dbg_out,
                    glyph_dbg_out,
                    glyph_cache,
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
                extract_shape_geometry(world_from_item, shape, line_out, shape_out, stats);
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

                // Real glyph outlines: tessellate each glyph outline into triangles.
                //
                // This replaces the previous "glyph boxes" placeholder.
                append_glyph_outlines(
                    glyph_dbg_out,
                    glyph_cache,
                    world_from_item,
                    text,
                    bb.scale,
                    stats,
                );
            }
            _ => {}
        }
    }
}

fn extract_shape_geometry(
    world_from_item: Affine2,
    shape: &visualize::Shape,
    line_out: &mut Mesh2D,
    shape_out: &mut Mesh2D,
    stats: &mut ExtractStats,
) {
    match &shape.geometry {
        visualize::Geometry::Line(delta) => {
            let (x0, y0) = world_from_item.transform_point(0.0, 0.0);
            let (x1, y1) =
                world_from_item.transform_point(delta.x.to_pt() as f32, delta.y.to_pt() as f32);

            let thickness_pt = shape
                .stroke
                .as_ref()
                .map(|s| s.thickness.to_pt() as f32)
                .unwrap_or(0.75)
                .max(0.25);

            append_line_as_rect(line_out, [x0, y0], [x1, y1], thickness_pt);
            stats.lines += 1;
        }

        visualize::Geometry::Rect(size) => {
            // Rect has its origin at the top-left corner in Typst.
            // We'll tessellate a rectangle path and fill it.
            let w = size.x.to_pt() as f32;
            let h = size.y.to_pt() as f32;

            let mut b = Path::builder();
            b.begin(lyon_point(0.0, 0.0));
            b.line_to(lyon_point(w, 0.0));
            b.line_to(lyon_point(w, h));
            b.line_to(lyon_point(0.0, h));
            b.close();
            let path = b.build();

            let xf = locus::font::tessellate::Affine2x3 {
                a: world_from_item.m[0][0],
                b: world_from_item.m[0][1],
                c: world_from_item.m[1][0],
                d: world_from_item.m[1][1],
                tx: world_from_item.m[2][0],
                ty: world_from_item.m[2][1],
            };

            let before = shape_out.indices.len();
            let _ = locus::font::tessellate::append_tessellated_path(
                shape_out,
                &path,
                xf,
                locus::font::tessellate::TessellateOptions::default(),
            );
            let after = shape_out.indices.len();
            if after > before {
                stats.filled_shapes += 1;
            }
        }

        visualize::Geometry::Curve(curve) => {
            // Convert Typst curve items into a lyon path.
            // Points are relative to the item's origin.
            let mut b = Path::builder();
            let mut started = false;

            for item in curve.0.iter() {
                match item {
                    visualize::CurveItem::Move(p) => {
                        if started {
                            b.close();
                        }
                        b.begin(lyon_point(p.x.to_pt() as f32, p.y.to_pt() as f32));
                        started = true;
                    }
                    visualize::CurveItem::Line(p) => {
                        if !started {
                            b.begin(lyon_point(0.0, 0.0));
                            started = true;
                        }
                        b.line_to(lyon_point(p.x.to_pt() as f32, p.y.to_pt() as f32));
                    }
                    visualize::CurveItem::Cubic(p1, p2, p) => {
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
                    visualize::CurveItem::Close => {
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

            let xf = locus::font::tessellate::Affine2x3 {
                a: world_from_item.m[0][0],
                b: world_from_item.m[0][1],
                c: world_from_item.m[1][0],
                d: world_from_item.m[1][1],
                tx: world_from_item.m[2][0],
                ty: world_from_item.m[2][1],
            };

            let before = shape_out.indices.len();
            let _ = locus::font::tessellate::append_tessellated_path(
                shape_out,
                &path,
                xf,
                locus::font::tessellate::TessellateOptions::default(),
            );
            let after = shape_out.indices.len();
            if after > before {
                stats.filled_shapes += 1;
            }
        }
    }
}

/// Emit real glyph outline meshes for a shaped Typst `TextItem`.
///
/// Implementation:
/// - For each glyph:
///   - Get the outline via `ttf-parser` from Typst's `Font`
///   - Tessellate with lyon
///   - Place using pen position + glyph offsets
fn append_glyph_outlines(
    mesh: &mut Mesh2D,
    cache: &mut GlyphMeshCache,
    world_from_item: Affine2,
    text: &TextItem,
    scale: f32,
    stats: &mut ExtractStats,
) {
    // NOTE:
    // This uses Typst's internal font face access (`font.ttf()`) which returns a `ttf_parser::Face`.
    // We convert its outline callbacks into a lyon `Path`, then tessellate.
    let face = text.font.ttf();
    let upm = face.units_per_em() as f32;
    if upm <= 0.0 {
        return;
    }

    let font_units_to_pt = (text.size.to_pt() as f32) / upm;

    // Convert `scene::Affine2` (column-major 3x3) into tessellator `Affine2x3` once per text item.
    let world_from_item_2x3 = locus::font::tessellate::Affine2x3 {
        a: world_from_item.m[0][0],
        b: world_from_item.m[0][1],
        c: world_from_item.m[1][0],
        d: world_from_item.m[1][1],
        tx: world_from_item.m[2][0],
        ty: world_from_item.m[2][1],
    };

    let mut pen_x_pt = 0.0f32;
    let pen_y_pt = 0.0f32;

    for g in text.glyphs.iter() {
        // Advances/offsets are `Em` relative to font size.
        let adv_pt = g.x_advance.at(text.size).to_pt() as f32;
        let x_off_pt = g.x_offset.at(text.size).to_pt() as f32;
        let y_off_pt = g.y_offset.at(text.size).to_pt() as f32;

        // Transform parameters for this glyph.
        let sx = font_units_to_pt * scale;
        let sy = font_units_to_pt * scale;
        let tx = (pen_x_pt + x_off_pt) as f32;
        let ty = (pen_y_pt + y_off_pt) as f32;

        // Count glyph-level calls separately from tessellation calls.
        stats.glyph_calls += 1;

        // Cache key: glyph id + transform parameters that affect tessellation result.
        let key = GlyphCacheKey {
            glyph_id: g.id,
            sx_bits: sx.to_bits(),
            sy_bits: sy.to_bits(),
            // Translation is NOT part of the cached mesh, we apply it via transform composition below.
            // We cache a mesh tessellated at origin (tx=0, ty=0) and then translate via composed transform.
            //
            // Therefore we cache only scaling (sx/sy) and glyph id.
        };

        // Ensure we have a cached outline path tessellation for this glyph at this scale.
        // We tessellate in glyph-local space with scale, but without translation.
        let cached = cache.get_or_insert_with(key, || {
            let gid = ttf_parser::GlyphId(g.id);
            let mut builder = LyonOutlineBuilder::new();
            let bbox = face.outline_glyph(gid, &mut builder);
            if bbox.is_none() {
                return None;
            }

            let path = builder.build();

            let local_no_translate = locus::font::tessellate::Affine2x3 {
                a: sx,
                b: 0.0,
                c: 0.0,
                d: sy,
                tx: 0.0,
                ty: 0.0,
            };

            let mut tmp = Mesh2D::default();
            if locus::font::tessellate::append_tessellated_path(
                &mut tmp,
                &path,
                local_no_translate,
                locus::font::tessellate::TessellateOptions::default(),
            )
            .is_err()
            {
                return None;
            }

            Some(tmp)
        });

        if let Some(src) = cached {
            // Compose: out = world_from_item_2x3 * translate(tx, ty)
            let t_only = locus::font::tessellate::Affine2x3 {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 1.0,
                tx,
                ty,
            };

            let composed = mul_affine2x3(world_from_item_2x3, t_only);

            // Append cached mesh with transformed positions (no re-tessellation).
            let before_indices = mesh.indices.len();
            append_mesh_with_transform(mesh, src, composed);
            let after_indices = mesh.indices.len();
            let added_indices = after_indices.saturating_sub(before_indices);

            stats.glyph_triangles += added_indices / 3;
            stats.glyph_tess_calls += 1;
        }

        pen_x_pt += adv_pt;
    }
}

/// Glyph mesh cache key.
///
/// We cache tessellated meshes in glyph-local space (scaled to pt), but without translation.
/// Translation is applied when appending to the final mesh.
///
/// NOTE:
/// - We hash scale factors by their bit pattern to avoid float hashing pitfalls.
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

/// Cache for tessellated glyph meshes.
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

/// Multiply two `Affine2x3` transforms: out = a * b.
fn mul_affine2x3(
    a: locus::font::tessellate::Affine2x3,
    b: locus::font::tessellate::Affine2x3,
) -> locus::font::tessellate::Affine2x3 {
    locus::font::tessellate::Affine2x3 {
        a: a.a * b.a + a.c * b.b,
        b: a.b * b.a + a.d * b.b,
        c: a.a * b.c + a.c * b.d,
        d: a.b * b.c + a.d * b.d,
        tx: a.a * b.tx + a.c * b.ty + a.tx,
        ty: a.b * b.tx + a.d * b.ty + a.ty,
    }
}

/// Append `src` into `dst` after transforming src positions by `xf`.
///
/// This avoids re-tessellating cached glyph meshes; it just transforms vertices and appends indices.
fn append_mesh_with_transform(
    dst: &mut Mesh2D,
    src: &Mesh2D,
    xf: locus::font::tessellate::Affine2x3,
) {
    let base = dst.positions.len();
    // Keep u16 indices safety consistent with existing helpers.
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
        self.builder.begin(lyon::math::point(x, y));
        self.contour_open = true;
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.builder.line_to(lyon::math::point(x, y));
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.builder
            .quadratic_bezier_to(lyon::math::point(x1, y1), lyon::math::point(x, y));
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.builder.cubic_bezier_to(
            lyon::math::point(x1, y1),
            lyon::math::point(x2, y2),
            lyon::math::point(x, y),
        );
    }
    fn close(&mut self) {
        if self.contour_open {
            self.builder.close();
            self.contour_open = false;
        }
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
