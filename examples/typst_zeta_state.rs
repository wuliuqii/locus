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
//!   - `Text` items as **debug rectangles** (approximate), so we can validate layout without glyph outlines
//!
//! Non-goals (for now):
//! - Glyph outline rendering (next phase).
//! - Full shape geometry support beyond simple lines.
//!
//! Notes:
//! - Typst's coordinate system here is page-local in `pt` units.
//! - We accumulate `Group.transform` and item positions so rules render in correct locations.

use std::{sync::Arc, time::Instant};

use anyhow::Context as _;
use winit::window::Window;

use locus::{
    render::{app::AppState, gpu::Gpu, mesh_renderer::MeshRenderer},
    scene::{Affine2, Mesh2D, Mobject2D, Rgba, Scene2D},
};

// In this file we need Typst's public types (`layout`, `visualize`).
// The `locus::typst` module is our integration layer, but it doesn't re-export these namespaces.
use typst::{layout, visualize};

/// Demo state for "Typst ζ → extract line shapes → render".
pub struct State {
    pub window: Arc<Window>,
    pub gpu: Gpu,

    pub scene: Scene2D,
    pub renderer: MeshRenderer,

    // Simple camera animation (optional).
    start_time: Instant,
    base_zoom: f32,
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
        // - `text_dbg_mesh`: approximate text rectangles (debug overlay)
        let mut line_mesh = Mesh2D::default();
        let mut text_dbg_mesh = Mesh2D::default();

        let mut stats = ExtractStats::default();
        extract_primitives_into_meshes(
            &compiled.document,
            &mut line_mesh,
            &mut text_dbg_mesh,
            &mut stats,
        );

        // Log extraction stats once at startup so `timeout ... cargo run --example typst_zeta`
        // still gives useful feedback even if the window is killed quickly.
        log::info!(
            "typst_zeta: extracted lines={} text_dbg_rects={} groups={} shapes={} texts={} pages={}",
            stats.lines,
            stats.text_debug_rects,
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

        // Debug overlay: approximate text boxes.
        scene.add_root(
            Mobject2D::new("typst_text_dbg")
                .with_mesh(text_dbg_mesh)
                .with_fill(Rgba {
                    r: 0.30,
                    g: 0.85,
                    b: 0.95,
                    a: 0.55,
                }),
        );

        // Frame the camera around extracted geometry (prefer text overlay if present).
        //
        // This avoids framing around a single thin rule which can feel "too zoomed in".
        if stats.text_debug_rects > 0 {
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
    stats: &mut ExtractStats,
) {
    stats.pages = doc.pages.len();

    for page in &doc.pages {
        let frame = &page.frame;
        walk_frame_for_primitives(frame, Affine2::IDENTITY, line_out, text_dbg_out, stats);
    }
}

fn walk_frame_for_primitives(
    frame: &layout::Frame,
    world_from_frame: Affine2,
    line_out: &mut Mesh2D,
    text_dbg_out: &mut Mesh2D,
    stats: &mut ExtractStats,
) {
    for (pos, item) in frame.items() {
        match item {
            layout::FrameItem::Group(group) => {
                stats.groups += 1;
                let world_from_group =
                    world_from_frame.mul(affine2_from_typst_transform(group.transform));
                walk_frame_for_primitives(
                    &group.frame,
                    world_from_group,
                    line_out,
                    text_dbg_out,
                    stats,
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
            layout::FrameItem::Text(_text) => {
                // Typst text items are shaped runs, but the public API surface doesn't expose
                // reliable glyph boxes for all versions. For Phase A layout debugging, we draw
                // a conservative placeholder rectangle at the text position.
                //
                // This still answers the key question: "where did Typst place text items?"
                stats.texts += 1;

                let world_from_item = world_from_frame.mul(Affine2::translate(
                    pos.x.to_pt() as f32,
                    pos.y.to_pt() as f32,
                ));

                // Conservative box size (pt). This is intentionally "visible" rather than accurate.
                // We'll replace this with real glyph boxes once we wire text shaping data.
                let w = 6.0f32;
                let h = 10.0f32;

                append_axis_aligned_rect_transformed(text_dbg_out, world_from_item, w, h);
                stats.text_debug_rects += 1;
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

/// Append an axis-aligned rectangle centered at the item's local origin, transformed by `world_from_item`.
///
/// This is used for **debug text boxes**. The rectangle is:
/// - centered at (0, 0) in item-local space
/// - width `w_pt`, height `h_pt` in pt
fn append_axis_aligned_rect_transformed(
    mesh: &mut Mesh2D,
    world_from_item: Affine2,
    w_pt: f32,
    h_pt: f32,
) {
    let hw = 0.5 * w_pt;
    let hh = 0.5 * h_pt;

    let (x0, y0) = world_from_item.transform_point(-hw, -hh);
    let (x1, y1) = world_from_item.transform_point(hw, -hh);
    let (x2, y2) = world_from_item.transform_point(hw, hh);
    let (x3, y3) = world_from_item.transform_point(-hw, hh);

    let base = mesh.positions.len() as u16;
    mesh.positions
        .extend_from_slice(&[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]);
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
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
