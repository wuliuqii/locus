//! Example: Pythagoras teaching demo (Step 1) using the minimal `Timeline` system.
//!
//! Goal of this step:
//! - Show a right triangle on the left.
//! - Fade in labels "a", "b", "c", and the equation `a^2 + b^2 = c^2`.
//! - When the equation appears, dim the triangle to shift attention.
//! - Highlight the hypotenuse with a bright stroke (`edge_c`) alongside the equation.
//! - Subtly scale the equation in as it fades.
//! - Move the labels slightly into place to demonstrate animation basics.
//!
//! Run (recommended to avoid blocking your terminal session):
//! - `RUST_LOG=info timeout 10s cargo run --example pythagoras_step1`
//!
//! Notes:
//! - This demo is intentionally simple. It focuses on wiring the `anim::Timeline` into the
//!   winit render loop and animating `Mobject2D` properties.
//! - Geometry here is basic CPU meshes (triangles/quads) built directly.
//! - Labels are rendered as Typst glyph outline meshes.
//! - With alpha propagation in scene flatten, we keep Typst child mesh alpha at 1.0 and animate
//!   only the group/root alpha via the Timeline.
//!
//! Debugging:
//! - This example includes a one-time stderr dump of flattened draw items (mesh sizes, color, z,
//!   and world_from_local matrix) to diagnose blank-screen issues after transform refactors.

use std::{sync::Arc, time::Instant};

use anyhow::Context as _;
use winit::window::Window;

use locus::{
    anim::{AnimTarget, Ease, Keyframe, Timeline, Track},
    render::{app::AppState, gpu::Gpu, mesh_renderer::MeshRenderer},
    scene::{Affine2, Mesh2D, Mobject2D, Rgba, Scene2D},
    typst::demo::{TypstGroupOptions, compile_snippet_to_group_mobject_baseline},
};

// One-time debug dump to stderr (kept local to this example).
static DEBUG_DUMP_ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();

/// Demo state for step 1.
pub struct State {
    pub window: Arc<Window>,
    pub gpu: Gpu,
    pub scene: Scene2D,
    pub renderer: MeshRenderer,

    start_time: Instant,
    timeline: Timeline,

    // Camera baseline (framed once).
    base_zoom: f32,
    base_center: [f32; 2],
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let gpu = Gpu::new(window.clone()).await?;

        let mut scene = Scene2D::new();
        scene
            .camera
            .set_viewport_px(gpu.size.width.max(1), gpu.size.height.max(1));

        // --- Build geometry (triangle + Typst labels as baseline-anchored group mobjects) ---
        //
        // Coordinate convention:
        // - world units are pt
        // - y-up in our scene
        // - we place the construction around the origin and then frame the camera
        let tri = right_triangle_mesh([0.0, 0.0], 240.0, 160.0);

        // Triangle object.
        scene.add_root(Mobject2D::new("tri").with_mesh(tri).with_fill(Rgba {
            r: 0.20,
            g: 0.25,
            b: 0.30,
            a: 1.0,
        }));

        // Highlight the hypotenuse (edge c) as a thin rectangle (stroke approximation).
        // Right triangle vertices:
        // - O = (0, 0)
        // - A = (a, 0) = (240, 0)
        // - B = (0, b) = (0, 160)
        //
        // Hypotenuse is A -> B.
        scene.add_root(
            Mobject2D::new("edge_c")
                .with_mesh(line_as_rect_mesh([240.0, 0.0], [0.0, 160.0], 2.5))
                .with_fill(Rgba {
                    r: 0.95,
                    g: 0.95,
                    b: 0.96,
                    a: 0.0,
                }),
        );

        // Labels as Typst-rendered glyph outline meshes (baseline-anchored groups).
        //
        // Important: we keep the same root names ("label_a"/"label_b"/"label_c"/"label_eq") so the Timeline
        // tracks continue to target the correct objects.
        //
        // Visibility note:
        // - The Typst glyph triangles live on the child mesh node.
        // - `Timeline` animates `fill.a` on the target root/group.
        // - Scene flatten propagates parent alpha down the tree.
        //
        // Therefore:
        // - keep child mesh alpha at 1.0
        // - initialize the group alpha at 0.0 and let the Timeline fade the whole group in
        let mut label_a = compile_snippet_to_group_mobject_baseline(
            "$a$",
            TypstGroupOptions {
                name: "label_a".to_string(),
                child_name: "mesh".to_string(),
                fill: Rgba {
                    r: 0.90,
                    g: 0.55,
                    b: 0.25,
                    a: 1.0,
                },
                local_from_parent: Affine2::translate(-40.0, 70.0),
                render: Default::default(),
                include_shapes: false,
                include_text_debug: false,
            },
        )
        .context("failed to build typst label_a")?;
        // Fade is driven by group/root alpha (propagated to children during flatten).
        label_a.fill.a = 0.0;
        scene.add_root(label_a);

        let mut label_b = compile_snippet_to_group_mobject_baseline(
            "$b$",
            TypstGroupOptions {
                name: "label_b".to_string(),
                child_name: "mesh".to_string(),
                fill: Rgba {
                    r: 0.35,
                    g: 0.85,
                    b: 0.45,
                    a: 1.0,
                },
                local_from_parent: Affine2::translate(110.0, -20.0),
                render: Default::default(),
                include_shapes: false,
                include_text_debug: false,
            },
        )
        .context("failed to build typst label_b")?;
        // Fade is driven by group/root alpha (propagated to children during flatten).
        label_b.fill.a = 0.0;
        scene.add_root(label_b);

        let mut label_c = compile_snippet_to_group_mobject_baseline(
            "$c$",
            TypstGroupOptions {
                name: "label_c".to_string(),
                child_name: "mesh".to_string(),
                fill: Rgba {
                    r: 0.45,
                    g: 0.65,
                    b: 0.95,
                    a: 1.0,
                },
                local_from_parent: Affine2::translate(70.0, 130.0),
                render: Default::default(),
                include_shapes: false,
                include_text_debug: false,
            },
        )
        .context("failed to build typst label_c")?;
        // Fade is driven by group/root alpha (propagated to children during flatten).
        label_c.fill.a = 0.0;
        scene.add_root(label_c);

        // Main equation label (baseline-anchored).
        let mut label_eq = compile_snippet_to_group_mobject_baseline(
            "$a^2 + b^2 = c^2$",
            TypstGroupOptions {
                name: "label_eq".to_string(),
                child_name: "mesh".to_string(),
                fill: Rgba {
                    r: 0.95,
                    g: 0.95,
                    b: 0.96,
                    a: 1.0,
                },
                // Start slightly lower and slightly smaller; Timeline will move + scale it into place.
                local_from_parent: Affine2::translate(-10.0, 210.0).mul(Affine2::scale(0.96, 0.96)),
                render: Default::default(),
                include_shapes: false,
                include_text_debug: false,
            },
        )
        .context("failed to build typst label_eq")?;
        // Fade is driven by group/root alpha (propagated to children during flatten).
        label_eq.fill.a = 0.0;
        scene.add_root(label_eq);

        // Frame camera around triangle + all labels so Typst glyphs stay in view.
        //
        // NOTE: `compute_local_bounds()` includes children bounds (transforming their corners),
        // so using it on the label group roots accounts for the glyph mesh child.
        let mut frame_bounds = if let Some(root) = scene.get("tri") {
            root.compute_local_bounds()
        } else {
            locus::scene::Aabb2::empty()
        };

        for name in ["label_a", "label_b", "label_c", "label_eq", "edge_c"] {
            if let Some(root) = scene.get(name) {
                frame_bounds = frame_bounds.union(root.compute_local_bounds());
            }
        }

        scene.camera.frame_bounds(frame_bounds, 80.0, 0.85);

        let base_zoom = scene.camera.zoom;
        let base_center = scene.camera.center_pt;

        // --- Timeline (Step 1) ---
        let mut timeline = Timeline::new();

        // Fade in and nudge labels into place.
        // We animate only translation and alpha.
        //
        // Each label:
        // - at t=0.0: alpha=0
        // - at t=0.6: alpha=1
        // - translate to a final position by t=0.9 (slight movement)
        //
        // "a" label: move right/down a bit
        timeline.add_track(
            Track::new_alpha(AnimTarget::Name("label_a".to_string())).with_keyframes(vec![
                Keyframe::at(0.0, 0.0).ease(Ease::OutCubic),
                Keyframe::at(0.6, 1.0).ease(Ease::OutCubic),
            ]),
        );
        timeline.add_track(
            Track::new_translate_x(AnimTarget::Name("label_a".to_string())).with_keyframes(vec![
                Keyframe::at(0.0, -40.0).ease(Ease::OutCubic),
                Keyframe::at(0.9, -20.0).ease(Ease::OutCubic),
            ]),
        );
        timeline.add_track(
            Track::new_translate_y(AnimTarget::Name("label_a".to_string())).with_keyframes(vec![
                Keyframe::at(0.0, 70.0).ease(Ease::OutCubic),
                Keyframe::at(0.9, 60.0).ease(Ease::OutCubic),
            ]),
        );

        // "b" label: move left/up a bit
        timeline.add_track(
            Track::new_alpha(AnimTarget::Name("label_b".to_string())).with_keyframes(vec![
                Keyframe::at(0.15, 0.0).ease(Ease::OutCubic),
                Keyframe::at(0.75, 1.0).ease(Ease::OutCubic),
            ]),
        );
        timeline.add_track(
            Track::new_translate_x(AnimTarget::Name("label_b".to_string())).with_keyframes(vec![
                Keyframe::at(0.15, 110.0).ease(Ease::OutCubic),
                Keyframe::at(1.05, 95.0).ease(Ease::OutCubic),
            ]),
        );
        timeline.add_track(
            Track::new_translate_y(AnimTarget::Name("label_b".to_string())).with_keyframes(vec![
                Keyframe::at(0.15, -20.0).ease(Ease::OutCubic),
                Keyframe::at(1.05, -5.0).ease(Ease::OutCubic),
            ]),
        );

        // "c" label: move slightly down
        timeline.add_track(
            Track::new_alpha(AnimTarget::Name("label_c".to_string())).with_keyframes(vec![
                Keyframe::at(0.30, 0.0).ease(Ease::OutCubic),
                Keyframe::at(0.90, 1.0).ease(Ease::OutCubic),
            ]),
        );
        timeline.add_track(
            Track::new_translate_y(AnimTarget::Name("label_c".to_string())).with_keyframes(vec![
                Keyframe::at(0.30, 130.0).ease(Ease::OutCubic),
                Keyframe::at(1.10, 118.0).ease(Ease::OutCubic),
            ]),
        );
        timeline.add_track(
            Track::new_translate_x(AnimTarget::Name("label_c".to_string())).with_keyframes(vec![
                Keyframe::at(0.30, 70.0).ease(Ease::OutCubic),
                Keyframe::at(1.10, 78.0).ease(Ease::OutCubic),
            ]),
        );

        // Equation: fade in, scale in, and settle upward a bit (baseline-anchored).
        timeline.add_track(
            Track::new_alpha(AnimTarget::Name("label_eq".to_string())).with_keyframes(vec![
                Keyframe::at(0.55, 0.0).ease(Ease::OutCubic),
                Keyframe::at(1.25, 1.0).ease(Ease::OutCubic),
            ]),
        );
        timeline.add_track(
            Track::new_scale(AnimTarget::Name("label_eq".to_string())).with_keyframes(vec![
                Keyframe::at(0.55, 0.96).ease(Ease::OutCubic),
                Keyframe::at(1.35, 1.00).ease(Ease::OutCubic),
            ]),
        );
        timeline.add_track(
            Track::new_translate_y(AnimTarget::Name("label_eq".to_string())).with_keyframes(vec![
                Keyframe::at(0.55, 210.0).ease(Ease::OutCubic),
                Keyframe::at(1.35, 195.0).ease(Ease::OutCubic),
            ]),
        );

        // When the equation appears, dim the triangle to shift attention.
        timeline.add_track(
            Track::new_alpha(AnimTarget::Name("tri".to_string())).with_keyframes(vec![
                Keyframe::at(0.55, 1.0).ease(Ease::OutCubic),
                Keyframe::at(1.35, 0.30).ease(Ease::OutCubic),
            ]),
        );

        // Highlight the hypotenuse alongside the equation.
        timeline.add_track(
            Track::new_alpha(AnimTarget::Name("edge_c".to_string())).with_keyframes(vec![
                Keyframe::at(0.55, 0.0).ease(Ease::OutCubic),
                Keyframe::at(1.15, 1.0).ease(Ease::OutCubic),
            ]),
        );

        let mut renderer = MeshRenderer::new(&gpu)?;
        // Debug: draw a full-screen triangle in clip space to validate pipeline output.
        // If you see a magenta triangle, the render pass + pipeline + presentation are working.
        renderer.set_debug_fullscreen_triangle(true);

        Ok(Self {
            window,
            gpu,
            scene,
            renderer,
            start_time: Instant::now(),
            timeline,
            base_zoom,
            base_center,
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.gpu.resize(new_size);
        self.scene
            .camera
            .set_viewport_px(self.gpu.size.width.max(1), self.gpu.size.height.max(1));
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        // Apply timeline.
        let t = self.start_time.elapsed().as_secs_f32();
        self.timeline.apply(&mut self.scene, t);

        // Sanity-check mode:
        // Force a fixed camera independent of any framing/bounds logic.
        //
        // If geometry still doesn't show up with this, the issue is likely in the camera->clip
        // transform, matrix conventions, or the renderer pipeline (not camera framing).
        self.scene.camera.center_pt = [120.0, 80.0];
        self.scene.camera.zoom = 0.01;

        // One-time debug dump of what we are about to draw (useful for diagnosing blank screens).
        DEBUG_DUMP_ONCE.get_or_init(|| {
            let items = self.scene.flatten();
            let non_empty = items
                .iter()
                .filter(|it| !it.mesh.positions.is_empty() && !it.mesh.indices.is_empty())
                .count();

            eprintln!(
                "[pythagoras_step1 debug] camera center={:?} zoom={} aspect={} items={} non_empty_items={}",
                self.scene.camera.center_pt,
                self.scene.camera.zoom,
                self.scene.camera.viewport_aspect,
                items.len(),
                non_empty
            );

            for (i, it) in items.iter().enumerate() {
                eprintln!(
                    "[pythagoras_step1 debug] item[{i}] z={} color=({:.3},{:.3},{:.3},{:.3}) mesh(v={}, i={}) world_from_local={:?}",
                    it.z,
                    it.fill.r,
                    it.fill.g,
                    it.fill.b,
                    it.fill.a,
                    it.mesh.positions.len(),
                    it.mesh.indices.len(),
                    it.world_from_local.m
                );
            }
        });

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
                label: Some("Pythagoras Step1 Encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Pythagoras Step1 Pass"),
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

        // Keep animating.
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

/// Create a filled right triangle mesh.
/// Right angle at `origin`. Legs:
/// - along +X with length `a`
/// - along +Y with length `b`
fn right_triangle_mesh(origin: [f32; 2], a: f32, b: f32) -> Mesh2D {
    let ox = origin[0];
    let oy = origin[1];

    // Triangle vertices: O, A, B
    // O = (0,0), A = (a,0), B = (0,b)
    let positions = vec![[ox, oy], [ox + a, oy], [ox, oy + b]];
    let indices = vec![0, 1, 2];

    Mesh2D { positions, indices }
}

/// Approximate a line segment as a thin rectangle mesh (two triangles).
fn line_as_rect_mesh(a: [f32; 2], b: [f32; 2], thickness_pt: f32) -> Mesh2D {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let len = (dx * dx + dy * dy).sqrt();

    if len < 1e-6 {
        return Mesh2D::default();
    }

    let nx = -dy / len;
    let ny = dx / len;
    let half = 0.5 * thickness_pt.max(0.25);

    let p0 = [a[0] + nx * half, a[1] + ny * half];
    let p1 = [a[0] - nx * half, a[1] - ny * half];
    let p2 = [b[0] - nx * half, b[1] - ny * half];
    let p3 = [b[0] + nx * half, b[1] + ny * half];

    Mesh2D {
        positions: vec![p0, p1, p2, p3],
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}

/// Create a simple label block as a filled rectangle mesh, placed using local transform.

/// Thin wrapper main: uses the library app runner with the demo state.
fn main() -> anyhow::Result<()> {
    env_logger::init();

    locus::render::app::run_with_builder(
        locus::render::app::AppConfig {
            title: "locus: pythagoras_step1".to_string(),
            ..Default::default()
        },
        |window| async move { State::new(window).await },
    )
    .context("failed to run pythagoras_step1 demo")
}
