//! Example: Pythagoras teaching demo (Step 1) using the minimal `Timeline` system.
//!
//! Goal of this step:
//! - Show a right triangle on the left.
//! - Fade in labels "a", "b", "c".
//! - Move the labels slightly into place to demonstrate animation basics.
//!
//! Run (recommended to avoid blocking your terminal session):
//! - `RUST_LOG=info timeout 10s cargo run --example pythagoras_step1`
//!
//! Notes:
//! - This demo is intentionally simple. It focuses on wiring the `anim::Timeline` into the
//!   winit render loop and animating `Mobject2D` properties.
//! - Geometry here is basic CPU meshes (triangles/quads) built directly.
//! - Later steps will replace labels with Typst-rendered math and add more complex constructions.

use std::{sync::Arc, time::Instant};

use anyhow::Context as _;
use winit::window::Window;

use locus::{
    anim::{AnimTarget, Ease, Keyframe, Timeline, Track},
    render::{app::AppState, gpu::Gpu, mesh_renderer::MeshRenderer},
    scene::{Affine2, Mesh2D, Mobject2D, Rgba, Scene2D},
};

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

        // --- Build geometry (triangle + simple label blocks) ---
        //
        // Coordinate convention:
        // - world units are pt
        // - y-up in our scene
        // - we place the construction around the origin and then frame the camera
        let tri = right_triangle_mesh([0.0, 0.0], 240.0, 160.0);

        // Triangle object.
        scene.add_root(
            Mobject2D::new("tri")
                .with_mesh(tri)
                .with_fill(Rgba {
                    r: 0.20,
                    g: 0.25,
                    b: 0.30,
                    a: 1.0,
                }),
        );

        // Labels as simple rectangles for now (we'll replace with Typst text in step 2).
        // Start them slightly offset and fully transparent; animation will move + fade them in.
        scene.add_root(label_block(
            "label_a",
            [-40.0, 70.0],
            [22.0, 14.0],
            Rgba {
                r: 0.90,
                g: 0.55,
                b: 0.25,
                a: 0.0,
            },
        ));

        scene.add_root(label_block(
            "label_b",
            [110.0, -20.0],
            [22.0, 14.0],
            Rgba {
                r: 0.35,
                g: 0.85,
                b: 0.45,
                a: 0.0,
            },
        ));

        scene.add_root(label_block(
            "label_c",
            [70.0, 130.0],
            [22.0, 14.0],
            Rgba {
                r: 0.45,
                g: 0.65,
                b: 0.95,
                a: 0.0,
            },
        ));

        // Frame camera around triangle + expected label area.
        // We approximate by framing around the triangle's bounds.
        if let Some(root) = scene.get("tri") {
            let bounds = root.compute_local_bounds();
            scene.camera.frame_bounds(bounds, 80.0, 0.85);
        }

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

        let renderer = MeshRenderer::new(&gpu)?;

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

        // Keep camera stable for teaching (no breathing here), but allow future steps to animate camera.
        self.scene.camera.zoom = self.base_zoom;
        self.scene.camera.center_pt = self.base_center;

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

/// Create a simple label block as a filled rectangle mesh, placed using local transform.
fn label_block(name: &str, pos: [f32; 2], size: [f32; 2], color: Rgba) -> Mobject2D {
    let mesh = rect_mesh_centered(size[0], size[1]);

    Mobject2D::new(name)
        .with_mesh(mesh)
        .with_fill(color)
        .with_transform(Affine2::translate(pos[0], pos[1]))
}

/// Create a centered rectangle mesh in local coordinates.
fn rect_mesh_centered(w: f32, h: f32) -> Mesh2D {
    let hw = 0.5 * w;
    let hh = 0.5 * h;

    let positions = vec![
        [-hw, -hh],
        [hw, -hh],
        [hw, hh],
        [-hw, hh],
    ];

    let indices = vec![0, 1, 2, 0, 2, 3];
    Mesh2D { positions, indices }
}

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
