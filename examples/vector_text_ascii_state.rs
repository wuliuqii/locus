//! Reusable state-only implementation for the vector text ASCII demo.
//!
//! This file is intentionally **not** an example binary (no `fn main()`).
//! It is meant to be reused by:
//! - `examples/vector_text_ascii.rs` (thin wrapper that calls the library runner)
//! - `src/main.rs` (default `cargo run` entrypoint)
//!
//! The state renders:
//! - a short ASCII string as vector glyph outlines (tessellated)
//! - baseline / ascender / descender guide lines
//! - a gentle camera zoom animation (breathing) to validate smooth scaling

use std::{sync::Arc, time::Instant};

use anyhow::Context as _;
use winit::window::Window;

use locus::font::text::{TextLayoutOptions, baseline_guides_meshes, layout_ascii_text_to_mesh};
use locus::font::{FontQuery, FontSystem};
use locus::render::{app::AppState, gpu::Gpu, mesh_renderer::MeshRenderer};
use locus::scene::{Mobject2D, Rgba, Scene2D};

/// Vector text demo state.
///
/// This is designed to be driven by the library runner:
/// - `locus::render::app::run_with_builder(..., |window| async move { State::new(window).await })`
pub struct State {
    pub window: Arc<Window>,
    pub gpu: Gpu,

    pub scene: Scene2D,
    pub renderer: MeshRenderer,

    // Animation clock for camera zoom proof.
    start_time: Instant,
    // Baseline zoom computed from framing; animation should be relative to this
    // to avoid drift/accumulation.
    base_zoom: f32,
}

impl State {
    /// Construct the demo state.
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let gpu = Gpu::new(window.clone()).await?;

        let mut scene = Scene2D::new();
        scene
            .camera
            .set_viewport_px(gpu.size.width.max(1), gpu.size.height.max(1));

        // --- Vector text bring-up: render a short ASCII string as one merged mesh ---
        let font_system = FontSystem::new().context("font: failed to initialize FontSystem")?;

        // Use a kerning-sensitive string to validate pair kerning (e.g. "AV", "To").
        // Keep it ASCII-only for this bring-up stage.
        let text = "AV To s = 1";

        // Keep the font query explicit so we can also read its metrics for guides.
        let font_query = FontQuery {
            families: vec![
                "STIX Two Text".to_string(),
                "Latin Modern Roman".to_string(),
                "Linux Libertine".to_string(),
                "serif".to_string(),
            ],
            weight: 400,
            italic: false,
        };

        // Resolve the face once so we can access its vertical metrics for guides.
        let face = font_system
            .resolve(&font_query)
            .context("font: failed to resolve face for guide metrics")?;

        let font_size_pt = 140.0f32;
        let scale = face.font_units_to_pt_scale(font_size_pt);

        let text_mesh = layout_ascii_text_to_mesh(
            &font_system,
            text,
            &TextLayoutOptions {
                font: font_query,
                font_size_pt,
                tolerance: 0.01,
                letter_spacing_pt: 0.0,
                kerning_strength: 1.0,
                ..Default::default()
            },
        )
        .context("text: failed to build mesh")?;

        // Center the text by shifting it left by half the advance width.
        // Baseline remains at y=0.
        let mut mesh = text_mesh.mesh;
        let shift_x = -0.5 * text_mesh.advance_width_pt;
        for p in &mut mesh.positions {
            p[0] += shift_x;
        }

        let text_obj = Mobject2D::new("text_ascii")
            .with_mesh(mesh)
            .with_fill(Rgba {
                r: 0.95,
                g: 0.95,
                b: 0.95,
                a: 1.0,
            });
        scene.add_root(text_obj);

        // --- Baseline + ascender/descender guides (debug) ---
        // Convert font-unit metrics to pt using the same scale as the text.
        let asc_pt = face.v_metrics.ascender * scale;
        let desc_pt = face.v_metrics.descender * scale;

        let guide_w = text_mesh.advance_width_pt.max(400.0);
        let (baseline_mesh, asc_mesh, desc_mesh) =
            baseline_guides_meshes(guide_w, 2.0, asc_pt, desc_pt);

        scene.add_root(
            Mobject2D::new("baseline")
                .with_mesh(baseline_mesh)
                .with_fill(Rgba {
                    r: 0.2,
                    g: 0.9,
                    b: 0.3,
                    a: 0.85,
                }),
        );

        scene.add_root(
            Mobject2D::new("ascender")
                .with_mesh(asc_mesh)
                .with_fill(Rgba {
                    r: 0.9,
                    g: 0.7,
                    b: 0.2,
                    a: 0.65,
                }),
        );

        scene.add_root(
            Mobject2D::new("descender")
                .with_mesh(desc_mesh)
                .with_fill(Rgba {
                    r: 0.9,
                    g: 0.2,
                    b: 0.4,
                    a: 0.65,
                }),
        );

        // Frame the scene with padding (fits text and guides).
        if let Some(root) = scene.get("text_ascii") {
            let bounds = root.compute_local_bounds();
            scene.camera.frame_bounds(bounds, 60.0, 0.85);
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

    /// Forward resize events to the GPU and camera.
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.gpu.resize(new_size);
        self.scene
            .camera
            .set_viewport_px(self.gpu.size.width.max(1), self.gpu.size.height.max(1));
    }

    /// Render one frame and schedule the next redraw (continuous animation).
    pub fn render(&mut self) -> anyhow::Result<()> {
        // Gentle camera zoom animation (breathing) to validate smooth vector scaling.
        // IMPORTANT: Use a stable baseline (the framed zoom) so we don't accumulate drift
        // by feeding the animated zoom back into itself each frame.
        let t = self.start_time.elapsed().as_secs_f32();
        let zoom = self.base_zoom * (1.0 + 0.06 * (t * 0.8).sin());
        self.scene.camera.zoom = zoom;

        // Acquire frame (handle recoverable surface errors).
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

        // Encode pass
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Main Encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
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

        // Submit and present
        self.gpu.queue.submit(Some(encoder.finish()));
        self.window.pre_present_notify();
        surface_texture.present();

        // Drive animation without a tight loop by requesting the next frame here.
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
