//! Typst ζ demo state (windowed): compile the canonical validation formula and render it.
//!
//! This is a **state-only** module (no `fn main()`), used by `examples/typst_zeta.rs`.
//!
//! This file is intentionally thin: all Typst → mesh extraction logic lives in the library:
//! - `locus::typst::render`
//!
//! Runtime toggles (env vars):
//! - `LOCUS_TY_ZETA_SHOW_TEXT_DBG=0|1` (default: 0)
//! - `LOCUS_TY_ZETA_SHOW_GLYPHS=0|1`   (default: 1)
//! - `LOCUS_TY_ZETA_SHOW_LINES=0|1`    (default: 1)
//! - `LOCUS_TY_ZETA_SHOW_SHAPES=0|1`   (default: 1)

use std::{sync::Arc, time::Instant};

use anyhow::Context as _;
use winit::window::Window;

use locus::{
    render::{app::AppState, gpu::Gpu, mesh_renderer::MeshRenderer},
    scene::{Mobject2D, Rgba, Scene2D},
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

/// Demo state for "Typst ζ → extract meshes (library) → render".
pub struct State {
    pub window: Arc<Window>,
    pub gpu: Gpu,

    pub scene: Scene2D,
    pub renderer: MeshRenderer,

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

        // Compile the canonical formula (math-only, Phase A).
        let compiled = locus::typst::engine::compile_zeta_validation()
            .context("typst: failed to compile ζ validation formula")?;

        // Env toggles (see module docs).
        let show_lines = env_flag("LOCUS_TY_ZETA_SHOW_LINES", true);
        let show_shapes = env_flag("LOCUS_TY_ZETA_SHOW_SHAPES", true);
        let show_glyphs = env_flag("LOCUS_TY_ZETA_SHOW_GLYPHS", true);
        let show_text_dbg = env_flag("LOCUS_TY_ZETA_SHOW_TEXT_DBG", false);

        // Build meshes using the library extraction pipeline.
        let opts = locus::typst::render::RenderOptions {
            enable_lines: show_lines,
            enable_shapes: show_shapes,
            enable_glyphs: show_glyphs,
            ..Default::default()
        };

        let (meshes, stats) =
            locus::typst::render::build_meshes_from_paged_document(&compiled.document, &opts);

        log::info!(
            "typst_zeta(render): pages={} groups={} shapes_seen={} texts_seen={} lines={} filled_shapes={} glyph_calls={} glyph_tess_calls={} glyph_tris={}",
            stats.pages,
            stats.groups,
            stats.shapes_seen,
            stats.texts_seen,
            stats.lines_emitted,
            stats.filled_shapes_emitted,
            stats.glyph_calls,
            stats.glyph_tess_calls,
            stats.glyph_triangles
        );

        // Scene population.
        if show_lines {
            scene.add_root(
                Mobject2D::new("typst_lines")
                    .with_mesh(meshes.lines)
                    .with_fill(Rgba {
                        r: 0.90,
                        g: 0.90,
                        b: 0.95,
                        a: 1.0,
                    }),
            );
        }

        if show_shapes {
            scene.add_root(
                Mobject2D::new("typst_shapes")
                    .with_mesh(meshes.shapes)
                    .with_fill(Rgba {
                        r: 0.95,
                        g: 0.95,
                        b: 0.95,
                        a: 0.90,
                    }),
            );
        }

        if show_glyphs {
            scene.add_root(
                Mobject2D::new("typst_glyphs")
                    .with_mesh(meshes.glyphs)
                    .with_fill(Rgba {
                        r: 0.95,
                        g: 0.95,
                        b: 0.95,
                        a: 1.0,
                    }),
            );
        }

        if show_text_dbg {
            scene.add_root(
                Mobject2D::new("typst_text_dbg")
                    .with_mesh(meshes.text_debug)
                    .with_fill(Rgba {
                        r: 0.30,
                        g: 0.85,
                        b: 0.95,
                        a: 0.40,
                    }),
            );
        }

        // Camera framing preference:
        // glyphs → text debug → shapes → lines (whichever exists).
        if show_glyphs {
            if let Some(root) = scene.get("typst_glyphs") {
                let bounds = root.compute_local_bounds();
                if !bounds.is_empty() {
                    scene.camera.frame_bounds(bounds, 60.0, 0.85);
                }
            }
        }

        if show_text_dbg {
            if let Some(root) = scene.get("typst_text_dbg") {
                let bounds = root.compute_local_bounds();
                if !bounds.is_empty() {
                    scene.camera.frame_bounds(bounds, 60.0, 0.85);
                }
            }
        }

        if show_shapes {
            if let Some(root) = scene.get("typst_shapes") {
                let bounds = root.compute_local_bounds();
                if !bounds.is_empty() {
                    scene.camera.frame_bounds(bounds, 60.0, 0.85);
                }
            }
        }

        if show_lines {
            if let Some(root) = scene.get("typst_lines") {
                let bounds = root.compute_local_bounds();
                if !bounds.is_empty() {
                    scene.camera.frame_bounds(bounds, 60.0, 0.85);
                }
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
