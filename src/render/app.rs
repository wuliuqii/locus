//! App entrypoint for the rendering layer.
//!
//! This module owns:
//! - the winit application lifecycle + event loop
//! - creating the window
//! - delegating to an injected async state builder
//!
//! Library API goal:
//! - Provide a stable `run_with_config(...)` entrypoint.
//! - Provide a configurable `run_with_builder(...)` entrypoint so demos can live under
//!   `examples/` without duplicating the event loop boilerplate.
//!
//! Design:
//! - The app runner is generic over a user-defined state type `S`.
//! - `S` must implement `AppState` (resize + render, and a way to request redraw).
//! - The builder is async and receives the created window.
//!
//! This keeps the library surface clean and allows example binaries to supply their own state.

use std::{future::Future, pin::Pin, sync::Arc};

use anyhow::Context as _;
use log::info;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

/// App-facing configuration for running the winit event loop.
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// Window title.
    pub title: String,
    /// ControlFlow for the event loop. Default is `Poll` (current behavior).
    pub control_flow: ControlFlow,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            title: "locus".to_string(),
            control_flow: ControlFlow::Poll,
        }
    }
}

/// Minimal trait a demo state must implement to be driven by the app runner.
pub trait AppState: 'static {
    /// Handle window resize.
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>);

    /// Render one frame.
    fn render(&mut self) -> anyhow::Result<()>;

    /// Request a redraw on the underlying window (used for continuous animation).
    fn request_redraw(&self);
}

/// Run the winit event loop with an explicit configuration.
///
/// This uses the library's default state (an "empty scene" state). For demos,
/// prefer `run_with_builder(...)` from an example binary.
pub fn run_with_config(config: AppConfig) -> anyhow::Result<()> {
    run_with_builder::<DefaultState, _, _>(config, |window| async move {
        DefaultState::new(window).await
    })
}

/// Run the winit event loop using defaults.
pub fn run() -> anyhow::Result<()> {
    run_with_config(AppConfig::default())
}

/// Run the winit event loop with an injected async state builder.
///
/// This is the recommended entrypoint for `examples/` so they can customize scene creation
/// without duplicating the winit event loop boilerplate.
///
/// Notes:
/// - The builder is called once when the app is resumed (after the window is created).
/// - The builder runs on the current thread using `pollster::block_on`.
pub fn run_with_builder<S, B, Fut>(config: AppConfig, builder: B) -> anyhow::Result<()>
where
    S: AppState,
    B: FnOnce(Arc<Window>) -> Fut + 'static,
    Fut: Future<Output = anyhow::Result<S>> + 'static,
{
    let event_loop = EventLoop::new().context("winit: failed to create EventLoop")?;
    event_loop.set_control_flow(config.control_flow);

    let mut app = App::<S>::new_with_builder(config, builder);
    event_loop
        .run_app(&mut app)
        .context("winit: run_app failed")?;

    Ok(())
}

/// Type-erased async builder for creating a state `S` from a created window.
///
/// Important:
/// - We must return a **pinned** boxed future so `pollster::block_on(...)` can drive it.
///   (`dyn Future` is not `Unpin` by default.)
type BoxedStateBuilder<S> = Box<
    dyn FnOnce(Arc<Window>) -> Pin<Box<dyn Future<Output = anyhow::Result<S>> + 'static>> + 'static,
>;

/// Application state used by winit.
struct App<S: AppState> {
    config: AppConfig,
    builder: Option<BoxedStateBuilder<S>>,
    state: Option<S>,
    exiting: bool,
}

impl<S: AppState> App<S> {
    fn new_with_builder<B, Fut>(config: AppConfig, builder: B) -> Self
    where
        B: FnOnce(Arc<Window>) -> Fut + 'static,
        Fut: Future<Output = anyhow::Result<S>> + 'static,
    {
        Self {
            config,
            builder: Some(Box::new(|window| Box::pin(builder(window)))),
            state: None,
            exiting: false,
        }
    }
}

impl<S: AppState> ApplicationHandler for App<S> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create window
        let window = Arc::new(
            event_loop
                .create_window(WindowAttributes::default().with_title(self.config.title.as_str()))
                .expect("winit: failed to create window"),
        );

        // Create state (via injected builder)
        let builder = self
            .builder
            .take()
            .expect("app state builder can only be consumed once");
        let state = pollster::block_on(builder(window)).expect("failed to initialize renderer");
        self.state = Some(state);

        // Kick off rendering
        self.state.as_ref().unwrap().request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                info!("Close requested; exiting");
                self.exiting = true;
                self.state = None;
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if self.exiting {
                    return;
                }
                state.resize(size);
                state.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                if self.exiting {
                    return;
                }
                if let Err(err) = state.render() {
                    info!("render error: {:#}", err);
                }
            }
            _ => {}
        }
    }
}

/// Default library state: empty scene, just clears the window.
///
/// This is intentionally minimal; real demos should provide their own `AppState` via
/// `run_with_builder(...)`.
struct DefaultState {
    window: Arc<Window>,
    gpu: crate::render::gpu::Gpu,
}

impl DefaultState {
    async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let gpu = crate::render::gpu::Gpu::new(window.clone()).await?;
        Ok(Self { window, gpu })
    }
}

impl AppState for DefaultState {
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.gpu.resize(new_size);
    }

    fn render(&mut self) -> anyhow::Result<()> {
        // Acquire frame (handle recoverable surface errors).
        let (surface_texture, view) = match self.gpu.acquire_frame() {
            Ok(v) => v,
            Err(wgpu::SurfaceError::Outdated) | Err(wgpu::SurfaceError::Lost) => {
                self.gpu.resize(self.gpu.size);
                self.request_redraw();
                return Ok(());
            }
            Err(wgpu::SurfaceError::Timeout) => {
                self.request_redraw();
                return Ok(());
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                return Err(anyhow::anyhow!("wgpu SurfaceError::OutOfMemory"));
            }
            Err(wgpu::SurfaceError::Other) => {
                self.gpu.resize(self.gpu.size);
                self.request_redraw();
                return Ok(());
            }
        };

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Main Encoder"),
            });

        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
        }

        self.gpu.queue.submit(Some(encoder.finish()));
        self.window.pre_present_notify();
        surface_texture.present();

        Ok(())
    }

    fn request_redraw(&self) {
        self.window.request_redraw();
    }
}
