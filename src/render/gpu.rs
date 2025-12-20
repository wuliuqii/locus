use std::sync::Arc;

use anyhow::Context as _;
use winit::window::Window;

/// Minimal GPU context wrapper:
/// - Owns `wgpu::Instance`, `wgpu::Adapter`, `wgpu::Device`, `wgpu::Queue`
/// - Owns the window `Surface` and the current `SurfaceConfiguration`
///
/// This is intended as a shared foundation for:
/// - simple primitive rendering (triangles/rects)
/// - Typst frame rendering (vector glyph outlines, shapes, etc.)
pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    /// The surface is tied to the window.
    pub surface: wgpu::Surface<'static>,
    pub surface_format: wgpu::TextureFormat,

    pub size: winit::dpi::PhysicalSize<u32>,
    pub config: wgpu::SurfaceConfiguration,
}

impl Gpu {
    /// Create a GPU context for the given window.
    ///
    /// Notes:
    /// - Chooses the first surface format from surface capabilities.
    /// - Configures the surface immediately.
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .context("wgpu: failed to request adapter")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .context("wgpu: failed to request device")?;

        let size = window.inner_size();

        // Create a 'static surface by cloning the Arc<Window>.
        // This is safe because the surface must not outlive the window; we keep the
        // window alive elsewhere in the app state.
        let surface = instance
            .create_surface(window)
            .context("wgpu: failed to create surface")?;

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .first()
            .copied()
            .context("wgpu: surface reported no supported formats")?;

        let config = Self::make_surface_config(size, surface_format);

        surface.configure(&device, &config);

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface,
            surface_format,
            size,
            config,
        })
    }

    /// Reconfigure the surface for a new size.
    ///
    /// You should call this on `WindowEvent::Resized`.
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        // Avoid configuring 0-sized surfaces; winit can report 0 during minimize.
        if new_size.width == 0 || new_size.height == 0 {
            self.size = new_size;
            self.config.width = 0;
            self.config.height = 0;
            return;
        }

        self.size = new_size;
        self.config = Self::make_surface_config(new_size, self.surface_format);
        self.surface.configure(&self.device, &self.config);
    }

    /// Acquire the next frame from the surface.
    ///
    /// Returns the surface texture and its view in the SRGB view format.
    ///
    /// Robust error handling:
    /// - Surface acquisition can fail transiently (e.g. during resize) with `wgpu::SurfaceError`.
    /// - We return `SurfaceError` explicitly so callers can decide whether to reconfigure,
    ///   retry, or exit.
    pub fn acquire_frame(
        &self,
    ) -> Result<(wgpu::SurfaceTexture, wgpu::TextureView), wgpu::SurfaceError> {
        let surface_texture = self.surface.get_current_texture()?;

        // Use SRGB view for correct color-space when the surface supports it.
        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.surface_format.add_srgb_suffix()),
                ..Default::default()
            });

        Ok((surface_texture, view))
    }

    fn make_surface_config(
        size: winit::dpi::PhysicalSize<u32>,
        surface_format: wgpu::TextureFormat,
    ) -> wgpu::SurfaceConfiguration {
        wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            // We render into SRGB view format for correct gamma.
            view_formats: vec![surface_format.add_srgb_suffix()],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: size.width,
            height: size.height,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::AutoNoVsync,
        }
    }
}
