//! A simple 2D mesh renderer.
//!
//! This renderer draws a list of scene draw items (`scene::DrawItem2D`) using a
//! solid-color pipeline and a camera MVP.
//!
//! Design goals:
//! - Keep it minimal and easy to iterate on.
//! - Work in **pt** world coordinates.
//! - Let the camera provide the world->clip transform.
//! - Allow per-item local transforms (world_from_local) and per-item fill color.
//!
//! Debug mode:
//! - You can enable a "full-screen triangle" draw path to validate that the render pass,
//!   pipeline, and surface presentation are working, independent of camera math and scene data.
//! - This is intentionally simple and should only be used for debugging.
//!
//! Notes / current limitations:
//! - Uses a uniform buffer that stores (mvp, color) and is updated per draw call.
//! - No batching beyond "one draw per item" (fine for early phase; we can batch later).
//! - Uses u16 indices (matches `scene::Mesh2D`).
//! - No depth buffer; painter's order is based on `z` sorting in the caller.

use std::{borrow::Cow, mem};

use crate::render::gpu::Gpu;
use crate::scene::{Affine2, Camera2D, DrawItem2D, Rgba};

fn round_up_to(v: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (v + (align - 1)) & !(align - 1)
}

/// GPU vertex format for 2D meshes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex2D {
    pub position: [f32; 2],
}

impl Vertex2D {
    pub const ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x2];

    #[inline]
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex2D>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

/// Uniform layout for the solid-color pipeline:
/// - `mvp`: clip_from_world * world_from_local (embedded affine into 4x4)
/// - `color`: RGBA
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SolidUniforms {
    mvp: [[f32; 4]; 4],
    color: [f32; 4],
}

impl SolidUniforms {
    #[inline]
    fn new(mvp: [[f32; 4]; 4], color: Rgba) -> Self {
        Self {
            mvp,
            color: [color.r, color.g, color.b, color.a],
        }
    }
}

// (removed) rgba_to_array: no longer needed

fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // Column-major 4x4: out = a * b
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }
    out
}

fn affine_to_mat4(a: Affine2) -> [[f32; 4]; 4] {
    // Affine2 is stored as a 3x3 column-major matrix. Embed it into a 4x4:
    //
    // [ a00 a01 0 a02 ]   but note our Affine2 uses columns:
    // [ a10 a11 0 a12 ]   Affine2::to_mat4 already exists, but we keep local here
    // [  0   0  1  0  ]
    // [ a20 a21 0 a22 ]
    //
    // However in our Affine2, the translation is in the 3rd column's first two rows
    // (m[2][0], m[2][1]) and bottom-right is m[2][2].
    a.to_mat4()
}

/// Convert a scene mesh into GPU vertex/index arrays.
///
/// Notes:
/// - Keeps indices as u16 (matching scene).
/// - Positions are copied directly; they are assumed to be in local(pt) coordinates.
fn build_gpu_mesh(mesh: &crate::scene::Mesh2D) -> (Vec<Vertex2D>, Vec<u16>) {
    let vertices = mesh
        .positions
        .iter()
        .copied()
        .map(|p| Vertex2D { position: p })
        .collect::<Vec<_>>();

    (vertices, mesh.indices.clone())
}

/// A minimal mesh renderer that draws solid-colored meshes.
pub struct MeshRenderer {
    pipeline: wgpu::RenderPipeline,

    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,

    // Reusable GPU buffers; resized on demand.
    vertex_buffer: wgpu::Buffer,
    vertex_capacity_bytes: u64,

    index_buffer: wgpu::Buffer,
    index_capacity_bytes: u64,

    /// If enabled, ignore scene items and draw a single full-screen triangle in clip-space.
    ///
    /// This is a debugging aid to validate that the render pipeline outputs pixels.
    debug_fullscreen_triangle: bool,
}

impl MeshRenderer {
    /// Enable or disable the full-screen triangle debug mode.
    ///
    /// When enabled, `draw_items()` will ignore the provided scene and render a single
    /// magenta triangle directly in clip space.
    pub fn set_debug_fullscreen_triangle(&mut self, enabled: bool) {
        self.debug_fullscreen_triangle = enabled;
    }

    /// Create a new renderer with a solid-color pipeline.
    ///
    /// This expects the surface format to be SRGB-view compatible; we target
    /// `gpu.surface_format.add_srgb_suffix()` for rendering.
    pub fn new(gpu: &Gpu) -> anyhow::Result<Self> {
        // Shader: keep it in a file so it can be iterated independently.
        // If you later want hot-reload, replace include_str! with runtime file loading.
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MeshRenderer Solid Shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "shaders/solid_mesh.wgsl"
                ))),
            });

        let uniform_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("MeshRenderer Uniform BGL"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                wgpu::BufferSize::new(mem::size_of::<SolidUniforms>() as u64)
                                    .unwrap(),
                            ),
                        },
                        count: None,
                    }],
                });

        let uniform_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MeshRenderer Uniform Buffer"),
            size: mem::size_of::<SolidUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MeshRenderer Uniform BG"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MeshRenderer Pipeline Layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("MeshRenderer Solid Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex2D::layout()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.surface_format.add_srgb_suffix(),
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        // Create small initial buffers; they'll grow as needed.
        let initial_vb = 1024u64;
        let initial_ib = 1024u64;

        let vertex_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MeshRenderer Vertex Buffer"),
            size: initial_vb,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MeshRenderer Index Buffer"),
            size: initial_ib,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            pipeline,
            uniform_buffer,
            uniform_bind_group,
            vertex_buffer,
            vertex_capacity_bytes: initial_vb,
            index_buffer,
            index_capacity_bytes: initial_ib,
            debug_fullscreen_triangle: false,
        })
    }

    /// Ensure the internal vertex/index buffers can hold at least `vb_bytes` / `ib_bytes`.
    fn ensure_capacity(&mut self, gpu: &Gpu, vb_bytes: u64, ib_bytes: u64) {
        if vb_bytes > self.vertex_capacity_bytes {
            // Grow to next power-ish to reduce realloc frequency.
            let new_size = vb_bytes.next_power_of_two().max(1024);
            self.vertex_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MeshRenderer Vertex Buffer (resized)"),
                size: new_size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.vertex_capacity_bytes = new_size;
        }

        if ib_bytes > self.index_capacity_bytes {
            let new_size = ib_bytes.next_power_of_two().max(1024);
            self.index_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MeshRenderer Index Buffer (resized)"),
                size: new_size,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.index_capacity_bytes = new_size;
        }
    }

    /// Draw all items into the provided render pass.
    ///
    /// The caller is responsible for:
    /// - creating the render pass
    /// - clearing background
    /// - sorting items by z if desired
    ///
    /// Coordinate mapping:
    /// - Item vertices are in *local pt* coordinates.
    /// - Each item has a `world_from_local` affine.
    /// - Camera provides `clip_from_world`.
    ///
    /// Final transform: `clip_from_local = clip_from_world * world_from_local`.
    pub fn draw_items<'pass>(
        &'pass mut self,
        gpu: &Gpu,
        pass: &mut wgpu::RenderPass<'pass>,
        camera: &Camera2D,
        items: &[DrawItem2D],
    ) -> anyhow::Result<()> {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.uniform_bind_group, &[]);

        // Debug: render a full-screen triangle in clip space (no camera dependency).
        //
        // Use the conservative "big triangle" that covers the full viewport in NDC.
        // Keeping vertices well within clip space avoids any surprises from precision
        // or coordinate convention issues.
        if self.debug_fullscreen_triangle {
            let vertices = vec![
                Vertex2D {
                    position: [-1.0, -1.0],
                },
                Vertex2D {
                    position: [1.0, -1.0],
                },
                Vertex2D {
                    position: [-1.0, 1.0],
                },
                Vertex2D {
                    position: [1.0, 1.0],
                },
            ];
            let indices: Vec<u16> = vec![0, 1, 2, 2, 1, 3];

            let vb_bytes = (vertices.len() * mem::size_of::<Vertex2D>()) as u64;
            let ib_bytes = (indices.len() * mem::size_of::<u16>()) as u64;

            let align = wgpu::COPY_BUFFER_ALIGNMENT;
            let vb_upload = round_up_to(vb_bytes, align);
            let ib_upload = round_up_to(ib_bytes, align);

            self.ensure_capacity(gpu, vb_upload, ib_upload);

            // Upload (padded) vertex data.
            let v_raw = bytemuck::cast_slice(&vertices);
            if vb_upload == vb_bytes {
                gpu.queue.write_buffer(&self.vertex_buffer, 0, v_raw);
            } else {
                let mut padded = Vec::<u8>::with_capacity(vb_upload as usize);
                padded.extend_from_slice(v_raw);
                padded.resize(vb_upload as usize, 0);
                gpu.queue.write_buffer(&self.vertex_buffer, 0, &padded);
            }

            // Upload (padded) index data.
            let i_raw = bytemuck::cast_slice(&indices);
            if ib_upload == ib_bytes {
                gpu.queue.write_buffer(&self.index_buffer, 0, i_raw);
            } else {
                let mut padded = Vec::<u8>::with_capacity(ib_upload as usize);
                padded.extend_from_slice(i_raw);
                padded.resize(ib_upload as usize, 0);
                gpu.queue.write_buffer(&self.index_buffer, 0, &padded);
            }

            // Identity MVP: positions are already in clip space.
            let mvp = Affine2::IDENTITY.to_mat4();
            let uniforms = SolidUniforms::new(
                mvp,
                Rgba {
                    r: 1.0,
                    g: 0.0,
                    b: 1.0,
                    a: 1.0,
                },
            );
            gpu.queue
                .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..vb_bytes));
            pass.set_index_buffer(
                self.index_buffer.slice(..ib_bytes),
                wgpu::IndexFormat::Uint16,
            );
            pass.draw_indexed(0..(indices.len() as u32), 0, 0..1);

            return Ok(());
        }

        // Precompute camera matrix (affine -> mat4).
        let clip_from_world = affine_to_mat4(camera.clip_from_world());

        for item in items {
            if item.mesh.positions.is_empty() || item.mesh.indices.is_empty() {
                continue;
            }

            // Convert CPU mesh into GPU-friendly arrays.
            let (vertices, indices) = build_gpu_mesh(&item.mesh);

            if vertices.is_empty() || indices.is_empty() {
                continue;
            }

            // Upload geometry.
            //
            // wgpu requires `Queue::write_buffer` writes to respect COPY_BUFFER_ALIGNMENT (4 bytes).
            // Our vertex/index data lengths can be arbitrary, so we pad uploads to the next multiple
            // of the alignment and slice only the real ranges when drawing.
            let vb_bytes = (vertices.len() * mem::size_of::<Vertex2D>()) as u64;
            let ib_bytes = (indices.len() * mem::size_of::<u16>()) as u64;

            let align = wgpu::COPY_BUFFER_ALIGNMENT;
            let vb_upload = round_up_to(vb_bytes, align);
            let ib_upload = round_up_to(ib_bytes, align);

            self.ensure_capacity(gpu, vb_upload, ib_upload);

            // Pad vertex upload if needed.
            let v_raw = bytemuck::cast_slice(&vertices);
            if vb_upload == vb_bytes {
                gpu.queue.write_buffer(&self.vertex_buffer, 0, v_raw);
            } else {
                let mut padded = Vec::<u8>::with_capacity(vb_upload as usize);
                padded.extend_from_slice(v_raw);
                padded.resize(vb_upload as usize, 0);
                gpu.queue.write_buffer(&self.vertex_buffer, 0, &padded);
            }

            // Pad index upload if needed.
            let i_raw = bytemuck::cast_slice(&indices);
            if ib_upload == ib_bytes {
                gpu.queue.write_buffer(&self.index_buffer, 0, i_raw);
            } else {
                let mut padded = Vec::<u8>::with_capacity(ib_upload as usize);
                padded.extend_from_slice(i_raw);
                padded.resize(ib_upload as usize, 0);
                gpu.queue.write_buffer(&self.index_buffer, 0, &padded);
            }

            // Compute MVP.
            let world_from_local = affine_to_mat4(item.world_from_local);
            let mvp = mat4_mul(clip_from_world, world_from_local);

            // Update uniforms (per item).
            let uniforms = SolidUniforms::new(mvp, item.fill);
            gpu.queue
                .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

            // Issue draw (slice only the real data, not the padded upload).
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..vb_bytes));
            pass.set_index_buffer(
                self.index_buffer.slice(..ib_bytes),
                wgpu::IndexFormat::Uint16,
            );
            pass.draw_indexed(0..(indices.len() as u32), 0, 0..1);
        }

        Ok(())
    }
}

/// --- Shader file expectation ---
///
/// This renderer expects `locus/src/render/shaders/solid_mesh.wgsl` to exist with:
/// - `@group(0) @binding(0)` uniform containing:
///   - `mvp: mat4x4<f32>`
///   - `color: vec4<f32>`
/// - a vertex shader that transforms `@location(0) position: vec2<f32>`
/// - a fragment shader that outputs `color`
///
/// Example WGSL:
/// ```wgsl
/// struct Uniforms {
///   mvp: mat4x4<f32>,
///   color: vec4<f32>,
/// };
///
/// @group(0) @binding(0)
/// var<uniform> u: Uniforms;
///
/// struct VsOut { @builtin(position) position: vec4<f32> };
///
/// @vertex
/// fn vs_main(@location(0) a_pos: vec2<f32>) -> VsOut {
///   var out: VsOut;
///   out.position = u.mvp * vec4<f32>(a_pos, 0.0, 1.0);
///   return out;
/// }
///
/// @fragment
/// fn fs_main() -> @location(0) vec4<f32> {
///   return u.color;
/// }
/// ```
const _SHADER_EXPECTATION_DOC: () = ();
