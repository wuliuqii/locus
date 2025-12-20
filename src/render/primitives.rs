//! Minimal shared GPU primitives for 2D rendering.
//!
//! This module intentionally stays small and dependency-light so it can be reused by:
//! - simple demos (rects/triangles)
//! - future Typst rendering (vector glyph meshes, rule lines, etc.)
//!
//! Coordinate convention:
//! - Vertex positions are in "world" coordinates for a 2D scene.
//! - A `Uniforms2D` matrix maps world -> clip space.
//! - You can pick your own world units (pixels, points, NDC-ish); just be consistent.

use std::mem;

/// A 2D vertex with position only.
///
/// Add more attributes later (color, UV, etc.) as needed.
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

/// Standard 2D uniforms: a single MVP matrix.
///
/// - Stored column-major to match WGSL `mat4x4<f32>` expectation.
/// - Use `Uniforms2D::from_mat4` to populate.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms2D {
    /// Column-major mat4.
    pub mvp: [[f32; 4]; 4],
}

impl Uniforms2D {
    #[inline]
    pub fn from_mat4(mvp: glam::Mat4) -> Self {
        Self {
            mvp: mvp.to_cols_array_2d(),
        }
    }

    /// Handy helper: an ortho projection that maps world coords directly into clip space.
    ///
    /// If you already author geometry in clip space, you can use identity instead.
    #[inline]
    pub fn ortho_clip() -> Self {
        Self::from_mat4(glam::Mat4::orthographic_rh(
            -1.0, 1.0, // left, right
            -1.0, 1.0, // bottom, top
            -1.0, 1.0, // near, far
        ))
    }
}

/// A tiny mesh container for 2D drawing.
///
/// This is CPU-side only. Upload it via `wgpu::util::DeviceExt::create_buffer_init`.
#[derive(Clone, Debug, Default)]
pub struct Mesh2D {
    pub vertices: Vec<Vertex2D>,
    pub indices: Vec<u16>,
}

impl Mesh2D {
    /// Create an empty mesh with preallocated capacities.
    #[inline]
    pub fn with_capacity(v: usize, i: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(v),
            indices: Vec::with_capacity(i),
        }
    }

    /// Append a mesh into `self`, offsetting indices automatically.
    ///
    /// Panics if the resulting vertex count exceeds `u16::MAX`.
    #[inline]
    pub fn append(&mut self, other: &Mesh2D) {
        let base = self.vertices.len();
        assert!(
            base + other.vertices.len() <= u16::MAX as usize,
            "Mesh2D::append: vertex count overflow for u16 indices"
        );

        self.vertices.extend_from_slice(&other.vertices);

        let base = base as u16;
        self.indices
            .extend(other.indices.iter().copied().map(|idx| base + idx));
    }

    /// Convenience: build a CCW rectangle (two triangles) centered at `center`, with size `size`.
    ///
    /// Units are whatever you decide (e.g. NDC-ish, pixels, points).
    #[inline]
    pub fn rect(center: [f32; 2], size: [f32; 2]) -> Self {
        let cx = center[0];
        let cy = center[1];
        let hw = size[0] * 0.5;
        let hh = size[1] * 0.5;

        let vertices = vec![
            Vertex2D {
                position: [cx - hw, cy - hh],
            }, // 0
            Vertex2D {
                position: [cx + hw, cy - hh],
            }, // 1
            Vertex2D {
                position: [cx + hw, cy + hh],
            }, // 2
            Vertex2D {
                position: [cx - hw, cy + hh],
            }, // 3
        ];

        let indices = vec![0, 1, 2, 0, 2, 3];

        Self { vertices, indices }
    }
}
