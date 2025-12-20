//! Glyph/path tessellation helpers.
//!
//! This module converts vector outlines (`lyon::path::Path`) into renderer-friendly
//! triangle meshes (`crate::scene::Mesh2D`) using `lyon::tessellation::FillTessellator`.
//!
//! Intended use (Phase A: single-glyph proof, later Typst frames):
//! 1. Resolve a font face and extract a glyph outline as a `Path` (font units).
//! 2. Apply a transform from font units -> pt and apply positioning (advance/offset).
//! 3. Tessellate into triangles.
//! 4. Add to an `Mobject2D` and render via the scene pipeline.
//!
//! Notes:
//! - This uses *fill* tessellation (closed contours). That's correct for typical
//!   font outlines.
//! - Winding / fill rule matters. Fonts are usually authored to work with
//!   non-zero winding, but some environments prefer even-odd. We expose both.
//! - For performance, you will likely want to reuse a `FillTessellator` and
//!   `VertexBuffers` across many glyphs; APIs here are designed to allow that.

use lyon::math::point;
use lyon::path::Path;
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, FillVertexConstructor,
    VertexBuffers,
};

use crate::scene::Mesh2D;

/// Tessellation options tailored for glyph outlines.
///
/// - `tolerance`: smaller => more triangles (smoother curves), larger => fewer triangles.
/// - `fill_rule`: NonZero is a common default for fonts; EvenOdd can be useful in some cases.
#[derive(Debug, Copy, Clone)]
pub struct TessellateOptions {
    pub tolerance: f32,
    pub fill_rule: FillRule,
}

impl Default for TessellateOptions {
    fn default() -> Self {
        Self {
            tolerance: 0.02,
            fill_rule: FillRule::NonZero,
        }
    }
}

/// Simple affine transform for 2D points (column-vector convention).
///
/// This is intentionally minimal to avoid pulling render/scene transform types
/// into the font module. You can convert from your scene transforms later.
///
/// Matrix:
/// [ a c tx ]
/// [ b d ty ]
/// [ 0 0  1 ]
#[derive(Debug, Copy, Clone)]
pub struct Affine2x3 {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub tx: f32,
    pub ty: f32,
}

impl Default for Affine2x3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Affine2x3 {
    pub const IDENTITY: Self = Self {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        tx: 0.0,
        ty: 0.0,
    };

    #[inline]
    pub fn scale_translate(scale: f32, tx: f32, ty: f32) -> Self {
        Self {
            a: scale,
            b: 0.0,
            c: 0.0,
            d: scale,
            tx,
            ty,
        }
    }

    #[inline]
    pub fn transform_point(&self, x: f32, y: f32) -> (f32, f32) {
        // Column-vector:
        // [x'] = [a c tx] [x]
        // [y']   [b d ty] [y]
        // [1 ]   [0 0  1] [1]
        let nx = self.a * x + self.c * y + self.tx;
        let ny = self.b * x + self.d * y + self.ty;
        (nx, ny)
    }
}

/// A vertex for tessellation output (2D position only).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TessVertex {
    pub position: [f32; 2],
}

/// Builds `TessVertex` from lyon's `FillVertex`.
struct TessVertexCtor {
    xf: Affine2x3,
}

impl FillVertexConstructor<TessVertex> for TessVertexCtor {
    fn new_vertex(&mut self, v: FillVertex) -> TessVertex {
        let p = v.position();
        let (x, y) = self.xf.transform_point(p.x, p.y);
        TessVertex { position: [x, y] }
    }
}

/// Tessellate a single outline path into a `scene::Mesh2D`.
///
/// - `path`: outline path, typically in font units.
/// - `transform`: applied to all points before output (e.g. font units -> pt + positioning).
/// - `opts`: tessellation tolerance and fill rule.
///
/// Returns a triangle mesh suitable for filling.
///
/// Errors:
/// - Returns `Err(String)` if tessellation fails (e.g. invalid path data).
pub fn tessellate_path_to_mesh(
    path: &Path,
    transform: Affine2x3,
    opts: TessellateOptions,
) -> Result<Mesh2D, String> {
    let mut tess = FillTessellator::new();
    let mut buffers: VertexBuffers<TessVertex, u16> = VertexBuffers::new();

    let fill = FillOptions::tolerance(opts.tolerance).with_fill_rule(opts.fill_rule);

    // Fonts can have self-intersections; turning on "assume_no_intersections" is risky.
    // Keep defaults for correctness.
    let ctor = TessVertexCtor { xf: transform };
    let res = tess.tessellate_path(path, &fill, &mut BuffersBuilder::new(&mut buffers, ctor));

    res.map_err(|e| format!("lyon tessellation failed: {e:?}"))?;

    Ok(mesh_from_buffers(&buffers))
}

/// Convert lyon `VertexBuffers<TessVertex, u16>` into `scene::Mesh2D`.
#[inline]
pub fn mesh_from_buffers(buffers: &VertexBuffers<TessVertex, u16>) -> Mesh2D {
    Mesh2D {
        positions: buffers.vertices.iter().map(|v| v.position).collect(),
        indices: buffers.indices.clone(),
    }
}

/// Append a tessellated path into an existing scene mesh, offsetting indices.
///
/// This is a convenience for batching many glyphs into one mesh:
/// - Create a big `Mesh2D`
/// - For each glyph outline path:
///   - tessellate into a temporary `VertexBuffers`
///   - append into the big mesh with index offset.
///
/// Panics if the vertex count exceeds `u16::MAX`.
pub fn append_tessellated_path(
    out: &mut Mesh2D,
    path: &Path,
    transform: Affine2x3,
    opts: TessellateOptions,
) -> Result<(), String> {
    let mesh = tessellate_path_to_mesh(path, transform, opts)?;
    append_mesh(out, &mesh);
    Ok(())
}

/// Append `src` into `dst` with index offset (u16 indices).
pub fn append_mesh(dst: &mut Mesh2D, src: &Mesh2D) {
    let base = dst.positions.len();
    assert!(
        base + src.positions.len() <= u16::MAX as usize,
        "append_mesh: vertex count overflow for u16 indices"
    );

    dst.positions.extend_from_slice(&src.positions);

    let base_u16 = base as u16;
    dst.indices
        .extend(src.indices.iter().copied().map(|i| base_u16 + i));
}

/// A helper to build a debug triangle mesh directly (useful when bringing up the pipeline).
///
/// This is not used by the glyph pipeline, but it's a handy smoke test that uses the same
/// output mesh type as tessellation.
pub fn debug_triangle_pt() -> Mesh2D {
    Mesh2D {
        positions: vec![[-50.0, -30.0], [50.0, -30.0], [0.0, 60.0]],
        indices: vec![0, 1, 2],
    }
}

/// Create a tiny rectangle path in local units (for testing tessellation quickly).
///
/// This uses lyon `Path` builder directly. It is a utility for tests / bring-up.
pub fn rect_path(w: f32, h: f32) -> Path {
    let hw = w * 0.5;
    let hh = h * 0.5;

    let mut b = Path::builder();
    b.begin(point(-hw, -hh));
    b.line_to(point(hw, -hh));
    b.line_to(point(hw, hh));
    b.line_to(point(-hw, hh));
    b.close();
    b.build()
}
