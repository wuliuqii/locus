//! Scene graph abstractions.
//!
//! This module is inspired by (but not a copy of) manim's mental model:
//! - You build a scene out of objects ("mobjects").
//! - A camera maps world coordinates (we use **pt**) to clip space.
//! - Renderers consume a flattened list of draw items (meshes + style + z-order).
//!
//! Design goals for this project:
//! - Coordinate system uses **pt** end-to-end for layout outputs (Typst) and for
//!   scene-level composition. This makes typography and spacing predictable.
//! - Camera provides a clean separation between "layout space" and "screen space".
//! - Scene graph keeps transforms explicit and composable (matrix order matters).
//!
//! Notes:
//! - This file intentionally does not depend on Typst yet.
//! - This file intentionally does not depend on wgpu yet; it is renderer-agnostic.
//!
//! Next steps after introducing this module:
//! - Add `scene::flatten()` that emits renderer draw items.
//! - Add a renderer that draws `Mesh2D` in pt-space with camera MVP.
//! - Add a Typst -> Mobject adapter that produces glyph-outline meshes and rule meshes.

use std::collections::BTreeMap;

/// 2D affine transform stored as a 3x3 matrix in column-major order.
///
/// We use a 3x3 matrix because:
/// - it is sufficient for 2D translation/rotation/scale/shear
/// - it composes cleanly
/// - it avoids accidental use of perspective (for now)
///
/// Convention:
/// - Column vectors (x, y, 1)
/// - Composition is `world_from_local = parent * local`
///
/// This matches common graphics conventions and is compatible with mapping into a 4x4 MVP
/// later (by embedding the affine matrix into a 4x4).
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Affine2 {
    /// Column-major 3x3 matrix.
    pub m: [[f32; 3]; 3],
}

impl Default for Affine2 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Affine2 {
    pub const IDENTITY: Self = Self {
        m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    #[inline]
    pub fn translate(tx: f32, ty: f32) -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [tx, ty, 1.0]],
        }
    }

    #[inline]
    pub fn scale(sx: f32, sy: f32) -> Self {
        Self {
            m: [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    #[inline]
    pub fn rotate(rad: f32) -> Self {
        let (s, c) = rad.sin_cos();
        Self {
            m: [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Compose transforms: `self * rhs`.
    ///
    /// With column-vector convention, this means:
    /// - `p' = (self * rhs) * p`
    /// - rhs applies first, then self.
    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        let a = self.m;
        let b = rhs.m;

        // Column-major 3x3 multiply: out = a * b.
        let mut out = [[0.0f32; 3]; 3];
        for col in 0..3 {
            for row in 0..3 {
                out[col][row] =
                    a[0][row] * b[col][0] + a[1][row] * b[col][1] + a[2][row] * b[col][2];
            }
        }
        Self { m: out }
    }

    #[inline]
    pub fn transform_point(self, x: f32, y: f32) -> (f32, f32) {
        // Column vector [x, y, 1]
        let nx = self.m[0][0] * x + self.m[1][0] * y + self.m[2][0];
        let ny = self.m[0][1] * x + self.m[1][1] * y + self.m[2][1];
        (nx, ny)
    }

    /// Embed into a 4x4 (column-major) for GPU MVP use.
    ///
    /// Result transforms (x, y, 0, 1) with:
    /// - z unchanged
    /// - w = 1
    #[inline]
    pub fn to_mat4(self) -> [[f32; 4]; 4] {
        let m = self.m;
        [
            [m[0][0], m[0][1], 0.0, m[0][2]],
            [m[1][0], m[1][1], 0.0, m[1][2]],
            [0.0, 0.0, 1.0, 0.0],
            [m[2][0], m[2][1], 0.0, m[2][2]],
        ]
    }
}

/// Axis-aligned bounding box in pt-space.
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Aabb2 {
    pub min: [f32; 2],
    pub max: [f32; 2],
}

impl Aabb2 {
    #[inline]
    pub fn from_min_max(min: [f32; 2], max: [f32; 2]) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn empty() -> Self {
        Self {
            min: [f32::INFINITY, f32::INFINITY],
            max: [f32::NEG_INFINITY, f32::NEG_INFINITY],
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.min[0] > self.max[0] || self.min[1] > self.max[1]
    }

    #[inline]
    pub fn include_point(&mut self, p: [f32; 2]) {
        self.min[0] = self.min[0].min(p[0]);
        self.min[1] = self.min[1].min(p[1]);
        self.max[0] = self.max[0].max(p[0]);
        self.max[1] = self.max[1].max(p[1]);
    }

    #[inline]
    pub fn union(self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        Self {
            min: [self.min[0].min(other.min[0]), self.min[1].min(other.min[1])],
            max: [self.max[0].max(other.max[0]), self.max[1].max(other.max[1])],
        }
    }

    #[inline]
    pub fn center(&self) -> [f32; 2] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
        ]
    }

    #[inline]
    pub fn size(&self) -> [f32; 2] {
        [self.max[0] - self.min[0], self.max[1] - self.min[1]]
    }
}

/// Simple RGBA color (linear space assumed; your renderer may treat as sRGB).
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Rgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Rgba {
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
}

/// A renderer-agnostic mesh handle.
///
/// For now, we keep this as an owned CPU mesh. Later you can:
/// - keep CPU geometry for caching/boolean ops
/// - upload to GPU and store an opaque GPU handle
///
/// This version is intentionally minimal. We store 2D positions only.
#[derive(Debug, Clone, Default)]
pub struct Mesh2D {
    pub positions: Vec<[f32; 2]>,
    pub indices: Vec<u16>,
}

/// A draw item produced by flattening the scene graph.
///
/// - `world_from_local` must already be fully composed for this item.
/// - `z` is a simple painter's-order; higher draws later.
#[derive(Debug, Clone)]
pub struct DrawItem2D {
    pub mesh: Mesh2D,
    pub fill: Rgba,
    pub world_from_local: Affine2,
    pub z: i32,
}

/// A "mobject": a node in the scene graph.
///
/// Each node has:
/// - a local transform
/// - children
/// - optional geometry payload (meshes)
///
/// In a more sophisticated system, geometry would be an enum:
/// - fill mesh
/// - stroke path
/// - image
/// - etc.
///
/// For Phase A (math-only Typst), we primarily need:
/// - filled glyph outline meshes (or later: stroke/ink effects)
/// - simple rule meshes
#[derive(Debug, Clone)]
pub struct Mobject2D {
    pub name: String,
    pub local_from_parent: Affine2,
    pub z: i32,

    pub fill: Rgba,
    pub mesh: Option<Mesh2D>,

    pub children: Vec<Mobject2D>,

    /// Optional cached bounds in local space (pt).
    ///
    /// If present, this should bound `mesh` and all children in this node's local space.
    /// You can compute it lazily and store it here to support camera framing.
    pub local_bounds: Option<Aabb2>,
}

impl Default for Mobject2D {
    fn default() -> Self {
        Self {
            name: "mobject".to_string(),
            local_from_parent: Affine2::IDENTITY,
            z: 0,
            fill: Rgba::WHITE,
            mesh: None,
            children: Vec::new(),
            local_bounds: None,
        }
    }
}

impl Mobject2D {
    #[inline]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    #[inline]
    pub fn with_mesh(mut self, mesh: Mesh2D) -> Self {
        self.mesh = Some(mesh);
        self
    }

    #[inline]
    pub fn with_fill(mut self, fill: Rgba) -> Self {
        self.fill = fill;
        self
    }

    #[inline]
    pub fn with_transform(mut self, local_from_parent: Affine2) -> Self {
        self.local_from_parent = local_from_parent;
        self
    }

    #[inline]
    pub fn add_child(&mut self, child: Mobject2D) {
        self.children.push(child);
    }

    /// Compute a conservative local-space AABB from the mesh and children.
    ///
    /// This is currently a simple implementation:
    /// - Mesh bounds are computed from positions.
    /// - Child bounds are transformed into this node's local space using the child's transform.
    ///
    /// Note:
    /// - Transforming an AABB by an affine transform isn't tight unless you transform corners.
    /// - We do that here (transform 4 corners) for a conservative bound.
    pub fn compute_local_bounds(&self) -> Aabb2 {
        let mut bounds = Aabb2::empty();

        if let Some(mesh) = &self.mesh {
            for &p in &mesh.positions {
                bounds.include_point(p);
            }
        }

        for child in &self.children {
            let child_bounds = child
                .local_bounds
                .unwrap_or_else(|| child.compute_local_bounds());
            if child_bounds.is_empty() {
                continue;
            }

            let corners = [
                [child_bounds.min[0], child_bounds.min[1]],
                [child_bounds.max[0], child_bounds.min[1]],
                [child_bounds.max[0], child_bounds.max[1]],
                [child_bounds.min[0], child_bounds.max[1]],
            ];

            for c in corners {
                let (x, y) = child.local_from_parent.transform_point(c[0], c[1]);
                bounds.include_point([x, y]);
            }
        }

        bounds
    }

    /// Flatten this subtree into draw items, composing transforms.
    pub fn flatten(&self, parent_from_world: Affine2, out: &mut Vec<DrawItem2D>) {
        let world_from_local = parent_from_world.mul(self.local_from_parent);

        if let Some(mesh) = &self.mesh {
            out.push(DrawItem2D {
                mesh: mesh.clone(),
                fill: self.fill,
                world_from_local,
                z: self.z,
            });
        }

        for child in &self.children {
            child.flatten(world_from_local, out);
        }
    }
}

/// A simple 2D camera operating in pt-space.
///
/// The camera maps world pt coordinates into clip space (-1..1).
/// This makes it easy to implement:
/// - zoom/pan (animations)
/// - framing a formula bbox
///
/// Camera model:
/// - `center_pt`: the world point that maps to the center of the viewport
/// - `zoom`: scale factor (world pt -> NDC)
/// - `viewport_aspect`: width/height
///
/// The mapping is:
/// - Translate by -center
/// - Scale by zoom
/// - Apply aspect correction so that zoom is isotropic in world space
#[derive(Debug, Copy, Clone)]
pub struct Camera2D {
    pub center_pt: [f32; 2],
    pub zoom: f32,
    pub viewport_aspect: f32,
}

impl Default for Camera2D {
    fn default() -> Self {
        Self {
            center_pt: [0.0, 0.0],
            zoom: 1.0,
            viewport_aspect: 1.0,
        }
    }
}

impl Camera2D {
    /// Set the viewport size in pixels to update aspect ratio.
    #[inline]
    pub fn set_viewport_px(&mut self, width: u32, height: u32) {
        let w = width.max(1) as f32;
        let h = height.max(1) as f32;
        self.viewport_aspect = w / h;
    }

    /// Compute an affine transform from world(pt) to clip space.
    ///
    /// This returns a 3x3 affine transform suitable for embedding into a 4x4 MVP.
    pub fn clip_from_world(&self) -> Affine2 {
        // Move world so that center becomes origin.
        let t = Affine2::translate(-self.center_pt[0], -self.center_pt[1]);

        // Scale isotropically by zoom, then apply aspect correction:
        // If aspect > 1 (wider than tall), we scale X down so that shapes don't stretch.
        let ax = if self.viewport_aspect > 1.0 {
            1.0 / self.viewport_aspect
        } else {
            1.0
        };
        let ay = if self.viewport_aspect < 1.0 {
            self.viewport_aspect
        } else {
            1.0
        };

        let s = Affine2::scale(self.zoom * ax, self.zoom * ay);

        // Apply translate then scale: p_clip = s * t * p_world
        s.mul(t)
    }

    /// Frame the given world-space bounds into the viewport with padding.
    ///
    /// - `padding_pt`: extra margin around the bounds in world units (pt).
    /// - `fill_ratio`: fraction of viewport to occupy (e.g. 0.8).
    ///
    /// This updates:
    /// - `center_pt`
    /// - `zoom`
    pub fn frame_bounds(&mut self, bounds: Aabb2, padding_pt: f32, fill_ratio: f32) {
        if bounds.is_empty() {
            return;
        }

        let mut b = bounds;
        b.min[0] -= padding_pt;
        b.min[1] -= padding_pt;
        b.max[0] += padding_pt;
        b.max[1] += padding_pt;

        let size = b.size();
        let size_x = size[0].max(1e-3);
        let size_y = size[1].max(1e-3);

        self.center_pt = b.center();

        // Choose zoom so that the bbox fits in clip space (-1..1), i.e. size maps into 2.0 units.
        // Apply aspect correction similarly to `clip_from_world`.
        //
        // We aim for fill_ratio of the viewport.
        let fill = fill_ratio.clamp(0.05, 0.98);

        let ax = if self.viewport_aspect > 1.0 {
            1.0 / self.viewport_aspect
        } else {
            1.0
        };
        let ay = if self.viewport_aspect < 1.0 {
            self.viewport_aspect
        } else {
            1.0
        };

        // After applying s = zoom*(ax, ay), effective scale differs on axes.
        // We need zoom such that:
        //   size_x * zoom * ax <= 2*fill
        //   size_y * zoom * ay <= 2*fill
        let zoom_x = (2.0 * fill) / (size_x * ax);
        let zoom_y = (2.0 * fill) / (size_y * ay);

        self.zoom = zoom_x.min(zoom_y);
    }
}

/// A top-level scene that holds named mobjects.
///
/// This is intentionally minimal; you can evolve it into:
/// - a timeline of animations
/// - layers
/// - selection by name
/// - etc.
#[derive(Debug, Default)]
pub struct Scene2D {
    pub camera: Camera2D,
    pub roots: Vec<Mobject2D>,
    /// A simple name index for convenience.
    pub index: BTreeMap<String, usize>,
}

impl Scene2D {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_root(&mut self, m: Mobject2D) {
        let idx = self.roots.len();
        self.index.insert(m.name.clone(), idx);
        self.roots.push(m);
    }

    pub fn get(&self, name: &str) -> Option<&Mobject2D> {
        self.index.get(name).and_then(|&i| self.roots.get(i))
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut Mobject2D> {
        let i = *self.index.get(name)?;
        self.roots.get_mut(i)
    }

    /// Flatten the full scene into draw items.
    ///
    /// Caller typically sorts by `z` before rendering.
    pub fn flatten(&self) -> Vec<DrawItem2D> {
        let mut items = Vec::new();
        for root in &self.roots {
            root.flatten(Affine2::IDENTITY, &mut items);
        }
        items
    }
}
