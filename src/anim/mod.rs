//! Minimal animation timeline for `Scene2D`.
//!
//! Design goals (Phase 1):
//! - Keep it small, deterministic, and easy to extend.
//! - Animate a subset of properties that are essential for teaching demos:
//!   - `Mobject2D.local_from_parent` (translate/scale/rotate as full Affine2)
//!   - `Mobject2D.fill.a` (fade in/out)
//! - Provide keyframes with easing.
//! - Allow sequential and parallel composition.
//!
//! Non-goals (for now):
//! - Physics, constraints, spring animations
//! - Per-vertex deformation / morphing
//! - Text-level (glyph) animations
//!
//! Usage sketch:
//! ```ignore
//! use locus::anim::{Timeline, Ease, AnimTarget, Track, Keyframe};
//! let mut tl = Timeline::new();
//! tl.add_track(Track::new_alpha(AnimTarget::Name("label".into()))
//!     .with_keyframes(vec![
//!         Keyframe::at(0.0, 0.0).ease(Ease::OutCubic),
//!         Keyframe::at(0.6, 1.0).ease(Ease::OutCubic),
//!     ]));
//! ```
//!
//! Then in your render loop per frame:
//! ```ignore
//! let t = start.elapsed().as_secs_f32();
//! timeline.apply(&mut scene, t);
//! ```

use crate::scene::{Affine2, Scene2D};

/// How to map animation time into a normalized [0,1] parameter.
#[derive(Debug, Copy, Clone)]
pub enum Ease {
    Linear,
    InQuad,
    OutQuad,
    InOutQuad,
    InCubic,
    OutCubic,
    InOutCubic,
    InQuart,
    OutQuart,
    InOutQuart,
}

impl Ease {
    #[inline]
    pub fn sample(self, x: f32) -> f32 {
        let t = x.clamp(0.0, 1.0);
        match self {
            Ease::Linear => t,
            Ease::InQuad => t * t,
            Ease::OutQuad => 1.0 - (1.0 - t) * (1.0 - t),
            Ease::InOutQuad => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(2) * 0.5
                }
            }
            Ease::InCubic => t * t * t,
            Ease::OutCubic => 1.0 - (1.0 - t).powi(3),
            Ease::InOutCubic => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(3) * 0.5
                }
            }
            Ease::InQuart => t.powi(4),
            Ease::OutQuart => 1.0 - (1.0 - t).powi(4),
            Ease::InOutQuart => {
                if t < 0.5 {
                    8.0 * t.powi(4)
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(4) * 0.5
                }
            }
        }
    }
}

/// Identifies which object in the scene to animate.
#[derive(Debug, Clone)]
pub enum AnimTarget {
    /// Target a root object by name (recommended for now).
    Name(String),
}

impl AnimTarget {
    fn resolve_index<'a>(&self, scene: &'a Scene2D) -> Option<usize> {
        match self {
            AnimTarget::Name(name) => scene.index.get(name).copied(),
        }
    }
}

/// A keyframe in seconds with a scalar value.
#[derive(Debug, Copy, Clone)]
pub struct Keyframe {
    pub time_s: f32,
    pub value: f32,
    pub ease: Ease,
}

impl Keyframe {
    #[inline]
    pub fn at(time_s: f32, value: f32) -> Self {
        Self {
            time_s: time_s.max(0.0),
            value,
            ease: Ease::Linear,
        }
    }

    #[inline]
    pub fn ease(mut self, ease: Ease) -> Self {
        self.ease = ease;
        self
    }
}

/// Interpolate a scalar track across keyframes.
fn sample_keyframes(frames: &[Keyframe], t_s: f32) -> Option<f32> {
    if frames.is_empty() {
        return None;
    }
    if frames.len() == 1 {
        return Some(frames[0].value);
    }

    // Find segment [i, i+1] such that time_i <= t < time_{i+1}
    // We keep it simple O(n) for Phase 1.
    let mut prev = frames[0];
    if t_s <= prev.time_s {
        return Some(prev.value);
    }

    for next in &frames[1..] {
        if t_s < next.time_s {
            let dt = (next.time_s - prev.time_s).max(1e-6);
            let u = (t_s - prev.time_s) / dt;
            let k = prev.ease.sample(u);
            return Some(lerp(prev.value, next.value, k));
        }
        prev = *next;
    }

    // Past end: hold last value.
    Some(frames[frames.len() - 1].value)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// What property to animate for a target.
#[derive(Debug, Clone)]
pub enum Property {
    /// Animate alpha channel (`fill.a`) on the target root object.
    Alpha { keyframes: Vec<Keyframe> },

    /// Animate translation X (pt) on `local_from_parent`.
    TranslateX { keyframes: Vec<Keyframe> },

    /// Animate translation Y (pt) on `local_from_parent`.
    TranslateY { keyframes: Vec<Keyframe> },

    /// Animate rotation (radians) around local origin on `local_from_parent`.
    Rotate { keyframes: Vec<Keyframe> },

    /// Animate uniform scale on `local_from_parent`.
    Scale { keyframes: Vec<Keyframe> },
}

impl Property {
    fn start_end(&self) -> Option<(f32, f32)> {
        let frames = match self {
            Property::Alpha { keyframes }
            | Property::TranslateX { keyframes }
            | Property::TranslateY { keyframes }
            | Property::Rotate { keyframes }
            | Property::Scale { keyframes } => keyframes,
        };
        if frames.is_empty() {
            None
        } else {
            Some((
                frames
                    .iter()
                    .map(|k| k.time_s)
                    .fold(f32::INFINITY, f32::min),
                frames
                    .iter()
                    .map(|k| k.time_s)
                    .fold(f32::NEG_INFINITY, f32::max),
            ))
        }
    }

    fn sample(&self, t_s: f32) -> Option<f32> {
        let frames = match self {
            Property::Alpha { keyframes }
            | Property::TranslateX { keyframes }
            | Property::TranslateY { keyframes }
            | Property::Rotate { keyframes }
            | Property::Scale { keyframes } => keyframes,
        };
        sample_keyframes(frames, t_s)
    }
}

/// One animation track applies one property to one target.
#[derive(Debug, Clone)]
pub struct Track {
    pub target: AnimTarget,
    pub property: Property,
}

impl Track {
    pub fn new_alpha(target: AnimTarget) -> Self {
        Self {
            target,
            property: Property::Alpha {
                keyframes: Vec::new(),
            },
        }
    }

    pub fn new_translate_x(target: AnimTarget) -> Self {
        Self {
            target,
            property: Property::TranslateX {
                keyframes: Vec::new(),
            },
        }
    }

    pub fn new_translate_y(target: AnimTarget) -> Self {
        Self {
            target,
            property: Property::TranslateY {
                keyframes: Vec::new(),
            },
        }
    }

    pub fn new_rotate(target: AnimTarget) -> Self {
        Self {
            target,
            property: Property::Rotate {
                keyframes: Vec::new(),
            },
        }
    }

    pub fn new_scale(target: AnimTarget) -> Self {
        Self {
            target,
            property: Property::Scale {
                keyframes: Vec::new(),
            },
        }
    }

    pub fn with_keyframes(mut self, keyframes: Vec<Keyframe>) -> Self {
        match &mut self.property {
            Property::Alpha { keyframes: k }
            | Property::TranslateX { keyframes: k }
            | Property::TranslateY { keyframes: k }
            | Property::Rotate { keyframes: k }
            | Property::Scale { keyframes: k } => {
                *k = keyframes;
                // Keep deterministic ordering.
                k.sort_by(|a, b| {
                    a.time_s
                        .partial_cmp(&b.time_s)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
        self
    }

    pub fn bounds(&self) -> Option<(f32, f32)> {
        self.property.start_end()
    }
}

/// A timeline holds multiple tracks. Tracks are evaluated independently at time t.
#[derive(Debug, Default)]
pub struct Timeline {
    pub tracks: Vec<Track>,
}

impl Timeline {
    pub fn new() -> Self {
        Self { tracks: Vec::new() }
    }

    pub fn add_track(&mut self, track: Track) -> &mut Self {
        self.tracks.push(track);
        self
    }

    /// Compute the time bounds of the whole timeline.
    pub fn bounds(&self) -> Option<(f32, f32)> {
        let mut start = f32::INFINITY;
        let mut end = f32::NEG_INFINITY;
        let mut any = false;

        for tr in &self.tracks {
            if let Some((s, e)) = tr.bounds() {
                start = start.min(s);
                end = end.max(e);
                any = true;
            }
        }

        if any { Some((start, end)) } else { None }
    }

    /// Apply this timeline to the scene at time `t_s`.
    ///
    /// Notes:
    /// - For now, we only target root mobjects (by name).
    /// - We apply properties onto the existing `local_from_parent` (composing a new transform).
    /// - Order of application when multiple tracks affect the same object:
    ///   - Alpha is last (simple scalar overwrite)
    ///   - Transform-related tracks are composed in a fixed order:
    ///     Scale → Rotate → Translate
    ///
    /// This ordering is stable and predictable for demos, but may be revised later.
    pub fn apply(&self, scene: &mut Scene2D, t_s: f32) {
        // We build per-target transform parts.
        // For Phase 1, only root targets exist, so we use root index as key.
        #[derive(Debug, Default, Copy, Clone)]
        struct Parts {
            sx: Option<f32>,
            rot: Option<f32>,
            tx: Option<f32>,
            ty: Option<f32>,
            alpha: Option<f32>,
        }

        // Use a HashMap to avoid borrow-checker issues from "get-or-insert then return &mut".
        // This is still deterministic because:
        // - we apply onto `scene.roots` after collection
        // - per-object property conflicts are resolved by "last write wins" in track iteration order
        let mut parts: std::collections::HashMap<usize, Parts> = std::collections::HashMap::new();

        for tr in &self.tracks {
            let Some(idx) = tr.target.resolve_index(scene) else {
                continue;
            };

            let Some(v) = tr.property.sample(t_s) else {
                continue;
            };

            let p = parts.entry(idx).or_insert_with(Parts::default);

            match tr.property {
                Property::Alpha { .. } => p.alpha = Some(v),
                Property::TranslateX { .. } => p.tx = Some(v),
                Property::TranslateY { .. } => p.ty = Some(v),
                Property::Rotate { .. } => p.rot = Some(v),
                Property::Scale { .. } => p.sx = Some(v),
            }
        }

        for (idx, p) in parts {
            let Some(obj) = scene.roots.get_mut(idx) else {
                continue;
            };

            // Transform composition for the animated layer:
            // anim_from_parent = T * R * S
            //
            // IMPORTANT:
            // We intentionally do NOT overwrite the static placement transform.
            // Static layout/baseline anchoring lives in `base_from_parent`.
            // Rendering composes: local_from_parent = base_from_parent * anim_from_parent.
            //
            // Track semantics remain "absolute" for the animated layer (i.e. tx/ty are absolute
            // within anim space, not deltas).
            let s = p.sx.unwrap_or(1.0);
            let r = p.rot.unwrap_or(0.0);
            let tx = p.tx.unwrap_or(0.0);
            let ty = p.ty.unwrap_or(0.0);

            let xf_s = Affine2::scale(s, s);
            let xf_r = Affine2::rotate(r);
            let xf_t = Affine2::translate(tx, ty);

            obj.anim_from_parent = xf_t.mul(xf_r.mul(xf_s));

            // Keep the legacy/compat field in sync for any remaining call sites that read it
            // directly (during migration).
            obj.local_from_parent = obj.base_from_parent.mul(obj.anim_from_parent);

            if let Some(a) = p.alpha {
                obj.fill.a = a.clamp(0.0, 1.0);
            }
        }
    }
}
