//! Render utility helpers.
//!
//! This module is intentionally small and dependency-light. It provides helpers
//! commonly needed for interactive demos:
//! - time tracking (dt / elapsed)
//! - simple oscillators / easing
//! - numeric helpers for animation

use std::time::{Duration, Instant};

/// A simple frame timer that tracks:
/// - `elapsed`: seconds since creation
/// - `dt`: seconds since the last `tick()`
///
/// Typical usage:
/// - Create once in your state: `let mut clock = FrameClock::new();`
/// - Each frame: `let dt = clock.tick();`
///
/// Note:
/// - `tick()` clamps unreasonable `dt` (e.g. when resuming from a breakpoint).
#[derive(Debug, Clone)]
pub struct FrameClock {
    start: Instant,
    last: Instant,
    /// Max dt allowed from `tick()` (in seconds).
    max_dt: f32,
}

impl FrameClock {
    /// Create a new clock with a reasonable default `max_dt` clamp.
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            start: now,
            last: now,
            max_dt: 0.1, // 100ms
        }
    }

    /// Set the `max_dt` clamp for `tick()`.
    #[inline]
    pub fn with_max_dt(mut self, max_dt: f32) -> Self {
        self.max_dt = max_dt.max(0.0);
        self
    }

    /// Seconds since this clock was created.
    #[inline]
    pub fn elapsed_s(&self) -> f32 {
        self.start.elapsed().as_secs_f32()
    }

    /// Duration since this clock was created.
    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Advance the clock and return `dt` in seconds.
    ///
    /// `dt` is clamped to `[0, max_dt]` to avoid destabilizing animations.
    #[inline]
    pub fn tick(&mut self) -> f32 {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32();
        self.last = now;
        dt.clamp(0.0, self.max_dt)
    }

    /// Reset the clock start time (and last tick) to now.
    #[inline]
    pub fn reset(&mut self) {
        let now = Instant::now();
        self.start = now;
        self.last = now;
    }
}

impl Default for FrameClock {
    fn default() -> Self {
        Self::new()
    }
}

/// A sinusoidal oscillator in `[0, 1]`.
///
/// - `t`: time in seconds
/// - `hz`: cycles per second
#[inline]
pub fn osc_01(t: f32, hz: f32) -> f32 {
    // sin in [-1,1] -> map to [0,1]
    0.5 + 0.5 * (std::f32::consts::TAU * hz * t).sin()
}

/// A sinusoidal oscillator in `[-1, 1]`.
///
/// - `t`: time in seconds
/// - `hz`: cycles per second
#[inline]
pub fn osc_pm1(t: f32, hz: f32) -> f32 {
    (std::f32::consts::TAU * hz * t).sin()
}

/// A "breathing" scale around 1.0.
///
/// - `amplitude`: e.g. `0.04` for +/-4%
/// - `hz`: e.g. `0.3` for a slow breathe
#[inline]
pub fn breathe(t: f32, amplitude: f32, hz: f32) -> f32 {
    1.0 + amplitude * osc_pm1(t, hz)
}

/// Linear interpolation.
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Smoothstep easing in `[0,1]`.
///
/// Returns 0 at t<=0, 1 at t>=1.
#[inline]
pub fn smoothstep01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// An ease-in-out curve (smoother than `smoothstep`).
#[inline]
pub fn smootherstep01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Ping-pong a value over `[0, 1]` given a monotonically increasing time.
///
/// Useful for "write-in" progress that goes 0->1->0->1...
#[inline]
pub fn ping_pong_01(t: f32, period_s: f32) -> f32 {
    let p = if period_s <= 0.0 { 1.0 } else { period_s };
    let x = (t / p) % 2.0; // 0..2
    if x <= 1.0 { x } else { 2.0 - x }
}

/// Loop a value over `[0, 1)` given a monotonically increasing time.
#[inline]
pub fn loop_01(t: f32, period_s: f32) -> f32 {
    let p = if period_s <= 0.0 { 1.0 } else { period_s };
    (t / p) % 1.0
}
