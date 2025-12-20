//! Rendering module root.
//!
//! This crate is being modularized from a single-file prototype into a set of
//! focused modules. The `render` module owns the window/event-loop integration
//! and the GPU renderer(s).
//!
//! Current entrypoint: `render::app::run()`.

pub mod app;

/// Common GPU types used across render submodules.
pub mod gpu;

/// Minimal color/shape pipelines that don't depend on Typst.
pub mod primitives;

/// Utility helpers for render-time math, conversions, etc.
pub mod util;

/// A simple solid-color mesh renderer for scene draw items.
pub mod mesh_renderer;
