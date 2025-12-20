//! Font database module (placeholder).
//!
//! This file exists to satisfy the module tree during the project's modularization.
//! The real implementation will live here shortly and is expected to:
//! - Build and own a `fontdb::Database`
//! - Load system fonts (cross-platform) and optional embedded fallback fonts
//! - Provide access to font face bytes (file-backed or embedded)
//! - Provide stable IDs / caching hooks for parsed font faces
//!
//! For now, this module provides only minimal stubs so other modules can compile.

/// Placeholder type for the font database.
///
/// Once implemented, this will likely wrap `fontdb::Database` and additional caches.
#[derive(Debug, Default)]
pub struct FontDatabase {
    _private: (),
}

impl FontDatabase {
    /// Create a new database.
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Load system fonts.
    ///
    /// Planned behavior:
    /// - delegate to `fontdb::Database::load_system_fonts()`
    /// - optionally add embedded fallback fonts
    pub fn load_system_fonts(&mut self) {
        // TODO: implement
    }

    /// Returns the number of discovered font faces.
    pub fn face_count(&self) -> usize {
        // TODO: implement
        0
    }
}
