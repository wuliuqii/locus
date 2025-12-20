//! Example binary: compile the ζ validation formula with Typst and log the frame tree.
//!
//! This is intentionally an `examples/` target so the main library surface stays clean.
//!
//! Run:
//! - `cargo run --example typst_zeta_log_tree`

use anyhow::Context as _;
use log::info;

fn main() -> anyhow::Result<()> {
    // Keep logging setup in the example binary (library stays unopinionated).
    env_logger::init();

    // Compile the canonical Phase A validation formula and log the frame tree.
    //
    // IMPORTANT:
    // - The canonical formula is defined in `locus::typst::math::ZETA_VALIDATION_FORMULA`.
    // - It uses Typst-native math syntax (not LaTeX backslash commands).
    let compiled = locus::typst::engine::compile_zeta_validation()
        .context("typst: failed to compile ζ validation formula")?;

    info!("Typst compile ok: {}", compiled.debug_summary);

    // Log nested frame structure to validate group nesting and transforms.
    locus::typst::engine::extract::log_paged_document_tree(&compiled.document, 6);

    Ok(())
}
