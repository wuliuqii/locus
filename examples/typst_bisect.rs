//! Example binary: bisect the `unknown variable: um` Typst compile error.
//!
//! Run:
//! - `cargo run --example typst_bisect`
//! - `RUST_LOG=info cargo run --example typst_bisect`
//!
//! What it does:
//! - Compiles a sequence of increasingly complex Typst snippets.
//! - Prints PASS/FAIL per snippet and shows the error if it fails.
//!
//! Goal:
//! - Determine the smallest snippet that triggers `unknown variable: um`.
//! - Once we have that, we can inspect Typst diagnostics and adjust our `World`
//!   provisioning (library/packages/units) accordingly.
//!
//! Important:
//! - These cases use Typst's native math commands **without** LaTeX-style backslashes.
//!   Typst math uses `sum`, `frac`, `infty`, `zeta`, etc.

use std::time::Instant;

fn main() -> anyhow::Result<()> {
    // Keep logging setup in the example binary.
    // (The library should remain unopinionated.)
    env_logger::init();

    println!("typst_bisect: starting…");

    // Each case is (name, source).
    // Keep snippets short and deterministic.
    let cases: Vec<(&str, &str)> = vec![
        ("plain_text", "hello"),
        ("math_x", "$x$"),
        ("math_zeta_symbol", "$zeta$"),
        ("math_infty_symbol", "$oo$"),
        ("math_sum_symbol", "$sum$"),
        ("math_sum_subsup_min", "$sum_(n=1)^(oo) 1$"),
        ("math_frac_min", "$frac(1, n)$"),
        ("math_power_min", "$n^s$"),
        ("math_zeta_full", "$zeta(s) = sum_(n=1)^(oo) frac(1, n^s)$"),
        // If the above fails, try breaking it slightly differently:
        ("math_zeta_eq_sum", "$zeta(s) = sum_(n=1)^(oo) 1$"),
        ("math_sum_frac_only", "$sum_(n=1)^(oo) frac(1, n)$"),
        ("math_frac_power_only", "$frac(1, n^s)$"),
    ];

    let mut passed = 0usize;
    let mut failed = 0usize;

    let t0 = Instant::now();

    for (idx, (name, src)) in cases.iter().enumerate() {
        println!();
        println!("=== Case {:02}: {} ===", idx + 1, name);
        println!("source: {}", src);

        let start = Instant::now();
        let res = locus::typst::engine::compile_math_only(src);
        let elapsed = start.elapsed();

        match res {
            Ok(compiled) => {
                passed += 1;
                println!(
                    "PASS ({} ms): {}",
                    elapsed.as_millis(),
                    compiled.debug_summary
                );
            }
            Err(err) => {
                failed += 1;
                println!("FAIL ({} ms)", elapsed.as_millis());
                println!("error: {:#}", err);
                println!();
                println!("--- FAILING SOURCE (for copy/paste) ---");
                println!("{}", src);
                println!("--- END SOURCE ---");

                // If you want to stop at first failure (typical bisect), uncomment:
                // break;
            }
        }
    }

    println!();
    println!(
        "typst_bisect: done in {} ms (passed={}, failed={})",
        t0.elapsed().as_millis(),
        passed,
        failed
    );

    // If we saw any failures, return an error to make it obvious in CI/automation.
    if failed > 0 {
        anyhow::bail!(
            "typst_bisect: {} case(s) failed; inspect output above for smallest failing snippet",
            failed
        );
    }

    Ok(())
}
