mod font;
mod render;
mod scene;
mod typst;

fn main() -> anyhow::Result<()> {
    // Keep logs available for when we start wiring Typst and font loading.
    env_logger::init();

    // Hand control to the rendering/app layer.
    //
    // This keeps `main.rs` tiny and makes it easy to swap demos:
    // - write-in path effect
    // - typst math-only rendering (A)
    // - later: full page rendering (B)
    render::app::run()
}
