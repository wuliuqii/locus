struct Uniforms2D {
  mvp: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> u: Uniforms2D;

struct VsOut {
  @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
  // Placeholder: draw nothing meaningful yet.
  // We still exercise the uniform binding and pipeline creation.
  var out: VsOut;
  out.position = u.mvp * vec4<f32>(0.0, 0.0, 0.0, 1.0);
  return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
  // Placeholder: no draw calls yet.
  return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
