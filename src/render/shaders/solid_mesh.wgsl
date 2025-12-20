struct Uniforms {
  mvp: mat4x4<f32>,
  color: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> u: Uniforms;

struct VsOut {
  @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@location(0) a_pos: vec2<f32>) -> VsOut {
  var out: VsOut;
  out.position = u.mvp * vec4<f32>(a_pos, 0.0, 1.0);
  return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
  return u.color;
}
