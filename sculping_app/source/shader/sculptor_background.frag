#version 450
layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 outColor;

void main() {
  vec3 topColor = vec3(0.05, 0.05, 0.07);
  vec3 bottomColor = vec3(0.0, 0.0, 0.0);
  outColor = vec4(mix(bottomColor, topColor, uv.y), 1.0);
}
