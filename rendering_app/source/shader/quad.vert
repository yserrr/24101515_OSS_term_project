#version 450
#extension GL_EXT_nonuniform_qualifier: require
layout (location = 0) out vec2 uv;

vec2 positions[6] = vec2[](vec2(-1, -1),
                           vec2(-1, 1),
                           vec2(1, 1),
                           vec2(1, 1),
                           vec2(1, -1),
                           vec2(-1, -1));
void main() {
  gl_Position = vec4(positions[gl_VertexIndex], 1.0, 1.0);
  uv = (positions[gl_VertexIndex] + 1.0) * 0.5;
}
