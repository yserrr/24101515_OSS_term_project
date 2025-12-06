#version 450
#extension GL_EXT_debug_printf: enable

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inTangent;
layout (location = 4) in vec3 inBitangent;
layout (location = 5) in vec4 inColor;
layout (location = 6) in ivec4 inBoneIndices;
layout (location = 7) in vec4 inBoneWeights;

layout (location = 0) out vec3 outPos;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec2 fragTexCoord;
layout (location = 3) out mat3 TBN;
layout (location = 6) out vec3 camPos;
layout (location = 7) out vec2 background;

layout (set = 0, binding = 0) uniform cameraUBD {
  mat4 view;
  mat4 proj;
  vec3 camPos;
} camera;

layout (push_constant) uniform PushConstants {
  mat4 model;
  vec4 color;
  uint bindlessIndex;
} constant;

vec2 positions[4] = vec2[](vec2(-1,-1), vec2(1,-1), vec2(1,1), vec2(-1,1));
void main() {

  vec3 N = normalize(inNormal);
  vec3 B = normalize(inBitangent);

  float handedness = 1.0;
  vec3 T = cross(B, N) * handedness;

  vec4 worldPos = constant.model * vec4(inPos.xyz, 1.0);
  gl_Position = camera.proj * camera.view * worldPos;
  outPos = worldPos.xyz;
  outNormal = inNormal;
  fragTexCoord = inUV;
  camPos = camera.camPos;
  TBN = mat3(T, B, N);
  background = (positions[gl_VertexIndex] + 1.0) * 0.5;
}
