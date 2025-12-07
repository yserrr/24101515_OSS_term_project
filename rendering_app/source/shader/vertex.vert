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

layout(push_constant) uniform PushData {
    mat4 modelMatrix_;
    vec4 color;
    int albedoTexture;
    int normalTexture;
    int roughnessTexture;
    int metalnessTexture;
} pc;


layout (set = 0, binding = 0) uniform cameraUBD {
    mat4 view;
    mat4 proj;
    vec3 camPos;
    float padding;
    float fov_;
    float aspect;
    float nearPlane;
    float farPlane;
    mat4 invView;
    mat4 invProj;
} camera;

vec2 positions[4] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, 1));
void main() {

    vec3 N = normalize(inNormal);
    vec3 B = normalize(inBitangent);

    float handedness = 1.0;
    vec3 T = cross(B, N) * handedness;

    vec4 worldPos = vec4(inPos.xyz, 1.0);
    gl_Position = camera.proj * camera.view * pc.modelMatrix_ *worldPos;
    outPos = worldPos.xyz;

    outNormal = inNormal;

    fragTexCoord = inUV;
}
