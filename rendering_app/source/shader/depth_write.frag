#version 450
#extension GL_EXT_nonuniform_qualifier: require
#define PI = 3.141592
struct GPULight {
    vec4 position;
    vec4 direction;
    vec4 color;
    vec2 coneAngles;
    vec2 padding;
};
layout(push_constant) uniform PushData {
    int gBufferPositon;
    int gBufferNormal;
    int gBufferAlbedo;
    int gBufferRoughness;

    int depthBuffer;
    int shadowBuffer;
    int lightningBuffer;
    int postProcessBuffer;

    int albedoTex;
    int normalTex;
    int roughnessTex;
    int cubeTex;

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
layout (location = 0) in vec3 fragPos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;

void main() {
}