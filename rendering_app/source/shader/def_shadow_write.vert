#version 450
#extension GL_EXT_debug_printf: enable
layout (location = 0) out vec3 outPos;

layout(push_constant) uniform PushData {
    mat4 modelMatrix_;
    vec4 color;
    int albedoTexture;
    int normalTexture;
    int roughnessTexture;
    int metalnessTexture;
    int lightBufferIndex;
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


layout (set = 0, binding = 3) uniform lightUBO{
    mat4 view;
    mat4 proj;
}lightBuffer;

void main() {
    vec4 worldPos = vec4(inPos.xyz, 1.0);
    gl_Position = lightBuffer.proj * lightBuffer.view * pc.modelMatrix_ *worldPos;

}
