#version 450
#extension GL_EXT_nonuniform_qualifier: require
layout (location = 0) in vec2 uv;
layout (location =0) out vec4 outColor;
layout (set = 0, binding = 2) uniform sampler2D bindlessTexture[];

layout(push_constant) uniform PushData {
    int offscreen;
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
void main()
{
    vec3 albedo = texture(bindlessTexture[pc.offscreen], uv).rgb;
    outColor = vec4(albedo, 1.0);
}
