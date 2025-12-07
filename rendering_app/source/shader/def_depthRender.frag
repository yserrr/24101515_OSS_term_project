#version 450
#extension GL_EXT_nonuniform_qualifier: require
layout (location = 0) in vec2 uv;
layout (location =0) out vec4 outColor;
layout (set = 0, binding = 2) uniform sampler2D bindlessTexture[];

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
void main()
{
    float f  = texture(bindlessTexture[pc.depthBuffer], uv).r;
    float d = 1-f;
    float l = pow(d, 0.9);
    outColor = vec4(l, l, l, 1.0);
}
