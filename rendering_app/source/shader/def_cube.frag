#version 450
#extension GL_EXT_nonuniform_qualifier: require

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 gPosition;
layout(location = 1) out vec4 gAlbedo;
layout(location = 2) out vec4 gNormal;

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


void main() {
    vec2 uv = (gl_FragCoord.xy / screenSize) * 2.0 - 1.0;
    vec4 dir = camera.invProj * vec4(uv, 1.0, 1.0);
    vec3 worldDir = normalize((camera.invView * vec4(dir.xyz, 0.0)).xyz);
    outColor = texture(uSkybox, worldDir);
}

