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
    int DepthBuffer;
    int ShaderBuffer;
    int lightningBuffer;
    int postProcessBuffer;

    int albedoTex;
    int normalTex;
    int roughnessTex;
    int cubeTex;

} pc;

struct GPULight {
    vec4 position;
    vec4 direction;
    vec4 color;
    vec4 padding;
    mat4 view;
    mat4 proj;
};
layout (std140, set = 0, binding = 1) uniform LightBuffer {
    GPULight lights[4];
    int lightCount;
} lightBuffer;
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
    vec3 albedo = texture(bindlessTexture[pc.gBufferAlbedo], uv).rgb;
    vec3 normal = normalize(texture(bindlessTexture[pc.gBufferNormal], uv).rgb * 2.0 - 1.0);
    vec3 worldPos = texture(bindlessTexture[pc.gBufferPositon], uv).rgb;
    float roughness = max(texture(bindlessTexture[pc.gBufferRoughness], uv).r, 0.05);

    vec3 color = vec3(0.0);

    for (int i = 0; i < lightBuffer.lightCount; ++i) {
        GPULight light = lightBuffer.lights[i];
        vec3 lightDir = normalize(light.position.xyz-worldPos);
        vec3 viewDir  = normalize(camera.camPos - worldPos);
        vec3 halfDir  = normalize(lightDir + viewDir);
        float NdotL = max(dot(normal, lightDir), 0.0);
        float NdotH = max(dot(normal, halfDir), 0.0);
        vec3 diffuse = albedo * light.color.rgb * NdotL;
        vec3 specular = light.color.rgb * pow(NdotH, 1.0/roughness);
        vec3 lightning = mix(diffuse, specular, 0.8);
        color += lightning;
    }
    color = clamp(color, 0.0, 1.0);
    outColor = vec4(color, 1.0);
}
