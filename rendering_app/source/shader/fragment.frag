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
layout (set = 0, binding = 0) uniform cameraUBD {
    mat4 view;
    mat4 proj;
    vec3 camPos;
    float fov_ = glm::radians(45.0f);
    float aspect = 20 / 12;
    float nearPlane = 0.01f;
    float farPlane = 1000.0f;
} camera;
layout (location = 0) in vec3 fragPos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;
layout (location = 3) in mat3 TBN;
layout (location = 6) in vec3 camPosIn;
layout (location = 7) in vec2 uv;
layout (location = 0) out vec4 outColor;

layout (set = 1, binding = 0) uniform sampler2D bindlessTexture[];

layout (push_constant) uniform PushConstants {
    mat4 model;
    vec4 color;

    int albedoIndex;
    int normalTextureIndex;
    int metalicTextureIndex;
    int padding;

    vec3 emissiveColor;
    float fFadding;

    float metalic;
    float roughness;
    float ao;
    float normalScale;

    float alpha;
    float paddingf;
    float paddingf2;
    float paddingf3;
} mat;

layout (std140, set = 0, binding = 1) uniform LightBuffer {
    GPULight lights[16];
    int lightCount;
};


vec3 getNormal() {
    vec3 N = normalize(normal);
    if (mat.normalTextureIndex < 0) return N;
    vec3 n = texture(bindlessTexture[mat.normalTextureIndex], texCoord).rgb;
    n = n * 2.0 - 1.0;
    vec3 T = normalize(TBN[0]);
    vec3 B = normalize(TBN[1]);
    vec3 N_tbn = normalize(TBN[2]);
    mat3 tbnMatrix = mat3(T, B, N_tbn);
    return normalize(tbnMatrix * n * mat.normalScale);
}

vec3 PBRLighting(vec3 albedo, float metallic, float roughness, float ao, vec3 N, vec3 V) {
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < lightCount; ++i) {
        GPULight light = lights[i];
        vec3 L;
        float attenuation = 1.0;
        if (light.position.w == 0.0) {
            L = normalize(- vec3(light.position));
        } else {
            vec3 lightPos = vec3(light.position);
            L = normalize(lightPos - fragPos);
            float distance = length(lightPos - fragPos);
            attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
            if (light.position.w == 2.0) {
                float theta = dot(L, normalize(- vec3(light.direction)));
                float innerCutoff = cos(light.coneAngles.x);
                float outerCutoff = cos(light.coneAngles.y);
                float epsilon = innerCutoff - outerCutoff;
                float spotFactor = clamp((theta - outerCutoff) / epsilon, 0.0, 1.0);
                attenuation *= spotFactor;
            }
        }

        vec3 H = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        float VdotH = max(dot(V, H), 0.0);
        vec3 F0 = mix(vec3(0.04), albedo, metallic);
        vec3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
        float alpha = roughness * roughness;
        float alpha2 = alpha * alpha;
        float denom = (NdotH * NdotH) * (alpha2 - 1.0) + 1.0;
        float D = alpha2 / (3.141592 * denom * denom);
        float k = alpha / 2.0;
        float G_V = NdotV / (NdotV * (1.0 - k) + k);
        float G_L = NdotL / (NdotL * (1.0 - k) + k);
        float G = G_V * G_L;
        vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);
        vec3 kD = (1.0 - F) * (1.0 - metallic);
        vec3 diffuse = kD * albedo / 3.141592;

        Lo += (diffuse + specular) * light.color.rgb * light.color.a * NdotL * attenuation;
    }

    vec3 ambient = albedo * ao;
    return Lo + ambient;
}

void main() {
    vec3 N = getNormal();
    vec3 V = normalize(camPosIn - fragPos);

    vec3 albedo = mat.color.rgb;
    if (mat.albedoIndex >= 0){
        albedo *= texture(bindlessTexture[mat.albedoIndex], texCoord).rgb;
    }

    float metallic = mat.metalic;
    if (mat.metalicTextureIndex >= 0){
        metallic *= texture(bindlessTexture[mat.metalicTextureIndex], texCoord).r;
    }

    float roughness = mat.roughness;
    float ao = mat.ao;

    vec3 color = PBRLighting(albedo, metallic, roughness, ao, N, V);
    color += mat.emissiveColor;

    color = pow(color, vec3(1.0 / 2.2));


    outColor = vec4(color, 1);
}
