#version 450
#extension GL_EXT_nonuniform_qualifier: require

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 gPosition;
layout(location = 1) out vec4 gAlbedo;
layout(location = 2) out vec4 gNormal;
layout(location = 3) out vec4 gBufferRoughness;

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
    mat4 modelMatrix_;
    vec4 color;
    int albedoTexture;
    int normalTexture;
    int roughnessTexture;
    int metalnessTexture;
} pc;

void main()
{
    vec3 albedo = texture(bindlessTexture[pc.albedoTexture], fragUV).xyz;
    gAlbedo = vec4(albedo, 1.0);
    gPosition = vec4(fragPosition, 1.0);

    vec3 normal = fragNormal;
    gNormal = vec4(normalize(normal) * 0.5 + 0.5, 1.0);
    if (pc.normalTexture>0){
        normal = texture(bindlessTexture[pc.normalTexture], fragUV).xyz;
        gNormal = vec4(normal,1.0);
    }
    vec3 roughness = vec3(1.0);
    if (pc.roughnessTexture>=0){
        roughness = texture(bindlessTexture[pc.roughnessTexture], fragUV).xyz;
    }
    gBufferRoughness = vec4(roughness, 1.0);

}
