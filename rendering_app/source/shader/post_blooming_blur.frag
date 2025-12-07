#version 450
#extension GL_EXT_nonuniform_qualifier: require
layout (location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout (set = 0, binding = 2) uniform sampler2D bindlessTexture[];

layout(push_constant) uniform PushData {
    int gBufferPositon;
    int gBufferNormal;
    int gBufferAlbedo;
    int gBufferRoughness;
    int DepthBuffer;
    int ShaderBuffer;
    int lightningBuffer;
    int bloomingExtract;
    int bloomingBlur;
    int ToneMapping;
    int gammaCollection;
} pc;

void main() {
    vec2 texelSize = 1.0 / vec2(textureSize(bindlessTexture[pc.bloomingExtract], 0));
    vec3 result = vec3(0.0);
    for (int x=-1;x<=1;x++){
        for (int y=-1;y<=1;y++){
            vec2 offset = vec2(x, y) * texelSize;
            result += texture(bindlessTexture[pc.bloomingExtract], uv + offset).rgb;
        }
    }
    outColor = vec4(result / 9.0, 1.0);
}
