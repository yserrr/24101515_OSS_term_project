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
    int bloomingExtract;
    int lightningBuffer;
    int shadowXp ;

    int shadowXm ;
    int shadowYp ;
    int shadowYm ;
    int shadowZp ;
    int shadowZm ;

    int bloomingBlur;
    int ToneMapping;
    int gammaCollection;
} pc;

const float gamma = 0.5;
void main() {
    vec3 color = texture(bindlessTexture[pc.ToneMapping], uv).rgb;
    vec3 corrected = pow(color, vec3(1.0 / gamma));
    outColor = vec4(corrected, 1.0);
}
