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

void main() {
    vec3 color = texture(bindlessTexture[pc.lightningBuffer], uv).rgb;

    // 가중치 기반 밝기 계산
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));

    float threshold = 0.4;
    float knee = 0.1; // 부드럽게 시작
    float factor = smoothstep(threshold, threshold + knee, brightness);
    vec3 bloom = color * factor;
    outColor = vec4(bloom, 1.0);

}
