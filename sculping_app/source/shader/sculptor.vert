
#version 450
layout(location = 0) in  vec3 inPosition;
layout(location = 1) in  vec3 inNormal;
layout(location = 2) in  vec2 inUV;
layout(location =0) out vec2 fragTexCoord;
layout(set = 0, binding = 0 ) uniform cameraUBD{
    mat4 view; 
    mat4 proj;
} camera;

layout(push_constant) uniform PushConstants {
    mat4 model;
    vec4 color;
} pc;

void main() {
    gl_Position = camera.proj* camera.view*vec4(inPosition, 1.0);
    fragTexCoord = inUV;
}
