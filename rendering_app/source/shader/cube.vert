#version 450
#extension GL_EXT_nonuniform_qualifier: require
layout (location = 0) out vec2 uv;

float cubeVertices[] = {
-1.0f, -1.0f, 1.0f,
1.0f, -1.0f, 1.0f,
1.0f, 1.0f, 1.0f,

-1.0f, -1.0f, 1.0f,
1.0f, 1.0f, 1.0f,
-1.0f, 1.0f, 1.0f,

-1.0f, -1.0f, -1.0f,
1.0f, 1.0f, -1.0f,
1.0f, -1.0f, -1.0f,

-1.0f, -1.0f, -1.0f,
-1.0f, 1.0f, -1.0f,
1.0f, 1.0f, -1.0f,

-1.0f, -1.0f, -1.0f,
-1.0f, -1.0f, 1.0f,
-1.0f, 1.0f, 1.0f,

-1.0f, -1.0f, -1.0f,
-1.0f, 1.0f, 1.0f,
-1.0f, 1.0f, -1.0f,

1.0f, -1.0f, -1.0f,
1.0f, 1.0f, 1.0f,
1.0f, -1.0f, 1.0f,

1.0f, -1.0f, -1.0f,
1.0f, 1.0f, -1.0f,
1.0f, 1.0f, 1.0f,

-1.0f, 1.0f, -1.0f,
-1.0f, 1.0f, 1.0f,
1.0f, 1.0f, 1.0f,

-1.0f, 1.0f, -1.0f,
1.0f, 1.0f, 1.0f,
1.0f, 1.0f, -1.0f,

-1.0f, -1.0f, -1.0f,
1.0f, -1.0f, 1.0f,
-1.0f, -1.0f, 1.0f,

-1.0f, -1.0f, -1.0f,
1.0f, -1.0f, -1.0f,
1.0f, -1.0f, 1.0f
};
void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 1.0, 1.0);
    uv = (positions[gl_VertexIndex] + 1.0) * 0.5;
}
