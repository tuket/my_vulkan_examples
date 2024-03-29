#version 450
#pragma shader_stage(vertex)

layout(location = 0)in vec2 a_pos;
layout(location = 1)in vec2 a_tc;

// redefine the default gl_PerVertex: https://www.khronos.org/opengl/wiki/Built-in_Variable_(GLSL)#Vertex_shader_outputs
out gl_PerVertex
{
    vec4 gl_Position;
};
layout (location = 1)out vec2 v_tc;

void main()
{
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_tc = a_tc;
}
