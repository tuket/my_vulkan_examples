#version 450
#pragma shader_stage(vertex)

layout(location = 0)in vec2 a_pos;
layout(location = 1)in vec3 a_color;

layout(binding = 0) uniform UBO {
    vec3 u_color;
};

// redefine the default gl_PerVertex: https://www.khronos.org/opengl/wiki/Built-in_Variable_(GLSL)#Vertex_shader_outputs
out gl_PerVertex
{
    vec4 gl_Position;
};
layout (location = 0)out vec3 v_color;

void main()
{
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_color = mix(a_color, u_color, 0.5);
}
