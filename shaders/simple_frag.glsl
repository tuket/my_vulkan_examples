#version 450
#pragma shader_stage(fragment)

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec2 v_tc;

layout(location = 0) out vec4 o_color;

layout(set = 0, binding = 1) uniform sampler2D u_colorTex;

void main()
{
    vec4 color = texture(u_colorTex, v_tc);
    o_color = vec4(v_color * color.rgb, color.a);
}