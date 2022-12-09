#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_uv;
out vec2 out_uv;
void main(){
    out_uv = vec2(a_uv.x,a_uv.y);
    gl_Position = vec4(a_position, 1.0);
}