#version 330 core
in vec2 out_uv;
uniform sampler2D texture0;
out vec4 fragcolor;
void main(){
    //fragcolor = vec4(1.0,0,0,1.0);
    fragcolor = texture2D(texture0,out_uv);
}