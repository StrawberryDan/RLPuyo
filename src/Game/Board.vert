#version 450


layout (push_constant) uniform PushConstants
{
    vec2 position;
    vec3 color;
};


void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
}