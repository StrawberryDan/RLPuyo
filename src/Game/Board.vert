#version 450


layout (set=0,binding=0) uniform Constants
{
    mat4 projection;
    uvec2 boardSize;
    vec2 tileSize;
    vec2 gapSize;
};


const vec2 Positions[] = {
    {0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0},
    {0.0, 0.0}, {1.0, 1.0}, {1.0, 0.0}
};


layout (location = 0) out vec2 fragPosition;
layout (location = 1) out flat uint tileIndex;


void main()
{
    tileIndex = gl_InstanceIndex;

    vec2 offset = vec2((gl_InstanceIndex % boardSize.x) * (tileSize.x + gapSize.x), (gl_InstanceIndex / boardSize.x) * (tileSize.y + gapSize.y));

    fragPosition = Positions[gl_VertexIndex];
    gl_Position = projection *  vec4(offset + tileSize * fragPosition, 0.0, 1.0);
}