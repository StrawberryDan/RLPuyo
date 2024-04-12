#version 450


layout (set=0,binding=0) uniform Constants
{
    mat4 projection;
    uvec2 boardSize;
    vec2 tileSize;
    vec2 gapSize;
};


layout (set=1, binding=0) buffer BoardState
{
    uvec2 fallingTilesPosition;
    uint  tiles[];
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
    fragPosition = Positions[gl_VertexIndex];

    if (tileIndex >= boardSize.x * boardSize.y)
    {
        bool top = tileIndex == boardSize.x * boardSize.y;
        vec2 offset = fallingTilesPosition * (tileSize + gapSize);
        if (!top) offset.y -= (tileSize.y + gapSize.y);
        offset.y = (boardSize.y - 1) * (tileSize.y + gapSize.y) - offset.y;
        gl_Position = projection *  vec4(offset + tileSize * fragPosition, 0.0, 1.0);
    }
    else
    {
        vec2 offset = vec2((gl_InstanceIndex % boardSize.x) * (tileSize.x + gapSize.x), (gl_InstanceIndex / boardSize.x) * (tileSize.y + gapSize.y));
        gl_Position = projection *  vec4(offset + tileSize * fragPosition, 0.0, 1.0);
    }
}