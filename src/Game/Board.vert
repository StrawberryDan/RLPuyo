#version 450


layout (set=0,binding=0) uniform Constants
{
    mat4 projection;
    uvec2 boardSize;
};


layout (set=1, binding=0) buffer BoardState
{
    uint  boardIndex;
    ivec2 fallingTilesPosition;
    uint  tiles[];
};


const vec2 Positions[] = {
    {0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0},
    {0.0, 0.0}, {1.0, 1.0}, {1.0, 0.0}
};


layout (location = 0) out vec2 fragPosition;
layout (location = 1) out flat uint tileIndex;


const uvec2 BOARD_OFFSET = uvec2(160, 10);
const float BOARD_GAP    = 100;
const float TILE_SIZE    =  25;
const float GAP_SIZE     =   2;


void main()
{
    vec2 boardOffset = BOARD_OFFSET + vec2(boardIndex * ((boardSize.x * (TILE_SIZE + GAP_SIZE)) + BOARD_GAP), 0.0);

    tileIndex = gl_InstanceIndex;
    fragPosition = Positions[gl_VertexIndex];

    // If this is a falling til
    if (tileIndex >= boardSize.x * boardSize.y)
    {
        bool top = tileIndex == boardSize.x * boardSize.y;
        vec2 tileOffset = fallingTilesPosition * (TILE_SIZE + GAP_SIZE);
        if (!top) tileOffset.y -= (TILE_SIZE + GAP_SIZE);
        tileOffset.y = (boardSize.y - 2) * (TILE_SIZE + GAP_SIZE) - tileOffset.y;
        gl_Position = projection *  vec4(boardOffset + tileOffset + TILE_SIZE * fragPosition, 0.0, 1.0);
    }
    // If this is a stationary tile
    else
    {
        vec2 tileOffset = vec2((gl_InstanceIndex % boardSize.x) * (TILE_SIZE + GAP_SIZE), (gl_InstanceIndex / boardSize.x) * (TILE_SIZE + GAP_SIZE));
        gl_Position = projection * vec4(boardOffset + tileOffset + TILE_SIZE * fragPosition, 0.0, 1.0);
    }
}