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
    uint tiles[];
};


layout (location = 0) in vec2 fragPosition;
layout (location = 1) in flat uint tileIndex;


layout (location = 0) out vec4 fragColor;


const float TILE_CENTER_RADIUS = 0.35;


vec3 GetTileColor(uint id)
{
    switch (id)
    {
        case 1:
            return vec3(0.5, 0.0, 0.0);
        case 2:
            return vec3(0.0, 0.0, 0.5);
        case 3:
            return vec3(0.5, 0.5, 0.0);
        case 4:
            return vec3(0.0, 0.5, 0.0);
    }

    return vec3(1.0, 1.0, 1.0);
}


void main()
{
    uint tileType = tiles[tileIndex];
    if (tileType == 0)
    {
        fragColor = vec4(0.0);
        return;
    }

    vec3 tileColor = GetTileColor(tileType);

    vec2 fromCenter = fragPosition - vec2(0.5, 0.5);
    bool inCenter = abs(fromCenter.x) < TILE_CENTER_RADIUS && abs(fromCenter.y) < TILE_CENTER_RADIUS;
    bool edgesA = fragPosition.x > fragPosition.y;
    bool edgesB = fragPosition.x > 1.0 - fragPosition.y;


    if (!inCenter)
    {
        if (edgesA && edgesB)
        {
            tileColor *= 2.00;
        }
        else if (edgesA && ! edgesB)
        {
            tileColor *= 0.50;
        }
        else if (!edgesA && edgesB)
        {
            tileColor  *= 1.50;
        }
        else
        {
            tileColor *= 0.25;
        }
    }

    fragColor = vec4(tileColor, 1.0);
}