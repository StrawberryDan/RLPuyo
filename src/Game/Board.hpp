#pragma once
//======================================================================================================================
//  Includes
//----------------------------------------------------------------------------------------------------------------------
// Strawberry Core
#include "Strawberry/Core/Math/Vector.hpp"
#include "Strawberry/Core/Types/Optional.hpp"
// Standard Library
#include <cstdint>
#include <deque>
#include <random>


//======================================================================================================================
//  Class Declaration
//----------------------------------------------------------------------------------------------------------------------
namespace Strawberry::RLPuyo
{
    static constexpr uint8_t BOARD_WIDTH  =  5;
    static constexpr uint8_t BOARD_HEIGHT = 16;


    enum class Tile : uint8_t
    {
        EMPTY = 0,
        RED,
        BLUE,
        YELLOW,
        GREEN,
    };


    class PlaceableTiles
    {
    public:
        PlaceableTiles(Tile first, Tile second)
            : mPosition((BOARD_WIDTH + 1) / 2, 0)
            , mTiles{first, second}
        {}


        void Descend();
        void Swap();


        Core::Math::Vec2u Position() const;
        Tile              Top() const;
        Tile              Bottom() const;


    private:
        Core::Math::Vec2u mPosition;
        Tile              mTiles[2];
    };


    class Board
    {
    public:
        Board();


        void Step();


    private:
        std::random_device                                          mRandomDevice;
        std::uniform_int_distribution<std::underlying_type_t<Tile>> mTileDistribution{1, 4};


        Tile                           mTiles[BOARD_WIDTH][BOARD_HEIGHT] = {};
        Core::Optional<PlaceableTiles> mCurrentTiles;
        std::deque<Tile>               mTileQueue;
    };
}