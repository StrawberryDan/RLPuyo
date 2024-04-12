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
#include <unordered_set>
#include <set>


//======================================================================================================================
//  Class Declaration
//----------------------------------------------------------------------------------------------------------------------
namespace Strawberry::RLPuyo
{
    static constexpr uint8_t BOARD_WIDTH  = 10;
    static constexpr uint8_t BOARD_HEIGHT = 20;


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


        void MoveLeft();
        void MoveRight();
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


		Tile GetTile(Core::Math::Vec2u position) const noexcept;
        void Step();


    protected:
		/// Refill the queue with random tiles until it is full again.
        void ReplenishQueue();
		/// Resolve the board after tiles have been placed.
		/// @param candidates The set of tiles which may have formed groups.
		void Resolve(std::unordered_set<Core::Math::Vec2u> candidates);
		/// Finds tiles of the same type which are adjacent to the root tile provided.
		/// @param root the cell from which to search.
		/// @return The positions of all the tiles connected to the root (inclusive).
		std::unordered_set<Core::Math::Vec2u> FindConnectedTiles(Core::Math::Vec2u root);
		/// Removes the given tiles and returns the columns to which gravity need be applied.
		/// @param The tiles to eliminate.
		/// @return The set of columns to which gravity need be applied.
		std::set<unsigned int> EliminateTiles(std::unordered_set<Core::Math::Vec2u> tiles);
		/// Applys gravity to the provided columns.
		/// @param columns The columns to which to apply gravity.
		/// @return The new positions of the tiles which descended due to gravity.
		std::unordered_set<Core::Math::Vec2u> ApplyGravity(std::set<unsigned int> columns) noexcept;
		/// Closes the first gap between two sets of non-contiguous tiles in the given column.
		/// @param column The column in which to close the gap.
		/// @return The new positions of tiles which have been moved.
		std::unordered_set<Core::Math::Vec2u> CloseGap(unsigned int column) noexcept;
		/// Returns whether there are any tiles (non-empty) above the given position (non-inclusive).
		bool AnyTilesAbove(Core::Math::Vec2u position) const noexcept;
		/// Returns the position of the top most tile which is contiguously above the given position.
		Core::Math::Vec2u FindTopmostTile(Core::Math::Vec2u position) const noexcept;
		/// Returns the number of contiguous empty tiles directly above the given position.
		unsigned int CountEmptyTilesAbove(Core::Math::Vec2u position) const noexcept;
		/// Returns the number of non-empty tiles above the given tile, including itself.
		unsigned int CountVerticalTiles(Core::Math::Vec2u position) const noexcept;


    private:
        std::random_device                                          mRandomDevice;
        std::uniform_int_distribution<std::underlying_type_t<Tile>> mTileDistribution{1, 4};


        Tile                           mTiles[BOARD_WIDTH][BOARD_HEIGHT] = {};
        Core::Optional<PlaceableTiles> mCurrentTiles;
        std::deque<Tile>               mTileQueue;
    };
}