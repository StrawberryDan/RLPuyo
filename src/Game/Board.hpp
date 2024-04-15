#pragma once
//======================================================================================================================
//  Includes
//----------------------------------------------------------------------------------------------------------------------
// Strawberry Window
#include "Strawberry/Window/Event.hpp"
// Strawberry Core
#include "Strawberry/Core/Math/Vector.hpp"
#include "Strawberry/Core/Types/Optional.hpp"
#include "Action.hpp"
// Nlohmann
#include "nlohmann/json.hpp"
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


	using Chain = std::vector<unsigned int>;


	using TilePosition = Core::Math::Vec2i;


    class PlaceableTiles
    {
    public:
        PlaceableTiles(Tile first, Tile second)
            : mPosition((BOARD_WIDTH + 1) / 2, -2)
            , mTiles{first, second}
        {}


        void MoveLeft();
        void MoveRight();
        void Descend();
        void Swap();


        TilePosition Position() const;
        Tile         Top() const;
        Tile         Bottom() const;


    private:
        TilePosition mPosition;
        Tile         mTiles[2];
    };


    class Board
    {
    public:
        Board();


		void ProcessAction(Action action);


		Core::Optional<Tile> FallingTilesTop() const noexcept;
		Core::Optional<Tile> FallingTilesBottom() const noexcept;
		Core::Optional<TilePosition> FallingTilesPosition() const noexcept;
		Tile GetTile(TilePosition position) const noexcept;
		Core::Optional<Chain> Step();


		nlohmann::json QueueAsJson() const noexcept;
		nlohmann::json TilesAsJson() const noexcept;


		static unsigned int ChainValue(const Chain& chain);


    protected:
		/// Sets the tile at the given position to the given value.
		void SetTile(TilePosition position, Tile tile);
		/// Refill the queue with random tiles until it is full again.
        void ReplenishQueue();
		/// Initialises new falling tiles from the tile queue
		void PullTilesFromQueue();
		/// Resolve the board after tiles have been placed.
		/// @param candidates The set of tiles which may have formed groups.
		Chain Resolve(std::unordered_set<TilePosition> candidates);
		/// Finds tiles of the same type which are adjacent to the root tile provided.
		/// @param root the cell from which to search.
		/// @return The positions of all the tiles connected to the root (inclusive).
		std::unordered_set<TilePosition> FindConnectedTiles(TilePosition root) const;
		/// Removes the given tiles and returns the columns to which gravity need be applied.
		/// @param The tiles to eliminate.
		/// @return The set of columns to which gravity need be applied.
		std::set<unsigned int> EliminateTiles(const std::unordered_set<TilePosition>& tiles);
		/// Applys gravity to the provided columns.
		/// @param columns The columns to which to apply gravity.
		/// @return The new positions of the tiles which descended due to gravity.
		std::unordered_set<TilePosition> ApplyGravity(const std::set<unsigned int>& columns) noexcept;
		/// Closes the first gap between two sets of non-contiguous tiles in the given column.
		/// @param column The column in which to close the gap.
		/// @return The new positions of tiles which have been moved.
		std::unordered_set<TilePosition> CloseGaps(unsigned int column) noexcept;


    private:
        std::random_device                                          mRandomDevice;
        std::uniform_int_distribution<std::underlying_type_t<Tile>> mTileDistribution{1, 4};


        Tile                           mTiles[BOARD_WIDTH][BOARD_HEIGHT] = {};
        Core::Optional<PlaceableTiles> mCurrentTiles;
        std::deque<Tile>               mTileQueue;
	};
}