//======================================================================================================================
//  Includes
//----------------------------------------------------------------------------------------------------------------------
#include "Board.hpp"
// Standard Library
#include <random>

//======================================================================================================================
//  Method Definitions
//----------------------------------------------------------------------------------------------------------------------
namespace Strawberry::RLPuyo
{
	void PlaceableTiles::Descend()
	{
		mPosition[1] += 1;
	}


	void PlaceableTiles::Swap()
	{
		std::swap(mTiles[0], mTiles[1]);
	}


	Core::Math::Vec2u PlaceableTiles::Position() const
	{
		return mPosition;
	}


	Tile PlaceableTiles::Top() const
	{
		return mTiles[0];
	}


	Tile PlaceableTiles::Bottom() const
	{
		return mTiles[1];
	}


	Board::Board()
	{
		// Populate Tile Queue
		while (mTileQueue.size() < 8)
		{
			mTileQueue.push_back(static_cast<Tile>(mTileDistribution(mRandomDevice)));
		}

		// Assign Random Values to each tile bellow the half way point down the map.
		for (uint8_t y = BOARD_HEIGHT / 2; y < BOARD_HEIGHT; y++)
		{
			for (uint8_t x = 0; x < BOARD_WIDTH; x++)
			{
				mTiles[x][y] = static_cast<Tile>(mTileDistribution(mRandomDevice));
			}
		}
	}


	void Board::Step()
	{
		// Check if there is a tile currently being placed.
		if (mCurrentTiles)
		{
			// Descend the current placing tile
			mCurrentTiles->Descend();
			auto hasHitBottom = [this]{ return mCurrentTiles->Position()[1] == 1; };
			auto hasHitAnotherTile = [this]{ return mTiles[mCurrentTiles->Position()[0]][mCurrentTiles->Position()[1] + 2] != Tile::EMPTY; };
			// Check if the tile has landed on
			if (hasHitBottom() || hasHitAnotherTile())
			{
				Core::AssertEQ(mTiles[mCurrentTiles->Position()[0]][mCurrentTiles->Position()[1] + 0], Tile::EMPTY);
				mTiles[mCurrentTiles->Position()[0]][mCurrentTiles->Position()[1] + 0] = mCurrentTiles->Top();
				Core::AssertEQ(mTiles[mCurrentTiles->Position()[0]][mCurrentTiles->Position()[1] + 1], Tile::EMPTY);
				mTiles[mCurrentTiles->Position()[0]][mCurrentTiles->Position()[1] + 1] = mCurrentTiles->Bottom();
				mCurrentTiles.Reset();
			}
		}
		else
		{
			Tile top = mTileQueue.front();
			mTileQueue.pop_front();
			Tile bottom = mTileQueue.front();
			mTileQueue.pop_front();
			mCurrentTiles.Emplace(top, bottom);

			while (mTileQueue.size() < 8)
			{
				mTileQueue.push_back(static_cast<Tile>(mTileDistribution(mRandomDevice)));
			}
		}
	}
}
