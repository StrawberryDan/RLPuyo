//======================================================================================================================
//  Includes
//----------------------------------------------------------------------------------------------------------------------
#include "Board.hpp"
// Standard Library
#include <algorithm>
#include <random>
#include <utility>


//======================================================================================================================
//  Method Definitions
//----------------------------------------------------------------------------------------------------------------------
namespace Strawberry::RLPuyo
{
	void PlaceableTiles::MoveLeft()
	{
		if (mPosition[0] > 0)
		{
			mPosition[0] -= 1;
		}
	}


	void PlaceableTiles::MoveRight()
	{
		if (mPosition[0] < BOARD_WIDTH - 1)
		{
			mPosition[0] += 1;
		}
	}


	void PlaceableTiles::Descend()
	{
		mPosition[1] += 1;
	}


	void PlaceableTiles::Swap()
	{
		std::swap(mTiles[0], mTiles[1]);
	}


	TilePosition PlaceableTiles::Position() const
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
		ReplenishQueue();

		// Assign Random Values to each tile bellow the half way point down the map.
		for (uint8_t y = BOARD_HEIGHT / 2; y < BOARD_HEIGHT; y++)
		{
			for (uint8_t x = 0; x < BOARD_WIDTH; x++)
			{
				mTiles[x][y] = static_cast<Tile>(mTileDistribution(mRandomDevice));

				while (FindConnectedTiles({x, y}).size() > 2)
				{
					mTiles[x][y] = static_cast<Tile>(mTileDistribution(mRandomDevice));
				}
			}
		}
	}


	Core::Optional<Tile> Board::FallingTilesTop() const noexcept
	{
		return mCurrentTiles.Map([](const auto& x) { return x.Top(); });
	}


	Core::Optional<Tile> Board::FallingTilesBottom() const noexcept
	{
		return mCurrentTiles.Map([](const auto& x) { return x.Bottom(); });
	}


	Core::Optional<TilePosition> Board::FallingTilesPosition() const noexcept
	{
		return mCurrentTiles.Map([](const auto& x) { return x.Position(); });
	}


	Tile Board::GetTile(TilePosition position) const noexcept
	{
		return mTiles[position[0]][position[1]];
	}


	void Board::Step()
	{
		// Check if there is a tile currently being placed.
		if (mCurrentTiles)
		{
			auto hasHitBottom = [this]{ return mCurrentTiles->Position()[1] == BOARD_HEIGHT - 2; };
			auto hasHitAnotherTile = [this]{ return mTiles[mCurrentTiles->Position()[0]][mCurrentTiles->Position()[1] + 2] != Tile::EMPTY; };
			// Check if the tile has landed on
			if (hasHitBottom() || hasHitAnotherTile())
			{
				Core::AssertEQ(mTiles[mCurrentTiles->Position()[0]][mCurrentTiles->Position()[1] + 0], Tile::EMPTY);
				mTiles[mCurrentTiles->Position()[0]][mCurrentTiles->Position()[1] + 0] = mCurrentTiles->Top();
				Core::AssertEQ(mTiles[mCurrentTiles->Position()[0]][mCurrentTiles->Position()[1] + 1], Tile::EMPTY);
				mTiles[mCurrentTiles->Position()[0]][mCurrentTiles->Position()[1] + 1] = mCurrentTiles->Bottom();
				Resolve({mCurrentTiles->Position(), mCurrentTiles->Position().Offset(0, 1)});
				mCurrentTiles.Reset();
			}
			else
			{
				// Descend the current placing tile
				mCurrentTiles->Descend();
			}
		}
		else
		{
			Tile top = mTileQueue.front();
			mTileQueue.pop_front();
			Tile bottom = mTileQueue.front();
			mTileQueue.pop_front();
			mCurrentTiles.Emplace(top, bottom);
			ReplenishQueue();
		}
	}


	void Board::ReplenishQueue()
	{
		while (mTileQueue.size() < 8)
		{
			mTileQueue.push_back(static_cast<Tile>(mTileDistribution(mRandomDevice)));
		}
	}


	void Board::Resolve(std::unordered_set<TilePosition> candidates)
	{
		while (!candidates.empty())
		{
			std::set<unsigned int> affectedColumns;
			for (auto candidate: candidates)
			{
				if (GetTile(candidate) != Tile::EMPTY)
				{
					auto group = FindConnectedTiles(candidate);
					if (group.size() >= 3)
					{
						auto columns = EliminateTiles(group);
						for (auto c: columns) affectedColumns.emplace(c);
					}
				}
			}

			candidates = ApplyGravity(affectedColumns);
		}
	}


	std::unordered_set<TilePosition> Board::FindConnectedTiles(TilePosition root)
	{
		std::unordered_set<TilePosition> closed, frontier{root};

		// Expand the frontier until it can no longer be expanded.
		while (!frontier.empty())
		{
			// Expand the frontier;
			std::unordered_set<TilePosition> newFrontier;
			for (auto tile : frontier)
			{
				Core::AssertNEQ(GetTile(tile), Tile::EMPTY);
				// Try to add left tile
				if (!closed.contains(tile.Offset(-1, 0)) && tile[0] > 0 && GetTile(tile.Offset(-1, 0)) == GetTile(tile))
				{
					newFrontier.emplace(tile.Offset(-1, 0));
				}

				// Try to add right tile
				if (!closed.contains(tile.Offset(1, 0)) && tile[0] < BOARD_WIDTH - 1 && GetTile(tile.Offset(1, 0)) == GetTile(tile))
				{
					newFrontier.emplace(tile.Offset(1, 0));
				}

				// Try to add above tile
				if (!closed.contains(tile.Offset(0, -1)) && tile[1] > 0 && GetTile(tile.Offset(0, -1)) == GetTile(tile))
				{
					newFrontier.emplace(tile.Offset(0, -1));
				}

				// Try to add below tile
				if (!closed.contains(tile.Offset(0, 1)) && tile[1] < BOARD_HEIGHT - 1 && GetTile(tile.Offset(0, 1)) == GetTile(tile))
				{
					newFrontier.emplace(tile.Offset(0, 1));
				}
				// Add this tile to the closed set
				closed.emplace(tile);
			}

			frontier = newFrontier;
		}

		return closed;
	}


	std::set<unsigned int> Board::EliminateTiles(std::unordered_set<TilePosition> tiles)
	{
		std::set<unsigned int> columns;

		for (auto tile : tiles)
		{
			// Remove tile.
			mTiles[tile[0]][tile[1]] = Tile::EMPTY;
			// Add tile to column list.
			columns.emplace(tile[0]);
		}

		return columns;
	}


	std::unordered_set<TilePosition> Board::ApplyGravity(std::set<unsigned int> columns) noexcept
	{
		std::unordered_set<TilePosition> affectedTiles;

		for (unsigned int column : columns)
		{
			auto affectedTilesInColumn = CloseGap(column);
			for (auto tile : affectedTilesInColumn)
			{
				affectedTiles.emplace(tile);
			}
		}

		return affectedTiles;
	}


	std::unordered_set<TilePosition> Board::CloseGap(unsigned int column) noexcept
	{
		TilePosition base;
		if (GetTile({column, BOARD_HEIGHT - 1}) == Tile::EMPTY)
		{
			base = TilePosition(column, BOARD_HEIGHT);
		}
		else
		{
			base = FindTopmostTile({column, BOARD_HEIGHT - 1});
		}

		std::unordered_set<TilePosition> affectedTiles;
		while (AnyTilesAbove(base))
		{
			auto gapSize = CountEmptyTilesAbove(base);
			auto chunkSize = CountVerticalTiles(base.Offset(0, -gapSize - 1));

			for (int i = 0; i < chunkSize; i++)
			{
				auto srcRow = base[1] - i - gapSize - 1;
				auto dstRow = base[1] - i - 1;
				mTiles[column][dstRow] = std::exchange(mTiles[column][srcRow], Tile::EMPTY);
				affectedTiles.emplace(column, dstRow);
			}

			base = FindTopmostTile(base);
		}

		return affectedTiles;
	}


	bool Board::AnyTilesAbove(TilePosition position) const noexcept
	{
		if (position[1] == 0) return false;

		position[1] -= 1;
		while (true)
		{
			if (GetTile(position) != Tile::EMPTY) return true;
			if (position[1] == 0) return false;
			position[1] -= 1;
		}
	}


	TilePosition Board::FindTopmostTile(TilePosition position) const noexcept
	{
		Core::AssertNEQ(GetTile(position), Tile::EMPTY);

		while (GetTile(position) != Tile::EMPTY)
		{
			if (position[1] == 0) return position;
			position[1] -= 1;
		}
		return position.Offset(0, 1);
	}


	unsigned int Board::CountEmptyTilesAbove(TilePosition position) const noexcept
	{
		if (position[1] == 0) return 0;
		position[1] -= 1;

		unsigned int emptyTiles = 0;
		while (GetTile(position) == Tile::EMPTY)
		{
			emptyTiles += 1;
			if (position[1] == 0) break;
			position[1] -= 1;
		}

		return emptyTiles;
	}


	unsigned int Board::CountVerticalTiles(TilePosition position) const noexcept
	{
		Core::AssertNEQ(GetTile(position), Tile::EMPTY);

		unsigned int count = 0;
		while (GetTile(position) != Tile::EMPTY)
		{
			if (position[1] == 0) return count;
			position[1] -= 1;
			count += 1;
		}
		return count;
	}
}
