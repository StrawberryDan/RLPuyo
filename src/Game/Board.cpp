//======================================================================================================================
//  Includes
//----------------------------------------------------------------------------------------------------------------------
#include "Board.hpp"
#include "Action.hpp"
// Standard Library
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
		: mRandomDevice(std::random_device()())
	{
		// Populate Tile Queue
		ReplenishQueue();
		PullTilesFromQueue();

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


	void Board::ProcessAction(Action action)
	{
		switch (action)
		{
			case Action::None:
				break;
			case Action::Left:
				if (mCurrentTiles &&
					mCurrentTiles->Position()[0] > 0 &&
					GetTile(mCurrentTiles->Position().Offset(-1, 0)) == Tile::EMPTY &&
					GetTile(mCurrentTiles->Position().Offset(-1, 1)) == Tile::EMPTY)
				{
					mCurrentTiles->MoveLeft();
				}
				break;
			case Action::Right:
				if (mCurrentTiles &&
					mCurrentTiles->Position()[0] <= BOARD_WIDTH - 1 &&
					GetTile(mCurrentTiles->Position().Offset(1, 0)) == Tile::EMPTY &&
					GetTile(mCurrentTiles->Position().Offset(1, 1)) == Tile::EMPTY)
				{
					mCurrentTiles->MoveRight();
				}
				break;
			case Action::Swap:
				if (mCurrentTiles)
				{
					mCurrentTiles->Swap();
				}
				break;
			default:
				Core::Unreachable();
		}
	}


	void Board::ProcessDefenseSkill(unsigned int points)
	{
		unsigned int rows = points / 100;


		for (int y = BOARD_HEIGHT - 1; y >= 0 && rows > 0; --y, --rows)
		{
			for (auto x = 0; x < BOARD_WIDTH; x++)
			{
				SetTile({x, y}, Tile::EMPTY);
			}
		}


		std::set<unsigned int> columns;
		for (int i = 0; i < BOARD_WIDTH; ++i) columns.emplace(i);
		ApplyGravity(columns);
	}


	void Board::ProcessOffensiveSkill(unsigned int points)
	{
		unsigned int rows = points / 100;

		for (int i = 0; i < rows; ++i)
		{
			for (int column = 0; column < BOARD_WIDTH; ++column)
			{
				ShiftColumnUp(column);

				do
				{
					SetTile({column, BOARD_HEIGHT - 1}, static_cast<Tile>(mTileDistribution(mRandomDevice)));
				}
				while (FindConnectedTiles({column, BOARD_HEIGHT - 1}).size() > 2);
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
		if (position[1] < 0)
		{
			const auto& column = mAboveBoardTiles[position[0]];
			const int index = static_cast<int>(column.size()) + position[1];
			if (index < 0) return Tile::EMPTY;
			else return column[index];
		}
		else
		{
			return mTiles[position[0]][position[1]];
		}
	}


	Core::Optional<Chain> Board::Step()
	{
		Core::Assert(mCurrentTiles.HasValue());


		auto hasHitBottom = [this]{ return mCurrentTiles->Position()[1] == BOARD_HEIGHT - 2; };
		auto hasHitAnotherTile = [this]{ return mTiles[mCurrentTiles->Position()[0]][mCurrentTiles->Position()[1] + 2] != Tile::EMPTY; };
		// Check if the tile has landed on
		if (hasHitBottom() || hasHitAnotherTile())
		{
			// Place the current tiles in place
			Core::AssertEQ(GetTile(mCurrentTiles->Position().Offset(0, 0)), Tile::EMPTY);
			SetTile(mCurrentTiles->Position().Offset(0, 0), mCurrentTiles->Top());
			Core::AssertEQ(GetTile(mCurrentTiles->Position().Offset(0, 1)), Tile::EMPTY);
			SetTile(mCurrentTiles->Position().Offset(0, 1), mCurrentTiles->Bottom());

			// Resolve the board state until it is stable;
			auto chain = Resolve({mCurrentTiles->Position(), mCurrentTiles->Position().Offset(0, 1)});

			// Grab fresh tiles off the queue
			PullTilesFromQueue();

			// Return the chain
			return chain;
		}
		else
		{
			// Descend the current placing tile
			mCurrentTiles->Descend();
		}

		return Core::NullOpt;
	}


	bool Board::HasLost() const noexcept
	{
		return GetTile({(BOARD_WIDTH + 1) / 2, -2}) != Tile::EMPTY || GetTile({(BOARD_WIDTH + 1) / 2, -1}) != Tile::EMPTY;
	}


	void Board::PullTilesFromQueue()
	{
		Tile top = mTileQueue.front();
		mTileQueue.pop_front();
		Tile bottom = mTileQueue.front();
		mTileQueue.pop_front();
		mCurrentTiles.Emplace(top, bottom);
		ReplenishQueue();
	}


	nlohmann::json Board::QueueAsJson() const noexcept
	{
		Core::AssertEQ(mTileQueue.size(), 8);


		auto queue = nlohmann::json::array();
		for (auto tile : mTileQueue)
		{
			queue.push_back(tile);
		}
		return queue;
	}


	nlohmann::json Board::TilesAsJson() const noexcept
	{
		auto tiles = nlohmann::json::object();

		auto falling      = nlohmann::json::object();
		falling["x"]      = mCurrentTiles->Position()[0];
		falling["y"]      = mCurrentTiles->Position()[1];
		falling["top"]    = FallingTilesTop().Unwrap();
		falling["bottom"] = FallingTilesBottom().Unwrap();
		tiles["falling"]  = falling;


		auto board = nlohmann::json::array();
		for (int x = 0; x < BOARD_WIDTH; ++x)
		{
			auto column = nlohmann::json::array();
			for (int y = 0; y < BOARD_HEIGHT; ++y)
			{
				column.push_back(GetTile({x, y}));
			}
			board.push_back(column);
		}
		tiles["board"] = board;


		return tiles;
	}


	unsigned int Board::ChainValue(const Chain& chain)
	{
		unsigned int value = 0;

		for (int i = 0; i < chain.size(); ++i)
		{
			value += (i + 1) * chain[i];
		}

		return value;
	}


	void Board::SetTile(TilePosition position, Tile tile)
	{
		if (position[1] < 0)
		{
			auto& column = mAboveBoardTiles[position[0]];
			// Make sure that the column vector is big enough
			while (column.size() < -position[1])
			{
				column.insert(column.begin(), Tile::EMPTY);
			}

			auto index = static_cast<int>(column.size()) + position[1];
			column[index] = tile;
		}
		else
		{
			mTiles[position[0]][position[1]] = tile;
		}
	}


	void Board::ReplenishQueue()
	{
		while (mTileQueue.size() < 8)
		{
			mTileQueue.push_back(static_cast<Tile>(mTileDistribution(mRandomDevice)));
		}
	}


	Chain Board::Resolve(std::unordered_set<TilePosition> candidates)
	{
#if STRAWBERRY_DEBUG
		Tile originalState[BOARD_WIDTH][BOARD_HEIGHT];
		for (int x = 0; x < BOARD_WIDTH; ++x)
		{
			for (int y = 0; y < BOARD_HEIGHT; ++y)
			{
				originalState[x][y] = mTiles[x][y];
			}
		}
#endif


		Chain chain;
		while (!candidates.empty())
		{
			chain.emplace_back(0);
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
						chain.back() += group.size();
					}
				}
			}

			candidates = ApplyGravity(affectedColumns);
		}
		return chain;
	}


	std::unordered_set<TilePosition> Board::FindConnectedTiles(TilePosition root) const
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


	std::set<unsigned int> Board::EliminateTiles(const std::unordered_set<TilePosition>& tiles)
	{
		std::set<unsigned int> columns;

		for (auto tile : tiles)
		{
			// Remove tile.
			SetTile(tile, Tile::EMPTY);
			// Add tile to column list.
			columns.emplace(tile[0]);
		}

		return columns;
	}


	std::unordered_set<TilePosition> Board::ApplyGravity(const std::set<unsigned int>& columns) noexcept
	{
		std::unordered_set<TilePosition> affectedTiles;
		for (unsigned int column : columns)
		{
			auto affectedInColumn = CloseGaps(column);
			for (auto tile : affectedInColumn) affectedTiles.emplace(tile);
		}
		return affectedTiles;
	}


	std::unordered_set<TilePosition> Board::CloseGaps(unsigned int column) noexcept
	{
		std::vector<Tile> tiles;
		Core::Optional<int> stableRow;

		int aboveBoardColSize = static_cast<int>(mAboveBoardTiles[column].size());
		for (int y = BOARD_HEIGHT - 1; y >= -aboveBoardColSize; --y)
		{
			Tile tile = GetTile({column, y});
			if (tile != Tile::EMPTY)
			{
				tiles.emplace_back(tile);
				SetTile({column, y}, Tile::EMPTY);
			}
			else if (y > 0 && tile == Tile::EMPTY && !stableRow)
			{
				stableRow = y + 1;
			}
		}


		std::unordered_set<TilePosition> affectedTiles;
		for (int i = 0; i < tiles.size(); ++i)
		{
			int row = BOARD_HEIGHT - 1 - i;
			SetTile({column, row}, tiles[i]);
			if (!stableRow || (stableRow && row < *stableRow))
			{
				affectedTiles.emplace(column, row);
			}
		}

		return affectedTiles;
	}


	void Board::ShiftColumnUp(unsigned int column)
	{
		// Make sure that there is space to move tiles up
		if (mAboveBoardTiles[column].empty() || mAboveBoardTiles[column][0] != Tile::EMPTY)
		{
			mAboveBoardTiles[column].insert(mAboveBoardTiles[column].begin(), Tile::EMPTY);
		}

		for (int y = -static_cast<int>(mAboveBoardTiles[column].size()); y < BOARD_HEIGHT - 1; ++y)
		{
			SetTile({column, y}, GetTile({column, y + 1}));
		}

		SetTile({column, BOARD_HEIGHT - 1}, Tile::EMPTY);
	}
}
