//======================================================================================================================
//  Includes
//----------------------------------------------------------------------------------------------------------------------
#include "Environment.hpp"
#include "Strawberry/Window/Window.hpp"

//======================================================================================================================
//  Method Definitions
//----------------------------------------------------------------------------------------------------------------------
namespace Strawberry::RLPuyo
{
	Environment::Environment()
		: mSkillPoints{0, 0}
	{}


	Environment::Environment(const Window::Window& window)
		: mRenderer(window)
		, mSkillPoints{0, 0}
	{}


	void Environment::ProcessAction(unsigned int playerIndex, Action action) noexcept
	{
		switch (action)
		{
			case Action::OffensiveSkill:
				if (mSkillPoints[playerIndex] >= 100)
					mBoards[(playerIndex + 1) % 2].ProcessOffensiveSkill(std::exchange(mSkillPoints[playerIndex], 0));
				break;
			case Action::DefensiveSkill:
				if (mSkillPoints[playerIndex] >= 100)
					mBoards[playerIndex].ProcessDefenseSkill(std::exchange(mSkillPoints[playerIndex], 0));
				break;
			default:
				mBoards[playerIndex].ProcessAction(action);
		}
	}


	void Environment::Step() noexcept
	{
		unsigned int pointsEarned[] = {
			mBoards[0].Step().Map([](auto& x) { return Board::ChainValue(x); }).UnwrapOr(0),
			mBoards[1].Step().Map([](auto& x) { return Board::ChainValue(x); }).UnwrapOr(0)
		};

		mSkillPoints[0] += pointsEarned[0];
		mSkillPoints[1] += pointsEarned[1];

		if (GameOver())
		{
			if (mBoards[0].HasLost())
			{
				mPreviousRewards[0] = -1000;
				mPreviousRewards[1] = 1000;
			}
			else if (mBoards[1].HasLost())
			{
				mPreviousRewards[0] = 1000;
				mPreviousRewards[1] = -1000;
			}
			else
			{
				Core::Unreachable();
			}
		}
		else
		{
			mPreviousRewards[0] = pointsEarned[0] == 0 ? -1 : static_cast<int>(pointsEarned[0]);
			mPreviousRewards[1] = pointsEarned[1] == 0 ? -1 : static_cast<int>(pointsEarned[1]);
		}
	}


	void Environment::Render() noexcept
	{
		mRenderer->Submit(0, mBoards[0]);
		mRenderer->Submit(1, mBoards[1]);
		mRenderer->Render();
	}


	std::tuple<int, int> Environment::GetRewards() const noexcept
	{
		return {mPreviousRewards[0], mPreviousRewards[1]};
	}


	bool Environment::GameOver() const noexcept
	{
		return mBoards[0].HasLost() || mBoards[1].HasLost();
	}


	nlohmann::json Environment::StateAsJson() const noexcept
	{
		auto state = nlohmann::json::object();

		state["gameOver"] = GameOver();
		auto players = nlohmann::json::array();
		players[0] = PlayerStateAsJson(0);
		players[1] = PlayerStateAsJson(1);
		state["players"] = players;

		return state;
	}


	nlohmann::json Environment::PlayerStateAsJson(unsigned int playerIndex) const noexcept
	{
		auto playerState = nlohmann::json::object();

		playerState["ap"]     = mSkillPoints[playerIndex];
		playerState["queue"]  = mBoards[playerIndex].QueueAsJson();
		playerState["tiles"]  = mBoards[playerIndex].TilesAsJson();
		playerState["reward"] = mPreviousRewards[playerIndex];

		return playerState;
	}
}