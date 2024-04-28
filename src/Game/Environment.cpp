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
		: mPreviousReward(0)
	{}


	Environment::Environment(const Window::Window& window)
		: mRenderer(window)
		, mPreviousReward(0)
	{}


	void Environment::ProcessAction(Action action) noexcept
	{
		mBoards.ProcessAction(action);
	}


	void Environment::Step() noexcept
	{
		Core::Optional<unsigned int> pointsEarned = mBoards.Step().Map([](auto& x) { return Board::ChainValue(x); });

		if (GameOver())
		{
			mPreviousReward = 0;
		}
		else
		{
			mPreviousReward = pointsEarned.ValueOr(0);
		}
	}


	void Environment::Render() noexcept
	{
		mRenderer->Submit(0, mBoards);
		mRenderer->Render();
	}


	int Environment::GetReward() const noexcept
	{
		return mPreviousReward;
	}


	bool Environment::GameOver() const noexcept
	{
		return mBoards.HasLost();
	}


	nlohmann::json Environment::StateAsJson() const noexcept
	{
		auto state = nlohmann::json::object();

		state["gameOver"] = GameOver();
		state["players"] = PlayerStateAsJson();

		return state;
	}


	nlohmann::json Environment::PlayerStateAsJson() const noexcept
	{
		auto playerState = nlohmann::json::object();

		playerState["queue"]  = mBoards.QueueAsJson();
		playerState["tiles"]  = mBoards.TilesAsJson();
		playerState["reward"] = GetReward();

		return playerState;
	}
}