//======================================================================================================================
//  Includes
//----------------------------------------------------------------------------------------------------------------------
#include "Environment.hpp"


//======================================================================================================================
//  Method Definitions
//----------------------------------------------------------------------------------------------------------------------
namespace Strawberry::RLPuyo
{
	Environment::Environment(const Window::Window& window)
		: mRenderer(window)
	{}


	void Environment::ProcessAction(unsigned int playerIndex, Action action) noexcept
	{
		switch (action)
		{
			default:
				mBoards[playerIndex].ProcessAction(action);
		}
	}


	void Environment::Step() noexcept
	{
		mSkillPoints[0] += mBoards[0].Step().Map([](auto& x) { return Board::ChainValue(x); }).UnwrapOr(0);
		mSkillPoints[1] += mBoards[1].Step().Map([](auto& x) { return Board::ChainValue(x); }).UnwrapOr(0);
	}


	void Environment::Render() noexcept
	{
		mRenderer.Submit(0, mBoards[0]);
		mRenderer.Submit(1, mBoards[1]);
		mRenderer.Render();
	}


	nlohmann::json Environment::StateAsJson() const noexcept
	{
		auto state = nlohmann::json::object();

		auto players = nlohmann::json::array();
		players[0] = PlayerStateAsJson(0);
		players[1] = PlayerStateAsJson(1);
		state["players"] = players;

		return state;
	}


	nlohmann::json Environment::PlayerStateAsJson(unsigned int playerIndex) const noexcept
	{
		auto playerState = nlohmann::json::object();

		playerState["ap"]    = mSkillPoints[playerIndex];
		playerState["queue"] = mBoards[playerIndex].QueueAsJson();
		playerState["tiles"] = mBoards[playerIndex].TilesAsJson();

		return playerState;
	}
}