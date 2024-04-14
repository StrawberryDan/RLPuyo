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
		mBoards[0].Step();
		mBoards[1].Step();
	}


	void Environment::Render() noexcept
	{
		mRenderer.Submit(0, mBoards[0]);
		mRenderer.Submit(1, mBoards[1]);
		mRenderer.Render();
	}
}