#pragma once
//======================================================================================================================
//  Includes
//----------------------------------------------------------------------------------------------------------------------
// RLPuyo
#include "Board.hpp"
#include "Renderer.hpp"
// Nlohmann
#include "nlohmann/json.hpp"


//======================================================================================================================
//  Class Declaration
//----------------------------------------------------------------------------------------------------------------------
namespace Strawberry::RLPuyo
{
	class Environment
	{
	public:
		Environment();
		Environment(const Window::Window& window);


		void ProcessAction(Action action) noexcept;


		void Step() noexcept;


		void Render() noexcept;


		int GetReward() const noexcept;
		bool GameOver() const noexcept;


		nlohmann::json StateAsJson() const noexcept;
		nlohmann::json PlayerStateAsJson() const noexcept;


	private:
		int          mPreviousReward;
		Board        mBoards;


		Core::Optional<Renderer> mRenderer;
	};
}