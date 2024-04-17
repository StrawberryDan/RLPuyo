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


		void ProcessAction(unsigned int playerIndex, Action action) noexcept;


		void Step() noexcept;


		void Render() noexcept;


		bool GameOver() const noexcept;


		nlohmann::json StateAsJson() const noexcept;
		nlohmann::json PlayerStateAsJson(unsigned int playerIndex) const noexcept;


	private:
		unsigned int mSkillPoints[2];
		Board        mBoards[2];


		Core::Optional<Renderer> mRenderer;
	};
}