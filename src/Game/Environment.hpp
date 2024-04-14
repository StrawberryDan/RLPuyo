#pragma once
//======================================================================================================================
//  Includes
//----------------------------------------------------------------------------------------------------------------------
#include "Board.hpp"
#include "Renderer.hpp"


//======================================================================================================================
//  Class Declaration
//----------------------------------------------------------------------------------------------------------------------
namespace Strawberry::RLPuyo
{
	class Environment
	{
	public:
		Environment(const Window::Window& window);


		void ProcessAction(unsigned int playerIndex, Action action) noexcept;


		void Step() noexcept;


		void Render() noexcept;


	private:
		unsigned int mSkillPoints[2];
		Board        mBoards[2];


		Renderer mRenderer;
	};
}