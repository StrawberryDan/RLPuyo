#include <iostream>
#include <Strawberry/Window/Window.hpp>
#include "Game/Board.hpp"
#include "Game/Renderer.hpp"


constexpr double UPDATE_INTERVAL = 0.5;


int main()
{
	using namespace Strawberry;
	using namespace RLPuyo;

	Strawberry::RLPuyo::Board board;


	Window::Window window("RLPuyo", {4 * 480, 4 * 360});
	Renderer renderer(window);

	Core::Clock mUpdateTimer(true);
	while (!window.CloseRequested())
	{
		Window::PollInput();

		if (mUpdateTimer.Read() >= UPDATE_INTERVAL)
		{
			board.Step();
			mUpdateTimer.Restart();
		}

		renderer.Submit(board);
		renderer.Render();

		window.SwapBuffers();
	}

	return 0;
}
