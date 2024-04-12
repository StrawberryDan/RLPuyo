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


	Window::Window window("RLPuyo", {2 * 480, 2 * 360});
	Renderer renderer(window);

#if RLPUYO_REALTIME
	Core::Clock mUpdateTimer(true);
#endif

	while (!window.CloseRequested())
	{
		Window::PollInput();

		while (auto event = window.NextEvent())
		{
			board.ProcessEvent(event.Unwrap());
		}

#if RLPUYO_REALTIME
		if (mUpdateTimer.Read() >= UPDATE_INTERVAL)
		{
			board.Step();
			mUpdateTimer.Restart();
		}
#else
		board.Step();
#endif

		renderer.Submit(board);
		renderer.Render();

		window.SwapBuffers();
	}

	return 0;
}
