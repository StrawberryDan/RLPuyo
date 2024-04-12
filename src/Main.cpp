#include <iostream>
#include <Strawberry/Window/Window.hpp>
#include "Game/Board.hpp"
#include "Game/Renderer.hpp"


constexpr double UPDATE_INTERVAL = 0.2;


int main()
{
	using namespace Strawberry;
	using namespace RLPuyo;

	Strawberry::RLPuyo::Board board;


	Window::Window window("RLPuyo", {2 * 480, 2 * 360});
	Renderer renderer(window);

#if RLPUYO_REALTIME
	Core::Clock updateTimer(true);
#endif

	Core::Clock frameTimer;
	while (!window.CloseRequested())
	{
		frameTimer.Restart();
		Window::PollInput();

		while (auto event = window.NextEvent())
		{
			board.ProcessEvent(event.Unwrap());
		}

#if RLPUYO_REALTIME
		if (updateTimer.Read() >= UPDATE_INTERVAL)
		{
			board.Step();
			updateTimer.Restart();
		}
#else
		board.Step();
#endif

		renderer.Submit(board);
		renderer.Render();
		window.SwapBuffers();

		std::cout << "FPS: " << 1.0 / frameTimer.Read() << std::endl;
		frameTimer.Restart();
	}

	return 0;
}
