#include <iostream>
#include <Strawberry/Window/Window.hpp>
#include "Game/Board.hpp"
#include "Game/Renderer.hpp"
#include "Strawberry/Net/Socket/TCPListener.hpp"


constexpr double UPDATE_INTERVAL = 0.2;


int main()
{
	using namespace Strawberry;
	using namespace RLPuyo;

#if RLPUYO_REALTIME
	Strawberry::RLPuyo::Board board;


	Window::Window window("RLPuyo", {2 * 480, 2 * 360});
	Renderer renderer(window);


	Core::Clock updateTimer(true);
	while (!window.CloseRequested())
	{
		Window::PollInput();


		if (window.HasFocus())
		{
			while (auto event = window.NextEvent())
			{
				board.ProcessEvent(event.Unwrap());
			}

			if (updateTimer.Read() >= UPDATE_INTERVAL)
			{
				board.Step();
				updateTimer.Restart();
			}
		}

		renderer.Submit(board);
		renderer.Render();
		window.SwapBuffers();
	}
#endif

	return 0;
}
