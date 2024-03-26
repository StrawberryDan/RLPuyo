#include <iostream>
#include <Strawberry/Window/Window.hpp>
#include "Game/Board.hpp"
#include "Game/Renderer.hpp"


int main()
{
	using namespace Strawberry;
	using namespace RLPuyo;

	Strawberry::RLPuyo::Board board;


	Window::Window window("RLPuyo", {480, 360});
	Renderer renderer(window);

	while (!window.CloseRequested())
	{
		Window::PollInput();

		renderer.Render();

		window.SwapBuffers();
	}

	return 0;
}
