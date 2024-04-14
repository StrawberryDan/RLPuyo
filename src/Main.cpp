#include <iostream>
#include <Strawberry/Window/Window.hpp>
#include "Game/Board.hpp"
#include "Game/Renderer.hpp"
#include "Strawberry/Net/Socket/TCPListener.hpp"
#include "Strawberry/Net/Socket/API.hpp"
#include "Strawberry/Core/IO/Logging.hpp"


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
				if (auto action = ActionFromEvent(*event))
				{
					board.ProcessAction(action.Unwrap());
				}
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
#else
	Net::Socket::API::Initialise();
	Net::Socket::TCPListener listener = Net::Socket::TCPListener::Bind(Net::Endpoint(Net::IPv4Address(127, 0, 0, 1), 25500)).Unwrap();
	auto client = listener.Accept().Unwrap();
	Core::Logging::Info("Client Connected Successfully");

	auto message = client.Read(1024);
	std::cout << message->AsString() << std::endl;

	Net::Socket::API::Terminate();
#endif

	return 0;
}
