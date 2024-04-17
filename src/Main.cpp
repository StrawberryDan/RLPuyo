// RLPuyo
#include "Game/Board.hpp"
#include "Game/Environment.hpp"
#include "Game/Renderer.hpp"
// Strawberry Window
#include "Strawberry/Window/Window.hpp"
// Strawberry Net
#include "Strawberry/Core/IO/Logging.hpp"
#include "Strawberry/Net/Socket/API.hpp"
#include "Strawberry/Net/Socket/TCPListener.hpp"


using namespace Strawberry;
using namespace RLPuyo;


constexpr double UPDATE_INTERVAL = 0.2;


std::vector<Action> ActionsFromMessage(const Core::IO::DynamicByteBuffer& bytes)
{
	return {
		static_cast<Action>(bytes[0]),
		static_cast<Action>(bytes[1]),
	};
}


int main()
{
#if RLPUYO_REALTIME
	Window::Window window("RLPuyo", {2 * 480, 2 * 360});
	Environment environment(window);


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
					environment.ProcessAction(0, action.Unwrap());
				}
			}

			if (updateTimer.Read() >= UPDATE_INTERVAL)
			{
				environment.Step();
				updateTimer.Restart();
			}
		}

		environment.Render();
		window.SwapBuffers();
	}
#else
	Net::Socket::API::Initialise();
	Net::Socket::TCPListener listener = Net::Socket::TCPListener::Bind(Net::Endpoint(Net::IPv4Address(127, 0, 0, 1), 25500)).Unwrap();
	auto client = listener.Accept().Unwrap();
	Core::Logging::Info("Client Connected Successfully");

	Environment environment;
	while (true)
	{
		auto state = environment.StateAsJson().dump();
		client.Write(state).Unwrap();

		auto actionMessage = client.ReadAll(2).Unwrap();
		std::vector<Action> chosenActions = ActionsFromMessage(actionMessage);

		if (environment.GameOver())
		{
			environment = Environment();
		}

		environment.ProcessAction(0, chosenActions[0]);
		environment.ProcessAction(1, chosenActions[1]);

		environment.Step();
	}

	Net::Socket::API::Terminate();
#endif

	return 0;
}
