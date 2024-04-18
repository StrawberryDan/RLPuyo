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


void HandleConnection()
{
	Net::Socket::TCPListener listener = Net::Socket::TCPListener::Bind(Net::Endpoint(Net::IPv4Address(127, 0, 0, 1), 25500)).Unwrap();
	auto client = listener.Accept().Unwrap();
	Core::Logging::Info("Client Connected Successfully");

	Environment environment;
	while (true)
	{
		auto state = environment.StateAsJson().dump();
		auto writeResult = client.Write(state);
		if (writeResult.IsErr()) switch (writeResult.Err())
		{
			case Core::IO::Error::Closed:
				return;
			default:
				Core::Unreachable();
		}

		if (environment.GameOver())
		{
			environment = Environment();
			continue;
		}

		auto actionMessage = client.ReadAll(2);
		if (actionMessage.IsErr()) switch (actionMessage.Err())
			{
				case Core::IO::Error::Closed:
					return;
				default:
					Core::Unreachable();
			}

		std::vector<Action> chosenActions = ActionsFromMessage(actionMessage.Unwrap());


		environment.ProcessAction(0, chosenActions[0]);
		environment.ProcessAction(1, chosenActions[1]);

		environment.Step();
	}
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

	while (true)
	{
		HandleConnection();
	}

	Net::Socket::API::Terminate();
#endif

	return 0;
}
