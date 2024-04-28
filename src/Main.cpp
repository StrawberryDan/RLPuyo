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


constexpr double UPDATE_INTERVAL = RLPUYO_REALTIME ? 0.2 : 0.001;
constexpr float  RENDER_INTERVAL = 0.0;


Action ActionsFromMessage(const Core::IO::DynamicByteBuffer& bytes)
{
	return static_cast<Action>(bytes[0]);
}


void HandleConnection(Net::Socket::TCPSocket client)
{
	Window::Window window("RLPuyo", {2 * 480, 2 * 360});

	Core::Optional<Environment> environment(window);
	Core::Clock mUpdateTimer;
	Core::Clock mRenderTimer;
	while (true)
	{
		Window::PollInput();

		if (mUpdateTimer.Read() > UPDATE_INTERVAL)
		{
			auto state = environment->StateAsJson().dump();
			auto writeResult = client.Write(state);
			if (writeResult.IsErr())
				switch (writeResult.Err())
				{
					case Core::IO::Error::Closed:
						return;
					default:
						Core::Unreachable();
				}

			if (environment->GameOver())
			{
				environment.Emplace(window);
				continue;
			}

			auto actionMessage = client.ReadAll(1);
			if (actionMessage.IsErr())
				switch (actionMessage.Err())
				{
					case Core::IO::Error::Closed:
						return;
					default:
						Core::Unreachable();
				}

			Action chosenAction = ActionsFromMessage(actionMessage.Unwrap());
			environment->ProcessAction(chosenAction);
			environment->Step();
			mUpdateTimer.Restart();
		}

		if (mRenderTimer.Read() > (RENDER_INTERVAL <= 0.0 ? UPDATE_INTERVAL : RENDER_INTERVAL))
		{
			environment->Render();
			mRenderTimer.Restart();
			window.SwapBuffers();
		}
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
		Net::Socket::TCPListener listener = Net::Socket::TCPListener::Bind(Net::Endpoint(Net::IPv4Address(127, 0, 0, 1), 25500)).Unwrap();
		auto client = listener.Accept().Unwrap();
		Core::Logging::Info("Client Connected Successfully");

		HandleConnection(std::move(client));
	}

	Net::Socket::API::Terminate();
#endif

	return 0;
}
