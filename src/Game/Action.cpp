//======================================================================================================================
//  Includes
//----------------------------------------------------------------------------------------------------------------------
#include "Action.hpp"


//======================================================================================================================
//  Method Definitions
//----------------------------------------------------------------------------------------------------------------------
namespace Strawberry::RLPuyo
{
	Core::Optional<Action> ActionFromEvent(Window::Event& event)
	{
		if (auto key = event.Value<Window::Events::Key>())
		{
			if (key->action == Window::Input::KeyAction::Release)
			{
				switch (key->keyCode)
				{
					case Window::Input::KeyCode::LEFT:
						return Action::Left;
					case Window::Input::KeyCode::RIGHT:
						return Action::Right;
					case Window::Input::KeyCode::SPACE:
						return Action::Swap;
					default:
						break;
				}
			}
		}

		return Core::NullOpt;
	}
}