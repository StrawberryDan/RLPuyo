#pragma once
//======================================================================================================================
//  Includes
//----------------------------------------------------------------------------------------------------------------------
#include "Strawberry/Window/Event.hpp"

//======================================================================================================================
//  Class Declaration
//----------------------------------------------------------------------------------------------------------------------
namespace Strawberry::RLPuyo
{
	enum class Action
	{
		None,
		Left,
		Right,
		Swap,
		OffensiveSkill,
		DefensiveSkill,
	};


	Core::Optional<Action> ActionFromEvent(Window::Event& event);
}