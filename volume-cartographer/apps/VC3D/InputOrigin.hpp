#pragma once

// Whether the input being handled was sent by an agent rather than typed by a
// person. The agent bridge marks its synthetic key events with an
// AgentInputScope; a handler that acts for "the user" (a shortcut) asks
// handlingAgentInput() and acts for the agent instead. GUI thread only.

namespace vc3d {

[[nodiscard]] inline int& agentInputDepth()
{
    static int depth = 0;
    return depth;
}

[[nodiscard]] inline bool handlingAgentInput()
{
    return agentInputDepth() > 0;
}

class AgentInputScope {
public:
    AgentInputScope() { ++agentInputDepth(); }
    ~AgentInputScope() { --agentInputDepth(); }
    AgentInputScope(const AgentInputScope&) = delete;
    AgentInputScope& operator=(const AgentInputScope&) = delete;
};

}  // namespace vc3d
