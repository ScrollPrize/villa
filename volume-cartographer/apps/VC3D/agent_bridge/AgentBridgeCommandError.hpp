#pragma once

#include "CommandLaunchError.hpp"
#include "agent_bridge/AgentBridgeError.hpp"

inline AgentBridgeError commandLaunchErrorToBridgeError(
    const CommandLaunchError& error,
    const QString& fallbackMessage,
    const QString& segmentId,
    const QString& source)
{
    QJsonObject data;
    data["detail"] = error.message;

    switch (error.kind) {
    case CommandLaunchError::SegmentNotFound:
        data["kind"] = "segment";
        data["id"] = segmentId;
        return {-32007, "Segment not found", data};
    case CommandLaunchError::VolumeNotFound:
        data["kind"] = "volume";
        return {-32007, "Volume not found", data};
    case CommandLaunchError::InputNotFound:
        data["kind"] = "file";
        return {-32007, "Required input not found", data};
    case CommandLaunchError::RemoteVolume:
        return {-32009, "Remote volume not supported", data};
    case CommandLaunchError::ToolUnavailable:
        return {-32006, "Command line tool unavailable", data};
    case CommandLaunchError::Busy:
        data["source"] = source;
        return {-32004, "A job is already running", data};
    case CommandLaunchError::InvalidState:
    case CommandLaunchError::Other:
        return {-32005, fallbackMessage, data};
    }
    return {-32005, fallbackMessage, data};
}
