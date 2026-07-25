#pragma once

#include <QString>

struct CommandLaunchError {
    enum Kind {
        Other,
        InvalidState,
        SegmentNotFound,
        VolumeNotFound,
        InputNotFound,
        RemoteVolume,
        ToolUnavailable,
        Busy,
    };

    Kind kind{Other};
    QString message;
};
