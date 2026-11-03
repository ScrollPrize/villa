#pragma once

#include "vc/core/types/Volume.hpp"

#include <QString>

#include <memory>

namespace vc3d {

// The volume argument for a command-line tool. A fully remote (streaming-only)
// volume has an empty Volume::path(), so it passes its remote locator instead;
// an empty path made the child fail instantly (issue #1188).
inline QString commandPathForVolume(const std::shared_ptr<Volume>& volume)
{
    if (!volume) {
        return {};
    }
    if (volume->isRemote()) {
        return QString::fromStdString(volume->remoteLocator());
    }
    return QString::fromStdString(volume->path().string());
}

} // namespace vc3d
