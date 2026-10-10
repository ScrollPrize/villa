#pragma once

#include "VCSettings.hpp"
#include "vc/core/types/VolumePkg.hpp"

#include <QDir>
#include <QSettings>
#include <QString>

namespace vc3d {

inline QString defaultNewProjectDirectory(QSettings& settings)
{
    const auto autosaveFile = VolumePkg::autosaveFile();
    if (!autosaveFile.empty()) {
        const auto directory = QString::fromStdString(autosaveFile.parent_path().string());
        QDir().mkpath(directory);
        if (!directory.isEmpty()) return directory;
    }
    return settings.value(vc3d::settings::project::DEFAULT_PATH).toString();
}

} // namespace vc3d
