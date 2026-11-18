#pragma once

#include "VCSettings.hpp"

#include <QSettings>

namespace vc3d::line_annotation
{

// Annotation without Lasagna or fiber predictions (Fibers panel checkbox,
// persisted in the VC3D ini): each solved span is a cubic spline through the
// controls, both open ends are straight display rays, and no normal source,
// inference dataset or fiber-follow model is opened. Read on the GUI thread
// when a request is built; a running solve keeps the mode it was built with
// (FiberModeOptimizationRequest::csplineOnly).
inline constexpr auto kCsplineOnlySettingKey = "lineAnnotation/cspline_only";

inline bool csplineOnlyAnnotationEnabled()
{
    const QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    return settings.value(kCsplineOnlySettingKey, false).toBool();
}

inline void setCsplineOnlyAnnotationEnabled(bool enabled)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(kCsplineOnlySettingKey, enabled);
}

}  // namespace vc3d::line_annotation
