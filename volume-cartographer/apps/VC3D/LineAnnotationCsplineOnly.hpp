#pragma once

#include "VCSettings.hpp"

#include <QSettings>

#include <atomic>

namespace vc3d::line_annotation
{

// Annotation without Lasagna or fiber predictions (Fibers panel checkbox,
// persisted in the VC3D ini): every span is a cubic spline through the
// controls, both open ends are straight display rays, and no normal source,
// inference dataset or fiber-follow model is opened. Read by solve workers,
// hence the atomic; the first read loads the persisted value.
inline constexpr auto kCsplineOnlySettingKey = "lineAnnotation/cspline_only";

namespace detail
{
inline std::atomic<int> csplineOnlyState{-1};
}  // namespace detail

inline bool csplineOnlyAnnotationEnabled()
{
    int state = detail::csplineOnlyState.load(std::memory_order_relaxed);
    if (state < 0) {
        const QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        state = settings.value(kCsplineOnlySettingKey, false).toBool() ? 1 : 0;
        int expected = -1;
        detail::csplineOnlyState.compare_exchange_strong(expected, state);
        state = detail::csplineOnlyState.load(std::memory_order_relaxed);
    }
    return state == 1;
}

inline void setCsplineOnlyAnnotationEnabled(bool enabled)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(kCsplineOnlySettingKey, enabled);
    detail::csplineOnlyState.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

}  // namespace vc3d::line_annotation
