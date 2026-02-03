#pragma once

#include "elements/JsonProfileEditor.hpp"
#include "elements/UserProfileStore.hpp"

#include <QObject>
#include <QString>
#include <QVector>

namespace vc3d::json_profiles {

inline QString robustTracerParamsJson()
{
    return QStringLiteral(
        "{\n"
        "  \"snap_weight\": 0.0,\n"
        "  \"normal_weight\": 0.0,\n"
        "  \"normal3dline_weight\": 1.0,\n"
        "  \"straight_weight\": 10.0,\n"
        "  \"dist_weight\": 1.0,\n"
        "  \"direction_weight\": 0.0,\n"
        "  \"sdir_weight\": 1.0,\n"
        "  \"correction_weight\": 1.0,\n"
        "  \"reference_ray_weight\": 0.0,\n"
        "  \"flipback_weight\": 0.0\n"
        "}\n");
}

inline QString robustTracerParams2Json()
{
    return QStringLiteral(
        "{\n"
        "  \"snap_weight\": 0.0,\n"
        "  \"normal_weight\": 4.0,\n"
        "  \"normal3dline_weight\": 1.0,\n"
        "  \"straight_weight\": 10.0,\n"
        "  \"dist_weight\": 1.0,\n"
        "  \"direction_weight\": 0.0,\n"
        "  \"sdir_weight\": 1.0,\n"
        "  \"correction_weight\": 1.0,\n"
        "  \"reference_ray_weight\": 0.0,\n"
        "  \"flipback_weight\": 0.0\n"
        "}\n");
}

inline QString robustTracerParams_w_SpaceLineCopyJson()
{
    return QStringLiteral(
        "{\n"
        "  \"space_line_weight\": 0.1,\n"
        "  \"resume_local_row_flip_dot_thresh\": 0.5,\n"
        "  \"resume_local_row_self_intersection_cell_factor\": 32.0,\n"
        "  \"resume_local_row_flip_window\": 4.0,\n"
        "  \"snap_weight\": 0.0,\n"
        "  \"normal_weight\": 1.0,\n"
        "  \"normal3dline_weight\": 1.0,\n"
        "  \"straight_weight\": 10.0,\n"
        "  \"dist_weight\": 1.0,\n"
        "  \"direction_weight\": 0.0,\n"
        "  \"sdir_weight\": 0.0,\n"
        "  \"correction_weight\": 1.0,\n"
        "  \"reference_ray_weight\": 0.0,\n"
        "  \"flipback_weight\": 0.0\n"
        "}\n");
}

inline QString robustTracerParams_w_SpaceLineGrowJson()
{
    return QStringLiteral(
        "{\n"
        "  \"space_line_weight\": 0.02,\n"
        "  \"snap_weight\": 0.1,\n"
        "  \"normal_weight\": 5.0,\n"
        "  \"normal3dline_weight\": 0.1,\n"
        "  \"straight_weight\": 0.1,\n"
        "  \"dist_weight\": 1.0,\n"
        "  \"direction_weight\": 0.0,\n"
        "  \"sdir_weight\": 0.1,\n"
        "  \"correction_weight\": 1.0,\n"
        "  \"reference_ray_weight\": 0.0,\n"
        "  \"flipback_weight\": 0.0\n"
        "}\n");
}

template <typename TrFn>
inline QVector<JsonProfileEditor::Profile> tracerParamProfiles(TrFn&& trFn)
{
    QVector<JsonProfileEditor::Profile> profiles;
    profiles.push_back({QStringLiteral("custom"), trFn("Custom"), QString(), true});
    profiles.push_back({QStringLiteral("default"), trFn("Default"), QString(), false});
    profiles.push_back({QStringLiteral("robust"), trFn("Robust"), robustTracerParamsJson(), false});
    profiles.push_back({QStringLiteral("robust_2"), trFn("Robust 2"), robustTracerParams2Json(), false});
    profiles.push_back({QStringLiteral("robust_3"), trFn("Robust with Spaceline (Copy)"), robustTracerParams_w_SpaceLineCopyJson(), false});
    profiles.push_back({QStringLiteral("robust_3"), trFn("Robust with Spaceline (Grow)"), robustTracerParams_w_SpaceLineGrowJson(), false});
    return profiles;
}

inline QVector<JsonProfileEditor::Profile> tracerParamProfiles()
{
    return tracerParamProfiles([](const char* text) { return QObject::tr(text); });
}

inline bool isTracerParamProfileId(const QString& profileId)
{
    const auto profiles = tracerParamProfiles();
    for (const auto& profile : profiles) {
        if (profile.id == profileId) {
            return true;
        }
    }
    return false;
}

inline QString tracerParamProfileJson(const QString& profileId)
{
    const auto profiles = tracerParamProfiles();
    for (const auto& profile : profiles) {
        if (profile.id == profileId) {
            return profile.jsonText;
        }
    }
    return QString();
}

inline bool isValidProfileId(const QString& profileId)
{
    if (vc3d::user_profiles::isUserProfileId(profileId)) {
        return true;
    }
    return isTracerParamProfileId(profileId);
}

} // namespace vc3d::json_profiles
