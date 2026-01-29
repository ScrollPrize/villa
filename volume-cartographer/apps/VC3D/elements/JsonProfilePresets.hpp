#pragma once

#include "elements/JsonProfileEditor.hpp"

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
        "  \"reference_ray_weight\": 0.0\n"
        "}\n");
}

inline QString robustTracerParams2Json()
{
    return QStringLiteral(
        "{\n"
        "  \"snap_weight\": 0.0,\n"
        "  \"normal_weight\": 8.0,\n"
        "  \"normal3dline_weight\": 1.0,\n"
        "  \"straight_weight\": 7.0,\n"
        "  \"dist_weight\": 1.0,\n"
        "  \"direction_weight\": 0.0,\n"
        "  \"sdir_weight\": 1.0,\n"
        "  \"correction_weight\": 1.0,\n"
        "  \"reference_ray_weight\": 0.0\n"
        "}\n");
}

inline QString robustTracerParams3Json()
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
        "  \"reference_ray_weight\": 0.0\n"
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
    profiles.push_back({QStringLiteral("robust_3"), trFn("Robust 3"), robustTracerParams2Json(), false});
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

} // namespace vc3d::json_profiles
