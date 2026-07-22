#pragma once

// Shared free helpers for the Agent Bridge server. The server's member
// functions are defined across several per-domain translation units; these
// helpers are used by more than one of them, so they live here as plain
// `inline` free functions (ODR-safe across TUs). Bodies are moved verbatim
// from AgentBridgeServer.cpp.

#include <cmath>
#include <optional>

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QString>

#include <opencv2/core.hpp>

#include "agent_bridge/AgentBridgeServer.hpp"     // AgentBridgeError
#include "segmentation/tools/ManualAddTool.hpp"   // ManualAddTool enums

inline QJsonObject vec3ToJson(const cv::Vec3f& v)
{
    QJsonObject o;
    o["x"] = static_cast<double>(v[0]);
    o["y"] = static_cast<double>(v[1]);
    o["z"] = static_cast<double>(v[2]);
    return o;
}

inline QJsonObject paramsObject(const QJsonValue& params)
{
    if (params.isObject())
        return params.toObject();
    return QJsonObject();
}

inline cv::Vec3f jsonToVec3(const QJsonValue& value, const char* paramName)
{
    if (!value.isObject()) {
        QJsonObject data;
        data["param"] = QString::fromLatin1(paramName);
        throw AgentBridgeError{-32602,
            QStringLiteral("%1 must be an object {x, y, z}").arg(QLatin1String(paramName)), data};
    }
    const QJsonObject o = value.toObject();
    if (!o.contains("x") || !o.contains("y") || !o.contains("z")) {
        QJsonObject data;
        data["param"] = QString::fromLatin1(paramName);
        throw AgentBridgeError{-32602,
            QStringLiteral("%1 requires x, y and z").arg(QLatin1String(paramName)), data};
    }
    const double x = o.value("x").toDouble();
    const double y = o.value("y").toDouble();
    const double z = o.value("z").toDouble();
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        QJsonObject data;
        data["param"] = QString::fromLatin1(paramName);
        throw AgentBridgeError{-32602,
            QStringLiteral("%1 has non-finite coordinates").arg(QLatin1String(paramName)), data};
    }
    return cv::Vec3f(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
}

inline QString linePreviewModeToString(ManualAddTool::LinePreviewMode mode)
{
    switch (mode) {
    case ManualAddTool::LinePreviewMode::VerticalOnly:   return QStringLiteral("vertical");
    case ManualAddTool::LinePreviewMode::HorizontalOnly: return QStringLiteral("horizontal");
    case ManualAddTool::LinePreviewMode::Cross:          return QStringLiteral("cross");
    case ManualAddTool::LinePreviewMode::CrossFill:      return QStringLiteral("cross_fill");
    }
    return QStringLiteral("cross");
}

inline QString interpolationModeToString(ManualAddTool::InterpolationMode mode)
{
    switch (mode) {
    case ManualAddTool::InterpolationMode::ThinPlateSpline:        return QStringLiteral("thin_plate_spline");
    case ManualAddTool::InterpolationMode::TracerRestrictedToFill: return QStringLiteral("tracer_restricted_to_fill");
    }
    return QStringLiteral("thin_plate_spline");
}
