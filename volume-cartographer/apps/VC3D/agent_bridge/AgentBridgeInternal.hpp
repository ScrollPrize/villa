#pragma once

// Shared inline free helpers for the Agent Bridge server, used across several
// per-domain translation units (ODR-safe as inline free functions).

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QPointF>
#include <QString>

#include <opencv2/core.hpp>

#include "agent_bridge/AgentBridgeError.hpp"       // AgentBridgeError
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
    if (params.isObject()) {
        return params.toObject();
    }
    if (params.isUndefined() || params.isNull()) {
        return {};
    }
    QJsonObject data;
    data["param"] = "params";
    throw AgentBridgeError{-32602, "params must be an object", data};
}

// ---------------------------------------------------------------------------
// Strict wire-parameter parsing (SPEC §5). QJsonValue::toDouble()/toInt()/
// toBool() silently coerce a wrong-typed value to 0/false, which turns a
// malformed request into a plausible-but-wrong operation. These helpers reject a
// PRESENT-but-malformed value with -32602 + data.param instead. Callers still
// decide required vs optional: an omitted optional may keep its documented
// default, but a value that IS present must have the right type.
// ---------------------------------------------------------------------------

[[noreturn]] inline void throwParamError(const char* paramName, const QString& detail)
{
    QJsonObject data;
    data["param"] = QString::fromLatin1(paramName);
    throw AgentBridgeError{-32602,
        QStringLiteral("%1 %2").arg(QLatin1String(paramName), detail), data};
}

// Require a JSON number. Rejects strings, booleans, null, objects, arrays.
inline double jsonRequireNumber(const QJsonValue& v, const char* paramName)
{
    if (!v.isDouble())
        throwParamError(paramName, QStringLiteral("must be a number"));
    return v.toDouble();
}

// Require a finite JSON number (rejects NaN/Inf in addition to wrong types).
inline double jsonRequireFinite(const QJsonValue& v, const char* paramName)
{
    const double d = jsonRequireNumber(v, paramName);
    if (!std::isfinite(d))
        throwParamError(paramName, QStringLiteral("must be a finite number"));
    return d;
}

// Require a finite JSON number that also stays finite when narrowed to float.
// A finite double beyond float's range (e.g. 1e300) becomes +/-inf on cast, which
// would silently violate the finite-coordinate promise of the {x,y,z} helpers (SPEC §5).
inline double jsonRequireFiniteFloat(const QJsonValue& v, const char* paramName)
{
    const double d = jsonRequireFinite(v, paramName);
    if (std::abs(d) > static_cast<double>(std::numeric_limits<float>::max()))
        throwParamError(paramName, QStringLiteral("is out of range for a float coordinate"));
    return d;
}

inline QString jsonRequireString(const QJsonValue& value, const char* paramName)
{
    if (!value.isString())
        throwParamError(paramName, QStringLiteral("must be a string"));
    return value.toString();
}

// Require a JSON string on an object key (rejects absent + wrong type).
inline QString jsonRequireString(const QJsonObject& o, const char* key)
{
    return jsonRequireString(o.value(QLatin1String(key)), key);
}

// Optional string: absent -> default; present-but-wrong-type -> reject.
inline QString jsonOptionalString(const QJsonObject& o, const char* key,
                                  const QString& def = QString())
{
    if (!o.contains(QLatin1String(key)))
        return def;
    return jsonRequireString(o, key);
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
    // Reject wrong-typed (string/bool) or non-finite coordinates rather than
    // silently coercing them to 0 (SPEC §5). jsonRequireFiniteFloat additionally
    // rejects a finite double that would overflow to +/-inf on the float cast below.
    const double x = jsonRequireFiniteFloat(o.value("x"), "x");
    const double y = jsonRequireFiniteFloat(o.value("y"), "y");
    const double z = jsonRequireFiniteFloat(o.value("z"), "z");
    return cv::Vec3f(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
}

inline bool volumePointInBounds(const cv::Vec3f& point,
                                const std::array<int, 3>& shapeXyz)
{
    for (std::size_t axis = 0; axis < shapeXyz.size(); ++axis) {
        if (!std::isfinite(point[axis]) ||
            point[axis] < 0.0f ||
            point[axis] >= static_cast<float>(shapeXyz[axis])) {
            return false;
        }
    }
    return true;
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
