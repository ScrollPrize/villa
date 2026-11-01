#include "agent_bridge/AgentBridgeServer.hpp"
#include "agent_bridge/AgentBridgeEditSettings.hpp"

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QString>
#include <QStringList>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "CWindow.hpp"
#include "CState.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "segmentation/SegmentationWidget.hpp"
#include "segmentation/growth/SegmentationGrowth.hpp"

#include "vc/core/types/VolumePkg.hpp"

// ===========================================================================
// Segmentation editing settings. Brush, edit-scale and
// smoothing values go through SegmentationModule's setters, which the mouse
// wheel shortcuts use and which update and save the editing panel; the other
// settings go through SegmentationWidget, whose panels clamp, save and signal
// the module. Neural tracer executable and checkpoint paths are deliberately
// not exposed.
// ===========================================================================

namespace {

QString growthMethodName(SegmentationGrowthMethod method)
{
    switch (method) {
    case SegmentationGrowthMethod::Tracer:      return QStringLiteral("tracer");
    case SegmentationGrowthMethod::Corrections: return QStringLiteral("corrections");
    case SegmentationGrowthMethod::ManualAdd:   return QStringLiteral("manual_add");
    }
    return QStringLiteral("tracer");
}

SegmentationGrowthMethod growthMethodFromName(const QString& name)
{
    if (name == QLatin1String("corrections"))
        return SegmentationGrowthMethod::Corrections;
    return SegmentationGrowthMethod::Tracer;
}

QString directionName(SegmentationGrowthDirection direction)
{
    switch (direction) {
    case SegmentationGrowthDirection::Up:    return QStringLiteral("up");
    case SegmentationGrowthDirection::Down:  return QStringLiteral("down");
    case SegmentationGrowthDirection::Left:  return QStringLiteral("left");
    case SegmentationGrowthDirection::Right: return QStringLiteral("right");
    default:                                 return QStringLiteral("all");
    }
}

SegmentationGrowthDirection directionFromName(const QString& name)
{
    if (name == QLatin1String("up"))
        return SegmentationGrowthDirection::Up;
    if (name == QLatin1String("down"))
        return SegmentationGrowthDirection::Down;
    if (name == QLatin1String("left"))
        return SegmentationGrowthDirection::Left;
    return SegmentationGrowthDirection::Right;
}

SegmentationDirectionFieldOrientation orientationFromName(const QString& name)
{
    if (name == QLatin1String("horizontal"))
        return SegmentationDirectionFieldOrientation::Horizontal;
    if (name == QLatin1String("vertical"))
        return SegmentationDirectionFieldOrientation::Vertical;
    return SegmentationDirectionFieldOrientation::Normal;
}

SegmentationWidget* requireWidget(SegmentationWidget* widget)
{
    if (!widget) {
        throw AgentBridgeError{
            -32009, "Segmentation editing unavailable",
            QJsonObject{{"detail", "segmentation widget is not available"}}};
    }
    return widget;
}

SegmentationModule* requireModule(SegmentationModule* module)
{
    if (!module) {
        throw AgentBridgeError{
            -32009, "Segmentation editing unavailable",
            QJsonObject{{"detail", "segmentation module is not available"}}};
    }
    return module;
}

[[noreturn]] void throwParam(const QString& param, const QString& message)
{
    throw AgentBridgeError{-32602, message, QJsonObject{{"param", param}}};
}

}  // namespace


// ---------------------------------------------------------------------------
// segmentation.get_settings / segmentation.set_settings
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::segmentationSettingsJson() const
{
    const SegmentationWidget* widget =
        requireWidget(_window ? _window->_segmentationWidget : nullptr);
    CState* state = _window ? _window->_state : nullptr;

    auto brush = [](float radius, float sigma) {
        return QJsonObject{{"radius", radius}, {"sigma", sigma}};
    };
    QJsonObject pushPull = brush(widget->pushPullRadius(), widget->pushPullSigma());
    pushPull["step"] = widget->pushPullStep();

    QJsonArray directions;
    for (const auto direction : widget->allowedGrowthDirections())
        directions.append(directionName(direction));

    const auto [zMin, zMax] = widget->correctionsZBounds();
    const QJsonObject zRange{
        {"enabled", widget->correctionsZRange().has_value()},
        {"min", zMin},
        {"max", zMax},
    };

    QJsonArray directionFields;
    for (const auto& field : widget->directionFieldConfigs()) {
        directionFields.append(QJsonObject{
            {"path", field.path},
            {"orientation", segmentationDirectionFieldOrientationKey(field.orientation)},
            {"scale", field.scale},
            {"weight", field.weight},
        });
    }

    const std::string growthVolume = state ? state->segmentationGrowthVolumeId() : std::string{};

    return {
        {"editingEnabled", widget->isEditingEnabled()},
        {"drag", brush(widget->dragRadius(), widget->dragSigma())},
        {"pushPull", pushPull},
        {"editScale", widget->editScale()},
        {"smoothingStrength", widget->smoothingStrength()},
        {"smoothingIterations", widget->smoothingIterations()},
        {"showHoverMarker", widget->showHoverMarker()},
        // Manual Add is entered through segmentation.manual_add.begin; the
        // growth method reported is the one growth uses, so get_settings can
        // be passed back to set_settings.
        {"growthMethod", growthMethodName(
                             widget->growthMethod() == SegmentationGrowthMethod::ManualAdd
                                 ? widget->lastNonManualGrowthMethod()
                                 : widget->growthMethod())},
        {"manualAddSelected", widget->growthMethod() == SegmentationGrowthMethod::ManualAdd},
        {"growthSteps", widget->growthSteps()},
        {"growthScale", widget->growthScale()},
        {"growthDirections", directions},
        {"zRange", zRange},
        {"growthVolumeId", growthVolume.empty() ? QJsonValue(QJsonValue::Null)
                                                : QJsonValue(QString::fromStdString(growthVolume))},
        {"directionFields", directionFields},
    };
}

QJsonObject AgentBridgeServer::handleSegmentationGetSettings(const QJsonValue&)
{
    return segmentationSettingsJson();
}

QJsonObject AgentBridgeServer::handleSegmentationSetSettings(const QJsonValue& params)
{
    SegmentationWidget* widget = requireWidget(_window ? _window->_segmentationWidget : nullptr);
    SegmentationModule* module = requireModule(_window ? _window->_segmentationModule.get() : nullptr);
    CState* state = _window ? _window->_state : nullptr;
    const QJsonObject p = params.toObject();

    // Validate everything that can fail before changing any value.
    const bool setGrowthVolume = p.contains("growthVolumeId") && !p.value("growthVolumeId").isNull();
    if (setGrowthVolume) {
        const QString volumeId = p.value("growthVolumeId").toString();
        bool known = false;
        if (state && state->vpkg()) {
            for (const auto& id : state->vpkg()->volumeIDs())
                known = known || QString::fromStdString(id) == volumeId;
        }
        if (!known) {
            throw AgentBridgeError{
                -32007, QStringLiteral("Unknown volume id: %1").arg(volumeId),
                QJsonObject{{"kind", "volume"}, {"id", volumeId}}};
        }
    }
    std::vector<SegmentationDirectionFieldConfig> directionFields;
    for (const QJsonValue& value : p.value("directionFields").toArray()) {
        const QJsonObject field = value.toObject();
        SegmentationDirectionFieldConfig config;
        config.path = field.value("path").toString();
        if (config.path.trimmed().isEmpty())
            throwParam(QStringLiteral("directionFields"), QStringLiteral("direction field path must not be blank"));
        config.orientation = orientationFromName(field.value("orientation").toString());
        config.scale = field.value("scale").toInt(0);
        config.weight = field.value("weight").toDouble(1.0);
        directionFields.push_back(config);
    }
    // A Z range given with one bound keeps the other field's value; the
    // panel's own bounds are taken back even when its spin boxes hold them
    // in the wrong order.
    const QJsonObject zRange = p.value("zRange").toObject();
    const auto [currentZMin, currentZMax] = widget->correctionsZBounds();
    const auto [zMin, zMax] = p.contains("zRange")
        ? agentBridgeZRangeUpdate(zRange, currentZMin, currentZMax)
        : std::pair<int, int>{currentZMin, currentZMax};

    // The growth volume is the one change that can still be refused (the
    // panel may not offer a volume the project has), so it goes first.
    if (setGrowthVolume && !widget->selectGrowthVolume(p.value("growthVolumeId").toString())) {
        const QString volumeId = p.value("growthVolumeId").toString();
        throw AgentBridgeError{
            -32007, QStringLiteral("Volume %1 is not offered for growth").arg(volumeId),
            QJsonObject{{"kind", "volume"}, {"id", volumeId},
                        {"detail", "the Segmentation panel does not list this volume"}}};
    }

    // Brushes, edit scale and smoothing: the module's setters clamp the value,
    // apply it and update the editing panel, which saves it.
    const QJsonObject drag = p.value("drag").toObject();
    if (drag.contains("radius"))
        module->setDragRadius(static_cast<float>(drag.value("radius").toDouble()));
    if (drag.contains("sigma"))
        module->setDragSigma(static_cast<float>(drag.value("sigma").toDouble()));
    const QJsonObject pushPull = p.value("pushPull").toObject();
    if (pushPull.contains("radius"))
        module->setPushPullRadius(static_cast<float>(pushPull.value("radius").toDouble()));
    if (pushPull.contains("sigma"))
        module->setPushPullSigma(static_cast<float>(pushPull.value("sigma").toDouble()));
    if (pushPull.contains("step"))
        module->setPushPullStepMultiplier(static_cast<float>(pushPull.value("step").toDouble()));
    if (p.contains("editScale"))
        module->setEditScale(static_cast<float>(p.value("editScale").toDouble()));
    if (p.contains("smoothingStrength"))
        module->setSmoothingStrength(static_cast<float>(p.value("smoothingStrength").toDouble()));
    if (p.contains("smoothingIterations"))
        module->setSmoothingIterations(p.value("smoothingIterations").toInt());
    // The panel signals hover-marker, growth-method and Z-range changes
    // itself; the module follows through its connections.
    if (p.contains("showHoverMarker"))
        widget->setShowHoverMarker(p.value("showHoverMarker").toBool());
    if (p.contains("growthMethod"))
        widget->setGrowthMethod(growthMethodFromName(p.value("growthMethod").toString()));
    if (p.contains("growthSteps"))
        widget->setGrowthSteps(p.value("growthSteps").toInt());
    if (p.contains("growthScale"))
        widget->setGrowthScale(p.value("growthScale").toInt());
    if (p.contains("growthDirections")) {
        std::vector<SegmentationGrowthDirection> directions;
        for (const QJsonValue& value : p.value("growthDirections").toArray())
            directions.push_back(directionFromName(value.toString()));
        widget->setAllowedGrowthDirections(directions);
    }
    if (p.contains("zRange"))
        widget->setCorrectionsZRange(zRange.value("enabled").toBool(), zMin, zMax);
    if (p.contains("directionFields"))
        widget->setDirectionFieldConfigs(directionFields);


    return segmentationSettingsJson();
}

