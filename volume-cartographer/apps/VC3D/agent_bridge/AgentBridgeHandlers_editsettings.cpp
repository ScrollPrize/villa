#include "agent_bridge/AgentBridgeServer.hpp"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QString>
#include <QStringList>

#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "CWindow.hpp"
#include "CState.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "segmentation/SegmentationWidget.hpp"
#include "segmentation/growth/SegmentationGrowth.hpp"
#include "overlays/SegmentationOverlayController.hpp"

#include "utils/Json.hpp"
#include "vc/core/types/VolumePkg.hpp"

// ===========================================================================
// Segmentation editing settings and approval-mask mode. Brush, edit-scale and
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
    case SegmentationGrowthMethod::PatchTracer: return QStringLiteral("patch_tracer");
    case SegmentationGrowthMethod::ManualAdd:   return QStringLiteral("manual_add");
    }
    return QStringLiteral("tracer");
}

SegmentationGrowthMethod growthMethodFromName(const QString& name)
{
    if (name == QLatin1String("corrections"))
        return SegmentationGrowthMethod::Corrections;
    if (name == QLatin1String("patch_tracer"))
        return SegmentationGrowthMethod::PatchTracer;
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

QJsonObject jsonFromUtils(const utils::Json& value)
{
    return QJsonDocument::fromJson(QByteArray::fromStdString(value.dump())).object();
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

    QJsonObject patchTracer = jsonFromUtils(widget->patchTracerParamsJson());
    patchTracer.remove(QStringLiteral("umbilicus_path"));

    QJsonArray directionFields;
    for (const auto& field : widget->directionFieldConfigs()) {
        directionFields.append(QJsonObject{
            {"path", field.path},
            {"orientation", segmentationDirectionFieldOrientationKey(field.orientation)},
            {"scale", field.scale},
            {"weight", field.weight},
        });
    }

    const QString approvalMode = widget->editApprovedMask()     ? QStringLiteral("approve")
                               : widget->editUnapprovedMask()   ? QStringLiteral("unapprove")
                                                                : QStringLiteral("off");
    const std::string growthVolume = state ? state->segmentationGrowthVolumeId() : std::string{};

    return {
        {"editingEnabled", widget->isEditingEnabled()},
        {"drag", brush(widget->dragRadius(), widget->dragSigma())},
        {"line", brush(widget->lineRadius(), widget->lineSigma())},
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
        {"patchTracer", patchTracer},
        {"directionFields", directionFields},
        {"approvalMask", QJsonObject{
            {"show", widget->showApprovalMask()},
            {"mode", approvalMode},
            {"brushRadius", widget->approvalBrushRadius()},
            {"brushDepth", widget->approvalBrushDepth()},
            {"opacity", widget->approvalMaskOpacity() / 100.0},
            {"autoApprove", widget->autoApprovalEnabled()},
            {"autoApproveRadius", widget->autoApprovalRadius()},
            {"autoApproveThreshold", widget->autoApprovalThreshold()},
            {"autoApproveMaxDistance", widget->autoApprovalMaxDistance()},
        }},
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
    utils::Json patchTracer = utils::Json::object();
    if (p.contains("patchTracer")) {
        const QJsonObject requested = p.value("patchTracer").toObject();
        const utils::Json current = widget->patchTracerParamsJson();
        for (auto it = requested.begin(); it != requested.end(); ++it) {
            const std::string key = it.key().toStdString();
            if (it.key() == QLatin1String("umbilicus_path") || !current.contains(key)) {
                throwParam(QStringLiteral("patchTracer"),
                           QStringLiteral("Unknown patch tracer parameter: %1").arg(it.key()));
            }
            const utils::Json& known = current.at(key);
            const QJsonValue value = it.value();
            const bool typeMatches = known.is_boolean()
                ? value.isBool()
                : value.isDouble() && std::isfinite(value.toDouble()) &&
                      (!known.is_number_integer() ||
                       value.toDouble() == std::floor(value.toDouble()));
            if (!typeMatches) {
                throwParam(QStringLiteral("patchTracer"),
                           QStringLiteral("Patch tracer parameter %1 has the wrong type").arg(it.key()));
            }
        }
        patchTracer = utils::Json::parse(
            QJsonDocument(requested).toJson(QJsonDocument::Compact).toStdString());
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
    // A Z range given with one bound keeps the other field's value.
    const QJsonObject zRange = p.value("zRange").toObject();
    const auto [currentZMin, currentZMax] = widget->correctionsZBounds();
    const int zMin = zRange.value("min").toInt(currentZMin);
    const int zMax = zRange.value("max").toInt(currentZMax);
    if (p.contains("zRange") && zMin > zMax) {
        throwParam(QStringLiteral("zRange"), QStringLiteral("zRange.min must not exceed zRange.max"));
    }

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
    const QJsonObject line = p.value("line").toObject();
    if (line.contains("radius"))
        module->setLineRadius(static_cast<float>(line.value("radius").toDouble()));
    if (line.contains("sigma"))
        module->setLineSigma(static_cast<float>(line.value("sigma").toDouble()));
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
    // The panel signals hover-marker, growth-method, Z-range and approval
    // changes itself; the module follows through its connections.
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
    if (p.contains("patchTracer"))
        widget->setPatchTracerParams(patchTracer);
    if (p.contains("directionFields"))
        widget->setDirectionFieldConfigs(directionFields);

    const QJsonObject approval = p.value("approvalMask").toObject();
    if (approval.contains("show"))
        widget->setShowApprovalMask(approval.value("show").toBool());
    if (approval.contains("brushRadius"))
        widget->setApprovalBrushRadius(static_cast<float>(approval.value("brushRadius").toDouble()));
    if (approval.contains("brushDepth"))
        widget->setApprovalBrushDepth(static_cast<float>(approval.value("brushDepth").toDouble()));
    if (approval.contains("opacity")) {
        widget->setApprovalMaskOpacity(
            static_cast<int>(std::lround(approval.value("opacity").toDouble() * 100.0)));
    }
    if (approval.contains("autoApprove"))
        widget->setAutoApprovalEnabled(approval.value("autoApprove").toBool());
    if (approval.contains("autoApproveRadius"))
        widget->setAutoApprovalRadius(static_cast<float>(approval.value("autoApproveRadius").toDouble()));
    if (approval.contains("autoApproveThreshold"))
        widget->setAutoApprovalThreshold(static_cast<float>(approval.value("autoApproveThreshold").toDouble()));
    if (approval.contains("autoApproveMaxDistance"))
        widget->setAutoApprovalMaxDistance(static_cast<float>(approval.value("autoApproveMaxDistance").toDouble()));

    return segmentationSettingsJson();
}


// ---------------------------------------------------------------------------
// segmentation.approval_mask.set_mode / undo
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleApprovalMaskSetMode(const QJsonValue& params)
{
    SegmentationWidget* widget = requireWidget(_window ? _window->_segmentationWidget : nullptr);
    CState* state = _window ? _window->_state : nullptr;
    const QJsonObject p = params.toObject();
    const QString mode = p.value("mode").toString();

    if (mode != QLatin1String("off")) {
        if (!state || !state->surface("segmentation")) {
            throw AgentBridgeError{
                -32007, "No active segmentation surface",
                QJsonObject{{"kind", "segment"}, {"detail", "activate a segment first"}}};
        }
        // Painting needs the mask on screen, as the B and N keys do.
        if (!widget->showApprovalMask() && !p.value("show").toBool(false)) {
            throwParam(QStringLiteral("show"),
                       QStringLiteral("the approval mask is hidden; pass show=true to show it"));
        }
    }

    if (p.contains("show"))
        widget->setShowApprovalMask(p.value("show").toBool());
    // The panel enforces that approve and unapprove exclude each other and
    // signals the module.
    widget->setEditApprovedMask(mode == QLatin1String("approve"));
    widget->setEditUnapprovedMask(mode == QLatin1String("unapprove"));

    const QString applied = widget->editApprovedMask()   ? QStringLiteral("approve")
                          : widget->editUnapprovedMask() ? QStringLiteral("unapprove")
                                                         : QStringLiteral("off");
    return {{"mode", applied}, {"show", widget->showApprovalMask()}};
}

QJsonObject AgentBridgeServer::handleApprovalMaskUndo(const QJsonValue&)
{
    SegmentationModule* module = requireModule(_window ? _window->_segmentationModule.get() : nullptr);
    SegmentationOverlayController* overlay = module->overlay();
    if (!module->isEditingApprovalMask()) {
        throw AgentBridgeError{
            -32009, "Approval mask painting is off",
            QJsonObject{{"detail", "set an approve or unapprove mode first"}}};
    }
    const bool canUndo = overlay && overlay->canUndoApprovalMaskPaint();
    if (canUndo)
        module->undoApprovalStroke();
    return {{"undone", canUndo}};
}
