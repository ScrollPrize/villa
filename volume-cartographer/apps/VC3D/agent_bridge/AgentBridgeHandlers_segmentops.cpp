#include "agent_bridge/AgentBridgeServer.hpp"

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QString>
#include <QStringList>

#include <algorithm>
#include <string>
#include <vector>

#include "CWindow.hpp"
#include "CState.hpp"
#include "SegmentationCommandHandler.hpp"
#include "segmentation/SegmentationWidget.hpp"

#include "vc/core/types/VolumePkg.hpp"

// ===========================================================================
// Segment file operations over SegmentationCommandHandler's dialog-free
// cores. The context-menu and Actions-menu slots run the same cores with
// their dialogs, so both callers share validation, persistence and UI
// refresh.
// ===========================================================================

namespace {

CState* requireProject(CState* state)
{
    if (!state || !state->hasVpkg() || !state->vpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    return state;
}

// Checks the package's segment ids without loading any surface; the operation
// cores load what they need and report load failures themselves.
void requireKnownSegment(CState* state, const QString& segmentId)
{
    const std::vector<std::string> ids = state->vpkg()->segmentationIDs();
    if (std::find(ids.begin(), ids.end(), segmentId.toStdString()) == ids.end()) {
        throw AgentBridgeError{
            -32007, QStringLiteral("Segment not found: %1").arg(segmentId),
            QJsonObject{{"kind", "segment"}, {"id", segmentId}}};
    }
}

SegmentationCommandHandler* requireCommandHandler(SegmentationCommandHandler* handler,
                                                  const QString& operation)
{
    if (!handler) {
        throw AgentBridgeError{
            -32009, QStringLiteral("%1 unavailable").arg(operation),
            QJsonObject{{"detail", "segmentation command handler is not available"}}};
    }
    return handler;
}

// Destructive or lossy calls need an explicit confirm:true, like
// segments.delete.
void requireConfirm(const QJsonObject& p, const QString& reason)
{
    if (!p.value("confirm").toBool(false)) {
        throw AgentBridgeError{
            -32602, QStringLiteral("confirm must be true (%1)").arg(reason),
            QJsonObject{{"param", "confirm"},
                        {"reason", "destructive; pass confirm=true"}}};
    }
}

}  // namespace


// Refuses an in-place rewrite of a segment's files while something else may
// be reading or writing them: an editing session on that segment, a mask
// render, or a command-line tool run.
void AgentBridgeServer::requireSegmentFilesIdle(const QString& segmentId,
                                                const QString& operation) const
{
    CState* state = _window ? _window->_state : nullptr;
    if (_window && _window->_segmentationWidget &&
        _window->_segmentationWidget->isEditingEnabled() && state &&
        (segmentId.isEmpty() || state->activeSurfaceId() == segmentId.toStdString())) {
        throw AgentBridgeError{
            -32004, QStringLiteral("Cannot %1 the segment being edited").arg(operation),
            QJsonObject{{"detail", "disable segmentation editing first"}}};
    }
    if (_window && _window->_maskRenderInProgress) {
        throw AgentBridgeError{
            -32004, "A mask render is in progress",
            QJsonObject{{"detail", "a mask render is in progress"}}};
    }
    requireSourceIdle(QStringLiteral("tool"));
}


// ---------------------------------------------------------------------------
// segment.transform
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleSegmentTransform(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = requireProject(_window ? _window->_state : nullptr);
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const QString segmentId = p.value("segmentId").toString();
    requireKnownSegment(state, segmentId);

    // The descriptor restricts op to these spellings.
    const QString op = p.value("op").toString();
    using Transform = SegmentationCommandHandler::SurfaceTransform;
    const Transform transform = op == QLatin1String("flip_u")   ? Transform::FlipU
                              : op == QLatin1String("rotate90") ? Transform::Rotate90
                                                                : Transform::FlipV;
    // Flips are exact and their own inverse; a rotation resamples the grid.
    if (transform == Transform::Rotate90)
        requireConfirm(p, QStringLiteral("rotate90 resamples the grid in place"));

    // Transforming the grid under an open editing session would leave the
    // session's preview points indexed against the old layout.
    requireSegmentFilesIdle(segmentId, QStringLiteral("transform"));

    SegmentationCommandHandler* handler = requireCommandHandler(
        _window->_segmentationCommandHandler.get(), QStringLiteral("Transform"));

    QString err;
    if (!handler->transformSurface(segmentId.toStdString(), transform, &err)) {
        throw AgentBridgeError{-32005, "Transform failed", QJsonObject{{"detail", err}}};
    }

    QJsonObject result;
    result["transformed"] = true;
    result["segmentId"] = segmentId;
    result["op"] = op;
    return result;
}
