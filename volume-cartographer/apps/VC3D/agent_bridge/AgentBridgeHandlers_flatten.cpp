#include "agent_bridge/AgentBridgeServer.hpp"
#include "agent_bridge/AgentBridgeInternal.hpp"

#include <QBuffer>
#include <QByteArray>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFutureWatcher>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLocalServer>
#include <QLocalSocket>
#include <QMdiSubWindow>
#include <QPixmap>
#include <QPointF>
#include <QTabWidget>
#include <QTimer>
#include <QVector3D>
#include <QWidget>
#include <QtConcurrent>

#include <algorithm>
#include <filesystem>
#include <set>
#include <string>
#include <unordered_set>

#include "CWindow.hpp"
#include "AxisAlignedSliceController.hpp"
#include "CState.hpp"
#include "LasagnaServiceManager.hpp"
#include "LineAnnotationController.hpp"
#include "LineAnnotationDialog.hpp"
#include "MenuActionController.hpp"
#include "OpenDataManifest.hpp"
#include "OpenDataSampleProject.hpp"
#include "OpenDataSegmentCache.hpp"
#include "SeedingWidget.hpp"
#include "SegmentationCommandHandler.hpp"
#include "SurfacePanelController.hpp"
#include "CommandLineToolRunner.hpp"
#include "ViewerManager.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "segmentation/SegmentationPushPullConfig.hpp"
#include "segmentation/SegmentationWidget.hpp"
#include "segmentation/tools/SegmentationPushPullTool.hpp"
#include "segmentation/panels/SegmentationLasagnaPanel.hpp"
#include "segmentation/growth/SegmentationGrowth.hpp"
#include "segmentation/growth/SegmentationGrower.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include "vc/core/types/Segmentation.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/ui/VCCollection.hpp"


// ---------------------------------------------------------------------------
// Flattening RPCs: flatten.slim / flatten.abf / flatten.straighten
// ---------------------------------------------------------------------------

// Shared launch body for all three flatten RPCs: the handler builds `launch`
// (a closure over the concrete start* launcher); this registers the job and
// maps typed launch failures to JSON-RPC errors. Completion is driven by
// SegmentationCommandHandler::flattenJobFinished rather than a
// CommandLineToolRunner signal, since flatten jobs own their own
// QProcess/QtConcurrent lifecycle.
QJsonObject AgentBridgeServer::launchFlattenJob(
    const QString& kind, const QString& label, const QString& segmentId,
    const std::function<bool(CommandLaunchError* error,
                             QString* outDir)>& launch)
{
    // Only one flatten at a time (its own source, so it may run concurrently
    // with a "tool", "growth", or other source-specific job).
    requireSourceIdle(QStringLiteral("flatten"));

    SegmentationCommandHandler* handler =
        _window ? _window->_segmentationCommandHandler.get() : nullptr;
    if (!handler) {
        QJsonObject data;
        data["detail"] = "segmentation command handler is not available";
        throw AgentBridgeError{-32009, "Flatten unavailable", data};
    }

    const QString jobId = beginJob(QStringLiteral("flatten"), kind, label,
                                   /*broadcastStart=*/false);

    CommandLaunchError error;
    QString outputDir;
    if (!launch(&error, &outputDir)) {
        _activeJobs.remove(QStringLiteral("flatten"));
        throwCommandLaunchError(error, "Failed to start flatten",
                                segmentId, "flatten");
    }

    // The job ctor already emitted flattenJobStarted (adopted by
    // handleFlattenStarted into the active record); attach the resolved output
    // path and broadcast the start now that we have it.
    if (auto it = _activeJobs.find(QStringLiteral("flatten")); it != _activeJobs.end()) {
        it.value().outputPath = outputDir;
        broadcastJobProgress(it.value(), QStringLiteral("started"));
    }

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = kind;
    result["source"] = "flatten";
    result["outputDir"] = outputDir;
    return result;
}

QJsonObject AgentBridgeServer::handleFlattenSlim(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const QString segmentId = p.value("segmentId").toString();
    if (segmentId.isEmpty()) {
        QJsonObject data;
        data["param"] = "segmentId";
        throw AgentBridgeError{-32602, "segmentId is required", data};
    }

    SegmentationCommandHandler::SlimFlattenParams sp;  // headless defaults

    sp.iterations = p.value("iterations").toInt(sp.iterations);
    sp.tolerance = p.value("tolerance").toDouble(sp.tolerance);
    sp.energyType = p.value("energyType").toString(sp.energyType);
    sp.keepPercent = p.value("keepPercent").toDouble(sp.keepPercent);
    sp.inpaintHoles = p.value("inpaintHoles").toBool(sp.inpaintHoles);
    sp.outputDir = p.value("outputDir").toString();

    SegmentationCommandHandler* handler =
        _window ? _window->_segmentationCommandHandler.get() : nullptr;
    return launchFlattenJob(
        QStringLiteral("flatten.slim"), QStringLiteral("SLIM flatten"), segmentId,
        [handler, segmentId, sp](CommandLaunchError* error, QString* outDir) {
            return handler && handler->startSlimFlatten(
                segmentId.toStdString(), sp, error, outDir);
        });
}


QJsonObject AgentBridgeServer::handleFlattenAbf(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QString segmentId = p.value("segmentId").toString();
    if (segmentId.isEmpty()) {
        QJsonObject data;
        data["param"] = "segmentId";
        throw AgentBridgeError{-32602, "segmentId is required", data};
    }

    const int iterations = p.value("iterations").toInt(10);
    const int downsampleFactor = p.value("downsampleFactor").toInt(1);

    SegmentationCommandHandler* handler =
        _window ? _window->_segmentationCommandHandler.get() : nullptr;
    return launchFlattenJob(
        QStringLiteral("flatten.abf"), QStringLiteral("ABF++ flatten"), segmentId,
        [handler, segmentId, iterations, downsampleFactor](CommandLaunchError* error,
                                                           QString* outDir) {
            return handler && handler->startAbfFlatten(segmentId.toStdString(),
                                                       iterations, downsampleFactor,
                                                       error, outDir);
        });
}


QJsonObject AgentBridgeServer::handleFlattenStraighten(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!state->currentVolume())
        throw AgentBridgeError{-32001, "No volume loaded", {}};

    const QString segmentId = p.value("segmentId").toString();
    if (segmentId.isEmpty()) {
        QJsonObject data;
        data["param"] = "segmentId";
        throw AgentBridgeError{-32602, "segmentId is required", data};
    }

    SegmentationCommandHandler::StraightenParams stp;  // defaults mirror the dialog

    stp.unbend = p.value("unbend").toBool(stp.unbend);
    stp.unbendSmoothCols =
        p.value("unbendSmoothCols").toDouble(stp.unbendSmoothCols);
    stp.overlapPasses = p.value("overlapPasses").toInt(stp.overlapPasses);
    stp.orthogonalize =
        p.value("orthogonalize").toBool(stp.orthogonalize);
    stp.trim = p.value("trim").toBool(stp.trim);
    stp.trimMaxEdge = p.value("trimMaxEdge").toDouble(stp.trimMaxEdge);
    stp.outputDir = p.value("outputDir").toString();

    SegmentationCommandHandler* handler =
        _window ? _window->_segmentationCommandHandler.get() : nullptr;
    return launchFlattenJob(
        QStringLiteral("flatten.straighten"), QStringLiteral("Straighten"), segmentId,
        [handler, segmentId, stp](CommandLaunchError* error, QString* outDir) {
            return handler && handler->startStraighten(
                segmentId.toStdString(), stp, error, outDir);
        });
}
