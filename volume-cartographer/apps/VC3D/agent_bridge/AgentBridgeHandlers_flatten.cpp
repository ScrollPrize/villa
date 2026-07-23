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
#include <cmath>
#include <filesystem>
#include <limits>
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
// Flattening RPCs (SPEC §20): flatten.slim / flatten.abf / flatten.straighten
// ---------------------------------------------------------------------------

// Shared launch body for all three flatten RPCs: the handler builds `launch`
// (a closure over the concrete start* launcher); this registers the job and
// maps failure sentences to codes. Completion is driven by
// SegmentationCommandHandler::flattenJobFinished rather than a
// CommandLineToolRunner signal, since flatten jobs own their own
// QProcess/QtConcurrent lifecycle.
QJsonObject AgentBridgeServer::launchFlattenJob(
    const QString& kind, const QString& label, const QString& segmentId,
    const std::function<bool(QString* err, QString* outDir)>& launch)
{
    // Only one flatten at a time (its own source, so it may run concurrently
    // with a "tool"/"growth"/etc. job, §8.3).
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

    QString err;
    QString outputDir;
    if (!launch(&err, &outputDir)) {
        _activeJobs.remove(QStringLiteral("flatten"));
        // Map the distinct failure sentences the start* launchers produce (§20).
        if (err.contains(QLatin1String("Invalid segment"))) {
            QJsonObject data;
            data["kind"] = "segment";
            data["id"] = segmentId;
            data["detail"] = err;
            throw AgentBridgeError{-32007, "Segment not found", data};
        }
        if (err.contains(QLatin1String("not found or not executable"))) {
            QJsonObject data;
            data["detail"] = err;
            throw AgentBridgeError{-32006, "Flatten tool unavailable", data};
        }
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "Failed to start flatten", data};
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
    const QJsonObject p = paramsObject(params);
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

    SegmentationCommandHandler::SlimFlattenParams sp;  // headless defaults (§20)

    if (p.contains("iterations")) {
        // jsonRequireInt rejects a fractional value (e.g. 1.5) that toInt() would
        // silently truncate, plus wrong types and int overflow.
        sp.iterations = jsonRequireInt(p.value("iterations"), "iterations");
        if (sp.iterations < 1) {
            QJsonObject data; data["param"] = "iterations";
            throw AgentBridgeError{-32602, "iterations must be >= 1", data};
        }
    }
    if (p.contains("tolerance")) {
        if (!p.value("tolerance").isDouble()) {
            QJsonObject data; data["param"] = "tolerance";
            throw AgentBridgeError{-32602, "tolerance must be a number", data};
        }
        sp.tolerance = p.value("tolerance").toDouble();
        if (!std::isfinite(sp.tolerance) || sp.tolerance < 0.0) {
            QJsonObject data; data["param"] = "tolerance";
            throw AgentBridgeError{-32602, "tolerance must be a finite value >= 0", data};
        }
    }
    if (p.contains("energyType") && !p.value("energyType").isNull()) {
        const QString e = p.value("energyType").toString();
        if (e != QLatin1String("symmetric_dirichlet") && e != QLatin1String("conformal")) {
            QJsonObject data; data["param"] = "energyType";
            data["detail"] = "energyType must be \"symmetric_dirichlet\" or \"conformal\"";
            throw AgentBridgeError{-32602, "Invalid energyType", data};
        }
        sp.energyType = e;
    }
    if (p.contains("keepPercent")) {
        if (!p.value("keepPercent").isDouble()) {
            QJsonObject data; data["param"] = "keepPercent";
            throw AgentBridgeError{-32602, "keepPercent must be a number", data};
        }
        sp.keepPercent = p.value("keepPercent").toDouble();
        if (!std::isfinite(sp.keepPercent) || sp.keepPercent <= 0.0 || sp.keepPercent > 100.0) {
            QJsonObject data; data["param"] = "keepPercent";
            throw AgentBridgeError{-32602, "keepPercent must be in (0, 100]", data};
        }
    }
    if (p.contains("inpaintHoles")) {
        if (!p.value("inpaintHoles").isBool()) {
            QJsonObject data; data["param"] = "inpaintHoles";
            throw AgentBridgeError{-32602, "inpaintHoles must be a boolean", data};
        }
        sp.inpaintHoles = p.value("inpaintHoles").toBool();
    }
    sp.outputDir = jsonOptionalString(p, "outputDir");

    SegmentationCommandHandler* handler =
        _window ? _window->_segmentationCommandHandler.get() : nullptr;
    return launchFlattenJob(
        QStringLiteral("flatten.slim"), QStringLiteral("SLIM flatten"), segmentId,
        [handler, segmentId, sp](QString* err, QString* outDir) {
            return handler && handler->startSlimFlatten(segmentId.toStdString(), sp, err, outDir);
        });
}


QJsonObject AgentBridgeServer::handleFlattenAbf(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QString segmentId = p.value("segmentId").toString();
    if (segmentId.isEmpty()) {
        QJsonObject data;
        data["param"] = "segmentId";
        throw AgentBridgeError{-32602, "segmentId is required", data};
    }

    int iterations = 10;        // ABFFlattenDialog session default
    int downsampleFactor = 1;
    if (p.contains("iterations")) {
        iterations = jsonRequireInt(p.value("iterations"), "iterations");
        if (iterations < 1) {
            QJsonObject data; data["param"] = "iterations";
            throw AgentBridgeError{-32602, "iterations must be >= 1", data};
        }
    }
    if (p.contains("downsampleFactor")) {
        downsampleFactor = jsonRequireInt(p.value("downsampleFactor"), "downsampleFactor");
        if (downsampleFactor < 1) {
            QJsonObject data; data["param"] = "downsampleFactor";
            throw AgentBridgeError{-32602, "downsampleFactor must be >= 1", data};
        }
    }

    SegmentationCommandHandler* handler =
        _window ? _window->_segmentationCommandHandler.get() : nullptr;
    return launchFlattenJob(
        QStringLiteral("flatten.abf"), QStringLiteral("ABF++ flatten"), segmentId,
        [handler, segmentId, iterations, downsampleFactor](QString* err, QString* outDir) {
            return handler && handler->startAbfFlatten(segmentId.toStdString(),
                                                       iterations, downsampleFactor, err, outDir);
        });
}


QJsonObject AgentBridgeServer::handleFlattenStraighten(const QJsonValue& params)
{
    const QJsonObject p = paramsObject(params);
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

    if (p.contains("unbend")) {
        if (!p.value("unbend").isBool()) {
            QJsonObject data; data["param"] = "unbend";
            throw AgentBridgeError{-32602, "unbend must be a boolean", data};
        }
        stp.unbend = p.value("unbend").toBool();
    }
    if (p.contains("unbendSmoothCols")) {
        if (!p.value("unbendSmoothCols").isDouble()) {
            QJsonObject data; data["param"] = "unbendSmoothCols";
            throw AgentBridgeError{-32602, "unbendSmoothCols must be a number", data};
        }
        stp.unbendSmoothCols = p.value("unbendSmoothCols").toDouble();
        if (!std::isfinite(stp.unbendSmoothCols) || stp.unbendSmoothCols < 0.0) {
            QJsonObject data; data["param"] = "unbendSmoothCols";
            throw AgentBridgeError{-32602, "unbendSmoothCols must be a finite value >= 0", data};
        }
    }
    if (p.contains("overlapPasses")) {
        // jsonRequireInt rejects fractional (e.g. 1.5) / wrong-typed / overflowing
        // values that toInt() would otherwise silently accept.
        stp.overlapPasses = jsonRequireInt(p.value("overlapPasses"), "overlapPasses");
        if (stp.overlapPasses < 0) {
            QJsonObject data; data["param"] = "overlapPasses";
            throw AgentBridgeError{-32602, "overlapPasses must be >= 0", data};
        }
    }
    if (p.contains("orthogonalize")) {
        if (!p.value("orthogonalize").isBool()) {
            QJsonObject data; data["param"] = "orthogonalize";
            throw AgentBridgeError{-32602, "orthogonalize must be a boolean", data};
        }
        stp.orthogonalize = p.value("orthogonalize").toBool();
    }
    if (p.contains("trim")) {
        if (!p.value("trim").isBool()) {
            QJsonObject data; data["param"] = "trim";
            throw AgentBridgeError{-32602, "trim must be a boolean", data};
        }
        stp.trim = p.value("trim").toBool();
    }
    if (p.contains("trimMaxEdge")) {
        if (!p.value("trimMaxEdge").isDouble()) {
            QJsonObject data; data["param"] = "trimMaxEdge";
            throw AgentBridgeError{-32602, "trimMaxEdge must be a number", data};
        }
        stp.trimMaxEdge = p.value("trimMaxEdge").toDouble();
        if (!std::isfinite(stp.trimMaxEdge) || stp.trimMaxEdge < 0.0) {
            QJsonObject data; data["param"] = "trimMaxEdge";
            throw AgentBridgeError{-32602, "trimMaxEdge must be a finite value >= 0", data};
        }
    }
    stp.outputDir = jsonOptionalString(p, "outputDir");

    SegmentationCommandHandler* handler =
        _window ? _window->_segmentationCommandHandler.get() : nullptr;
    return launchFlattenJob(
        QStringLiteral("flatten.straighten"), QStringLiteral("Straighten"), segmentId,
        [handler, segmentId, stp](QString* err, QString* outDir) {
            return handler && handler->startStraighten(segmentId.toStdString(), stp, err, outDir);
        });
}
