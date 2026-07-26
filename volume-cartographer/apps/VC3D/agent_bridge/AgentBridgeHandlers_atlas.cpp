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
// Atlas RPCs
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleAtlasOpen(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    std::shared_ptr<VolumePkg> vpkg = state ? state->vpkg() : nullptr;
    if (!state || !state->hasVpkg() || !vpkg)
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QJsonObject p = params.toObject();
    const QString atlasDirStr = p.value("atlasDir").toString();
    if (atlasDirStr.isEmpty()) {
        QJsonObject data;
        data["param"] = "atlasDir";
        throw AgentBridgeError{-32602, "atlasDir is required", data};
    }

    std::filesystem::path dir(atlasDirStr.toStdString());
    if (dir.is_relative()) {
        // Match CWindow::loadAndDisplayAtlas by resolving relative paths
        // against the volpkg root.
        const std::filesystem::path volpkgRoot = vpkg->path().empty()
            ? std::filesystem::path(vpkg->getVolpkgDirectory())
            : vpkg->path().parent_path();
        dir = volpkgRoot / dir;
    }
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) {
        QJsonObject data;
        data["kind"] = "atlas";
        data["path"] = QString::fromStdString(dir.string());
        throw AgentBridgeError{-32007, "Atlas directory not found", data};
    }

    // The headless path never opens the rebuild prompt or a
    // warning dialog — failures come back as the exception text.
    QString err;
    if (!_window->displayAtlasFromDirectoryHeadless(dir, &err)) {
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "Atlas load failed", data};
    }

    QJsonObject result;
    result["opened"] = true;
    result["atlasDir"] = _window->_currentAtlasDir
        ? QString::fromStdString(_window->_currentAtlasDir->string())
        : QString::fromStdString(dir.string());
    result["atlasName"] = QString::fromStdString(_window->_currentAtlasName);
    return result;
}

QJsonObject AgentBridgeServer::handleAtlasStatus(const QJsonValue&)
{
    QJsonObject result;
    if (_window && _window->_currentAtlasDir) {
        result["atlasDir"] = QString::fromStdString(_window->_currentAtlasDir->string());
        result["atlasName"] = QString::fromStdString(_window->_currentAtlasName);
    } else {
        result["atlasDir"] = QJsonValue::Null;
        result["atlasName"] = QJsonValue::Null;
    }

    QJsonObject search;
    search["running"] = jobIsRunning(QStringLiteral("atlas"));
    // 1-based phase numbering, mirroring atlasSearchPhaseNumber (CWindow.cpp).
    search["phase"] = _window
        ? static_cast<int>(_window->_atlasSearchProgressPhase) + 1 : 1;
    search["phaseCount"] = 5;  // ATLAS_SEARCH_PHASE_COUNT (CWindow.cpp)
    search["completed"] = _window
        ? static_cast<double>(_window->_atlasSearchPhaseCompleted) : 0.0;
    search["total"] = _window
        ? static_cast<double>(_window->_atlasSearchPhaseTotal) : 0.0;
    search["cancelRequested"] = _window ? _window->_atlasSearchCancelRequested : false;
    search["resultCount"] = _window
        ? static_cast<double>(_window->_atlasSearchResults.size()) : 0.0;
    result["search"] = search;
    return result;
}


QJsonObject AgentBridgeServer::handleAtlasSearchStart(const QJsonValue& params)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    const QJsonObject p = params.toObject();

    AtlasFiberSearchParams sp;
    const QString modeStr =
        p.value("mode").toString(QStringLiteral("atlas_to_non_atlas"));
    if (modeStr == QLatin1String("atlas_to_non_atlas")) {
        sp.searchMode = 0;  // ATLAS_SEARCH_MODE_ATLAS_TO_NON_ATLAS (CWindow.cpp)
    } else {
        sp.searchMode = 1;  // ATLAS_SEARCH_MODE_NON_ATLAS_ONLY (CWindow.cpp)
    }

    auto readTags = [](const QJsonObject& obj, const char* key) -> QStringList {
        QStringList tags;
        const QJsonValue v = obj.value(QLatin1String(key));
        if (v.isUndefined() || v.isNull())
            return tags;
        for (const QJsonValue& tv : v.toArray()) {
            const QString s = tv.toString();
            if (!s.isEmpty())
                tags.append(s);
        }
        return tags;
    };
    sp.requiredTags = readTags(p, "requiredTags");
    sp.excludedTags = readTags(p, "excludedTags");

    if (p.contains("maxDistance")) {
        sp.maxDistance = p.value("maxDistance").toDouble();
    }
    // else: keep the current spin-box value — the headless
    // launcher reads the persisted QSettings key.

    requireSourceIdle(QStringLiteral("atlas"));

    if (sp.searchMode == 0 && !_window->_currentAtlasDir) {
        QJsonObject data;
        data["kind"] = "atlas";
        data["detail"] =
            "no atlas is open; call atlas.open first or use mode \"non_atlas_only\"";
        throw AgentBridgeError{-32007, "No atlas open", data};
    }

    QString err;
    if (!_window->startAtlasFiberIntersectionSearchHeadless(sp, &err)) {
        QJsonObject data;
        data["detail"] = err;
        // -32007 for the atlas-shaped preconditions; everything
        // else (no fibers / no lasagna dataset / already running) is -32005.
        if (err.contains(QLatin1String("no saved fiber mappings"), Qt::CaseInsensitive) ||
            err.contains(QLatin1String("Select an atlas"), Qt::CaseInsensitive)) {
            data["kind"] = "atlas";
            throw AgentBridgeError{-32007, "Atlas search preconditions not met", data};
        }
        throw AgentBridgeError{-32005, "Atlas search failed to start", data};
    }

    const QString jobId = beginJob(QStringLiteral("atlas"),
                                   QStringLiteral("atlas.fiber_search"),
                                   QStringLiteral("Atlas fiber search started"),
                                   /*broadcastStart=*/true);

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "atlas.fiber_search";
    result["source"] = "atlas";
    return result;
}


QJsonObject AgentBridgeServer::handleAtlasSearchCancel(const QJsonValue&)
{
    if (!_window || !jobIsRunning(QStringLiteral("atlas"))) {
        QJsonObject data;
        data["kind"] = "job";
        throw AgentBridgeError{-32007, "No atlas search running", data};
    }
    // Dialog-free: sets the cancel flags and updates the progress bars. The
    // job still terminates through atlasSearchFinished (success=false).
    _window->cancelAtlasFiberIntersectionSearch();
    QJsonObject result;
    result["cancelRequested"] = true;
    return result;
}


QJsonObject AgentBridgeServer::handleAtlasSearchResults(const QJsonValue& params)
{
    if (!_window)
        throw AgentBridgeError{-32010, "Window unavailable", {}};

    const QJsonObject p = params.toObject();
    const int offset = p.value("offset").toInt(0);
    const int limit = std::min(p.value("limit").toInt(100), 1000);

    const auto& results = _window->_atlasSearchResults;
    const auto& signedWindings = _window->_atlasSearchSignedWindings;
    const int total = static_cast<int>(results.size());

    auto vec3dJson = [](const cv::Vec3d& v) {
        QJsonObject o;
        o["x"] = v[0];
        o["y"] = v[1];
        o["z"] = v[2];
        return o;
    };

    QJsonArray rows;
    for (int i = offset; i < total && i < offset + limit; ++i) {
        const auto& r = results[static_cast<size_t>(i)];
        QJsonObject row;
        row["index"] = i;  // vector order; the id atlas.open_result takes
        row["sourceFiberId"] = QString::number(r.sourceFiberId);
        row["targetFiberId"] = QString::number(r.targetFiberId);
        row["candidateDistance"] = r.candidateDistance;
        row["refinedScore"] = r.refinedScore;
        row["windingDistance"] = std::isfinite(r.windingDistance)
            ? QJsonValue(r.windingDistance)
            : QJsonValue(QJsonValue::Null);
        const bool haveSigned = static_cast<size_t>(i) < signedWindings.size() &&
                                std::isfinite(signedWindings[static_cast<size_t>(i)]);
        row["signedWinding"] = haveSigned
            ? QJsonValue(signedWindings[static_cast<size_t>(i)])
            : QJsonValue(QJsonValue::Null);
        row["sourcePoint"] = vec3dJson(r.sourcePoint);
        row["targetPoint"] = vec3dJson(r.targetPoint);
        row["sourceArclength"] = r.sourceArclength;
        row["targetArclength"] = r.targetArclength;
        row["converged"] = r.converged;
        row["message"] = QString::fromStdString(r.message);
        rows.push_back(row);
    }

    QJsonObject result;
    result["total"] = total;
    result["offset"] = offset;
    result["results"] = rows;
    return result;
}


QJsonObject AgentBridgeServer::handleAtlasOpenResult(const QJsonValue& params)
{
    if (!_window)
        throw AgentBridgeError{-32010, "Window unavailable", {}};

    const QJsonObject p = params.toObject();
    const int index = p.value("index").toInt();
    const auto& results = _window->_atlasSearchResults;
    if (index < 0 || index >= static_cast<int>(results.size())) {
        QJsonObject data;
        data["kind"] = "result";
        data["index"] = index;
        data["total"] = static_cast<int>(results.size());
        throw AgentBridgeError{-32007, "Atlas search result index out of range", data};
    }

    // Mirror the interactive slot's preconditions (CWindow::openAtlasSearchResult)
    // without its QMessageBoxes.
    if (!_window->_lineAnnotationController || !_window->_intersectionsMdiArea) {
        QJsonObject data;
        data["detail"] = "intersections workspace is not available";
        throw AgentBridgeError{-32005, "Intersections workspace unavailable", data};
    }
    if (!_window->_currentAtlasDir) {
        QJsonObject data;
        data["kind"] = "atlas";
        throw AgentBridgeError{-32007, "No atlas open", data};
    }

    if (_window->_workspaceTabs && _window->_intersectionsWorkspaceWindow)
        _window->_workspaceTabs->setCurrentWidget(_window->_intersectionsWorkspaceWindow);

    QString err;
    if (!_window->_lineAnnotationController->showIntersectionInspectionHeadless(
            results[static_cast<size_t>(index)],
            _window->_intersectionsMdiArea,
            _window->_currentAtlasDir,
            &err)) {
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "Could not open atlas search result", data};
    }

    QJsonObject result;
    result["opened"] = true;
    result["index"] = index;
    return result;
}


QJsonObject AgentBridgeServer::handleAtlasRemap(const QJsonValue&)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!_window->_currentAtlasDir) {
        QJsonObject data;
        data["kind"] = "atlas";
        throw AgentBridgeError{-32007, "No atlas open", data};
    }

    QString err;
    if (!_window->startAtlasRemapHeadless(&err)) {
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "Atlas remap failed to start", data};
    }

    QJsonObject result;
    result["remapped"] = true;
    return result;
}


QJsonObject AgentBridgeServer::handleAtlasOptimizeSnapCandidates(const QJsonValue&)
{
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->hasVpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!_window->_currentAtlasDir) {
        QJsonObject data;
        data["kind"] = "atlas";
        throw AgentBridgeError{-32007, "No atlas open", data};
    }

    QString err;
    if (!_window->optimizeAtlasSnapCandidatesHeadless(&err)) {
        QJsonObject data;
        data["detail"] = err;
        throw AgentBridgeError{-32005, "Snap-candidate ranking failed to start", data};
    }

    QJsonObject result;
    result["requested"] = true;
    return result;
}
