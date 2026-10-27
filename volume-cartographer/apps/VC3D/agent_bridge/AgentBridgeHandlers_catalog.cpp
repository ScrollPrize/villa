#include "agent_bridge/AgentBridgeServer.hpp"

#include <QDateTime>
#include <QFutureWatcher>
#include <QJsonArray>
#include <QtConcurrent>

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "CWindow.hpp"
#include "MenuActionController.hpp"
#include "OpenDataManifest.hpp"

namespace {

QString representationKindToJson(vcr::catalog::OpenDataRepresentationKind kind)
{
    switch (kind) {
    case vcr::catalog::OpenDataRepresentationKind::NormalGrids: return QStringLiteral("normal_grids");
    case vcr::catalog::OpenDataRepresentationKind::Lasagna:     return QStringLiteral("lasagna");
    case vcr::catalog::OpenDataRepresentationKind::Prediction:  return QStringLiteral("prediction");
    }
    return QStringLiteral("prediction");
}
std::optional<vcr::catalog::OpenDataRepresentationKind>
representationKindFromJson(const QString& s)
{
    if (s == QLatin1String("normal_grids"))
        return vcr::catalog::OpenDataRepresentationKind::NormalGrids;
    if (s == QLatin1String("lasagna"))
        return vcr::catalog::OpenDataRepresentationKind::Lasagna;
    if (s == QLatin1String("prediction"))
        return vcr::catalog::OpenDataRepresentationKind::Prediction;
    return std::nullopt;
}

} // namespace

QJsonObject AgentBridgeServer::handleCatalogOpenSample(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    const QString sampleId = p.value("sampleId").toString();
    if (sampleId.isEmpty()) {
        QJsonObject data;
        data["param"] = "sampleId";
        throw AgentBridgeError{-32602, "sampleId is required", data};
    }
    if (!_window) {
        QJsonObject data;
        data["detail"] = "window is not available";
        throw AgentBridgeError{-32010, "Internal error", data};
    }

    // Resolve the sample first so we can distinguish "unknown sample" (-32007)
    // from "manifest unavailable" / "open failed" (-32005).
    const auto manifest = vc3d::opendata::cachedCatalog();
    if (!manifest) {
        QJsonObject data;
        data["detail"] = "Open Data manifest is unavailable";
        throw AgentBridgeError{-32005, "Manifest fetch failed", data};
    }
    const vcr::catalog::OpenDataSample* sample =
        manifest->findSample(sampleId.toStdString());
    if (!sample) {
        QJsonObject data;
        data["kind"] = "sample";
        data["id"] = sampleId;
        throw AgentBridgeError{-32007, QStringLiteral("Unknown sample: %1").arg(sampleId), data};
    }

    MenuActionController* mc = _window->_menuController.get();
    if (!mc) {
        QJsonObject data;
        data["detail"] = "menu controller is not available";
        throw AgentBridgeError{-32010, "Internal error", data};
    }

    // --- Optional resource selection. Validated in full against
    // the resolved sample *before* any download or project mutation begins. ---
    OpenDataSampleOpenOptions options;
    // An explicit remote call is consent to replace the current project.
    options.interactive = false;

    if (p.contains("resources")) {
        const QJsonObject res = p.value("resources").toObject();
        auto& selection = options.selection;

        // Set of the sample's real volume ids (validation + zero-volume check).
        std::vector<std::string> sampleVolumeIds;
        for (const auto& v : sample->volumes)
            sampleVolumeIds.push_back(v.id);

        // volumeIds axis.
        if (res.contains("volumeIds")) {
            const QJsonValue vidsv = res.value("volumeIds");
            std::vector<std::string> vids;
            for (const QJsonValue& vv : vidsv.toArray()) {
                const std::string vid = vv.toString().toStdString();
                if (std::find(sampleVolumeIds.begin(), sampleVolumeIds.end(), vid) ==
                    sampleVolumeIds.end()) {
                    QJsonObject data;
                    data["kind"] = "resource";
                    data["id"] = QString::fromStdString(vid);
                    throw AgentBridgeError{-32007,
                        QStringLiteral("Unknown volumeId: %1").arg(QString::fromStdString(vid)),
                        data};
                }
                vids.push_back(vid);
            }
            // A selection that leaves zero volumes is rejected.
            if (vids.empty()) {
                QJsonObject data;
                data["param"] = "resources.volumeIds";
                throw AgentBridgeError{-32602,
                    "resources.volumeIds selects zero volumes", data};
            }
            selection.volumeIds = std::move(vids);
        }

        // representationRefs axis. Each "vi:ai" must name a real derived
        // representation of the sample (from vcr::catalog::derivedRepresentations()).
        if (res.contains("representationRefs")) {
            const QJsonValue refsv = res.value("representationRefs");
            const auto derived = vcr::catalog::derivedRepresentations(*sample);
            std::vector<vcr::catalog::OpenDataRepresentationRef> refs;
            for (const QJsonValue& rv : refsv.toArray()) {
                const QString refStr = rv.toString();
                const auto reject = [&]() {
                    QJsonObject data;
                    data["kind"] = "resource";
                    data["id"] = refStr;
                    throw AgentBridgeError{-32007,
                        QStringLiteral("Invalid representation ref: %1").arg(refStr), data};
                };
                const int colon = refStr.indexOf(QLatin1Char(':'));
                if (colon <= 0 || colon == refStr.size() - 1)
                    reject();
                bool okVi = false, okAi = false;
                const qulonglong vi = refStr.left(colon).toULongLong(&okVi);
                const qulonglong ai = refStr.mid(colon + 1).toULongLong(&okAi);
                if (!okVi || !okAi)
                    reject();
                const auto match = std::find_if(derived.begin(), derived.end(),
                    [&](const vcr::catalog::OpenDataRepresentationRef& r) {
                        return r.volumeIndex == vi && r.artifactIndex == ai;
                    });
                if (match == derived.end())
                    reject();
                refs.push_back(*match);
            }
            selection.representations = std::move(refs);
        }

        // kinds axis.
        if (res.contains("kinds")) {
            const QJsonValue kindsv = res.value("kinds");
            std::vector<vcr::catalog::OpenDataRepresentationKind> kinds;
            for (const QJsonValue& kv : kindsv.toArray()) {
                const QString ks = kv.toString();
                const auto kind = representationKindFromJson(ks);
                if (!kind) {
                    QJsonObject data;
                    data["kind"] = "resource";
                    data["id"] = ks;
                    throw AgentBridgeError{-32007,
                        QStringLiteral("Unknown resource kind: %1").arg(ks), data};
                }
                kinds.push_back(*kind);
            }
            selection.kinds = std::move(kinds);
        }
    }

    // Async open: the RPC no longer blocks on the network. Reject a
    // second catalog open (bridge-started or human-initiated) up front, then
    // start the dialog-free asynchronous core and return a jobId immediately.
    requireSourceIdle(QStringLiteral("catalog"));
    if (mc->openDataSampleOpenInFlight()) {
        QJsonObject data;
        data["source"] = "catalog";
        data["detail"] = "an interactive Open Data open is in progress";
        throw AgentBridgeError{-32004,
            "An Open Data sample open is already in progress", data};
    }

    // Progress: forward the open's stages as job.progress "output"
    // notifications, rate-limited to <=10/s.
    auto onProgress =
        [this](const vc3d::opendata::OpenDataSampleDownloadProgress& progress) {
            auto it = _activeJobs.find(QStringLiteral("catalog"));
            if (it == _activeJobs.end())
                return;
            const qint64 now = QDateTime::currentMSecsSinceEpoch();
            if (now - _lastConsoleBroadcastMs < 100)
                return;
            _lastConsoleBroadcastMs = now;
            const QString label = progress.status == "project-ready"
                ? QStringLiteral("Open-data project is ready.")
                : QStringLiteral("Opening remote volumes in parallel...");
            broadcastJobProgress(it.value(), QStringLiteral("output"), label);
        };

    auto onFinished =
        [this](const MenuActionController::OpenDataSampleOpenOutcome& outcome) {
            completeCatalogOpenJob(outcome);
        };

    _catalogOpenSampleId = sampleId;
    QString err;
    if (!mc->startOpenDataSampleOpen(sampleId, options, onFinished, onProgress, &err)) {
        _catalogOpenSampleId.clear();
        QJsonObject data;
        data["detail"] = err.isEmpty()
            ? QStringLiteral("failed to open open-data sample %1").arg(sampleId)
            : err;
        throw AgentBridgeError{-32005, "Open Data sample open failed", data};
    }

    const QString jobId = beginJob(QStringLiteral("catalog"),
                                   QStringLiteral("catalog.open_sample"),
                                   QStringLiteral("open sample %1").arg(sampleId),
                                   /*broadcastStart=*/true);

    QJsonObject result;
    result["jobId"] = jobId;
    result["kind"] = "catalog.open_sample";
    result["source"] = "catalog";
    result["sampleId"] = sampleId;
    return result;
}


// ---------------------------------------------------------------------------
// Remote catalog resource selection
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::withOpenDataManifest(
    bool refresh,
    const std::function<QJsonObject(const vcr::catalog::OpenDataManifest&)>& build)
{
    if (!_window) {
        QJsonObject data;
        data["detail"] = "window is not available";
        throw AgentBridgeError{-32010, "Internal error", data};
    }

    if (!refresh) {
        if (const auto cached = vc3d::opendata::cachedCatalog()) {
            return build(*cached);  // AgentBridgeError from build propagates to dispatch.
        }
    }

    // No cached manifest (or a forced refresh): fetch off-thread and reply via
    // the deferred mechanism, with a 30-second deadline.
    const int token = beginDeferred(30000, "Open Data manifest fetch");
    startManifestFetch(token, build);
    throw AgentBridgeDeferred{};
}


void AgentBridgeServer::startManifestFetch(
    int token,
    const std::function<QJsonObject(const vcr::catalog::OpenDataManifest&)>& build)
{
    struct FetchResult {
        std::optional<vcr::catalog::OpenDataManifest> manifest;
        QString error;
    };

    auto* watcher = new QFutureWatcher<FetchResult>(this);
    connect(watcher, &QFutureWatcher<FetchResult>::finished, this,
            [this, token, build, watcher]() {
                FetchResult fr = watcher->result();
                watcher->deleteLater();
                if (!fr.manifest) {
                    QJsonObject data;
                    data["detail"] = fr.error.isEmpty()
                        ? QStringLiteral("Open Data manifest fetch failed")
                        : fr.error;
                    completeDeferredError(token, -32005, "Manifest fetch failed", data);
                    return;
                }
                try {
                    completeDeferredResult(token, build(*fr.manifest));
                } catch (const AgentBridgeError& e) {
                    completeDeferredError(token, e.code, e.message, e.data);
                } catch (const std::exception& e) {
                    QJsonObject data;
                    data["detail"] = QString::fromUtf8(e.what());
                    completeDeferredError(token, -32010, "Internal error", data);
                }
            });

    watcher->setFuture(QtConcurrent::run([]() {
        FetchResult fr;
        try {
            fr.manifest = vcr::catalog::loadCatalog(std::string(vcr::catalog::kDefaultManifestUrl),
                {vc3d::opendata::cachedOpenDataManifestPath(), true}).manifest;
        } catch (const std::exception& e) {
            fr.error = QString::fromUtf8(e.what());
        } catch (...) {
            fr.error = QStringLiteral("unknown error fetching Open Data manifest");
        }
        return fr;
    }));
}


QJsonObject AgentBridgeServer::handleCatalogListSamples(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    const bool refresh = p.value("refresh").toBool(false);

    auto build = [](const vcr::catalog::OpenDataManifest& manifest) -> QJsonObject {
        QJsonObject result;
        result["manifestUrl"] = QString::fromStdString(manifest.manifestUrl);
        QJsonArray samples;
        for (const auto& s : manifest.samples) {
            QJsonObject o;
            o["id"] = QString::fromStdString(s.id);
            o["type"] = QString::fromStdString(s.type);
            o["description"] = QString::fromStdString(s.description);
            o["volumeCount"] = static_cast<int>(s.volumeCount());
            o["segmentCount"] = static_cast<int>(s.segmentCount());
            o["scanCount"] = static_cast<int>(s.scanCount());
            samples.append(o);
        }
        result["samples"] = samples;
        return result;
    };

    return withOpenDataManifest(refresh, build);
}


QJsonObject AgentBridgeServer::handleCatalogDescribeSample(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    const QString sampleId = p.value("sampleId").toString();
    if (sampleId.isEmpty()) {
        QJsonObject data;
        data["param"] = "sampleId";
        throw AgentBridgeError{-32602, "sampleId is required", data};
    }
    const bool refresh = p.value("refresh").toBool(false);
    const std::string sid = sampleId.toStdString();

    auto build = [sampleId, sid](const vcr::catalog::OpenDataManifest& manifest) -> QJsonObject {
        const vcr::catalog::OpenDataSample* sample = manifest.findSample(sid);
        if (!sample) {
            QJsonObject data;
            data["kind"] = "sample";
            data["id"] = sampleId;
            throw AgentBridgeError{-32007,
                QStringLiteral("Unknown sample: %1").arg(sampleId), data};
        }

        QJsonObject result;
        result["sampleId"] = QString::fromStdString(sample->id);
        result["type"] = QString::fromStdString(sample->type);
        result["description"] = QString::fromStdString(sample->description);
        result["segmentCount"] = static_cast<int>(sample->segmentCount());

        QJsonArray volumes;
        for (const auto& v : sample->volumes) {
            QJsonObject o;
            o["id"] = QString::fromStdString(v.id);
            o["scanId"] = QString::fromStdString(v.scanId);
            if (v.shapeZYX) {
                QJsonArray shape;
                shape.append(static_cast<double>((*v.shapeZYX)[0]));
                shape.append(static_cast<double>((*v.shapeZYX)[1]));
                shape.append(static_cast<double>((*v.shapeZYX)[2]));
                o["shapeZYX"] = shape;
            } else {
                o["shapeZYX"] = QJsonValue::Null;
            }
            o["pixelSizeUm"] = v.pixelSizeUm ? QJsonValue(*v.pixelSizeUm)
                                             : QJsonValue(QJsonValue::Null);
            o["dataFormat"] = QString::fromStdString(v.dataFormat);
            volumes.append(o);
        }
        result["volumes"] = volumes;

        QJsonArray representations;
        for (const auto& ref : vcr::catalog::derivedRepresentations(*sample)) {
            const auto& volume = sample->volumes[ref.volumeIndex];
            const auto& artifact = volume.artifacts[ref.artifactIndex];
            QJsonObject o;
            o["ref"] = QStringLiteral("%1:%2")
                           .arg(static_cast<qulonglong>(ref.volumeIndex))
                           .arg(static_cast<qulonglong>(ref.artifactIndex));
            o["volumeId"] = QString::fromStdString(volume.id);
            o["artifactType"] = QString::fromStdString(artifact.type);
            o["kind"] = representationKindToJson(ref.kind);
            o["url"] = artifact.hasResolvedUrl()
                           ? QJsonValue(QString::fromStdString(artifact.url()))
                           : QJsonValue(QJsonValue::Null);
            o["targetVolumeId"] = artifact.targetVolumeId
                ? QJsonValue(QString::fromStdString(*artifact.targetVolumeId))
                : QJsonValue(QJsonValue::Null);
            o["modelId"] = artifact.modelId
                ? QJsonValue(QString::fromStdString(*artifact.modelId))
                : QJsonValue(QJsonValue::Null);
            representations.append(o);
        }
        result["representations"] = representations;

        return result;
    };

    return withOpenDataManifest(refresh, build);
}
