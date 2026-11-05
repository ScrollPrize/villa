#include "agent_bridge/AgentBridgeServer.hpp"
#include "agent_bridge/AgentBridgeInternal.hpp"

#include <QFutureWatcher>
#include <QJsonArray>
#include <QPointer>
#include <QtConcurrent>

#include <algorithm>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "CState.hpp"
#include "CWindow.hpp"
#include "LineAnnotationController.hpp"
#include "StoreImportController.hpp"
#include "segmentation/SegmentationModule.hpp"

#include "vc/core/util/QuadSurface.hpp"
#include "vc/ui/VCCollection.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/GeometryExport.hpp>

// Object stores of the open project: importing source files into them and
// exporting their objects as native files.

namespace {

CState& requireProject(CState* state)
{
    if (!state || !state->vpkg())
        throw AgentBridgeError{-32000, "No project open", {}};
    return *state;
}

QJsonArray stringsJson(const std::vector<std::string>& values)
{
    QJsonArray result;
    for (const auto& value : values)
        result.append(QString::fromStdString(value));
    return result;
}

QJsonObject importReportJson(const vc::objects::GeometryIngestReport& report, const std::string& store)
{
    return {
        {"store", QString::fromStdString(store)},
        {"ingested", static_cast<double>(report.ingested)},
        {"edgesImported", static_cast<double>(report.edgesImported)},
        {"problems", stringsJson(report.problems)},
        {"uuids", stringsJson(report.uuids)},
        {"existing", stringsJson(report.existing)},
        {"cancelled", report.cancelled},
        {"fatalError", report.fatalError
             ? QJsonValue(QJsonObject{{"message", QString::fromStdString(*report.fatalError)}})
             : QJsonValue(QJsonValue::Null)},
    };
}

std::filesystem::path absolutePathParam(const QJsonObject& p, const char* name)
{
    const QString value = p.value(QLatin1String(name)).toString();
    const std::filesystem::path path(value.toStdString());
    if (value.trimmed().isEmpty() || !path.is_absolute())
        throwParamError(name, QStringLiteral("must be an absolute path"));
    return path.lexically_normal();
}

struct ExportTarget {
    std::string store;
    vc::objects::AttachmentIdentity attachment;
    std::string kind;
};

struct ExportResult {
    std::string objectRevision;
    std::string geometryRevision;
    QString error;
};

}  // namespace


QJsonObject AgentBridgeServer::handleObjectsImport(const QJsonValue& params)
{
    CState& state = requireProject(_window ? _window->_state : nullptr);
    StoreImportController* importer = _window->_storeImportController.get();
    if (!importer)
        throw AgentBridgeError{-32010, "Internal error", {{"detail", "store import controller is not available"}}};
    requireSourceIdle(QStringLiteral("import"));

    const QJsonObject p = params.toObject();
    StoreImportParams request;
    request.kind = p.value("kind").toString().toStdString();
    request.source = absolutePathParam(p, "path");
    if (p.contains("tags")) {
        std::vector<std::string> tags;
        for (const QJsonValue& tag : p.value("tags").toArray())
            tags.push_back(tag.toString().toStdString());
        request.tags = std::move(tags);
    }
    if (p.contains("bind"))
        request.bindScale = p.value("bind").toObject().value("scale").toDouble();
    if (p.contains("uuidNamespace"))
        request.uuidNamespace = p.value("uuidNamespace").toString().toStdString();
    if (p.contains("store"))
        request.storeReference = p.value("store").toString().toStdString();
    // Bridge imports name every source explicitly and stop at the store's failures.
    request.requireSourceUuids = true;
    request.stopOnInfrastructureError = true;
    request.refuseDuringFiberStructuralEdits = true;

    StoreImportRequest prepared;
    StoreImportFailure failure = StoreImportFailure::None;
    QString error;
    QString param;
    if (!importer->prepare(request, &prepared, &error, &failure, &param)) {
        switch (failure) {
        case StoreImportFailure::NoProject:
            throw AgentBridgeError{-32000, "No project open", {}};
        case StoreImportFailure::InvalidParams:
            throw AgentBridgeError{-32602, error, {{"param", param}}};
        case StoreImportFailure::SourceNotFound:
            throw AgentBridgeError{-32007, error, {{"kind", "path"}, {"path", p.value("path")}}};
        case StoreImportFailure::NoStore:
            throw AgentBridgeError{-32007, error, {{"kind", "store"}, {"id", p.value("store")}}};
        case StoreImportFailure::ReadOnlyStore:
            throw AgentBridgeError{-32009, error, {{"param", "store"}}};
        case StoreImportFailure::Busy:
            throw AgentBridgeError{-32004, error, {{"source", "fiber"}}};
        case StoreImportFailure::None:
            break;
        }
        throw AgentBridgeError{-32005, error, {}};
    }
    const QString store = QString::fromStdString(prepared.storeReference);

    QPointer<AgentBridgeServer> self(this);
    if (!importer->start(
            std::move(prepared),
            [self](const StoreImportProgress& update) {
                if (!self) return;
                auto it = self->_activeJobs.find(QStringLiteral("import"));
                if (it == self->_activeJobs.end()) return;
                it->progress = {
                    {"completed", static_cast<double>(update.completed)},
                    {"total", static_cast<double>(update.total)},
                    {"detail", update.detail},
                };
                self->broadcastJobProgress(*it, QStringLiteral("progress"),
                    QStringLiteral("%1/%2 %3").arg(update.completed).arg(update.total).arg(update.detail));
            },
            [self](const StoreImportOutcome& outcome) {
                if (self) self->completeImportJob(outcome);
            },
            &error)) {
        throw AgentBridgeError{-32004, error, {{"source", "import"}}};
    }
    const QString jobId = beginJob(
        QStringLiteral("import"), QStringLiteral("objects.import"),
        QStringLiteral("import %1 from %2").arg(p.value("kind").toString(), p.value("path").toString()),
        /*broadcastStart=*/true);
    return {
        {"jobId", jobId},
        {"kind", "objects.import"},
        {"source", "import"},
        {"store", store},
    };
}


void AgentBridgeServer::completeImportJob(const StoreImportOutcome& outcome)
{
    auto it = _activeJobs.find(QStringLiteral("import"));
    if (it == _activeJobs.end())
        return;
    const auto& report = outcome.report;
    it->resultJson = importReportJson(report, outcome.storeReference);
    const bool success = !report.fatalError && !report.cancelled;
    QString message;
    if (report.fatalError)
        message = QStringLiteral("Import stopped: %1").arg(QString::fromStdString(*report.fatalError));
    else if (report.cancelled)
        message = QStringLiteral("Import cancelled after %1 object(s)").arg(report.ingested);
    else
        message = QStringLiteral("Imported %1 object(s); %2 problem(s)").arg(report.ingested).arg(report.problems.size());
    finishJob(QStringLiteral("import"), success, message, QString());
}


QJsonObject AgentBridgeServer::handleObjectsExport(const QJsonValue& params)
{
    namespace fs = std::filesystem;
    CState& state = requireProject(_window ? _window->_state : nullptr);
    requireSourceIdle(QStringLiteral("export"));
    const QJsonObject p = params.toObject();

    const std::string requested = p.value("uuid").toString().toStdString();
    if (!vc::objects::util::isUuid(requested))
        throwParamError("uuid", QStringLiteral("must be a UUID"));
    const std::string uuid = vc::objects::util::canonicalUuid(requested);
    const fs::path destination = absolutePathParam(p, "path");
    std::error_code parentError;
    if (!fs::is_directory(destination.parent_path(), parentError))
        throwParamError("path", QStringLiteral("parent directory must exist"));
    const bool overwrite = p.value("overwrite").toBool(false);
    const bool validOnly = p.value("validOnly").toBool(false);

    // A collection still being edited reaches its store first, so it can be found there.
    if (const VCCollection* points = state.pointCollection()) {
        const auto& all = points->getAllCollections();
        if (std::any_of(all.begin(), all.end(), [&](const auto& entry) { return entry.second.uuid == uuid; })) {
            try {
                state.savePointCollections();
            } catch (const std::exception& error) {
                throw AgentBridgeError{-32005, "Pending point-collection edits could not be saved",
                                       {{"detail", QString::fromUtf8(error.what())}}};
            }
        }
    }

    // Only stores attached to the project are candidates; each attachment once.
    const auto session = state.objectStoreSession();
    std::vector<ExportTarget> candidates;
    const auto consider = [&](const std::string& store, const vc::objects::AttachmentIdentity& attachment) {
        if (std::any_of(candidates.begin(), candidates.end(), [&](const auto& c) { return c.attachment == attachment; }))
            return;
        std::optional<std::string> kind;
        session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
            const auto head = db.currentObjectRevision(txn, uuid);
            if (head && !head->deleted && !head->record.retired) kind = head->record.kind;
        });
        if (kind) candidates.push_back({store, attachment, *kind});
    };
    if (p.contains("store")) {
        const std::string store = p.value("store").toString().toStdString();
        const auto attachment = state.storeReference(store);
        if (!attachment)
            throw AgentBridgeError{-32007, "No attached store has that reference", {{"kind", "store"}, {"id", p.value("store")}}};
        consider(store, *attachment);
    } else {
        for (const auto& reference : state.vpkg()->document().data().at("stores")) {
            const auto store = reference.at("id").get<std::string>();
            if (const auto attachment = state.storeReference(store)) consider(store, *attachment);
        }
    }
    if (candidates.empty())
        throw AgentBridgeError{-32007, "Object not found in the project's stores", {{"kind", "object"}, {"id", QString::fromStdString(uuid)}}};
    if (candidates.size() > 1) {
        QJsonArray stores;
        for (const auto& candidate : candidates) stores.append(QString::fromStdString(candidate.store));
        throw AgentBridgeError{-32602, "The object is in several attached stores; choose one with store",
                               {{"param", "store"}, {"candidates", stores}}};
    }
    const ExportTarget target = candidates.front();
    if (target.kind != "surface" && target.kind != "fiber" && target.kind != "point_collection")
        throw AgentBridgeError{-32602, QStringLiteral("Objects of kind %1 cannot be exported").arg(QString::fromStdString(target.kind)),
                               {{"param", "uuid"}, {"kind", QString::fromStdString(target.kind)}}};
    if (validOnly && target.kind != "surface")
        throwParamError("validOnly", QStringLiteral("applies only to surfaces"));

    // Pending live edits of the object are saved first, through their own save paths.
    if (target.kind == "surface" && state.activeSurfaceId() == uuid && _window->_segmentationModule) {
        SegmentationModule& module = *_window->_segmentationModule;
        if (module.growthInProgress())
            throw AgentBridgeError{-32004, "Growth is running on this surface", {{"source", "growth"}}};
        const auto status = module.autosaveStatus();
        if (status.pending || status.saveInProgress || status.dirtyAfterSave) {
            const auto active = state.activeSurface().lock();
            if (!active || !module.saveSurface(active.get()))
                throw AgentBridgeError{-32005, "Pending surface edits could not be saved", {}};
        }
    } else if (target.kind == "fiber" && _window->_lineAnnotationController) {
        LineAnnotationController& fibers = *_window->_lineAnnotationController;
        if (fibers.structuralEditInProgress())
            throw AgentBridgeError{-32004, "A fiber merge, split or link is in progress", {{"source", "fiber"}}};
        const bool wasSuppressed = fibers.errorDialogsSuppressed();
        fibers.setErrorDialogsSuppressed(true);
        (void)fibers.takeLastSuppressedError();
        fibers.flushAllPendingSessionAutoSaves();
        const QString saveError = fibers.takeLastSuppressedError();
        fibers.setErrorDialogsSuppressed(wasSuppressed);
        if (!saveError.isEmpty())
            throw AgentBridgeError{-32005, "Pending fiber edits could not be saved", {{"detail", saveError}}};
    }

    // The object revision and the exported geometry come from one read snapshot.
    auto* watcher = new QFutureWatcher<ExportResult>(this);
    const QString path = QString::fromStdString(destination.string());
    _exportTargets.push_back(target.attachment);
    connect(watcher, &QFutureWatcher<ExportResult>::finished, this, [this, watcher, target, uuid, path]() {
        const ExportResult result = watcher->result();
        watcher->deleteLater();
        if (const auto found = std::find(_exportTargets.begin(), _exportTargets.end(), target.attachment);
            found != _exportTargets.end())
            _exportTargets.erase(found);
        auto it = _activeJobs.find(QStringLiteral("export"));
        if (it == _activeJobs.end())
            return;
        const bool success = result.error.isEmpty();
        if (success) {
            it->resultJson = {
                {"path", path},
                {"uuid", QString::fromStdString(uuid)},
                {"kind", QString::fromStdString(target.kind)},
                {"store", QString::fromStdString(target.store)},
                {"objectRevision", QString::fromStdString(result.objectRevision)},
                {"geometryRevision", QString::fromStdString(result.geometryRevision)},
            };
        }
        finishJob(QStringLiteral("export"), success,
                  success ? QStringLiteral("Exported %1").arg(QString::fromStdString(uuid)) : result.error,
                  success ? path : QString());
    });
    watcher->setFuture(QtConcurrent::run([session, target, uuid, destination, overwrite, validOnly]() {
        ExportResult result;
        try {
            session->read(target.attachment, [&](const auto& db, const auto& txn, const auto&) {
                const auto head = db.currentObjectRevision(txn, uuid);
                if (!head || head->deleted || head->record.retired)
                    throw std::runtime_error("the object was removed before it could be exported");
                result.objectRevision = head->uuid;
                result.geometryRevision = vc::objects::exportGeometry(db, txn, uuid, destination, std::nullopt, overwrite, validOnly);
            });
        } catch (const std::exception& error) {
            result.error = QString::fromUtf8(error.what());
        }
        return result;
    }));
    const QString jobId = beginJob(
        QStringLiteral("export"), QStringLiteral("objects.export"),
        QStringLiteral("export %1 to %2").arg(QString::fromStdString(uuid), path),
        /*broadcastStart=*/true);
    return {
        {"jobId", jobId},
        {"kind", "objects.export"},
        {"source", "export"},
        {"uuid", QString::fromStdString(uuid)},
        {"store", QString::fromStdString(target.store)},
        {"objectKind", QString::fromStdString(target.kind)},
    };
}

