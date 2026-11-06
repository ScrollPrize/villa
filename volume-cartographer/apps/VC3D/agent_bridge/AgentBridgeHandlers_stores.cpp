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
#include "MenuActionController.hpp"
#include "ProjectResources.hpp"
#include "StoreImportController.hpp"
#include "segmentation/SegmentationModule.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/ui/VCCollection.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/GeometryExport.hpp>
#include <vc_object_store/objects/StoreReader.hpp>
#include <vc_reader/zarr.hpp>

// Object stores of the open project (importing source files into them,
// exporting their objects, attaching, routing and detaching them), Save As,
// and the project's resource entries and umbilicus.

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
    vc::objects::ObjectVersion version;
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
    // Store references belong to the project that started the import.
    it->resultJson["projectReplaced"] = outcome.projectReplaced;
    if (outcome.projectReplaced) it->resultJson["store"] = QJsonValue::Null;
    const bool success = !report.fatalError && !report.cancelled && !outcome.projectReplaced;
    QString message;
    if (outcome.projectReplaced)
        message = QStringLiteral("Import stopped: the project changed after %1 object(s)").arg(report.ingested);
    else if (report.fatalError)
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
        session->readStore(attachment, [&](const vc::objects::StoreReader& reader, const auto&) {
            const auto head = reader.currentObject(uuid);
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
    // Native export reads the store file; a hosted store has none here.
    if (session->hosted(target.attachment))
        throw AgentBridgeError{-32602, "Objects of a hosted store cannot be exported", {{"param", "store"},
                               {"store", QString::fromStdString(target.store)}}};
    const std::filesystem::path storeFile = session->location(target.attachment);
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
                {"version", QJsonObject{{"store", QString::fromStdString(result.version.storeUuid)},
                                        {"generation", QString::fromStdString(result.version.generation)},
                                        {"object", QString::fromStdString(result.version.objectUuid)},
                                        {"version", qint64(result.version.version)}}},
                {"geometryRevision", QString::fromStdString(result.geometryRevision)},
            };
        }
        finishJob(QStringLiteral("export"), success,
                  success ? QStringLiteral("Exported %1").arg(QString::fromStdString(uuid)) : result.error,
                  success ? path : QString());
    });
    watcher->setFuture(QtConcurrent::run([storeFile, uuid, destination, overwrite, validOnly]() {
        ExportResult result;
        try {
            // A read-only connection of its own: the export and its reported
            // version come from one snapshot, and read-only stores export too.
            vc::objects::OpenOptions options;
            options.readOnly = true;
            auto db = vc::objects::ObjectStore::open(storeFile, options);
            const auto txn = db.read();
            const auto head = db.currentObjectRevision(txn, uuid);
            if (!head || head->deleted || head->record.retired)
                throw std::runtime_error("the object was removed before it could be exported");
            if (head->geometryRevision) result.geometryRevision = *head->geometryRevision;
            result.version = vc::objects::exportGeometry(db, txn, uuid, destination, std::nullopt, overwrite, validOnly);
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


// ---------------------------------------------------------------------------
// Store references and routing
// ---------------------------------------------------------------------------

namespace {

// The project's reference ids, as persisted.
bool referencesStore(const CState& state, const std::string& id)
{
    for (const auto& reference : state.vpkg()->document().data().at("stores"))
        if (reference.at("id") == id) return true;
    return false;
}

[[noreturn]] void throwUnknownStore(const QString& id)
{
    throw AgentBridgeError{-32007, QStringLiteral("Unknown store reference: %1").arg(id), {{"kind", "store"}, {"id", id}}};
}

QJsonObject storeAttachmentJson(const CState& state, const vc::objects::AttachmentIdentity& attachment)
{
    return {
        {"storeId", storeReferenceJson(state, attachment)},
        {"readOnly", !state.objectStoreSession()->writable(attachment)},
        {"path", QString::fromStdString(state.objectStoreSession()->path(attachment).string())},
    };
}

QJsonObject routesJson(const CState& state)
{
    QJsonObject routes;
    for (const auto& [kind, selected] : state.vpkg()->document().data().at("selections").items())
        routes[QString::fromStdString(kind)] = selected.is_null() ? QJsonValue(QJsonValue::Null)
                                                                  : QJsonValue(QString::fromStdString(selected.get<std::string>()));
    return routes;
}

}  // namespace


QJsonObject AgentBridgeServer::handleStoresAttach(const QJsonValue& params)
{
    CState& state = requireProject(_window ? _window->_state : nullptr);
    const QJsonObject p = params.toObject();
    const auto path = absolutePathParam(p, "path");
    try {
        const auto attachment = state.attachStore(path, CState::storeAuthor(), p.value("readOnly").toBool(false));
        return storeAttachmentJson(state, attachment);
    } catch (const std::exception& error) {
        throw AgentBridgeError{-32005, "Could not attach the store", {{"detail", QString::fromUtf8(error.what())}}};
    }
}


QJsonObject AgentBridgeServer::handleStoresCreate(const QJsonValue& params)
{
    namespace fs = std::filesystem;
    CState& state = requireProject(_window ? _window->_state : nullptr);
    const QJsonObject p = params.toObject();
    auto path = absolutePathParam(p, "path");
    if (!path.has_extension()) path += ".vcobj";
    std::error_code existsError;
    if (fs::exists(path, existsError))
        throw AgentBridgeError{-32005, "A file already exists at that path", {{"path", QString::fromStdString(path.string())}}};
    bool created = false;
    try {
        { auto store = vc::objects::ObjectStore::create(path, CState::storeOpenOptions(CState::storeAuthor(), false)); created = true; }
        const auto attachment = state.attachStore(path, CState::storeAuthor());
        return storeAttachmentJson(state, attachment);
    } catch (const std::exception& error) {
        if (created) { std::error_code ignored; fs::remove(path, ignored); }
        throw AgentBridgeError{-32005, "Could not create the store", {{"detail", QString::fromUtf8(error.what())}}};
    }
}


QJsonObject AgentBridgeServer::handleStoresDetach(const QJsonValue& params)
{
    CState& state = requireProject(_window ? _window->_state : nullptr);
    const QString id = params.toObject().value("storeId").toString();
    if (!referencesStore(state, id.toStdString())) throwUnknownStore(id);
    // Workers keep the attachments they write to or read until they finish.
    if (const auto attachment = state.storeReference(id.toStdString())) {
        const auto* importer = _window->_storeImportController.get();
        const bool importing = importer && importer->target() == attachment;
        const bool exporting = std::find(_exportTargets.begin(), _exportTargets.end(), *attachment) != _exportTargets.end();
        if (importing || exporting)
            throw AgentBridgeError{-32004, "A job is using this store", {{"source", importing ? "import" : "export"}}};
    }
    try {
        state.detachStoreReference(id.toStdString());
    } catch (const StoreSwitchRefused& error) {
        throw AgentBridgeError{-32004, "Pending edits could not be saved", {{"detail", QString::fromUtf8(error.what())}}};
    } catch (const std::exception& error) {
        throw AgentBridgeError{-32005, "Could not detach the store", {{"detail", QString::fromUtf8(error.what())}}};
    }
    return {{"storeId", id}, {"routes", routesJson(state)}};
}


QJsonObject AgentBridgeServer::handleStoresRoute(const QJsonValue& params)
{
    CState& state = requireProject(_window ? _window->_state : nullptr);
    const QJsonObject p = params.toObject();
    const std::string kind = p.value("kind").toString().toStdString();
    const QJsonValue storeId = p.value("storeId");
    const std::string id = storeId.isNull() ? std::string{} : storeId.toString().toStdString();
    if (storeId.isString() && id.empty()) throwParamError("storeId", QStringLiteral("must not be empty; pass null to route no store"));
    if (!id.empty() && !referencesStore(state, id)) throwUnknownStore(storeId.toString());
    try {
        state.setStoreSelection(kind, id);
    } catch (const StoreSwitchRefused& error) {
        throw AgentBridgeError{-32004, "Pending edits could not be saved", {{"detail", QString::fromUtf8(error.what())}}};
    } catch (const std::exception& error) {
        throw AgentBridgeError{-32005, "Could not route the store", {{"detail", QString::fromUtf8(error.what())}}};
    }
    return {{"kind", QString::fromStdString(kind)}, {"storeId", storeId}, {"routes", routesJson(state)}};
}


// ---------------------------------------------------------------------------
// The project file and its resource entries
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleProjectSaveAs(const QJsonValue& params)
{
    namespace fs = std::filesystem;
    CState& state = requireProject(_window ? _window->_state : nullptr);
    // Reopening detaches every store, which would stop an import or export.
    requireSourceIdle(QStringLiteral("import"));
    requireSourceIdle(QStringLiteral("export"));
    auto path = absolutePathParam(params.toObject(), "path");
    if (!path.has_extension()) path += ".vcproj";
    std::error_code existsError;
    if (fs::exists(path, existsError))
        throw AgentBridgeError{-32005, "A file already exists at that path", {{"path", QString::fromStdString(path.string())}}};
    const QString previous = QString::fromStdString(state.vpkg()->path().string());
    if (state.prepareStoreSwitch && !state.prepareStoreSwitch())
        throw AgentBridgeError{-32004, "Pending edits could not be saved, or an edit operation is running", {}};
    try {
        state.savePointCollections();
    } catch (const std::exception& error) {
        throw AgentBridgeError{-32004, "Pending point-collection edits could not be saved", {{"detail", QString::fromUtf8(error.what())}}};
    }
    try {
        state.vpkg()->document().saveAs(path);
    } catch (const std::exception& error) {
        throw AgentBridgeError{-32005, "Could not save the project", {{"detail", QString::fromUtf8(error.what())}}};
    }
    QString error;
    const QString file = QString::fromStdString(path.string());
    if (!_window->openVolumePackage(file, false, &error, QString::fromStdString(state.currentVolumeId())))
        throw AgentBridgeError{-32005, "The project was saved but could not be reopened", {{"path", file}, {"detail", error}}};
    return {{"path", file}, {"previousPath", previous}};
}


QJsonObject AgentBridgeServer::handleProjectAttachNormalGrid(const QJsonValue& params)
{
    CState& state = requireProject(_window ? _window->_state : nullptr);
    const QString location = params.toObject().value("location").toString().trimmed();
    if (location.isEmpty()) throwParamError("location", QStringLiteral("must not be blank"));
    const std::string input = location.toStdString();
    if (!vcr::zarr::isRemote(input) && !vc::project::resolveLocalPath(input).is_absolute())
        throwParamError("location", QStringLiteral("a local location must be absolute"));
    const auto invalid = vc::project::validateLocation(vc::project::Category::NormalGrids, input);
    if (!invalid.empty())
        throw AgentBridgeError{-32007, "Not a normal-grid location", {{"kind", "normal_grid"}, {"detail", QString::fromStdString(invalid)}}};
    bool attached = false;
    try {
        attached = state.vpkg()->addNormalGridEntry(input);
    } catch (const std::exception& error) {
        throw AgentBridgeError{-32005, "Could not save the project", {{"detail", QString::fromUtf8(error.what())}}};
    }
    if (attached) _window->refreshCurrentVolumePackageUi();
    return {{"location", location}, {"attached", attached}, {"alreadyAttached", !attached}};
}


QJsonObject AgentBridgeServer::handleProjectAttachLasagnaManifest(const QJsonValue& params)
{
    requireProject(_window ? _window->_state : nullptr);
    const QJsonObject p = params.toObject();
    QString location = p.value("location").toString().trimmed();
    if (location.isEmpty()) throwParamError("location", QStringLiteral("must not be blank"));
    if (!vcr::zarr::isRemote(location.toStdString())) {
        const std::filesystem::path local(location.toStdString());
        if (!local.is_absolute()) throwParamError("location", QStringLiteral("a local manifest path must be absolute"));
        location = QString::fromStdString(local.lexically_normal().string());
    }
    const bool fiber = p.value("role").toString() == QLatin1String("fiber");
    requireSourceIdle(QStringLiteral("manifest"));
    MenuActionController* menu = _window->_menuController.get();
    if (!menu)
        throw AgentBridgeError{-32010, "Internal error", {{"detail", "menu controller is not available"}}};
    QPointer<AgentBridgeServer> self(this);
    QString error;
    if (!menu->startLasagnaManifestAttachment(location, fiber, [self](const MenuActionController::LasagnaManifestOutcome& outcome) {
            if (!self) return;
            auto it = self->_activeJobs.find(QStringLiteral("manifest"));
            if (it == self->_activeJobs.end()) return;
            if (outcome.success) {
                it->resultJson = {
                    {"location", outcome.location},
                    {"role", outcome.fiberInference ? "fiber" : "regular"},
                    {"attached", !outcome.alreadyAttached},
                    {"alreadyAttached", outcome.alreadyAttached},
                    {"volumeIds", QJsonArray::fromStringList(outcome.volumeIds)},
                };
            }
            self->finishJob(QStringLiteral("manifest"), outcome.success,
                            outcome.success ? QStringLiteral("Attached %1").arg(outcome.location) : outcome.error, QString());
        }, &error)) {
        throw AgentBridgeError{-32004, error, {{"source", "manifest"}}};
    }
    const QString jobId = beginJob(QStringLiteral("manifest"), QStringLiteral("project.attach_lasagna_manifest"),
                                   QStringLiteral("attach Lasagna manifest %1").arg(location), /*broadcastStart=*/true);
    return {
        {"jobId", jobId},
        {"kind", "project.attach_lasagna_manifest"},
        {"source", "manifest"},
        {"location", location},
        {"role", fiber ? "fiber" : "regular"},
    };
}


QJsonObject AgentBridgeServer::handleProjectDetachEntry(const QJsonValue& params)
{
    CState& state = requireProject(_window ? _window->_state : nullptr);
    const QString location = params.toObject().value("location").toString();
    const auto pkg = state.vpkg();
    const auto listed = [&](const std::vector<vc::project::Entry>& entries) {
        return std::any_of(entries.begin(), entries.end(), [&](const auto& e) { return e.location == location.toStdString(); });
    };
    if (!listed(pkg->volumeEntries()) && !listed(pkg->normalGridEntries()) && !listed(pkg->allLasagnaDatasetEntries()))
        throw AgentBridgeError{-32007, "The project has no entry at that location", {{"kind", "entry"}, {"id", location}}};
    // Pending edits are saved first; nothing changes when they cannot be.
    if (state.prepareStoreSwitch && !state.prepareStoreSwitch())
        throw AgentBridgeError{-32004, "Pending edits could not be saved, or an edit operation is running", {}};
    const QString scope = pkg->entryOwner(location.toStdString()).empty() ? QStringLiteral("everyone") : QStringLiteral("only you");
    bool removed = false;
    try {
        removed = _window->_menuController->removeEntry(location);
    } catch (const std::exception& error) {
        throw AgentBridgeError{-32005, "Could not save the project", {{"detail", QString::fromUtf8(error.what())}}};
    }
    if (!removed)
        throw AgentBridgeError{-32007, "The project has no entry at that location", {{"kind", "entry"}, {"id", location}}};
    return {{"location", location}, {"scope", scope}};
}


// ---------------------------------------------------------------------------
// The annotation volume's umbilicus
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleUmbilicusImport(const QJsonValue& params)
{
    CState& state = requireProject(_window ? _window->_state : nullptr);
    const QJsonObject p = params.toObject();
    const auto path = absolutePathParam(p, "path");
    if (!state.currentVolume() || !state.annotationVolume())
        throw AgentBridgeError{-32001, "No volume selected", {}};
    const auto store = state.storeFor("umbilicus");
    if (!store)
        throw AgentBridgeError{-32007, "No store is routed for umbilici", {{"kind", "store"}}};
    if (!state.objectStoreSession()->writable(*store))
        throw AgentBridgeError{-32009, "The umbilicus store is read-only", {{"param", "store"}}};
    vc3d::UmbilicusImportPlan plan;
    try {
        plan = vc3d::planUmbilicusImport(state, path);
    } catch (const vc3d::UmbilicusFileUnreadable& error) {
        throw AgentBridgeError{-32602, "Not a readable umbilicus file", {{"param", "path"}, {"detail", QString::fromUtf8(error.what())}}};
    } catch (const std::exception& error) {
        throw AgentBridgeError{-32005, "Could not read the umbilicus store", {{"detail", QString::fromUtf8(error.what())}}};
    }
    const auto& binding = plan.binding;
    if (binding.refusal)
        throw AgentBridgeError{-32005, "The umbilicus does not fit the volume", {{"detail", QString::fromStdString(*binding.refusal)}}};
    const QJsonValue inferred = binding.storedUnitsPerVoxel ? QJsonValue(*binding.storedUnitsPerVoxel) : QJsonValue(QJsonValue::Null);
    double unitsPerVoxel = binding.storedUnitsPerVoxel.value_or(1.0);
    if (binding.confirm) {
        // The bridge form of the GUI's confirmation prompt.
        if (!p.contains("unitsPerVoxel"))
            throw AgentBridgeError{-32602, "Pass unitsPerVoxel: the file's frame needs confirming",
                                   {{"param", "unitsPerVoxel"}, {"inferred", inferred}, {"frame", QString::fromStdString(binding.frame)}}};
        unitsPerVoxel = p.value("unitsPerVoxel").toDouble();
    } else if (p.contains("unitsPerVoxel") && p.value("unitsPerVoxel").toDouble() != unitsPerVoxel) {
        throw AgentBridgeError{-32602, "The file states its units per voxel",
                               {{"param", "unitsPerVoxel"}, {"stated", inferred}, {"frame", QString::fromStdString(binding.frame)}}};
    }
    if (plan.replaces && !p.value("replace").toBool(false))
        throw AgentBridgeError{-32005, "The volume already has an umbilicus; pass replace=true to replace it",
                               {{"volume", QString::fromStdString(plan.target.volume)}}};
    std::string uuid;
    try {
        uuid = vc3d::importUmbilicus(state, plan, unitsPerVoxel);
    } catch (const std::exception& error) {
        throw AgentBridgeError{-32005, "Umbilicus import failed", {{"detail", QString::fromUtf8(error.what())}}};
    }
    return {
        {"uuid", QString::fromStdString(uuid)},
        {"volume", QString::fromStdString(plan.target.volume)},
        {"store", storeReferenceJson(state, plan.store)},
        {"unitsPerVoxel", unitsPerVoxel},
        {"frame", binding.frame.empty() ? QStringLiteral("entered") : QString::fromStdString(binding.frame)},
        {"replaced", plan.replaces},
    };
}


QJsonObject AgentBridgeServer::handleUmbilicusRemove(const QJsonValue&)
{
    CState& state = requireProject(_window ? _window->_state : nullptr);
    if (!state.currentVolume() || !state.annotationVolume())
        throw AgentBridgeError{-32001, "No volume selected", {}};
    std::optional<vc3d::AnnotationUmbilici> umbilici;
    try {
        umbilici = vc3d::annotationUmbilici(state);
    } catch (const std::exception& error) {
        throw AgentBridgeError{-32005, "Could not read the umbilicus store", {{"detail", QString::fromUtf8(error.what())}}};
    }
    if (!umbilici)
        throw AgentBridgeError{-32007, "No store is routed for umbilici", {{"kind", "store"}}};
    QJsonObject result{{"volume", QString::fromStdString(umbilici->volume)}, {"store", storeReferenceJson(state, umbilici->store)}};
    if (umbilici->uuids.empty()) {
        result["removed"] = QJsonArray();
        return result;
    }
    if (!state.objectStoreSession()->writable(umbilici->store))
        throw AgentBridgeError{-32009, "The umbilicus store is read-only", {{"param", "store"}}};
    try {
        result["removed"] = stringsJson(vc3d::retireUmbilici(state, *umbilici));
    } catch (const std::exception& error) {
        throw AgentBridgeError{-32005, "Umbilicus removal failed", {{"detail", QString::fromUtf8(error.what())}}};
    }
    return result;
}
