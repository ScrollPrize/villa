#include "agent_bridge/AgentBridgeServer.hpp"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QPointer>
#include <QString>

#include <memory>
#include <string>
#include <vector>

#include "CWindow.hpp"
#include "CState.hpp"
#include "FiberCollectionController.hpp"
#include "MenuActionController.hpp"
#include "ProjectAttachmentController.hpp"
#include "ProjectLocations.hpp"
#include "segmentation/SegmentationWidget.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"

// ===========================================================================
// Project attachments and Save As. The umbilicus, normal grids, Lasagna
// manifests, entry detach and Save As go through ProjectAttachmentController,
// the controller behind the File menu's actions; automated fiber volumes go
// through FiberCollectionController, which the File menu and the fiber dock
// use. Umbilicus and normal-grid attachments are synchronous; Lasagna
// manifests and fiber volumes load in the background as "project" jobs.
// ===========================================================================

namespace {

using AttachError = ProjectAttachmentOutcome::Error;

ProjectAttachmentController* requireController(ProjectAttachmentController* controller)
{
    if (!controller) {
        throw AgentBridgeError{
            -32009, "Project attachments unavailable",
            QJsonObject{{"detail", "project attachment controller is not available"}}};
    }
    return controller;
}

QJsonArray toJsonArray(const QStringList& values)
{
    QJsonArray out;
    for (const QString& value : values)
        out.append(value);
    return out;
}

QJsonValue nullableString(const QString& value)
{
    return value.isEmpty() ? QJsonValue(QJsonValue::Null) : QJsonValue(value);
}

// The error a refused or failed controller outcome maps to.
[[noreturn]] void throwOutcomeError(const ProjectAttachmentOutcome& outcome,
                                    const QString& kind,
                                    const QString& failedMessage)
{
    switch (outcome.errorKind) {
    case AttachError::NoProject:
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    case AttachError::InvalidLocation:
        throw AgentBridgeError{
            -32007, QStringLiteral("Invalid %1 location").arg(kind),
            QJsonObject{{"kind", kind}, {"id", outcome.location}, {"detail", outcome.error}}};
    case AttachError::NotFound:
        throw AgentBridgeError{
            -32007, QStringLiteral("No %1 entry at %2").arg(kind, outcome.location),
            QJsonObject{{"kind", kind}, {"id", outcome.location}, {"detail", outcome.error}}};
    case AttachError::NoVolume:
        throw AgentBridgeError{-32001, "No volume loaded", QJsonObject{{"detail", outcome.error}}};
    case AttachError::Unsupported:
        throw AgentBridgeError{-32009, failedMessage, QJsonObject{{"detail", outcome.error}}};
    case AttachError::Busy:
        throw AgentBridgeError{
            -32004, "An attachment of this kind is already loading",
            QJsonObject{{"source", "project"}, {"detail", outcome.error}}};
    case AttachError::None:
    case AttachError::Failed:
        break;
    }
    throw AgentBridgeError{-32005, failedMessage, QJsonObject{{"detail", outcome.error}}};
}

QJsonObject attachResultJson(const QString& kind, const ProjectAttachmentOutcome& outcome)
{
    return {
        {"attached", outcome.changed},
        {"alreadyAttached", !outcome.changed},
        {"kind", kind},
        {"location", outcome.location},
        {"previousLocation", nullableString(outcome.previousLocation)},
        {"warnings", toJsonArray(outcome.warnings)},
    };
}

}  // namespace


// ---------------------------------------------------------------------------
// project.attach
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleProjectAttach(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    const QString kind = p.value("kind").toString();
    const QString location = p.value("location").toString().trimmed();
    if (location.isEmpty()) {
        throw AgentBridgeError{-32602, "location must not be blank",
                               QJsonObject{{"param", "location"}}};
    }
    if (p.contains("fiberInference") && kind != QLatin1String("lasagna_manifest")) {
        throw AgentBridgeError{-32602, "fiberInference applies only to lasagna_manifest",
                               QJsonObject{{"param", "fiberInference"}}};
    }
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->vpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};

    // Only volumes and Lasagna manifests may be remote, as segments.attach
    // refuses remote segments.
    const bool remote = vc3d::project_locations::looksRemote(location);
    if (remote && kind != QLatin1String("lasagna_manifest")) {
        throw AgentBridgeError{
            -32007, QStringLiteral("Invalid %1 location").arg(kind),
            QJsonObject{{"kind", kind}, {"id", location},
                        {"detail", "remote locations are not supported for this kind"}}};
    }
    if (!remote && QDir::isRelativePath(location) &&
        !location.startsWith(QStringLiteral("file://"), Qt::CaseInsensitive)) {
        throw AgentBridgeError{-32602, "local paths must be absolute",
                               QJsonObject{{"param", "location"}}};
    }

    if (kind == QLatin1String("afv"))
        return attachFiberVolume(location);

    ProjectAttachmentController* controller =
        requireController(_window->_projectAttachmentController.get());

    if (kind == QLatin1String("umbilicus") || kind == QLatin1String("normal_grid")) {
        const ProjectAttachmentOutcome outcome = kind == QLatin1String("umbilicus")
            ? controller->attachUmbilicus(location)
            : controller->attachNormalGrid(location);
        if (!outcome.success)
            throwOutcomeError(outcome, kind, QStringLiteral("Attach failed"));
        return attachResultJson(kind, outcome);
    }

    // lasagna_manifest
    requireSourceIdle(QStringLiteral("project"));
    QPointer<AgentBridgeServer> self(this);
    ProjectAttachmentOutcome refusal;
    const QString jobId = beginJob(
        QStringLiteral("project"), QStringLiteral("project.attach"),
        QStringLiteral("attach %1 %2").arg(kind, location),
        /*broadcastStart=*/false);
    const bool started = controller->startLasagnaAttachment(
        location, p.value("fiberInference").toBool(false),
        [self, kind](const ProjectAttachmentOutcome& outcome) {
            if (self)
                self->completeProjectAttachmentJob(kind, outcome);
        },
        &refusal);
    if (!started) {
        _activeJobs.remove(QStringLiteral("project"));
        refusal.location = location;
        throwOutcomeError(refusal, kind, QStringLiteral("Could not start the attachment"));
    }
    if (auto it = _activeJobs.find(QStringLiteral("project")); it != _activeJobs.end())
        broadcastJobProgress(it.value(), QStringLiteral("started"));

    return {
        {"jobId", jobId},
        {"kind", "project.attach"},
        {"source", "project"},
        {"attachmentKind", kind},
        {"location", location},
    };
}

QJsonObject AgentBridgeServer::attachFiberVolume(const QString& location)
{
    const QString kind = QStringLiteral("afv");
    FiberCollectionController* collection = _window->_fiberCollection.get();
    if (!collection) {
        throw AgentBridgeError{
            -32009, "Automated fiber volumes unavailable",
            QJsonObject{{"detail", "fiber collection controller is not available"}}};
    }
    const QString path = QFileInfo(vc3d::project_locations::storedLocalLocation(
                                       location, _window->_state->vpkg()->path().parent_path()))
                             .absoluteFilePath();
    if (!QFileInfo(path).isFile()) {
        throw AgentBridgeError{
            -32007, "Invalid afv location",
            QJsonObject{{"kind", kind}, {"id", path},
                        {"detail", "automated fiber volume not found"}}};
    }
    switch (collection->openRefusal()) {
    case FiberCollectionController::OpenRefusal::NoVolume:
        throw AgentBridgeError{-32001, "No volume loaded",
                               QJsonObject{{"detail", "open the CT volume first"}}};
    case FiberCollectionController::OpenRefusal::NoCoordinateIdentity:
        throw AgentBridgeError{
            -32009, "The current volume has no verified coordinate identity",
            QJsonObject{{"detail", "attach a volume with coordinate metadata first"}}};
    case FiberCollectionController::OpenRefusal::None:
        break;
    }

    // The open collection is the attachment: attaching it again changes
    // nothing and does not reopen it.
    const QString previous = collection->collectionPath();
    if (!previous.isEmpty() &&
        QFileInfo(previous).canonicalFilePath() == QFileInfo(path).canonicalFilePath()) {
        ProjectAttachmentOutcome outcome;
        outcome.success = true;
        outcome.location = previous;
        QJsonObject result = attachResultJson(kind, outcome);
        result["jobId"] = QJsonValue::Null;
        return result;
    }

    requireSourceIdle(QStringLiteral("project"));
    const QString jobId = beginJob(
        QStringLiteral("project"), QStringLiteral("project.attach"),
        QStringLiteral("attach %1 %2").arg(kind, path),
        /*broadcastStart=*/false);

    // The collection reports every open once, carrying the id openCollection
    // returns; a refusal can be reported from inside the call, before the id
    // is known.
    QPointer<AgentBridgeServer> self(this);
    auto expectedId = std::make_shared<quint64>(0);
    auto connection = std::make_shared<QMetaObject::Connection>();
    *connection = connect(
        collection, &FiberCollectionController::collectionOpened, this,
        [self, connection, expectedId, path, previous, kind](
            quint64 openId, const QString& openedPath, bool success,
            const QString& message, bool attachmentSaved) {
            if (*expectedId != 0 ? openId != *expectedId : openedPath != path)
                return;
            disconnect(*connection);
            if (!self)
                return;
            ProjectAttachmentOutcome outcome;
            outcome.location = path;
            outcome.previousLocation = previous;
            outcome.success = success && attachmentSaved;
            outcome.changed = outcome.success;
            if (!success) {
                outcome.error = message;
            } else if (!attachmentSaved) {
                outcome.error = QObject::tr(
                    "The automated fiber volume opened, but its attachment to the project "
                    "could not be saved, so it will not reopen with the project.");
            }
            self->completeProjectAttachmentJob(kind, outcome);
        });
    const quint64 openId = collection->openCollection(path);
    if (*connection)
        *expectedId = openId;
    if (auto it = _activeJobs.find(QStringLiteral("project"));
        it != _activeJobs.end() && it->id == jobId) {
        broadcastJobProgress(it.value(), QStringLiteral("started"));
    }

    return {
        {"jobId", jobId},
        {"kind", "project.attach"},
        {"source", "project"},
        {"attachmentKind", kind},
        {"location", path},
    };
}

void AgentBridgeServer::completeProjectAttachmentJob(const QString& kind,
                                                     const ProjectAttachmentOutcome& outcome)
{
    auto it = _activeJobs.find(QStringLiteral("project"));
    if (it == _activeJobs.end())
        return;
    if (outcome.success)
        it->resultJson = attachResultJson(kind, outcome);
    const QString message = outcome.success
        ? QStringLiteral("Attached %1 %2").arg(kind, outcome.location)
        : outcome.error.isEmpty() ? QStringLiteral("Attachment failed") : outcome.error;
    finishJob(QStringLiteral("project"), outcome.success, message, outcome.location);
}


// ---------------------------------------------------------------------------
// project.detach
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleProjectDetach(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    const QString kind = p.value("kind").toString();
    const QString location = p.value("location").toString().trimmed();

    const bool singleValued = kind == QLatin1String("umbilicus") || kind == QLatin1String("afv");
    if (singleValued && p.contains("location")) {
        throw AgentBridgeError{
            -32602, QStringLiteral("%1 has no location to choose; omit location").arg(kind),
            QJsonObject{{"param", "location"}}};
    }
    if (!singleValued && location.isEmpty()) {
        throw AgentBridgeError{-32602, "location is required for this kind",
                               QJsonObject{{"param", "location"}}};
    }
    if (!p.value("confirm").toBool(false)) {
        throw AgentBridgeError{
            -32602, "confirm must be true (detaching changes the project file)",
            QJsonObject{{"param", "confirm"}, {"reason", "destructive; pass confirm=true"}}};
    }
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->vpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    auto pkg = state->vpkg();

    if (kind == QLatin1String("afv")) {
        FiberCollectionController* collection = _window->_fiberCollection.get();
        if (!collection) {
            throw AgentBridgeError{
                -32009, "Automated fiber volumes unavailable",
                QJsonObject{{"detail", "fiber collection controller is not available"}}};
        }
        requireSourceIdle(QStringLiteral("project"));
        const QString open = collection->collectionPath();
        const bool detached = collection->detachCollection();
        return {{"detached", detached}, {"kind", kind}, {"location", nullableString(open)}};
    }

    const bool entryOfData = kind == QLatin1String("volume") || kind == QLatin1String("segments");
    if (entryOfData) {
        // Removing a volume or segments entry reloads the package, which an
        // edit or a running job may still be using.
        if (_window->_segmentationWidget && _window->_segmentationWidget->isEditingEnabled()) {
            throw AgentBridgeError{
                -32004, QStringLiteral("Cannot detach %1 while editing").arg(kind),
                QJsonObject{{"detail", "disable segmentation editing first"}}};
        }
        for (const QString& source : _activeJobs.keys())
            requireSourceIdle(source);
    }
    if (kind == QLatin1String("volume") && state->currentVolume()) {
        // The entry backing the current volume would switch volumes on reload.
        const auto& volume = state->currentVolume();
        const std::string current = volume->isRemote() ? volume->remoteLocator()
                                                       : volume->path().string();
        const auto backing = pkg->matchingVolumeEntry(current);
        if (backing && vc::project::sameLocation(backing->location, location.toStdString(),
                                                 pkg->path().parent_path())) {
            throw AgentBridgeError{
                -32009, "Cannot detach the current volume",
                QJsonObject{{"detail", "select another volume first"}}};
        }
    }

    ProjectAttachmentController* controller =
        requireController(_window->_projectAttachmentController.get());
    ProjectAttachmentOutcome outcome;
    if (kind == QLatin1String("umbilicus")) {
        outcome = controller->detachUmbilicus();
    } else {
        const ProjectEntryKind entryKind =
            kind == QLatin1String("normal_grid")        ? ProjectEntryKind::NormalGrid
            : kind == QLatin1String("lasagna_manifest") ? ProjectEntryKind::Lasagna
            : kind == QLatin1String("volume")           ? ProjectEntryKind::Volume
                                                        : ProjectEntryKind::Segments;
        if (entryKind == ProjectEntryKind::Lasagna)
            requireSourceIdle(QStringLiteral("project"));
        outcome = controller->detachEntry(entryKind, location);
    }
    if (!outcome.success)
        throwOutcomeError(outcome, kind, QStringLiteral("Detach failed"));
    return {
        {"detached", outcome.changed},
        {"kind", kind},
        {"location", nullableString(outcome.location)},
    };
}


// ---------------------------------------------------------------------------
// project.save_as
// ---------------------------------------------------------------------------

QJsonObject AgentBridgeServer::handleProjectSaveAs(const QJsonValue& params)
{
    const QJsonObject p = params.toObject();
    QString path = p.value("path").toString().trimmed();
    if (path.isEmpty() || QDir::isRelativePath(path)) {
        throw AgentBridgeError{-32602, "path must be an absolute file path",
                               QJsonObject{{"param", "path"}}};
    }
    CState* state = _window ? _window->_state : nullptr;
    if (!state || !state->vpkg())
        throw AgentBridgeError{-32000, "No volume package loaded", {}};
    if (!path.endsWith(QStringLiteral(".volpkg.json"), Qt::CaseInsensitive))
        path += QStringLiteral(".volpkg.json");
    path = QDir::cleanPath(path);
    // The same rule and code as project.create.
    if (QFileInfo::exists(path) && !p.value("overwrite").toBool(false)) {
        throw AgentBridgeError{
            -32005, "output file already exists; pass overwrite=true to replace",
            QJsonObject{{"path", path}}};
    }
    ProjectAttachmentController* controller =
        requireController(_window->_projectAttachmentController.get());
    const ProjectAttachmentOutcome outcome = controller->saveProjectAs(path);
    if (!outcome.success)
        throwOutcomeError(outcome, QStringLiteral("project"), QStringLiteral("Save failed"));
    if (_window->_menuController)
        _window->_menuController->updateRecentVolpkgList(outcome.location);
    return {
        {"saved", true},
        {"path", outcome.location},
        {"warnings", toJsonArray(outcome.warnings)},
    };
}
