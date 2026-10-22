#include "ProjectAttachmentController.hpp"

#include <QDir>
#include <QFileInfo>
#include <QSettings>
#include <QFutureWatcher>
#include <QtConcurrent>

#include <algorithm>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "CState.hpp"
#include "CWindow.hpp"
#include "FiberCollectionController.hpp"
#include "ProjectLocations.hpp"
#include "VCSettings.hpp"
#include "VolumeAttachmentController.hpp"

#include "utils/Json.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/ScrollUmbilicus.hpp"
#include "vc/core/util/Umbilicus.hpp"
#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/ProjectVolumes.hpp"

namespace {

namespace fs = std::filesystem;
using Error = ProjectAttachmentOutcome::Error;

ProjectAttachmentOutcome failure(Error kind, const QString& location, const QString& error)
{
    ProjectAttachmentOutcome outcome;
    outcome.errorKind = kind;
    outcome.location = location;
    outcome.error = error;
    return outcome;
}

using vc3d::project_locations::findEntry;
using vc3d::project_locations::looksRemote;
using vc3d::project_locations::storedLocalLocation;

struct LasagnaTaskResult {
    std::vector<VolumePkg::PreparedVolumeAttachment> volumes;
    QString error;
};

}  // namespace

ProjectAttachmentController::ProjectAttachmentController(CWindow* window)
    : QObject(window)
    , _window(window)
{
}

bool ProjectAttachmentController::hasProject(ProjectAttachmentOutcome* outcome) const
{
    if (_window && _window->_state && _window->_state->vpkg()) {
        return true;
    }
    if (outcome) {
        outcome->errorKind = Error::NoProject;
        outcome->error = QObject::tr("Open or create a project first.");
    }
    return false;
}

ProjectAttachmentOutcome ProjectAttachmentController::attachUmbilicus(const QString& file)
{
    ProjectAttachmentOutcome outcome;
    if (!hasProject(&outcome)) {
        return outcome;
    }
    auto pkg = _window->_state->vpkg();
    const fs::path projectDir = pkg->path().parent_path();
    if (file.trimmed().isEmpty() || looksRemote(file)) {
        return failure(Error::InvalidLocation, file.trimmed(),
                       QObject::tr("An umbilicus must be a local file."));
    }
    const QString location = storedLocalLocation(file, projectDir);
    outcome.location = location;

    vc::core::util::UmbilicusFileInfo info;
    try {
        info = vc::core::util::Umbilicus::LoadFileInfo(
            std::filesystem::path(location.toStdString()));
    } catch (const std::exception& e) {
        return failure(Error::InvalidLocation, location,
                       QObject::tr("%1 is not a valid umbilicus file:\n\n%2")
                           .arg(location, QString::fromUtf8(e.what())));
    }

    // Malformed metadata makes the file unusable until it is fixed (the
    // resolver refuses it), so attaching it would store an attachment nothing
    // can use and report success beside a warning. Rejected instead, naming
    // the errors; fix the file and attach again.
    if (!info.metadataErrors.empty()) {
        QStringList errors;
        errors.reserve(static_cast<int>(info.metadataErrors.size()));
        for (const auto& error : info.metadataErrors) {
            errors << QString::fromStdString(error);
        }
        return failure(Error::InvalidLocation, location,
                       QObject::tr("%1 declares malformed frame metadata and was not "
                                   "attached:\n\n%2")
                           .arg(QFileInfo(location).fileName(),
                                errors.join(QStringLiteral("\n"))));
    }

    if (!vc::core::util::umbilicusFrameClaim(info).any()) {
        // Dimensions alone are a complete statement (the preferred one, in fact,
        // being exact integer counts), so warning about a missing voxelsize_um
        // there would call the better-stamped file unstamped.
        outcome.warnings << QObject::tr("%1 declares no frame, so consumers may have to guess "
                                        "the grid its coordinates index.")
                                .arg(QFileInfo(location).fileName());
    }

    outcome.success = true;
    const QString configured = QString::fromStdString(pkg->umbilicus());
    if (!configured.isEmpty() &&
        vc::project::sameLocation(configured.toStdString(), location.toStdString(), projectDir)) {
        outcome.location = configured;
        return outcome;
    }
    outcome.previousLocation = configured;
    pkg->setUmbilicus(location.toStdString());
    outcome.changed = true;
    // Consumers that placed geometry relative to the old umbilicus have no other
    // way to learn this happened.
    _window->_state->notifyUmbilicusChanged();
    _window->showStatusBarMessage(
        QObject::tr("Attached umbilicus %1").arg(QFileInfo(location).fileName()), 5000);
    return outcome;
}

ProjectAttachmentOutcome ProjectAttachmentController::detachUmbilicus()
{
    ProjectAttachmentOutcome outcome;
    if (!hasProject(&outcome)) {
        return outcome;
    }
    auto pkg = _window->_state->vpkg();
    const QString configured = QString::fromStdString(pkg->umbilicus());
    outcome.location = configured;
    outcome.success = true;
    if (configured.isEmpty()) {
        return outcome;
    }
    // Clearing the field persists the project and restores automatic
    // resolution; the umbilicus is a single-valued field, so it is not part of
    // the entry-based detach.
    pkg->setUmbilicus({});
    outcome.changed = true;
    _window->_state->notifyUmbilicusChanged();
    _window->showStatusBarMessage(
        QObject::tr("Detached umbilicus %1; resolution is automatic again").arg(configured),
        5000);
    return outcome;
}

ProjectAttachmentOutcome ProjectAttachmentController::attachNormalGrid(const QString& location)
{
    ProjectAttachmentOutcome outcome;
    if (!hasProject(&outcome)) {
        return outcome;
    }
    auto pkg = _window->_state->vpkg();
    const fs::path projectDir = pkg->path().parent_path();
    if (location.trimmed().isEmpty() || looksRemote(location)) {
        return failure(Error::InvalidLocation, location.trimmed(),
                       QObject::tr("Not a valid normal-grid location:\n\n%1")
                           .arg(QObject::tr("Remote locations are only supported for volumes.")));
    }
    const QString normalized = storedLocalLocation(location, projectDir);
    outcome.location = normalized;
    const auto err = vc::project::validateLocation(vc::project::Category::NormalGrids,
                                                   normalized.toStdString());
    if (!err.empty()) {
        return failure(Error::InvalidLocation, normalized,
                       QObject::tr("Not a valid normal-grid location:\n\n%1")
                           .arg(QString::fromStdString(err)));
    }
    outcome.success = true;
    if (const QString stored = findEntry(pkg->normalGridEntries(), normalized, projectDir);
        !stored.isEmpty()) {
        outcome.location = stored;
        return outcome;
    }
    if (!pkg->addNormalGridEntry(normalized.toStdString())) {
        return failure(Error::Failed, normalized, QObject::tr("Could not add normal grid."));
    }
    outcome.changed = true;
    _window->refreshCurrentVolumePackageUi(QString(), true);
    return outcome;
}

ProjectAttachmentOutcome ProjectAttachmentController::detachEntry(ProjectEntryKind kind,
                                                                  const QString& location)
{
    ProjectAttachmentOutcome outcome;
    if (!hasProject(&outcome)) {
        return outcome;
    }
    auto pkg = _window->_state->vpkg();
    const fs::path projectDir = pkg->path().parent_path();
    std::vector<vc::project::Entry> entries;
    switch (kind) {
    case ProjectEntryKind::Volume:     entries = pkg->volumeEntries(); break;
    case ProjectEntryKind::Segments:   entries = pkg->segmentEntries(); break;
    case ProjectEntryKind::NormalGrid: entries = pkg->normalGridEntries(); break;
    case ProjectEntryKind::Lasagna:    entries = pkg->allLasagnaDatasetEntries(); break;
    }
    const QString stored = findEntry(entries, location.trimmed(), projectDir);
    if (stored.isEmpty() || !pkg->removeEntry(stored.toStdString())) {
        return failure(Error::NotFound, location,
                       QObject::tr("Could not remove entry (not found)."));
    }
    outcome.success = true;
    outcome.changed = true;
    outcome.location = stored;
    _window->refreshCurrentVolumePackageUi(QString(), true);
    return outcome;
}

bool ProjectAttachmentController::startLasagnaAttachment(
    const QString& requestedLocation,
    bool fiberInference,
    std::function<void(const ProjectAttachmentOutcome&)> onFinished,
    ProjectAttachmentOutcome* refusal)
{
    auto fail = [refusal](Error kind, const QString& message) {
        if (refusal) {
            refusal->errorKind = kind;
            refusal->error = message;
        }
        return false;
    };
    ProjectAttachmentOutcome check;
    if (!hasProject(&check)) {
        return fail(check.errorKind, check.error);
    }
    if (_lasagnaInFlight) {
        return fail(Error::Busy,
                    QObject::tr("A Lasagna manifest attachment is already in progress."));
    }
    const QString trimmed = requestedLocation.trimmed();
    if (trimmed.isEmpty()) {
        return fail(Error::InvalidLocation,
                    QObject::tr("Lasagna manifest location must not be blank."));
    }
    // Lasagna treats schemes case-insensitively; a local manifest is stored
    // as an absolute, normal path like the other attachments.
    const bool remote = vc::lasagna::isRemoteLasagnaLocation(trimmed.toStdString());
    const QString location = remote
        ? trimmed
        : storedLocalLocation(trimmed, _window->_state->vpkg()->path().parent_path());
    if (!remote && !QFileInfo(location).isFile()) {
        return fail(Error::InvalidLocation,
                    QObject::tr("Lasagna manifest not found: %1").arg(location));
    }

    vc::lasagna::LasagnaDatasetOpenOptions openOptions;
    bool needsRemoteCache = remote;
    bool needsRemoteAuth = remote;
    QString authLocation = location;
    if (!remote) {
        try {
            const auto localManifest = std::filesystem::path(location.toStdString());
            const auto parsed = vc::lasagna::LasagnaDatasetManifest::parseFile(localManifest);
            const auto remoteGroup = std::find_if(parsed.groups.begin(), parsed.groups.end(), [](const auto& group) {
                return vc::lasagna::isRemoteLasagnaLocation(group.relativeZarrKey);
            });
            if (remoteGroup != parsed.groups.end()) {
                needsRemoteCache = true;
                needsRemoteAuth = true;
                authLocation = QString::fromStdString(remoteGroup->relativeZarrKey);
            } else {
                const auto markerPath = localManifest.parent_path() /
                    vc::lasagna::kLasagnaRemoteMarker;
                if (std::filesystem::is_regular_file(markerPath)) {
                    const auto marker = utils::Json::parse_file(markerPath);
                    const auto artifactUrl = marker.value(
                        "artifact_url", std::string{});
                    if (vc::lasagna::isRemoteLasagnaLocation(artifactUrl)) {
                        needsRemoteAuth = true;
                        authLocation = QString::fromStdString(artifactUrl);
                    }
                }
            }
        } catch (...) {
            // The background loader reports manifest parse errors uniformly.
        }
    }
    if (needsRemoteAuth || needsRemoteCache) {
        auto* attachment = _window->_volumeAttachmentController.get();
        if (!attachment) {
            return fail(Error::Failed, QObject::tr("Remote attachment settings are unavailable."));
        }
        QString authError;
        if (!attachment->resolveRemoteAuth(authLocation, &openOptions.remoteAuth, &authError)) {
            return fail(Error::Failed, authError);
        }
    }
    if (needsRemoteCache) {
        openOptions.remoteCacheRoot = vc3d::remoteCachePathFs();
    }

    const auto targetPackage = _window->_state->vpkg();
    // Another spelling of an attached manifest updates that entry; VolumePkg
    // matches manifests by their stored string.
    const QString attached = remote
        ? QString()
        : findEntry(targetPackage->allLasagnaDatasetEntries(), location,
                    targetPackage->path().parent_path());
    const std::string persistedLocation =
        (attached.isEmpty() ? location : attached).toStdString();
    auto* watcher = new QFutureWatcher<LasagnaTaskResult>(this);
    connect(watcher, &QFutureWatcher<LasagnaTaskResult>::finished, this,
            [this, watcher, targetPackage, persistedLocation, fiberInference,
             onFinished = std::move(onFinished)]() {
        auto task = watcher->result();
        watcher->deleteLater();
        _lasagnaInFlight = false;
        ProjectAttachmentOutcome outcome;
        outcome.location = QString::fromStdString(persistedLocation);
        outcome.errorKind = Error::Failed;
        if (!task.error.isEmpty()) {
            outcome.error = task.error;
        } else if (!_window || !_window->_state || _window->_state->vpkg() != targetPackage) {
            outcome.error = QObject::tr("The open project changed while the manifest was loading.");
        } else {
            try {
                const auto result = targetPackage->attachPreparedLasagnaDataset(
                    persistedLocation, {}, fiberInference, task.volumes);
                if (result == VolumePkg::AttachLasagnaResult::VolumeIdConflict) {
                    outcome.error = QObject::tr("A Lasagna volume conflicts with an existing volume id.");
                } else {
                    outcome.success = true;
                    outcome.errorKind = Error::None;
                    outcome.changed = result == VolumePkg::AttachLasagnaResult::Attached;
                    _window->refreshCurrentVolumePackageUi(QString(), true);
                }
            } catch (const std::exception& e) {
                outcome.error = QString::fromUtf8(e.what());
            }
        }
        if (onFinished) {
            onFinished(outcome);
        }
    });

    _lasagnaInFlight = true;
    watcher->setFuture(QtConcurrent::run([persistedLocation, openOptions, fiberInference]() {
        LasagnaTaskResult result;
        try {
            const auto dataset = vc::lasagna::LasagnaDataset::openLocation(persistedLocation, openOptions);
            if (dataset.manifest().groups.empty())
                throw std::runtime_error("Lasagna manifest contains no channel groups");
            if (fiberInference)
                (void)vc::fiber_tracer::resolveFiberPredictionTraceScales(dataset.manifest());
            else if (!dataset.hasNormalSource())
                throw std::runtime_error("regular Lasagna data requires grad_mag, nx, and ny normal channels");
            for (auto& prepared : vc::lasagna::prepareLasagnaProjectVolumes(
                     dataset, persistedLocation)) {
                result.volumes.push_back({std::move(prepared.location), std::move(prepared.tags), std::move(prepared.volume)});
            }
        } catch (const std::exception& e) {
            result.error = QString::fromUtf8(e.what());
        } catch (...) {
            result.error = QObject::tr("Unknown error while loading the Lasagna manifest.");
        }
        return result;
    }));
    return true;
}

ProjectAttachmentOutcome ProjectAttachmentController::saveProjectAs(const QString& path)
{
    ProjectAttachmentOutcome outcome;
    if (!hasProject(&outcome)) {
        return outcome;
    }
    QString file = path.trimmed();
    if (file.isEmpty()) {
        return failure(Error::InvalidLocation, file, QObject::tr("Project path must not be blank."));
    }
    if (!file.endsWith(QStringLiteral(".volpkg.json"), Qt::CaseInsensitive)) {
        file += QStringLiteral(".volpkg.json");
    }
    file = QFileInfo(file).absoluteFilePath();
    outcome.location = file;

    auto pkg = _window->_state->vpkg();
    const fs::path oldDir = pkg->path().parent_path();
    const fs::path newDir = fs::path(file.toStdString()).parent_path();
    try {
        pkg->save(fs::path(file.toStdString()));
    } catch (const std::exception& e) {
        return failure(Error::Failed, file, QString::fromUtf8(e.what()));
    }
    outcome.success = true;
    outcome.changed = true;

    std::error_code ec;
    if (!fs::equivalent(oldDir, newDir, ec)) {
        QStringList relative;
        auto collect = [&relative](const std::vector<vc::project::Entry>& entries) {
            for (const auto& entry : entries) {
                if (!vc::project::isLocationRemote(entry.location) &&
                    !vc::project::resolveLocalPath(entry.location).is_absolute()) {
                    relative << QString::fromStdString(entry.location);
                }
            }
        };
        collect(pkg->volumeEntries());
        collect(pkg->segmentEntries());
        collect(pkg->normalGridEntries());
        collect(pkg->allLasagnaDatasetEntries());
        collect({{pkg->umbilicus(), {}}});
        relative.removeAll(QString());
        if (!relative.isEmpty()) {
            outcome.warnings << QObject::tr("These relative locations now resolve against %1: %2")
                                    .arg(QString::fromStdString(newDir.string()),
                                         relative.join(QStringLiteral(", ")));
        }
    }

    FiberCollectionController* collection = _window->_fiberCollection.get();
    if (collection && !collection->collectionPath().isEmpty() &&
        !collection->saveAttachmentRecord()) {
        outcome.warnings << QObject::tr("The automated fiber volume's attachment could not "
                                        "be saved beside the new project file.");
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::project::DEFAULT_PATH, QFileInfo(file).absolutePath());
    return outcome;
}
