#include "ProjectAttachmentController.hpp"

#include <QFileInfo>
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
#include "VCSettings.hpp"
#include "VolumeAttachmentController.hpp"

#include "utils/Json.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/ProjectVolumes.hpp"

namespace {

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

bool ProjectAttachmentController::hasProject(QString* error) const
{
    if (_window && _window->_state && _window->_state->vpkg()) {
        return true;
    }
    if (error) {
        *error = QObject::tr("Open or create a project first.");
    }
    return false;
}

bool ProjectAttachmentController::startLasagnaAttachment(
    const QString& requestedLocation,
    bool fiberInference,
    std::function<void(const ProjectAttachmentOutcome&)> onFinished,
    QString* error)
{
    auto fail = [error](const QString& message) {
        if (error) *error = message;
        return false;
    };
    QString projectError;
    if (!hasProject(&projectError)) {
        return fail(projectError);
    }
    if (_lasagnaInFlight) {
        return fail(QObject::tr("A Lasagna manifest attachment is already in progress."));
    }
    const QString location = requestedLocation;
    const bool remote = vc::lasagna::isRemoteLasagnaLocation(location.toStdString());

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
            return fail(QObject::tr("Remote attachment settings are unavailable."));
        }
        QString authError;
        if (!attachment->resolveRemoteAuth(authLocation, &openOptions.remoteAuth, &authError)) {
            return fail(authError);
        }
    }
    if (needsRemoteCache) {
        openOptions.remoteCacheRoot = vc3d::remoteCachePathFs();
    }

    const auto targetPackage = _window->_state->vpkg();
    const std::string persistedLocation = location.toStdString();
    auto* watcher = new QFutureWatcher<LasagnaTaskResult>(this);
    connect(watcher, &QFutureWatcher<LasagnaTaskResult>::finished, this,
            [this, watcher, targetPackage, persistedLocation, location, fiberInference,
             onFinished = std::move(onFinished)]() {
        auto task = watcher->result();
        watcher->deleteLater();
        _lasagnaInFlight = false;
        ProjectAttachmentOutcome outcome;
        outcome.location = location;
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
