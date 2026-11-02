#include "VolumeAttachmentController.hpp"
#include <vc_reader/zarr.hpp>
#include "OpenDataManifest.hpp"

#include <QFutureWatcher>
#include <QtConcurrent>

#include <exception>
#include <filesystem>
#include <memory>
#include <utility>

#include "CState.hpp"
#include "CWindow.hpp"
#include "VCSettings.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/RemoteUrl.hpp"

struct VolumeAttachmentController::TaskResult {
    std::shared_ptr<Volume> volume;
    QString error;
};

VolumeAttachmentController::VolumeAttachmentController(CWindow* window)
    : QObject(window)
    , _window(window)
{
}

bool VolumeAttachmentController::prepare(
    const QString& location,
    std::vector<std::string> tags,
    VolumeAttachmentRequest* request,
    QString* errorMessage,
    VolumeAttachmentPreparationFailure* failure)
{
    if (failure)
        *failure = VolumeAttachmentPreparationFailure::None;
    if (!request) {
        if (errorMessage)
            *errorMessage = QObject::tr("No attachment request output was provided.");
        if (failure)
            *failure = VolumeAttachmentPreparationFailure::RemoteConfiguration;
        return false;
    }
    if (!_window || !_window->_state || !_window->_state->vpkg()) {
        if (errorMessage)
            *errorMessage = QObject::tr("Open a volume package before attaching a volume.");
        if (failure)
            *failure = VolumeAttachmentPreparationFailure::NoProject;
        return false;
    }

    const QString trimmed = location.trimmed();
    if (trimmed.isEmpty()) {
        if (errorMessage)
            *errorMessage = QObject::tr("Volume location must not be blank.");
        if (failure)
            *failure = VolumeAttachmentPreparationFailure::InvalidLocation;
        return false;
    }

    const std::string input = trimmed.toStdString();
    const std::string validationError =
        vc::project::validateSingleVolumeLocation(input);
    if (!validationError.empty()) {
        if (errorMessage)
            *errorMessage = QString::fromStdString(validationError);
        if (failure)
            *failure = VolumeAttachmentPreparationFailure::InvalidLocation;
        return false;
    }

    VolumeAttachmentRequest prepared;
    if (vcr::zarr::isRemote(input)) {
        const auto spec = vc::parseRemoteVolumeSpec(input);
        prepared.location = QString::fromStdString(spec.portableLocator);
    } else {
        auto localPath = vc::project::resolveLocalPath(input);
        if (localPath.is_relative())
            localPath = std::filesystem::absolute(localPath);
        prepared.location = QString::fromStdString(
            localPath.lexically_normal().string());
    }

    prepared.tags = std::move(tags);
    if (auto existing = _window->_state->vpkg()->matchingVolumeEntry(
            prepared.location.toStdString())) {
        prepared.tags = std::move(existing->tags);
    }

    *request = std::move(prepared);
    if (errorMessage)
        errorMessage->clear();
    return true;
}

bool VolumeAttachmentController::start(
    VolumeAttachmentRequest request,
    std::function<void(const VolumeAttachmentOutcome&)> onFinished,
    QString* errorMessage)
{
    if (!_window || !_window->_state || !_window->_state->vpkg()) {
        if (errorMessage)
            *errorMessage = QObject::tr("Open a volume package before attaching a volume.");
        return false;
    }
    if (_inFlight) {
        if (errorMessage)
            *errorMessage = QObject::tr("A volume attachment is already in progress.");
        return false;
    }

    const auto targetPackage = _window->_state->vpkg();
    const VolumeAttachmentRequest loadRequest = request;
    auto* watcher = new QFutureWatcher<TaskResult>(this);
    connect(
        watcher,
        &QFutureWatcher<TaskResult>::finished,
        this,
        [this,
         watcher,
         targetPackage,
         request = std::move(request),
         onFinished = std::move(onFinished)]() mutable {
            TaskResult task = watcher->result();
            _inFlight = false;
            watcher->deleteLater();

            VolumeAttachmentOutcome outcome;
            outcome.location = request.location;
            outcome.error = task.error;
            if (!task.volume) {
                if (outcome.error.isEmpty()) {
                    outcome.error = QObject::tr(
                        "The volume loader returned no volume.");
                }
            } else if (!_window || !_window->_state ||
                       _window->_state->vpkg() != targetPackage) {
                outcome.error = QObject::tr(
                    "The open project changed while the volume was loading.");
            } else {
                outcome.volumeId = QString::fromStdString(task.volume->id());
                outcome.projectPath =
                    QString::fromStdString(targetPackage->path().string());
                try {
                    const QString preferredVolumeId =
                        request.selection == VolumeAttachmentSelection::SelectAttached
                            ? outcome.volumeId
                            : QString{};
                    const auto result = _window->attachVolumeToCurrentPackage(
                        task.volume,
                        request.location,
                        std::move(request.tags),
                        preferredVolumeId);
                    if (result == CWindow::VolumeAttachResult::VolumeIdConflict) {
                        outcome.error = QObject::tr(
                            "A different volume with id '%1' is already attached.")
                                            .arg(outcome.volumeId);
                    } else {
                        outcome.success = true;
                        outcome.alreadyAttached =
                            result == CWindow::VolumeAttachResult::AlreadyAttached;
                    }
                } catch (const std::exception& error) {
                    outcome.error = QString::fromUtf8(error.what());
                } catch (...) {
                    outcome.error = QObject::tr(
                        "Unknown error while updating the volume package.");
                }
            }

            if (onFinished)
                onFinished(outcome);
        });

    _inFlight = true;
    watcher->setFuture(QtConcurrent::run(
        [request = loadRequest]() {
            TaskResult result;
            try {
                const std::string location = request.location.toStdString();
                if (vcr::zarr::isRemote(location)) {
                    const bool anonymous = std::find(
                        request.tags.begin(), request.tags.end(),
                        vc::project::kAnonymousRemoteAuthTag) !=
                        request.tags.end();
                    result.volume = Volume::NewFromUrl(
                        location,
                        vc::project::catalogVolumeMetadata(
                            vc3d::opendata::cachedCatalog().get(), location),
                        !anonymous);
                } else {
                    result.volume = Volume::New(std::filesystem::path(location), nlohmann::json::object());
                }
            } catch (const std::exception& error) {
                result.error = QString::fromUtf8(error.what());
            } catch (...) {
                result.error = QObject::tr("Unknown error while loading the volume.");
            }
            return result;
        }));
    if (errorMessage)
        errorMessage->clear();
    return true;
}

bool VolumeAttachmentController::inFlight() const
{
    return _inFlight;
}

