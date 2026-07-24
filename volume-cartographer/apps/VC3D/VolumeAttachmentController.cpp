#include "VolumeAttachmentController.hpp"

#include <QDir>
#include <QFutureWatcher>
#include <QInputDialog>
#include <QLineEdit>
#include <QSettings>
#include <QtConcurrent>

#include <exception>
#include <filesystem>
#include <memory>
#include <utility>

#include "CState.hpp"
#include "CWindow.hpp"
#include "RemoteVolumeCachePaths.hpp"
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
    VolumeAttachmentPresentation presentation,
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
    if (vc::project::isLocationRemote(input)) {
        const auto spec = vc::parseRemoteVolumeSpec(input);
        prepared.location = QString::fromStdString(spec.portableLocator);
        if (!resolveRemoteAuth(prepared.location, &prepared.auth, errorMessage)) {
            if (failure)
                *failure = VolumeAttachmentPreparationFailure::RemoteConfiguration;
            return false;
        }
        prepared.remoteCacheRoot = remoteCacheDirectory(presentation);
        if (prepared.remoteCacheRoot.isEmpty()) {
            if (errorMessage && errorMessage->isEmpty() &&
                presentation == VolumeAttachmentPresentation::Silent) {
                *errorMessage = QObject::tr("Could not resolve the remote volume cache.");
            }
            if (failure)
                *failure = VolumeAttachmentPreparationFailure::RemoteConfiguration;
            return false;
        }
    } else {
        prepared.location = QString::fromStdString(
            vc::project::resolveLocalPath(input).lexically_normal().string());
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
                        request.remoteCacheRoot,
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
                if (vc::project::isLocationRemote(location)) {
                    result.volume = Volume::NewFromUrl(
                        location,
                        request.remoteCacheRoot.toStdString(),
                        request.auth,
                        vc::project::volumeMetadataFromEntryTags(request.tags));
                } else {
                    result.volume = Volume::New(std::filesystem::path(location));
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

bool VolumeAttachmentController::resolveRemoteAuth(
    const QString& url,
    vc::HttpAuth* authOut,
    QString* errorMessage) const
{
    if (!authOut)
        return false;

    *authOut = {};
    const auto spec = vc::parseRemoteVolumeSpec(url.trimmed().toStdString());
    if (!spec.useAwsSigv4)
        return true;

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    *authOut = vc::loadAwsCredentials();
    if (authOut->region.empty())
        authOut->region = spec.awsRegion;

    if (authOut->access_key.empty() || authOut->secret_key.empty()) {
        const auto savedAccess =
            settings.value(vc3d::settings::aws::ACCESS_KEY).toString();
        const auto savedSecret =
            settings.value(vc3d::settings::aws::SECRET_KEY).toString();
        const auto savedToken =
            settings.value(vc3d::settings::aws::SESSION_TOKEN).toString();
        if (!savedAccess.isEmpty() && !savedSecret.isEmpty()) {
            authOut->access_key = savedAccess.toStdString();
            authOut->secret_key = savedSecret.toStdString();
            authOut->session_token = savedToken.toStdString();
        }
    }

    // Public S3 buckets should attempt anonymous access before the interactive
    // caller offers a credential retry.
    if (errorMessage)
        errorMessage->clear();
    return true;
}

QString VolumeAttachmentController::suggestedRemoteCacheDirectory() const
{
    if (_window && _window->_state && _window->_state->vpkg()) {
        const QString projectDir = QString::fromStdString(
            _window->_state->vpkg()->getVolpkgDirectory());
        if (!projectDir.isEmpty()) {
            return vc3d::remoteCachePath(
                QDir(projectDir).filePath("remote_cache"));
        }
    }
    return vc3d::remoteCachePath();
}

QString VolumeAttachmentController::configuredRemoteCacheDirectory() const
{
    if (_window && _window->_state && _window->_state->vpkg()) {
        const QString persisted = QString::fromStdString(
            _window->_state->vpkg()->remoteCacheRootOrEmpty()).trimmed();
        return vc3d::remoteCachePath(persisted);
    }
    return {};
}

QString VolumeAttachmentController::remoteCacheDirectory(
    VolumeAttachmentPresentation presentation)
{
    QString cacheDir = configuredRemoteCacheDirectory();
    if (cacheDir.isEmpty() &&
        presentation == VolumeAttachmentPresentation::Interactive) {
        bool ok = false;
        cacheDir = QInputDialog::getText(
            _window,
            QObject::tr("Remote Cache Location"),
            QObject::tr(
                "Choose where this project should store downloaded remote "
                "volume chunks."),
            QLineEdit::Normal,
            suggestedRemoteCacheDirectory(),
            &ok).trimmed();
        if (!ok)
            return {};
        cacheDir = vc3d::remoteCachePath(cacheDir);
    }

    if (cacheDir.isEmpty()) {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        cacheDir = vc3d::remoteCachePath(
            settings.value(vc3d::settings::viewer::REMOTE_CACHE_DIR).toString());
    }
    if (QDir::isRelativePath(cacheDir)) {
        cacheDir =
            QDir::cleanPath(QDir::current().absoluteFilePath(cacheDir));
    }
    return QDir().mkpath(cacheDir) ? cacheDir : QString{};
}
