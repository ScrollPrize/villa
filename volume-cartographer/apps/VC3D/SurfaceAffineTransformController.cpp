#include "SurfaceAffineTransformController.hpp"

#include "AxisAlignedSliceController.hpp"
#include "CState.hpp"
#include "SurfacePanelController.hpp"
#include "ViewerManager.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "viewer_controls/ViewerControlsPanel.hpp"
#include "viewer_controls/panels/ViewerTransformsPanel.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/AffineTransform.hpp"
#include "vc/core/util/HttpFetch.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/RemoteUrl.hpp"

#include <QDir>
#include <QFutureWatcher>
#include <QInputDialog>
#include <QLineEdit>
#include <QMessageBox>
#include <QRegularExpression>
#include <QSettings>
#include <QTemporaryDir>
#include <QtConcurrent/QtConcurrent>

#include <filesystem>
#include <stdexcept>
#include <utility>

#include "VCSettings.hpp"

using vc::core::util::cloneSurfaceForTransform;
using vc::core::util::invertAffineTransformMatrix;
using vc::core::util::loadAffineTransformMatrix;
using vc::core::util::loadAffineTransformMatrixFromString;
using vc::core::util::refreshTransformedSurfaceState;
using vc::core::util::transformSurfacePoints;

namespace
{

bool isRemoteTransformSource(const QString& source)
{
    const QString trimmed = source.trimmed();
    return trimmed.startsWith("http://", Qt::CaseInsensitive) ||
           trimmed.startsWith("https://", Qt::CaseInsensitive) ||
           trimmed.startsWith("s3://", Qt::CaseInsensitive) ||
           trimmed.startsWith("s3+", Qt::CaseInsensitive);
}

std::filesystem::path expandLocalTransformPath(const QString& source)
{
    QString path = source.trimmed();
    if (path.startsWith("~/")) {
        path.replace(0, 1, QDir::homePath());
    } else if (path == "~") {
        path = QDir::homePath();
    }

    std::filesystem::path fsPath = path.toStdString();
    if (fsPath.is_relative()) {
        fsPath = std::filesystem::absolute(fsPath);
    }
    return fsPath;
}

vc::HttpAuth authForRemoteTransformSource(const QString& source)
{
    vc::HttpAuth auth;
    const auto resolved = vc::resolveRemoteUrl(source.trimmed().toStdString());
    if (!resolved.useAwsSigv4) {
        return auth;
    }

    auth = vc::loadAwsCredentials();
    if (auth.region.empty())
        auth.region = resolved.awsRegion;
    if (auth.access_key.empty() || auth.secret_key.empty()) {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        const auto savedAccess = settings.value(vc3d::settings::aws::ACCESS_KEY).toString();
        const auto savedSecret = settings.value(vc3d::settings::aws::SECRET_KEY).toString();
        const auto savedToken = settings.value(vc3d::settings::aws::SESSION_TOKEN).toString();

        if (!savedAccess.isEmpty() && !savedSecret.isEmpty()) {
            auth.access_key = savedAccess.toStdString();
            auth.secret_key = savedSecret.toStdString();
            auth.session_token = savedToken.toStdString();
        }
    }

    return auth;
}

} // namespace

SurfaceAffineTransformController::SurfaceAffineTransformController(const Deps& deps, QObject* parent)
    : QObject(parent)
    , _state(deps.state)
    , _viewerControlsPanel(deps.viewerControlsPanel)
    , _viewerManager(deps.viewerManager)
    , _segmentationModule(deps.segmentationModule)
    , _surfacePanel(deps.surfacePanel)
    , _axisAlignedSliceController(deps.axisAlignedSliceController)
    , _dialogParent(deps.dialogParent)
    , _showStatus(deps.showStatus)
{
    connectPanelSignals();
}

void SurfaceAffineTransformController::connectPanelSignals()
{
    auto* transformPanel = transformsPanel();
    if (!transformPanel) {
        return;
    }

    connect(transformPanel, &ViewerTransformsPanel::previewToggled,
            this, &SurfaceAffineTransformController::onPreviewTransformToggled);
    connect(transformPanel, &ViewerTransformsPanel::stateChanged,
            this, &SurfaceAffineTransformController::refresh);
    connect(transformPanel, &ViewerTransformsPanel::loadAffineRequested,
            this, &SurfaceAffineTransformController::onLoadAffineRequested);
    connect(transformPanel, &ViewerTransformsPanel::saveTransformedRequested,
            this, &SurfaceAffineTransformController::onSaveTransformedRequested);
}

void SurfaceAffineTransformController::showStatus(const QString& text, int timeoutMs)
{
    if (_showStatus) {
        _showStatus(text, timeoutMs);
    }
}

std::shared_ptr<QuadSurface> SurfaceAffineTransformController::currentTransformSourceSurface() const
{
    if (!_state) {
        return nullptr;
    }

    if (auto active = _state->activeSurface().lock()) {
        return active;
    }

    if (_state->vpkg()) {
        const std::string activeId = _state->activeSurfaceId();
        if (!activeId.empty()) {
            if (auto surface = _state->vpkg()->getSurface(activeId)) {
                return surface;
            }
        }
    }

    auto segmentationSurface = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));
    if (segmentationSurface && segmentationSurface != _transformPreviewSurface) {
        return segmentationSurface;
    }

    return _transformPreviewSourceSurface;
}

QString SurfaceAffineTransformController::currentTransformSourceDescription() const
{
    if (!_customTransformSource.trimmed().isEmpty()) {
        return _customTransformSource.trimmed();
    }

    const auto currentVolume = _state ? _state->currentVolume() : nullptr;
    const auto remoteTransformUrl = currentRemoteTransformJsonUrl();
    if (currentVolume && currentVolume->isRemote() && !remoteTransformUrl.empty()) {
        return QString::fromStdString(remoteTransformUrl);
    }

    if (currentVolume && !currentVolume->path().empty()) {
        return QString::fromStdString((currentVolume->path() / "transform.json").string());
    }

    const auto localPath = localCurrentTransformJsonPath();
    if (!localPath.empty()) {
        return QString::fromStdString(localPath.string());
    }

    return {};
}

std::filesystem::path SurfaceAffineTransformController::localCurrentTransformJsonPath() const
{
    if (!_customTransformSource.trimmed().isEmpty()) {
        return _customTransformLocalPath;
    }

    if (!_state) {
        return {};
    }

    auto currentVolume = _state->currentVolume();
    if (!currentVolume) {
        return {};
    }

    const auto volumePath = currentVolume->path();
    if (volumePath.empty()) {
        return {};
    }

    const auto localTransformPath = volumePath / "transform.json";
    if (std::filesystem::exists(localTransformPath)) {
        return localTransformPath;
    }

    return {};
}

std::string SurfaceAffineTransformController::currentRemoteTransformJsonUrl() const
{
    if (!_state) {
        return {};
    }

    auto currentVolume = _state->currentVolume();
    if (!currentVolume || !currentVolume->isRemote() || currentVolume->remoteUrl().empty()) {
        return {};
    }

    std::string remoteTransformUrl = currentVolume->remoteUrl();
    while (!remoteTransformUrl.empty() && remoteTransformUrl.back() == '/') {
        remoteTransformUrl.pop_back();
    }
    remoteTransformUrl += "/transform.json";
    return remoteTransformUrl;
}

void SurfaceAffineTransformController::ensureCurrentRemoteTransformJsonAsync()
{
    if (!_state) {
        return;
    }

    auto currentVolume = _state->currentVolume();
    if (!currentVolume || !currentVolume->isRemote() || currentVolume->remoteUrl().empty()) {
        return;
    }

    const auto remoteTransformUrl = currentRemoteTransformJsonUrl();
    if (remoteTransformUrl.empty()) {
        return;
    }

    auto& fetchState = _remoteTransformFetchStates[remoteTransformUrl];
    if (fetchState != RemoteTransformFetchState::Unknown) {
        return;
    }

    fetchState = RemoteTransformFetchState::Pending;
    const auto auth = currentVolume->remoteAuth();
    auto* watcher = new QFutureWatcher<std::optional<cv::Matx44d>>(this);
    connect(watcher, &QFutureWatcher<std::optional<cv::Matx44d>>::finished, this,
            [this, watcher, remoteTransformUrl]() {
                watcher->deleteLater();

                std::optional<cv::Matx44d> matrix;
                try {
                    matrix = watcher->result();
                } catch (const std::exception&) {
                    matrix.reset();
                }

                if (matrix) {
                    _remoteTransformMatrices[remoteTransformUrl] = *matrix;
                    _remoteTransformFetchStates[remoteTransformUrl] = RemoteTransformFetchState::Available;
                } else {
                    _remoteTransformMatrices.erase(remoteTransformUrl);
                    _remoteTransformFetchStates[remoteTransformUrl] = RemoteTransformFetchState::Missing;
                }

                if (currentRemoteTransformJsonUrl() == remoteTransformUrl) {
                    refresh();
                }
            });
    watcher->setFuture(QtConcurrent::run(
        [remoteTransformUrl, auth]() -> std::optional<cv::Matx44d> {
            const auto body = vc::httpGetString(remoteTransformUrl, auth);
            if (body.empty())
                return std::nullopt;
            return loadAffineTransformMatrixFromString(body);
        }));
}

bool SurfaceAffineTransformController::setCustomTransformSource(const QString& source, QString* errorMessage)
{
    const QString trimmed = source.trimmed();
    if (trimmed.isEmpty()) {
        _customTransformSource.clear();
        _customTransformLocalPath.clear();
        _customTransformMatrix.reset();
        return true;
    }

    try {
        if (isRemoteTransformSource(trimmed)) {
            const auto resolved = vc::resolveRemoteUrl(trimmed.toStdString());
            const auto auth = authForRemoteTransformSource(trimmed);
            const auto body = vc::httpGetString(resolved.httpsUrl, auth);
            _customTransformMatrix = loadAffineTransformMatrixFromString(body);
            _customTransformLocalPath.clear();
        } else {
            auto resolvedPath = expandLocalTransformPath(trimmed);
            _customTransformMatrix = loadAffineTransformMatrix(resolvedPath);
            _customTransformLocalPath = std::move(resolvedPath);
        }

        _customTransformSource = trimmed;
        return true;
    } catch (const std::exception& ex) {
        if (errorMessage) {
            *errorMessage = QString::fromUtf8(ex.what());
        }
        return false;
    }
}

std::optional<cv::Matx44d> SurfaceAffineTransformController::currentTransformMatrix(bool allowRemoteFetch)
{
    if (!_customTransformSource.trimmed().isEmpty()) {
        return _customTransformMatrix;
    }

    if (const auto localTransformPath = localCurrentTransformJsonPath();
        !localTransformPath.empty()) {
        const auto remoteTransformUrl = currentRemoteTransformJsonUrl();
        if (!remoteTransformUrl.empty()) {
            _remoteTransformFetchStates[remoteTransformUrl] = RemoteTransformFetchState::Available;
        }
        return loadAffineTransformMatrix(localTransformPath);
    }

    if (!_state) {
        return std::nullopt;
    }

    auto currentVolume = _state->currentVolume();
    if (!currentVolume) {
        return std::nullopt;
    }

    if (!currentVolume->isRemote() || currentVolume->remoteUrl().empty()) {
        return std::nullopt;
    }

    const auto remoteTransformUrl = currentRemoteTransformJsonUrl();
    if (remoteTransformUrl.empty())
        return std::nullopt;

    if (auto it = _remoteTransformMatrices.find(remoteTransformUrl);
        it != _remoteTransformMatrices.end()) {
        return it->second;
    }

    if (!allowRemoteFetch) {
        return std::nullopt;
    }

    const auto body = vc::httpGetString(remoteTransformUrl, currentVolume->remoteAuth());
    if (!body.empty()) {
        auto matrix = loadAffineTransformMatrixFromString(body);
        _remoteTransformMatrices[remoteTransformUrl] = matrix;
        _remoteTransformFetchStates[remoteTransformUrl] = RemoteTransformFetchState::Available;
        return matrix;
    }

    _remoteTransformMatrices.erase(remoteTransformUrl);
    _remoteTransformFetchStates[remoteTransformUrl] = RemoteTransformFetchState::Missing;
    return std::nullopt;
}

void SurfaceAffineTransformController::clearPreview(bool restoreDisplayedSurface)
{
    if (!_state) {
        _transformPreviewSurface.reset();
        _transformPreviewSourceSurface.reset();
        return;
    }

    auto currentDisplayed = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));
    const bool showingPreview = (_transformPreviewSurface && currentDisplayed == _transformPreviewSurface);

    if (restoreDisplayedSurface && showingPreview) {
        auto restoreSurface = currentTransformSourceSurface();
        if (!restoreSurface) {
            restoreSurface = _transformPreviewSourceSurface;
        }
        _state->setSurface("segmentation", restoreSurface, false, false);
        if (_axisAlignedSliceController) {
            _axisAlignedSliceController->applyOrientation(restoreSurface.get());
        }
    }

    _transformPreviewSurface.reset();
    _transformPreviewSourceSurface.reset();
}

bool SurfaceAffineTransformController::applyTransformPreview(bool allowRemoteFetch)
{
    if (!_state || (_segmentationModule && _segmentationModule->editingEnabled())) {
        return false;
    }

    auto sourceSurface = currentTransformSourceSurface();
    if (!sourceSurface) {
        return false;
    }

    auto* transformPanel = transformsPanel();
    const int scale = transformPanel ? transformPanel->scaleValue() : 1;
    const bool scaleOnly = transformPanel && transformPanel->scaleOnlyChecked();
    std::optional<cv::Matx44d> matrix;
    if (!scaleOnly) {
        matrix = currentTransformMatrix(allowRemoteFetch);
        if (matrix) {
            if (transformPanel && transformPanel->invertChecked()) {
                matrix = invertAffineTransformMatrix(*matrix);
            }
        } else if (scale == 1) {
            return false;
        }
    } else if (scale == 1) {
        return false;
    }

    auto previewSurface = cloneSurfaceForTransform(sourceSurface);
    if (!previewSurface) {
        return false;
    }

    previewSurface->path.clear();
    previewSurface->id.clear();

    transformSurfacePoints(previewSurface.get(), scale, matrix);
    refreshTransformedSurfaceState(previewSurface.get());
    if (_viewerManager) {
        _viewerManager->refreshSurfacePatchIndex(previewSurface);
    }

    clearPreview(false);
    _transformPreviewSourceSurface = sourceSurface;
    _transformPreviewSurface = previewSurface;
    _state->setSurface("segmentation", previewSurface, false, false);
    if (_axisAlignedSliceController) {
        _axisAlignedSliceController->applyOrientation(previewSurface.get());
    }
    return true;
}

ViewerTransformsPanel* SurfaceAffineTransformController::transformsPanel() const
{
    return _viewerControlsPanel ? _viewerControlsPanel->transformsPanel() : nullptr;
}

void SurfaceAffineTransformController::refresh()
{
    auto* transformPanel = transformsPanel();
    if (!transformPanel) {
        return;
    }

    const bool editingEnabled = _segmentationModule && _segmentationModule->editingEnabled();
    const auto sourceSurface = currentTransformSourceSurface();
    const auto currentVolume = _state ? _state->currentVolume() : nullptr;
    const bool scaleOnly = transformPanel->scaleOnlyChecked();
    const auto localTransformPath = localCurrentTransformJsonPath();
    bool hasTransform = _customTransformMatrix.has_value() ||
                        (!localTransformPath.empty() && std::filesystem::exists(localTransformPath));
    const int scale = transformPanel->scaleValue();
    const bool hasScaleOnlyTransform = scale != 1;
    const bool hasCustomTransform = !_customTransformSource.trimmed().isEmpty();
    const auto remoteTransformUrl = currentRemoteTransformJsonUrl();
    RemoteTransformFetchState remoteFetchState = RemoteTransformFetchState::Unknown;
    if (!scaleOnly && !hasCustomTransform && currentVolume && currentVolume->isRemote() &&
        !remoteTransformUrl.empty()) {
        if (auto it = _remoteTransformMatrices.find(remoteTransformUrl);
            it != _remoteTransformMatrices.end()) {
            hasTransform = true;
            _remoteTransformFetchStates[remoteTransformUrl] = RemoteTransformFetchState::Available;
        } else {
            ensureCurrentRemoteTransformJsonAsync();
            auto stateIt = _remoteTransformFetchStates.find(remoteTransformUrl);
            if (stateIt != _remoteTransformFetchStates.end()) {
                remoteFetchState = stateIt->second;
            }
        }
    }
    const bool previewEnabled =
        sourceSurface && !editingEnabled &&
        (hasScaleOnlyTransform || (!scaleOnly && hasTransform));
    const bool saveEnabled = previewEnabled && sourceSurface && !sourceSurface->path.empty();

    const QString transformLocation = currentTransformSourceDescription();

    QString statusText;
    if (!_state || !_state->vpkg()) {
        statusText = tr("Open a volume package to use transforms.");
    } else if (!_state->currentVolume()) {
        statusText = tr("Select a volume to load transform.json.");
    } else if (!sourceSurface) {
        statusText = tr("Select a segmentation to preview or save its transform.");
    } else if (editingEnabled) {
        statusText = tr("Transform preview is unavailable while segmentation editing is enabled.");
    } else if (scaleOnly) {
        if (hasScaleOnlyTransform) {
            statusText = hasTransform
                ? tr("Scaling points by %1 only. Affine from %2 is ignored.")
                      .arg(scale)
                      .arg(transformLocation)
                : tr("Scaling points by %1 only. No affine will be applied.")
                      .arg(scale);
        } else {
            statusText = hasTransform
                ? tr("Scale only is enabled. Affine from %1 will be ignored until scale is greater than 1.")
                      .arg(transformLocation)
                : tr("Scale only is enabled. Increase scale above 1 to preview or save.");
        }
    } else if (!hasTransform && remoteFetchState == RemoteTransformFetchState::Pending) {
        if (hasScaleOnlyTransform) {
            statusText = tr("Scaling points by %1 while checking %2 for transform.json.")
                .arg(scale)
                .arg(transformLocation);
        } else {
            statusText = tr("Checking %1 for transform.json.")
                .arg(transformLocation);
        }
    } else if (!hasTransform) {
        if (hasScaleOnlyTransform) {
            statusText = hasCustomTransform
                ? tr("Scaling points by %1. No affine was loaded from %2.")
                      .arg(scale)
                      .arg(transformLocation)
                : tr("Scaling points by %1. No affine transform was found at %2.")
                      .arg(scale)
                      .arg(transformLocation);
        } else {
            statusText = hasCustomTransform
                ? tr("No affine was loaded from %1")
                      .arg(transformLocation)
                : tr("No transform.json found at %1")
                      .arg(transformLocation);
        }
    } else {
        statusText = hasCustomTransform
            ? tr("Using custom affine %1%2 with scale %3")
                  .arg(transformLocation,
                       (transformPanel->invertChecked() ? tr(" (inverted)") : QString()))
                  .arg(scale)
            : tr("Using %1%2 with scale %3")
                  .arg(transformLocation,
                       (transformPanel->invertChecked() ? tr(" (inverted)") : QString()))
                  .arg(scale);
    }

    if (!previewEnabled && transformPanel->previewChecked()) {
        transformPanel->setPreviewChecked(false);
        clearPreview(true);
    } else if (previewEnabled && transformPanel->previewChecked()) {
        try {
            if (!applyTransformPreview(false)) {
                clearPreview(true);
            }
        } catch (const std::exception& ex) {
            clearPreview(true);
            transformPanel->setPreviewChecked(false);
            statusText = tr("Failed to load transform.json: %1")
                .arg(QString::fromUtf8(ex.what()));
        }
    } else if (!transformPanel->previewChecked()) {
        clearPreview(true);
    }

    transformPanel->applyUiState(ViewerTransformsPanel::UiState{
        .previewAvailable = previewEnabled,
        .sourceAvailable = static_cast<bool>(sourceSurface),
        .editingEnabled = editingEnabled,
        .affineAvailable = hasTransform,
        .scaleOnly = scaleOnly,
        .saveAvailable = saveEnabled,
        .statusText = statusText,
    });
}

void SurfaceAffineTransformController::onPreviewTransformToggled(bool enabled)
{
    auto* transformPanel = transformsPanel();
    if (!transformPanel) {
        return;
    }

    if (!enabled) {
        clearPreview(true);
        refresh();
        return;
    }

    try {
        if (!applyTransformPreview()) {
            throw std::runtime_error("transform preview is unavailable for the current selection");
        }
    } catch (const std::exception& ex) {
        clearPreview(true);
        transformPanel->setPreviewChecked(false);
        showStatus(tr("Failed to preview transform: %1")
                                     .arg(QString::fromUtf8(ex.what())),
                                 5000);
    }

    refresh();
}

void SurfaceAffineTransformController::onSaveTransformedRequested()
{
    if (!_state || !_state->vpkg()) {
        return;
    }

    if (_segmentationModule && _segmentationModule->editingEnabled()) {
        QMessageBox::warning(_dialogParent, tr("Editing Active"),
                             tr("Disable segmentation editing before saving a transformed surface."));
        return;
    }

    auto sourceSurface = currentTransformSourceSurface();
    if (!sourceSurface || sourceSurface->path.empty()) {
        QMessageBox::warning(_dialogParent, tr("No Segmentation"),
                             tr("Select a segmentation with files on disk first."));
        return;
    }

    auto* transformPanel = transformsPanel();
    const int scale = transformPanel ? transformPanel->scaleValue() : 1;
    const bool scaleOnly = transformPanel && transformPanel->scaleOnlyChecked();
    std::optional<cv::Matx44d> matrix = scaleOnly
        ? std::nullopt
        : currentTransformMatrix();
    const bool hasTransform = matrix.has_value();
    if (!hasTransform && scale == 1) {
        QMessageBox::warning(_dialogParent, tr("Missing Transform"),
                             scaleOnly
                                 ? tr("Scale only is enabled, and scale is set to 1.")
                                 : (_customTransformSource.trimmed().isEmpty()
                                        ? tr("No transform.json was found for the current volume, and scale is set to 1.")
                                        : tr("The selected affine could not be loaded, and scale is set to 1.")));
        return;
    }

    const QString defaultName = QString::fromStdString(sourceSurface->id.empty()
        ? sourceSurface->path.filename().string() + "_transformed"
        : sourceSurface->id + "_transformed");
    bool ok = false;
    QString newName = QInputDialog::getText(_dialogParent,
                                            tr("Save Transformed"),
                                            tr("New surface name:"),
                                            QLineEdit::Normal,
                                            defaultName,
                                            &ok).trimmed();
    if (!ok || newName.isEmpty()) {
        return;
    }

    static const QRegularExpression validNameRegex(QStringLiteral("^[a-zA-Z0-9_-]+$"));
    if (!validNameRegex.match(newName).hasMatch()) {
        QMessageBox::warning(_dialogParent, tr("Invalid Name"),
                             tr("Surface name can only contain letters, numbers, underscores, and hyphens."));
        return;
    }

    const std::string newId = newName.toStdString();
    const std::filesystem::path sourcePath = sourceSurface->path;
    const std::filesystem::path parentDir = sourcePath.parent_path();
    const std::filesystem::path targetPath = parentDir / newId;
    if (std::filesystem::exists(targetPath)) {
        QMessageBox::warning(_dialogParent, tr("Name Exists"),
                             tr("A surface with the name '%1' already exists.").arg(newName));
        return;
    }

    QTemporaryDir stagingRoot;
    if (!stagingRoot.isValid()) {
        QMessageBox::critical(_dialogParent, tr("Temporary Directory Error"),
                              tr("Failed to create a temporary staging directory."));
        return;
    }

    try {
        if (hasTransform) {
            if (transformPanel && transformPanel->invertChecked()) {
                matrix = invertAffineTransformMatrix(*matrix);
            }
        }
        auto transformedSurface = cloneSurfaceForTransform(sourceSurface);
        if (!transformedSurface) {
            throw std::runtime_error("failed to clone source surface");
        }

        transformSurfacePoints(transformedSurface.get(), scale, matrix);
        refreshTransformedSurfaceState(transformedSurface.get());

        const std::filesystem::path stagingPath =
            std::filesystem::path(stagingRoot.path().toStdString()) / newId;
        transformedSurface->save(stagingPath.string(), newId, false);

        std::filesystem::copy(sourcePath,
                              targetPath,
                              std::filesystem::copy_options::recursive);
        std::filesystem::copy(stagingPath,
                              targetPath,
                              std::filesystem::copy_options::recursive |
                                  std::filesystem::copy_options::overwrite_existing);
    } catch (const std::exception& ex) {
        std::error_code cleanupError;
        std::filesystem::remove_all(targetPath, cleanupError);
        QMessageBox::critical(_dialogParent, tr("Save Failed"),
                              tr("Failed to save transformed surface: %1")
                                  .arg(QString::fromUtf8(ex.what())));
        return;
    }

    if (_state->vpkg()->addSingleSegmentation(newId)) {
        if (_surfacePanel) {
            _surfacePanel->addSingleSegmentation(newId);
        }
    } else {
        _state->vpkg()->refreshSegmentations();
        if (_surfacePanel) {
            _surfacePanel->reloadSurfacesFromDisk();
        }
    }

    showStatus(tr("Saved transformed surface as '%1'.").arg(newName), 5000);
    refresh();
}

void SurfaceAffineTransformController::onLoadAffineRequested()
{
    const QString promptText = tr("Enter a local path or URL for an affine JSON (http://, https://, s3://).\n"
                                  "Leave blank to use the current volume transform.json.");
    bool accepted = false;
    const QString source = QInputDialog::getText(_dialogParent,
                                                 tr("Load Affine"),
                                                 promptText,
                                                 QLineEdit::Normal,
                                                 _customTransformSource,
                                                 &accepted).trimmed();
    if (!accepted) {
        return;
    }

    QString errorMessage;
    if (!setCustomTransformSource(source, &errorMessage)) {
        QMessageBox::warning(_dialogParent,
                             tr("Load Affine Failed"),
                             tr("Failed to load affine from %1: %2")
                                 .arg(source.isEmpty() ? tr("(empty path)") : source, errorMessage));
        return;
    }

    auto* transformPanel = transformsPanel();
    if (source.isEmpty() && transformPanel && transformPanel->previewChecked()) {
        currentTransformMatrix();
    }

    if (source.isEmpty()) {
        showStatus(tr("Using the current volume transform.json."), 5000);
    } else {
        showStatus(tr("Loaded affine from %1").arg(source), 5000);
    }

    refresh();
}
