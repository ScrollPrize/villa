#include "MenuActionController.hpp"

#include "VCSettings.hpp"
#include "UnifiedBrowserDialog.hpp"
#include "CWindow.hpp"
#include "SurfacePanelController.hpp"
#include "ViewerManager.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "CommandLineToolRunner.hpp"
#include "SettingsDialog.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "ui_VCMain.h"
#include "Keybinds.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/Version.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/LoadJson.hpp"
#include "vc/core/util/RemoteUrl.hpp"
#include "vc/core/util/VolpkgConvert.hpp"
#include "vc/core/types/Segmentation.hpp"

#include <QAction>
#include <QApplication>
#include <QClipboard>
#include <QFutureWatcher>
#include <QUnhandledException>
#include <QInputDialog>
#include <QtConcurrent>
#include <QDateTime>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QLabel>
#include <QMainWindow>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QProcess>
#include <QScrollArea>
#include <QStringList>
#include <QSettings>
#include <QStyle>
#include <QTemporaryDir>
#include <QTreeWidget>
#include <QTimer>
#include <QTreeWidgetItem>
#include <QUrl>
#include <QVBoxLayout>

#include "utils/Json.hpp"

#include <algorithm>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <vector>

namespace
{
constexpr auto kRemoteVolumeRegistryFile = "remote_volumes.json";
constexpr int kMaxStoredRemoteUrls = 10;
QString extractExceptionMessage(const std::exception& e);
bool isAuthError(const QString& msg);

} // namespace

MenuActionController::MenuActionController(CWindow* window)
    : QObject(window)
    , _window(window)
{
    _recentActs.fill(nullptr);
}

void MenuActionController::populateMenus(QMenuBar* menuBar)
{
    if (!menuBar || !_window) {
        return;
    }

    auto* qWindow = _window;

    // Create actions
    _newProjectAct = new QAction(QObject::tr("&New Project"), this);
    connect(_newProjectAct, &QAction::triggered, this, &MenuActionController::newProject);

    _saveProjectAsAct = new QAction(QObject::tr("Save Project &As..."), this);
    connect(_saveProjectAsAct, &QAction::triggered, this, &MenuActionController::saveProjectAs);

    _attachVolumeAct = new QAction(QObject::tr("Attach &Volume..."), this);
    connect(_attachVolumeAct, &QAction::triggered, this, &MenuActionController::attachVolume);

    _attachSegmentsAct = new QAction(QObject::tr("Attach Se&gments..."), this);
    connect(_attachSegmentsAct, &QAction::triggered, this, &MenuActionController::attachSegments);

    _attachNormalGridAct = new QAction(QObject::tr("Attach &Normal Grid..."), this);
    connect(_attachNormalGridAct, &QAction::triggered, this, &MenuActionController::attachNormalGrid);

    _detachEntryAct = new QAction(QObject::tr("&Detach..."), this);
    connect(_detachEntryAct, &QAction::triggered, this, &MenuActionController::detachEntry);

    _setOutputSegmentsAct = new QAction(QObject::tr("Set Output Segments..."), this);
    connect(_setOutputSegmentsAct, &QAction::triggered, this, &MenuActionController::setOutputSegments);

    _convertLegacyAct = new QAction(QObject::tr("Convert Legacy Volpkg..."), this);
    connect(_convertLegacyAct, &QAction::triggered, this, &MenuActionController::convertLegacyVolpkg);

    _openAct = new QAction(qWindow->style()->standardIcon(QStyle::SP_DialogOpenButton), QObject::tr("&Open Project..."), this);
    _openAct->setShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::OpenVolpkg));
    connect(_openAct, &QAction::triggered, this, &MenuActionController::openVolpkg);

    _attachRemoteZarrAct = new QAction(QObject::tr("Attach Remote &Zarr..."), this);
    connect(_attachRemoteZarrAct, &QAction::triggered, this, &MenuActionController::attachRemoteZarr);

    _settingsAct = new QAction(QObject::tr("Settings"), this);
    connect(_settingsAct, &QAction::triggered, this, &MenuActionController::showSettingsDialog);

    _exitAct = new QAction(qWindow->style()->standardIcon(QStyle::SP_DialogCloseButton), QObject::tr("E&xit..."), this);
    connect(_exitAct, &QAction::triggered, this, &MenuActionController::exitApplication);

    _keybindsAct = new QAction(QObject::tr("&Keybinds"), this);
    connect(_keybindsAct, &QAction::triggered, this, &MenuActionController::showKeybindings);

    _aboutAct = new QAction(QObject::tr("&About..."), this);
    connect(_aboutAct, &QAction::triggered, this, &MenuActionController::showAboutDialog);

    _resetViewsAct = new QAction(QObject::tr("Reset Segmentation Views"), this);
    connect(_resetViewsAct, &QAction::triggered, this, &MenuActionController::resetSegmentationViews);

    _showConsoleAct = new QAction(QObject::tr("Show Console Output"), this);
    connect(_showConsoleAct, &QAction::triggered, this, &MenuActionController::toggleConsoleOutput);

    _drawBBoxAct = new QAction(QObject::tr("Draw BBox"), this);
    _drawBBoxAct->setCheckable(true);
    connect(_drawBBoxAct, &QAction::toggled, this, &MenuActionController::toggleDrawBBox);

    _mirrorCursorAct = new QAction(QObject::tr("Sync cursor to Surface view"), this);
    _mirrorCursorAct->setCheckable(true);
    if (qWindow) {
        _mirrorCursorAct->setChecked(qWindow->segmentationCursorMirroringEnabled());
    }
    connect(_mirrorCursorAct, &QAction::toggled, this, &MenuActionController::toggleCursorMirroring);

    _surfaceFromSelectionAct = new QAction(QObject::tr("Surface from Selection"), this);
    connect(_surfaceFromSelectionAct, &QAction::triggered, this, &MenuActionController::surfaceFromSelection);

    _selectionClearAct = new QAction(QObject::tr("Clear"), this);
    connect(_selectionClearAct, &QAction::triggered, this, &MenuActionController::clearSelection);

    _importObjAct = new QAction(QObject::tr("Import OBJ as Patch..."), this);
    connect(_importObjAct, &QAction::triggered, this, &MenuActionController::importObjAsPatch);

    _rotateSurfaceAct = new QAction(QObject::tr("Rotate"), this);
    connect(_rotateSurfaceAct, &QAction::triggered, this, &MenuActionController::beginRotateSurfaceTransform);

    // Build menus
    _fileMenu = new QMenu(QObject::tr("&File"), qWindow);
    _fileMenu->addAction(_newProjectAct);
    _fileMenu->addAction(_openAct);
    _fileMenu->addAction(_saveProjectAsAct);
    _fileMenu->addSeparator();
    _fileMenu->addAction(_attachVolumeAct);
    _fileMenu->addAction(_attachSegmentsAct);
    _fileMenu->addAction(_attachNormalGridAct);
    _fileMenu->addAction(_detachEntryAct);
    _fileMenu->addAction(_setOutputSegmentsAct);
    _fileMenu->addSeparator();
    _fileMenu->addAction(_convertLegacyAct);
    _fileMenu->addSeparator();
    _fileMenu->addAction(_attachRemoteZarrAct);

    _recentMenu = new QMenu(QObject::tr("Open &recent project"), _fileMenu);
    _recentMenu->setEnabled(false);
    _fileMenu->addMenu(_recentMenu);

    ensureRecentActions();

    _fileMenu->addSeparator();
    _fileMenu->addAction(_settingsAct);
    _fileMenu->addSeparator();
    _fileMenu->addAction(_importObjAct);
    _fileMenu->addSeparator();
    _fileMenu->addAction(_exitAct);

    _editMenu = new QMenu(QObject::tr("&Edit"), qWindow);

    _viewMenu = new QMenu(QObject::tr("&View"), qWindow);
    _viewMenu->addAction(qWindow->ui.dockWidgetVolumes->toggleViewAction());
    _viewMenu->addAction(qWindow->ui.dockWidgetSegmentation->toggleViewAction());
    _viewMenu->addAction(qWindow->ui.dockWidgetDistanceTransform->toggleViewAction());
    _viewMenu->addAction(qWindow->ui.dockWidgetViewerControls->toggleViewAction());

    if (qWindow->_point_collection_widget) {
        _viewMenu->addAction(qWindow->_point_collection_widget->toggleViewAction());
    }

    _viewMenu->addAction(_mirrorCursorAct);
    _viewMenu->addSeparator();
    _viewMenu->addAction(_resetViewsAct);
    _viewMenu->addSeparator();
    _viewMenu->addAction(_showConsoleAct);

    _actionsMenu = new QMenu(QObject::tr("&Actions"), qWindow);
    _actionsMenu->addAction(_drawBBoxAct);
    _actionsMenu->addSeparator();
    _transformsMenu = new QMenu(QObject::tr("&Transforms"), _actionsMenu);
    _transformsMenu->addAction(_rotateSurfaceAct);
    _actionsMenu->addMenu(_transformsMenu);

    _selectionMenu = new QMenu(QObject::tr("&Selection"), qWindow);
    _selectionMenu->addAction(_surfaceFromSelectionAct);
    _selectionMenu->addAction(_selectionClearAct);

    _helpMenu = new QMenu(QObject::tr("&Help"), qWindow);
    _helpMenu->addAction(_keybindsAct);
    _helpMenu->addAction(_aboutAct);

    menuBar->addMenu(_fileMenu);
    menuBar->addMenu(_editMenu);
    menuBar->addMenu(_viewMenu);
    menuBar->addMenu(_actionsMenu);
    menuBar->addMenu(_selectionMenu);
    menuBar->addMenu(_helpMenu);

    refreshRecentMenu();
}

void MenuActionController::ensureRecentActions()
{
    if (!_recentMenu) {
        return;
    }

    for (auto& act : _recentActs) {
        if (!act) {
            act = new QAction(this);
            act->setVisible(false);
            connect(act, &QAction::triggered, this, &MenuActionController::openRecentVolpkg);
            _recentMenu->addAction(act);
        }
    }
}

QStringList MenuActionController::loadRecentPaths() const
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    return settings.value(vc3d::settings::project::RECENT).toStringList();
}

void MenuActionController::saveRecentPaths(const QStringList& paths)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::project::RECENT, paths);
}

void MenuActionController::refreshRecentMenu()
{
    ensureRecentActions();

    QStringList files = loadRecentPaths();
    if (!files.isEmpty() && files.last().isEmpty()) {
        files.removeLast();
    }

    const int numRecentFiles = std::min(static_cast<int>(files.size()), kMaxRecentVolpkg);

    for (int i = 0; i < numRecentFiles; ++i) {
        QString fileName = QFileInfo(files[i]).fileName();
        fileName.replace("&", "&&");

        QString path = QFileInfo(files[i]).canonicalPath();
        if (path == ".") {
            path = QObject::tr("Directory not available!");
        } else {
            path.replace("&", "&&");
        }

        QString text = QObject::tr("&%1 | %2 (%3)").arg(i + 1).arg(fileName).arg(path);
        _recentActs[i]->setText(text);
        _recentActs[i]->setData(files[i]);
        _recentActs[i]->setVisible(true);
    }

    for (int j = numRecentFiles; j < kMaxRecentVolpkg; ++j) {
        if (_recentActs[j]) {
            _recentActs[j]->setVisible(false);
            _recentActs[j]->setData(QVariant());
        }
    }

    if (_recentMenu) {
        _recentMenu->setEnabled(numRecentFiles > 0);
    }
}

void MenuActionController::updateRecentVolpkgList(const QString& path)
{
    QStringList files = loadRecentPaths();
    const QString canonical = QFileInfo(path).absoluteFilePath();
    files.removeAll(canonical);
    files.prepend(canonical);
    while (files.size() > MAX_RECENT_VOLPKG) {
        files.removeLast();
    }
    saveRecentPaths(files);
    refreshRecentMenu();
}

void MenuActionController::removeRecentVolpkgEntry(const QString& path)
{
    QStringList files = loadRecentPaths();
    files.removeAll(path);
    saveRecentPaths(files);
    refreshRecentMenu();
}

void MenuActionController::openVolpkg()
{
    if (!_window) {
        return;
    }
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    auto file = promptLocation(QObject::tr("Open Project"),
        QObject::tr("Pick a .volpkg.json file, a legacy volpkg folder, or a remote volpkg URL."),
        settings.value(vc3d::settings::project::DEFAULT_PATH).toString(),
        {"*.volpkg.json"}, true, true);
    if (file.isEmpty()) return;

    if (!file.endsWith(".volpkg.json", Qt::CaseInsensitive)) {
        QMessageBox::information(_window, QObject::tr("Convert to .volpkg.json"),
            QObject::tr("This needs to be converted to a .volpkg.json — pick where to save the converted project."));
        QString converted;
        if (!runLegacyVolpkgConvert(file, &converted)) return;
        file = converted;
    }
    _window->CloseVolume();
    _window->OpenVolume(file);
    loadAttachedRemoteVolumesForCurrentPackage();
    _window->UpdateView();
}

void MenuActionController::openRecentVolpkg()
{
    if (!_window) {
        return;
    }

    if (auto* action = qobject_cast<QAction*>(sender())) {
        const QString path = action->data().toString();
        if (!path.isEmpty()) {
            _window->CloseVolume();
            _window->OpenVolume(path);
            loadAttachedRemoteVolumesForCurrentPackage();
            _window->UpdateView();
        }
    }
}

void MenuActionController::openVolpkgAt(const QString& path)
{
    if (!_window) {
        return;
    }

    _window->CloseVolume();
    _window->OpenVolume(path);
    loadAttachedRemoteVolumesForCurrentPackage();
    _window->UpdateView();
}

// --- Remote recents management ---

QStringList MenuActionController::loadRecentRemoteUrls() const
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    return settings.value(vc3d::settings::viewer::REMOTE_RECENT_URLS).toStringList();
}

void MenuActionController::saveRecentRemoteUrls(const QStringList& urls)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::REMOTE_RECENT_URLS, urls);
}

void MenuActionController::updateRecentRemoteList(const QString& url)
{
    QStringList urls = loadRecentRemoteUrls();
    urls.removeAll(url);
    urls.prepend(url);
    while (urls.size() > kMaxStoredRemoteUrls) {
        urls.removeLast();
    }
    saveRecentRemoteUrls(urls);
}

void MenuActionController::attachRemoteZarr()
{
    if (!_window) return;

    if (!_window->_state || !_window->_state->vpkg()) {
        QMessageBox::warning(_window,
                             QObject::tr("No Volume Package Loaded"),
                             QObject::tr("Open a volpkg before attaching a remote zarr."));
        return;
    }

    QStringList recentUrls = loadRecentRemoteUrls();
    QString lastUrl = recentUrls.isEmpty() ? QString() : recentUrls.first();

    bool ok = false;
    QString url = QInputDialog::getText(
        _window,
        QObject::tr("Attach Remote Zarr"),
        QObject::tr("Enter remote OME-Zarr URL (http://, https://, s3://):"),
        QLineEdit::Normal,
        lastUrl,
        &ok);

    if (!ok || url.trimmed().isEmpty()) {
        return;
    }

    attachRemoteZarrUrl(url.trimmed(), true);
}

bool MenuActionController::tryResolveRemoteAuth(const QString& url,
                                                vc::HttpAuth* authOut,
                                                bool allowPrompt,
                                                QString* errorMessage) const
{
    if (!authOut) {
        return false;
    }

    *authOut = {};
    auto resolved = vc::resolveRemoteUrl(url.trimmed().toStdString());
    if (!resolved.useAwsSigv4) {
        return true;
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    *authOut = vc::loadAwsCredentials();
    if (authOut->region.empty()) authOut->region = resolved.awsRegion;

    if (authOut->access_key.empty() || authOut->secret_key.empty()) {
        const auto savedAccess = settings.value(vc3d::settings::aws::ACCESS_KEY).toString();
        const auto savedSecret = settings.value(vc3d::settings::aws::SECRET_KEY).toString();
        const auto savedToken = settings.value(vc3d::settings::aws::SESSION_TOKEN).toString();

        if (!savedAccess.isEmpty() && !savedSecret.isEmpty()) {
            authOut->access_key = savedAccess.toStdString();
            authOut->secret_key = savedSecret.toStdString();
            authOut->session_token = savedToken.toStdString();
        }
    }

    if (!authOut->access_key.empty() && !authOut->secret_key.empty()) {
        return true;
    }

    // Public S3 buckets can be read anonymously. Do not block the first
    // request on credential entry just because the URL is S3-shaped; if the
    // server returns an auth error, the caller's existing retry path prompts.
    if (errorMessage) {
        errorMessage->clear();
    }

    return true;
}

QString MenuActionController::remoteCacheDirectory() const
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    QString defaultCache = vc3d::defaultCacheBase() + "/remote_cache";
    QString cacheDir = settings.value(vc3d::settings::viewer::REMOTE_CACHE_DIR, defaultCache).toString();
    QDir().mkpath(cacheDir);
    return cacheDir;
}

QString MenuActionController::remoteVolumeRegistryPath() const
{
    if (!_window || !_window->_state || !_window->_state->vpkg()) {
        return {};
    }
    return QDir(_window->_state->vpkgPath()).filePath(QString::fromLatin1(kRemoteVolumeRegistryFile));
}

void MenuActionController::persistAttachedRemoteVolume(const QString& url, const std::shared_ptr<Volume>& volume)
{
    const QString registryPath = remoteVolumeRegistryPath();
    if (registryPath.isEmpty() || !volume) {
        return;
    }

    utils::Json root = {
        {"version", utils::Json(1)},
        {"volumes", utils::Json::array()}
    };

    try {
        if (QFileInfo::exists(registryPath)) {
            root = utils::Json::parse_file(registryPath.toStdString());
        }
    } catch (const std::exception& e) {
        Logger()->warn("Failed reading remote volume registry '{}': {}", registryPath.toStdString(), e.what());
        root = {
            {"version", utils::Json(1)},
            {"volumes", utils::Json::array()}
        };
    }

    if (!root.is_object()) {
        root = utils::Json::object();
    }
    if (!root.contains("volumes") || !root["volumes"].is_array()) {
        root["volumes"] = utils::Json::array();
    }
    root["version"] = 1;

    const std::string urlStd = url.trimmed().toStdString();
    const std::string idStd = volume->id();
    utils::Json updated = utils::Json::array();
    bool replaced = false;

    for (const auto& entry : root["volumes"]) {
        if (!entry.is_object()) {
            continue;
        }
        const std::string existingUrl = entry.value("url", std::string{});
        const std::string existingId = entry.value("id", std::string{});
        if (existingUrl == urlStd || (!idStd.empty() && existingId == idStd)) {
            if (!replaced) {
                updated.push_back({
                    {"url", urlStd},
                    {"id", idStd},
                    {"name", volume->name()}
                });
                replaced = true;
            }
            continue;
        }
        updated.push_back(entry);
    }

    if (!replaced) {
        updated.push_back({
            {"url", urlStd},
            {"id", idStd},
            {"name", volume->name()}
        });
    }

    root["volumes"] = std::move(updated);

    std::ofstream output(registryPath.toStdString(), std::ofstream::out | std::ofstream::trunc);
    output << root.dump(2) << '\n';
}

void MenuActionController::loadAttachedRemoteVolumesForCurrentPackage()
{
    const QString registryPath = remoteVolumeRegistryPath();
    if (registryPath.isEmpty() || !QFileInfo::exists(registryPath) || !_window || !_window->_state || !_window->_state->vpkg()) {
        return;
    }

    utils::Json root;
    try {
        root = utils::Json::parse_file(registryPath.toStdString());
    } catch (const std::exception& e) {
        Logger()->warn("Failed to parse remote volume registry '{}': {}", registryPath.toStdString(), e.what());
        if (_window->statusBar()) {
            _window->statusBar()->showMessage(QObject::tr("Failed to read remote_volumes.json"), 5000);
        }
        return;
    }

    if (!root.contains("volumes") || !root["volumes"].is_array() || root["volumes"].empty()) {
        return;
    }

    const QString currentId = QString::fromStdString(_window->_state->currentVolumeId());
    const QString cacheDir = remoteCacheDirectory();
    int attachedCount = 0;
    int skippedCount = 0;
    QString firstAttachedId;

    for (const auto& entry : root["volumes"]) {
        if (!entry.is_object()) {
            continue;
        }

        const QString url = QString::fromStdString(entry.value("url", std::string{})).trimmed();
        if (url.isEmpty()) {
            continue;
        }

        vc::HttpAuth auth;
        QString authError;
        if (!tryResolveRemoteAuth(url, &auth, false, &authError)) {
            Logger()->warn("Skipping persisted remote volume '{}': {}", url.toStdString(), authError.toStdString());
            skippedCount++;
            continue;
        }

        try {
            auto volume = Volume::NewFromUrl(url.toStdString(), cacheDir.toStdString(), auth);
            if (_window->_state->vpkg()->hasVolume(volume->id())) {
                continue;
            }
            if (_window->_state->vpkg()->addVolume(volume)) {
                if (firstAttachedId.isEmpty()) {
                    firstAttachedId = QString::fromStdString(volume->id());
                }
                attachedCount++;
            } else {
                skippedCount++;
            }
        } catch (const std::exception& e) {
            Logger()->warn("Failed to attach persisted remote volume '{}': {}", url.toStdString(), e.what());
            skippedCount++;
        }
    }

    if (attachedCount > 0) {
        _window->refreshCurrentVolumePackageUi(firstAttachedId.isEmpty() ? currentId : firstAttachedId, false);
        _window->UpdateView();
    }

    if (_window->statusBar() && (attachedCount > 0 || skippedCount > 0)) {
        _window->statusBar()->showMessage(
            QObject::tr("Attached %1 persisted remote volume(s), skipped %2.")
                .arg(attachedCount)
                .arg(skippedCount),
            5000);
    }
}

void MenuActionController::attachRemoteZarrUrl(const QString& url, bool persistEntry)
{
    if (!_window || !_window->_state || !_window->_state->vpkg()) {
        QMessageBox::warning(_window,
                             QObject::tr("No Volume Package Loaded"),
                             QObject::tr("Open a volpkg before attaching a remote zarr."));
        return;
    }

    auto resolved = vc::resolveRemoteUrl(url.trimmed().toStdString());
    std::string trimmed = resolved.httpsUrl;
    while (!trimmed.empty() && trimmed.back() == '/') {
        trimmed.pop_back();
    }
    const bool looksLikeZarr = trimmed.size() >= 5 && trimmed.substr(trimmed.size() - 5) == ".zarr";
    if (!looksLikeZarr) {
        QMessageBox::warning(_window,
                             QObject::tr("Expected Remote Zarr"),
                             QObject::tr("Attach Remote Zarr expects a direct .zarr URL, not a scroll root."));
        return;
    }

    vc::HttpAuth auth;
    QString authError;
    if (!tryResolveRemoteAuth(url, &auth, true, &authError)) {
        if (!authError.isEmpty() && authError != QObject::tr("AWS credential entry canceled.")) {
            QMessageBox::warning(_window, QObject::tr("Authentication Error"), authError);
        }
        return;
    }

    const QString cacheDir = remoteCacheDirectory();
    updateRecentRemoteList(url);
    if (_attachRemoteZarrAct) {
        _attachRemoteZarrAct->setEnabled(false);
    }
    if (_window->statusBar()) {
        _window->statusBar()->showMessage(QObject::tr("Attaching remote zarr..."));
    }

    auto* watcher = new QFutureWatcher<std::shared_ptr<Volume>>(this);
    connect(watcher, &QFutureWatcher<std::shared_ptr<Volume>>::finished, this,
            [this, watcher, url, persistEntry]() {
                watcher->deleteLater();
                if (_attachRemoteZarrAct) {
                    _attachRemoteZarrAct->setEnabled(true);
                }

                auto future = watcher->future();
                QString errorMsg;
                bool success = false;

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
                if (future.isValid() && !future.isCanceled() && future.isResultReadyAt(0)) {
#else
                if (future.isFinished() && !future.isCanceled()) {
#endif
                    try {
                        auto volume = future.result();
                        if (!_window || !_window->_state || !_window->_state->vpkg()) {
                            return;
                        }

                        if (!_window->attachVolumeToCurrentPackage(volume)) {
                            QMessageBox::warning(
                                _window,
                                QObject::tr("Attach Remote Zarr"),
                                QObject::tr("A volume with id '%1' is already present in this volume package.")
                                    .arg(QString::fromStdString(volume->id())));
                            return;
                        }

                        _window->_state->vpkg()->addVolumeEntry(url.trimmed().toStdString());

                        if (persistEntry) {
                            persistAttachedRemoteVolume(url, volume);
                        }

                        if (_window->statusBar()) {
                            _window->statusBar()->showMessage(
                                QObject::tr("Attached remote zarr: %1")
                                    .arg(QString::fromStdString(volume->id())),
                                5000);
                        }
                        success = true;
                    } catch (const std::exception& e) {
                        errorMsg = extractExceptionMessage(e);
                    } catch (...) {
                        errorMsg = QObject::tr("Unknown error attaching remote zarr");
                    }
                } else {
                    try {
                        future.waitForFinished();
                        future.result();
                    } catch (const std::exception& e) {
                        errorMsg = extractExceptionMessage(e);
                    } catch (...) {
                        errorMsg = QObject::tr("Unknown error attaching remote zarr");
                    }
                }

                if (success) return;

                if (isAuthError(errorMsg)) {
                    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
                    settings.remove(vc3d::settings::aws::ACCESS_KEY);
                    settings.remove(vc3d::settings::aws::SECRET_KEY);
                    settings.remove(vc3d::settings::aws::SESSION_TOKEN);

                    const auto reply = QMessageBox::warning(
                        _window,
                        QObject::tr("Authentication Error"),
                        QObject::tr("Failed to attach remote zarr:\n%1\n\n"
                                    "Would you like to enter new AWS credentials and retry?")
                            .arg(errorMsg),
                        QMessageBox::Yes | QMessageBox::No);
                    if (reply == QMessageBox::Yes) {
                        QTimer::singleShot(0, this, [this, url, persistEntry]() {
                            attachRemoteZarrUrl(url, persistEntry);
                        });
                        return;
                    }
                }

                QMessageBox::critical(
                    _window,
                    QObject::tr("Attach Remote Zarr Error"),
                    QObject::tr("Failed to attach remote zarr:\n%1").arg(errorMsg));
            });

    auto future = QtConcurrent::run([url, auth, cacheDir]() -> std::shared_ptr<Volume> {
        return Volume::NewFromUrl(url.toStdString(), cacheDir.toStdString(), auth);
    });
    watcher->setFuture(future);
}

namespace
{

// Helper: extract the real error message from a QFuture exception.
// Qt wraps task exceptions in QUnhandledException whose what() returns
// "std::exception" — useless. Unwrap to get the original message.
QString extractExceptionMessage(const std::exception& e)
{
    if (auto* unhandled = dynamic_cast<const QUnhandledException*>(&e)) {
        auto ptr = unhandled->exception();
        if (ptr) {
            try {
                std::rethrow_exception(ptr);
            } catch (const std::exception& inner) {
                return QString::fromStdString(inner.what());
            } catch (...) {
                return QObject::tr("Unknown error (non-std::exception)");
            }
        }
    }
    return QString::fromStdString(e.what());
}

bool isAuthError(const QString& msg)
{
    return msg.contains("Access denied", Qt::CaseInsensitive) ||
           msg.contains("403", Qt::CaseInsensitive) ||
           msg.contains("401", Qt::CaseInsensitive) ||
           msg.contains("credential", Qt::CaseInsensitive) ||
           msg.contains("Forbidden", Qt::CaseInsensitive);
}

} // namespace

void MenuActionController::showSettingsDialog()
{
    if (!_window) {
        return;
    }

    auto* dialog = new SettingsDialog(_window);
    dialog->exec();

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    bool showDirHints = settings.value(vc3d::settings::viewer::SHOW_DIRECTION_HINTS,
                                       vc3d::settings::viewer::SHOW_DIRECTION_HINTS_DEFAULT).toBool();
    if (_window->_viewerManager) {
        _window->_viewerManager->forEachBaseViewer([showDirHints](VolumeViewerBase* viewer) {
            if (viewer) {
                viewer->setShowDirectionHints(showDirHints);
            }
        });
    }

    dialog->deleteLater();
}

void MenuActionController::showAboutDialog()
{
    if (!_window) {
        return;
    }
    const QString repoShortHash = QString::fromStdString(ProjectInfo::RepositoryShortHash());
    QString commitText = repoShortHash;
    if (commitText.isEmpty() || commitText.compare("Untracked", Qt::CaseInsensitive) == 0) {
        commitText = QStringLiteral("unknown");
    }
    QMessageBox::information(
        _window,
        QObject::tr("About VC3D - Volume Cartographer 3D"),
        QObject::tr("Vesuvius Challenge Team\n\n"
                    "code: https://github.com/ScrollPrize/villa\n\n"
                    "discord: https://discord.com/channels/1079907749569237093/1243576621722767412\n\n"
                    "Commit: %1")
            .arg(commitText));
}

void MenuActionController::showKeybindings()
{
    if (!_window) {
        return;
    }

    if (_keybindsDialog) {
        _keybindsDialog->raise();
        _keybindsDialog->activateWindow();
        return;
    }

    _keybindsDialog = new QDialog(_window);
    _keybindsDialog->setAttribute(Qt::WA_DeleteOnClose);
    _keybindsDialog->setWindowTitle(QObject::tr("Keybindings for Volume Cartographer"));

    auto* layout = new QVBoxLayout(_keybindsDialog);
    auto* scrollArea = new QScrollArea(_keybindsDialog);
    scrollArea->setWidgetResizable(true);

    auto* content = new QWidget(scrollArea);
    auto* contentLayout = new QVBoxLayout(content);
    auto* label = new QLabel(content);
    label->setTextFormat(Qt::PlainText);
    label->setText(vc3d::keybinds::buildKeybindsHelpText());
    label->setTextInteractionFlags(Qt::TextSelectableByMouse);
    label->setWordWrap(false);
    contentLayout->addWidget(label);
    contentLayout->addStretch();
    content->setLayout(contentLayout);

    scrollArea->setWidget(content);
    layout->addWidget(scrollArea);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok, _keybindsDialog);
    connect(buttons, &QDialogButtonBox::accepted, _keybindsDialog, &QDialog::accept);
    layout->addWidget(buttons);

    _keybindsDialog->resize(640, 520);
    _keybindsDialog->setMinimumHeight(360);
    _keybindsDialog->show();
    _keybindsDialog->raise();
    _keybindsDialog->activateWindow();
}

void MenuActionController::exitApplication()
{
    if (_window) {
        _window->close();
    }
}

void MenuActionController::resetSegmentationViews()
{
    if (!_window) {
        return;
    }

    for (auto* sub : _window->mdiArea->subWindowList()) {
        sub->showNormal();
    }
    _window->mdiArea->tileSubWindows();
}

void MenuActionController::toggleConsoleOutput()
{
    if (!_window) {
        return;
    }

    if (_window->_cmdRunner) {
        _window->_cmdRunner->showConsoleOutput();
    } else {
        QMessageBox::information(_window, QObject::tr("Console Output"),
                                 QObject::tr("No command line tool has been run yet. The console will be available after running a tool."));
    }
}

void MenuActionController::toggleDrawBBox(bool enabled)
{
    if (!_window || !_window->_viewerManager) {
        return;
    }

    _window->_viewerManager->forEachBaseViewer([this, enabled](VolumeViewerBase* viewer) {
        if (viewer && viewer->surfName() == "segmentation") {
            viewer->setBBoxMode(enabled);
            if (_window->statusBar()) {
                _window->statusBar()->showMessage(enabled ? QObject::tr("BBox mode active: drag on Surface view")
                                                         : QObject::tr("BBox mode off"),
                                                  3000);
            }
        }
    });
}

void MenuActionController::toggleCursorMirroring(bool enabled)
{
    if (!_window) {
        return;
    }
    _window->setSegmentationCursorMirroring(enabled);
}

void MenuActionController::surfaceFromSelection()
{
    if (!_window || !_window->_viewerManager || !_window->_state->vpkg()) {
        return;
    }

    VolumeViewerBase* segViewer = _window->segmentationBaseViewer();

    if (!segViewer) {
        _window->statusBar()->showMessage(QObject::tr("No Surface viewer found"), 3000);
        return;
    }

    auto sels = segViewer->selections();
    if (sels.empty()) {
        _window->statusBar()->showMessage(QObject::tr("No selections to convert"), 3000);
        return;
    }

    if (_window->_state->activeSurfaceId().empty() || !_window->_state->vpkg()->getSurface(_window->_state->activeSurfaceId())) {
        _window->statusBar()->showMessage(QObject::tr("Select a segmentation first"), 3000);
        return;
    }

    auto surf = _window->_state->vpkg()->getSurface(_window->_state->activeSurfaceId());
    std::filesystem::path baseSegPath = surf->path;
    std::filesystem::path parentDir = baseSegPath.parent_path();

    int idx = 1;
    int created = 0;
    QString ts = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    for (const auto& pr : sels) {
        const QRectF& rect = pr.first;
        std::unique_ptr<QuadSurface> filtered(segViewer->makeBBoxFilteredSurfaceFromSceneRect(rect));
        if (!filtered) {
            continue;
        }

        std::string newId = _window->_state->activeSurfaceId() + std::string("_sel_") + ts.toStdString() + std::string("_") + std::to_string(idx++);
        std::filesystem::path outDir = parentDir / newId;
        try {
            filtered->save(outDir.string(), newId);
            created++;
        } catch (const std::exception& e) {
            _window->statusBar()->showMessage(QObject::tr("Failed to save selection: ") + e.what(), 5000);
        }
    }

    if (created > 0) {
        if (_window->_surfacePanel) {
            _window->_surfacePanel->reloadSurfacesFromDisk();
        }
        _window->statusBar()->showMessage(QObject::tr("Created %1 surface(s) from selection").arg(created), 5000);
    } else {
        if (_window->_surfacePanel) {
            _window->_surfacePanel->refreshFiltersOnly();
        }
        _window->statusBar()->showMessage(QObject::tr("No surfaces created from selection"), 3000);
    }
}

void MenuActionController::clearSelection()
{
    if (!_window) {
        return;
    }

    VolumeViewerBase* segViewer = _window->segmentationBaseViewer();
    if (!segViewer) {
        _window->statusBar()->showMessage(QObject::tr("No Surface viewer found"), 3000);
        return;
    }

    segViewer->clearSelections();
    _window->statusBar()->showMessage(QObject::tr("Selections cleared"), 2000);
}

void MenuActionController::importObjAsPatch()
{
    if (!_window || !_window->_state->vpkg()) {
        QMessageBox::warning(_window, QObject::tr("Error"), QObject::tr("No volume package loaded."));
        return;
    }

    QStringList objFiles = QFileDialog::getOpenFileNames(
        _window,
        QObject::tr("Select OBJ Files"),
        QDir::homePath(),
        QObject::tr("OBJ Files (*.obj);;All Files (*)"));

    if (objFiles.isEmpty()) {
        return;
    }

    const auto outDir = _window->_state->vpkg()->outputSegmentsPath();
    if (outDir.empty()) {
        QMessageBox::warning(_window, QObject::tr("Error"),
                             QObject::tr("Project has no output segments directory configured."));
        return;
    }
    QString pathsDir = QString::fromStdString(outDir.string());

    QStringList successfulIds;
    QStringList failedFiles;

    for (const QString& objFile : objFiles) {
        QFileInfo fileInfo(objFile);
        QString baseName = fileInfo.completeBaseName();
        QString outputDir = pathsDir + "/" + baseName;

        if (QDir(outputDir).exists()) {
            if (QMessageBox::question(_window, QObject::tr("Overwrite?"),
                                      QObject::tr("'%1' exists. Overwrite?").arg(baseName),
                                      QMessageBox::Yes | QMessageBox::No) == QMessageBox::No) {
                continue;
            }
        }

        QProcess process;
        process.setProcessChannelMode(QProcess::MergedChannels);

        QStringList args;
        args << objFile << outputDir;
        args << QString::number(1000.0f)
             << QString::number(1.0f)
             << QString::number(20);

        QString toolPath = QCoreApplication::applicationDirPath() + "/vc_obj2tifxyz_legacy";
        process.start(toolPath, args);

        if (!process.waitForStarted(5000)) {
            failedFiles.append(fileInfo.fileName());
            continue;
        }

        process.waitForFinished(-1);

        if (process.exitCode() == 0 && process.exitStatus() == QProcess::NormalExit) {
            successfulIds.append(baseName);
        } else {
            failedFiles.append(fileInfo.fileName());
        }
    }

    if (!successfulIds.isEmpty() && _window->_surfacePanel) {
        _window->_surfacePanel->reloadSurfacesFromDisk();
    } else if (_window->_surfacePanel) {
        _window->_surfacePanel->refreshFiltersOnly();
    }

    QString message = QObject::tr("Imported: %1\nFailed: %2").arg(successfulIds.size()).arg(failedFiles.size());
    if (!failedFiles.isEmpty()) {
        message += QObject::tr("\n\nFailed files:\n%1").arg(failedFiles.join("\n"));
    }

    QMessageBox::information(_window, QObject::tr("Import Results"), message);
}

void MenuActionController::beginRotateSurfaceTransform()
{
    if (!_window || !_window->_surfaceRotationOverlay) {
        return;
    }

    auto activeSurface = _window->_state ? _window->_state->activeSurface().lock() : nullptr;
    if (!activeSurface) {
        activeSurface = _window->_state
            ? std::dynamic_pointer_cast<QuadSurface>(_window->_state->surface("segmentation"))
            : nullptr;
    }
    if (!activeSurface) {
        QMessageBox::information(_window,
                                 QObject::tr("Rotate Surface"),
                                 QObject::tr("Select a segmentation surface before rotating."));
        return;
    }

    _window->_surfaceRotationOverlay->beginRotate();
    if (_window->statusBar()) {
        _window->statusBar()->showMessage(QObject::tr("Surface rotation active"), 3000);
    }
}

void MenuActionController::newProject()
{
    if (!_window) return;
    auto pkg = VolumePkg::newEmpty();
    pkg->saveAutosave();
    _window->_state->setVpkg(pkg);
    _window->refreshCurrentVolumePackageUi(QString(), true);
}

void MenuActionController::saveProjectAs()
{
    if (!_window || !_window->_state || !_window->_state->vpkg()) {
        QMessageBox::information(_window, QObject::tr("No project"), QObject::tr("Open or create a project first."));
        return;
    }
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    QString defaultDir = settings.value(vc3d::settings::project::DEFAULT_PATH).toString();
    QString file = QFileDialog::getSaveFileName(
        _window, QObject::tr("Save Project As"), defaultDir,
        QObject::tr("Project (*.volpkg.json)"));
    if (file.isEmpty()) return;
    if (!file.endsWith(".volpkg.json", Qt::CaseInsensitive)) file += ".volpkg.json";
    try {
        _window->_state->vpkg()->save(std::filesystem::path(file.toStdString()));
        settings.setValue(vc3d::settings::project::DEFAULT_PATH, QFileInfo(file).absolutePath());
        updateRecentVolpkgList(file);
    } catch (const std::exception& e) {
        QMessageBox::warning(_window, QObject::tr("Save failed"), QString::fromUtf8(e.what()));
    }
}

QString MenuActionController::promptLocation(const QString& title,
                                             const QString& hint,
                                             const QString& defaultDir,
                                             const QStringList& localFilters,
                                             bool acceptFiles,
                                             bool acceptDirs)
{
    UnifiedBrowserDialog dlg(_window);
    dlg.setWindowTitle(title);
    dlg.setHint(hint);
    dlg.setStartUri(defaultDir);
    dlg.setLocalNameFilters(localFilters);
    dlg.setAcceptsFiles(acceptFiles);
    dlg.setAcceptsDirs(acceptDirs);
    dlg.setAuthResolver([this](const QString& url, vc::HttpAuth* out, QString* err) {
        return tryResolveRemoteAuth(url, out, true, err);
    });
    if (dlg.exec() != QDialog::Accepted) return {};
    QString uri = dlg.selectedUri();
    if (uri.startsWith("file://", Qt::CaseInsensitive)) uri = uri.mid(7);
    const int schemeSep = uri.indexOf("://");
    const int minLen = (schemeSep < 0) ? 1 : schemeSep + 4;
    while (uri.size() > minLen && uri.endsWith('/')) uri.chop(1);
    return uri;
}

void MenuActionController::attachVolume()
{
    if (!_window || !_window->_state || !_window->_state->vpkg()) {
        QMessageBox::information(_window, QObject::tr("No project"), QObject::tr("Open or create a project first."));
        return;
    }
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    auto loc = promptLocation(QObject::tr("Attach Volume"),
                              QObject::tr("Pick a zarr volume or a folder of zarrs (local or s3://, https://)."),
                              settings.value(vc3d::settings::project::DEFAULT_PATH).toString(),
                              {}, false, true);
    if (loc.isEmpty()) return;
    const auto err = vc::project::validateLocation(vc::project::Category::Volumes, loc.toStdString());
    if (!err.empty()) {
        QMessageBox::warning(_window, QObject::tr("Attach failed"),
            QObject::tr("Not a valid volume location:\n\n%1").arg(QString::fromStdString(err)));
        return;
    }
    bool ok = false;
    QString tagsStr = QInputDialog::getText(_window, QObject::tr("Attach Volume"),
        QObject::tr("Tags (comma-separated, optional; e.g. normal3d):"),
        QLineEdit::Normal, QString(), &ok);
    if (!ok) return;
    std::vector<std::string> tags;
    for (const auto& t : tagsStr.split(',', Qt::SkipEmptyParts)) {
        const auto trimmed = t.trimmed().toStdString();
        if (!trimmed.empty()) tags.push_back(trimmed);
    }
    if (!_window->_state->vpkg()->addVolumeEntry(loc.toStdString(), tags)) {
        QMessageBox::warning(_window, QObject::tr("Attach failed"), QObject::tr("Could not add volume (already attached?)"));
        return;
    }
    _window->refreshCurrentVolumePackageUi(QString(), true);
}

void MenuActionController::attachSegments()
{
    if (!_window || !_window->_state || !_window->_state->vpkg()) {
        QMessageBox::information(_window, QObject::tr("No project"), QObject::tr("Open or create a project first."));
        return;
    }
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    auto loc = promptLocation(QObject::tr("Attach Segments"),
                              QObject::tr("Pick a segment or a folder of segments (local or s3://, https://)."),
                              settings.value(vc3d::settings::project::DEFAULT_PATH).toString(),
                              {}, false, true);
    if (loc.isEmpty()) return;
    const auto err = vc::project::validateLocation(vc::project::Category::Segments, loc.toStdString());
    if (!err.empty()) {
        QMessageBox::warning(_window, QObject::tr("Attach failed"),
            QObject::tr("Not a valid segments location:\n\n%1").arg(QString::fromStdString(err)));
        return;
    }
    auto pkg = _window->_state->vpkg();
    if (!pkg->addSegmentsEntry(loc.toStdString())) {
        QMessageBox::warning(_window, QObject::tr("Attach failed"), QObject::tr("Could not add segments (already attached?)"));
        return;
    }
    if (!pkg->hasOutputSegments()) pkg->setOutputSegments(loc.toStdString());
    _window->refreshCurrentVolumePackageUi(QString(), true);
}

void MenuActionController::attachNormalGrid()
{
    if (!_window || !_window->_state || !_window->_state->vpkg()) {
        QMessageBox::information(_window, QObject::tr("No project"), QObject::tr("Open or create a project first."));
        return;
    }
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    auto loc = promptLocation(QObject::tr("Attach Normal Grid"),
                              QObject::tr("Pick a normal_grids root (xy/xz/yz subdirs)."),
                              settings.value(vc3d::settings::project::DEFAULT_PATH).toString(),
                              {}, false, true);
    if (loc.isEmpty()) return;
    const auto err = vc::project::validateLocation(vc::project::Category::NormalGrids, loc.toStdString());
    if (!err.empty()) {
        QMessageBox::warning(_window, QObject::tr("Attach failed"),
            QObject::tr("Not a valid normal-grid location:\n\n%1").arg(QString::fromStdString(err)));
        return;
    }
    if (!_window->_state->vpkg()->addNormalGridEntry(loc.toStdString())) {
        QMessageBox::warning(_window, QObject::tr("Attach failed"), QObject::tr("Could not add normal grid (already attached?)"));
        return;
    }
    _window->refreshCurrentVolumePackageUi(QString(), true);
}

void MenuActionController::detachEntry()
{
    if (!_window || !_window->_state || !_window->_state->vpkg()) {
        QMessageBox::information(_window, QObject::tr("No project"), QObject::tr("Open or create a project first."));
        return;
    }
    auto pkg = _window->_state->vpkg();
    QStringList items;
    QStringList locations;
    for (const auto& e : pkg->volumeEntries()) {
        items << QObject::tr("[volume] %1").arg(QString::fromStdString(e.location));
        locations << QString::fromStdString(e.location);
    }
    for (const auto& e : pkg->segmentEntries()) {
        items << QObject::tr("[segments] %1").arg(QString::fromStdString(e.location));
        locations << QString::fromStdString(e.location);
    }
    for (const auto& e : pkg->normalGridEntries()) {
        items << QObject::tr("[normal_grid] %1").arg(QString::fromStdString(e.location));
        locations << QString::fromStdString(e.location);
    }
    if (items.isEmpty()) {
        QMessageBox::information(_window, QObject::tr("Detach"),
            QObject::tr("Nothing to detach — project has no entries."));
        return;
    }
    bool ok = false;
    const QString picked = QInputDialog::getItem(_window, QObject::tr("Detach"),
        QObject::tr("Select an entry to remove from the project:"),
        items, 0, false, &ok);
    if (!ok || picked.isEmpty()) return;
    const int idx = items.indexOf(picked);
    if (idx < 0) return;
    const QString loc = locations.at(idx);
    if (QMessageBox::question(_window, QObject::tr("Detach"),
            QObject::tr("Remove this entry from the project?\n\n%1").arg(loc),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No) != QMessageBox::Yes) {
        return;
    }
    if (!pkg->removeEntry(loc.toStdString())) {
        QMessageBox::warning(_window, QObject::tr("Detach failed"),
            QObject::tr("Could not remove entry (not found)."));
        return;
    }
    _window->refreshCurrentVolumePackageUi(QString(), true);
}

void MenuActionController::setOutputSegments()
{
    if (!_window || !_window->_state || !_window->_state->vpkg()) {
        QMessageBox::information(_window, QObject::tr("No project"), QObject::tr("Open or create a project first."));
        return;
    }
    auto pkg = _window->_state->vpkg();
    QStringList items;
    for (const auto& e : pkg->segmentEntries()) items << QString::fromStdString(e.location);
    if (items.isEmpty()) {
        QMessageBox::information(_window, QObject::tr("No segments"),
                                 QObject::tr("Attach a segments source first."));
        return;
    }
    bool ok = false;
    int currentIdx = 0;
    if (pkg->hasOutputSegments()) {
        const auto cur = QString::fromStdString(pkg->outputSegmentsPath().string());
        for (int i = 0; i < items.size(); ++i) {
            if (items[i] == cur) { currentIdx = i; break; }
        }
    }
    QString chosen = QInputDialog::getItem(_window, QObject::tr("Set Output Segments"),
        QObject::tr("Where new segments will land:"), items, currentIdx, false, &ok);
    if (!ok || chosen.isEmpty()) return;
    pkg->setOutputSegments(chosen.toStdString());
}

bool MenuActionController::runLegacyVolpkgConvert(const QString& inputLocation, QString* convertedOut)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    QString out = QFileDialog::getSaveFileName(_window,
        QObject::tr("Save converted .volpkg.json"),
        settings.value(vc3d::settings::project::DEFAULT_PATH).toString(),
        QObject::tr("Project (*.volpkg.json)"));
    if (out.isEmpty()) return false;
    if (!out.endsWith(".volpkg.json", Qt::CaseInsensitive)) out += ".volpkg.json";

    QApplication::setOverrideCursor(Qt::WaitCursor);
    auto r = vc::convertVolpkg(inputLocation.toStdString(), std::filesystem::path(out.toStdString()));
    QApplication::restoreOverrideCursor();

    if (!r.ok) {
        QMessageBox box(_window);
        box.setWindowTitle(QObject::tr("Volpkg conversion failed"));
        box.setIcon(QMessageBox::Warning);
        box.setText(QObject::tr("Could not convert this volpkg to .volpkg.json."));
        box.setInformativeText(QObject::tr("Input:  %1\nOutput: %2\n\nReason: %3")
            .arg(inputLocation, out, QString::fromStdString(r.message)));
        box.setStandardButtons(QMessageBox::Ok);
        box.setStyleSheet("QLabel{min-width:600px;}");
        box.exec();
        return false;
    }
    if (!r.message.empty()) {
        QMessageBox::information(_window, QObject::tr("Volpkg converted with warnings"),
            QString::fromStdString(r.message));
    }
    if (convertedOut) *convertedOut = out;
    return true;
}

void MenuActionController::convertLegacyVolpkg()
{
    if (!_window) return;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    auto in = promptLocation(QObject::tr("Convert Legacy Volpkg"),
                             QObject::tr("Pick the legacy volpkg directory or remote URL."),
                             settings.value(vc3d::settings::project::DEFAULT_PATH).toString(),
                             {}, false, true);
    if (in.isEmpty()) return;
    QString out;
    if (!runLegacyVolpkgConvert(in, &out)) return;
    auto reply = QMessageBox::question(_window, QObject::tr("Open converted project?"),
        QObject::tr("Wrote %1 — open it now?").arg(out));
    if (reply == QMessageBox::Yes) openVolpkgAt(out);
}
