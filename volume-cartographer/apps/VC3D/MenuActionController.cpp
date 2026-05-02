#include "MenuActionController.hpp"

#include "VCSettings.hpp"
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
#include <filesystem>
#include <unordered_map>

namespace
{
constexpr auto kRemoteVolumeRegistryFile = "remote_volumes.json";
constexpr int kMaxStoredRemoteUrls = 10;
QString extractExceptionMessage(const std::exception& e);
bool isAuthError(const QString& msg);

static bool run_cli(QWidget* parent, const QString& program, const QStringList& args, QString* outLog = nullptr)
{
    QProcess process;
    process.setProcessChannelMode(QProcess::MergedChannels);
    process.start(program, args);
    if (!process.waitForStarted()) {
        QMessageBox::critical(parent, QObject::tr("Error"), QObject::tr("Failed to start %1").arg(program));
        return false;
    }
    process.waitForFinished(-1);
    const QString log = process.readAll();
    if (outLog) {
        *outLog = log;
    }
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
        QMessageBox::critical(parent, QObject::tr("Command Failed"),
                              QObject::tr("%1 exited with code %2.\n\n%3")
                                  .arg(program)
                                  .arg(process.exitCode())
                                  .arg(log));
        return false;
    }
    return true;
}

static QString find_tool(const char* baseName)
{
#ifdef _WIN32
    const QString exe = QString::fromLatin1(baseName) + ".exe";
#else
    const QString exe = QString::fromLatin1(baseName);
#endif
    const QString appDir = QCoreApplication::applicationDirPath();
    const QString local = appDir + QDir::separator() + exe;
    if (QFileInfo::exists(local)) {
        return local;
    }
    return exe;
}

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
    _openAct = new QAction(qWindow->style()->standardIcon(QStyle::SP_DialogOpenButton), QObject::tr("&Open volpkg..."), this);
    _openAct->setShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::OpenVolpkg));
    connect(_openAct, &QAction::triggered, this, &MenuActionController::openVolpkg);

    _openLocalZarrAct = new QAction(QObject::tr("Open Local &Zarr..."), this);
    connect(_openLocalZarrAct, &QAction::triggered, this, &MenuActionController::openLocalZarr);

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
    _fileMenu->addAction(_openAct);
    _fileMenu->addAction(_openLocalZarrAct);
    _fileMenu->addAction(_attachRemoteZarrAct);

    _recentMenu = new QMenu(QObject::tr("Open &recent volpkg"), _fileMenu);
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
    return settings.value(vc3d::settings::volpkg::RECENT).toStringList();
}

void MenuActionController::saveRecentPaths(const QStringList& paths)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::volpkg::RECENT, paths);
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

    _window->CloseVolume();
    _window->OpenVolume(QString());
    loadAttachedRemoteVolumesForCurrentPackage();
    _window->UpdateView();
}

void MenuActionController::openLocalZarr()
{
    if (!_window) return;

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    QString dir = QFileDialog::getExistingDirectory(
        _window,
        QObject::tr("Open Local OME-Zarr Directory"),
        settings.value(vc3d::settings::volpkg::DEFAULT_PATH).toString(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks |
        QFileDialog::ReadOnly | QFileDialog::DontUseNativeDialog);

    if (dir.isEmpty()) return;

    auto path = std::filesystem::path(dir.toStdString());

    // Validate that this looks like a zarr directory
    if (!Volume::checkDir(path)) {
        QMessageBox::warning(
            _window, QObject::tr("Not a Zarr Volume"),
            QObject::tr("The selected directory does not appear to be an "
                         "OME-Zarr volume (no .zgroup, .zattrs, or meta.json found)."));
        return;
    }

    try {
        auto vol = Volume::New(path);
        _window->CloseVolume();
        _window->setVolume(vol);
        _window->UpdateView();

        if (_window->statusBar()) {
            _window->statusBar()->showMessage(
                QObject::tr("Opened local zarr: %1")
                    .arg(QString::fromStdString(vol->id())),
                5000);
        }
    } catch (const std::exception& e) {
        QMessageBox::critical(
            _window, QObject::tr("Error Opening Zarr"),
            QObject::tr("Failed to open zarr volume:\n%1")
                .arg(QString::fromStdString(e.what())));
    }
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

    auto pathsDirFs = std::filesystem::path(_window->_state->vpkg()->getVolpkgDirectory()) /
                      std::filesystem::path(_window->_state->vpkg()->getSegmentationDirectory());
    QString pathsDir = QString::fromStdString(pathsDirFs.string());

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
    if (!_window || !_window->_transformOverlay) {
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

    _window->_transformOverlay->beginRotate();
    if (_window->statusBar()) {
        _window->statusBar()->showMessage(QObject::tr("Surface rotation active"), 3000);
    }
}
