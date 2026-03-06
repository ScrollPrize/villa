#include "MenuActionController.hpp"

#include "VCSettings.hpp"
#include "CWindow.hpp"
#include "SurfacePanelController.hpp"
#include "ViewerManager.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "tiled/CTiledVolumeViewer.hpp"
#include "CVolumeViewerView.hpp"
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
#include "vc/core/cache/HttpMetadataFetcher.hpp"
#include "vc/core/util/RemoteScroll.hpp"
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
#include <QFile>
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
#include <QTextStream>
#include <QUrl>
#include <QVBoxLayout>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <map>
#include <unordered_map>

namespace
{
constexpr auto kRemoteVolumeRegistryFile = "remote_volumes.json";
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
    _recentRemoteActs.fill(nullptr);
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

    _openRemoteAct = new QAction(QObject::tr("Open &Remote Volume..."), this);
    connect(_openRemoteAct, &QAction::triggered, this, &MenuActionController::openRemoteVolume);

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

    _reportingAct = new QAction(QObject::tr("Generate Review Report..."), this);
    connect(_reportingAct, &QAction::triggered, this, &MenuActionController::generateReviewReport);

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

    _teleaAct = new QAction(QObject::tr("Inpaint (Telea) && Rebuild Segment"), this);
    _teleaAct->setToolTip(QObject::tr("Generate RGB, Telea-inpaint it, then convert back to tifxyz into a new segment"));
#if (QT_VERSION >= QT_VERSION_CHECK(5, 10, 0))
    _teleaAct->setShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::TeleaInpaint));
#endif
    connect(_teleaAct, &QAction::triggered, this, &MenuActionController::runTeleaInpaint);

    _importObjAct = new QAction(QObject::tr("Import OBJ as Patch..."), this);
    connect(_importObjAct, &QAction::triggered, this, &MenuActionController::importObjAsPatch);

    // Build menus
    _fileMenu = new QMenu(QObject::tr("&File"), qWindow);
    _fileMenu->addAction(_openAct);
    _fileMenu->addAction(_openLocalZarrAct);
    _fileMenu->addAction(_openRemoteAct);
    _fileMenu->addAction(_attachRemoteZarrAct);

    _recentMenu = new QMenu(QObject::tr("Open &recent volpkg"), _fileMenu);
    _recentMenu->setEnabled(false);
    _fileMenu->addMenu(_recentMenu);

    _recentRemoteMenu = new QMenu(QObject::tr("Open recent re&mote volume"), _fileMenu);
    _recentRemoteMenu->setEnabled(false);
    _fileMenu->addMenu(_recentRemoteMenu);

    ensureRecentActions();
    ensureRecentRemoteActions();

    _fileMenu->addSeparator();
    _fileMenu->addAction(_reportingAct);
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
    _viewMenu->addAction(qWindow->ui.dockWidgetDrawing->toggleViewAction());
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
    _actionsMenu->addAction(_teleaAct);

    _selectionMenu = new QMenu(QObject::tr("&Selection"), qWindow);
    _selectionMenu->addAction(_surfaceFromSelectionAct);
    _selectionMenu->addAction(_selectionClearAct);
    _selectionMenu->addSeparator();
    _selectionMenu->addAction(_teleaAct);

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
    refreshRecentRemoteMenu();
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
    while (urls.size() > kMaxRecentRemote) {
        urls.removeLast();
    }
    saveRecentRemoteUrls(urls);
    refreshRecentRemoteMenu();
}

void MenuActionController::ensureRecentRemoteActions()
{
    if (!_recentRemoteMenu) return;

    for (auto& act : _recentRemoteActs) {
        if (!act) {
            act = new QAction(this);
            act->setVisible(false);
            connect(act, &QAction::triggered, this, &MenuActionController::openRecentRemoteVolume);
            _recentRemoteMenu->addAction(act);
        }
    }
}

void MenuActionController::refreshRecentRemoteMenu()
{
    ensureRecentRemoteActions();

    QStringList urls = loadRecentRemoteUrls();
    if (!urls.isEmpty() && urls.last().isEmpty()) {
        urls.removeLast();
    }

    const int count = std::min(static_cast<int>(urls.size()), kMaxRecentRemote);

    for (int i = 0; i < count; ++i) {
        QString text = QObject::tr("&%1 | %2").arg(i + 1).arg(urls[i]);
        _recentRemoteActs[i]->setText(text);
        _recentRemoteActs[i]->setData(urls[i]);
        _recentRemoteActs[i]->setVisible(true);
    }

    for (int j = count; j < kMaxRecentRemote; ++j) {
        if (_recentRemoteActs[j]) {
            _recentRemoteActs[j]->setVisible(false);
            _recentRemoteActs[j]->setData(QVariant());
        }
    }

    if (_recentRemoteMenu) {
        _recentRemoteMenu->setEnabled(count > 0);
    }
}

void MenuActionController::openRecentRemoteVolume()
{
    if (!_window) return;

    if (auto* action = qobject_cast<QAction*>(sender())) {
        const QString url = action->data().toString();
        if (!url.isEmpty()) {
            openRemoteUrl(url, false);
        }
    }
}

void MenuActionController::openRemoteVolume()
{
    if (!_window) return;

    // Pre-fill with the most recent remote URL
    QStringList recentUrls = loadRecentRemoteUrls();
    QString lastUrl = recentUrls.isEmpty() ? QString() : recentUrls.first();

    bool ok = false;
    QString url = QInputDialog::getText(
        _window,
        QObject::tr("Open Remote Volume"),
        QObject::tr("Enter volume URL (http://, https://, s3://):"),
        QLineEdit::Normal,
        lastUrl,
        &ok);

    if (!ok || url.trimmed().isEmpty()) return;

    openRemoteUrl(url.trimmed(), false);
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
                                                vc::cache::HttpAuth* authOut,
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
    authOut->awsSigv4 = true;
    authOut->region = resolved.awsRegion;

    auto getEnv = [](const char* name) -> std::string {
        const char* v = std::getenv(name);
        return v ? v : "";
    };

    authOut->accessKey = getEnv("AWS_ACCESS_KEY_ID");
    authOut->secretKey = getEnv("AWS_SECRET_ACCESS_KEY");
    authOut->sessionToken = getEnv("AWS_SESSION_TOKEN");

    if (authOut->accessKey.empty() || authOut->secretKey.empty()) {
        const auto savedAccess = settings.value(vc3d::settings::aws::ACCESS_KEY).toString();
        const auto savedSecret = settings.value(vc3d::settings::aws::SECRET_KEY).toString();
        const auto savedToken = settings.value(vc3d::settings::aws::SESSION_TOKEN).toString();

        if (!savedAccess.isEmpty() && !savedSecret.isEmpty()) {
            authOut->accessKey = savedAccess.toStdString();
            authOut->secretKey = savedSecret.toStdString();
            authOut->sessionToken = savedToken.toStdString();
        }
    }

    if (!authOut->accessKey.empty() && !authOut->secretKey.empty()) {
        return true;
    }

    if (!allowPrompt) {
        if (errorMessage) {
            *errorMessage = QObject::tr("Missing AWS credentials for %1").arg(url);
        }
        return false;
    }

    bool credOk = false;
    QString accessKey = QInputDialog::getText(
        _window,
        QObject::tr("AWS Credentials"),
        QObject::tr("AWS_ACCESS_KEY_ID:"),
        QLineEdit::Normal, QString(), &credOk);
    if (!credOk || accessKey.trimmed().isEmpty()) {
        if (errorMessage) {
            *errorMessage = QObject::tr("AWS credential entry canceled.");
        }
        return false;
    }

    QString secretKey = QInputDialog::getText(
        _window,
        QObject::tr("AWS Credentials"),
        QObject::tr("AWS_SECRET_ACCESS_KEY:"),
        QLineEdit::Password, QString(), &credOk);
    if (!credOk || secretKey.trimmed().isEmpty()) {
        if (errorMessage) {
            *errorMessage = QObject::tr("AWS credential entry canceled.");
        }
        return false;
    }

    QString sessionToken = QInputDialog::getText(
        _window,
        QObject::tr("AWS Credentials"),
        QObject::tr("AWS_SESSION_TOKEN (optional, leave blank if not using STS):"),
        QLineEdit::Normal, QString(), &credOk);
    if (!credOk) {
        if (errorMessage) {
            *errorMessage = QObject::tr("AWS credential entry canceled.");
        }
        return false;
    }

    authOut->accessKey = accessKey.trimmed().toStdString();
    authOut->secretKey = secretKey.trimmed().toStdString();
    authOut->sessionToken = sessionToken.trimmed().toStdString();

    settings.setValue(vc3d::settings::aws::ACCESS_KEY, accessKey.trimmed());
    settings.setValue(vc3d::settings::aws::SECRET_KEY, secretKey.trimmed());
    settings.setValue(vc3d::settings::aws::SESSION_TOKEN, sessionToken.trimmed());
    return true;
}

QString MenuActionController::remoteCacheDirectory() const
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    QString defaultCache = QDir::homePath() + "/.VC3D/remote_cache";
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

    nlohmann::json root = {
        {"version", 1},
        {"volumes", nlohmann::json::array()}
    };

    try {
        if (QFileInfo::exists(registryPath)) {
            std::ifstream input(registryPath.toStdString());
            if (input.good()) {
                input >> root;
            }
        }
    } catch (const std::exception& e) {
        Logger()->warn("Failed reading remote volume registry '{}': {}", registryPath.toStdString(), e.what());
        root = {
            {"version", 1},
            {"volumes", nlohmann::json::array()}
        };
    }

    if (!root.is_object()) {
        root = nlohmann::json::object();
    }
    if (!root.contains("volumes") || !root["volumes"].is_array()) {
        root["volumes"] = nlohmann::json::array();
    }
    root["version"] = 1;

    const std::string urlStd = url.trimmed().toStdString();
    const std::string idStd = volume->id();
    nlohmann::json updated = nlohmann::json::array();
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

    nlohmann::json root;
    try {
        std::ifstream input(registryPath.toStdString());
        if (!input.good()) {
            return;
        }
        input >> root;
    } catch (const std::exception& e) {
        Logger()->warn("Failed to parse remote volume registry '{}': {}", registryPath.toStdString(), e.what());
        if (_window->statusBar()) {
            _window->statusBar()->showMessage(QObject::tr("Failed to read remote_volumes.json"), 5000);
        }
        return;
    }

    const auto volumesIt = root.find("volumes");
    if (volumesIt == root.end() || !volumesIt->is_array() || volumesIt->empty()) {
        return;
    }

    const QString currentId = QString::fromStdString(_window->_state->currentVolumeId());
    const QString cacheDir = remoteCacheDirectory();
    int attachedCount = 0;
    int skippedCount = 0;

    for (const auto& entry : *volumesIt) {
        if (!entry.is_object()) {
            continue;
        }

        const QString url = QString::fromStdString(entry.value("url", std::string{})).trimmed();
        if (url.isEmpty()) {
            continue;
        }

        vc::cache::HttpAuth auth;
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
        _window->refreshCurrentVolumePackageUi(currentId, false);
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

    vc::cache::HttpAuth auth;
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

                try {
                    auto volume = watcher->result();
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
                } catch (const std::exception& e) {
                    const QString msg = extractExceptionMessage(e);

                    if (isAuthError(msg)) {
                        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
                        settings.remove(vc3d::settings::aws::ACCESS_KEY);
                        settings.remove(vc3d::settings::aws::SECRET_KEY);
                        settings.remove(vc3d::settings::aws::SESSION_TOKEN);

                        const auto reply = QMessageBox::warning(
                            _window,
                            QObject::tr("Authentication Error"),
                            QObject::tr("Failed to attach remote zarr:\n%1\n\n"
                                        "Would you like to enter new AWS credentials and retry?")
                                .arg(msg),
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
                        QObject::tr("Failed to attach remote zarr:\n%1").arg(msg));
                }
            });

    auto future = QtConcurrent::run([url, auth, cacheDir]() -> std::shared_ptr<Volume> {
        return Volume::NewFromUrl(url.toStdString(), cacheDir.toStdString(), auth);
    });
    watcher->setFuture(future);
}

void MenuActionController::openRemoteUrl(const QString& url, bool isRetry)
{
    if (!_window || url.isEmpty()) return;

    if (!isRetry) {
        _remoteOpenAuthRetries = 0;
        _remoteScrollAuthRetries = 0;
    }

    auto urlStr = url.toStdString();
    auto resolved = vc::resolveRemoteUrl(urlStr);
    vc::cache::HttpAuth auth;
    QString authError;
    if (!tryResolveRemoteAuth(url, &auth, true, &authError)) {
        return;
    }

    const QString cacheDir = remoteCacheDirectory();

    // Save the URL to recents
    updateRecentRemoteList(url);

    // Disable the action while loading to prevent double-open
    _openRemoteAct->setEnabled(false);
    if (_window->statusBar()) {
        _window->statusBar()->showMessage(QObject::tr("Opening remote volume..."));
    }

    auto cachePath = cacheDir.toStdString();

    // Check if this might be a scroll root URL (not ending with .zarr)
    bool isLikelyZarr = resolved.httpsUrl.size() >= 5 &&
        resolved.httpsUrl.substr(resolved.httpsUrl.size() - 5) == ".zarr";
    // Also check without trailing slash
    {
        std::string trimmed = resolved.httpsUrl;
        while (!trimmed.empty() && trimmed.back() == '/') trimmed.pop_back();
        if (trimmed.size() >= 5 && trimmed.substr(trimmed.size() - 5) == ".zarr") {
            isLikelyZarr = true;
        }
    }

    if (!isLikelyZarr) {
        // Try scroll discovery first
        openRemoteScroll(resolved.httpsUrl, auth, cachePath);
    } else {
        // Direct zarr volume open (existing flow)
        openRemoteZarr(resolved.httpsUrl, auth, cachePath);
    }
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

void MenuActionController::openRemoteZarr(
    const std::string& httpsUrl,
    const vc::cache::HttpAuth& auth,
    const std::string& cachePath)
{
    auto* watcher = new QFutureWatcher<std::shared_ptr<Volume>>(this);

    connect(watcher, &QFutureWatcher<std::shared_ptr<Volume>>::finished, this,
        [this, watcher, httpsUrl, cachePath]() {
            watcher->deleteLater();
            _openRemoteAct->setEnabled(true);

            try {
                auto vol = watcher->result();
                _remoteOpenAuthRetries = 0;
                _remoteScrollAuthRetries = 0;
                _window->CloseVolume();
                _window->setVolume(vol);
                _window->UpdateView();

                if (_window->statusBar()) {
                    _window->statusBar()->showMessage(
                        QObject::tr("Opened remote volume: %1")
                            .arg(QString::fromStdString(vol->id())),
                        5000);
                }
            } catch (const std::exception& e) {
                QString msg = extractExceptionMessage(e);

                if (_window->statusBar()) {
                    _window->statusBar()->clearMessage();
                }

                // If it looks like an auth error, offer to re-enter credentials
                if (isAuthError(msg)) {
                    // Clear stale saved credentials
                    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
                    settings.remove(vc3d::settings::aws::ACCESS_KEY);
                    settings.remove(vc3d::settings::aws::SECRET_KEY);
                    settings.remove(vc3d::settings::aws::SESSION_TOKEN);

                    auto reply = QMessageBox::warning(
                        _window,
                        QObject::tr("Authentication Error"),
                        QObject::tr("Failed to open remote volume:\n%1\n\n"
                                    "Would you like to enter new AWS credentials and retry?").arg(msg),
                        QMessageBox::Yes | QMessageBox::No);

                    if (reply == QMessageBox::Yes) {
                        // Re-prompt for credentials by calling openRemoteUrl again.
                        // Use QTimer::singleShot to break the call stack and avoid
                        // deep recursion on repeated auth failures (Issue 31).
                        if (_remoteOpenAuthRetries >= 3) {
                            if (_window->statusBar()) {
                                _window->statusBar()->showMessage(
                                    QObject::tr("Authentication failed after 3 attempts"), 5000);
                            }
                            _remoteOpenAuthRetries = 0;
                            return;
                        }
                        ++_remoteOpenAuthRetries;
                        _openRemoteAct->setEnabled(false);
                        QTimer::singleShot(0, this, [this, httpsUrl]() {
                            openRemoteUrl(QString::fromStdString(httpsUrl), true);
                        });
                        return;
                    }

                    _remoteOpenAuthRetries = 0;
                } else {
                    _remoteOpenAuthRetries = 0;
                    _remoteScrollAuthRetries = 0;
                    QMessageBox::critical(
                        _window,
                        QObject::tr("Remote Volume Error"),
                        QObject::tr("Failed to open remote volume:\n%1").arg(msg));
                }
            }
        });

    auto future = QtConcurrent::run(
        [httpsUrl, cachePath, auth]() -> std::shared_ptr<Volume> {
            return Volume::NewFromUrl(httpsUrl, cachePath, auth);
        });
    watcher->setFuture(future);
}

// Scroll discovery result that can be passed between threads
struct ScrollOpenResult {
    std::shared_ptr<Volume> volume;
    std::vector<std::pair<std::string, std::shared_ptr<Surface>>> surfaces;
    std::string errorMsg;
};

void MenuActionController::openRemoteScroll(
    const std::string& httpsUrl,
    const vc::cache::HttpAuth& auth,
    const std::string& cachePath)
{
    // Phase 1: Discover scroll structure on background thread
    auto* discoveryWatcher = new QFutureWatcher<vc::RemoteScrollInfo>(this);

    connect(discoveryWatcher, &QFutureWatcher<vc::RemoteScrollInfo>::finished, this,
        [this, discoveryWatcher, httpsUrl, auth, cachePath]() {
            discoveryWatcher->deleteLater();

            vc::RemoteScrollInfo scrollInfo;
            try {
                scrollInfo = discoveryWatcher->result();
            } catch (const std::exception& e) {
                // Discovery failed — fall back to direct zarr open
                std::fprintf(stderr, "[RemoteScroll] Discovery failed: %s, falling back to zarr\n", e.what());
                openRemoteZarr(httpsUrl, auth, cachePath);
                return;
            }

            // Auth error — prompt for fresh credentials and retry
            if (scrollInfo.authError) {
                QString msg = QObject::tr("AWS credentials error: %1\n\n"
                    "Please enter fresh credentials.")
                    .arg(QString::fromStdString(scrollInfo.authErrorMessage));
                QMessageBox::warning(_window, QObject::tr("Credentials Expired"), msg);

                vc::cache::HttpAuth freshAuth = auth;
                bool credOk = false;
                QString accessKey = QInputDialog::getText(
                    _window, QObject::tr("AWS Credentials"),
                    QObject::tr("AWS_ACCESS_KEY_ID:"),
                    QLineEdit::Normal, QString(), &credOk);
                if (!credOk || accessKey.trimmed().isEmpty()) {
                    _remoteScrollAuthRetries = 0;
                    _openRemoteAct->setEnabled(true);
                    if (_window->statusBar()) _window->statusBar()->clearMessage();
                    return;
                }

                QString secretKey = QInputDialog::getText(
                    _window, QObject::tr("AWS Credentials"),
                    QObject::tr("AWS_SECRET_ACCESS_KEY:"),
                    QLineEdit::Password, QString(), &credOk);
                if (!credOk || secretKey.trimmed().isEmpty()) {
                    _remoteScrollAuthRetries = 0;
                    _openRemoteAct->setEnabled(true);
                    if (_window->statusBar()) _window->statusBar()->clearMessage();
                    return;
                }

                QString sessionToken = QInputDialog::getText(
                    _window, QObject::tr("AWS Credentials"),
                    QObject::tr("AWS_SESSION_TOKEN (optional):"),
                    QLineEdit::Normal, QString(), &credOk);
                if (!credOk) {
                    _remoteScrollAuthRetries = 0;
                    _openRemoteAct->setEnabled(true);
                    if (_window->statusBar()) _window->statusBar()->clearMessage();
                    return;
                }

                freshAuth.accessKey = accessKey.trimmed().toStdString();
                freshAuth.secretKey = secretKey.trimmed().toStdString();
                freshAuth.sessionToken = sessionToken.trimmed().toStdString();

                // Save the fresh credentials
                QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
                settings.setValue(vc3d::settings::aws::ACCESS_KEY,
                                  QString::fromStdString(freshAuth.accessKey));
                settings.setValue(vc3d::settings::aws::SECRET_KEY,
                                  QString::fromStdString(freshAuth.secretKey));
                settings.setValue(vc3d::settings::aws::SESSION_TOKEN,
                                  QString::fromStdString(freshAuth.sessionToken));

                // Retry discovery with fresh credentials.
                // Use QTimer::singleShot to break the call stack and limit
                // recursive re-entry on repeated auth failures (Issue 31).
                if (_remoteScrollAuthRetries >= 3) {
                    if (_window->statusBar()) {
                        _window->statusBar()->showMessage(
                            QObject::tr("Authentication failed after 3 attempts"), 5000);
                    }
                    _remoteScrollAuthRetries = 0;
                    _openRemoteAct->setEnabled(true);
                    return;
                }
                ++_remoteScrollAuthRetries;
                QTimer::singleShot(0, this, [this, httpsUrl, freshAuth, cachePath]() {
                    openRemoteScroll(httpsUrl, freshAuth, cachePath);
                });
                return;
            }

            _remoteScrollAuthRetries = 0;

            if (scrollInfo.volumeNames.empty()) {
                // No volumes found — fall back to direct zarr open
                std::fprintf(stderr, "[RemoteScroll] No volumes found, falling back to zarr\n");
                openRemoteZarr(httpsUrl, auth, cachePath);
                return;
            }

            // Pick volume: if multiple, ask user; if one, auto-select
            std::string volumeName;
            if (scrollInfo.volumeNames.size() == 1) {
                volumeName = scrollInfo.volumeNames.front();
            } else {
                QStringList items;
                for (const auto& v : scrollInfo.volumeNames) {
                    items << QString::fromStdString(v);
                }
                bool ok = false;
                QString picked = QInputDialog::getItem(
                    _window,
                    QObject::tr("Select Volume"),
                    QObject::tr("Multiple volumes found. Select one:"),
                    items, 0, false, &ok);
                if (!ok || picked.isEmpty()) {
                    _openRemoteAct->setEnabled(true);
                    if (_window->statusBar()) _window->statusBar()->clearMessage();
                    return;
                }
                volumeName = picked.toStdString();
            }

            // Continuation: Phase 2 — open volume + download segments on
            // background thread.  Captured by value so it can be invoked from
            // either the fast path (no external-segment probe) or the async
            // s3ListObjects watcher path below.
            auto continueWithPhase2 = [this, auth, cachePath](
                vc::RemoteScrollInfo scrollInfo, const std::string& volumeName) {

            // Phase 2: Open volume + download segments on background thread
            if (_window->statusBar()) {
                _window->statusBar()->showMessage(
                    QObject::tr("Opening remote scroll (volume: %1, %2 segments)...")
                        .arg(QString::fromStdString(volumeName))
                        .arg(scrollInfo.segmentIds.size()));
            }

            auto* loadWatcher = new QFutureWatcher<ScrollOpenResult>(this);

            connect(loadWatcher, &QFutureWatcher<ScrollOpenResult>::finished, this,
                [this, loadWatcher, scrollInfo, cachePath]() {
                    loadWatcher->deleteLater();
                    _openRemoteAct->setEnabled(true);

                    ScrollOpenResult result;
                    try {
                        result = loadWatcher->result();
                    } catch (const std::exception& e) {
                        QMessageBox::critical(_window,
                            QObject::tr("Remote Scroll Error"),
                            QObject::tr("Failed to open remote scroll:\n%1").arg(e.what()));
                        if (_window->statusBar()) _window->statusBar()->clearMessage();
                        return;
                    }

                    if (!result.errorMsg.empty()) {
                        QMessageBox::critical(_window,
                            QObject::tr("Remote Scroll Error"),
                            QObject::tr("Failed to open remote scroll:\n%1")
                                .arg(QString::fromStdString(result.errorMsg)));
                        if (_window->statusBar()) _window->statusBar()->clearMessage();
                        return;
                    }

                    _window->CloseVolume();

                    _window->setVolume(result.volume);

                    if (!result.surfaces.empty()) {
                        _window->setRemoteSurfaces(result.surfaces);
                    }

                    // Populate volume combo with all discovered volumes
                    if (_window->volSelect && scrollInfo.volumeNames.size() > 1) {
                        const QSignalBlocker blocker{_window->volSelect};
                        _window->volSelect->clear();
                        for (const auto& vname : scrollInfo.volumeNames) {
                            QString label = QString::fromStdString(vname);
                            // Strip .zarr suffix for display
                            if (label.endsWith(QStringLiteral(".zarr"))) {
                                label.chop(5);
                            }
                            _window->volSelect->addItem(label, QString::fromStdString(vname));
                        }
                        // Select the currently loaded volume
                        const QString currentId = QString::fromStdString(result.volume->id());
                        for (int i = 0; i < _window->volSelect->count(); ++i) {
                            if (_window->volSelect->itemData(i).toString().contains(currentId)) {
                                _window->volSelect->setCurrentIndex(i);
                                break;
                            }
                        }
                    }

                    _window->UpdateView();

                    if (_window->statusBar()) {
                        _window->statusBar()->showMessage(
                            QObject::tr("Opened remote scroll: %1 (%2 segments)")
                                .arg(QString::fromStdString(result.volume->id()))
                                .arg(result.surfaces.size()),
                            5000);
                    }
                });

            auto segIds = scrollInfo.segmentIds;
            auto scrollAuth = scrollInfo.auth;
            auto baseUrl = scrollInfo.baseUrl;
            auto segSource = scrollInfo.segmentSource;
            auto segBaseUrl = scrollInfo.segmentsBaseUrl;

            auto loadFuture = QtConcurrent::run(
                [baseUrl, volumeName, segIds, cachePath, scrollAuth, segSource, segBaseUrl]() -> ScrollOpenResult {
                    ScrollOpenResult result;
                    try {
                        // Derive volpkg name from base URL (last path component)
                        std::string volpkgName = baseUrl;
                        while (!volpkgName.empty() && volpkgName.back() == '/') volpkgName.pop_back();
                        auto slash = volpkgName.rfind('/');
                        if (slash != std::string::npos) volpkgName = volpkgName.substr(slash + 1);

                        std::filesystem::path volpkgCache = std::filesystem::path(cachePath) / volpkgName;

                        // Open the volume
                        std::string volumeUrl = baseUrl + "/volumes/" + volumeName;
                        result.volume = Volume::NewFromUrl(volumeUrl, (volpkgCache / "volumes").string(), scrollAuth);

                        // Pick the right base URL for segment downloads
                        const std::string& dlBase = (segSource == vc::RemoteSegmentSource::Direct)
                            ? segBaseUrl : baseUrl;

                        // Download and load segments
                        for (const auto& segId : segIds) {
                            try {
                                auto localDir = vc::downloadRemoteSegment(
                                    dlBase, segId, volpkgCache, scrollAuth, segSource);

                                // Check that meta.json exists (download succeeded)
                                if (!std::filesystem::exists(localDir / "meta.json")) {
                                    std::fprintf(stderr, "[RemoteScroll] Skipping segment %s: no meta.json\n",
                                                 segId.c_str());
                                    continue;
                                }

                                auto seg = Segmentation::New(localDir);
                                if (seg && seg->canLoadSurface()) {
                                    auto surf = seg->loadSurface();
                                    if (surf) {
                                        result.surfaces.emplace_back(segId, surf);
                                    }
                                }
                            } catch (const std::exception& e) {
                                std::fprintf(stderr, "[RemoteScroll] Failed to load segment %s: %s\n",
                                             segId.c_str(), e.what());
                            }
                        }
                    } catch (const std::exception& e) {
                        result.errorMsg = e.what();
                    }
                    return result;
                });
            loadWatcher->setFuture(loadFuture);
            };  // end continueWithPhase2

            // If no segments found, ask user for an external segments URL
            if (scrollInfo.segmentIds.empty()) {
                bool segOk = false;
                QString segUrl = QInputDialog::getText(
                    _window,
                    QObject::tr("Segments Location"),
                    QObject::tr("No segments found in the volpkg.\n"
                                "Enter S3/HTTPS URL of directory containing segments\n"
                                "(leave empty to skip):"),
                    QLineEdit::Normal, QString(), &segOk);

                if (segOk && !segUrl.trimmed().isEmpty()) {
                    auto segResolved = vc::resolveRemoteUrl(segUrl.trimmed().toStdString());
                    vc::cache::HttpAuth segAuth = auth;
                    if (segResolved.useAwsSigv4 && !segAuth.awsSigv4) {
                        segAuth.awsSigv4 = true;
                        segAuth.region = segResolved.awsRegion;
                    }

                    // Normalize trailing slash
                    std::string segBaseUrl = segResolved.httpsUrl;
                    while (!segBaseUrl.empty() && segBaseUrl.back() == '/')
                        segBaseUrl.pop_back();

                    std::fprintf(stderr, "[RemoteScroll] Probing external segments URL: %s\n",
                                 segBaseUrl.c_str());

                    // Run s3ListObjects on a background thread to avoid
                    // blocking the GUI (it has a 30s timeout).
                    auto* s3Watcher = new QFutureWatcher<vc::cache::S3ListResult>(this);
                    connect(s3Watcher, &QFutureWatcher<vc::cache::S3ListResult>::finished, this,
                        [this, s3Watcher, scrollInfo, volumeName, segBaseUrl, segAuth,
                         continueWithPhase2]() mutable {
                            s3Watcher->deleteLater();

                            try {
                                auto extList = s3Watcher->result();
                                if (!extList.prefixes.empty()) {
                                    scrollInfo.segmentSource = vc::RemoteSegmentSource::Direct;
                                    scrollInfo.segmentsBaseUrl = segBaseUrl;
                                    scrollInfo.auth = segAuth;
                                    for (const auto& name : extList.prefixes) {
                                        std::fprintf(stderr, "[RemoteScroll]   external segment: %s\n",
                                                     name.c_str());
                                        scrollInfo.segmentIds.push_back(name);
                                    }
                                    std::fprintf(stderr, "[RemoteScroll] Found %zu external segments\n",
                                                 scrollInfo.segmentIds.size());
                                } else {
                                    std::fprintf(stderr, "[RemoteScroll] No segments found at external URL\n");
                                }
                            } catch (const std::exception& e) {
                                std::fprintf(stderr, "[RemoteScroll] s3ListObjects failed: %s\n", e.what());
                            }

                            continueWithPhase2(std::move(scrollInfo), volumeName);
                        });

                    auto s3Future = QtConcurrent::run(
                        [segBaseUrl, segAuth]() -> vc::cache::S3ListResult {
                            return vc::cache::s3ListObjects(segBaseUrl + "/", segAuth);
                        });
                    s3Watcher->setFuture(s3Future);
                    return;  // Phase 2 will be triggered by s3Watcher callback
                }
            }

            // Fast path: no external-segment probe needed
            continueWithPhase2(std::move(scrollInfo), volumeName);
        });

    auto discoveryFuture = QtConcurrent::run(
        [httpsUrl, auth]() -> vc::RemoteScrollInfo {
            return vc::discoverRemoteScroll(httpsUrl, auth);
        });
    discoveryWatcher->setFuture(discoveryFuture);
}

void MenuActionController::triggerTeleaInpaint()
{
    runTeleaInpaint();
}

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
        _window->_viewerManager->forEachViewer([showDirHints](CTiledVolumeViewer* viewer) {
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

void MenuActionController::generateReviewReport()
{
    if (!_window || !_window->_state->vpkg()) {
        QMessageBox::warning(_window, QObject::tr("Error"), QObject::tr("No volume package loaded."));
        return;
    }

    QString fileName = QFileDialog::getSaveFileName(_window,
        QObject::tr("Save Review Report"),
        "review_report.csv",
        QObject::tr("CSV Files (*.csv)"));

    if (fileName.isEmpty()) {
        return;
    }

    struct UserStats {
        double totalArea = 0.0;
        int surfaceCount = 0;
    };

    std::map<QString, std::map<QString, UserStats>> dailyStats;
    int totalReviewedCount = 0;
    double grandTotalArea = 0.0;

    for (const auto& id : _window->_state->vpkg()->getLoadedSurfaceIDs()) {
        auto surf = _window->_state->vpkg()->getSurface(id);
        if (!surf || !surf->meta) {
            continue;
        }

        nlohmann::json* meta = surf->meta.get();
        const auto tags = vc::json::tags_or_empty(meta);
        const auto itReviewed = tags.find("reviewed");
        if (itReviewed == tags.end() || !itReviewed->is_object()) {
            continue;
        }

        const nlohmann::json& reviewed = *itReviewed;

        QString reviewDate = "Unknown";
        const std::string reviewDateRaw = vc::json::string_or(&reviewed, "date", std::string{});
        if (!reviewDateRaw.empty()) {
            reviewDate = QString::fromStdString(reviewDateRaw).left(10);
        } else {
            QFileInfo metaFile(QString::fromStdString(surf->path.string()) + "/meta.json");
            if (metaFile.exists()) {
                reviewDate = metaFile.lastModified().toString("yyyy-MM-dd");
            }
        }

        QString username = "Unknown";
        const std::string reviewerUser = vc::json::string_or(&reviewed, "user", std::string{});
        if (!reviewerUser.empty()) {
            username = QString::fromStdString(reviewerUser);
        }

        const double area = vc::json::number_or(meta, "area_cm2", 0.0);

        dailyStats[reviewDate][username].totalArea += area;
        dailyStats[reviewDate][username].surfaceCount++;
        totalReviewedCount++;
        grandTotalArea += area;
    }

    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(_window, QObject::tr("Error"), QObject::tr("Could not open file for writing."));
        return;
    }

    QTextStream stream(&file);
    stream << "Date,Username,CM² Reviewed,Surface Count\n";

    for (const auto& dateEntry : dailyStats) {
        const QString& date = dateEntry.first;
        for (const auto& userEntry : dateEntry.second) {
            const QString& username = userEntry.first;
            const UserStats& stats = userEntry.second;
            stream << date << ","
                   << username << ","
                   << QString::number(stats.totalArea, 'f', 3) << ","
                   << stats.surfaceCount << "\n";
        }
    }

    file.close();

    QString message = QObject::tr("Review report saved successfully.\n\n"
                                   "Total reviewed surfaces: %1\n"
                                   "Total area reviewed: %2 cm²\n"
                                   "Days covered: %3")
                           .arg(totalReviewedCount)
                           .arg(grandTotalArea, 0, 'f', 3)
                           .arg(dailyStats.size());

    QMessageBox::information(_window, QObject::tr("Report Generated"), message);
}

void MenuActionController::toggleDrawBBox(bool enabled)
{
    if (!_window || !_window->_viewerManager) {
        return;
    }

    _window->_viewerManager->forEachViewer([this, enabled](CTiledVolumeViewer* viewer) {
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

    CTiledVolumeViewer* segViewer = nullptr;
    _window->_viewerManager->forEachViewer([&segViewer](CTiledVolumeViewer* viewer) {
        if (viewer && viewer->surfName() == "segmentation") {
            segViewer = viewer;
        }
    });

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

    CTiledVolumeViewer* segViewer = _window->segmentationViewer();
    if (!segViewer) {
        _window->statusBar()->showMessage(QObject::tr("No Surface viewer found"), 3000);
        return;
    }

    segViewer->clearSelections();
    _window->statusBar()->showMessage(QObject::tr("Selections cleared"), 2000);
}

void MenuActionController::runTeleaInpaint()
{
    if (!_window) {
        return;
    }

    QList<QTreeWidgetItem*> selectedItems = _window->treeWidgetSurfaces->selectedItems();
    if (selectedItems.isEmpty()) {
        QMessageBox::information(_window, QObject::tr("Info"), QObject::tr("Select a patch/trace first in the Surfaces list."));
        return;
    }

    const QString vc_tifxyz2rgb = find_tool("vc_tifxyz2rgb");
    const QString vc_telea_inpaint = find_tool("vc_telea_inpaint");
    const QString vc_rgb2tifxyz = find_tool("vc_rgb2tifxyz");

    int successCount = 0;
    int failCount = 0;

    for (QTreeWidgetItem* item : selectedItems) {
        const std::string id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString();
        auto surf = _window->_state->vpkg() ? _window->_state->vpkg()->getSurface(id) : nullptr;
        if (!surf) {
            ++failCount;
            continue;
        }

        const std::filesystem::path segDir = surf->path;
        const std::filesystem::path parentDir = segDir.parent_path();
        const std::filesystem::path metaJson = segDir / "meta.json";

        if (!std::filesystem::exists(metaJson)) {
            QMessageBox::warning(_window, QObject::tr("Error"),
                                 QObject::tr("Missing meta.json for %1").arg(QString::fromStdString(id)));
            ++failCount;
            continue;
        }

        const QString stamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmsszzz");
        const QString rgbPngName = QString::fromStdString(id) + "_xyz_rgb_" + stamp + ".png";
        const QString newSegName = QString::fromStdString(id) + "_telea_" + stamp;

        QTemporaryDir tmpInDir;
        QTemporaryDir tmpOutDir;
        if (!tmpInDir.isValid() || !tmpOutDir.isValid()) {
            QMessageBox::warning(_window, QObject::tr("Error"), QObject::tr("Failed to create temporary directories."));
            ++failCount;
            continue;
        }

        const QString rgbPng = QDir(tmpInDir.path()).filePath(rgbPngName);
        {
            QStringList args;
            args << QString::fromStdString(segDir.string())
                 << rgbPng;
            QString log;
            if (!run_cli(_window, vc_tifxyz2rgb, args, &log)) {
                ++failCount;
                continue;
            }
        }

        QString inpaintedPng;
        {
            QStringList args;
            args << rgbPng
                 << (inpaintedPng = QDir(tmpOutDir.path()).filePath(QString::fromStdString(id) + "_inpainted_" + stamp + ".png"))
                 << "--patch" << QString::number(9)
                 << "--iterations" << QString::number(100);
            QString log;
            if (!run_cli(_window, vc_telea_inpaint, args, &log)) {
                ++failCount;
                continue;
            }
        }

        {
            QStringList args;
            args << inpaintedPng
                 << QString::fromStdString(metaJson.string())
                 << QString::fromStdString(parentDir.string())
                 << newSegName
                 << "--invalid-black";
            QString log;
            if (!run_cli(_window, vc_rgb2tifxyz, args, &log)) {
                ++failCount;
                continue;
            }
        }

        ++successCount;
    }

    if (successCount > 0 && _window->_surfacePanel) {
        _window->_surfacePanel->reloadSurfacesFromDisk();
    }

    _window->statusBar()->showMessage(QObject::tr("Telea inpaint pipeline complete. Success: %1, Failed: %2")
                                         .arg(successCount)
                                         .arg(failCount),
                                     6000);
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
