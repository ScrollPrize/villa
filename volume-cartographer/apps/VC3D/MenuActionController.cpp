#include "MenuActionController.hpp"

#include "vc/core/cache/HttpMetadataFetcher.hpp"
#include "VCSettings.hpp"
#include "UnifiedBrowserDialog.hpp"
#include "CWindow.hpp"
#include "SurfacePanelController.hpp"
#include "ViewerManager.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "adaptive/CAdaptiveVolumeViewer.hpp"
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

#include "utils/Json.hpp"

#include <algorithm>
#include <filesystem>
#include <map>
#include <unordered_map>

namespace
{

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

    _recentMenu = new QMenu(QObject::tr("Open &recent project"), _fileMenu);
    _recentMenu->setEnabled(false);
    _fileMenu->addMenu(_recentMenu);

    ensureRecentActions();

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
    _window->UpdateView();
}

// --- Remote recents management ---

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
    *authOut = vc::cache::loadAwsCredentials();
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
        if (!surf || surf->meta.is_null()) {
            continue;
        }

        const auto tags = vc::json::tags_or_empty(surf->meta);
        if (!tags.contains("reviewed") || !tags["reviewed"].is_object()) {
            continue;
        }

        const auto& reviewed = tags["reviewed"];

        QString reviewDate = "Unknown";
        const std::string reviewDateRaw = vc::json::string_or(reviewed, "date", std::string{});
        if (!reviewDateRaw.empty()) {
            reviewDate = QString::fromStdString(reviewDateRaw).left(10);
        } else {
            QFileInfo metaFile(QString::fromStdString(surf->path.string()) + "/meta.json");
            if (metaFile.exists()) {
                reviewDate = metaFile.lastModified().toString("yyyy-MM-dd");
            }
        }

        QString username = "Unknown";
        const std::string reviewerUser = vc::json::string_or(reviewed, "user", std::string{});
        if (!reviewerUser.empty()) {
            username = QString::fromStdString(reviewerUser);
        }

        const double area = vc::json::number_or(surf->meta, "area_cm2", 0.0);

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
    dlg.setAuthResolver([this](const QString& url, vc::cache::HttpAuth* out, QString* err) {
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

    QString tool = QCoreApplication::applicationDirPath() + "/vc_volpkg_convert";
    QProcess proc;
    proc.start(tool, {inputLocation, out});
    proc.waitForFinished(-1);
    if (proc.exitCode() != 0) {
        QMessageBox::warning(_window, QObject::tr("Convert failed"),
            QString::fromUtf8(proc.readAllStandardError()) + "\n" + QString::fromUtf8(proc.readAllStandardOutput()));
        return false;
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
