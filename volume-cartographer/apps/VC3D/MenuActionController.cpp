#include "MenuActionController.hpp"

#include "VCSettings.hpp"
#include "CWindow.hpp"
#include "SurfacePanelController.hpp"
#include "ViewerManager.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "CVolumeViewer.hpp"
#include "CVolumeViewerView.hpp"
#include "CSurfaceCollection.hpp"
#include "CommandLineToolRunner.hpp"
#include "SettingsDialog.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "ui_VCMain.h"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/Version.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/JsonSafe.hpp"

#include <QAction>
#include <QApplication>
#include <QClipboard>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QMainWindow>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QProcess>
#include <QStringList>
#include <QSettings>
#include <QStyle>
#include <QTemporaryDir>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QTextStream>
#include <QUrl>

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
    _openAct = new QAction(qWindow->style()->standardIcon(QStyle::SP_DialogOpenButton), QObject::tr("&Open volpkg..."), this);
    _openAct->setShortcut(QKeySequence::Open);
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
    _teleaAct->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_I));
#endif
    connect(_teleaAct, &QAction::triggered, this, &MenuActionController::runTeleaInpaint);

    _importObjAct = new QAction(QObject::tr("Import OBJ as Patch..."), this);
    connect(_importObjAct, &QAction::triggered, this, &MenuActionController::importObjAsPatch);

    // Build menus
    _fileMenu = new QMenu(QObject::tr("&File"), qWindow);
    _fileMenu->addAction(_openAct);

    _recentMenu = new QMenu(QObject::tr("Open &recent volpkg"), _fileMenu);
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
    _viewMenu->addAction(qWindow->ui.dockWidgetComposite->toggleViewAction());
    _viewMenu->addAction(qWindow->ui.dockWidgetView->toggleViewAction());
    _viewMenu->addAction(qWindow->ui.dockWidgetOverlay->toggleViewAction());

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
    return settings.value("volpkg/recent").toStringList();
}

void MenuActionController::saveRecentPaths(const QStringList& paths)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue("volpkg/recent", paths);
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
    bool showDirHints = settings.value("viewer/show_direction_hints", true).toBool();
    if (_window->_viewerManager) {
        _window->_viewerManager->forEachViewer([showDirHints](CVolumeViewer* viewer) {
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

    QMessageBox::information(
        _window,
        QObject::tr("Keybindings for Volume Cartographer"),
        QObject::tr(
            "=== File Menu ===\n"
            "Ctrl+O: Open Volume Package\n"
            "Ctrl+I: Inpaint (Telea) & Rebuild Segment\n"
            "\n"
            "=== View Menu ===\n"
            "Ctrl+J: Toggle axis-aligned slice planes\n"
            "Ctrl+T: Toggle direction hints (flip_x arrows)\n"
            "C: Toggle composite view\n"
            "Ctrl+Shift+D: Toggle drawing mode\n"
            "Space: Toggle volume overlay visibility\n"
            "\n"
            "=== Navigation ===\n"
            "R: Center focus on cursor\n"
            "F: Step backward in focus history\n"
            "Ctrl+F: Step forward in focus history\n"
            "\n"
            "=== Segmentation Editing ===\n"
            "Ctrl+Z: Undo last change (segmentation or approval mask)\n"
            "Shift (hold): Activate invalidation brush\n"
            "S (hold): Line draw mode\n"
            "E: Apply pending brush changes\n"
            "T: Create correction annotation\n"
            "Escape: Cancel current drag/operation\n"
            "\n"
            "=== Approval Mask ===\n"
            "B: Toggle approval painting (when mask is shown)\n"
            "N: Toggle unapproval painting (when mask is shown)\n"
            "Ctrl+B: Undo last approval mask stroke\n"
            "\n"
            "=== Segmentation Growth ===\n"
            "Ctrl+G: Grow segmentation (all directions)\n"
            "1: Grow left\n"
            "2: Grow up\n"
            "3: Grow down\n"
            "4: Grow right\n"
            "5: Grow all directions\n"
            "6: Grow one step (all directions)\n"
            "\n"
            "=== Push/Pull ===\n"
            "A (hold): Push (move inward)\n"
            "D (hold): Pull (move outward)\n"
            "Ctrl+A/D: Push/Pull with alpha override\n"
            "Q: Decrease push/pull radius\n"
            "E: Increase push/pull radius\n"
            "\n"
            "=== Point Collection ===\n"
            "Delete: Remove selected point\n"
            "\n"
            "=== Mouse Controls ===\n"
            "Left Click: Select/Add point or drag surface\n"
            "Left Drag: Drag surface point or draw with active tool\n"
            "Right Drag: Pan slice image\n"
            "Middle Drag: Pan (if enabled)\n"
            "Mouse Wheel: Zoom in/out (when editing: adjust tool radius)\n"));
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
    if (!_window || !_window->fVpkg) {
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

    for (const auto& id : _window->fVpkg->getLoadedSurfaceIDs()) {
        auto surfMeta = _window->fVpkg->getSurface(id);
        if (!surfMeta || !surfMeta->surface() || !surfMeta->surface()->meta) {
            continue;
        }

        nlohmann::json* meta = surfMeta->surface()->meta;
        const auto tags = vc::json_safe::tags_or_empty(meta);
        const auto itReviewed = tags.find("reviewed");
        if (itReviewed == tags.end() || !itReviewed->is_object()) {
            continue;
        }

        const nlohmann::json& reviewed = *itReviewed;

        QString reviewDate = "Unknown";
        const std::string reviewDateRaw = vc::json_safe::string_or(&reviewed, "date", std::string{});
        if (!reviewDateRaw.empty()) {
            reviewDate = QString::fromStdString(reviewDateRaw).left(10);
        } else {
            QFileInfo metaFile(QString::fromStdString(surfMeta->path.string()) + "/meta.json");
            if (metaFile.exists()) {
                reviewDate = metaFile.lastModified().toString("yyyy-MM-dd");
            }
        }

        QString username = "Unknown";
        const std::string reviewerUser = vc::json_safe::string_or(&reviewed, "user", std::string{});
        if (!reviewerUser.empty()) {
            username = QString::fromStdString(reviewerUser);
        }

        const double area = vc::json_safe::number_or(meta, "area_cm2", 0.0);

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

    _window->_viewerManager->forEachViewer([this, enabled](CVolumeViewer* viewer) {
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
    if (!_window || !_window->_viewerManager || !_window->fVpkg) {
        return;
    }

    CVolumeViewer* segViewer = nullptr;
    _window->_viewerManager->forEachViewer([&segViewer](CVolumeViewer* viewer) {
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

    if (_window->_surfID.empty() || !_window->fVpkg->getSurface(_window->_surfID)) {
        _window->statusBar()->showMessage(QObject::tr("Select a segmentation first"), 3000);
        return;
    }

    auto surfMeta = _window->fVpkg->getSurface(_window->_surfID);
    std::filesystem::path baseSegPath = surfMeta->path;
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

        std::string newId = _window->_surfID + std::string("_sel_") + ts.toStdString() + std::string("_") + std::to_string(idx++);
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

    CVolumeViewer* segViewer = _window->segmentationViewer();
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
        auto surfMeta = _window->fVpkg ? _window->fVpkg->getSurface(id) : nullptr;
        if (!surfMeta) {
            ++failCount;
            continue;
        }

        const std::filesystem::path segDir = surfMeta->path;
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
    if (!_window || !_window->fVpkg) {
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

    auto pathsDirFs = std::filesystem::path(_window->fVpkg->getVolpkgDirectory()) /
                      std::filesystem::path(_window->fVpkg->getSegmentationDirectory());
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
