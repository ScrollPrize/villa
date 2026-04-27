#include "MenuActionController.hpp"

#include "vc/core/cache/HttpMetadataFetcher.hpp"
#include "VCSettings.hpp"
#include "CWindow.hpp"
#include "SurfacePanelController.hpp"
#include "ViewerManager.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "adaptive/CAdaptiveVolumeViewer.hpp"
#include "CVolumeViewerView.hpp"
#include "CommandLineToolRunner.hpp"
#include "SettingsDialog.hpp"
#include "UnifiedBrowserDialog.hpp"
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
#include <QPushButton>
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
#include "utils/http_fetch.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <map>
#include <unordered_map>
#include <unordered_set>

namespace
{
QString extractExceptionMessage(const std::exception& e);
bool isAuthError(const QString& msg);
void autosave_project(const vc::Volpkg& proj);
std::string unique_source_id(const vc::Volpkg& proj, std::string base);

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
    _newVolpkgAct = new QAction(QObject::tr("&New Volpkg"), this);
    connect(_newVolpkgAct, &QAction::triggered, this, &MenuActionController::newVolpkg);

    _openAct = new QAction(qWindow->style()->standardIcon(QStyle::SP_DialogOpenButton), QObject::tr("&Open Volpkg..."), this);
    _openAct->setShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::OpenVolpkg));
    connect(_openAct, &QAction::triggered, this, &MenuActionController::openVolpkg);

    _saveAsAct = new QAction(QObject::tr("&Save Volpkg As..."), this);
    connect(_saveAsAct, &QAction::triggered, this, &MenuActionController::saveVolpkgAs);

    _attachVolumeAct = new QAction(QObject::tr("Attach &Volume..."), this);
    connect(_attachVolumeAct, &QAction::triggered, this, &MenuActionController::attachVolume);

    _attachSegmentsAct = new QAction(QObject::tr("Attach &Segments..."), this);
    connect(_attachSegmentsAct, &QAction::triggered, this, &MenuActionController::attachSegments);

    _attachOtherAct = new QAction(QObject::tr("Attach &Other..."), this);
    connect(_attachOtherAct, &QAction::triggered, this, &MenuActionController::attachOther);

    _attachFromVolpkgAct = new QAction(QObject::tr("Attach &From Volpkg..."), this);
    connect(_attachFromVolpkgAct, &QAction::triggered, this, &MenuActionController::attachFromVolpkg);

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

    _fileMenu = new QMenu(QObject::tr("&File"), qWindow);
    _fileMenu->addAction(_newVolpkgAct);
    _fileMenu->addAction(_openAct);

    _recentMenu = new QMenu(QObject::tr("Open R&ecent"), _fileMenu);
    _recentMenu->setEnabled(false);
    _fileMenu->addMenu(_recentMenu);

    _recentRemoteMenu = new QMenu(QObject::tr("Open recent re&mote"), _fileMenu);
    _recentRemoteMenu->setEnabled(false);
    _fileMenu->addMenu(_recentRemoteMenu);

    ensureRecentActions();
    ensureRecentRemoteActions();

    _fileMenu->addSeparator();
    _fileMenu->addAction(_saveAsAct);
    _fileMenu->addSeparator();
    _fileMenu->addAction(_attachVolumeAct);
    _fileMenu->addAction(_attachSegmentsAct);
    _fileMenu->addAction(_attachOtherAct);
    _fileMenu->addAction(_attachFromVolpkgAct);
    _fileMenu->addSeparator();
    _fileMenu->addAction(_reportingAct);
    _fileMenu->addAction(_importObjAct);
    _fileMenu->addAction(_settingsAct);
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

// --- Recent projects (JSON-backed project files) ---

void MenuActionController::newVolpkg()
{
    loadEmptyVolumePackage();
}

void MenuActionController::openVolpkg()
{
    if (!_window) return;
    UnifiedBrowserDialog dlg(_window);
    dlg.setWindowTitle(QObject::tr("Open Volpkg"));
    dlg.setHint(QObject::tr(
        "Pick a .volpkg.json file, a .volpkg/ directory, or a remote URL"));
    dlg.setAcceptsFiles(true);
    dlg.setAcceptsDirs(true);
    dlg.setLocalNameFilters({QStringLiteral("*.volpkg.json"),
                              QStringLiteral("*.volpkg")});
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    dlg.setStartUri(settings.value(vc3d::settings::volpkg::DEFAULT_PATH).toString());
    dlg.setAuthResolver(
        [this](const QString& url, vc::cache::HttpAuth* out, QString* err) {
            return tryResolveRemoteAuth(url, out, true, err);
        });
    if (dlg.exec() != QDialog::Accepted) return;
    QString uri = dlg.selectedUri();
    if (uri.isEmpty()) return;
    if (uri.startsWith(QLatin1String("file://"))) uri = uri.mid(7);
    while (uri.size() > 1 && uri.endsWith('/')) uri.chop(1);
    openFile(uri);
}

void MenuActionController::openRecentVolpkg()
{
    if (!_window) return;
    if (auto* action = qobject_cast<QAction*>(sender())) {
        openFile(action->data().toString());
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

void MenuActionController::openFile(const QString& pathOrUrl)
{
    if (!_window || pathOrUrl.isEmpty()) return;

    const QString trimmed = pathOrUrl.trimmed();
    const auto kind = vc::infer_location_kind(trimmed.toStdString());
    if (kind == vc::LocationKind::Remote) {
        openRemoteUrl(trimmed, false);
        return;
    }

    namespace fs = std::filesystem;
    const fs::path p(trimmed.toStdString());
    std::error_code ec;
    if (fs::is_regular_file(p, ec)) {
        const QString lower = trimmed.toLower();
        if (lower.endsWith(".volpkg.json") || lower.endsWith(".json")) {
            try {
                auto proj = std::make_shared<vc::Volpkg>(
                    vc::Volpkg::load_from_file(p));
                _window->CloseVolume();
                _window->_state->setPackage(VolumePkg::New(proj), proj);
                auto loadAllOfType = [&](vc::DataSourceType t) {
                    for (const auto* ds : proj->sources_of_type(t))
                        if (ds) loadSource(*proj, *ds);
                };
                loadAllOfType(vc::DataSourceType::ZarrVolume);
                loadAllOfType(vc::DataSourceType::VolumesDir);
                loadAllOfType(vc::DataSourceType::SegmentsDir);
                if (auto vpkg = _window->_state->vpkg()) {
                    if (!_window->_state->currentVolume() && vpkg->numberOfVolumes() > 0) {
                        const auto& ids = vpkg->volumeIDs();
                        if (!ids.empty()) {
                            try { _window->setVolume(vpkg->volume(ids.front())); }
                            catch (...) {}
                        }
                    }
                }
                _window->refreshCurrentVolumePackageUi(QString(), true);
                autosave_project(*proj);
                updateRecentVolpkgList(trimmed);
                _window->UpdateView();
            } catch (const std::exception& e) {
                QMessageBox::critical(_window, QObject::tr("Open"),
                    QObject::tr("Failed to load %1:\n%2")
                        .arg(trimmed, QString::fromStdString(e.what())));
            }
        } else {
            QMessageBox::warning(_window, QObject::tr("Open"),
                QObject::tr("Unsupported file: %1").arg(trimmed));
        }
        return;
    }

    if (fs::is_directory(p, ec)) {
        if (vc::Volpkg::looks_like_volpkg(p)) {
            openVolpkgAt(trimmed);
            return;
        }
        if (Volume::checkDir(p)) {
            try {
                auto vol = Volume::New(p);
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
                QMessageBox::critical(_window, QObject::tr("Open"),
                    QObject::tr("Failed to open zarr:\n%1")
                        .arg(QString::fromStdString(e.what())));
            }
            return;
        }
        QMessageBox::warning(_window, QObject::tr("Open"),
            QObject::tr("Directory isn't a volpkg or zarr volume: %1").arg(trimmed));
        return;
    }

    QMessageBox::warning(_window, QObject::tr("Open"),
        QObject::tr("Cannot open: %1").arg(trimmed));
}

bool MenuActionController::tryRestoreAutosavedProject()
{
    if (!_window) return false;
    const QString path = vc3d::currentProjectFilePath();
    if (!QFileInfo::exists(path)) return false;

    std::shared_ptr<vc::Volpkg> proj;
    try {
        proj = std::make_shared<vc::Volpkg>(
            vc::Volpkg::load_from_file(path.toStdString()));
    } catch (const std::exception& e) {
        Logger()->warn("Failed to restore autosaved project '{}': {}",
                       path.toStdString(), e.what());
        return false;
    }

    _window->CloseVolume();
    if (proj->is_volpkg_compatible()) {
        const auto rootStr = proj->origin->root.string();
        if (!std::filesystem::exists(rootStr)) {
            Logger()->info("Autosaved project's volpkg '{}' no longer exists; skipping",
                           rootStr);
            return false;
        }
    }

    _window->_state->setPackage(VolumePkg::New(proj), proj);

    // Kick off any remote sources so populated cache metadata can arrive
    // asynchronously (dedup-safe for local sources already ingested by
    // VolumePkg::New above).
    auto loadAllOfType = [&](vc::DataSourceType type) {
        for (const auto* ds : proj->sources_of_type(type)) {
            if (ds) loadSource(*proj, *ds);
        }
    };
    loadAllOfType(vc::DataSourceType::ZarrVolume);
    loadAllOfType(vc::DataSourceType::VolumesDir);
    loadAllOfType(vc::DataSourceType::SegmentsDir);

    if (auto vpkg = _window->_state->vpkg()) {
        if (!_window->_state->currentVolume()
            && vpkg->numberOfVolumes() > 0) {
            const auto& ids = vpkg->volumeIDs();
            if (!ids.empty()) {
                try { _window->setVolume(vpkg->volume(ids.front())); }
                catch (...) { /* best-effort */ }
            }
        }
    }

    _window->refreshCurrentVolumePackageUi(QString(), true);
    _window->UpdateView();
    return true;
}

int MenuActionController::attachDataSource(vc::DataSource ds)
{
    if (!_window) return 0;
    auto proj = _window->_state->project();
    if (!proj) return 0;

    if (ds.id.empty()) ds.id = "source";
    ds.id = unique_source_id(*proj, ds.id);

    proj->data_sources.push_back(std::move(ds));
    const int loaded = loadSource(*proj, proj->data_sources.back());
    autosave_project(*proj);
    _window->_state->setProject(proj);
    _window->refreshCurrentVolumePackageUi(QString(), true);
    _window->UpdateView();
    return loaded;
}

void MenuActionController::loadEmptyVolumePackage()
{
    if (!_window) return;

    auto proj = std::make_shared<vc::Volpkg>();
    proj->name = "Untitled";
    proj->version = 1;

    _window->CloseVolume();
    _window->_state->setPackage(VolumePkg::New(proj), proj);

    // Overwrite the session autosave so a relaunch doesn't restore the
    // previous project. Without this the in-memory state is empty but
    // ~/.VC3D/current_project.json still points at whatever was open.
    try {
        proj->save_to_file(vc3d::currentProjectFilePath().toStdString());
    } catch (const std::exception& e) {
        Logger()->warn("New Volpkg: failed to overwrite autosave: {}", e.what());
    }

    _window->refreshCurrentVolumePackageUi(QString(), true);
    _window->UpdateView();
}

// --- Project / Data menu slots ---

namespace {

void autosave_project(const vc::Volpkg& proj)
{
    const auto out = vc3d::currentProjectFilePath().toStdString();
    try {
        proj.save_to_file(out);
    } catch (const std::exception& e) {
        Logger()->warn("Project autosave failed ({}): {}", out, e.what());
    }
}

QStringList source_type_options()
{
    return {
        QObject::tr("Segments directory"),
        QObject::tr("Single segment"),
        QObject::tr("Volumes directory"),
        QObject::tr("Zarr volume"),
        QObject::tr("Normal grid"),
        QObject::tr("Normal/dir volume"),
    };
}

QString prompt_source_type(QWidget* parent, int defaultIndex = 0)
{
    const QStringList options = source_type_options();
    bool ok = false;
    const QString pick = QInputDialog::getItem(
        parent, QObject::tr("Data type"),
        QObject::tr("Select the kind of data to add:"),
        options, defaultIndex, false, &ok);
    if (!ok) return {};
    return pick;
}

// Guess the most likely source type from a URL string. Rough heuristic —
// user can always pick something else from the dropdown.
int guess_type_index_for_url(const QString& url)
{
    const auto options = source_type_options();
    const QString u = url.trimmed().toLower();
    if (u.endsWith(".zarr") || u.endsWith(".zarr/")) {
        return options.indexOf(QObject::tr("Zarr volume"));
    }
    if (u.contains("/volumes") && !u.contains("/volumes/")) {
        return options.indexOf(QObject::tr("Volumes directory"));
    }
    if (u.contains("/paths") || u.contains("/segments")
        || u.endsWith("/paths/") || u.endsWith("/segments/"))
    {
        return options.indexOf(QObject::tr("Segments directory"));
    }
    return 0;
}

vc::DataSourceType source_type_from_label(const QString& label)
{
    if (label == QObject::tr("Segments directory"))  return vc::DataSourceType::SegmentsDir;
    if (label == QObject::tr("Single segment"))      return vc::DataSourceType::Segment;
    if (label == QObject::tr("Volumes directory"))   return vc::DataSourceType::VolumesDir;
    if (label == QObject::tr("Zarr volume"))         return vc::DataSourceType::ZarrVolume;
    if (label == QObject::tr("Normal grid"))         return vc::DataSourceType::NormalGrid;
    if (label == QObject::tr("Normal/dir volume"))   return vc::DataSourceType::NormalDirVolume;
    return vc::DataSourceType::SegmentsDir;
}

std::string unique_source_id(const vc::Volpkg& proj, std::string base)
{
    if (base.empty()) base = "source";
    if (!proj.find_source(base)) return base;
    for (int i = 2; i < 10000; ++i) {
        auto candidate = base + "_" + std::to_string(i);
        if (!proj.find_source(candidate)) return candidate;
    }
    return base + "_dup";
}

} // namespace

int MenuActionController::loadSource(const vc::Volpkg& proj,
                                     const vc::DataSource& ds)
{
    if (!_window) return 0;
    if (!_window->_state->vpkg()) return 0;
    if (!ds.enabled) return 0;

    // SyncDir always goes async — even when location is a local path, the
    // remote side of the mirror needs network.
    if (ds.type == vc::DataSourceType::SyncDir) {
        loadSourceRemoteAsync(proj, ds);
        return 0;
    }

    if (ds.location_kind == vc::LocationKind::Local) {
        return loadSourceLocal(proj, ds);
    }
    loadSourceRemoteAsync(proj, ds);
    return 0;
}

int MenuActionController::loadSourceLocal(const vc::Volpkg& proj,
                                          const vc::DataSource& ds)
{
    auto vpkg = _window->_state->vpkg();
    std::filesystem::path root;
    try { root = proj.resolve_local(ds); }
    catch (const std::exception&) { return 0; }
    if (!std::filesystem::exists(root)) return 0;

    int n = 0;
    switch (ds.type) {
        case vc::DataSourceType::ZarrVolume:
            if (vpkg->addVolumeAt(root)) ++n;
            break;
        case vc::DataSourceType::VolumesDir:
            if (!ds.recursive) {
                if (vpkg->addVolumeAt(root)) ++n;
            } else {
                for (const auto& entry : std::filesystem::directory_iterator(root)) {
                    if (entry.is_directory() && vpkg->addVolumeAt(entry.path())) ++n;
                }
            }
            break;
        case vc::DataSourceType::Segment:
            if (vpkg->addSegmentationAt(root, ds.id)) ++n;
            break;
        case vc::DataSourceType::SegmentsDir:
            if (!ds.recursive) {
                if (vpkg->addSegmentationAt(root, ds.id)) ++n;
            } else {
                for (const auto& entry : std::filesystem::directory_iterator(root)) {
                    if (!entry.is_directory()) continue;
                    const auto name = entry.path().filename().string();
                    if (name.empty() || name[0] == '.' || name == ".tmp") continue;
                    if (vpkg->addSegmentationAt(entry.path(), ds.id)) ++n;
                }
            }
            break;
        case vc::DataSourceType::NormalGrid:
        case vc::DataSourceType::NormalDirVolume:
            // No VolumePkg loader for these yet; recorded but not loaded.
            break;
        case vc::DataSourceType::SyncDir:
            // Should have been routed to the async path by loadSource.
            break;
    }
    return n;
}

namespace {
// Return value of the remote-load background worker. Holds plain data so it
// can be marshalled back to the main thread without touching Qt objects.
struct RemoteSegmentEntry {
    std::filesystem::path localDir;
    std::string segId;
    std::string groupId;
    bool fullyCached = false;     // true iff TIFFs already on disk (no stub needed)
    CState::RemoteSegmentInfo info;
};
struct RemoteLoadResult {
    std::vector<std::shared_ptr<Volume>> volumes;
    std::vector<RemoteSegmentEntry> segments;
    std::string logSummary;
};

// List remote "directories" directly at `url` — used as a fallback when a URL
// isn't laid out as a volpkg root (no volumes/, paths/, or segments/ subdir).
// For an S3-compatible bucket `s3ListObjects` returns common prefixes which
// are the immediate subdirectory names.
std::vector<std::string> listBareRemotePrefixes(
    const std::string& url, const vc::cache::HttpAuth& auth)
{
    std::string u = url;
    while (!u.empty() && u.back() == '/') u.pop_back();
    auto r = vc::cache::s3ListObjects(u + "/", auth);
    if (r.authError) return {};
    return r.prefixes;
}

// Pull-mirror a remote S3/HTTP directory tree into a local directory.
// Downloads every file that isn't already present locally. Honours the
// shared cancel flag by bailing out of the recursion.
int syncDirPull(const std::string& remoteUrl,
                const std::filesystem::path& localRoot,
                const vc::cache::HttpAuth& auth,
                const std::shared_ptr<std::atomic<bool>>& cancelled)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(localRoot, ec);
    int count = 0;

    std::vector<std::pair<std::string, fs::path>> stack;
    stack.emplace_back(remoteUrl, localRoot);
    while (!stack.empty()) {
        if (cancelled && cancelled->load()) break;

        auto [ru, ld] = stack.back();
        stack.pop_back();

        std::string url = ru;
        while (!url.empty() && url.back() == '/') url.pop_back();
        url += "/";

        auto list = vc::cache::s3ListObjects(url, auth);
        if (list.authError) {
            Logger()->warn("syncDirPull: auth error on {}: {}",
                           url, list.errorMessage);
            break;
        }
        fs::create_directories(ld, ec);

        for (const auto& objRel : list.objects) {
            if (cancelled && cancelled->load()) break;
            const std::string objUrl = url + objRel;
            const fs::path destPath = ld / objRel;
            fs::create_directories(destPath.parent_path(), ec);
            if (fs::exists(destPath)) continue;
            if (vc::cache::httpDownloadFile(objUrl, destPath, auth)) {
                ++count;
            } else {
                Logger()->warn("syncDirPull: failed to download {}", objUrl);
            }
        }
        for (const auto& pref : list.prefixes) {
            stack.emplace_back(url + pref, ld / pref);
        }
    }
    return count;
}

// Walk a remote tree once and collect the relative paths of every object.
// Used by syncDirPush so we can skip uploading what's already there without
// HEADing each file individually.
void collectRemoteObjects(const std::string& remoteUrl,
                          const vc::cache::HttpAuth& auth,
                          const std::shared_ptr<std::atomic<bool>>& cancelled,
                          std::unordered_set<std::string>& out)
{
    std::vector<std::pair<std::string, std::string>> stack;
    stack.emplace_back(remoteUrl, std::string());
    while (!stack.empty()) {
        if (cancelled && cancelled->load()) break;

        auto [ru, prefix] = stack.back();
        stack.pop_back();

        std::string url = ru;
        while (!url.empty() && url.back() == '/') url.pop_back();
        url += "/";

        auto list = vc::cache::s3ListObjects(url, auth);
        if (list.authError) {
            Logger()->warn("syncDirPush: auth error listing {}: {}",
                           url, list.errorMessage);
            break;
        }
        for (const auto& objRel : list.objects) {
            out.insert(prefix + objRel);
        }
        for (const auto& pref : list.prefixes) {
            stack.emplace_back(url + pref, prefix + pref);
        }
    }
}

// Push-mirror a local directory tree to a remote S3/HTTP location. Uploads
// every local file whose relative path is not already present remotely.
int syncDirPush(const std::string& remoteUrl,
                const std::filesystem::path& localRoot,
                const vc::cache::HttpAuth& auth,
                const std::shared_ptr<std::atomic<bool>>& cancelled)
{
    namespace fs = std::filesystem;
    if (!fs::exists(localRoot) || !fs::is_directory(localRoot)) return 0;

    std::unordered_set<std::string> remoteSet;
    collectRemoteObjects(remoteUrl, auth, cancelled, remoteSet);
    if (cancelled && cancelled->load()) return 0;

    std::string base = remoteUrl;
    while (!base.empty() && base.back() == '/') base.pop_back();
    base += "/";

    utils::HttpClient::Config cfg;
    cfg.aws_auth = auth;
    cfg.transfer_timeout = std::chrono::seconds(120);
    utils::HttpClient client(cfg);

    int count = 0;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(localRoot, ec);
         it != fs::recursive_directory_iterator(); ++it)
    {
        if (cancelled && cancelled->load()) break;
        if (!it->is_regular_file(ec)) continue;
        const auto rel = fs::relative(it->path(), localRoot, ec);
        if (ec) continue;
        std::string relStr = rel.generic_string();
        if (remoteSet.count(relStr)) continue;
        const std::string objUrl = base + relStr;
        try {
            auto resp = client.put_file(objUrl, it->path());
            if (resp.ok()) {
                ++count;
            } else {
                Logger()->warn("syncDirPush: PUT {} failed (status {})",
                               objUrl, resp.status_code);
            }
        } catch (const std::exception& e) {
            Logger()->warn("syncDirPush: PUT {} threw: {}", objUrl, e.what());
        }
    }
    return count;
}
} // namespace

void MenuActionController::loadSourceRemoteAsync(const vc::Volpkg& proj,
                                                 const vc::DataSource& ds)
{
    // For ordinary remote sources, `url` is the HTTP/S3 location. For
    // SyncDir it's the *local* target directory; the remote side is
    // carried separately in `syncRemote`.
    std::string url;
    if (ds.type == vc::DataSourceType::SyncDir) {
        try { url = proj.resolve_local(ds).string(); }
        catch (const std::exception&) { url = ds.location; }
    } else {
        url = ds.location;
    }
    const std::string syncRemote = ds.sync_remote;
    const auto syncDirection = ds.sync_direction;

    // Auth resolution happens on the main thread (may prompt).
    vc::cache::HttpAuth auth;
    QString authError;
    // For SyncDir the URL used for auth resolution is the remote side,
    // not the local target path.
    const std::string authUrl = ds.type == vc::DataSourceType::SyncDir
                                    ? syncRemote : url;
    if (!tryResolveRemoteAuth(QString::fromStdString(authUrl), &auth, false, &authError)) {
        Logger()->warn("loadSource: auth failed for {}: {}",
                       url, authError.toStdString());
        if (_window && _window->statusBar()) {
            _window->statusBar()->showMessage(
                tr("Could not resolve auth for %1").arg(QString::fromStdString(url)),
                5000);
        }
        return;
    }
    const std::string cacheDir = remoteCacheDirectory().toStdString();

    // Cooperative cancellation: worker checks this between segments.
    auto cancelled = std::make_shared<std::atomic<bool>>(false);

    // Temporary status-bar cancel button, cleaned up when the future
    // completes (see the finished handler below).
    QPushButton* cancelBtn = nullptr;
    QLabel* statusLbl = nullptr;
    if (_window && _window->statusBar()) {
        statusLbl = new QLabel(
            tr("Loading remote %1...").arg(QString::fromStdString(url)),
            _window->statusBar());
        _window->statusBar()->addPermanentWidget(statusLbl);
        cancelBtn = new QPushButton(tr("Cancel"), _window->statusBar());
        _window->statusBar()->addPermanentWidget(cancelBtn);
        QObject::connect(cancelBtn, &QPushButton::clicked, _window, [cancelled]() {
            cancelled->store(true);
        });
    }

    // Capture everything the worker needs by value.
    const auto type = ds.type;
    const std::string groupId = ds.id;

    // Keep a weak-ish handle on the vpkg that was active when we kicked off,
    // so we can discard the result if the user switched volumes mid-flight.
    auto vpkgAtDispatch = _window->_state->vpkg();

    auto* watcher = new QFutureWatcher<RemoteLoadResult>(this);
    connect(watcher, &QFutureWatcher<RemoteLoadResult>::finished, this,
        [this, watcher, vpkgAtDispatch, groupId, cancelled, cancelBtn, statusLbl]() {
            watcher->deleteLater();
            // Remove the cancel button + loading label from status bar.
            if (cancelBtn) { cancelBtn->hide(); cancelBtn->deleteLater(); }
            if (statusLbl) { statusLbl->hide(); statusLbl->deleteLater(); }
            if (!_window || _window->_state->vpkg() != vpkgAtDispatch) {
                // Volpkg changed mid-download — discard.
                return;
            }
            if (cancelled->load()) {
                if (_window->statusBar()) {
                    _window->statusBar()->showMessage(tr("Remote load cancelled"), 3000);
                }
                return;
            }
            RemoteLoadResult r;
            try { r = watcher->result(); }
            catch (const std::exception& e) {
                Logger()->warn("loadSource: remote worker failed: {}", e.what());
                if (_window->statusBar()) {
                    _window->statusBar()->showMessage(
                        tr("Remote load failed: %1").arg(QString::fromStdString(e.what())),
                        5000);
                }
                return;
            }

            int n = 0;
            auto vpkg = _window->_state->vpkg();
            for (auto& v : r.volumes) {
                if (v && !vpkg->hasVolume(v->id()) && vpkg->addVolume(v)) ++n;
            }
            std::vector<std::string> stubIds;
            for (const auto& e : r.segments) {
                if (!vpkg->addSegmentationAt(e.localDir, e.groupId)) continue;
                ++n;
                // Record per-segment download info so an on-demand fetch
                // uses the right base URL / auth / cache / layout.
                _window->_state->registerRemoteSegment(e.segId, e.info);
                if (!e.fullyCached) {
                    stubIds.push_back(e.segId);
                }
            }
            if (_window->statusBar()) {
                _window->statusBar()->showMessage(
                    tr("Loaded %1 remote items (%2)")
                        .arg(n)
                        .arg(QString::fromStdString(r.logSummary)),
                    5000);
            }
            if (n > 0) {
                // Pure-JSON projects land here with a stub VolumePkg that
                // has no active segmentation dir and no current volume
                // picked. Set reasonable defaults so widgets aren't stuck
                // greyed out after the async load lands.
                if (!groupId.empty() && vpkg->getSegmentationDirectory().empty()) {
                    vpkg->setSegmentationDirectory(groupId);
                }
                if (!_window->_state->currentVolume()
                    && vpkg->numberOfVolumes() > 0) {
                    const auto& ids = vpkg->volumeIDs();
                    if (!ids.empty()) {
                        try { _window->setVolume(vpkg->volume(ids.front())); }
                        catch (...) { /* best-effort */ }
                    }
                }
                _window->refreshCurrentVolumePackageUi(QString(), true);
                // Tree items now exist; mark the metadata-only entries as
                // stubs so the click-to-download path fires on selection.
                if (_window->_surfacePanel) {
                    for (const auto& segId : stubIds) {
                        _window->_surfacePanel->markAsRemoteStub(segId);
                    }
                }
            }
        });

    auto future = QtConcurrent::run(
        [url, cacheDir, auth, type, groupId, cancelled,
         syncRemote, syncDirection]() -> RemoteLoadResult {
            // Install a per-thread cancel token so any curl transfer on
            // *this* worker thread aborts immediately when Cancel is hit
            // (CURLE_ABORTED_BY_CALLBACK fires on the next xfer tick).
            utils::CancelScope cancelScope(cancelled.get());
            RemoteLoadResult r;
            switch (type) {
                case vc::DataSourceType::ZarrVolume: {
                    try {
                        auto v = Volume::NewFromUrl(url, cacheDir, auth);
                        if (v) r.volumes.push_back(std::move(v));
                    } catch (const std::exception& e) {
                        Logger()->warn("loadSource[worker]: zarr '{}' failed: {}",
                                       url, e.what());
                    }
                    r.logSummary = std::to_string(r.volumes.size()) + " zarr";
                    break;
                }
                case vc::DataSourceType::VolumesDir: {
                    vc::RemoteScrollInfo info;
                    try { info = vc::discoverRemoteScroll(url, auth); }
                    catch (const std::exception& e) {
                        Logger()->warn("loadSource[worker]: discover '{}' failed: {}",
                                       url, e.what());
                    }
                    std::vector<std::string> volumeNames = info.volumeNames;
                    std::string base = info.baseUrl.empty() ? url : info.baseUrl;
                    std::string prefix = "/volumes/";
                    if (volumeNames.empty()) {
                        // Bare-list fallback: treat the URL itself as a
                        // flat directory whose children are volume dirs.
                        volumeNames = listBareRemotePrefixes(url, auth);
                        base = url;
                        prefix = "/";
                    }
                    for (const auto& name : volumeNames) {
                        if (cancelled->load()) break;
                        std::string trimmedBase = base;
                        while (!trimmedBase.empty() && trimmedBase.back() == '/') trimmedBase.pop_back();
                        std::string volName = name;
                        while (!volName.empty() && volName.back() == '/') volName.pop_back();
                        const std::string volUrl = trimmedBase + prefix + volName;
                        try {
                            auto v = Volume::NewFromUrl(volUrl, cacheDir, auth);
                            if (v) r.volumes.push_back(std::move(v));
                        } catch (const std::exception& e) {
                            Logger()->warn("loadSource[worker]: volume '{}' failed: {}",
                                           volUrl, e.what());
                        }
                    }
                    r.logSummary = std::to_string(r.volumes.size()) + " volumes";
                    break;
                }
                case vc::DataSourceType::Segment: {
                    // Single-segment is an explicit "load this one" action —
                    // fetch everything now so it's ready for surface display.
                    try {
                        std::string u = url;
                        while (!u.empty() && u.back() == '/') u.pop_back();
                        const auto slash = u.find_last_of('/');
                        if (slash == std::string::npos) break;
                        const std::string base = u.substr(0, slash);
                        const std::string segId = u.substr(slash + 1);
                        auto localDir = vc::downloadRemoteSegment(
                            base, segId, cacheDir, auth,
                            vc::RemoteSegmentSource::Direct);
                        if (std::filesystem::exists(localDir / "meta.json")) {
                            RemoteSegmentEntry entry;
                            entry.localDir = localDir;
                            entry.segId = segId;
                            entry.groupId = groupId;
                            entry.fullyCached = vc::isRemoteSegmentFullyCached(
                                cacheDir, segId, vc::RemoteSegmentSource::Direct);
                            entry.info = {base, cacheDir, auth,
                                          vc::RemoteSegmentSource::Direct};
                            r.segments.push_back(std::move(entry));
                        }
                    } catch (const std::exception& e) {
                        Logger()->warn("loadSource[worker]: segment '{}' failed: {}",
                                       url, e.what());
                    }
                    r.logSummary = std::to_string(r.segments.size()) + " segments";
                    break;
                }
                case vc::DataSourceType::SegmentsDir: {
                    // Lazy: list segment ids and pull meta.json only. TIFF
                    // payloads are fetched on demand when the user clicks.
                    vc::RemoteScrollInfo info;
                    try { info = vc::discoverRemoteScroll(url, auth); }
                    catch (const std::exception& e) {
                        Logger()->warn("loadSource[worker]: discover '{}' failed: {}",
                                       url, e.what());
                    }
                    std::vector<std::string> segIds = info.segmentIds;
                    std::string base = info.segmentsBaseUrl.empty()
                        ? info.baseUrl : info.segmentsBaseUrl;
                    vc::RemoteSegmentSource src = info.segmentSource;

                    if (segIds.empty()) {
                        // Bare-list fallback: URL children are segment dirs.
                        segIds = listBareRemotePrefixes(url, auth);
                        std::string u = url;
                        while (!u.empty() && u.back() == '/') u.pop_back();
                        base = u;
                        src = vc::RemoteSegmentSource::Direct;
                    }

                    // Normalize ids (strip trailing '/').
                    for (auto& segId : segIds) {
                        while (!segId.empty() && segId.back() == '/') segId.pop_back();
                    }

                    // Parallel metadata fetch — each meta.json is tiny and
                    // the latency dominates, so fanning out across Qt's
                    // global thread pool keeps listing responsive even for
                    // thousands of segments.
                    auto fetchOne = [base, cacheDir, auth, src, groupId, cancelled]
                                    (const std::string& segId) -> RemoteSegmentEntry
                    {
                        // Each parallel task may run on a different pool
                        // thread, so install the cancel scope per-call.
                        utils::CancelScope innerScope(cancelled.get());
                        RemoteSegmentEntry e;
                        e.segId = segId;
                        e.groupId = groupId;
                        if (cancelled && cancelled->load()) return e;
                        try {
                            auto localDir = vc::downloadRemoteSegmentMetadataOnly(
                                base, segId, cacheDir, auth, src);
                            if (std::filesystem::exists(localDir / "meta.json")) {
                                e.localDir = localDir;
                                e.fullyCached = vc::isRemoteSegmentFullyCached(
                                    cacheDir, segId, src);
                                e.info = {base, cacheDir, auth, src};
                            }
                        } catch (const std::exception& ex) {
                            Logger()->warn("loadSource[worker]: segment '{}' failed: {}",
                                           segId, ex.what());
                        }
                        return e;
                    };

                    const auto mapped = QtConcurrent::blockingMapped(segIds, fetchOne);
                    for (const auto& e : mapped) {
                        if (!e.localDir.empty()) {
                            r.segments.push_back(e);
                        }
                    }
                    r.logSummary = std::to_string(r.segments.size()) + "/"
                                 + std::to_string(segIds.size()) + " segments (lazy)";
                    break;
                }
                case vc::DataSourceType::NormalGrid:
                case vc::DataSourceType::NormalDirVolume:
                    break;
                case vc::DataSourceType::SyncDir: {
                    int pulled = 0;
                    int pushed = 0;
                    const std::filesystem::path localPath(url);
                    if (syncDirection == vc::SyncDirection::Pull
                        || syncDirection == vc::SyncDirection::Both)
                    {
                        pulled = syncDirPull(
                            syncRemote, localPath, auth, cancelled);
                    }
                    if (syncDirection == vc::SyncDirection::Push
                        || syncDirection == vc::SyncDirection::Both)
                    {
                        pushed = syncDirPush(
                            syncRemote, localPath, auth, cancelled);
                    }
                    r.logSummary =
                        std::to_string(pulled) + " pulled, "
                        + std::to_string(pushed) + " pushed";
                    break;
                }
            }
            return r;
        });
    watcher->setFuture(future);
}

void MenuActionController::unloadSource(const vc::Volpkg& proj,
                                        const vc::DataSource& ds)
{
    if (!_window) return;
    auto vpkg = _window->_state->vpkg();
    if (!vpkg) return;

    // Segments are tracked by group == ds.id, regardless of local/remote.
    if (ds.type == vc::DataSourceType::Segment
        || ds.type == vc::DataSourceType::SegmentsDir)
    {
        for (const auto& segId : vpkg->segmentationIDsInGroup(ds.id)) {
            vpkg->removeSingleSegmentation(segId);
        }
        return;
    }

    if (ds.type == vc::DataSourceType::ZarrVolume
        || ds.type == vc::DataSourceType::VolumesDir)
    {
        std::filesystem::path localRoot;
        if (ds.location_kind == vc::LocationKind::Local) {
            try { localRoot = proj.resolve_local(ds); }
            catch (const std::exception&) { return; }
        } else {
            // Remote: the cached copy sits under <remote cache>/volumes/<id>
            // but volumes are keyed by their remote URL in hasVolume(id).
            // Match by URL prefix.
            std::vector<std::string> victims;
            for (const auto& volId : vpkg->volumeIDs()) {
                if (volId.rfind(ds.location, 0) == 0) {
                    victims.push_back(volId);
                }
            }
            for (const auto& id : victims) vpkg->removeSingleVolume(id);
            return;
        }

        std::vector<std::string> victims;
        for (const auto& volId : vpkg->volumeIDs()) {
            auto vol = vpkg->volume(volId);
            if (!vol) continue;
            std::error_code ec;
            auto rel = std::filesystem::relative(vol->path(), localRoot, ec);
            if (!ec && !rel.empty() && *rel.begin() != "..") {
                victims.push_back(volId);
            }
        }
        for (const auto& id : victims) vpkg->removeSingleVolume(id);
    }
}

void MenuActionController::saveVolpkgAs()
{
    if (!_window) return;
    auto proj = _window->_state->project();
    if (!proj) return;

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QString start = settings.value(vc3d::settings::volpkg::DEFAULT_PATH).toString();
    QString defaultName = proj->name.empty() ? QString("untitled") : QString::fromStdString(proj->name);
    QString path = QFileDialog::getSaveFileName(
        _window, QObject::tr("Save Volpkg As"),
        start.isEmpty() ? defaultName + ".volpkg.json"
                        : start + "/" + defaultName + ".volpkg.json",
        QObject::tr("Volpkg (*.volpkg.json)"),
        nullptr, QFileDialog::DontUseNativeDialog);
    if (path.isEmpty()) return;
    if (!path.endsWith(".json", Qt::CaseInsensitive)) {
        path += ".volpkg.json";
    }

    try {
        proj->save_to_file(path.toStdString());
        proj->set_path(path.toStdString());
        updateRecentVolpkgList(path);
        if (_window->statusBar()) {
            _window->statusBar()->showMessage(
                QObject::tr("Saved volpkg to %1").arg(path), 5000);
        }
    } catch (const std::exception& e) {
        QMessageBox::critical(_window, QObject::tr("Save Volpkg"),
            QObject::tr("Failed to save:\n%1").arg(QString::fromStdString(e.what())));
    }
}

namespace {

// Browse for a directory or file. Returns "" if cancelled. Resolves to a
// plain local path (no file:// prefix) or a remote URL.
QString runUnifiedBrowse(MenuActionController* controller,
                         CWindow* window,
                         const QString& title,
                         const QString& hint,
                         bool acceptFiles,
                         bool acceptDirs,
                         const QStringList& localFilters)
{
    UnifiedBrowserDialog dlg(window);
    dlg.setWindowTitle(title);
    dlg.setHint(hint);
    dlg.setAcceptsFiles(acceptFiles);
    dlg.setAcceptsDirs(acceptDirs);
    dlg.setLocalNameFilters(localFilters);
    dlg.setAuthResolver(
        [controller](const QString& url, vc::cache::HttpAuth* out, QString* err) {
            return controller ? controller->resolveAuthForBrowser(url, out, err) : false;
        });
    if (dlg.exec() != QDialog::Accepted) return {};
    QString uri = dlg.selectedUri().trimmed();
    if (uri.isEmpty()) return {};
    if (uri.startsWith(QLatin1String("file://"))) {
        uri = uri.mid(7);
    }
    return uri;
}

}  // namespace

void MenuActionController::attachVolume()
{
    if (!_window) return;
    if (!_window->_state || !_window->_state->vpkg()) {
        QMessageBox::warning(_window, QObject::tr("Attach Volume"),
            QObject::tr("Open a volpkg first."));
        return;
    }
    QString uri = runUnifiedBrowse(this, _window,
        QObject::tr("Attach Volume"),
        QObject::tr("Pick a single zarr or a directory containing zarrs"),
        false, true, {});
    if (uri.isEmpty()) return;

    const auto kind = vc::infer_location_kind(uri.toStdString());
    namespace fs = std::filesystem;

    vc::DataSourceType type = vc::DataSourceType::ZarrVolume;
    bool recursive = false;
    if (kind == vc::LocationKind::Local) {
        const fs::path p(uri.toStdString());
        std::error_code ec;
        bool isZarr = fs::exists(p / ".zarray", ec)
                   || fs::exists(p / "zarr.json", ec);
        if (isZarr) {
            type = vc::DataSourceType::ZarrVolume;
        } else if (fs::is_directory(p, ec)) {
            type = vc::DataSourceType::VolumesDir;
            recursive = true;
        }
    } else {
        type = vc::DataSourceType::ZarrVolume;
    }

    QString suggested;
    if (kind == vc::LocationKind::Local) {
        suggested = QFileInfo(uri).fileName();
    } else {
        suggested = QUrl(uri).fileName();
    }
    bool ok = false;
    const QString id = QInputDialog::getText(_window,
        QObject::tr("Source ID"),
        QObject::tr("ID for this volume source:"),
        QLineEdit::Normal, suggested, &ok);
    if (!ok) return;

    vc::DataSource ds;
    ds.id = id.toStdString();
    ds.type = type;
    ds.location = uri.toStdString();
    ds.location_kind = kind;
    ds.recursive = recursive;
    attachDataSource(std::move(ds));
}

void MenuActionController::attachSegments()
{
    if (!_window) return;
    if (!_window->_state || !_window->_state->vpkg()) {
        QMessageBox::warning(_window, QObject::tr("Attach Segments"),
            QObject::tr("Open a volpkg first."));
        return;
    }
    QString uri = runUnifiedBrowse(this, _window,
        QObject::tr("Attach Segments"),
        QObject::tr("Pick a directory containing segment subdirectories"),
        false, true, {});
    if (uri.isEmpty()) return;

    QString suggested;
    if (vc::infer_location_kind(uri.toStdString()) == vc::LocationKind::Local) {
        suggested = QFileInfo(uri).fileName();
    } else {
        suggested = QUrl(uri).fileName();
    }
    bool ok = false;
    const QString id = QInputDialog::getText(_window,
        QObject::tr("Source ID"),
        QObject::tr("ID for this segments source:"),
        QLineEdit::Normal, suggested, &ok);
    if (!ok) return;

    vc::DataSource ds;
    ds.id = id.toStdString();
    ds.type = vc::DataSourceType::SegmentsDir;
    ds.location = uri.toStdString();
    ds.location_kind = vc::infer_location_kind(uri.toStdString());
    ds.recursive = true;
    attachDataSource(std::move(ds));
}

void MenuActionController::attachOther()
{
    if (!_window) return;
    if (!_window->_state || !_window->_state->vpkg()) {
        QMessageBox::warning(_window, QObject::tr("Attach"),
            QObject::tr("Open a volpkg first."));
        return;
    }

    const QStringList typeLabels{
        QObject::tr("Single segment"),
        QObject::tr("Normal grid"),
        QObject::tr("Normal-direction volume")};
    bool ok = false;
    const QString chosen = QInputDialog::getItem(_window,
        QObject::tr("Attach Other"),
        QObject::tr("Type:"), typeLabels, 0, false, &ok);
    if (!ok) return;
    const int idx = typeLabels.indexOf(chosen);

    vc::DataSourceType type;
    bool wantsFile = false;
    bool wantsDir = false;
    QStringList filters;
    QString hint;
    if (idx == 0) {
        type = vc::DataSourceType::Segment;
        wantsDir = true;
        hint = QObject::tr("Pick a single segment directory");
    } else if (idx == 1) {
        type = vc::DataSourceType::NormalGrid;
        wantsFile = true;
        hint = QObject::tr("Pick a normal grid file");
    } else {
        type = vc::DataSourceType::NormalDirVolume;
        wantsDir = true;
        hint = QObject::tr("Pick a normal-direction volume directory");
    }

    QString uri = runUnifiedBrowse(this, _window,
        QObject::tr("Attach Other"), hint, wantsFile, wantsDir, filters);
    if (uri.isEmpty()) return;

    QString suggested;
    if (vc::infer_location_kind(uri.toStdString()) == vc::LocationKind::Local) {
        suggested = QFileInfo(uri).fileName();
    } else {
        suggested = QUrl(uri).fileName();
    }
    const QString id = QInputDialog::getText(_window,
        QObject::tr("Source ID"),
        QObject::tr("ID for this source:"),
        QLineEdit::Normal, suggested, &ok);
    if (!ok) return;

    vc::DataSource ds;
    ds.id = id.toStdString();
    ds.type = type;
    ds.location = uri.toStdString();
    ds.location_kind = vc::infer_location_kind(uri.toStdString());
    attachDataSource(std::move(ds));
}

void MenuActionController::attachFromVolpkg()
{
    if (!_window) return;
    auto proj = _window->_state->project();
    if (!proj) return;

    QString uri = runUnifiedBrowse(this, _window,
        QObject::tr("Attach From Volpkg"),
        QObject::tr("Pick another volpkg's .volpkg.json to import sources from"),
        true, false,
        {QStringLiteral("*.volpkg.json"), QStringLiteral("*.json")});
    if (uri.isEmpty()) return;

    vc::Volpkg other;
    try {
        other = vc::Volpkg::load_from_file(uri.toStdString());
    } catch (const std::exception& e) {
        QMessageBox::critical(_window, QObject::tr("Attach From Volpkg"),
            QObject::tr("Failed to load:\n%1").arg(QString::fromStdString(e.what())));
        return;
    }

    if (other.data_sources.empty()) {
        QMessageBox::information(_window, QObject::tr("Attach From Volpkg"),
            QObject::tr("Selected volpkg has no data sources."));
        return;
    }

    QStringList labels;
    for (const auto& ds : other.data_sources) {
        labels.push_back(QString::fromStdString(
            ds.id + " [" + vc::data_source_type_to_string(ds.type) + "]"));
    }
    bool ok = false;
    const QString picked = QInputDialog::getItem(
        _window, QObject::tr("Attach From Volpkg"),
        QObject::tr("Select source to import:"),
        labels, 0, false, &ok);
    if (!ok) return;

    const int idx = labels.indexOf(picked);
    if (idx < 0) return;

    auto ds = other.data_sources[idx];
    ds.id = unique_source_id(*proj, ds.id);
    proj->data_sources.push_back(std::move(ds));
    const int loaded = loadSource(*proj, proj->data_sources.back());
    autosave_project(*proj);
    _window->_state->setProject(proj);
    if (loaded > 0) {
        _window->refreshCurrentVolumePackageUi(QString(), true);
    }
}

namespace {
int findSourceIdx(const vc::Volpkg& proj, const std::string& id)
{
    for (std::size_t i = 0; i < proj.data_sources.size(); ++i) {
        if (proj.data_sources[i].id == id) return int(i);
    }
    return -1;
}
} // namespace

void MenuActionController::reloadSourceById(const QString& sourceId)
{
    if (!_window) return;
    auto proj = _window->_state->project();
    if (!proj) return;
    const int idx = findSourceIdx(*proj, sourceId.toStdString());
    if (idx < 0) return;
    auto ds = proj->data_sources[idx];
    unloadSource(*proj, ds);
    const int loaded = loadSource(*proj, ds);
    _window->_state->setProject(proj);
    if (loaded > 0) {
        _window->refreshCurrentVolumePackageUi(QString(), true);
    }
}

void MenuActionController::removeSourceById(const QString& sourceId)
{
    if (!_window) return;
    auto proj = _window->_state->project();
    if (!proj) return;
    const int idx = findSourceIdx(*proj, sourceId.toStdString());
    if (idx < 0) return;
    auto ds = proj->data_sources[idx];
    if (ds.imported) {
        QMessageBox::information(_window, QObject::tr("Remove Source"),
            QObject::tr("Imported sources can only be removed by editing the linked project."));
        return;
    }
    unloadSource(*proj, ds);
    proj->data_sources.erase(proj->data_sources.begin() + idx);
    if (proj->active_segments_source_id == ds.id) proj->active_segments_source_id.clear();
    if (proj->output_segments_source_id == ds.id) proj->output_segments_source_id.clear();
    autosave_project(*proj);
    _window->_state->setProject(proj);
    _window->refreshCurrentVolumePackageUi(QString(), true);
}

void MenuActionController::renameSourceById(const QString& sourceId)
{
    if (!_window) return;
    auto proj = _window->_state->project();
    if (!proj) return;
    const int idx = findSourceIdx(*proj, sourceId.toStdString());
    if (idx < 0) return;
    if (proj->data_sources[idx].imported) {
        QMessageBox::information(_window, QObject::tr("Rename Source"),
            QObject::tr("Imported sources cannot be renamed here."));
        return;
    }

    bool ok = false;
    const QString newIdQ = QInputDialog::getText(
        _window, QObject::tr("Rename Source"),
        QObject::tr("New ID:"), QLineEdit::Normal, sourceId, &ok);
    if (!ok) return;
    const std::string newId = newIdQ.toStdString();
    if (newId.empty() || newId == sourceId.toStdString()) return;
    if (proj->find_source(newId)) {
        QMessageBox::warning(_window, QObject::tr("Rename Source"),
            QObject::tr("Another source already uses that ID."));
        return;
    }
    proj->data_sources[idx].id = newId;
    if (proj->active_segments_source_id == sourceId.toStdString()) {
        proj->active_segments_source_id = newId;
    }
    if (proj->output_segments_source_id == sourceId.toStdString()) {
        proj->output_segments_source_id = newId;
    }
    autosave_project(*proj);
    _window->_state->setProject(proj);
}

void MenuActionController::setSourceEnabled(const QString& sourceId, bool enabled)
{
    if (!_window) return;
    auto proj = _window->_state->project();
    if (!proj) return;
    const int idx = findSourceIdx(*proj, sourceId.toStdString());
    if (idx < 0) return;
    if (proj->data_sources[idx].enabled == enabled) return;
    proj->data_sources[idx].enabled = enabled;
    if (enabled) {
        loadSource(*proj, proj->data_sources[idx]);
    } else {
        unloadSource(*proj, proj->data_sources[idx]);
    }
    autosave_project(*proj);
    _window->_state->setProject(proj);
    _window->refreshCurrentVolumePackageUi(QString(), true);
}

void MenuActionController::editSourceTags(const QString& sourceId)
{
    if (!_window) return;
    auto proj = _window->_state->project();
    if (!proj) return;
    const int idx = findSourceIdx(*proj, sourceId.toStdString());
    if (idx < 0) return;

    QStringList existing;
    for (const auto& t : proj->data_sources[idx].tags) {
        existing.push_back(QString::fromStdString(t));
    }
    bool ok = false;
    const QString input = QInputDialog::getText(
        _window, QObject::tr("Edit Tags"),
        QObject::tr("Comma-separated tags:"),
        QLineEdit::Normal, existing.join(QStringLiteral(", ")), &ok);
    if (!ok) return;

    std::vector<std::string> newTags;
    for (const auto& piece : input.split(',', Qt::SkipEmptyParts)) {
        const QString trimmed = piece.trimmed();
        if (!trimmed.isEmpty()) newTags.push_back(trimmed.toStdString());
    }
    proj->data_sources[idx].tags = std::move(newTags);
    autosave_project(*proj);
    _window->_state->setProject(proj);
}

void MenuActionController::revealSourceLocation(const QString& sourceId)
{
    if (!_window) return;
    auto proj = _window->_state->project();
    if (!proj) return;
    const auto* ds = proj->find_source(sourceId.toStdString());
    if (!ds) return;
    if (ds->location_kind == vc::LocationKind::Remote) {
        QDesktopServices::openUrl(QUrl(QString::fromStdString(ds->location)));
        return;
    }
    try {
        const auto p = proj->resolve_local(*ds);
        QDesktopServices::openUrl(QUrl::fromLocalFile(QString::fromStdString(p.string())));
    } catch (const std::exception&) {}
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

bool MenuActionController::resolveAuthForBrowser(const QString& url,
                                                  vc::cache::HttpAuth* out,
                                                  QString* err)
{
    return tryResolveRemoteAuth(url, out, true, err);
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

    authOut->access_key = accessKey.trimmed().toStdString();
    authOut->secret_key = secretKey.trimmed().toStdString();
    authOut->session_token = sessionToken.trimmed().toStdString();

    settings.setValue(vc3d::settings::aws::ACCESS_KEY, accessKey.trimmed());
    settings.setValue(vc3d::settings::aws::SECRET_KEY, secretKey.trimmed());
    settings.setValue(vc3d::settings::aws::SESSION_TOKEN, sessionToken.trimmed());
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


void MenuActionController::attachRemoteZarrUrl(const QString& url, bool /*persistEntry*/)
{
    if (!_window) return;

    const QString trimmedUrl = url.trimmed();
    const std::string urlStd = trimmedUrl.toStdString();

    auto resolved = vc::resolveRemoteUrl(urlStd);
    std::string canonical = resolved.httpsUrl;
    while (!canonical.empty() && canonical.back() == '/') canonical.pop_back();
    const bool looksLikeZarr = canonical.size() >= 5
        && canonical.substr(canonical.size() - 5) == ".zarr";
    if (!looksLikeZarr) {
        QMessageBox::warning(_window,
                             QObject::tr("Expected Remote Zarr"),
                             QObject::tr("Attach Remote Zarr expects a direct .zarr URL, not a scroll root."));
        return;
    }

    updateRecentRemoteList(trimmedUrl);

    vc::DataSource ds;
    QString suggestedId = QUrl(trimmedUrl).fileName();
    if (suggestedId.endsWith(".zarr", Qt::CaseInsensitive)) {
        suggestedId.chop(5);
    }
    ds.id = suggestedId.isEmpty() ? std::string("zarr") : suggestedId.toStdString();
    ds.type = vc::DataSourceType::ZarrVolume;
    ds.location = urlStd;
    ds.location_kind = vc::LocationKind::Remote;
    attachDataSource(std::move(ds));
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
    _openAct->setEnabled(false);
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
        [this, watcher, httpsUrl, cachePath, auth]() {
            watcher->deleteLater();
            _openAct->setEnabled(true);

            auto future = watcher->future();
            QString errorMsg;

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
            if (future.isValid() && !future.isCanceled() && future.isResultReadyAt(0)) {
#else
            if (future.isFinished() && !future.isCanceled()) {
#endif
                try {
                    auto vol = future.result();
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

                    // If the user previously attached a remote-segments dir
                    // to this exact zarr URL, auto-re-attach it silently.
                    // Otherwise offer the prompt.
                    {
                        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
                        const QString prevZarr = settings.value(
                            vc3d::settings::viewer::LAST_REMOTE_SEGMENTS_FOR_ZARR)
                                .toString();
                        const QString prevSeg = settings.value(
                            vc3d::settings::viewer::LAST_REMOTE_SEGMENTS_URL)
                                .toString();
                        const QString currentZarr = QString::fromStdString(httpsUrl);
                        if (!prevSeg.isEmpty() && prevZarr == currentZarr) {
                            loadRemoteSegmentsWithUrl(prevSeg, auth, cachePath,
                                                       currentZarr);
                            return;
                        }
                    }
                    // Offer to load remote segments
                    promptAndLoadRemoteSegments(auth, cachePath);
                    return;
                } catch (const std::exception& e) {
                    errorMsg = extractExceptionMessage(e);
                } catch (...) {
                    errorMsg = QObject::tr("Unknown error opening remote volume");
                }
            } else {
                // Future finished but no result ready — exception was stored
                try {
                    future.waitForFinished();
                    future.result(); // will re-throw
                } catch (const std::exception& e) {
                    errorMsg = extractExceptionMessage(e);
                } catch (...) {
                    errorMsg = QObject::tr("Unknown error opening remote volume");
                }
            }

            // Error path
            if (_window->statusBar()) {
                _window->statusBar()->clearMessage();
            }

            // If it looks like an auth error, offer to re-enter credentials
            if (isAuthError(errorMsg)) {
                // Clear stale saved credentials
                QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
                settings.remove(vc3d::settings::aws::ACCESS_KEY);
                settings.remove(vc3d::settings::aws::SECRET_KEY);
                settings.remove(vc3d::settings::aws::SESSION_TOKEN);

                auto reply = QMessageBox::warning(
                    _window,
                    QObject::tr("Authentication Error"),
                    QObject::tr("Failed to open remote volume:\n%1\n\n"
                                "Would you like to enter new AWS credentials and retry?").arg(errorMsg),
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
                    _openAct->setEnabled(false);
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
                    QObject::tr("Failed to open remote volume:\n%1").arg(errorMsg));
            }
        });

    auto future = QtConcurrent::run(
        [httpsUrl, cachePath, auth]() -> std::shared_ptr<Volume> {
            return Volume::NewFromUrl(httpsUrl, cachePath, auth);
        });
    watcher->setFuture(future);
}

// Result struct for background volume+segment loading
struct ScrollOpenResult {
    std::shared_ptr<Volume> volume;
    std::vector<std::pair<std::string, std::shared_ptr<Surface>>> surfaces;
    std::string errorMsg;
};

void MenuActionController::promptAndLoadRemoteSegments(
    const vc::cache::HttpAuth& auth,
    const std::string& cachePath)
{
    bool ok = false;
    QString segUrl = QInputDialog::getText(
        _window,
        QObject::tr("Remote Segments"),
        QObject::tr("Enter S3/HTTPS URL of directory containing segments\n"
                     "(leave empty to skip):"),
        QLineEdit::Normal, QString(), &ok);

    if (!ok || segUrl.trimmed().isEmpty())
        return;

    QString zarrUrl;
    if (_window && _window->_state) {
        if (auto vol = _window->_state->currentVolume())
            zarrUrl = QString::fromStdString(vol->remoteUrl());
    }
    loadRemoteSegmentsWithUrl(segUrl, auth, cachePath, zarrUrl);
}

void MenuActionController::loadRemoteSegmentsWithUrl(
    const QString& segUrlIn,
    const vc::cache::HttpAuth& auth,
    const std::string& cachePath,
    const QString& zarrUrl)
{
    const QString segUrlTrim = segUrlIn.trimmed();
    if (segUrlTrim.isEmpty())
        return;

    auto segResolved = vc::resolveRemoteUrl(segUrlTrim.toStdString());
    vc::cache::HttpAuth segAuth = auth;
    if (segResolved.useAwsSigv4 && segAuth.region.empty())
        segAuth.region = segResolved.awsRegion;

    std::string segBaseUrl = segResolved.httpsUrl;
    while (!segBaseUrl.empty() && segBaseUrl.back() == '/')
        segBaseUrl.pop_back();

    if (_window->statusBar())
        _window->statusBar()->showMessage(QObject::tr("Discovering remote segments..."));

    // Probe the URL for segment subdirectories
    auto* s3Watcher = new QFutureWatcher<vc::cache::S3ListResult>(this);
    connect(s3Watcher, &QFutureWatcher<vc::cache::S3ListResult>::finished, this,
        [this, s3Watcher, segBaseUrl, segAuth, cachePath, zarrUrl]() {
            s3Watcher->deleteLater();

            vc::cache::S3ListResult extList;
            try {
                extList = s3Watcher->result();
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[RemoteSegments] s3ListObjects failed: %s\n", e.what());
                if (_window->statusBar())
                    _window->statusBar()->showMessage(
                        QObject::tr("Failed to list segments: %1").arg(e.what()), 5000);
                return;
            }

            if (extList.prefixes.empty()) {
                if (_window->statusBar())
                    _window->statusBar()->showMessage(
                        QObject::tr("No segment directories found at that URL"), 5000);
                return;
            }

            std::fprintf(stderr, "[RemoteSegments] Found %zu segments\n", extList.prefixes.size());

            for (const auto& segId : extList.prefixes) {
                _window->_state->registerRemoteSegment(
                    segId,
                    {segBaseUrl, cachePath, segAuth, vc::RemoteSegmentSource::Direct});
            }

            // Persist (zarrUrl → segmentsUrl) so the next auto-open of the
            // same remote zarr can re-attach without prompting the user.
            if (!zarrUrl.isEmpty()) {
                QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
                settings.setValue(vc3d::settings::viewer::LAST_REMOTE_SEGMENTS_URL,
                                  QString::fromStdString(segBaseUrl));
                settings.setValue(vc3d::settings::viewer::LAST_REMOTE_SEGMENTS_FOR_ZARR,
                                  zarrUrl);
            }

            // Download metadata + load cached surfaces on background thread
            auto segIds = extList.prefixes;
            auto* loadWatcher = new QFutureWatcher<ScrollOpenResult>(this);
            connect(loadWatcher, &QFutureWatcher<ScrollOpenResult>::finished, this,
                [this, loadWatcher, segIds]() {
                    loadWatcher->deleteLater();

                    ScrollOpenResult result;
                    try {
                        result = loadWatcher->result();
                    } catch (const std::exception& e) {
                        if (_window->statusBar())
                            _window->statusBar()->showMessage(
                                QObject::tr("Failed to load segments: %1").arg(e.what()), 5000);
                        return;
                    }

                    _window->setRemoteStubs(segIds, result.surfaces);
                    _window->UpdateView();

                    int cached = static_cast<int>(result.surfaces.size());
                    int total = static_cast<int>(segIds.size());
                    if (_window->statusBar())
                        _window->statusBar()->showMessage(
                            QObject::tr("Loaded %1/%2 segments (rest on-demand)")
                                .arg(cached).arg(total), 5000);
                });

            auto loadFuture = QtConcurrent::run(
                [segBaseUrl, segIds, cachePath, segAuth]() -> ScrollOpenResult {
                    ScrollOpenResult result;
                    std::filesystem::path cacheDir = cachePath;
                    for (const auto& segId : segIds) {
                        try {
                            vc::downloadRemoteSegmentMetadataOnly(
                                segBaseUrl, segId, cacheDir, segAuth,
                                vc::RemoteSegmentSource::Direct);

                            if (vc::isRemoteSegmentFullyCached(
                                    cacheDir, segId, vc::RemoteSegmentSource::Direct)) {
                                // "Direct" uses "paths" subdir
                                auto localDir = cacheDir / "paths" / segId;
                                auto seg = Segmentation::New(localDir);
                                if (seg && seg->canLoadSurface()) {
                                    auto surf = seg->loadSurface();
                                    if (surf)
                                        result.surfaces.emplace_back(segId, surf);
                                }
                            }
                        } catch (const std::exception& e) {
                            std::fprintf(stderr, "[RemoteSegments] Failed to process segment %s: %s\n",
                                         segId.c_str(), e.what());
                        }
                    }
                    return result;
                });
            loadWatcher->setFuture(loadFuture);
        });

    auto s3Future = QtConcurrent::run(
        [segBaseUrl, segAuth]() -> vc::cache::S3ListResult {
            return vc::cache::s3ListObjects(segBaseUrl + "/", segAuth);
        });
    s3Watcher->setFuture(s3Future);
}

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
                    _openAct->setEnabled(true);
                    if (_window->statusBar()) _window->statusBar()->clearMessage();
                    return;
                }

                QString secretKey = QInputDialog::getText(
                    _window, QObject::tr("AWS Credentials"),
                    QObject::tr("AWS_SECRET_ACCESS_KEY:"),
                    QLineEdit::Password, QString(), &credOk);
                if (!credOk || secretKey.trimmed().isEmpty()) {
                    _remoteScrollAuthRetries = 0;
                    _openAct->setEnabled(true);
                    if (_window->statusBar()) _window->statusBar()->clearMessage();
                    return;
                }

                QString sessionToken = QInputDialog::getText(
                    _window, QObject::tr("AWS Credentials"),
                    QObject::tr("AWS_SESSION_TOKEN (optional):"),
                    QLineEdit::Normal, QString(), &credOk);
                if (!credOk) {
                    _remoteScrollAuthRetries = 0;
                    _openAct->setEnabled(true);
                    if (_window->statusBar()) _window->statusBar()->clearMessage();
                    return;
                }

                freshAuth.access_key = accessKey.trimmed().toStdString();
                freshAuth.secret_key = secretKey.trimmed().toStdString();
                freshAuth.session_token = sessionToken.trimmed().toStdString();

                // Save the fresh credentials
                QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
                settings.setValue(vc3d::settings::aws::ACCESS_KEY,
                                  QString::fromStdString(freshAuth.access_key));
                settings.setValue(vc3d::settings::aws::SECRET_KEY,
                                  QString::fromStdString(freshAuth.secret_key));
                settings.setValue(vc3d::settings::aws::SESSION_TOKEN,
                                  QString::fromStdString(freshAuth.session_token));

                // Retry discovery with fresh credentials.
                // Use QTimer::singleShot to break the call stack and limit
                // recursive re-entry on repeated auth failures (Issue 31).
                if (_remoteScrollAuthRetries >= 3) {
                    if (_window->statusBar()) {
                        _window->statusBar()->showMessage(
                            QObject::tr("Authentication failed after 3 attempts"), 5000);
                    }
                    _remoteScrollAuthRetries = 0;
                    _openAct->setEnabled(true);
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
                    _openAct->setEnabled(true);
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

            // Phase 2: Open volume + fetch segment metadata on background
            // thread.  Only downloads meta.json for each segment (fast, tiny
            // files).  Segments whose TIFFs are already cached get loaded
            // immediately; the rest appear as stubs that download on demand
            // when the user selects them.
            if (_window->statusBar()) {
                _window->statusBar()->showMessage(
                    QObject::tr("Opening remote scroll (volume: %1, discovering %2 segments)...")
                        .arg(QString::fromStdString(volumeName))
                        .arg(scrollInfo.segmentIds.size()));
            }

            auto* loadWatcher = new QFutureWatcher<ScrollOpenResult>(this);

            connect(loadWatcher, &QFutureWatcher<ScrollOpenResult>::finished, this,
                [this, loadWatcher, scrollInfo, cachePath]() {
                    loadWatcher->deleteLater();
                    _openAct->setEnabled(true);

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

                    {
                        const auto& base = scrollInfo.segmentSource == vc::RemoteSegmentSource::Direct
                            ? scrollInfo.segmentsBaseUrl
                            : scrollInfo.baseUrl;
                        for (const auto& segId : scrollInfo.segmentIds) {
                            _window->_state->registerRemoteSegment(
                                segId,
                                {base, cachePath, scrollInfo.auth, scrollInfo.segmentSource});
                        }
                    }

                    _window->setRemoteStubs(scrollInfo.segmentIds, result.surfaces);

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

                    int cachedCount = static_cast<int>(result.surfaces.size());
                    int totalCount = static_cast<int>(scrollInfo.segmentIds.size());
                    if (_window->statusBar()) {
                        _window->statusBar()->showMessage(
                            QObject::tr("Opened remote scroll: %1 (%2/%3 segments cached, rest on-demand)")
                                .arg(QString::fromStdString(result.volume->id()))
                                .arg(cachedCount)
                                .arg(totalCount),
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

                        // For Direct sources the on-demand download in CWindow
                        // uses flat cachePath (no volpkgName nesting). Match that
                        // layout here so preloaded segments are found on-demand.
                        const std::filesystem::path segCache =
                            (segSource == vc::RemoteSegmentSource::Direct)
                                ? std::filesystem::path(cachePath) : volpkgCache;

                        // Lazy loading: only download meta.json for each segment,
                        // and fully load only those whose TIFFs are already cached.
                        for (const auto& segId : segIds) {
                            try {
                                // Download metadata only (fast)
                                vc::downloadRemoteSegmentMetadataOnly(
                                    dlBase, segId, segCache, scrollAuth, segSource);

                                // If TIFFs are already cached, load the surface
                                if (vc::isRemoteSegmentFullyCached(segCache, segId, segSource)) {
                                    const char* subdir = (segSource == vc::RemoteSegmentSource::Segments)
                                        ? "segments" : "paths";
                                    auto localDir = segCache / subdir / segId;
                                    auto seg = Segmentation::New(localDir);
                                    if (seg && seg->canLoadSurface()) {
                                        auto surf = seg->loadSurface();
                                        if (surf) {
                                            result.surfaces.emplace_back(segId, surf);
                                        }
                                    }
                                }
                            } catch (const std::exception& e) {
                                std::fprintf(stderr, "[RemoteScroll] Failed to process segment %s: %s\n",
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
                    if (segResolved.useAwsSigv4 && segAuth.region.empty()) {
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

    auto pathsDirFs = _window->_state->activeSegmentsPath();
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
