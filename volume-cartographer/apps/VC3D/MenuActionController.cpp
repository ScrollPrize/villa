#include "MenuActionController.hpp"
#include <vc_reader/zarr.hpp>
#include "OpenedVolumes.hpp"
#include <vc_object_store/Uuid.hpp>

#include "VolumeAttachmentController.hpp"
#include "VCSettings.hpp"
#include "UnifiedBrowserDialog.hpp"
#include "OpenDataCatalogWindow.hpp"
#include "OpenDataLasagna.hpp"
#include "OpenDataNormalGrids.hpp"
#include "OpenDataSampleProject.hpp"
#include "ProjectCreationDefaults.hpp"
#include "OpenDataVolumePrefill.hpp"
#include "CWindow.hpp"
#include "SurfaceCatalog.hpp"
#include "SurfacePanelController.hpp"
#include "ViewerManager.hpp"
#include "WrapAnnotationWidget.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "CommandLineToolRunner.hpp"
#include "SettingsDialog.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "ui_VCMain.h"
#include "Keybinds.hpp"
#include "LineAnnotationController.hpp"
#include "LineAnnotationFiberNaming.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/Version.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc_object_store/objects/Hub.hpp"
#include "vc/core/util/LoadJson.hpp"
#include <vc_object_store/objects/UmbilicusFile.hpp>
#include "BindCoordinatesPrompt.hpp"
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include <vc_object_store/geometry/Geometry.hpp>
#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/CatalogStore.hpp>
#include <vc_object_store/objects/Project.hpp>
#include <vc_object_store/objects/GeometryIngest.hpp>
#include <vc_object_store/objects/VolumeUmbilicus.hpp>
#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/ProjectVolumes.hpp"

#include <QAction>
#include <QActionGroup>
#include <QApplication>
#include <QClipboard>
#include <QFutureWatcher>
#include <QInputDialog>
#include <QtConcurrent>
#include <QDateTime>
#include <QDesktopServices>
#include <QDialog>
#include <QFormLayout>
#include <QComboBox>
#include <QCheckBox>
#include <QDialogButtonBox>
#include <QDir>
#include <QEventLoop>
#include <QFileDialog>
#include <QFileInfo>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMetaObject>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPointer>
#include <QProgressDialog>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QStringList>
#include <QSettings>
#include <QStyle>
#include <QTreeWidget>
#include <QTimer>
#include <QTreeWidgetItem>
#include <QUrl>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <filesystem>
#include <memory>
#include <unordered_map>
#include <vector>

namespace
{
constexpr int kMaxStoredRemoteUrls = 10;
} // namespace

// Kept local so the QtConcurrent result type does not leak into the header.
struct MenuActionController::OpenDataOpenTaskResult {
    std::shared_ptr<VolumePkg> pkg;
    vc3d::opendata::OpenDataSampleProjectResult result;
    QString error;
    QString sampleId;
    bool freshProject{false};
    bool reopened{false};
};

struct MenuActionController::LasagnaAttachTaskResult {
    std::vector<VolumePkg::PreparedVolumeAttachment> volumes;
    QString error;
};

MenuActionController::MenuActionController(CWindow* window)
    : QObject(window)
    , _window(window)
{
    _recentActs.fill(nullptr);
}

MenuActionController::~MenuActionController()
{
    cancelOpenDataVolumePrefills();
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


    _attachNormalGridAct = new QAction(QObject::tr("Attach &Normal Grid..."), this);
    connect(_attachNormalGridAct, &QAction::triggered, this, &MenuActionController::attachNormalGrid);

    _importUmbilicusAct = new QAction(QObject::tr("Import &Umbilicus..."), this);
    connect(_importUmbilicusAct, &QAction::triggered, this, &MenuActionController::importUmbilicus);

    _removeUmbilicusAct = new QAction(QObject::tr("Remove Um&bilicus"), this);
    connect(_removeUmbilicusAct, &QAction::triggered, this, &MenuActionController::removeUmbilicus);

    _attachLasagnaManifestAct = new QAction(QObject::tr("Attach Lasagna Manifest..."), this);
    connect(_attachLasagnaManifestAct, &QAction::triggered, this, &MenuActionController::attachLasagnaManifest);

    _attachRemoteLasagnaManifestAct = new QAction(QObject::tr("Attach Remote Lasagna Manifest..."), this);
    connect(_attachRemoteLasagnaManifestAct, &QAction::triggered, this, &MenuActionController::attachRemoteLasagnaManifest);

    _detachEntryAct = new QAction(QObject::tr("&Detach..."), this);
    connect(_detachEntryAct, &QAction::triggered, this, &MenuActionController::detachEntry);

    _openAct = new QAction(qWindow->style()->standardIcon(QStyle::SP_DialogOpenButton), QObject::tr("&Open Project..."), this);
    _openAct->setShortcut(vc3d::keybinds::sequenceFor(vc3d::keybinds::shortcuts::OpenVolpkg));
    connect(_openAct, &QAction::triggered, this, &MenuActionController::openVolpkg);

    _attachRemoteZarrAct = new QAction(QObject::tr("Attach Remote &Zarr..."), this);
    connect(_attachRemoteZarrAct, &QAction::triggered, this, &MenuActionController::attachRemoteZarr);

    _openDataCatalogAct = new QAction(QObject::tr("Open Data Catalog..."), this);
    connect(_openDataCatalogAct, &QAction::triggered, this, &MenuActionController::showOpenDataCatalog);

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

    _mirrorCursorAct = new QAction(QObject::tr("Sync cursor across views"), this);
    _mirrorCursorAct->setCheckable(true);
    if (qWindow) {
        _mirrorCursorAct->setChecked(qWindow->segmentationCursorMirroringEnabled());
    }
    connect(_mirrorCursorAct, &QAction::toggled, this, &MenuActionController::toggleCursorMirroring);

    _surfaceFromSelectionAct = new QAction(QObject::tr("Surface from Selection"), this);
    connect(_surfaceFromSelectionAct, &QAction::triggered, this, &MenuActionController::surfaceFromSelection);

    _selectionClearAct = new QAction(QObject::tr("Clear"), this);
    connect(_selectionClearAct, &QAction::triggered, this, &MenuActionController::clearSelection);

    _rotateSurfaceAct = new QAction(QObject::tr("Rotate"), this);
    connect(_rotateSurfaceAct, &QAction::triggered, this, &MenuActionController::beginRotateSurfaceTransform);



    _growTrackPatchesAct = new QAction(QObject::tr("Grow Track Patches..."), this);
    connect(_growTrackPatchesAct, &QAction::triggered,
            this, &MenuActionController::growTrackPatchesFromMenuRequested);


    // Build menus
    _fileMenu = new QMenu(QObject::tr("&File"), qWindow);
    _fileMenu->addAction(_newProjectAct);
    _fileMenu->addAction(_openAct);
    _fileMenu->addAction(QObject::tr("Clone Hosted Project..."), this, [this]() { cloneHostedStore(true); });
    _fileMenu->addAction(_saveProjectAsAct);
    _fileMenu->addSeparator();
    _fileMenu->addAction(_attachVolumeAct);
    _storesMenu = new QMenu(QObject::tr("&Stores"), _fileMenu);
    _fileMenu->addMenu(_storesMenu);
    connect(_storesMenu, &QMenu::aboutToShow, this, &MenuActionController::refreshStoresMenu);
    refreshStoresMenu();
    connect(qWindow->_state, &CState::objectStoreObservationFailed, this,
        [qWindow](const QString& error) {
            qWarning().noquote() << error;
            qWindow->showStatusBarMessage(error, 15000);
        });
    _fileMenu->addAction(_attachNormalGridAct);
    _fileMenu->addAction(_importUmbilicusAct);
    _fileMenu->addAction(_removeUmbilicusAct);
    _fileMenu->addAction(_attachLasagnaManifestAct);
    _fileMenu->addAction(_attachRemoteLasagnaManifestAct);
    _fileMenu->addAction(_detachEntryAct);
    _fileMenu->addSeparator();
    _fileMenu->addAction(_attachRemoteZarrAct);
    _fileMenu->addAction(_openDataCatalogAct);

    _recentMenu = new QMenu(QObject::tr("Open &recent project"), _fileMenu);
    _recentMenu->setEnabled(false);
    _fileMenu->addMenu(_recentMenu);

    ensureRecentActions();

    _fileMenu->addSeparator();
    _fileMenu->addAction(_settingsAct);
    _fileMenu->addSeparator();
    _fileMenu->addAction(_exitAct);

    _editMenu = new QMenu(QObject::tr("&Edit"), qWindow);

    _viewMenu = new QMenu(QObject::tr("&View"), qWindow);
    qWindow->populateDockToggleMenu(_viewMenu);
    _viewMenu->addAction(_mirrorCursorAct);
    _viewMenu->addSeparator();
    _viewMenu->addAction(_resetViewsAct);
    _viewMenu->addSeparator();
    _viewMenu->addAction(_showConsoleAct);

    _actionsMenu = new QMenu(QObject::tr("&Actions"), qWindow);
    _actionsMenu->addAction(_drawBBoxAct);
    _actionsMenu->addSeparator();
    _actionsMenu->addSeparator();
    _actionsMenu->addAction(_growTrackPatchesAct);
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
    auto paths = settings.value(vc3d::settings::project::RECENT).toStringList();
    paths.removeIf([](const QString& path) { return !path.endsWith(".vcproj", Qt::CaseInsensitive); });
    return paths;
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

void MenuActionController::openVolpkg()
{
    if (!_window) {
        return;
    }
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const auto file = QFileDialog::getOpenFileName(_window, QObject::tr("Open Project"),
        settings.value(vc3d::settings::project::DEFAULT_PATH).toString(),
        QObject::tr("Project (*.vcproj);;All files (*)"));
    if (file.isEmpty()) return;
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
            _window->OpenVolume(path);
            _window->UpdateView();
        }
    }
}

bool MenuActionController::openVolpkgAt(const QString& path,
                                        bool interactive,
                                        QString* errorMessage)
{
    if (!_window) {
        if (errorMessage) {
            *errorMessage = QObject::tr("Main window is not available.");
        }
        return false;
    }

    return _window->openVolumePackage(path, interactive, errorMessage);
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
    const QStringList recentUrls = loadRecentRemoteUrls();
    showAttachRemoteZarrDialog(recentUrls.isEmpty() ? QString() : recentUrls.first());
}

void MenuActionController::showAttachRemoteZarrDialog(const QString& initialUrl)
{
    if (!_window) return;

    if (!_window->_state || !_window->_state->vpkg()) {
        QMessageBox::warning(_window,
                             QObject::tr("No Volume Package Loaded"),
                             QObject::tr("Open a volpkg before attaching a remote zarr."));
        return;
    }

    const auto url = promptLocation(
        QObject::tr("Attach Remote Zarr"),
        QObject::tr("Select a remote OME-Zarr root (http://, https://, s3://, sftp://)."),
        initialUrl.isEmpty() ? QStringLiteral("s3://") : initialUrl,
        {}, false, true);

    if (url.trimmed().isEmpty()) {
        return;
    }
    if (!vcr::zarr::isRemote(url.toStdString())) {
        QMessageBox::warning(_window, QObject::tr("Attach failed"),
            QObject::tr("Select a remote volume, or use Attach Volume for a local one."));
        return;
    }

    attachRemoteZarrUrl(url.trimmed());
}

void MenuActionController::showOpenDataCatalog()
{
    if (!_window) {
        return;
    }

    if (_openDataCatalogDialog) {
        _openDataCatalogDialog->show();
        _openDataCatalogDialog->raise();
        _openDataCatalogDialog->activateWindow();
        emit openDataCatalogVisibilityChanged(true);
        return;
    }

    auto* dialog = new vc3d::opendata::OpenDataCatalogWindow(_window);
    _openDataCatalogDialog = dialog;
    connect(dialog, &vc3d::opendata::OpenDataCatalogWindow::attachVolumeRequested,
            this, &MenuActionController::showAttachRemoteZarrDialog);
    dialog->setAttachRepresentationHandler(
        [this](const vcr::catalog::OpenDataSample& sample, const vcr::catalog::OpenDataRepresentationRef& ref) {
            attachCatalogRepresentation(sample, ref);
        });
    dialog->setOpenSampleHandler([this](const vcr::catalog::OpenDataSample& sample) {
        return openOpenDataSample(sample);
    });
    dialog->setCreateProjectHandler([this](const auto& sample, const auto& selection,
                                           const auto& project) {
        return openOpenDataSample(sample, &selection, &project);
    });
    dialog->setRefreshSampleHandler([this](const vcr::catalog::OpenDataSample& sample) -> QString {
        if (!_window || !_window->_state) return tr("No viewer state.");
        QApplication::setOverrideCursor(Qt::WaitCursor);
        // Volume metadata is kept as immutable; a refresh is how a sample
        // republished in place is picked up on its next open.
        vcr::zarr::OpenOptions options;
        options.cacheDirs = {vc::settings::remoteCachePath().string()};
        for (const auto& volume : sample.volumes)
            for (const auto& artifact : volume.artifacts)
                vcr::zarr::forgetKeptMetadata(artifact.url(), options);
        try {
            auto& state = *_window->_state;
            const auto project = state.sampleStore();
            if (!project || vc3d::opendata::openDataProjectSample(*state.vpkg()) != sample.id) {
                QApplication::restoreOverrideCursor();
                return tr("Open the project of %1 to refresh it; its volumes are refreshed when it is opened.")
                    .arg(QString::fromStdString(sample.id));
            }
            const auto report = vc::objects::refreshCatalog(*state.objectStoreSession(), *project, sample);
            QApplication::restoreOverrideCursor();
            return tr("%1: %2 surfaces, %3 new, %4 updated, %5 refetched, %6 locally edited surfaces preserved.")
                .arg(QString::fromStdString(sample.id)).arg(report.surfaces).arg(report.added)
                .arg(report.updated).arg(report.stale).arg(report.preserved);
        } catch (const std::exception& e) {
            QApplication::restoreOverrideCursor();
            return tr("Refreshing %1 failed: %2").arg(QString::fromStdString(sample.id), QString::fromUtf8(e.what()));
        }
    });
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    connect(dialog, &QDialog::finished, this, [this]() {
        emit openDataCatalogVisibilityChanged(false);
    });
    connect(dialog, &QDialog::finished, dialog, &QObject::deleteLater);
    connect(dialog, &QObject::destroyed, this, [this]() {
        _openDataCatalogDialog = nullptr;
        emit openDataCatalogVisibilityChanged(false);
    });
    dialog->show();
    dialog->raise();
    dialog->activateWindow();
    emit openDataCatalogVisibilityChanged(true);
}

void MenuActionController::resetOpenDataProject()
{
    auto original = _window->_state->vpkg();
    if (!original || !original->writable() || _openDataSampleOpenInFlight) return;
    const auto sampleId = vc3d::opendata::openDataProjectSample(*original);
    const auto catalog = vc3d::opendata::cachedCatalog();
    const auto* sample = sampleId && catalog ? catalog->findSample(*sampleId) : nullptr;
    if (!sample) {
        QMessageBox::warning(_window, tr("Reset to Open Data"), tr("The catalog no longer lists this sample."));
        return;
    }
    if (QMessageBox::question(_window, tr("Reset to Open Data"),
        tr("Restore the sample's resources and select a fresh store? Existing stores and their objects will be kept.")) != QMessageBox::Yes) return;
    if (_window->_state->prepareStoreSwitch && !_window->_state->prepareStoreSwitch()) return;
    struct ResetResult { std::shared_ptr<VolumePkg> package; QString error; };
    const auto staging = original->directory() / (original->path().stem().string() + "-reset-" + vc::objects::util::newUuid() + ".vcproj");
    const auto name = original->name();
    const auto options = CState::storeOpenOptions(CState::storeAuthor(), false);
    const auto cache = vc3d::remoteCachePathFs();
    auto* progress = new QProgressDialog(tr("Preparing a fresh sample store..."), {}, 0, 0, _window);
    progress->setWindowModality(Qt::WindowModal);
    progress->setCancelButton(nullptr);
    progress->show();
    auto* watcher = new QFutureWatcher<ResetResult>(this);
    _openDataSampleOpenInFlight = true;
    connect(watcher, &QFutureWatcher<ResetResult>::finished, this, [this, watcher, progress, original, staging]() {
        auto result = watcher->result();
        watcher->deleteLater();
        progress->close();
        progress->deleteLater();
        _openDataSampleOpenInFlight = false;
        try {
            if (!result.error.isEmpty()) throw std::runtime_error(result.error.toStdString());
            if (_window->_state->vpkg() != original) throw std::runtime_error("The active project changed during reset");
            if (_window->_state->prepareStoreSwitch && !_window->_state->prepareStoreSwitch()) throw std::runtime_error("Pending edits prevented reset");
            _window->_state->savePointCollections();
            auto data = result.package->document().data();
            for (const auto& store : original->document().data().at("stores")) data["stores"].push_back(store);
            original->document().replace(std::move(data));
            auto package = VolumePkg::open(original->path(), {.catalog = vc3d::opendata::cachedCatalog()});
            _window->CloseVolume();
            _window->_state->setVpkg(std::move(package));
            _window->refreshCurrentVolumePackageUi();
        } catch (const std::exception& error) {
            QMessageBox::warning(_window, tr("Reset failed"), QString::fromUtf8(error.what()));
        }
        std::error_code ignored;
        std::filesystem::remove(staging, ignored);
    });
    watcher->setFuture(QtConcurrent::run([staging, name, sample = *sample, options, cache]() {
        ResetResult result;
        try {
            vc3d::opendata::createSampleProjectFiles(staging, sample, name, options);
            const auto document = vc::project::ProjectDocument::load(staging);
            auto session = std::make_shared<vc::objects::ObjectStoreSession>();
            const auto store = session->attach(document.resolve(document.data().at("stores")[0].at("location")), options);
            result.package = vc3d::opendata::createOpenDataSampleProject(session, staging, store, sample, cache);
        } catch (const std::exception& error) { result.error = QString::fromUtf8(error.what()); }
        return result;
    }));
}

bool MenuActionController::isOpenDataCatalogVisible() const
{
    return _openDataCatalogDialog && _openDataCatalogDialog->isVisible();
}

bool MenuActionController::openOpenDataSample(const vcr::catalog::OpenDataSample& sample,
                                             const vc3d::opendata::OpenDataResourceSelection* selection,
                                             const vc3d::opendata::OpenDataNewProject* newProject)
{
    if (!_window || !_window->_state) {
        return false;
    }

    if (newProject && _window->_state->vpkg()) {
        const auto current = QString::fromStdString(_window->_state->vpkg()->path().string());
        const auto destination = QString::fromStdString(newProject->path.string());
        if (!current.isEmpty() && QFileInfo(current).canonicalFilePath() ==
                                  QFileInfo(destination).canonicalFilePath() &&
            QFileInfo::exists(destination)) {
            QMessageBox::warning(_window, QObject::tr("Create Project"),
                QObject::tr("Choose a different file from the currently open project."));
            return false;
        }
    }

    // Create and Open already explicitly confirms the project switch.
    if (!newProject && _window->_state->vpkg()) {
        QMessageBox prompt(_window);
        prompt.setWindowTitle(QObject::tr("Open Project"));
        prompt.setText(QObject::tr("Open sample %1?").arg(QString::fromStdString(sample.id)));
        prompt.setInformativeText(
            QObject::tr("This will close the current project and open the selected project. "
                        "The current project file will not be overwritten."));
        auto* openButton = prompt.addButton(QObject::tr("Open Project"), QMessageBox::AcceptRole);
        prompt.addButton(QMessageBox::Cancel);
        prompt.setDefaultButton(openButton);
        prompt.exec();

        if (prompt.clickedButton() != openButton) {
            return false;
        }
    }

    // Do not interleave CloseVolume/setVpkg from overlapping opens. Only this
    // interactive wrapper waits through a nested event loop.
    if (_openDataSampleOpenInFlight) {
        return false;
    }

    OpenDataSampleOpenOutcome outcome;
    QEventLoop loop;
    beginOpenDataSampleOpenTask(
        sample, /*interactive=*/true, selection,
        [&outcome, &loop](const OpenDataSampleOpenOutcome& o) {
            outcome = o;
            loop.quit();
        },
        {}, newProject);
    loop.exec(QEventLoop::ExcludeUserInputEvents);
    return outcome.success;
}

bool MenuActionController::startOpenDataSampleOpen(
    const vcr::catalog::OpenDataSample& sample,
    const vc3d::opendata::OpenDataResourceSelection& selection,
    std::function<void(const OpenDataSampleOpenOutcome&)> onFinished,
    std::function<void(const vc3d::opendata::OpenDataSampleDownloadProgress&)> onProgress,
    QString* errorMessage)
{
    if (!_window || !_window->_state) {
        if (errorMessage) *errorMessage = QObject::tr("No application window available.");
        return false;
    }
    if (_openDataSampleOpenInFlight) {
        if (errorMessage)
            *errorMessage = QObject::tr("An Open Data sample open is already in progress.");
        return false;
    }

    // This entry point is always non-interactive. Completion is delivered
    // through onFinished on the GUI thread.
    beginOpenDataSampleOpenTask(sample, /*interactive=*/false, &selection,
                                std::move(onFinished), std::move(onProgress));
    return true;
}

bool MenuActionController::openDataSampleOpenInFlight() const
{
    return _openDataSampleOpenInFlight;
}

void MenuActionController::beginOpenDataSampleOpenTask(
    const vcr::catalog::OpenDataSample& sample,
    bool interactive,
    const vc3d::opendata::OpenDataResourceSelection* selection,
    std::function<void(const OpenDataSampleOpenOutcome&)> onFinished,
    std::function<void(const vc3d::opendata::OpenDataSampleDownloadProgress&)> onProgress,
    const vc3d::opendata::OpenDataNewProject* newProject)
{
    const QString cacheDir = vc3d::remoteCachePath();
    const vcr::catalog::OpenDataSample sampleCopy = sample;
    const auto projectCopy = newProject
        ? std::make_optional(*newProject) : std::nullopt;
    // Copy the selection by value so the worker thread never dereferences a
    // caller-owned pointer; a null selection means attach everything.
    const bool hasSelection = selection != nullptr;
    const vc3d::opendata::OpenDataResourceSelection selectionCopy =
        selection ? *selection : vc3d::opendata::OpenDataResourceSelection{};
    // Prepare edits now and check again before switching. Keep the old project
    // open while the sample is prepared, including when preparation fails.
    if (_window && _window->_state->prepareStoreSwitch && !_window->_state->prepareStoreSwitch()) {
        if (onFinished) QMetaObject::invokeMethod(this, [onFinished = std::move(onFinished)]() {
            OpenDataSampleOpenOutcome outcome;
            outcome.success = false;
            outcome.error = QObject::tr("Pending edits prevented the project switch.");
            onFinished(outcome);
        }, Qt::QueuedConnection);
        return;
    }

    // Create the project and its sample store, then populate them in the worker.
    const auto author = interactive ? objectStoreAuthor() : CState::storeAuthor();
    const auto projectPath = newProject ? newProject->path
                                        : vc3d::opendata::sampleProjectPath(vc3d::projectsDirectory().toStdString(), sample);
    const auto projectName = newProject ? newProject->name : (sample.id.empty() ? std::string("Open Data Sample") : sample.id);
    const bool populateProject = !std::filesystem::exists(projectPath);
    QString setupError;
    if (author.isEmpty()) {
        setupError = QObject::tr("A username is required to open the sample.");
    } else {
        try {
            if (newProject && std::filesystem::exists(projectPath))
                throw std::runtime_error(projectPath.string() + " already exists.");
            if (!std::filesystem::exists(projectPath))
                vc3d::opendata::createSampleProjectFiles(projectPath, sample, projectName,
                                                         CState::storeOpenOptions(author, false));

        } catch (const std::exception& error) {
            setupError = QString::fromUtf8(error.what());
        }
    }
    if (!setupError.isEmpty()) {
        if (onFinished) {
            QMetaObject::invokeMethod(this, [onFinished = std::move(onFinished), setupError]() {
                OpenDataSampleOpenOutcome outcome;
                outcome.success = false;
                outcome.error = setupError;
                onFinished(outcome);
            }, Qt::QueuedConnection);
        }
        return;
    }

    QPointer<QProgressDialog> progressDialog;
    if (interactive) {
        auto* dialog = new QProgressDialog(
            QObject::tr("Preparing catalog resources..."),
            QString(),
            0,
            0,
            _window);
        dialog->setWindowTitle(QObject::tr("Open Data Sample"));
        dialog->setCancelButton(nullptr);
        dialog->setWindowModality(Qt::WindowModal);
        dialog->setMinimumDuration(0);
        dialog->setAutoClose(false);
        dialog->setAutoReset(false);
        dialog->show();
        progressDialog = dialog;
    }

    // Forward one progress stream to the optional dialog and callback. Both
    // updates are marshalled to the GUI thread and guarded for object lifetime.
    QPointer<MenuActionController> self(this);
    auto progressCallback =
        [progressDialog, self, onProgress](
            const vc3d::opendata::OpenDataSampleDownloadProgress& progress) {
            if (onProgress) {
                QMetaObject::invokeMethod(
                    qApp,
                    [self, onProgress, progress]() {
                        if (self && onProgress) {
                            onProgress(progress);
                        }
                    },
                    Qt::QueuedConnection);
            }
            if (!progressDialog) {
                return;
            }
            QMetaObject::invokeMethod(
                progressDialog.data(),
                [progressDialog, progress]() {
                    if (!progressDialog) {
                        return;
                    }
                    const QString status = QString::fromStdString(progress.status);
                    const bool resolvingVolumes =
                        status == QStringLiteral("resolving-volumes");
                    const bool projectReady =
                        status == QStringLiteral("project-ready");
                    const QString label = resolvingVolumes
                        ? QObject::tr("Opening remote volumes in parallel...")
                        : projectReady
                        ? QObject::tr("Open-data project is ready.")
                        : QObject::tr("Preparing catalog resources...");
                    if (projectReady) {
                        progressDialog->setRange(0, 1);
                        progressDialog->setValue(1);
                    } else {
                        progressDialog->setRange(0, 0);
                    }
                    progressDialog->setLabelText(label);
                },
                Qt::QueuedConnection);
        };

    // The parented watcher completes on the GUI thread without a nested event
    // loop. QtConcurrent cannot interrupt this task; destruction simply
    // disconnects and discards a late result.
    auto* watcher = new QFutureWatcher<OpenDataOpenTaskResult>(this);
    connect(watcher,
            &QFutureWatcher<OpenDataOpenTaskResult>::finished,
            this,
            [this, watcher, progressDialog, interactive,
             onFinished = std::move(onFinished)]() mutable {
                OpenDataOpenTaskResult task = watcher->result();
                if (progressDialog) {
                    progressDialog->close();
                    progressDialog->deleteLater();
                }
                _openDataSampleOpenInFlight = false;
                watcher->deleteLater();

                OpenDataSampleOpenOutcome outcome;
                finishOpenDataSampleOpen(std::move(task), interactive, &outcome);
                if (onFinished) {
                    onFinished(outcome);
                }
            });

    _openDataSampleOpenInFlight = true;
    watcher->setFuture(QtConcurrent::run(
        [sampleCopy, cacheDir, progressCallback, hasSelection, selectionCopy, projectCopy, projectPath, populateProject,
         options = CState::storeOpenOptions(author, false)]() mutable {
            OpenDataOpenTaskResult taskResult;
            taskResult.freshProject = projectCopy.has_value();
            taskResult.reopened = !populateProject;
            taskResult.sampleId = QString::fromStdString(sampleCopy.id);
            try {
                if (populateProject) {
                    // Preparation must not publish attachment events to the current workspace.
                    auto session = std::make_shared<vc::objects::ObjectStoreSession>();
                    const auto document = vc::project::ProjectDocument::load(projectPath);
                    const auto sampleStoreId = document.data().at("open_data").at("store");
                    const auto& stores = document.data().at("stores");
                    const auto reference = std::find_if(stores.begin(), stores.end(), [&](const auto& store) { return store.at("id") == sampleStoreId; });
                    if (reference == stores.end()) throw std::runtime_error("Sample store reference is missing");
                    const auto attachment = session->attach(document.resolve(reference->at("location")), options);
                    taskResult.pkg = vc3d::opendata::createOpenDataSampleProject(
                        session, projectPath, attachment, sampleCopy, cacheDir.toStdString(),
                        &taskResult.result, progressCallback, hasSelection ? &selectionCopy : nullptr);
                } else {
                    taskResult.pkg = VolumePkg::open(projectPath, {.catalog = vc3d::opendata::cachedCatalog()});
                }
            } catch (const std::exception& e) {
                taskResult.error = QString::fromUtf8(e.what());
            } catch (...) {
                taskResult.error = QObject::tr("Unknown error while opening open-data sample.");
            }
            return taskResult;
        }));
}

void MenuActionController::finishOpenDataSampleOpen(OpenDataOpenTaskResult task,
                                                    bool interactive,
                                                    OpenDataSampleOpenOutcome* outcomeOut)
{
    const QString sampleId = task.sampleId;
    if (!task.error.isEmpty()) {
        if (interactive) {
            QMessageBox::warning(
                _window,
                QObject::tr("Open Data Sample"),
                QObject::tr("Failed to open sample %1:\n\n%2")
                    .arg(sampleId, task.error));
        }
        if (outcomeOut) {
            outcomeOut->success = false;
            outcomeOut->error = task.error;
        }
        return;
    }

    vc3d::opendata::OpenDataSampleProjectResult result = std::move(task.result);

    auto pkg = std::move(task.pkg);
    if (!pkg) {
        const QString msg =
            QObject::tr("Failed to create sample project for %1.").arg(sampleId);
        if (interactive) {
            QMessageBox::warning(
                _window, QObject::tr("Open Data Sample"), msg);
        }
        if (outcomeOut) {
            outcomeOut->success = false;
            outcomeOut->error = msg;
        }
        return;
    }

    // Same gate as CWindow::OpenVolume(): sessions end — finalized and saved —
    // while their own package is still the active one, or the switch does not
    // happen.
    if (_window && _window->_state->prepareStoreSwitch &&
        !_window->_state->prepareStoreSwitch()) {
        const QString msg = QObject::tr(
            "Open line annotation sessions could not be closed; the project "
            "was not switched.");
        if (outcomeOut) {
            outcomeOut->success = false;
            outcomeOut->error = msg;
        }
        return;
    }

    if (_window && _window->_state) {
        // The full close, not a bare setVpkg: another project may have been
        // opened while the sample fetch ran (nothing blocks OpenVolume()
        // during it), and replacing its package directly would skip the
        // segmentation-editing teardown CloseVolume() performs and leave its
        // active surface alive beside this sample's package. With nothing
        // open, CloseVolume() is a no-op.
        cancelOpenDataVolumePrefills();
        _window->CloseVolume();
        _window->_state->setVpkg(pkg);
    }
    if (!pkg->path().empty()) {
        updateRecentVolpkgList(QString::fromStdString(pkg->path().string()));
    }

    if (_window) {
        _window->refreshCurrentVolumePackageUi(
            QString::fromStdString(result.preferredVolumeId.empty() ? pkg->currentVolumeId() : result.preferredVolumeId));
        _window->UpdateView();
    }
    startOpenDataVolumePrefill(
        _window && _window->_state ? _window->_state->currentVolume() : nullptr);
    QString message = QObject::tr("Sample %1: attached %2 of %3 supported volume entries.")
                          .arg(sampleId)
                          .arg(result.attachedVolumeEntries)
                          .arg(result.supportedVolumes);
    if (task.reopened) message = tr("Opened sample %1.").arg(sampleId);
    if (_window && _window->statusBar()) {
        _window->showStatusBarMessage(message, 7000);
    }

    if (interactive &&
        ((!task.freshProject && !task.reopened && result.supportedVolumes == 0) ||
         result.failedVolumes > 0)) {
        QString details;
        for (const auto& item : result.messages) {
            if (!details.isEmpty()) {
                details += QLatin1Char('\n');
            }
            details += QString::fromStdString(item);
        }
        QMessageBox::information(
            _window,
            QObject::tr("Open Data Sample"),
            details.isEmpty() ? message : message + QObject::tr("\n\n%1").arg(details));
    }

    if (outcomeOut) {
        outcomeOut->success = true;
        outcomeOut->result = std::move(result);
    }
}

void MenuActionController::startOpenDataVolumePrefill(const std::shared_ptr<Volume>& volume)
{
    const int logicalPrefillLevel = volume
        ? vc3d::opendata::kOpenDataVolumePrefillLevel - volume->baseScaleLevel()
        : -1;
    if (!volume || !volume->isRemote() || logicalPrefillLevel < 0) {
        return;
    }
    try {
        if (!_window->_state->openedVolumes()->pyramid(*volume).has(logicalPrefillLevel))
            return;
    } catch (const std::exception&) {
        return;  // the volume's viewers report what cannot be opened
    }
    if (volume->remoteCacheRoot().empty()) {
        return;
    }

    auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
    _openDataPrefillCancelFlag = cancelFlag;
    auto* watcher = new QFutureWatcher<vc3d::opendata::OpenDataVolumePrefillResult>(this);
    _openDataPrefillWatchers.push_back(watcher);
    _openDataPrefillCancelFlags.push_back(cancelFlag);

    const QString volumeId = QString::fromStdString(volume->id());
    connect(watcher,
            &QFutureWatcher<vc3d::opendata::OpenDataVolumePrefillResult>::finished,
            this,
            [this, watcher, cancelFlag, volumeId]() {
                vc3d::opendata::OpenDataVolumePrefillResult result;
                // A canceled QFuture has no stored result. Check cancellation
                // before result() when a stale finished signal arrives.
                if (watcher->isCanceled()) {
                    result.status = vc3d::opendata::OpenDataVolumePrefillResult::Status::Cancelled;
                    result.volumeId = volumeId.toStdString();
                    result.message = "cancelled before completion";
                } else {
                    try {
                        result = watcher->result();
                    } catch (const std::exception& e) {
                        result.status = vc3d::opendata::OpenDataVolumePrefillResult::Status::Failed;
                        result.volumeId = volumeId.toStdString();
                        result.message = e.what();
                    } catch (...) {
                        result.status = vc3d::opendata::OpenDataVolumePrefillResult::Status::Failed;
                        result.volumeId = volumeId.toStdString();
                        result.message = "unknown error";
                    }
                }

                _openDataPrefillWatchers.erase(
                    std::remove(_openDataPrefillWatchers.begin(),
                                _openDataPrefillWatchers.end(),
                                watcher),
                    _openDataPrefillWatchers.end());
                _openDataPrefillCancelFlags.erase(
                    std::remove(_openDataPrefillCancelFlags.begin(),
                                _openDataPrefillCancelFlags.end(),
                                cancelFlag),
                    _openDataPrefillCancelFlags.end());
                if (_openDataPrefillCancelFlag == cancelFlag) {
                    _openDataPrefillCancelFlag.reset();
                }
                watcher->deleteLater();

                if (!_window) {
                    return;
                }

                using Status = vc3d::opendata::OpenDataVolumePrefillResult::Status;
                switch (result.status) {
                case Status::Completed:
                    if (result.fetchedChunks > 0) {
                        _window->showStatusBarMessage(
                            QObject::tr("Cached remote volume %1 physical /%2 (logical /%3): %4 chunks fetched.")
                                .arg(QString::fromStdString(result.volumeId))
                                .arg(result.physicalLevel)
                                .arg(result.level)
                                .arg(result.fetchedChunks),
                            7000);
                    }
                    Logger()->info(
                        "Open-data volume prefill completed for {} level {}: {} chunks (data={}, empty={}, fetched={})",
                        result.volumeId,
                        result.level,
                        result.totalChunks,
                        result.dataChunks,
                        result.emptyChunks,
                        result.fetchedChunks);
                    break;
                case Status::Failed:
                    _window->showStatusBarMessage(
                        QObject::tr("Remote volume %1 level /%2 cache failed: %3")
                            .arg(QString::fromStdString(result.volumeId))
                            .arg(result.level)
                            .arg(QString::fromStdString(result.message)),
                        9000);
                    Logger()->warn(
                        "Open-data volume prefill failed for {} level {}: {}",
                        result.volumeId,
                        result.level,
                        result.message);
                    break;
                case Status::Cancelled:
                    Logger()->info(
                        "Open-data volume prefill cancelled for {} level {} after {}/{} chunks",
                        result.volumeId,
                        result.level,
                        result.resolvedChunks,
                        result.totalChunks);
                    break;
                case Status::Skipped:
                    Logger()->info(
                        "Open-data volume prefill skipped for {} level {}: {}",
                        result.volumeId,
                        result.level,
                        result.message);
                    break;
                }
            });

    watcher->setFuture(QtConcurrent::run([volume, cancelFlag, logicalPrefillLevel]() {
        return vc3d::opendata::prefillOpenDataVolumeLevel(
            volume,
            logicalPrefillLevel,
            cancelFlag.get());
    }));
}

void MenuActionController::cancelOpenDataVolumePrefills()
{
    if (_openDataPrefillCancelFlag) {
        _openDataPrefillCancelFlag->store(true, std::memory_order_release);
    }
    for (const auto& cancelFlag : _openDataPrefillCancelFlags) {
        if (cancelFlag) {
            cancelFlag->store(true, std::memory_order_release);
        }
    }
    for (auto* watcher : _openDataPrefillWatchers) {
        if (watcher) {
            watcher->cancel();
        }
    }
}

void MenuActionController::attachRemoteZarrUrl(const QString& url)
{
    auto* attachment = _window
        ? _window->_volumeAttachmentController.get()
        : nullptr;
    if (!attachment) {
        QMessageBox::warning(
            _window,
            QObject::tr("Attach Remote Zarr"),
            QObject::tr("Volume attachment is unavailable."));
        return;
    }

    VolumeAttachmentRequest request;
    QString error;
    if (!attachment->prepare(
            url,
            {},
            &request,
            &error)) {
        if (!error.isEmpty()) {
            QMessageBox::warning(
                _window, QObject::tr("Attach Remote Zarr"), error);
        }
        return;
    }
    request.selection = VolumeAttachmentSelection::SelectAttached;
    const QString persistedLocator = request.location;
    if (_attachRemoteZarrAct) {
        _attachRemoteZarrAct->setEnabled(false);
    }
    if (_window->statusBar()) {
        _window->showStatusBarMessage(QObject::tr("Attaching remote zarr..."));
    }

    QPointer<MenuActionController> self(this);
    const bool started = attachment->start(
        std::move(request),
        [self, persistedLocator](const VolumeAttachmentOutcome& outcome) {
            if (!self)
                return;
            if (self->_attachRemoteZarrAct)
                self->_attachRemoteZarrAct->setEnabled(true);

            if (outcome.success) {
                self->updateRecentRemoteList(persistedLocator);
                if (self->_window && self->_window->statusBar()) {
                    self->_window->showStatusBarMessage(
                        outcome.alreadyAttached
                            ? QObject::tr("Remote zarr is already attached: %1")
                                  .arg(outcome.volumeId)
                            : QObject::tr("Attached remote zarr: %1")
                                  .arg(outcome.volumeId),
                        5000);
                }
                return;
            }

            QMessageBox::critical(
                self->_window,
                QObject::tr("Attach Remote Zarr Error"),
                QObject::tr("Failed to attach remote zarr:\n%1")
                    .arg(outcome.error));
        },
        &error);
    if (!started) {
        if (_attachRemoteZarrAct) {
            _attachRemoteZarrAct->setEnabled(true);
        }
        QMessageBox::warning(
            _window, QObject::tr("Attach Remote Zarr"), error);
    }
}

namespace
{

} // namespace

void MenuActionController::showSettingsDialog()
{
    if (!_window) {
        return;
    }

    auto* dialog = new SettingsDialog(_window);
    dialog->exec();

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const bool resetViewOnSurfaceChange =
        settings.value(vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE,
                       vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE_DEFAULT).toBool();
    const bool showPlaneLines =
        settings.value(vc3d::settings::viewer::SHOW_PLANE_INTERSECTION_LINES,
                       vc3d::settings::viewer::SHOW_PLANE_INTERSECTION_LINES_DEFAULT).toBool();
    if (_window->_viewerManager) {
        _window->_viewerManager->forEachBaseViewer([showPlaneLines](VolumeViewerBase* viewer) {
            if (viewer) {
                viewer->setPlaneIntersectionLinesVisible(showPlaneLines);
                // Re-read viewer settings so changes made in the dialog take effect immediately.
                viewer->reloadPerfSettings();
                viewer->renderVisible();
            }
        });
    }
    _window->onMoveOnSurfaceChangedToggled(resetViewOnSurfaceChange);

    if (_window->_viewerManager) {
        const int intersectionOpacity =
            settings.value(vc3d::settings::viewer::INTERSECTION_OPACITY,
                           vc3d::settings::viewer::INTERSECTION_OPACITY_DEFAULT).toInt();
        _window->_viewerManager->setIntersectionOpacity(
            std::clamp(static_cast<float>(intersectionOpacity) / 100.0f, 0.0f, 1.0f));
    }
    _window->onAxisOverlayOpacityChanged(
        settings.value(vc3d::settings::viewer::AXIS_OVERLAY_OPACITY,
                       vc3d::settings::viewer::AXIS_OVERLAY_OPACITY_DEFAULT).toInt());

    CState::applySurfaceGeometryCacheSettings();

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

    _window->resetSegmentationViews();
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
                _window->showStatusBarMessage(enabled ? QObject::tr("BBox mode active: drag on Surface view")
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
    if (!_window || !_window->_viewerManager || !_window->_surfacePanel || !_window->_surfaceCatalog) {
        return;
    }

    VolumeViewerBase* segViewer = _window->segmentationBaseViewer();

    if (!segViewer) {
        _window->showStatusBarMessage(QObject::tr("No Surface viewer found"), 3000);
        return;
    }

    auto sels = segViewer->selections();
    if (sels.empty()) {
        _window->showStatusBarMessage(QObject::tr("No selections to convert"), 3000);
        return;
    }

    const auto& opened = _window->_surfacePanel->opened();
    if (!opened || opened->surface->id != _window->_state->activeSurfaceId()) {
        _window->showStatusBarMessage(QObject::tr("Select a segmentation first"), 3000);
        return;
    }
    const auto sourceUuid = opened->surface->id;
    const auto* row = _window->_surfaceCatalog->row(sourceUuid);
    const std::string sourceName = row && !row->record.name.empty() ? row->record.name : sourceUuid;

    int idx = 1;
    int created = 0;
    QString ts = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    for (const auto& pr : sels) {
        const QRectF& rect = pr.first;
        std::unique_ptr<QuadSurface> filtered(segViewer->makeBBoxFilteredSurfaceFromSceneRect(rect));
        if (!filtered) {
            continue;
        }

        const std::string name = sourceName + "_sel_" + ts.toStdString() + "_" + std::to_string(idx++);
        const vc::objects::SurfaceProvenance provenance{
            "vc3d-surface-from-selection",
            {{"scene_rect", {rect.x(), rect.y(), rect.width(), rect.height()}}},
            {sourceUuid}};
        try {
            _window->_surfaceCatalog->create(*filtered, name, provenance);
            created++;
        } catch (const std::exception& e) {
            _window->showStatusBarMessage(QObject::tr("Failed to save selection: ") + e.what(), 5000);
        }
    }

    _window->showStatusBarMessage(created > 0
        ? QObject::tr("Created %1 surface(s) from selection").arg(created)
        : QObject::tr("No surfaces created from selection"), created > 0 ? 5000 : 3000);
}

void MenuActionController::clearSelection()
{
    if (!_window) {
        return;
    }

    VolumeViewerBase* segViewer = _window->segmentationBaseViewer();
    if (!segViewer) {
        _window->showStatusBarMessage(QObject::tr("No Surface viewer found"), 3000);
        return;
    }

    segViewer->clearSelections();
    _window->showStatusBarMessage(QObject::tr("Selections cleared"), 2000);
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
    // Rotation saves back to the store the surface was opened from.
    QString error;
    if (_window->_surfacePanel && !_window->_surfacePanel->prepareSurfaceEditing(activeSurface->id, &error)) {
        QMessageBox::information(_window, QObject::tr("Rotate Surface"), error);
        return;
    }

    _window->_surfaceRotationOverlay->beginRotate();
    if (_window->statusBar()) {
        _window->showStatusBarMessage(QObject::tr("Surface rotation active"), 3000);
    }
}

void MenuActionController::newProject()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    auto file = QFileDialog::getSaveFileName(_window, tr("New Project"),
        QDir(vc3d::defaultNewProjectDirectory(settings)).filePath("untitled.vcproj"), tr("Project (*.vcproj)"), nullptr, QFileDialog::DontConfirmOverwrite);
    if (file.isEmpty()) return;
    if (QFileInfo(file).suffix().isEmpty()) file += ".vcproj";
    if (QFileInfo::exists(file)) { QMessageBox::warning(_window, tr("New Project"), tr("Project already exists.")); return; }
    QMessageBox choice(QMessageBox::Question, tr("Project Store"), tr("Choose the first store for this project."), QMessageBox::Cancel, _window);
    auto* existing = choice.addButton(tr("Use Existing Store..."), QMessageBox::AcceptRole);
    auto* create = choice.addButton(tr("Create New Store..."), QMessageBox::AcceptRole);
    choice.exec();
    const bool makeStore = choice.clickedButton() == create;
    if (!makeStore && choice.clickedButton() != existing) return;
    auto suggested = std::filesystem::path(file.toStdString());
    suggested.replace_extension(".vcobj");
    QString store = makeStore
        ? QFileDialog::getSaveFileName(_window, tr("New Store"), QString::fromStdString(suggested.string()), tr("Object store (*.vcobj)"), nullptr, QFileDialog::DontConfirmOverwrite)
        : QFileDialog::getOpenFileName(_window, tr("Existing Store"), QFileInfo(file).absolutePath(), tr("Object store (*.vcobj)"));
    if (store.isEmpty()) return;
    if (QFileInfo(store).suffix().isEmpty()) store += ".vcobj";
    const auto author = objectStoreAuthor();
    if (author.isEmpty()) return;
    bool created = false;
    try {
        if (makeStore) { auto db = vc::objects::ObjectStore::create(store.toStdString(), CState::storeOpenOptions(author, false)); created = true; }
        else { auto db = vc::objects::ObjectStore::open(store.toStdString(), CState::storeOpenOptions(author, true)); }
        vc::project::ProjectDocument::create(file.toStdString(), QFileInfo(file).completeBaseName().toStdString(), store.toStdString(), !QFileInfo(store).isWritable());
    } catch (const std::exception& error) {
        if (created) { std::error_code ignored; std::filesystem::remove(store.toStdString(), ignored); }
        QMessageBox::warning(_window, tr("New Project failed"), QString::fromUtf8(error.what()));
        return;
    }
    settings.setValue(vc3d::settings::project::DEFAULT_PATH, QFileInfo(file).absolutePath());
    openVolpkgAt(file);
}

void MenuActionController::newStore()
{
    if (!requireProject(tr("Creating a store"))) return;
    auto file = QFileDialog::getSaveFileName(_window, tr("New Store"), _window->_state->projectDirectory(), tr("Object store (*.vcobj)"), nullptr, QFileDialog::DontConfirmOverwrite);
    if (file.isEmpty()) return;
    if (QFileInfo(file).suffix().isEmpty()) file += ".vcobj";
    bool created = false;
    try {
        { auto store = vc::objects::ObjectStore::create(file.toStdString(), CState::storeOpenOptions(CState::storeAuthor(), false)); created = true; }
        _window->_state->attachStore(file.toStdString(), CState::storeAuthor());
    } catch (const std::exception& error) {
        if (created) { std::error_code ignored; std::filesystem::remove(file.toStdString(), ignored); }
        QMessageBox::warning(_window, tr("New Store failed"), QString::fromUtf8(error.what()));
    }
}

void MenuActionController::saveProjectAs()
{
    auto& state = *_window->_state;
    if (!state.vpkg()) return;
    auto file = QFileDialog::getSaveFileName(_window, tr("Save Project As"), state.projectDirectory(), tr("Project (*.vcproj)"), nullptr, QFileDialog::DontConfirmOverwrite);
    if (file.isEmpty()) return;
    if (QFileInfo(file).suffix().isEmpty()) file += ".vcproj";
    try {
        if (state.prepareStoreSwitch && !state.prepareStoreSwitch()) return;
        state.savePointCollections();
        state.vpkg()->document().saveAs(file.toStdString());
        _window->openVolumePackage(file, true, nullptr, QString::fromStdString(state.currentVolumeId()));
    } catch (const std::exception& error) {
        QMessageBox::warning(_window, tr("Save failed"), QString::fromUtf8(error.what()));
    }
}

QString MenuActionController::promptLocation(const QString& title,
                                             const QString& hint,
                                             const QString& defaultDir,
                                             const QStringList& localFilters,
                                             bool acceptFiles,
                                             bool acceptDirs,
                                             bool startAtFile)
{
    UnifiedBrowserDialog dlg(_window);
    dlg.setWindowTitle(title);
    dlg.setHint(hint);
    dlg.setLocalNameFilters(localFilters);
    dlg.setAcceptsFiles(acceptFiles);
    dlg.setAcceptsDirs(acceptDirs);
    dlg.setStartUri(defaultDir, startAtFile);
    if (dlg.exec() != QDialog::Accepted) return {};
    QString uri = dlg.selectedUri();
    if (uri.startsWith(QLatin1String("file:"), Qt::CaseInsensitive)) {
        const QUrl localUrl(uri);
        if (localUrl.isLocalFile()) {
            QString localPath = QDir::fromNativeSeparators(localUrl.toLocalFile());
            while (localPath.size() > 1 && localPath.endsWith('/') &&
                   !QDir(localPath).isRoot()) {
                localPath.chop(1);
            }
            return localPath;
        }
    }
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
                              QObject::tr("Pick a zarr volume or a folder of zarrs (local or s3://, https://, sftp://)."),
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
        QObject::tr("Tags (comma-separated, optional):"),
        QLineEdit::Normal, QString(), &ok);
    if (!ok) return;
    std::vector<std::string> tags;
    for (const auto& t : tagsStr.split(',', Qt::SkipEmptyParts)) {
        const auto trimmed = t.trimmed().toStdString();
        if (!trimmed.empty()) tags.push_back(trimmed);
    }
    bool changed = false;
    try { changed = _window->_state->vpkg()->addVolumeEntry(loc.toStdString(), tags); }
    catch (const std::exception& error) {
        QMessageBox::warning(_window, tr("Project save failed"), QString::fromUtf8(error.what()));
        return;
    }
    if (!changed) {
        QMessageBox::warning(_window, QObject::tr("Attach failed"), QObject::tr("Could not add volume (already attached?)"));
        return;
    }
    _window->refreshCurrentVolumePackageUi();
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
    bool changed = false;
    try { changed = _window->_state->vpkg()->addNormalGridEntry(loc.toStdString()); }
    catch (const std::exception& error) {
        QMessageBox::warning(_window, tr("Project save failed"), QString::fromUtf8(error.what()));
        return;
    }
    if (!changed) {
        QMessageBox::warning(_window, QObject::tr("Attach failed"), QObject::tr("Could not add normal grid (already attached?)"));
        return;
    }
    _window->refreshCurrentVolumePackageUi();
}

namespace {
// The attached volume an umbilicus stamp names by store directory name, so the
// stamp can be checked against it. Remote volumes are skipped: the check needs
// local shape metadata.
std::shared_ptr<Volume> attachedVolumeByStoreName(const VolumePkg& pkg, const std::string& storeName)
{
    const std::string wanted = std::filesystem::path(storeName).filename().string();
    if (wanted.empty()) return nullptr;
    for (const std::string& id : pkg.volumeIDs()) {
        const std::shared_ptr<Volume> volume = pkg.volume(id);
        if (volume && !volume->isRemote() && volume->path().filename().string() == wanted) return volume;
    }
    return nullptr;
}
}

void MenuActionController::importUmbilicus()
{
    auto& state = *_window->_state;
    const auto attachment = state.storeFor("umbilicus");
    const auto volume = state.currentVolume();
    if (!attachment || !volume || !state.vpkg()) {
        QMessageBox::information(_window, QObject::tr("Import Umbilicus"),
            QObject::tr("Open a project and a volume first."));
        return;
    }
    const auto file = QFileDialog::getOpenFileName(
        _window, QObject::tr("Import Umbilicus"),
        QString::fromStdString(state.vpkg()->directory().string()),
        QObject::tr("Umbilicus (*.json *.txt *.csv)"));
    if (file.isEmpty()) return;

    vc::objects::UmbilicusFile umbilicus;
    try {
        umbilicus = vc::objects::readUmbilicusFile(std::filesystem::path(file.toStdString()));
    } catch (const std::exception& e) {
        QMessageBox::warning(_window, QObject::tr("Import failed"),
            QObject::tr("%1 is not a valid umbilicus file:\n\n%2").arg(file, QString::fromUtf8(e.what())));
        return;
    }

    // The store decides the frame: the file's stamp against the volume the
    // umbilicus is bound to, and against an attached volume the stamp names.
    const auto grid = [](const Volume& v) {
        const double voxelSize = v.voxelSize();
        return vc::objects::UmbilicusTargetGrid{
            {double(v.sliceWidth()), double(v.sliceHeight()), double(v.numSlices())},
            std::isfinite(voxelSize) && voxelSize > 0 ? std::optional(voxelSize) : std::nullopt};
    };
    std::optional<vc::objects::UmbilicusTargetGrid> named;
    if (umbilicus.volume)
        if (const auto volumeNamed = attachedVolumeByStoreName(*state.vpkg(), *umbilicus.volume)) named = grid(*volumeNamed);
    const auto binding = vc::objects::umbilicusBinding(umbilicus, grid(*volume), named);
    if (binding.refusal) {
        QMessageBox::warning(_window, QObject::tr("Import failed"),
            QObject::tr("%1 does not fit volume %2: %3")
                .arg(QFileInfo(file).fileName(), QString::fromStdString(volume->id()), QString::fromStdString(*binding.refusal)));
        return;
    }
    // An inferred frame is offered for confirmation, and a file stating
    // nothing asks, like an unbound fiber.
    double storedUnitsPerVoxel = binding.storedUnitsPerVoxel.value_or(1.0);
    if (binding.confirm) {
        const QString reason = binding.storedUnitsPerVoxel
            ? QObject::tr("Inferred from the points: %1").arg(QString::fromStdString(binding.frame))
            : QObject::tr("The file states no frame.");
        const auto entered = vc3d::promptStoredUnitsPerVoxel(
            _window, QObject::tr("Bind Umbilicus Coordinates"), QStringLiteral("umbilicus"), *volume,
            {{"file", QFileInfo(file).fileName().toStdString()}}, reason, binding.storedUnitsPerVoxel);
        if (!entered) return;
        storedUnitsPerVoxel = *entered;
    }

    // The umbilicus belongs to the open volume's store volume: units per voxel
    // of the open volume are units per voxel of that one at its scale.
    const auto target = state.annotationVolume().value_or(vc::project::AnnotationVolume{});
    vc::objects::RevisionSelection replaced;
    state.objectStoreSession()->read(*attachment, [&](const auto& db, const auto& txn, const auto&) {
        replaced = vc::objects::volumeUmbilicusRevisions(db, txn, target.volume);
    });
    if (!replaced.empty() &&
        QMessageBox::question(_window, QObject::tr("Import Umbilicus"),
            QObject::tr("Volume %1 already has an umbilicus. Replace it?").arg(QString::fromStdString(target.volume)))
            != QMessageBox::Yes)
        return;

    try {
        vc::objects::importVolumeUmbilicus(*state.objectStoreSession(), *attachment, target.volume, umbilicus,
            storedUnitsPerVoxel * target.scale, QFileInfo(file).completeBaseName().toStdString(),
            {{"tool", "vc3d-import-umbilicus"},
             {"parameters", {{"file", file.toStdString()}, {"units_per_voxel", storedUnitsPerVoxel},
                             {"frame", binding.frame.empty() ? std::string("entered") : binding.frame}}},
             {"sources", nlohmann::json::array()}});
    } catch (const std::exception& e) {
        QMessageBox::warning(_window, QObject::tr("Import failed"), QString::fromUtf8(e.what()));
        return;
    }
    _window->showStatusBarMessage(
        QObject::tr("Imported umbilicus %1 for volume %2").arg(QFileInfo(file).fileName(), QString::fromStdString(volume->id())), 5000);
}

void MenuActionController::removeUmbilicus()
{
    auto& state = *_window->_state;
    const auto attachment = state.storeFor("umbilicus");
    const auto volumeId = state.annotationVolume() ? state.annotationVolume()->volume : std::string{};
    if (!attachment || volumeId.empty()) {
        QMessageBox::information(_window, QObject::tr("Remove Umbilicus"),
            QObject::tr("Open a project and a volume first."));
        return;
    }
    vc::objects::RevisionSelection pins;
    state.objectStoreSession()->read(*attachment, [&](const auto& db, const auto& txn, const auto&) {
        pins = vc::objects::volumeUmbilicusRevisions(db, txn, volumeId);
    });
    if (pins.empty()) {
        QMessageBox::information(_window, QObject::tr("Remove Umbilicus"),
            QObject::tr("Volume %1 has no umbilicus.").arg(QString::fromStdString(volumeId)));
        return;
    }
    state.objectStoreSession()->act(*attachment, pins, [&](auto& db, auto& txn) {
        for (const auto& [id, revision] : pins) db.retireObject(txn, id);
    });
    _window->showStatusBarMessage(
        QObject::tr("Removed the umbilicus of volume %1").arg(QString::fromStdString(volumeId)), 5000);
}

void MenuActionController::attachLasagnaManifest()
{
    beginLasagnaManifestAttachment(false);
}

void MenuActionController::attachRemoteLasagnaManifest()
{
    beginLasagnaManifestAttachment(true);
}

void MenuActionController::attachCatalogRepresentation(const vcr::catalog::OpenDataSample& sample,
                                                       const vcr::catalog::OpenDataRepresentationRef& ref)
{
    if (!_window || !_window->_state || !_window->_state->vpkg()) {
        QMessageBox::information(_window, QObject::tr("No project"), QObject::tr("Open or create a project first."));
        return;
    }
    const auto targetPackage = _window->_state->vpkg();
    const auto reportMessages = [this](const QString& title, const std::vector<std::string>& messages) {
        QStringList lines;
        for (const auto& message : messages)
            lines.push_back(QString::fromStdString(message));
        QMessageBox::warning(_window, title, lines.join('\n'));
    };
    // Only this representation: no scan views beside it.
    vc3d::opendata::OpenDataResourceSelection selection;
    selection.rawVolumeIds.emplace();
    selection.representations = std::vector<vcr::catalog::OpenDataRepresentationRef>{ref};

    if (ref.kind == vcr::catalog::OpenDataRepresentationKind::Lasagna) {
        // Caching and validating the dataset reads the network; the project
        // only changes on the GUI thread, if it is still the open one.
        using Preparation = vc3d::opendata::OpenDataLasagnaPreparation;
        auto* watcher = new QFutureWatcher<Preparation>(this);
        connect(watcher, &QFutureWatcher<Preparation>::finished, this,
                [this, watcher, targetPackage, reportMessages]() {
                    auto preparation = watcher->result();
                    watcher->deleteLater();
                    if (!_window || !_window->_state || _window->_state->vpkg() != targetPackage) return;
                    auto messages = preparation.messages;
                    if (vc3d::opendata::attachOpenDataLasagna(*targetPackage, preparation, &messages) > 0)
                        _window->refreshCurrentVolumePackageUi();
                    if (!messages.empty())
                        reportMessages(QObject::tr("Attach Lasagna"), messages);
                });
        watcher->setFuture(QtConcurrent::run([sample, selection, cacheRoot = vc3d::remoteCachePathFs()]() {
            return vc3d::opendata::prepareOpenDataLasagna(sample, cacheRoot, &selection);
        }));
        return;
    }

    try {
        if (ref.kind == vcr::catalog::OpenDataRepresentationKind::NormalGrids) {
            if (vc3d::opendata::attachOpenDataNormalGrids(*targetPackage, sample, vc3d::remoteCachePathFs(),
                                                          &selection) == 0) {
                QMessageBox::warning(_window, tr("Attach failed"), tr("Could not attach normal grids."));
                return;
            }
            _window->refreshCurrentVolumePackageUi();
            return;
        }
        const auto result = vc3d::opendata::attachOpenDataSampleVolumes(*targetPackage, sample, &selection, vc3d::remoteCachePathFs());
        _window->refreshCurrentVolumePackageUi();
        if (result.failedVolumes)
            reportMessages(tr("Attach incomplete"), result.messages);
    } catch (const std::exception& e) {
        QMessageBox::warning(_window, tr("Attach failed"), QString::fromUtf8(e.what()));
    }
}

void MenuActionController::beginLasagnaManifestAttachment(bool remote)
{
    if (!_window || !_window->_state || !_window->_state->vpkg()) {
        QMessageBox::information(_window, QObject::tr("No project"), QObject::tr("Open or create a project first."));
        return;
    }
    if (_lasagnaAttachmentInFlight) {
        QMessageBox::information(_window, QObject::tr("Attachment in progress"), QObject::tr("A Lasagna manifest attachment is already in progress."));
        return;
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    QString location;
    if (remote) {
        location =
            promptLocation(QObject::tr("Attach Remote Lasagna Manifest"), QObject::tr("Pick a remote .lasagna.json manifest."), QStringLiteral("s3://"), {QStringLiteral("*.lasagna.json")}, true, false, false);
        if (location.isEmpty())
            return;
        if (!vcr::zarr::isRemote(location.toStdString())) {
            QMessageBox::warning(_window, QObject::tr("Attach failed"), QObject::tr("The remote action requires an s3://, http://, https://, or sftp:// manifest."));
            return;
        }
    } else {
        location = QFileDialog::
            getOpenFileName(_window, QObject::tr("Attach Lasagna Manifest"), settings.value(vc3d::settings::project::DEFAULT_PATH).toString(), QObject::tr("Lasagna manifests (*.lasagna.json);;JSON files (*.json)"));
        if (location.isEmpty())
            return;
        location = QFileInfo(location).absoluteFilePath();
        settings.setValue(vc3d::settings::project::DEFAULT_PATH, QFileInfo(location).absolutePath());
    }

    bool roleAccepted = false;
    const QString role =
        QInputDialog::getItem(_window, QObject::tr("Lasagna Data Role"), QObject::tr("Attach this manifest as:"), {QObject::tr("Regular Lasagna"), QObject::tr("Fiber inference")}, 0, false, &roleAccepted);
    if (!roleAccepted)
        return;
    const bool fiberInference = role == QObject::tr("Fiber inference");

    // Requests are signed with the environment's AWS credentials where a bucket needs
    // them (vcr::zarr::resolveAccess); a remote manifest or group needs a cache root.
    vc::lasagna::LasagnaDatasetOpenOptions openOptions;
    bool needsRemoteCache = remote;
    if (!remote) {
        try {
            const auto parsed = vc::lasagna::LasagnaDatasetManifest::parseFile(std::filesystem::path(location.toStdString()));
            needsRemoteCache = std::any_of(parsed.groups.begin(), parsed.groups.end(), [](const auto& group) {
                return vcr::zarr::isRemote(group.relativeZarrKey);
            });
        } catch (...) {
            // The background loader reports manifest parse errors uniformly.
        }
    }
    if (needsRemoteCache) {
        openOptions.remoteCacheRoot = vc3d::remoteCachePathFs();
    }

    const auto targetPackage = _window->_state->vpkg();
    const std::string persistedLocation = location.toStdString();
    auto* watcher = new QFutureWatcher<LasagnaAttachTaskResult>(this);
    connect(watcher, &QFutureWatcher<LasagnaAttachTaskResult>::finished, this, [this, watcher, targetPackage, persistedLocation, fiberInference]() {
        auto task = watcher->result();
        watcher->deleteLater();
        _lasagnaAttachmentInFlight = false;
        if (_attachLasagnaManifestAct)
            _attachLasagnaManifestAct->setEnabled(true);
        if (_attachRemoteLasagnaManifestAct)
            _attachRemoteLasagnaManifestAct->setEnabled(true);
        if (!task.error.isEmpty()) {
            QMessageBox::warning(_window, QObject::tr("Attach Lasagna failed"), task.error);
            return;
        }
        if (!_window || !_window->_state || _window->_state->vpkg() != targetPackage) {
            QMessageBox::warning(_window, QObject::tr("Attach Lasagna failed"), QObject::tr("The open project changed while the manifest was loading."));
            return;
        }
        try {
            const auto result = targetPackage->attachPreparedLasagnaDataset(
                persistedLocation, {}, fiberInference, task.volumes);
            if (result == VolumePkg::AttachLasagnaResult::VolumeIdConflict) {
                QMessageBox::warning(_window, QObject::tr("Attach Lasagna failed"), QObject::tr("A Lasagna volume conflicts with an existing volume id."));
                return;
            }
            _window->refreshCurrentVolumePackageUi();
        } catch (const std::exception& error) {
            QMessageBox::warning(_window, QObject::tr("Attach Lasagna failed"), QString::fromUtf8(error.what()));
        }
    });

    _lasagnaAttachmentInFlight = true;
    _attachLasagnaManifestAct->setEnabled(false);
    _attachRemoteLasagnaManifestAct->setEnabled(false);
    watcher->setFuture(QtConcurrent::run([persistedLocation, openOptions, fiberInference]() {
        LasagnaAttachTaskResult result;
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
        } catch (const std::exception& error) {
            result.error = QString::fromUtf8(error.what());
        } catch (...) {
            result.error = QObject::tr("Unknown error while loading the Lasagna manifest.");
        }
        return result;
    }));
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
    for (const auto& e : pkg->normalGridEntries()) {
        items << QObject::tr("[normal_grid] %1").arg(QString::fromStdString(e.location));
        locations << QString::fromStdString(e.location);
    }
    for (const auto& e : pkg->allLasagnaDatasetEntries()) {
        const bool fiber = vc::project::isFiberLasagnaEntry(e);
        items << QObject::tr("[%1 Lasagna] %2").arg(fiber ? QObject::tr("fiber") : QObject::tr("regular"), QString::fromStdString(e.location));
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
    detachEntryAt(locations.at(idx));
}

void MenuActionController::detachEntryAt(const QString& loc)
{
    auto pkg = _window && _window->_state ? _window->_state->vpkg() : nullptr;
    if (!pkg) return;
    const QString scope = pkg->entryOwner(loc.toStdString()).empty()
        ? QObject::tr("It is removed for everyone using this project.")
        : QObject::tr("Only you see this entry.");
    if (QMessageBox::question(_window, QObject::tr("Detach"),
            QObject::tr("Remove this entry from the project?\n\n%1\n\n%2").arg(loc, scope),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No) != QMessageBox::Yes) {
        return;
    }
    bool changed = false;
    try { changed = pkg->removeEntry(loc.toStdString()); }
    catch (const std::exception& error) {
        QMessageBox::warning(_window, tr("Project save failed"), QString::fromUtf8(error.what()));
        return;
    }
    if (!changed) {
        QMessageBox::warning(_window, QObject::tr("Detach failed"),
            QObject::tr("Could not remove entry (not found)."));
        return;
    }
    _window->refreshCurrentVolumePackageUi();
}

QString MenuActionController::objectStoreAuthor()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    QString name = settings.value(vc3d::settings::viewer::USERNAME,
                                  vc3d::settings::viewer::USERNAME_DEFAULT).toString().trimmed();
    if (!vc3d::line_annotation::isAnonymousFiberUsername(name.toStdString())) return name;
    name = qEnvironmentVariable("USER", qEnvironmentVariable("USERNAME"));
    QString prompt = QObject::tr("Username recorded on your object store edits and fiber names:");
    for (;;) {
        bool ok = false;
        name = QInputDialog::getText(_window, QObject::tr("Object Store Username"), prompt,
                                     QLineEdit::Normal, name, &ok).trimmed();
        if (!ok) return {};
        if (!vc3d::line_annotation::isAnonymousFiberUsername(name.toStdString())) break;
        prompt = QObject::tr("Enter a username other than \"anon\":");
    }
    settings.setValue(vc3d::settings::viewer::USERNAME, name);
    return name;
}

bool MenuActionController::attachStore(bool readOnly, const std::string& kind)
{
    auto& state = *_window->_state;
    if (!requireProject(QObject::tr("Attaching a store"))) return false;
    const auto path = QFileDialog::getOpenFileName(_window,
        readOnly ? QObject::tr("Attach Read-only Store") : QObject::tr("Attach Store"),
        {}, QObject::tr("Object store (*.vcobj);;All files (*)"));
    if (path.isEmpty()) return false;
    const auto author = readOnly ? CState::storeAuthor() : objectStoreAuthor();
    if (author.isEmpty()) return false;
    try {
        const auto attachment = state.attachStore(path.toStdString(), author, readOnly);
        if (!kind.empty()) state.setStoreFor(kind, attachment);
    } catch (const std::exception& error) {
        QMessageBox::critical(_window, QObject::tr("Cannot attach store"), QString::fromUtf8(error.what()));
        return false;
    }
    return true;
}

bool MenuActionController::requireProject(const QString& purpose)
{
    if (_window->_state->hasVpkg()) return true;
    QMessageBox box(QMessageBox::Information, QObject::tr("Project Required"),
        QObject::tr("%1 requires an open project, and none is open.\n\n"
                    "Create a new project (.vcproj), or open an existing one.").arg(purpose),
        QMessageBox::Cancel, _window);
    auto* create = box.addButton(QObject::tr("New Project..."), QMessageBox::AcceptRole);
    auto* open = box.addButton(QObject::tr("Open Project..."), QMessageBox::AcceptRole);
    box.setDefaultButton(create);
    box.exec();
    if (box.clickedButton() == create) newProject();
    else if (box.clickedButton() == open) openVolpkg();
    return _window->_state->hasVpkg();
}

void MenuActionController::refreshStoresMenu()
{
    if (!_storesMenu || !_window || !_window->_state) return;
    auto& state = *_window->_state;
    const auto session = state.objectStoreSession();
    _storesMenu->clear();
    const bool project = state.hasVpkg();
    const auto attached = state.attachedStores();
    const auto storeName = [&](const vc::objects::AttachmentIdentity& attachment) {
        auto name = QString::fromStdString(session->path(attachment).filename().string());
        if (!session->writable(attachment)) name += QObject::tr(" [read-only]");
        if (session->hubStatus(attachment)) name += QObject::tr(" [hosted]");
        return name.replace("&", "&&");
    };
    _storesMenu->addAction(QObject::tr("Attach Store..."), this, [this]() { attachStore(false); })->setEnabled(project);
    _storesMenu->addAction(QObject::tr("Attach Read-only Store..."), this, [this]() { attachStore(true); })->setEnabled(project);
    _storesMenu->addAction(QObject::tr("Clone Hosted Store..."), this, [this]() { cloneHostedStore(false); })->setEnabled(project);
    _storesMenu->addAction(tr("New Store..."), this, &MenuActionController::newStore)->setEnabled(project);
    _storesMenu->addSeparator();
    const std::vector<std::pair<std::string, QString>> kinds{
        {"surface", QObject::tr("Show Surfaces From")},
        {"fiber", QObject::tr("Show Fibers From")},
        {"point_collection", QObject::tr("Show Point Collections From")},
        {"umbilicus", QObject::tr("Show Umbilici From")},
        {"volume_transform", QObject::tr("Use Registrations From")}};
    for (const auto& [kind, title] : kinds) {
        auto* menu = _storesMenu->addMenu(title);
        menu->setEnabled(project);
        if (!project) continue;
        auto* group = new QActionGroup(menu);
        const auto current = state.storeFor(kind);
        auto* none = menu->addAction(tr("None"));
        none->setCheckable(true);
        none->setChecked(!current);
        group->addAction(none);
        connect(none, &QAction::triggered, this, [this, kind]() {
            try { _window->_state->setStoreFor(kind, std::nullopt); }
            catch (const std::exception& error) { QMessageBox::warning(_window, tr("Stores"), QString::fromUtf8(error.what())); }
        });
        menu->setEnabled(state.vpkg()->writable());
        const auto& choices = attached;
        for (const auto& choice : choices) {
            auto* action = menu->addAction(storeName(choice));
            action->setCheckable(true);
            action->setChecked(current == choice);
            group->addAction(action);
            connect(action, &QAction::triggered, this, [this, kind = kind, choice]() {
                try { _window->_state->setStoreFor(kind, choice); }
                catch (const std::exception& error) {
                    QMessageBox::warning(_window, QObject::tr("Stores"), QString::fromUtf8(error.what()));
                }
            });
        }
    }
    _storesMenu->addSeparator();
    _storesMenu->addAction(QObject::tr("Import from Folder..."), this, &MenuActionController::importIntoObjectStore)
        ->setEnabled(project);
    std::vector<vc::objects::AttachmentIdentity> hosted;
    for (const auto& attachment : attached) if (session->hubStatus(attachment)) hosted.push_back(attachment);
    auto* sync = _storesMenu->addAction(QObject::tr("Sync Hosted Stores Now"), this, [this, hosted]() {
        for (const auto& attachment : hosted) _window->_state->objectStoreSession()->syncHub(attachment);
        _window->promptHubConflicts();
    });
    sync->setEnabled(!hosted.empty());
    auto* detach = _storesMenu->addMenu(QObject::tr("Detach Store"));
    detach->setEnabled(!attached.empty());
    for (const auto& attachment : attached)
        detach->addAction(storeName(attachment), this, [this, attachment]() {
            try { _window->_state->detachStore(attachment); }
            catch (const std::exception& error) {
                QMessageBox::critical(_window, QObject::tr("Cannot detach store"), QString::fromUtf8(error.what()));
            }
        });
    if (project && vc3d::opendata::openDataProjectSample(*state.vpkg())) {
        _storesMenu->addSeparator();
        _storesMenu->addAction(QObject::tr("Reset Project to Open Data..."), this, &MenuActionController::resetOpenDataProject)
            ->setEnabled(state.vpkg()->writable());
    }
}

void MenuActionController::cloneHostedStore(bool asProject)
{
    if (!asProject && !requireProject(QObject::tr("Attaching a hosted store"))) return;
    const auto filter = QObject::tr("Object store (*.vcobj);;All files (*)");
    QDialog chooser(_window);
    chooser.setWindowTitle(asProject ? QObject::tr("Clone Hosted Project") : QObject::tr("Clone Hosted Store"));
    auto* layout = new QVBoxLayout(&chooser);
    layout->addWidget(new QLabel(QObject::tr("Hub on a network share, or an sftp:// or ssh:// address "
                                             "(sftp://user@host/path/hub.vcobj):"), &chooser));
    auto* row = new QHBoxLayout;
    auto* location = new QLineEdit(&chooser);
    location->setMinimumWidth(480);
    location->setPlaceholderText(QStringLiteral("sftp://user@host/path/hub.vcobj"));
    auto* browse = new QPushButton(QObject::tr("Browse..."), &chooser);
    QObject::connect(browse, &QPushButton::clicked, &chooser, [&] {
        const auto file = QFileDialog::getOpenFileName(&chooser,
            QObject::tr("Choose the hub on the network share"), {}, filter);
        if (!file.isEmpty()) location->setText(file);
    });
    row->addWidget(location);
    row->addWidget(browse);
    layout->addLayout(row);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &chooser);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &chooser, &QDialog::accept);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &chooser, &QDialog::reject);
    layout->addWidget(buttons);
    if (chooser.exec() != QDialog::Accepted) return;
    const auto hub = location->text().trimmed();
    if (hub.isEmpty()) return;
    // The last path component names the replica, for addresses and files alike.
    const auto hubName = QFileInfo(hub.section(QLatin1Char('/'), -1)).completeBaseName();
    const auto replica = QFileDialog::getSaveFileName(_window, QObject::tr("Local Replica"),
        QDir::home().filePath(hubName + QStringLiteral("-replica.vcobj")), filter);
    if (replica.isEmpty()) return;
    const auto author = objectStoreAuthor();
    if (author.isEmpty()) return;

    auto* progressDialog = new QProgressDialog(QObject::tr("Copying the hosted store..."), {}, 0, 1000, _window);
    progressDialog->setWindowModality(Qt::WindowModal);
    progressDialog->setMinimumDuration(0);
    progressDialog->setValue(0);
    QPointer<QProgressDialog> guardedDialog(progressDialog);
    auto progress = [guardedDialog](const vc::objects::util::ProgressUpdate& event) {
        if (!event.total) return;
        const int value = int(1000 * event.completed / event.total);
        QMetaObject::invokeMethod(guardedDialog, [guardedDialog, value] {
            if (guardedDialog) guardedDialog->setValue(value);
        }, Qt::QueuedConnection);
    };
    auto* watcher = new QFutureWatcher<QString>(this);
    connect(watcher, &QFutureWatcher<QString>::finished, this, [this, watcher, progressDialog, replica, author, asProject] {
        const auto error = watcher->result();
        watcher->deleteLater();
        progressDialog->close();
        progressDialog->deleteLater();
        if (!error.isEmpty()) {
            QMessageBox::critical(_window, QObject::tr("Cannot clone hosted store"), error);
            return;
        }
        if (asProject) {
            try {
                auto project = std::filesystem::path(replica.toStdString());
                project.replace_extension(".vcproj");
                vc::project::ProjectDocument::create(project, project.stem().string(), replica.toStdString());
                _window->OpenVolume(QString::fromStdString(project.string()));
            } catch (const std::exception& error) {
                QMessageBox::warning(_window, tr("Cannot create project"), QString::fromUtf8(error.what()));
            }
            _window->UpdateView();
            return;
        }
        try {
            _window->_state->attachStore(replica.toStdString(), author);
        } catch (const std::exception& attachError) {
            QMessageBox::critical(_window, QObject::tr("Cannot attach store"), QString::fromUtf8(attachError.what()));
        }
    });
    watcher->setFuture(QtConcurrent::run([hub, replica, author, progress]() {
        try {
            vc::objects::OpenOptions options;
            options.author = author.toStdString();
            vc::objects::cloneHub(hub.toStdString(), replica.toStdString(), options, progress);
            return QString();
        } catch (const std::exception& error) {
            return QString::fromUtf8(error.what());
        }
    }));
}

void MenuActionController::importIntoObjectStore()
{
    if (!requireProject(QObject::tr("Importing surfaces or fibers"))) return;
    auto& state = *_window->_state;
    const auto session = state.objectStoreSession();
    const auto volume = state.currentVolume();

    QDialog dialog(_window);
    dialog.setWindowTitle(QObject::tr("Import from Folder"));
    auto* layout = new QVBoxLayout(&dialog);
    auto* form = new QFormLayout;
    auto* kind = new QComboBox(&dialog);
    kind->addItem(QObject::tr("tifxyz surfaces"), QStringLiteral("surface"));
    kind->addItem(QObject::tr("Fibers (vc3d_fiber JSON)"), QStringLiteral("fiber"));
    form->addRow(QObject::tr("Import:"), kind);
    auto* row = new QHBoxLayout;
    auto* folder = new QLineEdit(&dialog);
    folder->setMinimumWidth(420);
    folder->setPlaceholderText(QObject::tr("A folder of tifxyz directories, or one tifxyz directory"));
    auto* browse = new QPushButton(QObject::tr("Browse..."), &dialog);
    QObject::connect(browse, &QPushButton::clicked, &dialog, [&] {
        const auto start = folder->text().isEmpty() && state.vpkg()
            ? QString::fromStdString(state.vpkg()->directory().string()) : folder->text();
        const auto chosen = QFileDialog::getExistingDirectory(&dialog, QObject::tr("Folder to Import"), start);
        if (!chosen.isEmpty()) folder->setText(chosen);
    });
    QObject::connect(kind, &QComboBox::currentIndexChanged, &dialog, [kind, folder] {
        folder->setPlaceholderText(kind->currentData().toString() == QStringLiteral("surface")
            ? QObject::tr("A folder of tifxyz directories, or one tifxyz directory")
            : QObject::tr("A folder of fiber JSON files"));
    });
    row->addWidget(folder);
    row->addWidget(browse);
    form->addRow(QObject::tr("Folder:"), row);
    // The folder's name until the user types tags of their own.
    auto* tags = new QLineEdit(&dialog);
    tags->setPlaceholderText(QObject::tr("comma-separated; added to every imported object"));
    auto tagsEdited = std::make_shared<bool>(false);
    QObject::connect(tags, &QLineEdit::textEdited, &dialog, [tagsEdited] { *tagsEdited = true; });
    QObject::connect(folder, &QLineEdit::textChanged, &dialog, [tags, tagsEdited](const QString& text) {
        if (!*tagsEdited) tags->setText(QFileInfo(QDir::cleanPath(text.trimmed())).fileName());
    });
    form->addRow(QObject::tr("Tags:"), tags);
    auto* bind = new QCheckBox(volume
        ? QObject::tr("Bind to the active volume (%1)").arg(QString::fromStdString(volume->id()))
        : QObject::tr("Bind to the active volume (none open: imports stay unbound)"), &dialog);
    bind->setEnabled(volume != nullptr);
    bind->setChecked(volume != nullptr);
    auto* scale = new QLineEdit(&dialog);
    scale->setPlaceholderText(QObject::tr("stored units per active-volume voxel; no scale is assumed"));
    scale->setEnabled(bind->isChecked());
    QObject::connect(bind, &QCheckBox::toggled, scale, &QLineEdit::setEnabled);
    form->addRow(bind);
    form->addRow(QObject::tr("Scale:"), scale);
    layout->addLayout(form);
    auto* note = new QLabel(QObject::tr("Sources declaring their own coordinates keep them; a conflict with "
                                        "this binding is reported. Objects already in the store are reported, "
                                        "not re-imported."), &dialog);
    note->setWordWrap(true);
    layout->addWidget(note);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, [&] {
        bool valid = false;
        const double value = scale->text().toDouble(&valid);
        if (!QFileInfo(folder->text().trimmed()).isDir())
            QMessageBox::warning(&dialog, dialog.windowTitle(), QObject::tr("Choose an existing folder."));
        else if (bind->isChecked() && (!valid || !(value > 0) || !std::isfinite(value)))
            QMessageBox::warning(&dialog, dialog.windowTitle(), QObject::tr("Enter a positive finite coordinate scale."));
        else dialog.accept();
    });
    layout->addWidget(buttons);
    if (dialog.exec() != QDialog::Accepted) return;

    const auto kindName = kind->currentData().toString().toStdString();
    // Into the store objects of the kind are kept in.
    const auto attachment = state.storeFor(kindName);
    if (!attachment || !session->writable(*attachment)) {
        QMessageBox::warning(_window, QObject::tr("Import from Folder"),
            QObject::tr("The store %1 are kept in is read-only.")
                .arg(kindName == "surface" ? QObject::tr("surfaces") : QObject::tr("fibers")));
        return;
    }
    const std::filesystem::path source = folder->text().trimmed().toStdString();
    const auto coordinates = bind->isChecked()
        ? vc::objects::volumeCoordinates(state.annotationVolume()->volume,
                                         scale->text().toDouble() * state.annotationVolume()->scale).dump()
        : std::string("{}");
    std::vector<std::string> tagNames;
    for (const auto& tag : tags->text().split(QLatin1Char(','), Qt::SkipEmptyParts))
        if (const auto name = tag.trimmed(); !name.isEmpty()) tagNames.push_back(name.toStdString());

    auto* progressDialog = new QProgressDialog(QObject::tr("Importing %1...").arg(folder->text().trimmed()), {}, 0, 0, _window);
    progressDialog->setWindowTitle(QObject::tr("Import from Folder"));
    progressDialog->setWindowModality(Qt::WindowModal);
    progressDialog->setMinimumDuration(0);
    progressDialog->setValue(0);
    QPointer<QProgressDialog> guardedDialog(progressDialog);
    auto progress = [guardedDialog](const vc::objects::util::ProgressUpdate& event) {
        if (!event.total) return;
        const int completed = int(event.completed), total = int(event.total);
        const auto detail = QString::fromUtf8(event.detail.data(), qsizetype(event.detail.size()));
        QMetaObject::invokeMethod(guardedDialog, [guardedDialog, completed, total, detail] {
            if (!guardedDialog) return;
            guardedDialog->setMaximum(total);
            guardedDialog->setValue(completed);
            if (!detail.isEmpty()) guardedDialog->setLabelText(QObject::tr("Importing %1 (%2 of %3)").arg(detail).arg(completed + 1).arg(total));
        }, Qt::QueuedConnection);
    };
    using Report = vc::objects::GeometryIngestReport;
    auto* watcher = new QFutureWatcher<Report>(this);
    connect(watcher, &QFutureWatcher<Report>::finished, this, [this, watcher, progressDialog, kindName] {
        const auto report = watcher->result();
        watcher->deleteLater();
        progressDialog->close();
        progressDialog->deleteLater();
        const auto what = kindName == "surface" ? QObject::tr("surface(s)") : QObject::tr("fiber(s)");
        const auto summary = QObject::tr("Imported %1 %2.").arg(report.ingested).arg(what);
        if (report.problems.empty()) {
            _window->showStatusBarMessage(summary, 5000);
            return;
        }
        QStringList problems;
        for (const auto& problem : report.problems) problems << QString::fromStdString(problem);
        QMessageBox box(QMessageBox::Warning, QObject::tr("Import from Folder"),
            summary + QLatin1Char(' ') + QObject::tr("%n item(s) could not be imported.", nullptr, int(problems.size())),
            QMessageBox::Ok, _window);
        box.setDetailedText(problems.join(QLatin1Char('\n')));
        box.exec();
    });
    watcher->setFuture(QtConcurrent::run([session, attachment = *attachment, kindName, source, progress, coordinates, tagNames]() {
        try {
            return vc::objects::ingestGeometrySource(*session, attachment, kindName, source, progress, coordinates, tagNames);
        } catch (const std::exception& error) {
            Report report;
            report.problems.push_back(error.what());
            return report;
        }
    }));
}

