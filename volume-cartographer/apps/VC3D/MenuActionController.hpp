#pragma once
#include <vc_reader/catalog.hpp>

#include <QObject>
#include <QFutureWatcher>
#include <QPointer>
#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "OpenDataVolumePrefill.hpp"
#include "OpenDataManifest.hpp"
#include "OpenDataSampleProject.hpp"

#include <functional>

class QAction;
class QDialog;
class QMenu;
class QMenuBar;
class CWindow;
class Volume;

namespace vc3d::opendata {

struct OpenDataSampleProjectResult;
class OpenDataCatalogWindow;
}

class MenuActionController : public QObject
{
    Q_OBJECT

public:
    static constexpr int kMaxRecentVolpkg = 10;

    explicit MenuActionController(CWindow* window);
    ~MenuActionController() override;

    void populateMenus(QMenuBar* menuBar);
    void updateRecentVolpkgList(const QString& path);
    void refreshRecentMenu();
    bool openVolpkgAt(const QString& path,
                      bool interactive = true,
                      QString* errorMessage = nullptr);
    void showOpenDataCatalog();
    bool isOpenDataCatalogVisible() const;
    // Prompts for a .vcobj (and a username unless a non-"anon" one is saved)
    // and attaches it beside the project; with `kind`, objects of that kind are
    // then kept in it. True once attached.
    bool attachStore(bool readOnly = false, const std::string& kind = {});
    // With no project open, explains that `purpose` needs one and offers to
    // create or open one. True once a project is open.
    bool requireProject(const QString& purpose);
    // Author recorded on edits: the saved username, asked for until it is not "anon".
    // Empty means cancelled.
    QString objectStoreAuthor();
    // Clones a hosted store to a local replica, then opens it as the project
    // or attaches it beside the project.
    void cloneHostedStore(bool asProject);
    // Collects a kind, a source folder and an optional volume binding, then ingests
    // it into the kind's store in the background (vc::objects::ingestGeometrySource).
    void importIntoObjectStore();
    // Terminal outcome of an asynchronous Open Data sample open.
    struct OpenDataSampleOpenOutcome {
        bool success{false};
        QString error;                                       // set iff !success
        vc3d::opendata::OpenDataSampleProjectResult result;  // attach counters/messages
    };

    // Starts a catalog sample open without spinning a nested event loop and
    // returns immediately.
    //
    // Returns false (with *errorMessage) on synchronous precondition failure (no
    // window/state, or an open already in flight). Returns true once the background task started; exactly one GUI-thread
    // call to `onFinished` follows. `onProgress`, if set, is invoked on the GUI thread
    // with the same stream the interactive QProgressDialog consumes. `selection`
    // restricts the attached resources. This entry point is always non-interactive.
    bool startOpenDataSampleOpen(
        const vcr::catalog::OpenDataSample& sample,
        const vc3d::opendata::OpenDataResourceSelection& selection,
        std::function<void(const OpenDataSampleOpenOutcome&)> onFinished,
        std::function<void(const vc3d::opendata::OpenDataSampleDownloadProgress&)> onProgress = {},
        QString* errorMessage = nullptr);

    // True while any sample open is in flight.
    bool openDataSampleOpenInFlight() const;

    // Remove a project entry (volume, normal grid or Lasagna dataset) by
    // location, after asking.
    void detachEntryAt(const QString& location);
    // Open Data: the sample's project store, made or refreshed from the catalog.
    void resetOpenDataProject();

public slots:
    void newProject();
    void newStore();
    void saveProjectAs();
    void attachVolume();
    void attachNormalGrid();
    void attachLasagnaManifest();
    void attachRemoteLasagnaManifest();
    void openVolpkg();
    void attachRemoteZarr();
    void showSettingsDialog();

private slots:
    void importUmbilicus();
    void removeUmbilicus();
    void detachEntry();
    void openRecentVolpkg();
    void showAboutDialog();
    void showKeybindings();
    void resetSegmentationViews();
    void toggleConsoleOutput();
    void toggleDrawBBox(bool enabled);
    void toggleCursorMirroring(bool enabled);
    void surfaceFromSelection();
    void clearSelection();
    void beginRotateSurfaceTransform();
    void exitApplication();

signals:
    // Emitted when the user picks Actions -> Merge Surfaces... CWindow
    // wires this to SegmentationCommandHandler::onMergeSurfaces with an
    // empty seed list so the dialog opens with an empty grid.
    void mergeSurfacesFromMenuRequested();
    // Emitted when the user picks Actions -> Patch Surface... CWindow
    // wires this to SegmentationCommandHandler::onMergePatch with an
    // empty seed list so the dialog opens with empty combo boxes.
    void mergePatchFromMenuRequested();
    void growTrackPatchesFromMenuRequested();
    void openDataCatalogVisibilityChanged(bool visible);

private:
    // The Stores menu: the stores beside the project and the store each kind uses.
    void refreshStoresMenu();
    QStringList loadRecentPaths() const;
    void saveRecentPaths(const QStringList& paths);
    void ensureRecentActions();

    QStringList loadRecentRemoteUrls() const;
    void saveRecentRemoteUrls(const QStringList& urls);
    void updateRecentRemoteList(const QString& url);
    void showAttachRemoteZarrDialog(const QString& initialUrl);
    void attachRemoteZarrUrl(const QString& url);
    void beginLasagnaManifestAttachment(bool remote);
    void attachCatalogRepresentation(const vcr::catalog::OpenDataSample& sample,
                                     const vcr::catalog::OpenDataRepresentationRef& ref);
    struct LasagnaAttachTaskResult;
    // Interactive: prompts before replacing the open project and shows progress.
    bool openOpenDataSample(const vcr::catalog::OpenDataSample& sample,
                            const vc3d::opendata::OpenDataResourceSelection* selection = nullptr,
                            const vc3d::opendata::OpenDataNewProject* newProject = nullptr);
    // .cpp-local payload keeps the QtConcurrent result type out of this header.
    struct OpenDataOpenTaskResult;
    // Launches the QtConcurrent task without a nested event loop. Its watcher is
    // parented to this; completion and onFinished run on the GUI thread.
    void beginOpenDataSampleOpenTask(
        const vcr::catalog::OpenDataSample& sample,
        bool interactive,
        const vc3d::opendata::OpenDataResourceSelection* selection,
        std::function<void(const OpenDataSampleOpenOutcome&)> onFinished,
        std::function<void(const vc3d::opendata::OpenDataSampleDownloadProgress&)> onProgress,
        const vc3d::opendata::OpenDataNewProject* newProject = nullptr);
    // Epilogue run on the GUI thread once the open task finishes: setVpkg, UI
    // refresh, prefill, status message, and (interactive-only) message boxes.
    void finishOpenDataSampleOpen(OpenDataOpenTaskResult task,
                                  bool interactive,
                                  OpenDataSampleOpenOutcome* outcomeOut);
    void startOpenDataVolumePrefill(const std::shared_ptr<Volume>& volume);
    void cancelOpenDataVolumePrefills();
    QString promptLocation(const QString& title,
                           const QString& hint,
                           const QString& defaultDir,
                           const QStringList& localFilters,
                           bool acceptFiles,
                           bool acceptDirs,
                           bool startAtFile = false);

    CWindow* _window{nullptr};

    QMenu* _fileMenu{nullptr};
    QMenu* _editMenu{nullptr};
    QMenu* _viewMenu{nullptr};
    QMenu* _actionsMenu{nullptr};
    QMenu* _transformsMenu{nullptr};
    QMenu* _selectionMenu{nullptr};
    QMenu* _helpMenu{nullptr};
    QMenu* _recentMenu{nullptr};
    QMenu* _storesMenu{nullptr};

    QAction* _newProjectAct{nullptr};
    QAction* _saveProjectAsAct{nullptr};
    QAction* _attachVolumeAct{nullptr};
    QAction* _attachNormalGridAct{nullptr};
    QAction* _importUmbilicusAct{nullptr};
    QAction* _removeUmbilicusAct{nullptr};
    QAction* _attachLasagnaManifestAct{nullptr};
    QAction* _attachRemoteLasagnaManifestAct{nullptr};
    QAction* _detachEntryAct{nullptr};
    QAction* _openAct{nullptr};
    QAction* _attachRemoteZarrAct{nullptr};
    QAction* _openDataCatalogAct{nullptr};
    std::array<QAction*, kMaxRecentVolpkg> _recentActs{};
    QAction* _settingsAct{nullptr};
    QAction* _exitAct{nullptr};
    QAction* _keybindsAct{nullptr};
    QAction* _aboutAct{nullptr};
    QAction* _resetViewsAct{nullptr};
    QAction* _showConsoleAct{nullptr};
    QAction* _drawBBoxAct{nullptr};
    QAction* _mirrorCursorAct{nullptr};
    QAction* _surfaceFromSelectionAct{nullptr};
    QAction* _selectionClearAct{nullptr};
    QAction* _rotateSurfaceAct{nullptr};
    QAction* _mergeSurfacesAct{nullptr};
    QAction* _mergePatchAct{nullptr};
    QAction* _growTrackPatchesAct{nullptr};

    QPointer<QDialog> _keybindsDialog;
    QPointer<vc3d::opendata::OpenDataCatalogWindow> _openDataCatalogDialog;
    std::vector<QFutureWatcher<vc3d::opendata::OpenDataVolumePrefillResult>*> _openDataPrefillWatchers;
    std::vector<std::shared_ptr<std::atomic<bool>>> _openDataPrefillCancelFlags;
    std::shared_ptr<std::atomic<bool>> _openDataPrefillCancelFlag;
    // True from launch until the finished slot runs; prevents overlapping opens.
    bool _openDataSampleOpenInFlight{false};
    bool _lasagnaAttachmentInFlight{false};
};
