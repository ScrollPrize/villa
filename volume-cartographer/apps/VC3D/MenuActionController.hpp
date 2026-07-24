#pragma once

#include <QObject>
#include <QFutureWatcher>
#include <QPointer>
#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "vc/core/util/RemoteAuth.hpp"
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
struct OpenDataSample;
struct OpenDataSampleProjectResult;
class OpenDataCatalogWindow;
}

// Options for opening an Open Data sample. `selection` restricts attached
// resources; `interactive` permits prompts, progress dialogs, and message boxes.
struct OpenDataSampleOpenOptions {
    vc3d::opendata::OpenDataResourceSelection selection;
    bool interactive{true};
};

enum class VolumeAttachmentSelection {
    PreserveCurrent,
    SelectAttached,
};

enum class VolumeAttachmentPresentation {
    Silent,
    Interactive,
};

enum class VolumeAttachmentPreparationFailure {
    None,
    NoProject,
    InvalidLocation,
    RemoteConfiguration,
};

struct VolumeAttachmentRequest {
    QString location;
    std::vector<std::string> tags;
    vc::HttpAuth auth;
    QString remoteCacheRoot;
    VolumeAttachmentSelection selection{VolumeAttachmentSelection::PreserveCurrent};
};

enum class VolumeAttachmentFailure {
    None,
    Load,
    ProjectChanged,
    VolumeIdConflict,
    Apply,
};

struct VolumeAttachmentOutcome {
    bool success{false};
    bool alreadyAttached{false};
    VolumeAttachmentFailure failure{VolumeAttachmentFailure::None};
    QString error;
    QString volumeId;
    QString location;
    QString projectPath;
};

class MenuActionController : public QObject
{
    Q_OBJECT

public:
    static constexpr int kMaxRecentVolpkg = 10;

    explicit MenuActionController(CWindow* window);
    ~MenuActionController() override;

    void populateMenus(QMenuBar* menuBar);
    void updateRecentVolpkgList(const QString& path);
    void removeRecentVolpkgEntry(const QString& path);
    void refreshRecentMenu();
    bool openVolpkgAt(const QString& path,
                      bool interactive = true,
                      QString* errorMessage = nullptr);
    void showOpenDataCatalog();
    bool isOpenDataCatalogVisible() const;
    // Resolves sampleId against the cached manifest and opens it. When
    // interactive is false, prompts, progress dialogs, and message boxes are
    // suppressed.
    bool openOpenDataSampleById(const QString& sampleId, bool interactive = true);

    // Opens a sample with explicit resource selection. resultOut receives the
    // attachment counters and messages when provided.
    bool openOpenDataSampleById(const QString& sampleId,
                                const OpenDataSampleOpenOptions& options,
                                QString* errorMessage = nullptr,
                                vc3d::opendata::OpenDataSampleProjectResult* resultOut = nullptr);

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
    // window/state, empty/unknown id, manifest unavailable, or an open already in
    // flight). Returns true once the background task started; exactly one GUI-thread
    // call to `onFinished` follows. `onProgress`, if set, is invoked on the GUI thread
    // with the same stream the interactive QProgressDialog consumes.
    // options.interactive is ignored; this entry point is always non-interactive.
    bool startOpenDataSampleOpen(
        const QString& sampleId,
        const OpenDataSampleOpenOptions& options,
        std::function<void(const OpenDataSampleOpenOutcome&)> onFinished,
        std::function<void(const vc3d::opendata::OpenDataSampleDownloadProgress&)> onProgress = {},
        QString* errorMessage = nullptr);

    // True while any sample open is in flight.
    bool openDataSampleOpenInFlight() const;
    bool prepareVolumeAttachment(
        const QString& location,
        std::vector<std::string> tags,
        VolumeAttachmentPresentation presentation,
        VolumeAttachmentRequest* request,
        QString* errorMessage = nullptr,
        VolumeAttachmentPreparationFailure* failure = nullptr);
    bool startVolumeAttachment(
        VolumeAttachmentRequest request,
        std::function<void(const VolumeAttachmentOutcome&)> onFinished,
        QString* errorMessage = nullptr);
    bool volumeAttachmentInFlight() const;

private slots:
    void newProject();
    void saveProjectAs();
    void attachVolume();
    void attachSegments();
    void attachNormalGrid();
    void detachEntry();
    void setOutputSegments();
    void convertLegacyVolpkg();
    void openVolpkg();
    void openRecentVolpkg();
    void attachRemoteZarr();
    void showSettingsDialog();
    void showAboutDialog();
    void showKeybindings();
    void resetSegmentationViews();
    void toggleConsoleOutput();
    void toggleDrawBBox(bool enabled);
    void toggleCursorMirroring(bool enabled);
    void surfaceFromSelection();
    void clearSelection();
    void importObjAsPatch();
    void beginRotateSurfaceTransform();
    void materializeCurrentOpenDataSegmentFolder();
    void exitApplication();

signals:
    // Emitted when the user picks Actions -> Merge tifxyz... CWindow
    // wires this to SegmentationCommandHandler::onMergeTifxyz with an
    // empty seed list so the dialog opens with an empty grid.
    void mergeTifxyzFromMenuRequested();
    // Emitted when the user picks Actions -> Patch tifxyz... CWindow
    // wires this to SegmentationCommandHandler::onMergePatch with an
    // empty seed list so the dialog opens with empty combo boxes.
    void mergePatchFromMenuRequested();
    void openDataCatalogVisibilityChanged(bool visible);

private:
    QStringList loadRecentPaths() const;
    void saveRecentPaths(const QStringList& paths);
    void ensureRecentActions();

    QStringList loadRecentRemoteUrls() const;
    void saveRecentRemoteUrls(const QStringList& urls);
    void updateRecentRemoteList(const QString& url);
    void attachRemoteZarrUrl(const QString& url);
    bool openOpenDataSample(const vc3d::opendata::OpenDataSample& sample,
                            bool interactive = true,
                            const vc3d::opendata::OpenDataResourceSelection* selection = nullptr,
                            QString* errorMessage = nullptr,
                            vc3d::opendata::OpenDataSampleProjectResult* resultOut = nullptr);
    // .cpp-local payload keeps the QtConcurrent result type out of this header.
    struct OpenDataOpenTaskResult;
    struct VolumeAttachmentTaskResult;
    // Launches the QtConcurrent task without a nested event loop. Its watcher is
    // parented to this; completion and onFinished run on the GUI thread.
    void beginOpenDataSampleOpenTask(
        const vc3d::opendata::OpenDataSample& sample,
        bool interactive,
        const vc3d::opendata::OpenDataResourceSelection* selection,
        std::function<void(const OpenDataSampleOpenOutcome&)> onFinished,
        std::function<void(const vc3d::opendata::OpenDataSampleDownloadProgress&)> onProgress);
    // Epilogue run on the GUI thread once the open task finishes: setVpkg, UI
    // refresh, prefill, status message, and (interactive-only) message boxes.
    void finishOpenDataSampleOpen(OpenDataOpenTaskResult task,
                                  bool interactive,
                                  OpenDataSampleOpenOutcome* outcomeOut);
    void startOpenDataVolumePrefill(const std::shared_ptr<Volume>& volume);
    void cancelOpenDataVolumePrefills();
    bool tryResolveRemoteAuth(const QString& url,
                              vc::HttpAuth* authOut,
                              QString* errorMessage = nullptr) const;
    // Runs vc_volpkg_convert against `inputLocation` (legacy folder or remote URL),
    // prompts the user for an output .volpkg.json, and returns the written path
    // via `convertedOut` on success.
    bool runLegacyVolpkgConvert(const QString& inputLocation, QString* convertedOut);
    QString remoteCacheDirectory(VolumeAttachmentPresentation presentation);
    QString configuredRemoteCacheDirectory() const;
    QString suggestedRemoteCacheDirectory() const;
    QString promptLocation(const QString& title,
                           const QString& hint,
                           const QString& defaultDir,
                           const QStringList& localFilters,
                           bool acceptFiles,
                           bool acceptDirs);

    CWindow* _window{nullptr};

    QMenu* _fileMenu{nullptr};
    QMenu* _editMenu{nullptr};
    QMenu* _viewMenu{nullptr};
    QMenu* _actionsMenu{nullptr};
    QMenu* _transformsMenu{nullptr};
    QMenu* _selectionMenu{nullptr};
    QMenu* _helpMenu{nullptr};
    QMenu* _recentMenu{nullptr};

    QAction* _newProjectAct{nullptr};
    QAction* _saveProjectAsAct{nullptr};
    QAction* _attachVolumeAct{nullptr};
    QAction* _attachSegmentsAct{nullptr};
    QAction* _attachNormalGridAct{nullptr};
    QAction* _detachEntryAct{nullptr};
    QAction* _setOutputSegmentsAct{nullptr};
    QAction* _convertLegacyAct{nullptr};
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
    QAction* _importObjAct{nullptr};
    QAction* _rotateSurfaceAct{nullptr};
    QAction* _mergeTifxyzAct{nullptr};
    QAction* _mergePatchAct{nullptr};
    QAction* _materializeOpenDataFolderAct{nullptr};
    QAction* _recalculateFiberScoresAct{nullptr};

    QPointer<QDialog> _keybindsDialog;
    QPointer<vc3d::opendata::OpenDataCatalogWindow> _openDataCatalogDialog;
    std::vector<QFutureWatcher<vc3d::opendata::OpenDataVolumePrefillResult>*> _openDataPrefillWatchers;
    std::vector<std::shared_ptr<std::atomic<bool>>> _openDataPrefillCancelFlags;
    std::shared_ptr<std::atomic<bool>> _openDataPrefillCancelFlag;
    // True from launch until the finished slot runs; prevents overlapping opens.
    bool _openDataSampleOpenInFlight{false};
    bool _volumeAttachmentInFlight{false};
};
