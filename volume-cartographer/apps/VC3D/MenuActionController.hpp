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

// Options for a non-interactive (agent-bridge) Open Data sample open (SPEC §14.1).
// `selection` restricts which resources attach (SPEC §10.3; default = attach everything).
// `interactive` gates every blocking UI element (prompts, progress dialog, message
// boxes); the bridge always passes false.
struct OpenDataSampleOpenOptions {
    vc3d::opendata::OpenDataResourceSelection selection;
    bool interactive{true};
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
    void openVolpkgAt(const QString& path);
    void showOpenDataCatalog();
    bool isOpenDataCatalogVisible() const;
    // Headless twin of double-clicking a catalog sample; resolves `sampleId` against the
    // cached manifest and forwards to openOpenDataSample() (agent_bridge/SPEC.md §3.16).
    // When `interactive` is false (SPEC §8.2), the replace-project prompt is skipped (an
    // explicit call is treated as consent), the progress dialog isn't created, and message
    // boxes are suppressed — bridge handlers must never spin a nested event loop
    // (SPEC §1.3), so they always pass false.
    bool openOpenDataSampleById(const QString& sampleId, bool interactive = true);

    // Non-interactive-aware overload with resource selection (SPEC §10.3, §14.1);
    // forwards to openOpenDataSample() with `options`. `resultOut`, when provided,
    // receives the attach counters/messages used to report catalog.open_sample's
    // `attached` block.
    bool openOpenDataSampleById(const QString& sampleId,
                                const OpenDataSampleOpenOptions& options,
                                QString* errorMessage = nullptr,
                                vc3d::opendata::OpenDataSampleProjectResult* resultOut = nullptr);

    // Terminal outcome of an asynchronous Open Data sample open (SPEC §18.3).
    struct OpenDataSampleOpenOutcome {
        bool success{false};
        QString error;                                       // set iff !success
        vc3d::opendata::OpenDataSampleProjectResult result;  // attach counters/messages
    };

    // Asynchronous, always non-interactive open of a catalog sample (the §1.3-safe twin
    // of openOpenDataSampleById(sampleId, options, ...)); starts the open without
    // spinning a nested event loop and returns immediately.
    //
    // Returns false (with *errorMessage) on synchronous precondition failure (no
    // window/state, empty/unknown id, manifest unavailable, or an open already in
    // flight). Returns true once the background task started; exactly one GUI-thread
    // call to `onFinished` follows. `onProgress`, if set, is invoked on the GUI thread
    // with the same stream the interactive QProgressDialog consumes.
    // `options.interactive` is ignored (SPEC §8.2).
    bool startOpenDataSampleOpen(
        const QString& sampleId,
        const OpenDataSampleOpenOptions& options,
        std::function<void(const OpenDataSampleOpenOutcome&)> onFinished,
        std::function<void(const vc3d::opendata::OpenDataSampleDownloadProgress&)> onProgress = {},
        QString* errorMessage = nullptr);

    // True while any sample open (interactive or bridge-started) is in flight.
    bool openDataSampleOpenInFlight() const;

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
    // .cpp-local task payload for an async open; kept out of the header so it doesn't
    // leak the QtConcurrent result type (SPEC §18.3).
    struct OpenDataOpenTaskResult;
    // Launches the QtConcurrent open task with a heap QFutureWatcher parented to this (no
    // QEventLoop); its finished slot clears the in-flight flag, runs
    // finishOpenDataSampleOpen(), then calls onFinished on the GUI thread (SPEC §18.3).
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
                              bool allowPrompt,
                              QString* errorMessage = nullptr) const;
    // Runs vc_volpkg_convert against `inputLocation` (legacy folder or remote URL),
    // prompts the user for an output .volpkg.json, and returns the written path
    // via `convertedOut` on success.
    bool runLegacyVolpkgConvert(const QString& inputLocation, QString* convertedOut);
    QString remoteCacheDirectory(bool allowPrompt);
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
    // True from launch until the finished slot runs; guards against overlapping opens (SPEC §18.3).
    bool _openDataSampleOpenInFlight{false};
};
