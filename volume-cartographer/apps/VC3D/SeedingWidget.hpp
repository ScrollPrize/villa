#pragma once

#include <QWidget>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QLabel>
#include <QProgressBar>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QProcess>
#include <QPointer>
#include <QToolButton>
#include <QFutureWatcher>
#include <QSet>
#include <QHash>
#include <QStringList>
#include <opencv2/core/mat.hpp>
#include <memory>
#include <filesystem>
#include <vector>

#include "elements/ProgressUtil.hpp"
#include "overlays/ViewerOverlayControllerBase.hpp"
#include "SeedingBatchTracker.hpp"

using PathPrimitive = ViewerOverlayControllerBase::PathPrimitive;
#include "vc/core/types/VolumePkg.hpp"
#include "vc/ui/VCCollection.hpp"

class CChunkedVolumeViewer;
class CState;
class ViewerManager;
class SurfacePatchIndex;
class QuadSurface;

class SeedingWidget : public QWidget {
    Q_OBJECT
    
public:
    explicit SeedingWidget(VCCollection* point_collection, CState* state, QWidget* parent = nullptr);
    ~SeedingWidget();
    
    enum class RelWindingIntersectionSource {
        CurrentVolume = 0,
        Patches = 1
    };

    void setState(CState* state);
    void setViewerManager(ViewerManager* viewerManager);

    // --- Agent bridge headless entry points (SPEC §15.2) ---
    // The seeding action entry points are private slots; these one-line public
    // wrappers let the agent bridge invoke the non-blocking ones without a Qt
    // signal connection. preview-rays (synchronous point math), cast-rays
    // (QtConcurrent background), and reset-points (pure UI reset) are simple
    // wrappers. setDialogsSuppressed() gates the precondition QMessageBox::warning
    // calls the wrapped slots would otherwise raise (a static QMessageBox spins a
    // nested event loop), for the bridge's lifetime only.
    void runPreviewRays() { onPreviewRaysClicked(); }
    void runCastRays() { onCastRaysClicked(); }
    void runResetPoints() { onResetPointsClicked(); }
    void setDialogsSuppressed(bool suppressed) { _dialogsSuppressed = suppressed; }
    [[nodiscard]] bool dialogsSuppressed() const { return _dialogsSuppressed; }

    // --- Agent bridge headless entry points for the batch seeding actions ---
    // (SPEC §15.2, as amended). onRunSegmentationClicked / onExpandSeedsClicked
    // used to end in a blocking `while (jobsRunning) QApplication::processEvents`
    // loop (a §1.3 violation). They now launch the QProcess batch and return; the
    // batch drains through the process finished callbacks and reports progress /
    // completion via seedingBatchProgressChanged / seedingBatchFinished. These
    // headless twins run the exact same validation + launch but report precondition
    // failures through *errorMessage (never a QMessageBox) and never block. Distinct
    // names (not overloads of the connect()-target slots), per the headless-split
    // doctrine. Return true when the batch was accepted (≥1 child launched).
    bool runSegmentationHeadless(QString* errorMessage);
    bool runExpandSeedsHeadless(QString* errorMessage);
    // Synchronous path-intensity analysis (Draw mode). Pure in-process compute; the
    // only reason the slot wasn't bridge-safe was a per-path QApplication::
    // processEvents repaint pump, now removed. Reports counts via the optional
    // out-params.
    bool runAnalyzePathsHeadless(QString* errorMessage, int* pathsAnalyzed = nullptr,
                                 int* peaksFound = nullptr);
    // Dialog-free cancel of the active run/expand (or neural-trace) batch. Bounded:
    // terminate() + waitForFinished(1000) then kill() per running child, so it is
    // safe to call synchronously from a bridge RPC handler. Fires
    // seedingBatchFinished(kind,false,...) when a run/expand batch was active.
    void cancelSeedingBatchHeadless();
    // Live batch introspection for the bridge job model (SPEC §8.3). "active" means
    // a run/expand batch specifically (not a neural trace, which shares the
    // jobsRunning flag). kind is "run" | "expand" | "" (idle).
    [[nodiscard]] bool seedingBatchActive() const { return jobsRunning && !_batch.kind().isEmpty(); }
    [[nodiscard]] QString seedingBatchKind() const { return _batch.kind(); }
    [[nodiscard]] int seedingBatchTotal() const { return _batch.total(); }

signals:
    void sendPathsChanged(const QList<ViewerOverlayControllerBase::PathPrimitive>& paths);
    void sendStatusMessageAvailable(QString text, int timeout);
    void relWindingAnnotationModeChanged(bool active);
    // Batch seeding lifecycle (SPEC §15.2). kind is "run" | "expand". Emitted from
    // the QProcess finished callbacks so the agent bridge can mirror the batch as a
    // source:"seeding" job: progress on each child completion, finished on
    // completion, failure, or cancel. success is true only when the whole batch
    // ran clean (no child failed and no cancel); canceled distinguishes a user
    // cancel from an execution failure (both map to a failed bridge job). message
    // carries meaningful terminal text (see finalizeSeedingBatch).
    void seedingBatchProgressChanged(const QString& kind, int completed, int total);
    void seedingBatchFinished(const QString& kind, bool success, bool canceled,
                              int completed, int total, const QString& message);
    
public slots:
    void onSurfacesLoaded();  // Called when surfaces have been loaded/reloaded
    void onCollectionsAdded(const std::vector<uint64_t>& collectionIds);
    void onCollectionChanged(uint64_t collectionId);
    void onCollectionRemoved(uint64_t collectionId);
    
public slots:
    void onVolumeChanged(std::shared_ptr<Volume> vol, const std::string& volumeId);
    void updateCurrentZSlice(int z);
    void onMousePress(cv::Vec3f vol_point, cv::Vec3f normal, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void onMouseMove(cv::Vec3f vol_point, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);
    void onMouseRelease(cv::Vec3f vol_point, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void setRelWindingAnnotationMode(bool active);
    void setRelWindingIntersectionSource(int source);
    void setRelWindingPatchTolerance(double tolerance);
    
private slots:
    void onPreviewRaysClicked();
    void onClearPreviewClicked();
    void onCastRaysClicked();
    void onCastRaysFinished();
    void onClearPeaksClicked();
    void onRunSegmentationClicked();
    void onExpandSeedsClicked();
    void onResetPointsClicked();
    void onCancelClicked();
    void onNeuralTraceClicked();
    void onNeuralCheckpointBrowseClicked();
    
private:
    // Mode enum
    enum class Mode {
        PointMode,
        DrawMode
    };
    
    void setupUI();
    void computeDistanceTransform();
    void castRays();
    void findPeaksAlongRay(const cv::Vec2f& rayDir, const cv::Vec3f& startPoint);
    void runSegmentation();
    QString findExecutablePath();
    QString findNeuralTracePyPath();
    QString findPythonExecutable();
    void updateParameterPreview();
    void updateModeUI();
    void analyzePaths();
    void findPeaksAlongPath(const ViewerOverlayControllerBase::PathPrimitive& path);
    void startDrawing(cv::Vec3f startPoint);
    void addPointToPath(cv::Vec3f point);
    void finalizePath();
    // Batch seeding internals (non-blocking replacements for the former
    // self-referencing local lambdas in onRun/onExpand — those lambdas captured
    // stack locals by reference, which would dangle now that the launching call
    // returns before the batch finishes). One batch (run OR expand) at a time.
    void startSegmentationProcessForPoint(int pointIndex);
    void startExpansionProcessForIteration(int iterationIndex);
    // Shared child launch: drains merged output (bounded tail), wires the
    // finished + errorOccurred(FailedToStart) completion paths, resolves the
    // nice/ionice priority wrappers (which may be absent) and falls back to
    // launching vc_grow_seg_from_seed directly. Appends to runningProcesses
    // before start() so a synchronous FailedToStart is still tracked.
    void launchBatchProcess(QProcess* process, int index, const QStringList& toolArgs);
    // A single QProcess reaches its terminal state exactly once here (deduped by
    // the tracker keyed on process identity): finished() and errorOccurred() may
    // both fire.
    // failedToStart routes FailedToStart through the same completion path so a
    // missing wrapper/tool fails the batch instead of stranding it.
    void handleBatchProcessFinished(QProcess* process, int index, int exitCode,
                                    QProcess::ExitStatus exitStatus, bool failedToStart);
    void finalizeSeedingBatch();
    // Relative winding annotation helpers
    void finalizePathLabelWraps(bool shiftHeld);
    void findPeaksAlongPathToCollection(const ViewerOverlayControllerBase::PathPrimitive& path, const std::string& collectionName);
    int findPatchIntersectionsAlongPathToCollection(const ViewerOverlayControllerBase::PathPrimitive& path, const std::string& collectionName);
    std::vector<std::shared_ptr<QuadSurface>> relWindingPatchSurfaces() const;
    QColor generatePathColor();
    void displayPaths();
    void updatePointsDisplay();
    void updateInfoLabel();
    void updateButtonStates();
    
private:
    
    // UI elements
    QLabel* infoLabel;
    QComboBox* collectionComboBox;
    QDoubleSpinBox* angleStepSpinBox;
    QSpinBox* processesSpinBox;
    QSpinBox* ompThreadsSpinBox;
    QSpinBox* thresholdSpinBox;  // Intensity threshold for peak detection
    QSpinBox* windowSizeSpinBox; // Window size for peak detection
    QSpinBox* maxRadiusSpinBox;  // Max radius for ray casting
    QSpinBox* expansionIterationsSpinBox; // Number of expansion iterations
    
    // Layout and label references for hiding/showing
    QHBoxLayout* maxRadiusLayout;
    QLabel* maxRadiusLabel;
    QHBoxLayout* angleStepLayout;
    QLabel* angleStepLabel;
    
    QString executablePath;
    
    QPushButton* previewRaysButton;
    QPushButton* clearPreviewButton;
    QPushButton* castRaysButton;
    QPushButton* clearPeaksButton;
    QPushButton* runSegmentationButton;
    QPushButton* expandSeedsButton;
    QPushButton* resetPointsButton;
    QPushButton* cancelButton;
    QProgressBar* progressBar;
    ProgressUtil* progressUtil;
    
    // Data
    bool _dialogsSuppressed{false};  // agent bridge suppresses precondition dialogs
    CState* _state{nullptr};
    ViewerManager* _viewerManager{nullptr};
    QPointer<CChunkedVolumeViewer> _relWindingDrawViewer;
    int currentZSlice;
    VCCollection* _point_collection;
    cv::Mat distanceTransform;
    
    // Drawing mode data
    Mode currentMode;
    QList<ViewerOverlayControllerBase::PathPrimitive> paths;  
    bool isDrawing;
    ViewerOverlayControllerBase::PathPrimitive currentPath;
    int colorIndex;
    bool labelWrapsMode = false; // special mode built on DrawMode
    RelWindingIntersectionSource _relWindingIntersectionSource{RelWindingIntersectionSource::CurrentVolume};
    double _relWindingPatchTolerance{1.0};
    
    // Process management
    QList<QPointer<QProcess>> runningProcesses;
    bool jobsRunning;

    // Async seeding batch state (run/expand). Promoted from onRun/onExpand locals
    // so the QProcess finished callbacks read stable members, not dangling
    // captures. Only one batch is active at a time (gated by jobsRunning).
    // Outcome aggregation (failure/cancel/success + terminal message) lives in
    // SeedingBatchTracker _batch; only genuine process-lifecycle bookkeeping
    // (scheduling cursor, config, per-child output tail) stays here.
    int _batchNextIndex{0};
    int _batchOmpThreads{0};
    SeedingBatchTracker _batch;                  // honest batch outcome (SPEC §1)
    QHash<QProcess*, QString> _batchProcessTail; // bounded per-child output tail for diagnostics
    std::vector<ColPoint> _batchPoints;      // run: source points; empty for expand
    // Resolved vc_grow_seg_from_seed volume argument: local zarr path for a
    // mirrored volume, or the remote locator for a streaming-only volume
    // (issue #1188). A QString (not filesystem::path) because a remote locator
    // is a URL, not a path.
    QString _batchVolumePath;
    std::filesystem::path _batchPathsDir;
    std::filesystem::path _batchConfigJson;  // seed.json (run) or expand.json (expand)
    QString _batchWorkingDir;

    // Async cast rays
    QFutureWatcher<void>* _castRaysWatcher{nullptr};
    std::vector<cv::Vec3f> _castRaysPeaks;  // Collected by background thread
    bool _castRaysWasPointMode{true};  // Track which mode triggered the cast

    // Neural trace UI and state
    QLineEdit* _neuralCheckpointEdit{nullptr};
    QToolButton* _neuralCheckpointBrowse{nullptr};
    QLineEdit* _neuralPythonEdit{nullptr};
    QToolButton* _neuralPythonBrowse{nullptr};
    QComboBox* _comboNeuralVolumeScale{nullptr};
    QSpinBox* _spinNeuralMaxSize{nullptr};
    QSpinBox* _spinNeuralStepsPerCrop{nullptr};
    QPushButton* _btnNeuralTrace{nullptr};
    QString _neuralCheckpointPath;
    QString _neuralPythonPath;
    int _neuralVolumeScale{0};
    int _neuralMaxSize{60};
    int _neuralStepsPerCrop{1};
};
