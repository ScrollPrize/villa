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

    // Dialog-free operation entry points.
    bool previewRaysHeadless(QString* errorMessage = nullptr);
    bool castRaysHeadless(QString* errorMessage = nullptr);
    void runResetPoints() { onResetPointsClicked(); }

    // Non-blocking batch entry points shared by the UI and automation. They
    // report validation failures through errorMessage and lifecycle through the
    // signals below. A true return means at least one child was launched.
    bool runSegmentationHeadless(QString* errorMessage);
    bool runExpandSeedsHeadless(QString* errorMessage);
    // Synchronous path-intensity analysis in Draw mode.
    bool runAnalyzePathsHeadless(QString* errorMessage, int* pathsAnalyzed = nullptr,
                                 int* peaksFound = nullptr);
    // Cancels the active run/expand or neural-trace batch. Each child is asked
    // to terminate, then killed if it does not exit within one second.
    void cancelSeedingBatchHeadless();
    // A neural trace shares jobsRunning but is not a seeding batch.
    [[nodiscard]] bool seedingBatchActive() const { return jobsRunning && !_batch.kind().isEmpty(); }
    [[nodiscard]] int seedingBatchTotal() const { return _batch.total(); }

signals:
    void sendPathsChanged(const QList<ViewerOverlayControllerBase::PathPrimitive>& paths);
    void sendStatusMessageAvailable(QString text, int timeout);
    void relWindingAnnotationModeChanged(bool active);
    // Batch lifecycle. kind is "run" or "expand"; success requires every child
    // to finish cleanly, and canceled distinguishes cancellation from failure.
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
    // Batch seeding internals; replace the former self-referencing local lambdas in
    // onRun/onExpand, whose captured stack locals would dangle now that the launching
    // call returns before the batch finishes. One batch (run OR expand) at a time.
    void startSegmentationProcessForPoint(int pointIndex);
    void startExpansionProcessForIteration(int iterationIndex);
    // Shared child launch: drains merged output (bounded tail), wires finished +
    // errorOccurred(FailedToStart), and falls back to vc_grow_seg_from_seed directly if
    // the nice/ionice wrappers are absent. Appends to runningProcesses before start() so a
    // synchronous FailedToStart is still tracked.
    void launchBatchProcess(QProcess* process, int index, const QStringList& toolArgs);
    // A single QProcess reaches its terminal state exactly once here (deduped by the
    // tracker keyed on process identity) since finished() and errorOccurred() may both
    // fire. failedToStart routes FailedToStart through the same path so a missing
    // wrapper/tool fails the batch rather than stranding it.
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

    // Async seeding batch state (run/expand), promoted from onRun/onExpand locals so the
    // QProcess finished callbacks read stable members, not dangling captures. Only one
    // batch is active at a time (gated by jobsRunning). Outcome aggregation lives in
    // SeedingBatchTracker _batch; only process-lifecycle bookkeeping stays here.
    int _batchNextIndex{0};
    int _batchOmpThreads{0};
    SeedingBatchTracker _batch;
    QHash<QProcess*, QString> _batchProcessTail; // bounded per-child output tail for diagnostics
    std::vector<ColPoint> _batchPoints;      // run: source points; empty for expand
    // Resolved vc_grow_seg_from_seed volume argument: local zarr path for a mirrored
    // volume, or the remote locator for a streaming-only volume (issue #1188). QString,
    // not filesystem::path, since a remote locator is a URL.
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
