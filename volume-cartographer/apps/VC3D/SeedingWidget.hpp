#pragma once

#include <QWidget>
#include <optional>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QLabel>
#include <QProgressBar>
#include <QHBoxLayout>
#include <QLineEdit>
#include "SurfaceCatalog.hpp"
#include "ToolProcess.hpp"
#include <QPointer>
#include <QToolButton>
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
    // Relative-winding paths cross the store surfaces of the current volume.
    void setSurfaceCatalog(SurfaceCatalog* catalog);

    // Dialog-free operation entry points.
    void runResetPoints() { onResetPointsClicked(); }

    // Non-blocking batch entry points shared by the UI and automation. They
    // report validation failures through errorMessage and lifecycle through the
    // signals below. A true return means at least one child was launched.
    bool runSegmentationHeadless(QString* errorMessage);
    bool runExpandSeedsHeadless(QString* errorMessage);
    // Synchronous path-intensity analysis of the drawn paths.
    bool runAnalyzePathsHeadless(QString* errorMessage, int* pathsAnalyzed = nullptr,
                                 int* peaksFound = nullptr);
    // Cancels the active run/expand batch. Each child is asked to terminate,
    // then killed if it does not exit within one second.
    void cancelSeedingBatchHeadless();
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
    void onAnalyzePathsClicked();
    void onClearPeaksClicked();
    void onRunSegmentationClicked();
    void onExpandSeedsClicked();
    void onResetPointsClicked();
    void onCancelClicked();
    
private:
    void setupUI();
    void setDrawEnabled(bool enabled);
    void computeDistanceTransform();
    std::vector<float> sampleIntensities(const Volume& volume, const std::vector<cv::Vec3f>& points) const;
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
    // Shared child launch: one vc_grow_seg_from_seed run at low priority. Appends to
    // runningProcesses before start() so a failed start is still tracked.
    void launchBatchProcess(int index, const QStringList& toolArgs);
    // A child reaches its terminal state exactly once here (deduped by the tracker
    // on the child index), a failed start included, so it fails the batch rather
    // than stranding it.
    void handleBatchProcessFinished(vc3d::ToolProcess* run, int index, const vc3d::ToolProcess::Result& result);
    void finalizeSeedingBatch();
    // Relative winding annotation helpers
    void finalizePathLabelWraps(bool shiftHeld);
    void findPeaksAlongPathToCollection(const ViewerOverlayControllerBase::PathPrimitive& path, const std::string& collectionName);
    int findPatchIntersectionsAlongPathToCollection(const ViewerOverlayControllerBase::PathPrimitive& path, const std::string& collectionName);
    QColor generatePathColor();
    void displayPaths();
    void updateButtonStates();
    
private:
    
    // UI elements
    QLabel* infoLabel;
    QComboBox* collectionComboBox;
    QPushButton* drawPathsButton;
    QSpinBox* processesSpinBox;
    QSpinBox* ompThreadsSpinBox;
    QSpinBox* thresholdSpinBox;  // Intensity threshold for peak detection
    QSpinBox* expansionIterationsSpinBox; // Number of expansion iterations

    QString executablePath;
    
    QPushButton* analyzePathsButton;
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
    // The store vc_grow_seg_from_seed writes to: new surfaces go to the
    // attached store, bound to the current volume.
    std::optional<SurfaceCatalog::ToolStore> batchStore(QString* error) const;
    SurfaceCatalog* _surfaceCatalog{nullptr};
    QPointer<CChunkedVolumeViewer> _relWindingDrawViewer;
    int currentZSlice;
    VCCollection* _point_collection;
    cv::Mat distanceTransform;
    
    // Drawing data: while drawing is on, a left-drag in a viewer draws a path.
    bool _drawEnabled{false};
    QList<ViewerOverlayControllerBase::PathPrimitive> paths;  
    bool isDrawing;
    ViewerOverlayControllerBase::PathPrimitive currentPath;
    int colorIndex;
    bool labelWrapsMode = false; // special mode built on drawing; forces it on
    RelWindingIntersectionSource _relWindingIntersectionSource{RelWindingIntersectionSource::CurrentVolume};
    double _relWindingPatchTolerance{1.0};
    
    // Process management
    QList<QPointer<vc3d::ToolProcess>> runningProcesses;
    bool jobsRunning;

    // Async seeding batch state (run/expand), promoted from onRun/onExpand locals so the
    // Child finished callbacks read stable members, not dangling captures. Only one
    // batch is active at a time (gated by jobsRunning). Outcome aggregation lives in
    // SeedingBatchTracker _batch; only process-lifecycle bookkeeping stays here.
    int _batchNextIndex{0};
    int _batchOmpThreads{0};
    SeedingBatchTracker _batch;
    std::vector<ColPoint> _batchPoints;      // run: source points; empty for expand
    // Resolved vc_grow_seg_from_seed volume argument: local zarr path, or the remote
    // locator for a remote volume (issue #1188). QString,
    // not filesystem::path, since a remote locator is a URL.
    QString _batchVolumePath;
    SurfaceCatalog::ToolStore _batchStore;  // the children's store contract (new surfaces are tagged "paths")
    std::filesystem::path _batchConfigJson;  // seed.json (run) or expand.json (expand)
    QString _batchWorkingDir;
};
