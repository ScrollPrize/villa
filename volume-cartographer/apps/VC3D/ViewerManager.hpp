#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QFutureWatcher>

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "vc/core/util/Compositing.hpp"
#include "vc/core/types/Sampling.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

class QMdiArea;
class QTimer;
class AxisAlignedSliceController;
class CChunkedVolumeViewer;
class CState;
struct POI;
class QWidget;
class VCCollection;
class SegmentationOverlayController;
class PointsOverlayController;
class RawPointsOverlayController;
class PathsOverlayController;
class BBoxOverlayController;
class VectorOverlayController;
class VolumeOverlayController;
class InkDetectionOverlayController;
class SegmentationModule;
class ViewerOverlayControllerBase;
class VolumeViewerBase;
class Volume;
class Surface;
class QuadSurface;

class ViewerManager : public QObject
{
    Q_OBJECT

public:
    enum class ViewerRole {
        Standard,
        Annotation,
    };

    ViewerManager(CState* state,
                  VCCollection* points,
                  QObject* parent = nullptr);
    ~ViewerManager() override;

    // All live managers (one per workspace). Use to apply user preferences
    // uniformly so viewers behave the same across workspaces.
    static const std::vector<ViewerManager*>& allManagers();

    VolumeViewerBase* createViewer(const std::string& surfaceName,
                                   const QString& title,
                                   QMdiArea* mdiArea,
                                   ViewerRole role = ViewerRole::Standard);
    VolumeViewerBase* createViewerInWidget(const std::string& surfaceName,
                                           QWidget* parent,
                                           ViewerRole role = ViewerRole::Standard);
    void unregisterViewer(VolumeViewerBase* viewer);

    const std::vector<VolumeViewerBase*>& baseViewers() const { return _baseViewers; }

    void setSegmentationOverlay(SegmentationOverlayController* overlay);
    SegmentationOverlayController* segmentationOverlay() const { return _segmentationOverlay; }
    void setSegmentationEditActive(bool active);
    void setSegmentationModule(SegmentationModule* module);
    void setPointsOverlay(PointsOverlayController* overlay);
    void setRawPointsOverlay(RawPointsOverlayController* overlay);
    RawPointsOverlayController* rawPointsOverlay() const { return _rawPointsOverlay; }
    void setPathsOverlay(PathsOverlayController* overlay);
    void setBBoxOverlay(BBoxOverlayController* overlay);
    void setVectorOverlay(VectorOverlayController* overlay);
    void setVolumeOverlay(VolumeOverlayController* overlay);
    void setInkDetectionOverlay(InkDetectionOverlayController* overlay);
    InkDetectionOverlayController* inkDetectionOverlay() const { return _inkDetectionOverlay; }

    void setIntersectionOpacity(float opacity);
    float intersectionOpacity() const { return _intersectionOpacity; }

    void setOverlayVolume(std::shared_ptr<Volume> volume, const std::string& volumeId);
    std::shared_ptr<Volume> overlayVolume() const { return _overlayVolume; }
    const std::string& overlayVolumeId() const { return _overlayVolumeId; }
    std::shared_ptr<Volume> currentVolume() const;
    std::string currentVolumeId() const;

    void setOverlayOpacity(float opacity);
    float overlayOpacity() const { return _overlayOpacity; }

    void setOverlayColormap(const std::string& colormapId);
    const std::string& overlayColormap() const { return _overlayColormapId; }
    void setOverlaySamplingMethod(vc::Sampling method);
    vc::Sampling overlaySamplingMethod() const { return _overlaySamplingMethod; }
    void setOverlayThreshold(float threshold);
    float overlayThreshold() const { return _overlayWindowLow; }

    void setOverlayWindow(float low, float high);
    float overlayWindowLow() const { return _overlayWindowLow; }
    float overlayWindowHigh() const { return _overlayWindowHigh; }
    void setOverlayMaxDisplayedResolution(int level);
    int overlayMaxDisplayedResolution() const { return _overlayMaxDisplayedResolution; }
    void setOverlayComposite(const OverlayCompositeSettings& settings);
    const OverlayCompositeSettings& overlayComposite() const { return _overlayComposite; }

    void setVolumeWindow(float low, float high);
    float volumeWindowLow() const { return _volumeWindowLow; }
    float volumeWindowHigh() const { return _volumeWindowHigh; }

    void setSurfacePatchSamplingStride(int stride, bool userInitiated = true);
    int surfacePatchSamplingStride() const { return _surfacePatchSamplingStride; }
    void setIntersectionMaxSurfaces(int limit);
    int intersectionMaxSurfaces() const { return _intersectionMaxSurfaces; }
    void primeSurfacePatchIndicesAsync();
    void resetStrideUserOverride() {}

    // Folder-selection keyed retention of built surface patch indexes. The
    // surface panel sets the key before rebinding surfaces on a folder
    // switch; the live index is stashed under the outgoing key and swapped
    // back in (if still valid) when the user returns to that selection.
    void setSurfacePatchIndexCacheKey(const QString& key);
    void clearSurfacePatchIndexCache();

    bool resetDefaultFor(VolumeViewerBase* viewer) const;
    void setResetDefaultFor(VolumeViewerBase* viewer, bool value);

    // Registered automatically by AxisAlignedSliceController::setViewerManager.
    void setAxisAlignedSliceController(AxisAlignedSliceController* slices) { _slices = slices; }
    AxisAlignedSliceController* axisAlignedSliceController() const { return _slices; }

    // --- Focus / navigation policy shared by all workspaces ---
    // Ensure the "focus" POI exists and lies inside the current volume: a
    // missing POI (or resetToCenter) is placed at the volume center, an
    // existing one is clamped to the volume bounds. overridePoint /
    // overrideNormal (already in the new volume's space) win when provided.
    // Returns false when no volume is set.
    bool resetFocusForVolumeChange(bool resetToCenter,
                                   const std::optional<cv::Vec3f>& overridePoint = std::nullopt,
                                   const std::optional<cv::Vec3f>& overrideNormal = std::nullopt);
    // Move the focus POI to a volume position and reorient the slice planes
    // around it (also nudges the segmentation viewer when the point projects
    // onto the active surface).
    bool centerFocusAt(const cv::Vec3f& position, const cv::Vec3f& normal, const std::string& sourceId);
    // centerFocusAt() at the volume position under the mouse cursor: prefers
    // the viewer under the cursor, then scans all visible viewer viewports.
    bool centerFocusOnCursor();
    bool recenterViewersOnCurrentFocus();
    void recenterPlaneViewersOn(const cv::Vec3f& position);
    void recenterSegmentationViewerNear(const cv::Vec3f& position);
    VolumeViewerBase* segmentationViewer() const;

    // Last viewer the user interacted with (mouse press / zoom / cursor move);
    // null when none or while it is being torn down.
    VolumeViewerBase* activeViewer() const;

    // Runs before the default volume-click policy (Shift ignored, Ctrl+click
    // centers the focus); return true to consume the click.
    using VolumeClickInterceptor = std::function<bool(const cv::Vec3f& volLoc,
                                                      const cv::Vec3f& normal,
                                                      Surface* surf,
                                                      Qt::MouseButton button,
                                                      Qt::KeyboardModifiers modifiers)>;
    void setVolumeClickInterceptor(VolumeClickInterceptor interceptor) { _volumeClickInterceptor = std::move(interceptor); }

    // --- Volume-switch policy shared by all workspaces ---
    // Snapshot of per-viewer cameras and view centers, retargetable through an
    // affine coordinate transform after a volume switch.
    struct ViewerNavigationSnapshot;
    std::shared_ptr<ViewerNavigationSnapshot> captureNavigation() const;
    void restoreNavigation(const std::shared_ptr<ViewerNavigationSnapshot>& snapshot,
                           const cv::Matx44d& transform);
    // Set the volume on the state, re-derive the focus POI (volume-centered
    // when new/absent, clamped otherwise, transformed when a coordinate
    // transform is supplied) and retarget the captured viewer navigation.
    void switchVolume(std::shared_ptr<Volume> volume,
                      const std::optional<cv::Matx44d>& navigationTransform = std::nullopt);

    // --- Fleet setters: apply a viewer preference to every viewer ---
    void setShowDirectionHints(bool show);
    void setShowSurfaceNormals(bool show);
    void setPlaneIntersectionLinesVisible(bool visible);
    void setSurfaceOverlaysEnabled(bool enabled);
    void setSurfaceOverlapThreshold(float threshold);
    void setSurfaceOverlays(const std::map<std::string, cv::Vec3b>& overlays);

    // --- Reset-view-on-surface-change policy ---
    // Store the new default on every viewer, keeping the segmentation viewer
    // suppressed while a segmentation edit session is active.
    void setResetViewOnSurfaceChangeDefault(bool enabled);
    // Temporarily force the segmentation viewers off (true) or restore their
    // stored defaults (false).
    void setSegmentationResetViewSuppressed(bool suppressed);

    void setSegmentationCursorMirroring(bool enabled);
    bool segmentationCursorMirroring() const { return _mirrorCursorToSegmentation; }
    void broadcastLinkedCursor(VolumeViewerBase* source,
                               const std::optional<cv::Vec3f>& point);

    void setZScrollSensitivity(double sensitivity);
    double zScrollSensitivity() const { return _zScrollSensitivity; }

    void forEachBaseViewer(const std::function<void(VolumeViewerBase*)>& fn) const;
    void setIntersectionThickness(float thickness);
    float intersectionThickness() const { return _intersectionThickness; }
    void setHighlightedSurfaceIds(const std::vector<std::string>& ids);
    SurfacePatchIndex* surfacePatchIndex();
    SurfacePatchIndex* surfacePatchIndexIfReady();
    SurfacePatchIndex* activeSegmentationEditSurfacePatchIndex() const;
    void refreshSurfacePatchIndex(const SurfacePatchIndex::SurfacePtr& surface);
    void refreshSurfacePatchIndex(const SurfacePatchIndex::SurfacePtr& surface, const cv::Rect& changedRegion);

    // Stop maintaining the SurfacePatchIndex. Any subsequent
    // surfaceWillBeDeleted signals will be ignored instead of triggering
    // an O(N) rtree removal. Intended to be called from the close path so
    // that tearing down thousands of cells doesn't block app exit.
    void beginShutdown() noexcept { _shuttingDown = true; }

signals:
    void baseViewerCreated(VolumeViewerBase* viewer);
    void baseViewerClosing(VolumeViewerBase* viewer);
    // Emitted whenever the user explicitly places the focus (Ctrl+click,
    // focus-on-cursor key, point activation, ...).
    void focusCenteredByUser(const cv::Vec3f& position);
    // Aggregated per-viewer cache statistics (RAM / disk / network).
    void sharedCacheStatsChanged(const QStringList& items);
    void overlayWindowChanged(float low, float high);
    void volumeWindowChanged(float low, float high);
    void overlayVolumeAvailabilityChanged(bool hasOverlay);
    void samplingStrideChanged(int stride);
    void zScrollSensitivityChanged(double sensitivity);

private slots:
    void onGlobalTick();
    void handleFocusPoiChanged(std::string name, POI* poi);
    void handleVolumeClicked(cv::Vec3f volLoc, cv::Vec3f normal, Surface* surf,
                             Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void handleSurfacePatchIndexPrimeFinished();
    void handleSurfacePatchIndexTaskFinished();
    void handleSurfaceChanged(std::string name, std::shared_ptr<Surface> surf, bool isEditUpdate = false);
    void handleSurfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf);

private:
    enum class SurfacePatchIndexTaskType {
        Update,
        Remove,
    };

    struct SurfacePatchIndexTask {
        SurfacePatchIndexTaskType type{SurfacePatchIndexTaskType::Update};
        std::string id;
        SurfacePatchIndex::SurfacePtr surface;
    };

    struct SurfacePatchIndexTaskResult {
        SurfacePatchIndexTaskType type{SurfacePatchIndexTaskType::Update};
        std::string id;
        SurfacePatchIndex::SurfacePtr surface;
        bool success{false};
    };

    void registerOverlay(ViewerOverlayControllerBase* overlay);
    VolumeViewerBase* initializeChunkedViewer(CChunkedVolumeViewer* chunkedViewer,
                                              const std::string& surfaceName,
                                              ViewerRole role);
    bool updateSurfacePatchIndexForSurface(const SurfacePatchIndex::SurfacePtr& quad, bool isEditUpdate);
    void queueSurfacePatchIndexTask(SurfacePatchIndexTask task);
    void startNextSurfacePatchIndexTask();

    CState* _state;
    VCCollection* _points;
    AxisAlignedSliceController* _slices{nullptr};
    SegmentationOverlayController* _segmentationOverlay{nullptr};
    PointsOverlayController* _pointsOverlay{nullptr};
    RawPointsOverlayController* _rawPointsOverlay{nullptr};
    PathsOverlayController* _pathsOverlay{nullptr};
    BBoxOverlayController* _bboxOverlay{nullptr};
    VectorOverlayController* _vectorOverlay{nullptr};
    // All overlay controllers that should be attached/detached from viewers.
    // Populated by the set*Overlay() methods. Does NOT include VolumeOverlayController
    // (which is not a ViewerOverlayControllerBase subclass).
    std::vector<ViewerOverlayControllerBase*> _allOverlays;
    bool _segmentationEditActive{false};
    SegmentationModule* _segmentationModule{nullptr};
    std::vector<VolumeViewerBase*> _baseViewers;
    VolumeViewerBase* _activeViewer{nullptr};
    VolumeClickInterceptor _volumeClickInterceptor;
    // The one maintenance clock for the whole app. Ticks ~60Hz; render requests
    // submit immediately, while deferred intersections/status are serviced here.
    QTimer* _globalClock{nullptr};
    std::unordered_map<VolumeViewerBase*, bool> _resetDefaults;
    float _intersectionOpacity{1.0f};
    float _intersectionThickness{0.0f};
    std::shared_ptr<Volume> _overlayVolume;
    std::string _overlayVolumeId;
    float _overlayOpacity{0.5f};
    std::string _overlayColormapId;
    vc::Sampling _overlaySamplingMethod{vc::Sampling::Nearest};
    float _overlayWindowLow{0.0f};
    float _overlayWindowHigh{255.0f};
    int _overlayMaxDisplayedResolution{0};
    OverlayCompositeSettings _overlayComposite;
    float _volumeWindowLow{0.0f};
    float _volumeWindowHigh{255.0f};
    bool _mirrorCursorToSegmentation{false};
    double _zScrollSensitivity{1.0};
    int _surfacePatchSamplingStride{1};
    std::atomic<bool> _shuttingDown{false};
    int _intersectionMaxSurfaces{0};  // 0 = unlimited

    VolumeOverlayController* _volumeOverlay{nullptr};
    InkDetectionOverlayController* _inkDetectionOverlay{nullptr};
    SurfacePatchIndex _surfacePatchIndex;
    struct CachedSurfacePatchIndex {
        SurfacePatchIndex index;
        std::unordered_set<std::string> ids;
    };
    // Indexes stashed on folder-selection switches, keyed by the selection
    // they were built for. Entries are validated against the current surface
    // instances (and stride) before being swapped back in.
    std::map<QString, CachedSurfacePatchIndex> _surfacePatchIndexCache;
    QString _surfacePatchIndexCacheKey;
    void invalidateSurfacePatchIndexCacheFor(const SurfacePatchIndex::SurfacePtr& surface);
    bool _surfacePatchIndexNeedsRebuild{true};
    // Use string IDs for surface tracking to avoid dangling pointers in async operations
    std::unordered_set<std::string> _indexedSurfaceIds;
    std::vector<std::string> _pendingSurfacePatchIndexSurfaceIds;
    std::vector<SurfacePatchIndexTask> _pendingSurfacePatchIndexTasks;
    std::vector<SurfacePatchIndexTask> _surfacesQueuedDuringRebuild;
    QFutureWatcher<std::shared_ptr<SurfacePatchIndex>>* _surfacePatchIndexWatcher{nullptr};
    QFutureWatcher<SurfacePatchIndexTaskResult>* _surfacePatchIndexTaskWatcher{nullptr};

    // Surfaces currently pinned in the LRU as "highlighted/visible".
    // We track them so we can unpin the right set when highlights change.
    std::vector<std::shared_ptr<QuadSurface>> _pinnedHighlightSurfaces;

    void rebuildSurfacePatchIndexIfNeeded();

public:
    // Async-intersection mutual exclusion. A viewer's plane-intersection query reads
    // _surfacePatchIndex on a worker thread; while any such read is in flight the
    // index must not be mutated (rebuild swap / updateSurface / clear / single-surface
    // task) or the worker tears. Mutation sites consult _indexReadsInFlight and defer
    // (mark dirty / stash result / hold task); endIndexRead() applies the deferred
    // work once reads drain. Begin/end bracket each worker read on the main thread.
    void beginIndexRead() { ++_indexReadsInFlight; }
    void endIndexRead();
    bool indexReadInFlight() const { return _indexReadsInFlight > 0; }
private:
    int _indexReadsInFlight{0};
    std::shared_ptr<SurfacePatchIndex> _deferredIndexSwap;
    std::vector<std::string> _deferredIndexSwapIds;
};
