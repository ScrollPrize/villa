#pragma once

#include <QObject>
#include <QString>
#include <QFutureWatcher>

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <opencv2/core.hpp>

#include "vc/core/util/SurfacePatchIndex.hpp"
#include "tiled/RenderPool.hpp"

class QMdiArea;
class CTiledVolumeViewer;
class CState;
class VCCollection;
class SegmentationOverlayController;
class PointsOverlayController;
class RawPointsOverlayController;
class PathsOverlayController;
class BBoxOverlayController;
class VectorOverlayController;
class VolumeOverlayController;
class SegmentationModule;
class ViewerOverlayControllerBase;
class Volume;
class Surface;
class QuadSurface;

class ViewerManager : public QObject
{
    Q_OBJECT

public:
    ViewerManager(CState* state,
                  VCCollection* points,
                  QObject* parent = nullptr);

    CTiledVolumeViewer* createViewer(const std::string& surfaceName,
                                     const QString& title,
                                     QMdiArea* mdiArea);

    const std::vector<CTiledVolumeViewer*>& viewers() const { return _viewers; }
    RenderPool* renderPool() { return &_renderPool; }

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

    void setIntersectionOpacity(float opacity);
    float intersectionOpacity() const { return _intersectionOpacity; }

    void setOverlayVolume(std::shared_ptr<Volume> volume, const std::string& volumeId);
    std::shared_ptr<Volume> overlayVolume() const { return _overlayVolume; }
    const std::string& overlayVolumeId() const { return _overlayVolumeId; }

    void setOverlayOpacity(float opacity);
    float overlayOpacity() const { return _overlayOpacity; }

    void setOverlayColormap(const std::string& colormapId);
    const std::string& overlayColormap() const { return _overlayColormapId; }
    void setOverlayThreshold(float threshold);
    float overlayThreshold() const { return _overlayWindowLow; }

    void setOverlayWindow(float low, float high);
    float overlayWindowLow() const { return _overlayWindowLow; }
    float overlayWindowHigh() const { return _overlayWindowHigh; }

    void setVolumeWindow(float low, float high);
    float volumeWindowLow() const { return _volumeWindowLow; }
    float volumeWindowHigh() const { return _volumeWindowHigh; }

    void setSurfacePatchSamplingStride(int stride, bool userInitiated = true);
    int surfacePatchSamplingStride() const { return _surfacePatchSamplingStride; }
    void primeSurfacePatchIndicesAsync();
    void resetStrideUserOverride() { _surfacePatchStrideUserSet = false; }

    bool resetDefaultFor(CTiledVolumeViewer* viewer) const;
    void setResetDefaultFor(CTiledVolumeViewer* viewer, bool value);

    void setSegmentationCursorMirroring(bool enabled);
    bool segmentationCursorMirroring() const { return _mirrorCursorToSegmentation; }

    void setSliceStepSize(int size);
    int sliceStepSize() const { return _sliceStepSize; }

    void forEachViewer(const std::function<void(CTiledVolumeViewer*)>& fn) const;
    void setIntersectionThickness(float thickness);
    float intersectionThickness() const { return _intersectionThickness; }
    void setHighlightedSurfaceIds(const std::vector<std::string>& ids);
    SurfacePatchIndex* surfacePatchIndex();
    void refreshSurfacePatchIndex(const SurfacePatchIndex::SurfacePtr& surface);
    void refreshSurfacePatchIndex(const SurfacePatchIndex::SurfacePtr& surface, const cv::Rect& changedRegion);
    void waitForPendingIndexRebuild();

signals:
    void viewerCreated(CTiledVolumeViewer* viewer);
    void overlayWindowChanged(float low, float high);
    void volumeWindowChanged(float low, float high);
    void overlayVolumeAvailabilityChanged(bool hasOverlay);
    void samplingStrideChanged(int stride);

private slots:
    void handleSurfacePatchIndexPrimeFinished();
    void handleSurfaceChanged(std::string name, std::shared_ptr<Surface> surf, bool isEditUpdate = false);
    void handleSurfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf);

private:
    void registerOverlay(ViewerOverlayControllerBase* overlay);
    bool updateSurfacePatchIndexForSurface(const SurfacePatchIndex::SurfacePtr& quad, bool isEditUpdate);

    CState* _state;
    VCCollection* _points;
    RenderPool _renderPool;  // shared across all viewers
    // Cache is obtained from Volume::tieredCache()
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
    std::vector<CTiledVolumeViewer*> _viewers;
    std::unordered_map<CTiledVolumeViewer*, bool> _resetDefaults;
    float _intersectionOpacity{1.0f};
    float _intersectionThickness{0.0f};
    std::shared_ptr<Volume> _overlayVolume;
    std::string _overlayVolumeId;
    float _overlayOpacity{0.5f};
    std::string _overlayColormapId;
    float _overlayWindowLow{0.0f};
    float _overlayWindowHigh{255.0f};
    float _volumeWindowLow{0.0f};
    float _volumeWindowHigh{255.0f};
    bool _mirrorCursorToSegmentation{false};
    int _sliceStepSize{1};
    int _surfacePatchSamplingStride{1};
    bool _surfacePatchStrideUserSet{false};
    int _targetRefinedStride{0};  // 0 = no refinement pending

    VolumeOverlayController* _volumeOverlay{nullptr};
    SurfacePatchIndex _surfacePatchIndex;
    bool _surfacePatchIndexNeedsRebuild{true};
    // Use string IDs for surface tracking to avoid dangling pointers in async operations
    std::unordered_set<std::string> _indexedSurfaceIds;
    std::vector<std::string> _pendingSurfacePatchIndexSurfaceIds;
    std::vector<std::string> _surfacesQueuedDuringRebuildIds;
    std::vector<std::pair<std::string, std::shared_ptr<Surface>>> _surfacesQueuedForRemovalDuringRebuild;
    QFutureWatcher<std::shared_ptr<SurfacePatchIndex>>* _surfacePatchIndexWatcher{nullptr};

    void rebuildSurfacePatchIndexIfNeeded();
};
