#pragma once

#include <QObject>
#include <QString>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class QMdiArea;
class CVolumeViewer;
class CSurfaceCollection;
class VCCollection;
class SegmentationOverlayController;
class PointsOverlayController;
class PathsOverlayController;
class BBoxOverlayController;
class VectorOverlayController;
class VolumeOverlayController;
class ChunkCache;
class SegmentationModule;
class Volume;

class ViewerManager : public QObject
{
    Q_OBJECT

public:
    ViewerManager(CSurfaceCollection* surfaces,
                  VCCollection* points,
                  ChunkCache* cache,
                  QObject* parent = nullptr);

    CVolumeViewer* createViewer(const std::string& surfaceName,
                                const QString& title,
                                QMdiArea* mdiArea);

    const std::vector<CVolumeViewer*>& viewers() const { return _viewers; }

    void setSegmentationOverlay(SegmentationOverlayController* overlay);
    void setSegmentationEditActive(bool active);
    void setSegmentationModule(SegmentationModule* module);
    void setPointsOverlay(PointsOverlayController* overlay);
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

    bool resetDefaultFor(CVolumeViewer* viewer) const;
    void setResetDefaultFor(CVolumeViewer* viewer, bool value);

    void forEachViewer(const std::function<void(CVolumeViewer*)>& fn) const;

    // Query segmentation growth state (via attached SegmentationModule)
    bool segmentationGrowthInProgress() const;

signals:
    void viewerCreated(CVolumeViewer* viewer);
    void overlayWindowChanged(float low, float high);
    void volumeWindowChanged(float low, float high);
    void overlayVolumeAvailabilityChanged(bool hasOverlay);

private:
    CSurfaceCollection* _surfaces;
    VCCollection* _points;
    ChunkCache* _chunkCache;
    SegmentationOverlayController* _segmentationOverlay{nullptr};
    PointsOverlayController* _pointsOverlay{nullptr};
    PathsOverlayController* _pathsOverlay{nullptr};
    BBoxOverlayController* _bboxOverlay{nullptr};
    VectorOverlayController* _vectorOverlay{nullptr};
    bool _segmentationEditActive{false};
    SegmentationModule* _segmentationModule{nullptr};
    std::vector<CVolumeViewer*> _viewers;
    std::unordered_map<CVolumeViewer*, bool> _resetDefaults;
    float _intersectionOpacity{1.0f};
    std::shared_ptr<Volume> _overlayVolume;
    std::string _overlayVolumeId;
    float _overlayOpacity{0.5f};
    std::string _overlayColormapId;
    float _overlayWindowLow{0.0f};
    float _overlayWindowHigh{255.0f};
    float _volumeWindowLow{0.0f};
    float _volumeWindowHigh{255.0f};

    VolumeOverlayController* _volumeOverlay{nullptr};
};
