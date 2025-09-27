#pragma once

#include <QObject>
#include <QString>

#include <functional>
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
class ChunkCache;
class SegmentationModule;

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

    bool resetDefaultFor(CVolumeViewer* viewer) const;
    void setResetDefaultFor(CVolumeViewer* viewer, bool value);

    void forEachViewer(const std::function<void(CVolumeViewer*)>& fn) const;

signals:
    void viewerCreated(CVolumeViewer* viewer);

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
};
