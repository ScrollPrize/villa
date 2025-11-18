#pragma once

#include "SegmentationGrowth.hpp"

#include <QFutureWatcher>
#include <QObject>
#include <QString>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class SegmentationModule;
class SegmentationWidget;
class CSurfaceCollection;
class ViewerManager;
class SurfacePanelController;
class VolumePkg;
class Volume;
template <typename T> class ChunkCache;
class QuadSurface;
class CVolumeViewer;

class SegmentationGrower : public QObject
{
    Q_OBJECT

public:
    struct Context
    {
        SegmentationModule* module{nullptr};
        SegmentationWidget* widget{nullptr};
        CSurfaceCollection* surfaces{nullptr};
        ViewerManager* viewerManager{nullptr};
        ChunkCache<uint8_t>* chunkCache{nullptr};
    };

    struct UiCallbacks
    {
        std::function<void(const QString&, int)> showStatus;
        std::function<void(QuadSurface*)> applySliceOrientation;
    };

    struct VolumeContext
    {
        std::shared_ptr<VolumePkg> package;
        std::shared_ptr<Volume> activeVolume;
        std::string activeVolumeId;
        std::string requestedVolumeId;
        QString normalGridPath;
    };

    SegmentationGrower(Context context,
                       UiCallbacks callbacks,
                       QObject* parent = nullptr);

    void updateContext(Context context);
    void updateUiCallbacks(UiCallbacks callbacks);
    void setSurfacePanel(SurfacePanelController* panel);

    bool start(const VolumeContext& volumeContext,
               SegmentationGrowthMethod method,
               SegmentationGrowthDirection direction,
               int steps,
               bool inpaintOnly);

    bool running() const { return _running; }

private:
    struct ActiveRequest
    {
        VolumeContext volumeContext;
        std::shared_ptr<Volume> growthVolume;
        std::string growthVolumeId;
        QuadSurface* segmentationSurface{nullptr};
        double growthVoxelSize{0.0};
        bool usingCorrections{false};
        bool inpaintOnly{false};
        std::optional<cv::Rect> correctionsDirtyBounds;
    };

    void finalize(bool ok);
    void handleFailure(const QString& message);
    void onFutureFinished();

    Context _context;
    UiCallbacks _callbacks;
    SurfacePanelController* _surfacePanel{nullptr};
    bool _running{false};
    std::unique_ptr<QFutureWatcher<TracerGrowthResult>> _watcher;
    std::optional<ActiveRequest> _activeRequest;
};
