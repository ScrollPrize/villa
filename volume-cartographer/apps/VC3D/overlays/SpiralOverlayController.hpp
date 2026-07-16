#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <QHash>
#include <QImage>
#include <memory>
#include <unordered_map>

#include "vc/core/util/PolylineIndex.hpp"

class QuadSurface;

class SpiralOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT
public:
    explicit SpiralOverlayController(QObject* parent = nullptr);
    void publishIndex(std::shared_ptr<const PolylineIndex> index, quint64 generation);
    void publishRunDiff(std::shared_ptr<QuadSurface> surface, QImage image);
    void setRunDiffVisible(bool visible);
    void reset();
    void setCategoryVisible(const QString& category, bool visible);
    void detachViewer(VolumeViewerBase* viewer) override;

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;

private:
    struct Cache {
        QString requestKey;
        quint64 requestGeneration = 0;
        std::vector<PathPrimitive> paths;
    };
    // One resident entry per fiber/pcl polyline: the whole generation stays
    // loaded so panning never waits on a viewport query. The points are owned
    // by _index (immutable once published); the AABB is the per-refresh cull.
    struct ChainEntry {
        QString category;
        const std::vector<cv::Vec3f>* points = nullptr;
        cv::Vec3f lo{0, 0, 0};
        cv::Vec3f hi{0, 0, 0};
    };
    void schedule(VolumeViewerBase* viewer, const QString& key,
                  const cv::Vec3f& lo, const cv::Vec3f& hi);
    void rebuildChains();
    bool hasRunDiffFor(VolumeViewerBase* viewer) const;

    std::shared_ptr<const PolylineIndex> _index;
    std::vector<ChainEntry> _chains;
    quint64 _indexGeneration = 0;
    quint64 _requestGeneration = 0;
    QHash<QString, bool> _visible;
    std::unordered_map<VolumeViewerBase*, Cache> _cache;
    std::shared_ptr<QuadSurface> _runDiffSurface;
    QImage _runDiffImage;
    bool _runDiffVisible = false;
};
