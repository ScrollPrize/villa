#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <QHash>
#include <memory>
#include <unordered_map>

#include "vc/core/util/PolylineIndex.hpp"

class SpiralOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT
public:
    explicit SpiralOverlayController(QObject* parent = nullptr);
    void publishIndex(std::shared_ptr<const PolylineIndex> index, quint64 generation);
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
    void schedule(VolumeViewerBase* viewer, const QString& key,
                  const cv::Vec3f& lo, const cv::Vec3f& hi);

    std::shared_ptr<const PolylineIndex> _index;
    quint64 _indexGeneration = 0;
    quint64 _requestGeneration = 0;
    QHash<QString, bool> _visible;
    std::unordered_map<VolumeViewerBase*, Cache> _cache;
};
