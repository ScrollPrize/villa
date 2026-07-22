#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <cstdint>
#include <memory>
#include <vector>

class FiberOverlayController final : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    struct Chain {
        uint64_t id = 0;
        std::vector<cv::Vec3f> points;
    };

    explicit FiberOverlayController(QObject* parent = nullptr);
    ~FiberOverlayController() override;

    void setChains(std::vector<Chain> chains);
    void setVisible(bool visible);
    void setViewDistance(double distance);
    [[nodiscard]] bool isVisible() const { return _visible; }
    [[nodiscard]] bool hasChains() const { return !_chains.empty(); }
    [[nodiscard]] double viewDistance() const { return _viewDistance; }

    static QColor fiberColor(uint64_t fiberId);
    static PointChainStyle fiberStyle(const QColor& color, float distanceTolerance);

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override;
    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override;
    void applyOverlayPrimitives(VolumeViewerBase* viewer,
                                std::vector<OverlayPrimitive> primitives) override;
    void clearOverlay(VolumeViewerBase* viewer) const override;

private:
    struct PersistentItems;

    std::vector<Chain> _chains;
    std::unique_ptr<PersistentItems> _persistentItems;
    bool _visible{false};
    float _viewDistance{10.0f};
};
