#pragma once

#include "ViewerOverlayControllerBase.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

class FiberOverlayController final : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    struct Chain {
        uint64_t id = 0;
        std::vector<cv::Vec3f> points;
        // Color-group representative when "show linked" is active; 0 = use id.
        uint64_t colorId = 0;
        // Parallel to points (may be empty): 0 none, 1 pending link, 2 approved.
        std::vector<uint8_t> pointLinkStates;
    };

    explicit FiberOverlayController(QObject* parent = nullptr);
    ~FiberOverlayController() override;

    void setChains(std::vector<Chain> chains);
    void setVisible(bool visible);
    void setViewDistance(double distance);
    void setShowLinked(bool show);
    [[nodiscard]] bool showLinked() const { return _showLinked; }
    [[nodiscard]] bool isVisible() const { return _visible; }
    [[nodiscard]] bool hasChains() const { return !_chains.empty(); }
    [[nodiscard]] double viewDistance() const { return _viewDistance; }

    struct ControlPointHit {
        uint64_t fiberId = 0;
        int controlPointIndex = -1;
    };
    // Nearest rendered fiber control point within maxDistancePx scene units
    // of scenePoint, or nullopt. Matches exactly what renderPointChain draws:
    // the overlay must be visible and the point must have opacity > 0.
    [[nodiscard]] std::optional<ControlPointHit> hitTestControlPoint(
        VolumeViewerBase* viewer,
        const QPointF& scenePoint,
        qreal maxDistancePx) const;

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
    bool _showLinked{false};
    float _viewDistance{10.0f};
};
