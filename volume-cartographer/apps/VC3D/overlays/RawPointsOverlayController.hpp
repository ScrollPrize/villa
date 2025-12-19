#pragma once

#include "ViewerOverlayControllerBase.hpp"
#include "../CSurfaceCollection.hpp"

#include <memory>
#include <optional>

#include <opencv2/core.hpp>

class QuadSurface;

// Overlay controller that displays the raw surface grid points
// For surfaces that are scaled down significantly (e.g., 20x), this shows
// where the actual data points are in the original high-resolution grid.
//
// Rendering is limited to avoid displaying millions of points:
// - Flattened view: shows points within radius (grid steps) from focus POI
// - Plane views (XY/XZ/YZ): shows points within distance from slice plane
class RawPointsOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    explicit RawPointsOverlayController(CSurfaceCollection* surfaces, QObject* parent = nullptr);
    ~RawPointsOverlayController() override;

    // Enable/disable the overlay
    void setEnabled(bool enabled);
    [[nodiscard]] bool isEnabled() const { return _enabled; }

    // Maximum number of points to render (performance limit)
    void setMaxPoints(int maxPoints);
    [[nodiscard]] int maxPoints() const { return _maxPoints; }

    // For flattened view: radius in grid steps from POI center
    void setGridRadius(int radius);
    [[nodiscard]] int gridRadius() const { return _gridRadius; }

    // For plane views: distance threshold from slice plane (in voxels)
    void setPlaneDistanceThreshold(float threshold);
    [[nodiscard]] float planeDistanceThreshold() const { return _planeDistanceThreshold; }

    // Point rendering style
    void setPointRadius(float radius);
    [[nodiscard]] float pointRadius() const { return _pointRadius; }

    void setPointColor(const QColor& color);
    [[nodiscard]] QColor pointColor() const { return _pointColor; }

    void setPointOpacity(float opacity);
    [[nodiscard]] float pointOpacity() const { return _pointOpacity; }

protected:
    bool isOverlayEnabledFor(CVolumeViewer* viewer) const override;
    void collectPrimitives(CVolumeViewer* viewer, OverlayBuilder& builder) override;

private slots:
    void onSurfaceChanged(std::string name, std::shared_ptr<Surface> surface);
    void onPoiChanged(std::string name, POI* poi);

private:
    // Get the focus POI position in grid coordinates for a surface
    std::optional<std::pair<int, int>> focusGridPosition(QuadSurface* surface) const;

    // Collect points for flattened (segmentation) view
    void collectFlattenedViewPoints(CVolumeViewer* viewer,
                                    QuadSurface* surface,
                                    OverlayBuilder& builder);

    // Collect points for plane (XY/XZ/YZ) view
    void collectPlaneViewPoints(CVolumeViewer* viewer,
                                QuadSurface* surface,
                                OverlayBuilder& builder);

    CSurfaceCollection* _surfaces{nullptr};
    bool _enabled{false};
    int _maxPoints{2000};
    int _gridRadius{50};  // Grid steps from POI for flattened view
    float _planeDistanceThreshold{30.0f};  // Voxels from plane for plane views (needs to be large enough to capture points at ~20 voxel spacing)
    float _pointRadius{4.0f};  // 2x larger
    QColor _pointColor{100, 200, 255};  // Cyan-ish
    float _pointOpacity{0.9f};  // More opaque
};
