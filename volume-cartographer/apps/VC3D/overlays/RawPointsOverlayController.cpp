#include "RawPointsOverlayController.hpp"

#include "../CSurfaceCollection.hpp"
#include "../CVolumeViewer.hpp"

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace
{
constexpr const char* kOverlayGroupRawPoints = "raw_points_overlay";
constexpr qreal kZValue = 85.0;  // Below vertex markers but above most overlays

// Check if a point is valid (not the sentinel -1 value)
bool isValidPoint(const cv::Vec3f& p)
{
    return p[0] != -1.f;
}

} // namespace

RawPointsOverlayController::RawPointsOverlayController(CSurfaceCollection* surfaces, QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroupRawPoints, parent)
    , _surfaces(surfaces)
{
    if (_surfaces) {
        connect(_surfaces, &CSurfaceCollection::sendSurfaceChanged,
                this, &RawPointsOverlayController::onSurfaceChanged);
        connect(_surfaces, &CSurfaceCollection::sendPOIChanged,
                this, &RawPointsOverlayController::onPoiChanged);
    }
}

RawPointsOverlayController::~RawPointsOverlayController() = default;

void RawPointsOverlayController::setEnabled(bool enabled)
{
    if (_enabled == enabled) {
        return;
    }
    _enabled = enabled;
    refreshAll();
}

void RawPointsOverlayController::setMaxPoints(int maxPoints)
{
    maxPoints = std::max(100, maxPoints);
    if (_maxPoints == maxPoints) {
        return;
    }
    _maxPoints = maxPoints;
    if (_enabled) {
        refreshAll();
    }
}

void RawPointsOverlayController::setGridRadius(int radius)
{
    radius = std::max(1, radius);
    if (_gridRadius == radius) {
        return;
    }
    _gridRadius = radius;
    if (_enabled) {
        refreshAll();
    }
}

void RawPointsOverlayController::setPlaneDistanceThreshold(float threshold)
{
    threshold = std::max(0.1f, threshold);
    if (std::abs(_planeDistanceThreshold - threshold) < 0.01f) {
        return;
    }
    _planeDistanceThreshold = threshold;
    if (_enabled) {
        refreshAll();
    }
}

void RawPointsOverlayController::setPointRadius(float radius)
{
    radius = std::clamp(radius, 0.5f, 10.0f);
    if (std::abs(_pointRadius - radius) < 0.01f) {
        return;
    }
    _pointRadius = radius;
    if (_enabled) {
        refreshAll();
    }
}

void RawPointsOverlayController::setPointColor(const QColor& color)
{
    if (_pointColor == color) {
        return;
    }
    _pointColor = color;
    if (_enabled) {
        refreshAll();
    }
}

void RawPointsOverlayController::setPointOpacity(float opacity)
{
    opacity = std::clamp(opacity, 0.0f, 1.0f);
    if (std::abs(_pointOpacity - opacity) < 0.01f) {
        return;
    }
    _pointOpacity = opacity;
    if (_enabled) {
        refreshAll();
    }
}

bool RawPointsOverlayController::isOverlayEnabledFor(CVolumeViewer* viewer) const
{
    return _enabled && _surfaces && viewer;
}

void RawPointsOverlayController::collectPrimitives(CVolumeViewer* viewer, OverlayBuilder& builder)
{
    if (!_enabled || !_surfaces || !viewer) {
        return;
    }

    // Get the segmentation surface
    auto surfacePtr = _surfaces->surface("segmentation");
    auto* quadSurface = dynamic_cast<QuadSurface*>(surfacePtr.get());
    if (!quadSurface) {
        return;
    }

    // Determine the type of viewer
    Surface* viewerSurface = viewer->currentSurface();
    auto* planeSurface = dynamic_cast<PlaneSurface*>(viewerSurface);

    if (planeSurface) {
        // XY/XZ/YZ slice view
        collectPlaneViewPoints(viewer, quadSurface, builder);
    } else if (dynamic_cast<QuadSurface*>(viewerSurface)) {
        // Flattened segmentation view
        collectFlattenedViewPoints(viewer, quadSurface, builder);
    }
}

void RawPointsOverlayController::onSurfaceChanged(std::string name, std::shared_ptr<Surface> /*surface*/)
{
    if (name == "segmentation" && _enabled) {
        refreshAll();
    }
}

void RawPointsOverlayController::onPoiChanged(std::string name, POI* /*poi*/)
{
    if (name == "focus" && _enabled) {
        refreshAll();
    }
}

std::optional<std::pair<int, int>> RawPointsOverlayController::focusGridPosition(QuadSurface* surface) const
{
    if (!_surfaces || !surface) {
        return std::nullopt;
    }

    POI* focusPoi = _surfaces->poi("focus");
    if (!focusPoi) {
        return std::nullopt;
    }

    // Convert world position to grid position using pointTo
    cv::Vec3f ptr = surface->pointer();
    float dist = surface->pointTo(ptr, focusPoi->p, 10.0f, 100);
    if (dist > 5.0f) {
        return std::nullopt;
    }

    // ptr is now in pointer space (relative to surface center)
    // Convert to grid indices
    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }

    int col = static_cast<int>(std::round(ptr[0] + points->cols / 2.0f));
    int row = static_cast<int>(std::round(ptr[1] + points->rows / 2.0f));

    if (row < 0 || row >= points->rows || col < 0 || col >= points->cols) {
        return std::nullopt;
    }

    return std::make_pair(row, col);
}

void RawPointsOverlayController::collectFlattenedViewPoints(CVolumeViewer* viewer,
                                                            QuadSurface* surface,
                                                            OverlayBuilder& builder)
{
    auto focusPos = focusGridPosition(surface);
    if (!focusPos) {
        return;
    }

    const int focusRow = focusPos->first;
    const int focusCol = focusPos->second;

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return;
    }

    const int rows = points->rows;
    const int cols = points->cols;

    // Collect points within grid radius, sorted by distance
    struct GridPoint {
        int row;
        int col;
        float distSq;
        cv::Vec3f world;
    };
    std::vector<GridPoint> candidates;
    candidates.reserve(static_cast<size_t>((2 * _gridRadius + 1) * (2 * _gridRadius + 1)));

    const int minRow = std::max(0, focusRow - _gridRadius);
    const int maxRow = std::min(rows - 1, focusRow + _gridRadius);
    const int minCol = std::max(0, focusCol - _gridRadius);
    const int maxCol = std::min(cols - 1, focusCol + _gridRadius);

    for (int r = minRow; r <= maxRow; ++r) {
        for (int c = minCol; c <= maxCol; ++c) {
            const cv::Vec3f& p = (*points)(r, c);
            if (!isValidPoint(p)) {
                continue;
            }

            const float dr = static_cast<float>(r - focusRow);
            const float dc = static_cast<float>(c - focusCol);
            const float distSq = dr * dr + dc * dc;

            // Only include points within circular radius
            if (distSq <= static_cast<float>(_gridRadius * _gridRadius)) {
                candidates.push_back({r, c, distSq, p});
            }
        }
    }

    // Sort by distance and limit count
    std::sort(candidates.begin(), candidates.end(),
              [](const GridPoint& a, const GridPoint& b) { return a.distSq < b.distSq; });

    const size_t count = std::min(candidates.size(), static_cast<size_t>(_maxPoints));

    // Prepare style
    OverlayStyle style;
    style.penColor = _pointColor;
    style.penColor.setAlphaF(_pointOpacity);
    style.brushColor = _pointColor;
    style.brushColor.setAlphaF(_pointOpacity * 0.5f);
    style.penWidth = 1.0;
    style.z = kZValue;

    // Scale point radius with zoom level
    const float viewerScale = viewer->getCurrentScale();
    const float scaledRadius = _pointRadius * std::sqrt(viewerScale);

    // Get surface scale for coordinate conversion
    // Grid indices -> pointer coords -> nominal coords -> scene coords
    const cv::Vec2f surfScale = surface->scale();
    const float centerX = cols / 2.0f;
    const float centerY = rows / 2.0f;

    // Render points using grid indices directly (not 3D world positions)
    for (size_t i = 0; i < count; ++i) {
        const GridPoint& gp = candidates[i];

        // Convert grid (row, col) to pointer-space coordinates
        const float ptrX = static_cast<float>(gp.col) - centerX;
        const float ptrY = static_cast<float>(gp.row) - centerY;

        // Convert to nominal coordinates (divide by surface scale)
        const float nomX = ptrX / surfScale[0];
        const float nomY = ptrY / surfScale[1];

        // Convert to scene coordinates (multiply by viewer scale)
        const float sceneX = nomX * viewerScale;
        const float sceneY = nomY * viewerScale;

        QPointF scenePos(sceneX, sceneY);
        builder.addCircle(scenePos, scaledRadius, true, style);
    }
}

void RawPointsOverlayController::collectPlaneViewPoints(CVolumeViewer* viewer,
                                                        QuadSurface* surface,
                                                        OverlayBuilder& builder)
{
    Surface* viewerSurface = viewer->currentSurface();
    auto* planeSurface = dynamic_cast<PlaneSurface*>(viewerSurface);
    if (!planeSurface) {
        return;
    }

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return;
    }

    // Get plane parameters - use origin() method to get actual plane position
    cv::Vec3f planeOrigin = planeSurface->origin();
    cv::Vec3f planeNormal = planeSurface->normal({0, 0, 0});

    // Normalize the plane normal
    float normalLen = cv::norm(planeNormal);
    if (normalLen < 1e-6f) {
        return;
    }
    planeNormal /= normalLen;

    // Collect points near the plane
    struct PlanePoint {
        cv::Vec3f world;
        float signedDistance;  // Positive = in front of plane (+ direction), negative = behind
    };
    std::vector<PlanePoint> candidates;
    candidates.reserve(1000);

    const int rows = points->rows;
    const int cols = points->cols;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const cv::Vec3f& p = (*points)(r, c);
            if (!isValidPoint(p)) {
                continue;
            }

            // Calculate signed distance to plane
            cv::Vec3f toPoint = p - planeOrigin;
            float signedDist = toPoint.dot(planeNormal);
            float absDist = std::abs(signedDist);

            if (absDist <= _planeDistanceThreshold) {
                candidates.push_back({p, signedDist});
            }
        }
    }

    // Sort by absolute distance to plane and limit count
    std::sort(candidates.begin(), candidates.end(),
              [](const PlanePoint& a, const PlanePoint& b) {
                  return std::abs(a.signedDistance) < std::abs(b.signedDistance);
              });

    const size_t count = std::min(candidates.size(), static_cast<size_t>(_maxPoints));

    // Scale point radius with zoom level
    const float viewerScale = viewer->getCurrentScale();
    const float scaledRadius = _pointRadius * std::sqrt(viewerScale);

    // Prepare styles for positive (red) and negative (green) directions
    OverlayStyle positiveStyle;
    positiveStyle.penColor = QColor(255, 80, 80);  // Red for + direction
    positiveStyle.penColor.setAlphaF(_pointOpacity);
    positiveStyle.brushColor = QColor(255, 80, 80);
    positiveStyle.brushColor.setAlphaF(_pointOpacity * 0.5f);
    positiveStyle.penWidth = 1.0;
    positiveStyle.z = kZValue;

    OverlayStyle negativeStyle;
    negativeStyle.penColor = QColor(80, 255, 80);  // Green for - direction
    negativeStyle.penColor.setAlphaF(_pointOpacity);
    negativeStyle.brushColor = QColor(80, 255, 80);
    negativeStyle.brushColor.setAlphaF(_pointOpacity * 0.5f);
    negativeStyle.penWidth = 1.0;
    negativeStyle.z = kZValue;

    // Render points with color based on signed distance
    for (size_t i = 0; i < count; ++i) {
        const PlanePoint& pp = candidates[i];
        QPointF scenePos = viewer->volumePointToScene(pp.world);
        const OverlayStyle& style = (pp.signedDistance >= 0) ? positiveStyle : negativeStyle;
        builder.addCircle(scenePos, scaledRadius, true, style);
    }
}
