#include "VectorOverlayController.hpp"

#include "../volume_viewers/VolumeViewerBase.hpp"
#include "../CState.hpp"
#include "../ViewerManager.hpp"

#include "vc/core/util/Surface.hpp"

#include <QFontMetricsF>

#include <algorithm>
#include <cmath>
#include <vector>

#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"

namespace
{
constexpr const char* kOverlayGroup = "vector_overlays";
constexpr qreal kArrowZ = 30.0;
constexpr qreal kLabelZ = 31.0;
}

VectorOverlayController::VectorOverlayController(CState* state, QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroup, parent)
    , _state(state)
{
    addProvider([this](VolumeViewerBase* viewer, OverlayBuilder& builder) {
        collectSurfaceNormals(viewer, builder);
    });
}

void VectorOverlayController::addProvider(Provider provider)
{
    if (provider) {
        _providers.push_back(std::move(provider));
    }
}

bool VectorOverlayController::isOverlayEnabledFor(VolumeViewerBase* viewer) const
{
    if (!viewer) {
        return false;
    }
    if (!viewer->isShowSurfaceNormals()) {
        return false;
    }
    for (const auto& provider : _providers) {
        if (provider) {
            return true;
        }
    }
    return false;
}

void VectorOverlayController::collectPrimitives(VolumeViewerBase* viewer,
                                                OverlayBuilder& builder)
{
    if (!viewer) {
        return;
    }
    for (const auto& provider : _providers) {
        if (provider) {
            provider(viewer, builder);
        }
    }
}

void VectorOverlayController::collectSurfaceNormals(VolumeViewerBase* viewer,
                                                     OverlayBuilder& builder) const
{
    if (!viewer->isShowSurfaceNormals()) {
        return;
    }

    auto* currentSurface = viewer->currentSurface();
    if (!currentSurface) {
        return;
    }

    const float viewerScale = viewer->getCurrentScale();
    const float arrowLengthScale = viewer->normalArrowLengthScale();
    const int maxArrowsPerAxis = viewer->normalMaxArrows();

    const float kArrowLen = 50.0f * arrowLengthScale;
    // 3-axis frame colors (standard convention): U-tangent = red, V-tangent = green,
    // Normal = blue. Drawn as a local frame at each sampled surface point.
    const QColor kAxisUColor(235, 60, 60);    // red
    const QColor kAxisVColor(60, 200, 60);    // green
    const QColor kAxisNColor(70, 120, 255);   // blue

    auto addNormalLegend = [&]() {
        const QRectF sceneRect = visibleSceneRect(viewer);
        const QPointF anchor = sceneRect.bottomRight() + QPointF(-42.0, -42.0);

        QFont font;
        font.setPointSizeF(9.0);
        auto legendLine = [&](const QPointF& pos, const QString& text, const QColor& color) {
            OverlayStyle style; style.penColor = color; style.z = kLabelZ;
            builder.addText(pos, text, font, style, true);
        };
        // 3-axis frame legend: U (red), V (green), N (blue).
        legendLine(anchor,                        QStringLiteral("U"), kAxisUColor);
        legendLine(anchor + QPointF(0.0, 14.0),   QStringLiteral("V"), kAxisVColor);
        legendLine(anchor + QPointF(0.0, 28.0),   QStringLiteral("N"), kAxisNColor);
    };

    auto addPlaneNormalsInstruction = [&]() {
        const QRectF sceneRect = visibleSceneRect(viewer);
        if (!sceneRect.isValid()) {
            return;
        }

        // 3-axis flip-check hint, using the empirically-verified directions of the
        // local frame: U (red) winds within the slice, V (green) points toward +Z
        // (up the scroll), N (blue) points INWARD toward the scroll center. Blue is
        // the reliable flip check -- if it points outward, the normals are reversed.
        // Each color word is drawn in its arrow's color, on two lines.
        QFont font;
        font.setPointSizeF(9.0);
        font.setBold(true);
        QFontMetricsF metrics(font);

        struct Seg { QString text; QColor color; };
        // Line 1: each axis' expected direction, drawn in its arrow's color.
        const std::vector<Seg> line1 = {
            {QStringLiteral("Blue (N) should point toward the center of the scroll.   "), kAxisNColor},
            {QStringLiteral("Red (U) should point along the winding.   "),               kAxisUColor},
            {QStringLiteral("Green (V) should point down the segment toward +Z."),       kAxisVColor},
        };
        // Line 2: the actionable check.
        const std::vector<Seg> line2 = {
            {QStringLiteral("If they don't, the surface normals are reversed — flip them."),
                            Qt::white},
        };
        auto drawCentered = [&](const std::vector<Seg>& segs, qreal y) {
            qreal totalW = 0.0;
            for (const auto& s : segs) totalW += metrics.horizontalAdvance(s.text);
            qreal x = sceneRect.center().x() - totalW * 0.5;
            for (const auto& s : segs) {
                OverlayStyle style; style.penColor = s.color; style.z = kLabelZ;
                builder.addText(QPointF(x, y), s.text, font, style, true);
                x += metrics.horizontalAdvance(s.text);
            }
        };
        const qreal yOffset = std::clamp(sceneRect.height() * 0.22, 56.0, 120.0);
        const qreal y0 = sceneRect.top() + yOffset;
        drawCentered(line1, y0);
        drawCentered(line2, y0 + metrics.height() + 2.0);
    };

    // Handle segmentation view (flattened UV space)
    if (viewer->surfName() == "segmentation") {
        auto* quad = dynamic_cast<QuadSurface*>(currentSurface);
        if (!quad) {
            return;
        }

        cv::Mat_<cv::Vec3f>* points = quad->rawPointsPtr();
        if (!points || points->empty()) {
            return;
        }

        const int rows = points->rows;
        const int cols = points->cols;
        const cv::Vec2f surfScale = quad->scale();

        // Arrow spacing: enforce a MINIMUM on-screen gap between arrows AND a hard cap
        // on the total count, so we don't draw thousands of cramped arrows when zoomed
        // out. scene px per grid cell = viewerScale / surfScale; the stride to hit the
        // min pixel gap is minGap / scenePxPerCell. The per-axis sampling cap
        // (maxArrowsPerAxis) is still respected as a floor on stride. Then, if the
        // resulting grid would still exceed the total cap, grow the stride to fit.
        constexpr double kMinArrowSpacingPx = 36.0;   // on-screen gap between arrows
        constexpr int    kMaxArrowsTotal    = 1000;   // hard cap (each is a +N/-N pair)
        const double scenePxPerCellX = double(viewerScale) / std::max(1e-6f, surfScale[0]);
        const double scenePxPerCellY = double(viewerScale) / std::max(1e-6f, surfScale[1]);
        int strideR = std::max({1,
            int(std::ceil(kMinArrowSpacingPx / std::max(1e-6, scenePxPerCellY))),
            rows / std::max(1, maxArrowsPerAxis)});
        int strideC = std::max({1,
            int(std::ceil(kMinArrowSpacingPx / std::max(1e-6, scenePxPerCellX))),
            cols / std::max(1, maxArrowsPerAxis)});
        // Grow strides uniformly until the sampled grid fits under the total cap.
        while ((std::size_t(rows / strideR + 1) * std::size_t(cols / strideC + 1))
                   > std::size_t(kMaxArrowsTotal)) {
            ++strideR; ++strideC;
        }

        auto drawAxisArrow = [&](const QPointF& origin, const cv::Vec3f& dir3d, const QColor& color) {
            OverlayStyle style;
            style.penColor = color;
            style.penWidth = 3.0;
            style.z = kArrowZ;

            // dir3d is a UNIT vector. Its (x,y) is its projection onto the view plane,
            // so |(x,y)| in [0,1] is the FORESHORTENING: 1 = fully in-plane, 0 = points
            // straight in/out of the screen. Draw the arrow at length proportional to
            // that, so an axis tilted out of plane reads as a shorter arrow (and a
            // perpendicular one as a dot) -- otherwise every axis drew the same fixed
            // length and you couldn't tell which way U/V actually pointed.
            const float inPlane = std::sqrt(dir3d[0] * dir3d[0] + dir3d[1] * dir3d[1]);
            if (inPlane < 0.1f) {
                OverlayStyle dotStyle;
                dotStyle.penColor = color;
                dotStyle.brushColor = color;
                dotStyle.penWidth = 1.0;
                dotStyle.z = kArrowZ;
                builder.addCircle(origin, 3.0f, true, dotStyle);
            } else {
                QPointF dir2d(dir3d[0] / inPlane, dir3d[1] / inPlane);
                QPointF end = origin + dir2d * (kArrowLen * inPlane);
                builder.addArrow(origin, end, 5.0, 3.0, style);
            }
        };

        // Draw the local 3-axis frame at a surface point: U-tangent (red),
        // V-tangent (green), Normal (blue). Each is projected to screen by
        // drawAxisArrow (a near-perpendicular axis collapses to a dot).
        auto drawAxisFrame = [&](const QPointF& origin, const cv::Vec3f& tangentU,
                                 const cv::Vec3f& tangentV, const cv::Vec3f& normal) {
            drawAxisArrow(origin, tangentU, kAxisUColor);
            drawAxisArrow(origin, tangentV, kAxisVColor);
            drawAxisArrow(origin, normal,   kAxisNColor);
        };

        addNormalLegend();

        for (int r = 0; r < rows; r += strideR) {
            for (int c = 0; c < cols; c += strideC) {
                const cv::Vec3f& p = (*points)(r, c);
                if (!vc::geometry::validSurfacePoint(p)) continue;

                cv::Vec3f pRight(-1, -1, -1), pLeft(-1, -1, -1);
                cv::Vec3f pDown(-1, -1, -1), pUp(-1, -1, -1);

                for (int d = 1; d <= 5; ++d) {
                    if (!vc::geometry::validSurfacePoint(pRight) && c + d < cols && vc::geometry::validSurfacePoint((*points)(r, c + d)))
                        pRight = (*points)(r, c + d);
                    if (!vc::geometry::validSurfacePoint(pLeft) && c - d >= 0 && vc::geometry::validSurfacePoint((*points)(r, c - d)))
                        pLeft = (*points)(r, c - d);
                    if (!vc::geometry::validSurfacePoint(pDown) && r + d < rows && vc::geometry::validSurfacePoint((*points)(r + d, c)))
                        pDown = (*points)(r + d, c);
                    if (!vc::geometry::validSurfacePoint(pUp) && r - d >= 0 && vc::geometry::validSurfacePoint((*points)(r - d, c)))
                        pUp = (*points)(r - d, c);
                }

                bool hasU = (vc::geometry::validSurfacePoint(pRight) && vc::geometry::validSurfacePoint(pLeft));
                bool hasV = (vc::geometry::validSurfacePoint(pDown) && vc::geometry::validSurfacePoint(pUp));
                if (!hasU && !hasV) continue;

                // Map the grid cell to scene via the viewer's real transform, which
                // includes the PAN offset (_surfacePtrX) + viewport-center term. The
                // old hand-rolled `c*gridToScene - centerOffset` applied only zoom, so
                // arrows didn't move with pan and spread from a fixed anchor on zoom.
                // grid cell (r,c) -> surface UV coords = ((c-cols/2)/surfScaleX, ...).
                const float surfX = (static_cast<float>(c) - cols / 2.0f) / surfScale[0];
                const float surfY = (static_cast<float>(r) - rows / 2.0f) / surfScale[1];
                QPointF origin = viewer->surfaceCoordsToScene(surfX, surfY);

                cv::Vec3f tangentU(0, 0, 0), tangentV(0, 0, 0);

                if (hasU) {
                    tangentU = pRight - pLeft;
                    const float len2 = tangentU.dot(tangentU);
                    if (len2 > 1e-12f) {
                        tangentU /= std::sqrt(len2);
                    }
                }

                if (hasV) {
                    tangentV = pDown - pUp;
                    const float len2 = tangentV.dot(tangentV);
                    if (len2 > 1e-12f) {
                        tangentV /= std::sqrt(len2);
                    }
                }

                if (hasU && hasV) {
                    // Left-hand rule: U x V gives normal pointing toward viewer
                    // (consistent with vcr::grid's patch normals)
                    cv::Vec3f normal = tangentU.cross(tangentV);
                    const float len2 = normal.dot(normal);
                    if (len2 > 1e-12f) {
                        normal /= std::sqrt(len2);
                        // U (red), V (green), N (blue) local frame.
                        drawAxisFrame(origin, tangentU, tangentV, normal);
                    }
                }
            }
        }
        return;
    }

    // Handle plane views (XY, XZ, YZ) - draw normals along the surface line
    auto* plane = dynamic_cast<PlaneSurface*>(currentSurface);
    if (!plane) {
        return;
    }

    // Get the segmentation surface
    QuadSurface* segSurface = nullptr;
    std::shared_ptr<Surface> segSurfaceHolder;
    if (_state) {
        segSurfaceHolder = _state->surface("segmentation");
        segSurface = dynamic_cast<QuadSurface*>(segSurfaceHolder.get());
    }
    if (!segSurface) {
        return;
    }

    // Find current position on surface
    cv::Vec3f targetWP = plane->origin();
    if (_state) {
        if (auto* poi = _state->poi("focus")) {
            targetWP = poi->p;
        }
    }

    const auto hit = manager() ? manager()->locate(*segSurface, targetWP, 50.0f) : std::nullopt;
    if (!hit) {
        return;
    }
    const cv::Vec3f segPtr = hit->ptr;

    // Helper to draw an arrow projected onto the plane. Anchor + tip go through the
    // viewer's volumeToScene() (plane projection + PAN offset + viewport-center term)
    // -- the old raw plane->project() applied only zoom, so arrows stuck to the
    // top-left and ignored pan. Direction is computed in SCENE space from the two
    // projected world points so it stays correct under the full transform.
    auto drawPlaneArrow = [&](const cv::Vec3f& worldPos, const cv::Vec3f& dir3d, const QColor& color) {
        const QPointF origin = viewer->volumeToScene(worldPos);
        const QPointF tip = viewer->volumeToScene(worldPos + dir3d * (kArrowLen / viewerScale));

        // tip-origin IS the honest foreshortened scene vector (projecting a fixed
        // world-space length): full kArrowLen when the axis lies in the plane, shorter
        // as it tilts out, ~0 when perpendicular. Draw it AS-IS (don't re-normalize to
        // a fixed length) so the in-plane direction + foreshortening are both readable.
        const QPointF dir2d(tip.x() - origin.x(), tip.y() - origin.y());
        const float len2d = static_cast<float>(std::hypot(dir2d.x(), dir2d.y()));

        OverlayStyle style;
        style.penColor = color;
        style.penWidth = 3.0;
        style.z = kArrowZ;

        if (len2d < 2.0f) {
            // Mostly perpendicular to the view plane -> dot.
            OverlayStyle dotStyle;
            dotStyle.penColor = color;
            dotStyle.brushColor = color;
            dotStyle.penWidth = 1.0;
            dotStyle.z = kArrowZ;
            builder.addCircle(origin, 3.0f, true, dotStyle);
        } else {
            builder.addArrow(origin, tip, 5.0, 3.0, style);
        }
    };

    // 3-axis frame at a world point projected onto the plane: U (red), V (green),
    // N (blue). An axis pointing into/out of the plane collapses to a dot.
    auto drawPlaneAxisFrame = [&](const cv::Vec3f& worldPos, const cv::Vec3f& tangentU,
                                  const cv::Vec3f& tangentV, const cv::Vec3f& normal) {
        drawPlaneArrow(worldPos, tangentU, kAxisUColor);
        drawPlaneArrow(worldPos, tangentV, kAxisVColor);
        drawPlaneArrow(worldPos, normal,   kAxisNColor);
    };

    addNormalLegend();
    addPlaneNormalsInstruction();

    // Get step size from settings or segment metadata
    float stepVal = 50.0f;  // Default step in nominal coords
    if (!segSurface->meta.is_null()) {
        try {
            if (segSurface->meta.contains("vc_grow_seg_from_segments_params")) {
                const auto& p = segSurface->meta.at("vc_grow_seg_from_segments_params");
                if (p.contains("step")) {
                    stepVal = p.at("step").get_float();
                }
            }
        } catch (...) {}
    }

    // Sample points along the surface, but only draw those close to the viewing plane
    // Walk in both U and V directions to find points that intersect this plane
    const int kMaxSamplesPerSide = maxArrowsPerAxis;
    const int kMaxTotalSamples = maxArrowsPerAxis * 3;
    constexpr float kPlaneDistThreshold = 3.0f;  // Only draw if within 3 voxels of plane

    const float sampleStep = stepVal * 2.0f;
    int samplesDrawn = 0;

    // Sample in a grid pattern around the current position
    for (int nu = -kMaxSamplesPerSide; nu <= kMaxSamplesPerSide && samplesDrawn < kMaxTotalSamples; ++nu) {
        for (int nv = -kMaxSamplesPerSide; nv <= kMaxSamplesPerSide && samplesDrawn < kMaxTotalSamples; ++nv) {
            float offsetU = nu * sampleStep;
            float offsetV = nv * sampleStep;

            cv::Vec3f worldPos = segSurface->coord(segPtr, {offsetU, offsetV, 0});
            if (!vc::geometry::validSurfacePoint(worldPos)) continue;

            // Only draw if this point is close to the viewing plane
            float planeDist = std::abs(plane->pointDist(worldPos));
            if (planeDist > kPlaneDistThreshold) continue;

            // Get tangents by sampling nearby points
            cv::Vec3f pPlusU = segSurface->coord(segPtr, {offsetU + 2.0f, offsetV, 0});
            cv::Vec3f pMinusU = segSurface->coord(segPtr, {offsetU - 2.0f, offsetV, 0});
            cv::Vec3f pPlusV = segSurface->coord(segPtr, {offsetU, offsetV + 2.0f, 0});
            cv::Vec3f pMinusV = segSurface->coord(segPtr, {offsetU, offsetV - 2.0f, 0});

            bool hasU = (vc::geometry::validSurfacePoint(pPlusU) && vc::geometry::validSurfacePoint(pMinusU));
            bool hasV = (vc::geometry::validSurfacePoint(pPlusV) && vc::geometry::validSurfacePoint(pMinusV));

            if (!hasU && !hasV) continue;

            cv::Vec3f tangentU(0, 0, 0), tangentV(0, 0, 0);

            if (hasU) {
                tangentU = pPlusU - pMinusU;
                const float len2 = tangentU.dot(tangentU);
                if (len2 > 1e-12f) {
                    tangentU /= std::sqrt(len2);
                }
            }

            if (hasV) {
                tangentV = pPlusV - pMinusV;
                const float len2 = tangentV.dot(tangentV);
                if (len2 > 1e-12f) {
                    tangentV /= std::sqrt(len2);
                }
            }

            if (hasU && hasV) {
                // Left-hand rule: U x V gives normal pointing toward viewer
                // (consistent with vcr::grid's patch normals)
                cv::Vec3f normal = tangentU.cross(tangentV);
                const float len2 = normal.dot(normal);
                if (len2 > 1e-12f) {
                    normal /= std::sqrt(len2);
                    drawPlaneAxisFrame(worldPos, tangentU, tangentV, normal);
                }
            }

            ++samplesDrawn;
        }
    }
}
