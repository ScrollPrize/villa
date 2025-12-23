#include "VectorOverlayController.hpp"

#include "../CVolumeViewer.hpp"
#include "../CSurfaceCollection.hpp"
#include "../VCSettings.hpp"
#include "../ViewerManager.hpp"

#include "vc/core/util/Surface.hpp"

#include <QSettings>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>

#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"

namespace
{
constexpr const char* kOverlayGroup = "vector_overlays";
constexpr qreal kArrowLength = 60.0;
constexpr qreal kArrowHeadLength = 10.0;
constexpr qreal kArrowHeadWidth = 6.0;
constexpr qreal kArrowZ = 30.0;
constexpr qreal kLabelZ = 31.0;
constexpr qreal kMarkerZ = 32.0;
constexpr float kStepCenterRadius = 4.0f;
constexpr float kStepMarkerRadius = 3.0f;
const QColor kCenterColor(255, 255, 0);
const QColor kArrowFalseColor(Qt::red);
const QColor kArrowTrueColor(Qt::green);
}

VectorOverlayController::VectorOverlayController(CSurfaceCollection* surfaces, QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroup, parent)
    , _surfaces(surfaces)
{
    addProvider([this](CVolumeViewer* viewer, OverlayBuilder& builder) {
        collectDirectionHints(viewer, builder);
    });
    addProvider([this](CVolumeViewer* viewer, OverlayBuilder& builder) {
        collectSurfaceNormals(viewer, builder);
    });
}

void VectorOverlayController::addProvider(Provider provider)
{
    if (provider) {
        _providers.push_back(std::move(provider));
    }
}

bool VectorOverlayController::isOverlayEnabledFor(CVolumeViewer* viewer) const
{
    if (!viewer) {
        return false;
    }
    if (!viewer->isShowDirectionHints() && !viewer->isShowSurfaceNormals()) {
        return false;
    }
    for (const auto& provider : _providers) {
        if (provider) {
            return true;
        }
    }
    return false;
}

void VectorOverlayController::collectPrimitives(CVolumeViewer* viewer,
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

void VectorOverlayController::collectDirectionHints(CVolumeViewer* viewer,
                                                    OverlayBuilder& builder) const
{
    if (!viewer->isShowDirectionHints()) {
        return;
    }

    auto* currentSurface = viewer->currentSurface();
    if (!currentSurface) {
        return;
    }

    const float scale = viewer->getCurrentScale();
    QPointF anchorScene = visibleSceneRect(viewer).center();

    auto addArrow = [&](const QPointF& origin, const QPointF& direction, const QColor& color) {
        if (direction.isNull()) {
            return;
        }
        QPointF dir = direction;
        double mag = std::hypot(dir.x(), dir.y());
        if (mag < 1e-3) {
            return;
        }
        dir.setX(dir.x() / mag);
        dir.setY(dir.y() / mag);
        QPointF end = origin + dir * kArrowLength;

        OverlayStyle style;
        style.penColor = color;
        style.penWidth = 2.0;
        style.z = kArrowZ;

        builder.addArrow(origin, end, kArrowHeadLength, kArrowHeadWidth, style);
    };

    auto addLabel = [&](const QPointF& pos, const QString& text, const QColor& color) {
        OverlayStyle textStyle;
        textStyle.penColor = color;
        textStyle.z = kLabelZ;

        QFont font;
        font.setPointSizeF(9.0);

        builder.addText(pos, text, font, textStyle);
    };

    auto addMarker = [&](const QPointF& center, const QColor& color, float radius) {
        OverlayStyle style;
        style.penColor = Qt::black;
        style.penWidth = 1.0;
        style.brushColor = color;
        style.z = kMarkerZ;
        builder.addCircle(center, radius, true, style);
    };

    QuadSurface* segSurface = nullptr;
    std::shared_ptr<Surface> segSurfaceHolder;  // Keep surface alive during this scope
    if (viewer->surfName() == "segmentation") {
        segSurface = dynamic_cast<QuadSurface*>(currentSurface);
    } else if (_surfaces) {
        segSurfaceHolder = _surfaces->surface("segmentation");
        segSurface = dynamic_cast<QuadSurface*>(segSurfaceHolder.get());
    }

    auto fetchFocusScene = [&](QPointF& anchor) {
        if (!segSurface || !_surfaces) {
            return;
        }
        if (auto* poi = _surfaces->poi("focus")) {
            auto ptr = segSurface->pointer();
            auto* patchIndex = manager() ? manager()->surfacePatchIndex() : nullptr;
            float dist = segSurface->pointTo(ptr, poi->p, 4.0, 100, patchIndex);
            if (dist >= 0 && dist < 20.0f / scale) {
                cv::Vec3f sp = segSurface->loc(ptr) * scale;
                anchor = QPointF(sp[0], sp[1]);
            }
        }
    };

    if (viewer->surfName() == "segmentation") {
        auto* quad = dynamic_cast<QuadSurface*>(currentSurface);
        if (!quad) {
            return;
        }

        fetchFocusScene(anchorScene);

        QPointF upOffset(0.0, -20.0);
        QPointF downOffset(0.0, 20.0);

        addArrow(anchorScene + upOffset, QPointF(1.0, 0.0), kArrowFalseColor);
        addArrow(anchorScene + downOffset, QPointF(-1.0, 0.0), kArrowTrueColor);

        addLabel(anchorScene + upOffset + QPointF(8.0, -8.0), QStringLiteral("false"), kArrowFalseColor);
        addLabel(anchorScene + downOffset + QPointF(8.0, -8.0), QStringLiteral("true"), kArrowTrueColor);

        auto ptr = quad->pointer();
        if (_surfaces) {
            if (auto* poi = _surfaces->poi("focus")) {
                auto* patchIndex = manager() ? manager()->surfacePatchIndex() : nullptr;
                quad->pointTo(ptr, poi->p, 4.0, 100, patchIndex);
            }
        }

        cv::Vec3f centerParam = quad->loc(ptr) * scale;
        addMarker(QPointF(centerParam[0], centerParam[1]), kCenterColor, kStepCenterRadius);

        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool useSegStep = settings.value(viewer::USE_SEG_STEP_FOR_HINTS, viewer::USE_SEG_STEP_FOR_HINTS_DEFAULT).toBool();
        int numPoints = std::max(0, std::min(100, settings.value(viewer::DIRECTION_STEP_POINTS, viewer::DIRECTION_STEP_POINTS_DEFAULT).toInt()));
        float stepVal = settings.value(viewer::DIRECTION_STEP, static_cast<float>(viewer::DIRECTION_STEP_DEFAULT)).toFloat();
        if (useSegStep && quad->meta) {
            try {
                if (quad->meta->contains("vc_grow_seg_from_segments_params")) {
                    auto& p = quad->meta->at("vc_grow_seg_from_segments_params");
                    if (p.contains("step")) {
                        stepVal = p.at("step").get<float>();
                    }
                }
            } catch (...) {
                // keep default
            }
        }
        if (stepVal <= 0.0f) {
            stepVal = settings.value(viewer::DIRECTION_STEP, static_cast<float>(viewer::DIRECTION_STEP_DEFAULT)).toFloat();
        }

        for (int n = 1; n <= numPoints; ++n) {
            cv::Vec3f pos = quad->loc(ptr, {n * stepVal, 0, 0}) * scale;
            addMarker(QPointF(pos[0], pos[1]), kArrowFalseColor, kStepMarkerRadius);

            cv::Vec3f neg = quad->loc(ptr, {-n * stepVal, 0, 0}) * scale;
            addMarker(QPointF(neg[0], neg[1]), kArrowTrueColor, kStepMarkerRadius);
        }
        return;
    }

    if (auto* plane = dynamic_cast<PlaneSurface*>(currentSurface)) {
        if (!segSurface) {
            return;
        }

        QPointF upOffset(0.0, -10.0);
        QPointF downOffset(0.0, 10.0);

        cv::Vec3f targetWP = plane->origin();
        if (_surfaces) {
            if (auto* poi = _surfaces->poi("focus")) {
                targetWP = poi->p;
            }
        }

        auto segPtr = segSurface->pointer();
        auto* patchIndex = manager() ? manager()->surfacePatchIndex() : nullptr;
        segSurface->pointTo(segPtr, targetWP, 4.0, 100, patchIndex);

        cv::Vec3f p0 = segSurface->coord(segPtr, {0, 0, 0});
        if (p0[0] == -1.0f) {
            return;
        }

        const float stepNominal = 2.0f;
        cv::Vec3f p1 = segSurface->coord(segPtr, {stepNominal, 0, 0});
        cv::Vec3f dir3 = p1 - p0;
        float len = std::sqrt(dir3.dot(dir3));
        if (len < 1e-5f) {
            return;
        }
        dir3 *= (1.0f / len);

        cv::Vec3f s0 = plane->project(p0, 1.0f, scale);
        QPointF anchor(QPointF(s0[0], s0[1]));

        cv::Vec3f s1 = plane->project(p0 + dir3 * (kArrowLength / scale), 1.0f, scale);
        QPointF dir2(s1[0] - s0[0], s1[1] - s0[1]);
        if (std::hypot(dir2.x(), dir2.y()) < 1e-3) {
            return;
        }

        addArrow(anchor + upOffset, dir2, kArrowFalseColor);
        addArrow(anchor + downOffset, QPointF(-dir2.x(), -dir2.y()), kArrowTrueColor);

        QPointF redTip = anchor + upOffset + dir2;
        QPointF greenTip = anchor + downOffset - dir2;
        addLabel(redTip + QPointF(8.0, -8.0), QStringLiteral("false"), kArrowFalseColor);
        addLabel(greenTip + QPointF(8.0, -8.0), QStringLiteral("true"), kArrowTrueColor);

        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool useSegStep = settings.value(viewer::USE_SEG_STEP_FOR_HINTS, viewer::USE_SEG_STEP_FOR_HINTS_DEFAULT).toBool();
        int numPoints = std::max(0, std::min(100, settings.value(viewer::DIRECTION_STEP_POINTS, viewer::DIRECTION_STEP_POINTS_DEFAULT).toInt()));
        float stepVal = settings.value(viewer::DIRECTION_STEP, static_cast<float>(viewer::DIRECTION_STEP_DEFAULT)).toFloat();
        if (useSegStep && segSurface->meta) {
            try {
                if (segSurface->meta->contains("vc_grow_seg_from_segments_params")) {
                    auto& p = segSurface->meta->at("vc_grow_seg_from_segments_params");
                    if (p.contains("step")) {
                        stepVal = p.at("step").get<float>();
                    }
                }
            } catch (...) {
                // keep default
            }
        }
        if (stepVal <= 0.0f) {
            stepVal = settings.value(viewer::DIRECTION_STEP, static_cast<float>(viewer::DIRECTION_STEP_DEFAULT)).toFloat();
        }

        addMarker(anchor, kCenterColor, kStepCenterRadius);

        for (int n = 1; n <= numPoints; ++n) {
            cv::Vec3f pPos = segSurface->coord(segPtr, {n * stepVal, 0, 0});
            cv::Vec3f pNeg = segSurface->coord(segPtr, {-n * stepVal, 0, 0});
            if (pPos[0] != -1) {
                cv::Vec3f s = plane->project(pPos, 1.0f, scale);
                addMarker(QPointF(s[0], s[1]), kArrowFalseColor, kStepMarkerRadius);
            }
            if (pNeg[0] != -1) {
                cv::Vec3f s = plane->project(pNeg, 1.0f, scale);
                addMarker(QPointF(s[0], s[1]), kArrowTrueColor, kStepMarkerRadius);
            }
        }
    }
}

void VectorOverlayController::collectSurfaceNormals(CVolumeViewer* viewer,
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

    constexpr float kArrowLen = 50.0f;
    // Colors: Blue = +U, Green = +V, Red = +Z (normal)
    const QColor kUColor(0, 100, 255);
    const QColor kVColor(0, 200, 0);
    const QColor kZColor(255, 50, 50);

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

        const float gridToSceneX = viewerScale / surfScale[0];
        const float gridToSceneY = viewerScale / surfScale[1];
        const float centerOffsetX = (cols / 2.0f) * gridToSceneX;
        const float centerOffsetY = (rows / 2.0f) * gridToSceneY;

        constexpr int kTargetSamplesPerAxis = 32;
        const int strideR = std::max(1, rows / kTargetSamplesPerAxis);
        const int strideC = std::max(1, cols / kTargetSamplesPerAxis);

        auto drawAxisArrow = [&](const QPointF& origin, const cv::Vec3f& dir3d, const QColor& color) {
            OverlayStyle style;
            style.penColor = color;
            style.penWidth = 3.0;
            style.z = kArrowZ;

            QPointF dir2d(dir3d[0], dir3d[1]);
            float len2d = std::sqrt(dir2d.x() * dir2d.x() + dir2d.y() * dir2d.y());

            if (len2d < 0.1f) {
                OverlayStyle dotStyle;
                dotStyle.penColor = color;
                dotStyle.brushColor = color;
                dotStyle.penWidth = 1.0;
                dotStyle.z = kArrowZ;
                builder.addCircle(origin, 3.0f, true, dotStyle);
            } else {
                dir2d /= len2d;
                QPointF end = origin + dir2d * kArrowLen;
                builder.addArrow(origin, end, 5.0, 3.0, style);
            }
        };

        for (int r = 0; r < rows; r += strideR) {
            for (int c = 0; c < cols; c += strideC) {
                const cv::Vec3f& p = (*points)(r, c);
                if (p[0] == -1.0f) continue;

                cv::Vec3f pRight(-1, -1, -1), pLeft(-1, -1, -1);
                cv::Vec3f pDown(-1, -1, -1), pUp(-1, -1, -1);

                for (int d = 1; d <= 5; ++d) {
                    if (pRight[0] == -1.0f && c + d < cols && (*points)(r, c + d)[0] != -1.0f)
                        pRight = (*points)(r, c + d);
                    if (pLeft[0] == -1.0f && c - d >= 0 && (*points)(r, c - d)[0] != -1.0f)
                        pLeft = (*points)(r, c - d);
                    if (pDown[0] == -1.0f && r + d < rows && (*points)(r + d, c)[0] != -1.0f)
                        pDown = (*points)(r + d, c);
                    if (pUp[0] == -1.0f && r - d >= 0 && (*points)(r - d, c)[0] != -1.0f)
                        pUp = (*points)(r - d, c);
                }

                bool hasU = (pRight[0] != -1.0f && pLeft[0] != -1.0f);
                bool hasV = (pDown[0] != -1.0f && pUp[0] != -1.0f);
                if (!hasU && !hasV) continue;

                QPointF origin(c * gridToSceneX - centerOffsetX, r * gridToSceneY - centerOffsetY);

                cv::Vec3f tangentU(0, 0, 0), tangentV(0, 0, 0);

                if (hasU) {
                    tangentU = pRight - pLeft;
                    float len = std::sqrt(tangentU.dot(tangentU));
                    if (len > 1e-6f) {
                        tangentU /= len;
                        drawAxisArrow(origin, tangentU, kUColor);
                    }
                }

                if (hasV) {
                    tangentV = pDown - pUp;
                    float len = std::sqrt(tangentV.dot(tangentV));
                    if (len > 1e-6f) {
                        tangentV /= len;
                        drawAxisArrow(origin, tangentV, kVColor);
                    }
                }

                if (hasU && hasV) {
                    cv::Vec3f normal = tangentU.cross(tangentV);
                    float len = std::sqrt(normal.dot(normal));
                    if (len > 1e-6f) {
                        normal /= len;
                        drawAxisArrow(origin, normal, kZColor);
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
    if (_surfaces) {
        segSurfaceHolder = _surfaces->surface("segmentation");
        segSurface = dynamic_cast<QuadSurface*>(segSurfaceHolder.get());
    }
    if (!segSurface) {
        return;
    }

    // Find current position on surface
    cv::Vec3f targetWP = plane->origin();
    if (_surfaces) {
        if (auto* poi = _surfaces->poi("focus")) {
            targetWP = poi->p;
        }
    }

    auto segPtr = segSurface->pointer();
    auto* patchIndex = manager() ? manager()->surfacePatchIndex() : nullptr;
    float dist = segSurface->pointTo(segPtr, targetWP, 4.0, 100, patchIndex);
    if (dist < 0 || dist > 50.0f) {
        return;
    }

    // Helper to draw an arrow projected onto the plane
    auto drawPlaneArrow = [&](const cv::Vec3f& worldPos, const cv::Vec3f& dir3d, const QColor& color) {
        cv::Vec3f p0 = plane->project(worldPos, 1.0f, viewerScale);
        cv::Vec3f p1 = plane->project(worldPos + dir3d * (kArrowLen / viewerScale), 1.0f, viewerScale);

        QPointF origin(p0[0], p0[1]);
        QPointF dir2d(p1[0] - p0[0], p1[1] - p0[1]);
        float len2d = std::sqrt(dir2d.x() * dir2d.x() + dir2d.y() * dir2d.y());

        OverlayStyle style;
        style.penColor = color;
        style.penWidth = 3.0;
        style.z = kArrowZ;

        if (len2d < 2.0f) {
            // Vector is mostly perpendicular to view plane
            OverlayStyle dotStyle;
            dotStyle.penColor = color;
            dotStyle.brushColor = color;
            dotStyle.penWidth = 1.0;
            dotStyle.z = kArrowZ;
            builder.addCircle(origin, 3.0f, true, dotStyle);
        } else {
            dir2d /= len2d;
            QPointF end = origin + dir2d * kArrowLen;
            builder.addArrow(origin, end, 5.0, 3.0, style);
        }
    };

    // Get step size from settings or segment metadata
    float stepVal = 50.0f;  // Default step in nominal coords
    if (segSurface->meta) {
        try {
            if (segSurface->meta->contains("vc_grow_seg_from_segments_params")) {
                auto& p = segSurface->meta->at("vc_grow_seg_from_segments_params");
                if (p.contains("step")) {
                    stepVal = p.at("step").get<float>();
                }
            }
        } catch (...) {}
    }

    // Sample points along the surface, but only draw those close to the viewing plane
    // Walk in both U and V directions to find points that intersect this plane
    constexpr int kMaxSamplesPerSide = 50;
    constexpr int kMaxTotalSamples = 100;
    constexpr float kPlaneDistThreshold = 3.0f;  // Only draw if within 3 voxels of plane

    const float sampleStep = stepVal * 2.0f;
    int samplesDrawn = 0;

    // Sample in a grid pattern around the current position
    for (int nu = -kMaxSamplesPerSide; nu <= kMaxSamplesPerSide && samplesDrawn < kMaxTotalSamples; ++nu) {
        for (int nv = -kMaxSamplesPerSide; nv <= kMaxSamplesPerSide && samplesDrawn < kMaxTotalSamples; ++nv) {
            float offsetU = nu * sampleStep;
            float offsetV = nv * sampleStep;

            cv::Vec3f worldPos = segSurface->coord(segPtr, {offsetU, offsetV, 0});
            if (worldPos[0] == -1.0f) continue;

            // Only draw if this point is close to the viewing plane
            float planeDist = std::abs(plane->pointDist(worldPos));
            if (planeDist > kPlaneDistThreshold) continue;

            // Get tangents by sampling nearby points
            cv::Vec3f pPlusU = segSurface->coord(segPtr, {offsetU + 2.0f, offsetV, 0});
            cv::Vec3f pMinusU = segSurface->coord(segPtr, {offsetU - 2.0f, offsetV, 0});
            cv::Vec3f pPlusV = segSurface->coord(segPtr, {offsetU, offsetV + 2.0f, 0});
            cv::Vec3f pMinusV = segSurface->coord(segPtr, {offsetU, offsetV - 2.0f, 0});

            bool hasU = (pPlusU[0] != -1.0f && pMinusU[0] != -1.0f);
            bool hasV = (pPlusV[0] != -1.0f && pMinusV[0] != -1.0f);

            if (!hasU && !hasV) continue;

            cv::Vec3f tangentU(0, 0, 0), tangentV(0, 0, 0);

            if (hasU) {
                tangentU = pPlusU - pMinusU;
                float len = std::sqrt(tangentU.dot(tangentU));
                if (len > 1e-6f) {
                    tangentU /= len;
                    drawPlaneArrow(worldPos, tangentU, kUColor);
                }
            }

            if (hasV) {
                tangentV = pPlusV - pMinusV;
                float len = std::sqrt(tangentV.dot(tangentV));
                if (len > 1e-6f) {
                    tangentV /= len;
                    drawPlaneArrow(worldPos, tangentV, kVColor);
                }
            }

            if (hasU && hasV) {
                cv::Vec3f normal = tangentU.cross(tangentV);
                float len = std::sqrt(normal.dot(normal));
                if (len > 1e-6f) {
                    normal /= len;
                    drawPlaneArrow(worldPos, normal, kZColor);
                }
            }

            ++samplesDrawn;
        }
    }
}
