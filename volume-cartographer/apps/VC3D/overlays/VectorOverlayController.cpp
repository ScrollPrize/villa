#include "VectorOverlayController.hpp"

#include "../CVolumeViewer.hpp"
#include "../CSurfaceCollection.hpp"

#include "vc/core/util/Surface.hpp"

#include <QSettings>

#include <algorithm>
#include <cmath>

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
}

void VectorOverlayController::addProvider(Provider provider)
{
    if (provider) {
        _providers.push_back(std::move(provider));
    }
}

bool VectorOverlayController::isOverlayEnabledFor(CVolumeViewer* viewer) const
{
    if (!viewer || !viewer->isShowDirectionHints()) {
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

    auto* segSurface = dynamic_cast<QuadSurface*>(viewer->surfName() == "segmentation"
                                                      ? currentSurface
                                                      : (_surfaces ? dynamic_cast<QuadSurface*>(_surfaces->surface("segmentation")) : nullptr));

    auto fetchFocusScene = [&](QPointF& anchor) {
        if (!segSurface || !_surfaces) {
            return;
        }
        if (auto* poi = _surfaces->poi("focus")) {
            auto ptr = segSurface->pointer();
            float dist = segSurface->pointTo(ptr, poi->p, 4.0, 100);
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
                quad->pointTo(ptr, poi->p, 4.0, 100);
            }
        }

        cv::Vec3f centerParam = quad->loc(ptr) * scale;
        addMarker(QPointF(centerParam[0], centerParam[1]), kCenterColor, kStepCenterRadius);

        QSettings settings("VC.ini", QSettings::IniFormat);
        bool useSegStep = settings.value("viewer/use_seg_step_for_hints", true).toBool();
        int numPoints = std::max(0, std::min(100, settings.value("viewer/direction_step_points", 5).toInt()));
        float stepVal = settings.value("viewer/direction_step", 10.0f).toFloat();
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
            stepVal = settings.value("viewer/direction_step", 10.0f).toFloat();
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
        segSurface->pointTo(segPtr, targetWP, 4.0, 100);

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

        QSettings settings("VC.ini", QSettings::IniFormat);
        bool useSegStep = settings.value("viewer/use_seg_step_for_hints", true).toBool();
        int numPoints = std::max(0, std::min(100, settings.value("viewer/direction_step_points", 5).toInt()));
        float stepVal = settings.value("viewer/direction_step", 10.0f).toFloat();
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
            stepVal = settings.value("viewer/direction_step", 10.0f).toFloat();
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
