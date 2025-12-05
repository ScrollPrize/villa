#include "CVolumeViewer.hpp"
#include "vc/ui/UDataManipulateUtils.hpp"

#include "ViewerManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QPainterPath>

#include "CVolumeViewerView.hpp"
#include "CSurfaceCollection.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/Slicing.hpp"

#include <omp.h>

#include <iostream>
#include <optional>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <utility>


#define COLOR_SEG_YZ Qt::yellow
#define COLOR_SEG_XZ Qt::red
#define COLOR_SEG_XY QColor(255, 140, 0)
#define COLOR_APPROVED QColor(0, 200, 0)
#define COLOR_PENDING QColor(0, 150, 255)
#define COLOR_PENDING_UNAPPROVE QColor(255, 50, 50)

#include <algorithm>
#include <cmath>

// Helper to compare intersection lines for caching
static bool intersectionLinesEqual(const std::vector<IntersectionLine>& a,
                                   const std::vector<IntersectionLine>& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    constexpr float epsilon = 1e-4f;
    for (size_t i = 0; i < a.size(); ++i) {
        for (int j = 0; j < 2; ++j) {
            const auto& wa = a[i].world[j];
            const auto& wb = b[i].world[j];
            if (std::abs(wa[0] - wb[0]) > epsilon ||
                std::abs(wa[1] - wb[1]) > epsilon ||
                std::abs(wa[2] - wb[2]) > epsilon) {
                return false;
            }
        }
    }
    return true;
}

void CVolumeViewer::renderIntersections()
{
    if (!volume || !volume->zarrDataset() || !_surf)
        return;

    // Refresh cached surface pointers if targets changed or if a surface object
    // was swapped out during an edit (common for segmentation updates).
    auto rebuildCachedIntersectSurfaces = [&]() {
        _cachedIntersectSurfaces.clear();
        if (!_surf_col) {
            return;
        }
        _cachedIntersectSurfaces.reserve(_intersect_tgts.size());
        for (const auto& key : _intersect_tgts) {
            if (auto* surf = dynamic_cast<QuadSurface*>(_surf_col->surface(key))) {
                _cachedIntersectSurfaces[key] = surf;
            }
        }
    };

    bool refreshCachedSurfaces = _cachedIntersectSurfaces.size() != _intersect_tgts.size();
    if (!refreshCachedSurfaces && _surf_col) {
        for (const auto& key : _intersect_tgts) {
            auto* current = dynamic_cast<QuadSurface*>(_surf_col->surface(key));
            const auto cachedIt = _cachedIntersectSurfaces.find(key);
            if (cachedIt == _cachedIntersectSurfaces.end() || cachedIt->second != current) {
                refreshCachedSurfaces = true;
                break;
            }
        }
    }
    if (refreshCachedSurfaces) {
        rebuildCachedIntersectSurfaces();
    }

    const QRectF viewRect = fGraphicsView
        ? fGraphicsView->mapToScene(fGraphicsView->viewport()->geometry()).boundingRect()
        : QRectF(curr_img_area);

    auto removeItemsForKey = [&](const std::string& key) {
        auto it = _intersect_items.find(key);
        if (it == _intersect_items.end()) {
            return;
        }
        for (auto* item : it->second) {
            fScene->removeItem(item);
            delete item;
        }
        _intersect_items.erase(it);
    };

    auto clearAllIntersectionItems = [&]() {
        std::vector<std::string> keys;
        keys.reserve(_intersect_items.size());
        for (const auto& pair : _intersect_items) {
            keys.push_back(pair.first);
        }
        for (const auto& key : keys) {
            removeItemsForKey(key);
        }
    };

    PlaneSurface *plane = dynamic_cast<PlaneSurface*>(_surf);
    QuadSurface* activeSegSurface =
        _surf_col ? dynamic_cast<QuadSurface*>(_surf_col->surface("segmentation")) : nullptr;
    const bool segmentationAliasRequested = _intersect_tgts.count("segmentation") > 0;


    if (plane) {
        cv::Rect plane_roi = {static_cast<int>(viewRect.x()/_scale),
                              static_cast<int>(viewRect.y()/_scale),
                              static_cast<int>(viewRect.width()/_scale),
                              static_cast<int>(viewRect.height()/_scale)};
        // Enlarge the sampled region so nearby intersections outside the viewport still get clipped.
        const int dominantSpan = std::max(plane_roi.width, plane_roi.height);
        const int planeRoiPadding = 8;
        plane_roi.x -= planeRoiPadding;
        plane_roi.y -= planeRoiPadding;
        plane_roi.width += planeRoiPadding * 2;
        plane_roi.height += planeRoiPadding * 2;

        cv::Vec3f corner = plane->coord(cv::Vec3f(0,0,0), {static_cast<float>(plane_roi.x), static_cast<float>(plane_roi.y), 0.0});
        Rect3D view_bbox = {corner, corner};
        view_bbox = expand_rect(view_bbox, plane->coord(cv::Vec3f(0,0,0), {static_cast<float>(plane_roi.br().x), static_cast<float>(plane_roi.y), 0}));
        view_bbox = expand_rect(view_bbox, plane->coord(cv::Vec3f(0,0,0), {static_cast<float>(plane_roi.x), static_cast<float>(plane_roi.br().y), 0}));
        view_bbox = expand_rect(view_bbox, plane->coord(cv::Vec3f(0,0,0), {static_cast<float>(plane_roi.br().x), static_cast<float>(plane_roi.br().y), 0}));
        const cv::Vec3f bboxExtent = view_bbox.high - view_bbox.low;
        const float maxExtent = std::max(std::abs(bboxExtent[0]),
                              std::max(std::abs(bboxExtent[1]), std::abs(bboxExtent[2])));
        const float viewPadding = std::max(64.0f, maxExtent * 0.25f);
        view_bbox.low -= cv::Vec3f(viewPadding, viewPadding, viewPadding);
        view_bbox.high += cv::Vec3f(viewPadding, viewPadding, viewPadding);

        using IntersectionCandidate = std::pair<std::string, QuadSurface*>;
        std::vector<IntersectionCandidate> intersectCandidates;
        intersectCandidates.reserve(_cachedIntersectSurfaces.size());
        for (const auto& [key, segmentation] : _cachedIntersectSurfaces) {
            if (!segmentation) {
                continue;
            }

            if (_segmentationEditActive && activeSegSurface && segmentationAliasRequested &&
                segmentation == activeSegSurface && key != "segmentation") {
                removeItemsForKey(key);
                continue;
            }

            intersectCandidates.emplace_back(key, segmentation);
        }

        // Build set of surfaces to filter query (avoids processing irrelevant triangles)
        std::unordered_set<QuadSurface*> targetSurfaces;
        targetSurfaces.reserve(intersectCandidates.size());
        for (const auto& candidate : intersectCandidates) {
            targetSurfaces.insert(candidate.second);
        }

        const SurfacePatchIndex* patchIndex =
            _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
        if (!patchIndex) {
            clearAllIntersectionItems();
            return;
        }

        const float clipTolerance = std::max(_intersectionThickness, 1e-4f);

        // Use member buffers to preserve capacity across frames
        patchIndex->queryTriangles(view_bbox, targetSurfaces, _triangleCandidates);

        // Clear and rebuild surface mapping (reuses allocated vectors)
        for (auto& [surf, indices] : _trianglesBySurface) {
            indices.clear();
        }
        for (size_t idx = 0; idx < _triangleCandidates.size(); ++idx) {
            auto* surface = _triangleCandidates[idx].surface;
            if (surface) {
                _trianglesBySurface[surface].push_back(idx);
            }
        }

        for (const auto& candidate : intersectCandidates) {
            const auto& key = candidate.first;
            QuadSurface* segmentation = candidate.second;

            const auto trianglesIt = _trianglesBySurface.find(segmentation);
            if (trianglesIt == _trianglesBySurface.end()) {
                removeItemsForKey(key);
                continue;
            }

            const auto& candidateIndices = trianglesIt->second;
            const size_t numCandidates = candidateIndices.size();

            // Parallel triangle clipping - each thread writes to its own slot
            std::vector<std::optional<IntersectionLine>> clipResults(numCandidates);

            #pragma omp parallel for schedule(dynamic, 64)
            for (size_t k = 0; k < numCandidates; ++k) {
                const auto& triCandidate = _triangleCandidates[candidateIndices[k]];
                auto segment = SurfacePatchIndex::clipTriangleToPlane(triCandidate, *plane, clipTolerance);
                if (segment) {
                    IntersectionLine line;
                    line.world[0] = segment->world[0];
                    line.world[1] = segment->world[1];
                    line.surfaceParams[0] = segment->surfaceParams[0];
                    line.surfaceParams[1] = segment->surfaceParams[1];
                    clipResults[k] = std::move(line);
                }
            }

            // Collect non-null results
            std::vector<IntersectionLine> intersectionLines;
            intersectionLines.reserve(numCandidates);
            for (auto& result : clipResults) {
                if (result) {
                    intersectionLines.push_back(std::move(*result));
                }
            }

            // Check if intersection lines match cached - if so, skip expensive recreation
            auto cachedIt = _cachedIntersectionLines.find(key);
            const bool hasCache = cachedIt != _cachedIntersectionLines.end();
            const bool hasExistingItems = _intersect_items.count(key) && !_intersect_items[key].empty();
            if (hasCache && hasExistingItems && intersectionLinesEqual(intersectionLines, cachedIt->second)) {
                // Lines unchanged - just update opacity and visibility on existing items and continue
                for (auto* item : _intersect_items[key]) {
                    item->setVisible(true);
                    item->setOpacity(_intersectionOpacity);
                }
                continue;
            }

            // Update cache
            _cachedIntersectionLines[key] = intersectionLines;

            QColor col;
            float width = 3;
            int z_value = 5;

            static const QColor palette[] = {
                // Vibrant saturated colors
                QColor(80, 180, 255),   // sky blue
                QColor(180, 80, 220),   // violet
                QColor(80, 220, 200),   // aqua/teal
                QColor(220, 80, 180),   // magenta
                QColor(80, 130, 255),   // medium blue
                QColor(160, 80, 255),   // purple
                QColor(80, 255, 220),   // cyan
                QColor(255, 80, 200),   // hot pink
                QColor(120, 220, 80),   // lime green
                QColor(80, 180, 120),   // spring green

                // Lighter/pastel variants
                QColor(150, 200, 255),  // light sky blue
                QColor(200, 150, 230),  // light violet
                QColor(150, 230, 210),  // light aqua
                QColor(230, 150, 200),  // light magenta
                QColor(150, 170, 255),  // light blue
                QColor(190, 150, 255),  // light purple
                QColor(150, 255, 230),  // light cyan
                QColor(255, 150, 210),  // light pink
                QColor(180, 240, 150),  // light lime
                QColor(150, 230, 170),  // light spring green

                // Deeper/darker variants
                QColor(50, 120, 200),   // deep blue
                QColor(140, 50, 180),   // deep violet
                QColor(50, 180, 160),   // deep teal
                QColor(180, 50, 140),   // deep magenta
                QColor(50, 90, 200),    // navy blue
                QColor(120, 50, 200),   // deep purple
                QColor(50, 200, 180),   // deep cyan
                QColor(200, 50, 160),   // deep pink
                QColor(80, 160, 60),    // forest green
                QColor(50, 140, 100),   // deep sea green

                // Extra variations with different saturation
                QColor(100, 160, 220),  // muted blue
                QColor(160, 100, 200),  // muted violet
                QColor(100, 200, 180),  // muted teal
                QColor(200, 100, 170),  // muted magenta
                QColor(120, 180, 240),  // soft blue
                QColor(180, 120, 220),  // soft purple
                QColor(120, 220, 200),  // soft cyan
                QColor(220, 120, 190),  // soft pink
                QColor(140, 200, 100),  // soft lime
                QColor(100, 180, 130),  // muted green
            };

            // Persistent color assignment: once a surface gets a color, it keeps it (up to 500 surfaces)
            size_t colorIndex;
            auto colorIt = _surfaceColorAssignments.find(key);
            if (colorIt != _surfaceColorAssignments.end()) {
                colorIndex = colorIt->second;
            } else if (_surfaceColorAssignments.size() < 500) {
                colorIndex = _nextColorIndex++;
                _surfaceColorAssignments[key] = colorIndex;
            } else {
                // Over 500 surfaces - fall back to hash-based assignment
                colorIndex = std::hash<std::string>{}(key);
            }
            col = palette[colorIndex % std::size(palette)];

            const bool isActiveSegmentation =
                activeSegSurface && segmentation == activeSegSurface;
            if (isActiveSegmentation) {
                col = (_surf_name == "seg yz"   ? COLOR_SEG_YZ
                       : _surf_name == "seg xz" ? COLOR_SEG_XZ
                                                : COLOR_SEG_XY);
                width = 3;
                z_value = 20;
            }

            if (!_highlightedSurfaceIds.empty() && _highlightedSurfaceIds.count(key)) {
                col = QColor(0, 220, 255);
                width = 4;
                z_value = 30;
            }

            // Get the segmentation overlay for approval mask checking (only for active segmentation)
            SegmentationOverlayController* segOverlay = nullptr;
            if (isActiveSegmentation && _viewerManager) {
                segOverlay = _viewerManager->segmentationOverlay();
            }
            const bool checkApproval = segOverlay && segOverlay->hasApprovalMaskData();

            // Cache surface properties for coordinate conversion (constant for all lines from this surface)
            float approvalOffsetX = 0.0f;
            float approvalOffsetY = 0.0f;
            if (checkApproval) {
                const cv::Vec3f center = segmentation->center();
                const cv::Vec2f scale = segmentation->scale();
                approvalOffsetX = center[0] * scale[0];
                approvalOffsetY = center[1] * scale[1];
            }

            // Batch lines by style (color/width/z) to reduce QGraphicsItem count
            struct LineStyle {
                QColor color;
                float width;
                int z;
                bool operator==(const LineStyle& o) const {
                    return color == o.color && width == o.width && z == o.z;
                }
            };
            struct LineStyleHash {
                size_t operator()(const LineStyle& s) const {
                    return std::hash<int>()(s.color.rgba()) ^
                           std::hash<int>()(static_cast<int>(s.width * 100)) ^
                           std::hash<int>()(s.z);
                }
            };
            std::unordered_map<LineStyle, QPainterPath, LineStyleHash> batchedPaths;

            for (const auto& line : intersectionLines) {
                // Determine color and width based on approval status
                QColor lineColor = col;
                float lineWidth = width;
                int lineZ = z_value;

                if (checkApproval) {
                    const float absCol0 = line.surfaceParams[0][0] + approvalOffsetX;
                    const float absRow0 = line.surfaceParams[0][1] + approvalOffsetY;
                    const float absCol1 = line.surfaceParams[1][0] + approvalOffsetX;
                    const float absRow1 = line.surfaceParams[1][1] + approvalOffsetY;

                    int status0 = 0, status1 = 0;
                    const float intensity0 = segOverlay->queryApprovalBilinear(absRow0, absCol0, &status0);
                    const float intensity1 = segOverlay->queryApprovalBilinear(absRow1, absCol1, &status1);

                    const int approvalState = std::max(status0, status1);
                    const float approvalIntensity = std::max(intensity0, intensity1);

                    if (approvalState > 0 && approvalIntensity > 0.0f) {
                        QColor baseColor;
                        if (approvalState == 3) {
                            baseColor = COLOR_PENDING_UNAPPROVE;
                        } else if (approvalState == 2) {
                            baseColor = COLOR_PENDING;
                        } else {
                            baseColor = COLOR_APPROVED;
                        }

                        const float blendFactor = std::min(1.0f, approvalIntensity * 2.0f);
                        lineColor = QColor(
                            static_cast<int>(col.red() * (1.0f - blendFactor) + baseColor.red() * blendFactor),
                            static_cast<int>(col.green() * (1.0f - blendFactor) + baseColor.green() * blendFactor),
                            static_cast<int>(col.blue() * (1.0f - blendFactor) + baseColor.blue() * blendFactor),
                            baseColor.alpha()
                        );

                        const float extraWidth = 6.0f * blendFactor;
                        lineWidth = width + extraWidth;
                        lineZ = z_value + 5;
                    }
                }

                // Add line to batched path for this style
                LineStyle style{lineColor, lineWidth, lineZ};
                QPainterPath& path = batchedPaths[style];
                cv::Vec3f p0 = plane->project(line.world[0], 1.0, _scale);
                cv::Vec3f p1 = plane->project(line.world[1], 1.0, _scale);
                path.moveTo(p0[0], p0[1]);
                path.lineTo(p1[0], p1[1]);
            }

            // Create one QGraphicsItem per style batch
            std::vector<QGraphicsItem*> items;
            items.reserve(batchedPaths.size());
            for (const auto& [style, path] : batchedPaths) {
                auto* item = fGraphicsView->scene()->addPath(path, QPen(style.color, style.width));
                item->setZValue(style.z);
                item->setOpacity(_intersectionOpacity);
                if (fBaseImageItem) {
                    item->setParentItem(fBaseImageItem);
                }
                items.push_back(item);
            }

            // Always remove old items before storing new ones (even if empty)
            removeItemsForKey(key);
            if (!items.empty()) {
                _intersect_items[key] = items;
            }
        }

        // Remove stale intersections that are no longer requested.
        std::vector<std::string> planeKeysToRemove;
        for (const auto& entry : _intersect_items) {
            if (!_intersect_tgts.count(entry.first)) {
                planeKeysToRemove.push_back(entry.first);
            }
        }
        for (const auto& key : planeKeysToRemove) {
            removeItemsForKey(key);
        }

    }
}


void CVolumeViewer::invalidateIntersect(const std::string &name)
{
    if (!name.size() || name == _surf_name) {
        for(auto &pair : _intersect_items) {
            for(auto &item : pair.second) {
                fScene->removeItem(item);
                delete item;
            }
        }
        _intersect_items.clear();
        _cachedIntersectionLines.clear();
        _cachedIntersectSurfaces.clear();
    }
    else if (_intersect_tgts.count(name)) {
        // Clear items and caches for this intersection target
        if (_intersect_items.count(name)) {
            for(auto &item : _intersect_items[name]) {
                fScene->removeItem(item);
                delete item;
            }
            _intersect_items.erase(name);
        }
        _cachedIntersectionLines.erase(name);
        // Also clear the surface cache for this key to force re-lookup
        // This handles cases where the surface object changed (e.g., switching segments)
        _cachedIntersectSurfaces.erase(name);
    }
}


void CVolumeViewer::setIntersects(const std::set<std::string> &set)
{
    _intersect_tgts = set;

    // Rebuild cached surface pointers to avoid string lookups during render
    _cachedIntersectSurfaces.clear();
    if (_surf_col) {
        _cachedIntersectSurfaces.reserve(set.size());
        for (const auto& key : set) {
            if (auto* surf = dynamic_cast<QuadSurface*>(_surf_col->surface(key))) {
                _cachedIntersectSurfaces[key] = surf;
            }
        }
    }

    renderIntersections();
}


void CVolumeViewer::setIntersectionOpacity(float opacity)
{
    _intersectionOpacity = std::clamp(opacity, 0.0f, 1.0f);
    for (auto& pair : _intersect_items) {
        for (auto* item : pair.second) {
            if (item) {
                item->setOpacity(_intersectionOpacity);
            }
        }
    }
}

void CVolumeViewer::setIntersectionThickness(float thickness)
{
    thickness = std::max(0.0f, thickness);
    if (std::abs(thickness - _intersectionThickness) < 1e-6f) {
        return;
    }
    _intersectionThickness = thickness;
    renderIntersections();
}
