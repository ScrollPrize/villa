#include "CVolumeViewer.hpp"
#include "vc/ui/UDataManipulateUtils.hpp"

#include "ViewerManager.hpp"

#include <QGraphicsView>
#include <QGraphicsScene>

#include "CVolumeViewerView.hpp"
#include "CSurfaceCollection.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/Slicing.hpp"

#include <omp.h>

#include <optional>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <utility>


#define COLOR_SEG_YZ Qt::yellow
#define COLOR_SEG_XZ Qt::red
#define COLOR_SEG_XY QColor(255, 140, 0)

#include <algorithm>
#include <cmath>

void CVolumeViewer::renderIntersections()
{
    if (!volume || !volume->zarrDataset() || !_surf)
        return;

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

        const SurfacePatchIndex* patchIndex =
            _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
        if (!patchIndex) {
            clearAllIntersectionItems();
            return;
        }
        const float clipTolerance = std::max(_intersectionThickness, 1e-4f);

        using IntersectionCandidate = std::pair<std::string, QuadSurface*>;
        std::vector<IntersectionCandidate> intersectCandidates;
        intersectCandidates.reserve(_intersect_tgts.size());
        for (const auto& key : _intersect_tgts) {
            Surface* surfacePtr = _surf_col->surface(key);
            if (!surfacePtr) {
                std::cout << "[CVolumeViewer] skip candidate '" << key << "' (surface missing)\n";
                continue;
            }
            auto* segmentation = dynamic_cast<QuadSurface*>(surfacePtr);
            if (!segmentation) {
                std::cout << "[CVolumeViewer] skip candidate '" << key << "' (not QuadSurface)\n";
                continue;
            }

            if (_segmentationEditActive && activeSegSurface && segmentationAliasRequested &&
                segmentation == activeSegSurface && key != "segmentation") {
                removeItemsForKey(key);
                continue;
            }

            intersectCandidates.emplace_back(key, segmentation);
        }

        std::vector<SurfacePatchIndex::TriangleCandidate> triangleCandidates;
        patchIndex->queryTriangles(view_bbox, nullptr, triangleCandidates);

        std::unordered_map<QuadSurface*, std::vector<size_t>> trianglesBySurface;
        trianglesBySurface.reserve(intersectCandidates.size());
        for (size_t idx = 0; idx < triangleCandidates.size(); ++idx) {
            auto* surface = triangleCandidates[idx].surface;
            if (!surface) {
                continue;
            }
            trianglesBySurface[surface].push_back(idx);
        }

        auto intersectionLinesEqual = [](const std::vector<IntersectionLine>& lhs,
                                         const std::vector<IntersectionLine>& rhs) {
            if (lhs.size() != rhs.size()) {
                return false;
            }
            for (size_t idx = 0; idx < lhs.size(); ++idx) {
                const auto& a = lhs[idx];
                const auto& b = rhs[idx];
                if (a.world.size() != b.world.size() ||
                    a.surfaceParams.size() != b.surfaceParams.size()) {
                    return false;
                }
                for (size_t pointIdx = 0; pointIdx < a.world.size(); ++pointIdx) {
                    if (a.world[pointIdx] != b.world[pointIdx]) {
                        return false;
                    }
                }
                for (size_t pointIdx = 0; pointIdx < a.surfaceParams.size(); ++pointIdx) {
                    if (a.surfaceParams[pointIdx] != b.surfaceParams[pointIdx]) {
                        return false;
                    }
                }
            }
            return true;
        };

        size_t colorIndex = 0;
        for (const auto& candidate : intersectCandidates) {
            const auto& key = candidate.first;
            QuadSurface* segmentation = candidate.second;

            const auto trianglesIt = trianglesBySurface.find(segmentation);
            if (trianglesIt == trianglesBySurface.end()) {
                removeItemsForKey(key);
                continue;
            }

            const auto& candidateIndices = trianglesIt->second;

            std::vector<IntersectionLine> intersectionLines;
            intersectionLines.reserve(candidateIndices.size());
            for (size_t candidateIndex : candidateIndices) {
                const auto& triCandidate = triangleCandidates[candidateIndex];
                auto segment = SurfacePatchIndex::clipTriangleToPlane(triCandidate, *plane, clipTolerance);
                if (!segment) {
                    continue;
                }

                IntersectionLine line;
                line.world[0] = segment->world[0];
                line.world[1] = segment->world[1];
                line.surfaceParams[0] = segment->surfaceParams[0];
                line.surfaceParams[1] = segment->surfaceParams[1];
                intersectionLines.push_back(std::move(line));
            }

            QColor col;
            float width = 3;
            int z_value = 5;

            static const QColor palette[] = {
                QColor(255, 50, 50),
                QColor(255, 161, 50),
                QColor(238, 255, 50),
                QColor(128, 255, 50),
                QColor(50, 255, 83),
                QColor(50, 255, 193),
                QColor(50, 206, 255),
                QColor(50, 95, 255),
                QColor(116, 50, 255),
                QColor(226, 50, 255),
                QColor(255, 50, 173),
                QColor(255, 50, 63),
                QColor(255, 148, 50),
                QColor(250, 255, 50),
                QColor(140, 255, 50),
                QColor(50, 255, 71),
                QColor(50, 255, 181),
                QColor(50, 218, 255),
                QColor(50, 108, 255),
                QColor(104, 50, 255),
                QColor(214, 50, 255),
                QColor(255, 50, 185),
                QColor(255, 50, 75),
                QColor(255, 136, 50),
                QColor(255, 246, 50),
                QColor(153, 255, 50),
                QColor(50, 255, 59),
                QColor(50, 255, 169),
                QColor(50, 230, 255),
                QColor(50, 120, 255),
                QColor(91, 50, 255),
                QColor(201, 50, 255),
            };
            col = palette[colorIndex % std::size(palette)];
            ++colorIndex;

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

            std::vector<QGraphicsItem*> items;
            items.reserve(intersectionLines.size());
            for (const auto& line : intersectionLines) {
                if (line.world.size() < 2) {
                    continue;
                }
                QPainterPath path;
                bool first = true;
                for (const auto& wp : line.world) {
                    cv::Vec3f p = plane->project(wp, 1.0, _scale);
                    if (first)
                        path.moveTo(p[0], p[1]);
                    else
                        path.lineTo(p[0], p[1]);
                    first = false;
                }
                auto* item = fGraphicsView->scene()->addPath(path, QPen(col, width));
                item->setZValue(z_value);
                item->setOpacity(_intersectionOpacity);
                if (fBaseImageItem) {
                    item->setParentItem(fBaseImageItem);
                }
                items.push_back(item);
            }

            if (!items.empty()) {
                removeItemsForKey(key);
                _intersect_items[key] = items;
            } else {
                removeItemsForKey(key);
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
    }
    else if (_intersect_items.count(name)) {
        for(auto &item : _intersect_items[name]) {
            fScene->removeItem(item);
            delete item;
        }
        _intersect_items.erase(name);
    }
}


void CVolumeViewer::setIntersects(const std::set<std::string> &set)
{
    _intersect_tgts = set;

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