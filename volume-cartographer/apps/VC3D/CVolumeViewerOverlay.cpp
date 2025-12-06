#include "CVolumeViewer.hpp"
#include "vc/ui/UDataManipulateUtils.hpp"

#include "VolumeViewerCmaps.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsItem>

#include "CVolumeViewerView.hpp"
#include "CSurfaceCollection.hpp"
#include "vc/ui/VCCollection.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/Slicing.hpp"

#include <omp.h>


#include <QPainter>
#include <QScopedValueRollback>

#include <optional>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "vc/core/util/Geometry.hpp"

bool scene2vol(cv::Vec3f &p, cv::Vec3f &n, Surface *_surf, const std::string &_surf_name, CSurfaceCollection *_surf_col, const QPointF &scene_loc, const cv::Vec2f &_vis_center, float _ds_scale);


void CVolumeViewer::setOverlayVolume(std::shared_ptr<Volume> volume)
{
    if (_overlayVolume == volume) {
        return;
    }
    _overlayVolume = std::move(volume);

    renderVisible(true);
}

void CVolumeViewer::setOverlayOpacity(float opacity)
{
    float clamped = std::clamp(opacity, 0.0f, 1.0f);
    if (std::abs(clamped - _overlayOpacity) < 1e-6f) {
        return;
    }
    _overlayOpacity = clamped;
    if (_overlayVolume) {
        renderVisible(true);
    }
}

void CVolumeViewer::setOverlayColormap(const std::string& colormapId)
{
    if (_overlayColormapId == colormapId) {
        return;
    }
    _overlayColormapId = colormapId;
    if (_overlayVolume) {
        renderVisible(true);
    }
}

void CVolumeViewer::setOverlayThreshold(float threshold)
{
    setOverlayWindow(std::max(threshold, 0.0f), _overlayWindowHigh);
}


void CVolumeViewer::setOverlayWindow(float low, float high)
{
    constexpr float kMaxOverlayValue = 255.0f;
    const float clampedLow = std::clamp(low, 0.0f, kMaxOverlayValue);
    float clampedHigh = std::clamp(high, 0.0f, kMaxOverlayValue);
    if (clampedHigh <= clampedLow) {
        clampedHigh = std::min(kMaxOverlayValue, clampedLow + 1.0f);
    }

    const bool unchanged = std::abs(clampedLow - _overlayWindowLow) < 1e-6f &&
                           std::abs(clampedHigh - _overlayWindowHigh) < 1e-6f;
    if (unchanged) {
        return;
    }

    _overlayWindowLow = clampedLow;
    _overlayWindowHigh = clampedHigh;

    if (_overlayVolume) {
        renderVisible(true);
    }
}

const std::vector<CVolumeViewer::OverlayColormapEntry>& CVolumeViewer::overlayColormapEntries()
{
    static std::vector<OverlayColormapEntry> entries;
    static bool initialized = false;
    if (!initialized) {
        const auto& sharedEntries = volume_viewer_cmaps::entries();
        entries.reserve(sharedEntries.size());
        for (const auto& entry : sharedEntries) {
            entries.push_back({entry.label, entry.id});
        }
        initialized = true;
    }
    return entries;
}


void CVolumeViewer::updateAllOverlays()
{
    // Validate _surf against collection to prevent use-after-free
    auto surf = _surf_weak.lock();
    if (!surf) {
        return;
    }

    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        POI *poi = _surf_col->poi("focus");
        if (poi) {
            cv::Vec3f planeOrigin = plane->origin();
            // If plane origin differs from POI, update POI
            if (std::abs(poi->p[2] - planeOrigin[2]) > 0.01) {
                poi->p = planeOrigin;
                _surf_col->setPOI("focus", poi);  // NOW we do the expensive update
                emit sendZSliceChanged(static_cast<int>(poi->p[2]));
            }
        }
    }

    QPoint viewportPos = fGraphicsView->mapFromGlobal(QCursor::pos());
    QPointF scenePos = fGraphicsView->mapToScene(viewportPos);

    cv::Vec3f p, n;
    if (scene2vol(p, n, surf.get(), _surf_name, _surf_col, scenePos, _vis_center, _scale)) {
        POI *cursor = _surf_col->poi("cursor");
        if (!cursor)
            cursor = new POI;
        cursor->p = p;
        _surf_col->setPOI("cursor", cursor);
    }

    if (_point_collection && _dragged_point_id == 0) {
        uint64_t old_highlighted_id = _highlighted_point_id;
        _highlighted_point_id = 0;

        const float highlight_dist_threshold = 10.0f;
        float min_dist_sq = highlight_dist_threshold * highlight_dist_threshold;

        const auto& collections = _point_collection->getAllCollections();
        for (const auto& col_pair : collections) {
            for (const auto& point_pair : col_pair.second.points) {
                QPointF point_scene_pos = volumeToScene(point_pair.second.p);
                QPointF diff = scenePos - point_scene_pos;
                float dist_sq = QPointF::dotProduct(diff, diff);
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    _highlighted_point_id = point_pair.second.id;
                }
            }
        }

    }

    invalidateVis();
    renderIntersections();

    emit overlaysUpdated();
}

void CVolumeViewer::setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items)
{
    // Remove and delete existing items in the group
    clearOverlayGroup(key);
    _overlay_groups[key] = items;
}

// Visualize the 'step' parameter used by vc_grow_seg_from_segments by placing
// three small markers in either direction along the same direction arrows.

void CVolumeViewer::clearOverlayGroup(const std::string& key)
{
    auto it = _overlay_groups.find(key);
    if (it == _overlay_groups.end()) return;
    for (auto* item : it->second) {
        if (!item) continue;
        if (auto* scene = item->scene()) {
            scene->removeItem(item);
        } else if (fScene) {
            fScene->removeItem(item);
        }
        delete item;
    }
    _overlay_groups.erase(it);
}

void CVolumeViewer::clearAllOverlayGroups()
{
    if (_overlay_groups.empty()) {
        return;
    }

    for (auto& entry : _overlay_groups) {
        for (auto* item : entry.second) {
            if (!item) {
                continue;
            }
            if (auto* scene = item->scene()) {
                scene->removeItem(item);
            } else if (fScene) {
                fScene->removeItem(item);
            }
            delete item;
        }
    }
    _overlay_groups.clear();
}