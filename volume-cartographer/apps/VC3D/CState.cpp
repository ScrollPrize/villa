#include "CState.hpp"
#include "OpenDataCoordinateIdentity.hpp"
#include "VCSettings.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <thread>
#include <optional>
#include <QSettings>

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/ui/VCCollection.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/Slicing.hpp"

namespace {

bool isValidSurfacePoint(const cv::Vec3f& point)
{
    return point[0] != -1.0f && point[1] != -1.0f && point[2] != -1.0f
        && std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

struct SurfaceFocusPoint {
    cv::Vec3f world{0, 0, 0};
    cv::Vec3f ptr{0, 0, 0};
    int row = -1;
    int col = -1;
};

std::optional<SurfaceFocusPoint> focusPointAtGrid(QuadSurface& surface, int row, int col)
{
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || row < 0 || col < 0 || row >= points->rows || col >= points->cols) {
        return std::nullopt;
    }
    if (row <= 0 || col <= 0 || row >= points->rows - 1 || col >= points->cols - 1) {
        return std::nullopt;
    }
    if (!surface.isQuadValid(row, col)) {
        return std::nullopt;
    }
    const cv::Vec3f normal = surface.gridNormal(row, col);
    if (!std::isfinite(normal[0]) || !std::isfinite(normal[1]) || !std::isfinite(normal[2])) {
        return std::nullopt;
    }

    const cv::Vec3f point = (*points)(row, col);
    if (!isValidSurfacePoint(point)) {
        return std::nullopt;
    }

    const cv::Vec3f center = surface.center();
    const cv::Vec2f scale = surface.scale();
    return SurfaceFocusPoint{
        point,
        cv::Vec3f(static_cast<float>(col) - center[0] * scale[0],
                  static_cast<float>(row) - center[1] * scale[1],
                  0.0f),
        row,
        col,
    };
}

std::optional<SurfaceFocusPoint> findSegmentFocusPoint(QuadSurface& surface)
{
    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }

    const cv::Vec2f centerGrid = surface.ptrToGrid({0, 0, 0});
    const int centerRow = std::clamp(static_cast<int>(std::lround(centerGrid[1])), 0, points->rows - 1);
    const int centerCol = std::clamp(static_cast<int>(std::lround(centerGrid[0])), 0, points->cols - 1);

    if (auto focus = focusPointAtGrid(surface, centerRow, centerCol)) {
        return focus;
    }

    const int maxHorizontalRadius = std::max(centerCol, points->cols - 1 - centerCol);
    for (int radius = 1; radius <= maxHorizontalRadius; ++radius) {
        if (auto focus = focusPointAtGrid(surface, centerRow, centerCol - radius)) {
            return focus;
        }
        if (auto focus = focusPointAtGrid(surface, centerRow, centerCol + radius)) {
            return focus;
        }
    }

    const int maxRadius = std::max(std::max(centerRow, centerCol),
                                   std::max(points->rows - 1 - centerRow, points->cols - 1 - centerCol));
    for (int radius = 1; radius <= maxRadius; ++radius) {
        const int rowMin = std::max(0, centerRow - radius);
        const int rowMax = std::min(points->rows - 1, centerRow + radius);
        const int colMin = std::max(0, centerCol - radius);
        const int colMax = std::min(points->cols - 1, centerCol + radius);

        for (int col = colMin; col <= colMax; ++col) {
            if (auto focus = focusPointAtGrid(surface, rowMin, col)) {
                return focus;
            }
            if (rowMax != rowMin) {
                if (auto focus = focusPointAtGrid(surface, rowMax, col)) {
                    return focus;
                }
            }
        }

        for (int row = rowMin + 1; row < rowMax; ++row) {
            if (auto focus = focusPointAtGrid(surface, row, colMin)) {
                return focus;
            }
            if (colMax != colMin) {
                if (auto focus = focusPointAtGrid(surface, row, colMax)) {
                    return focus;
                }
            }
        }
    }

    return std::nullopt;
}

std::string segmentationSurfaceIdForFocus(CState* state, QuadSurface& surface)
{
    if (!state) {
        return surface.id;
    }
    if (auto active = state->activeSurface().lock(); active.get() == &surface) {
        return state->activeSurfaceId();
    }
    if (!surface.id.empty()) {
        return surface.id;
    }
    return "segmentation";
}

bool resetViewOnSurfaceChangeEnabled()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    return settings.value(vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE,
                          vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE_DEFAULT).toBool();
}

std::unique_ptr<POI> createSegmentationFocusPoi(CState* state, QuadSurface& surface)
{
    if (!state) {
        return nullptr;
    }

    const auto focusPoint = findSegmentFocusPoint(surface);
    if (!focusPoint) {
        return nullptr;
    }

    auto vol = state->currentVolume();
    if (!vol) {
        return nullptr;
    }

    auto [w, h, d] = vol->shapeXyz();
    cv::Vec3f clamped = focusPoint->world;
    clamped[0] = std::clamp(clamped[0], 0.0f, static_cast<float>(w - 1));
    clamped[1] = std::clamp(clamped[1], 0.0f, static_cast<float>(h - 1));
    clamped[2] = std::clamp(clamped[2], 0.0f, static_cast<float>(d - 1));

    auto poi = std::make_unique<POI>();
    poi->p = clamped;
    poi->n = cv::Vec3f(0, 0, 0);
    poi->surfaceId = segmentationSurfaceIdForFocus(state, surface);
    poi->surfacePtr = focusPoint->ptr;
    poi->suppressViewerRecenter = false;

    return poi;
}

} // namespace

CState::CState(size_t cacheSizeBytes, QObject* parent)
    : QObject(parent)
    , _cacheSizeBytes(cacheSizeBytes)
{
    _pointCollection = new VCCollection(this);

    setSurface("xy plane",
        std::make_shared<PlaneSurface>(cv::Vec3f{2000,2000,2000}, cv::Vec3f{0,0,1}));
    setSurface("xz plane",
        std::make_shared<PlaneSurface>(cv::Vec3f{2000,2000,2000}, cv::Vec3f{0,1,0}));
    setSurface("yz plane",
        std::make_shared<PlaneSurface>(cv::Vec3f{2000,2000,2000}, cv::Vec3f{1,0,0}));
}

CState::~CState() = default;

std::shared_ptr<VolumePkg> CState::vpkg() const { return _vpkg; }

void CState::setVpkg(std::shared_ptr<VolumePkg> pkg)
{
    _vpkg = std::move(pkg);
    emit vpkgChanged(_vpkg);
}

QString CState::vpkgPath() const
{
    if (_vpkg) {
        return QString::fromStdString(_vpkg->getVolpkgDirectory());
    }
    return {};
}

bool CState::hasVpkg() const { return _vpkg != nullptr; }

std::shared_ptr<Volume> CState::currentVolume() const { return _currentVolume; }

std::string CState::currentVolumeId() const { return _currentVolumeId; }

void CState::setCurrentVolume(std::shared_ptr<Volume> vol)
{
    _currentVolume = std::move(vol);
    applyCacheBudget(_currentVolume);
    resolveCurrentVolumeId();
    _pointCollection->setFileMetadata(
        (_vpkg && !_currentVolumeId.empty())
            ? vc3d::opendata::coordinateIdentityJson(
                  vc3d::opendata::coordinateIdentityForVolume(
                      *_vpkg, _currentVolumeId))
            : utils::Json::object());
    emit volumeChanged(_currentVolume, _currentVolumeId);
}

std::string CState::segmentationGrowthVolumeId() const { return _segmentationGrowthVolumeId; }

void CState::setSegmentationGrowthVolumeId(const std::string& id)
{
    _segmentationGrowthVolumeId = id;
}

std::weak_ptr<QuadSurface> CState::activeSurface() const { return _activeSurface; }

std::string CState::activeSurfaceId() const { return _activeSurfaceId; }

void CState::setActiveSurface(const std::string& id, std::shared_ptr<QuadSurface> surf)
{
    // Drop derived caches on whichever surface was active before — we only
    // keep them populated for the segment the user is currently editing.
    auto prev = _activeSurface.lock();
    if (prev && prev != surf) {
        prev->unloadCaches();
    }
    _activeSurfaceId = id;
    _activeSurface = surf;
}

void CState::clearActiveSurface()
{
    if (auto prev = _activeSurface.lock()) {
        prev->unloadCaches();
    }
    _activeSurface.reset();
    _activeSurfaceId.clear();
}

VCCollection* CState::pointCollection() const { return _pointCollection; }

size_t CState::cacheSizeBytes() const { return _cacheSizeBytes; }

void CState::applyCacheBudget(const std::shared_ptr<Volume>& vol) const
{
    if (vol && _cacheSizeBytes > 0) {
        vol->setCacheBudget(_cacheSizeBytes);
    }
}

void CState::resolveCurrentVolumeId()
{
    _currentVolumeId.clear();
    if (_vpkg && _currentVolume) {
        for (const auto& id : _vpkg->volumeIDs()) {
            if (_vpkg->volume(id) == _currentVolume) {
                _currentVolumeId = id;
                return;
            }
        }
    }
    if (_currentVolume) {
        _currentVolumeId = _currentVolume->id();
    }
}

void CState::closeAll()
{
    emit volumeClosing();

    clearActiveSurface();

    setSurface("segmentation", nullptr, true);
    if (_vpkg) {
        for (const auto& id : _vpkg->getLoadedSurfaceIDs()) {
            setSurface(id, nullptr, true);
        }
        _vpkg->unloadAllSurfaces();
    } else {
        auto names = surfaceNames();
        for (const auto& name : names) {
            if (name != "segmentation") {
                setSurface(name, nullptr, true);
            }
        }
    }

    _currentVolume = nullptr;
    _currentVolumeId.clear();
    _segmentationGrowthVolumeId.clear();

    _pois.clear();
    _pointCollection->clearAll();
    _pointCollection->setFileMetadata(utils::Json::object());

    setVpkg(nullptr);
}

// --- Surface methods (from CSurfaceCollection) ---

void CState::setSurface(const std::string& name, std::shared_ptr<Surface> surf, bool noSignalSend, bool isEditUpdate)
{
    auto it = _surfs.find(name);
    const bool sameSurface = it != _surfs.end() && it->second == surf;
    if (sameSurface && !isEditUpdate && surf != nullptr) {
        return;
    }
    if (_surfaceBatchDepth == 0 && it != _surfs.end() && it->second &&
        it->second != surf) {
        emit surfaceWillBeDeleted(name, it->second);
    }

    POI* delayedFocusPoi = nullptr;
    if (name == "segmentation" && surf != nullptr && !isEditUpdate &&
        resetViewOnSurfaceChangeEnabled()) {
        if (auto quad = std::dynamic_pointer_cast<QuadSurface>(surf)) {
            try {
                auto focusPoi = createSurfaceFocusPoi(*quad);
                if (focusPoi) {
                    delayedFocusPoi = focusPoi.get();
                    _pois["focus"] = std::move(focusPoi);
                }
            } catch (const std::exception&) {
                // Reset recentering is optional; activation/orientation paths
                // handle and report lazy-load failures after the surface is set.
            } catch (...) {
                // Keep surface activation alive even if focus initialization fails.
            }
        }
    }

    _surfs[name] = surf;
    if (!sameSurface) {
        // Edit updates re-set the same pointer every frame; only a mapping
        // change should invalidate cached views of the surface map.
        ++_surfacesVersion;
        if (_surfaceBatchDepth > 0) {
            _surfaceBatchChanged = true;
        }
    }

    if (_surfaceBatchDepth == 0 && (!noSignalSend || surf == nullptr)) {
        emit surfaceChanged(name, surf, isEditUpdate);
    }

    if (delayedFocusPoi) {
        auto poiIt = _pois.find("focus");
        if (poiIt != _pois.end() && poiIt->second.get() == delayedFocusPoi) {
            emit poiChanged("focus", delayedFocusPoi);
            delayedFocusPoi->suppressViewerRecenter = false;
            delayedFocusPoi->suppressTransientPlaneIntersections = false;
        }
    }
}

void CState::setSurfacesBatch(
    const std::vector<std::pair<std::string, std::shared_ptr<Surface>>>& updates)
{
    const bool outermost = _surfaceBatchDepth == 0;
    if (outermost) {
        _surfaceBatchChanged = false;
    }
    ++_surfaceBatchDepth;

    auto finishBatch = [this, outermost]() {
        --_surfaceBatchDepth;
        if (outermost) {
            const bool changed = _surfaceBatchChanged;
            _surfaceBatchChanged = false;
            if (changed) {
                emit surfaceChanged("", nullptr, false);
            }
        }
    };

    try {
        for (const auto& [name, surface] : updates) {
            setSurface(name, surface, true, false);
        }
    } catch (...) {
        finishBatch();
        throw;
    }
    finishBatch();
}

void CState::emitSurfacesChanged()
{
    emit surfaceChanged("", nullptr, false);
}

std::shared_ptr<Surface> CState::surface(const std::string& name)
{
    auto it = _surfs.find(name);
    if (it == _surfs.end())
        return nullptr;
    return it->second;
}

Surface* CState::surfaceRaw(const std::string& name)
{
    auto it = _surfs.find(name);
    if (it == _surfs.end())
        return nullptr;
    return it->second.get();
}

std::string CState::findSurfaceId(Surface* surf)
{
    if (!surf) return {};
    for (const auto& [name, s] : _surfs) {
        if (s.get() == surf) {
            return name;
        }
    }
    return {};
}

std::vector<std::shared_ptr<Surface>> CState::surfaces()
{
    std::vector<std::shared_ptr<Surface>> result;
    result.reserve(_surfs.size());

    for (auto& surface : _surfs) {
        result.push_back(surface.second);
    }

    return result;
}

std::vector<std::string> CState::surfaceNames()
{
    std::vector<std::string> keys;
    for (auto& it : _surfs)
        keys.push_back(it.first);

    return keys;
}

// --- POI methods (from CSurfaceCollection) ---

std::unique_ptr<POI> CState::createSurfaceFocusPoi(QuadSurface& surface)
{
    return createSegmentationFocusPoi(this, surface);
}

void CState::setPOI(const std::string& name, POI* poi)
{
    auto it = _pois.find(name);
    if (it != _pois.end() && it->second.get() == poi) {
        // Same pointer re-submitted (caller mutated in place) - just signal
        emit poiChanged(name, poi);
        poi->suppressViewerRecenter = false;
        poi->suppressTransientPlaneIntersections = false;
        return;
    }
    _pois[name] = std::unique_ptr<POI>(poi);
    emit poiChanged(name, poi);
    poi->suppressViewerRecenter = false;
    poi->suppressTransientPlaneIntersections = false;
}

POI* CState::poi(const std::string& name)
{
    auto it = _pois.find(name);
    if (it == _pois.end())
        return nullptr;
    return it->second.get();
}

std::vector<POI*> CState::pois()
{
    std::vector<POI*> result;
    result.reserve(_pois.size());

    for (auto& [key, ptr] : _pois) {
        result.push_back(ptr.get());
    }

    return result;
}

std::vector<std::string> CState::poiNames()
{
    std::vector<std::string> keys;
    for (auto& it : _pois)
        keys.push_back(it.first);

    return keys;
}
