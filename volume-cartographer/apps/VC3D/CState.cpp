#include "CState.hpp"
#include "VCSettings.hpp"

#include <algorithm>
#include <thread>
#include <QSettings>

#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/Slicing.hpp"

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
    fprintf(stderr, "[CState] setCurrentVolume: begin (old=%p new=%p)\n",
            (void*)_currentVolume.get(), (void*)vol.get());

    // Shut down old pipeline workers and destroy the pipeline eagerly so
    // we don't have two 10 GB mmaps alive simultaneously. shutdown() makes
    // the destructor's abortAll() a no-op, and resetTieredCache() frees
    // the BlockCache mmap before the new pipeline allocates its own.
    if (_currentVolume) {
        auto* oldPipeline = _currentVolume->tieredCache();
        if (oldPipeline) {
            fprintf(stderr, "[CState] shutting down old pipeline %p\n", (void*)oldPipeline);
            oldPipeline->shutdown();
        }
        fprintf(stderr, "[CState] resetting old pipeline\n");
        _currentVolume->resetTieredCache();
    }

    fprintf(stderr, "[CState] replacing _currentVolume\n");
    _currentVolume = std::move(vol);
    applyCacheBudget(_currentVolume);
    resolveCurrentVolumeId();

    fprintf(stderr, "[CState] emitting volumeChanged\n");
    emit volumeChanged(_currentVolume, _currentVolumeId);
    fprintf(stderr, "[CState] setCurrentVolume: done\n");
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

    _vpkg = nullptr;
    _currentVolume = nullptr;
    _currentVolumeId.clear();
    _segmentationGrowthVolumeId.clear();

    _pois.clear();
    _pointCollection->clearAll();
}

// --- Surface methods (from CSurfaceCollection) ---

void CState::setSurface(const std::string& name, std::shared_ptr<Surface> surf, bool noSignalSend, bool isEditUpdate)
{
    auto it = _surfs.find(name);
    if (it != _surfs.end() && it->second && it->second != surf) {
        emit surfaceWillBeDeleted(name, it->second);
    }

    _surfs[name] = surf;

    if (!noSignalSend || surf == nullptr) {
        emit surfaceChanged(name, surf, isEditUpdate);
    }
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

void CState::setPOI(const std::string& name, POI* poi)
{
    auto it = _pois.find(name);
    if (it != _pois.end() && it->second.get() == poi) {
        // Same pointer re-submitted (caller mutated in place) - just signal
        emit poiChanged(name, poi);
        return;
    }
    _pois[name] = std::unique_ptr<POI>(poi);
    emit poiChanged(name, poi);
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
