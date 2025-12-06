#include "CSurfaceCollection.hpp"

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"


CSurfaceCollection::~CSurfaceCollection()
{
    // Surfaces are automatically cleaned up by shared_ptr

    for (auto& pair : _pois) {
        delete pair.second;
    }
}

void CSurfaceCollection::setSurface(const std::string &name, std::shared_ptr<Surface> surf, bool noSignalSend, bool isEditUpdate)
{
    auto it = _surfs.find(name);
    if (it != _surfs.end() && it->second && it->second != surf) {
        // Notify listeners BEFORE replacement so they can clear their references
        emit sendSurfaceWillBeDeleted(name, it->second);
    }

    _surfs[name] = surf;

    // Always emit signal when surface is deleted (nullptr) to prevent dangling pointers
    // Only suppress signal for non-deletion updates
    if (!noSignalSend || surf == nullptr) {
        emit sendSurfaceChanged(name, surf, isEditUpdate);
    }
}

void CSurfaceCollection::emitSurfacesChanged()
{
    // Emit a signal to notify listeners that surfaces have been modified in batch.
    // Use empty name and nullptr to indicate batch update.
    emit sendSurfaceChanged("", nullptr, false);
}

void CSurfaceCollection::setPOI(const std::string &name, POI *poi)
{
    _pois[name] = poi;
    emit sendPOIChanged(name, poi);
}

std::shared_ptr<Surface> CSurfaceCollection::surface(const std::string &name)
{
    auto it = _surfs.find(name);
    if (it == _surfs.end())
        return nullptr;
    return it->second;
}

Surface* CSurfaceCollection::surfaceRaw(const std::string &name)
{
    auto it = _surfs.find(name);
    if (it == _surfs.end())
        return nullptr;
    return it->second.get();
}

std::string CSurfaceCollection::findSurfaceId(Surface* surf)
{
    if (!surf) return {};
    for (const auto& [name, s] : _surfs) {
        if (s.get() == surf) {
            return name;
        }
    }
    return {};
}

POI *CSurfaceCollection::poi(const std::string &name)
{
    if (!_pois.count(name))
        return nullptr;
    return _pois[name];
}

std::vector<std::shared_ptr<Surface>> CSurfaceCollection::surfaces()
{
    std::vector<std::shared_ptr<Surface>> result;
    result.reserve(_surfs.size());

    for(auto& surface : _surfs) {
        result.push_back(surface.second);
    }

    return result;
}

std::vector<POI*> CSurfaceCollection::pois()
{
    std::vector<POI*> result;
    result.reserve(_pois.size());

    for(auto& poi : _pois) {
        result.push_back(poi.second);
    }

    return result;
}

std::vector<std::string> CSurfaceCollection::surfaceNames()
{
    std::vector<std::string> keys;
    for(auto &it : _surfs)
        keys.push_back(it.first);

    return keys;
}

std::vector<std::string> CSurfaceCollection::poiNames()
{
    std::vector<std::string> keys;
    for(auto &it : _pois)
        keys.push_back(it.first);

    return keys;
}
