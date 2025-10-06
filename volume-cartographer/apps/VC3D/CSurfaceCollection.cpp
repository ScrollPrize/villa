#include "CSurfaceCollection.hpp"

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"



CSurfaceCollection::~CSurfaceCollection()
{
    for (auto& pair : _surfs) {
        if (pair.second.owns && pair.second.ptr) {
            delete pair.second.ptr;
        }
    }

    for (auto& pair : _pois) {
        delete pair.second;
    }

    for (auto& pair : _intersections) {
        delete pair.second;
    }
}

void CSurfaceCollection::setSurface(const std::string &name, Surface* surf, bool noSignalSend, bool takeOwnership)
{
    auto it = _surfs.find(name);
    if (it != _surfs.end()) {
        if (it->second.owns && it->second.ptr && it->second.ptr != surf) {
            delete it->second.ptr;
        }
        it->second.ptr = surf;
        it->second.owns = takeOwnership;
    } else {
        _surfs[name] = {surf, takeOwnership};
    }
    if (!noSignalSend) {
        sendSurfaceChanged(name, surf);
    }
}

void CSurfaceCollection::setPOI(const std::string &name, POI *poi)
{
    _pois[name] = poi;
    sendPOIChanged(name, poi);
}

Surface* CSurfaceCollection::surface(const std::string &name)
{
    auto it = _surfs.find(name);
    if (it == _surfs.end())
        return nullptr;
    return it->second.ptr;
}

POI *CSurfaceCollection::poi(const std::string &name)
{
    if (!_pois.count(name))
        return nullptr;
    return _pois[name];
}

std::vector<Surface*> CSurfaceCollection::surfaces()
{
    std::vector<Surface*> surfaces;
    surfaces.reserve(_surfs.size());

    for(auto surface : _surfs) {
        surfaces.push_back(surface.second.ptr);
    } 

    return surfaces;
}

std::vector<POI*> CSurfaceCollection::pois()
{
    std::vector<POI*> pois;
    pois.reserve(_pois.size());

    for(auto poi : _pois) {
        pois.push_back(poi.second);  
    } 

    return pois;
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

void CSurfaceCollection::setIntersection(const std::string &a, const std::string &b, Intersection *intersect)
{
    auto key = std::make_pair(a, b);
    if (_intersections.count(key)) {
        delete _intersections[key];  // Delete old before overwriting
    }
    _intersections[key] = intersect;
    sendIntersectionChanged(a, b, intersect);
}

Intersection *CSurfaceCollection::intersection(const std::string &a, const std::string &b)
{
    if (_intersections.count({a,b}))
        return _intersections[{a,b}];
        
    if (_intersections.count({b,a}))
        return _intersections[{b,a}];
    
    return nullptr;
}

std::vector<std::pair<std::string,std::string>> CSurfaceCollection::intersections(const std::string &a)
{
    std::vector<std::pair<std::string,std::string>> res;

    if (!a.size()) {
        for(auto item : _intersections)
            res.push_back(item.first);
    }
    else
        for(auto item : _intersections) {
            if (item.first.first == a)
                res.push_back(item.first);
            else if (item.first.second == a)
                res.push_back(item.first);
        }
    return res;
}
