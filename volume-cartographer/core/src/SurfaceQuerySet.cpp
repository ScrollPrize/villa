#include "vc/core/util/SurfaceQuerySet.hpp"
#include "vc/core/util/QuadSurface.hpp"
void SurfaceQuerySet::rebuild(const std::vector<SurfacePtr>& surfaces) {
    std::vector<vc::object_links::SurfaceIndex::SurfacePtr> geometry;
    identities_.clear(); geometry_.clear();
    for (const auto& surface:surfaces) {
        if (!surface || geometry_.contains(surface.get())) continue;
        const auto* points=surface->rawPointsPtr();
        if (!points || points->empty()) continue;
        std::vector<uint8_t> validity;
        (void)surface->gridVertices(validity,points);
        auto resident=std::make_shared<vc::object_links::SurfaceGrid>(*points,surface->scale(),validity);
        identities_.emplace(resident.get(),Identity{surface,surface->gridToPtr({0,0})});
        geometry_.emplace(surface.get(),resident);
        geometry.push_back(std::move(resident));
    }
    index_.rebuild(geometry);
}
vc::object_links::SurfaceIndex::PointQuery SurfaceQuerySet::queryFor(const PointQuery& query) const {
    vc::object_links::SurfaceIndex::PointQuery native{query.worldPoint,query.tolerance,{}};
    if (query.surfaces.only) native.surfaces.only=geometry_.at(query.surfaces.only.get()).get();
    return native;
}
std::vector<SurfaceQuerySet::LookupResult> SurfaceQuerySet::locateAll(const PointQuery& query) const {
    if (query.surfaces.only && !geometry_.contains(query.surfaces.only.get())) return {};
    std::vector<LookupResult> result;
    for (const auto& hit:index_.locateAll(queryFor(query))) {
        const auto& identity=identities_.at(hit.surface.get());
        result.push_back({identity.surface,identity.ptrOrigin+cv::Vec3f(hit.grid[0],hit.grid[1],0),hit.distance});
    }
    return result;
}
std::optional<SurfaceQuerySet::LookupResult> SurfaceQuerySet::locate(const PointQuery& query) const {
    if (query.surfaces.only && !geometry_.contains(query.surfaces.only.get())) return {};
    const auto hit=index_.locate(queryFor(query));
    if (!hit) return {};
    const auto& identity=identities_.at(hit->surface.get());
    return LookupResult{identity.surface,identity.ptrOrigin+cv::Vec3f(hit->grid[0],hit->grid[1],0),hit->distance};
}
