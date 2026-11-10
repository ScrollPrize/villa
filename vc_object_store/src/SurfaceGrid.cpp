#include "vc_object_store/objects/SurfaceGrid.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
#include <stdexcept>

namespace vc::objects {
namespace {
SurfaceGrid surfaceGrid(const StoreReader& reader, const ObjectRevision& selected, const std::string& volume, double targetScale)
{
    if (!selected.geometryRevision) throw std::invalid_argument("surface has no geometry");
    // The geometry this version names, so record and grid come from one read.
    const auto canonical = GeometrySnapshot(reader).surface(selected.objectUuid, selected.geometryRevision);
    try {
        const auto view = canonical.viewTransformedAt(selected, volume, targetScale);
        return readSurfaceGrid(reader,selected,&view);
    } catch (const std::runtime_error& error) {
        throw std::invalid_argument("surface " + selected.objectUuid + " does not map to volume " + volume + ": " + error.what());
    }
}
}
SurfaceGrid readSurfaceGrid(const StoreReader& reader, const std::string& surfaceUuid, const std::string& volume,
                            double targetScale, std::optional<int64_t> version, const std::optional<std::string>& generation)
{
    const auto revision=reader.selectedObject(surfaceUuid,version,generation);
    if (!revision) throw std::invalid_argument("no surface " + surfaceUuid + (version ? " version " + std::to_string(*version) : ""));
    return surfaceGrid(reader,*revision,volume,targetScale);
}

SurfaceGrid readSurfaceGrid(const StoreReader& reader, const ObjectVersion& reference, const std::string& volume, double targetScale)
{
    const auto revision=reader.selectedObject(reference.objectUuid,std::nullopt,std::nullopt,reference);
    if(!revision)throw std::invalid_argument("no surface at the selected exact version");
    return surfaceGrid(reader,*revision,volume,targetScale);
}

SurfaceGrid readSurfaceGrid(const StoreReader& reader,const ObjectRevision& selected,
                            const TransformedGeometryView* transformed,bool applyValidity)
{
    if(selected.deleted||selected.record.kind!="surface"||!selected.geometryRevision)
        throw std::invalid_argument("selected object is not a stored surface");
    const auto canonical=GeometrySnapshot(reader).surface(selected.objectUuid,selected.geometryRevision);
    if(transformed&&transformed->revision().uuid!=canonical.revision().uuid)
        throw std::invalid_argument("transformed view revision differs from surface");
    SurfaceGrid grid;
    const auto copy=[&](const auto& tile){
        grid.rows=tile.rows;grid.columns=tile.columns;grid.valid=tile.validity;grid.scale=tile.scale;
        grid.points.reserve(tile.points.size()*3);
        for(size_t i=0;i<tile.points.size();++i){
            const bool valid=!applyValidity||grid.valid.empty()||grid.valid[i]!=0;
            for(auto value:tile.points[i])grid.points.push_back(valid?float(value):-1.0f);
        }
    };
    if(transformed)copy(transformed->readTile(0,0,canonical.rows(),canonical.columns()));
    else copy(canonical.readTile(0,0,canonical.rows(),canonical.columns()));
    grid.record = selected.record;
    grid.version = selected.version;
    grid.generation = selected.generation;
    grid.geometryRevision = *selected.geometryRevision;
    for (const auto& array : canonical.revision().arrays)
        if (array.name != "points" && array.name != "valid") grid.channels.push_back(array.name);
    return grid;
}
}
