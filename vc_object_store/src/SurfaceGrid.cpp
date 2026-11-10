#include "vc_object_store/objects/SurfaceGrid.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
#include <stdexcept>

namespace vc::objects {
namespace {
SurfaceGrid surfaceGrid(const StoreReader& reader, const ObjectRevision& selected, const std::string& volume, double targetScale);
}
SurfaceGrid readSurfaceGrid(const StoreReader& reader, const std::string& surfaceUuid, const std::string& volume,
                            double targetScale, std::optional<int64_t> version, const std::optional<std::string>& generation)
{
    std::optional<ObjectRevision> revision;
    if (version) {
        const auto current=reader.currentObject(surfaceUuid);
        if(current)revision=reader.objectsAtVersions({current->reference(version,generation)}).front();
        if (!revision || revision->deleted || revision->record.kind != "surface")
            throw std::invalid_argument("no surface " + surfaceUuid + " version " + std::to_string(*version));
    } else {
        revision = reader.currentObject(surfaceUuid);
        if (!revision || revision->record.kind != "surface") throw std::invalid_argument("no surface " + surfaceUuid);
    }
    return surfaceGrid(reader,*revision,volume,targetScale);
}

SurfaceGrid readSurfaceGrid(const StoreReader& reader, const ObjectVersion& reference, const std::string& volume, double targetScale)
{
    const auto revision=reader.objectsAtVersions({reference}).front();
    if(!revision||revision->deleted||revision->record.kind!="surface")
        throw std::invalid_argument("no surface at the selected exact version");
    return surfaceGrid(reader,*revision,volume,targetScale);
}

namespace {
SurfaceGrid surfaceGrid(const StoreReader& reader, const ObjectRevision& selected, const std::string& volume, double targetScale)
{
    const auto* revision=&selected;
    const auto& surfaceUuid=revision->objectUuid;
    if (!revision->geometryRevision)
        throw std::invalid_argument("surface " + surfaceUuid + " version " + std::to_string(revision->version) +
                                    " has no geometry");
    // The geometry this version names, so record and grid come from one read.
    GeometrySnapshot snapshot(reader);
    const auto canonical = snapshot.surface(surfaceUuid, *revision->geometryRevision);
    std::optional<TransformedGeometryView> view;
    try {
        view = canonical.viewTransformedAt(*revision, volume, targetScale);
    } catch (const std::runtime_error& error) {
        throw std::invalid_argument("surface " + surfaceUuid + " does not map to volume " + volume + ": " + error.what());
    }
    const auto tile = view->readTile(0, 0, canonical.rows(), canonical.columns());
    SurfaceGrid grid;
    grid.rows = tile.rows;
    grid.columns = tile.columns;
    grid.points.reserve(tile.points.size() * 3);
    for (size_t i = 0; i < tile.points.size(); ++i) {
        const bool valid = tile.validity[i] != 0;
        for (int axis = 0; axis < 3; ++axis) grid.points.push_back(valid ? float(tile.points[i][axis]) : -1.0f);
    }
    grid.valid = tile.validity;
    grid.scale = tile.scale;
    grid.record = revision->record;
    grid.version = revision->version;
    grid.generation = revision->generation;
    grid.geometryRevision = *revision->geometryRevision;
    for (const auto& array : canonical.revision().arrays)
        if (array.name != "points" && array.name != "valid") grid.channels.push_back(array.name);
    return grid;
}
}
}
