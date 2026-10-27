#pragma once
// Creating a surface object: one record shape for every writer (VC3D, the
// store-mode tools, Python).
#include "vc_object_store/objects/ObjectStore.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace vc::objects {
// How a surface was made: the tool, its parameters and the surfaces it came from.
struct SurfaceProvenance {
    std::string tool;
    nlohmann::json parameters = nlohmann::json::object();
    std::vector<std::string> sources;
};
// The keys of a tifxyz meta.json that describe the file rather than the
// surface (identity and grid layout); an object's record and geometry hold them.
const std::vector<std::string>& tifxyzFileKeys();
// A new surface `uuid` named `name` holding `geometry`, bound by `coordinates`
// ({volume_uuid, scale_factor}), with `tags`, meta.provenance, and the rest of
// `meta` without the tifxyz file keys, tags, provenance or coordinates (status
// and binding are the new record's own, never a source's).
void createSurface(ObjectStore&, WriteTxn&, const std::string& uuid, const std::string& name,
                   const std::vector<std::string>& tags, const nlohmann::json& coordinates,
                   const SurfaceProvenance&, const nlohmann::json& meta, const GeometryData&);
}
