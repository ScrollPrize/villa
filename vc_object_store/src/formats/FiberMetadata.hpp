#pragma once
#include "vc_object_store/objects/StoreReader.hpp"
#include <nlohmann/json.hpp>

namespace vc::objects::detail {
// Shared by version/network transport and native file export. Neither consults
// source filenames or moving heads: callers supply the selected records/geometry.
nlohmann::json fiberMetadata(const StoreReader&, const ObjectRecord&, const GeometryRevision&);
nlohmann::json fiberBranchJson(const StoreReader&, const EdgeRecord&,
    const Anchor& own, const Anchor& other, const GeometryRevision& ownGeometry,
    const GeometryRevision& otherGeometry);
}
