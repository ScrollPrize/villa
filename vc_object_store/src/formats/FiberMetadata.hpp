#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include <nlohmann/json.hpp>

namespace vc::objects::detail {
// Shared by revision/network transport and native file export. Neither consults
// source filenames or moving heads: callers supply the selected records/geometry.
nlohmann::json fiberMetadata(const ObjectStore&, const Txn&, const ObjectRecord&, const GeometryRevision&);
nlohmann::json fiberBranchJson(const ObjectStore&, const Txn&, const EdgeRecord&,
    const Anchor& own, const Anchor& other, const GeometryRevision& ownGeometry,
    const GeometryRevision& otherGeometry);
}
