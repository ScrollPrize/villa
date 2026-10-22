#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include <functional>
#include <map>

namespace vc::objects {
// Publish an owned geometry edit using sparse deltas and stable sequence IDs.
// The caller owns the checked transaction and decides what happens to affected
// relationships. Omitted annotations follow stable IDs (or unchanged row order).
// Resized sequences without IDs require explicit annotations when any exist.
// Array schema, coordinate frame, and patch dimensions must remain unchanged;
// use the dedicated surface operations for channel/grid changes.
std::string updateGeometryArrays(ObjectStore&, WriteTxn&, const std::string& objectUuid,
    const GeometryData& desired, const std::string& expectedGeometryRevision,
    const std::map<std::string, std::string>& annotations = {},
    const std::function<std::vector<GeometryEdgeEdit>(const std::vector<EdgeRecord>&)>& decideRelationships = {});
}
