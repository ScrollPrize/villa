#pragma once

#include <vc_object_store/objects/ObjectStore.hpp>
#include <vc_object_store/objects/VolumeTransforms.hpp>
#include <algorithm>
#include <map>
#include <stdexcept>

struct FiberConstraintGeometry {
    std::string uuid;
    std::string revision;
    // Stored coordinates to the target volume's voxels.
    vc::objects::VolumeAffine mapping;
};

// The ingest summary is not updated by geometry edits. Select and pin sampled
// lines from current geometry, retaining UUID order for database source_index.
inline std::vector<FiberConstraintGeometry> selectFiberConstraintGeometry(
    const vc::objects::ObjectStore& store, const vc::objects::Txn& txn,
    const std::map<std::string, vc::objects::VolumeAffine>& members)
{
    std::vector<FiberConstraintGeometry> result;
    for (const auto& [uuid, mapping] : members) {
        const auto head = store.currentGeometry(txn, uuid);
        if (!head) throw std::runtime_error("Fiber " + uuid + " has no geometry");
        const auto line = std::find_if(head->arrays.begin(), head->arrays.end(),
            [](const auto& a) { return a.name == "line_points"; });
        if (line == head->arrays.end() || line->shape.empty() || line->shape[0] < 2) continue;
        result.push_back({uuid, head->uuid, mapping});
    }
    return result;
}
