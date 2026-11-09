#pragma once
#include "vc_object_store/objects/AfvSource.hpp"
#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/objects/FiberImportExtension.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include <nlohmann/json.hpp>

namespace vc::objects::imports {
// Shared by streaming ingestion and matched prepared-input benchmarks.
inline ObjectRecord afvObjectRecord(const AfvFiber& fiber,const std::string& sourceUuid,
                                    const nlohmann::json& coordinates)
{
    ObjectRecord record;record.uuid=util::nameUuid(sourceUuid,"fiber:"+std::to_string(fiber.id));
    record.kind="fiber";record.name=fiber.name;record.formatVersion=4;record.fiber=fiber.detail;
    record.meta=nlohmann::json({{"coordinates",coordinates}}).dump();
    return record;
}
inline GeometryData afvGeometry(AfvFiber& fiber)
{
    GeometryData g;g.kind="fiber";
    // The source has no controls. Empty descriptors allocate no payload chunks.
    g.arrays={{"control_points","<f8",{0,3},{}},{"control_point_ids","<i8",{0},{}},
              {"line_points","<f8",{fiber.detail.linePointCount,3},std::move(fiber.xyz)}};
    return g;
}
inline InitialObjectRecord afvInitialRecord(AfvFiber& fiber,const std::string& sourceUuid,const nlohmann::json& coordinates) {
    InitialObjectRecord row;row.record=afvObjectRecord(fiber,sourceUuid,coordinates);
    row.extensions.push_back(fiberSourceRow(row.record.uuid,fiber.id,fiber.annotation));
    row.geometry=afvGeometry(fiber);return row;
}
inline InitialCollectionEdge afcInitialRecord(AfcConstraint&& constraint) {
    auto& edge=constraint.edge;
    return {{"",edge.aUuid},{"",edge.bUuid},std::move(edge),{std::move(constraint.measurement)}};
}
}
