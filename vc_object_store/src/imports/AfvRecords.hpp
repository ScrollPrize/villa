#pragma once
#include "vc_object_store/objects/AfvSource.hpp"
#include "vc_object_store/Uuid.hpp"
#include <nlohmann/json.hpp>

namespace vc::objects::imports {
// Shared by streaming ingestion and matched prepared-input benchmarks.
inline ObjectRecord afvObjectRecord(const AfvFiber& fiber,const std::string& sourceUuid,
                                    const nlohmann::json& coordinates)
{
    ObjectRecord record;record.uuid=util::nameUuid(sourceUuid,"fiber:"+std::to_string(fiber.id));
    record.kind="fiber";record.name=fiber.name;record.formatVersion=4;record.fiber=fiber.detail;
    record.meta=nlohmann::json({{"coordinates",coordinates},{"source",{{"afv",{{"collection_uuid",sourceUuid},
        {"fiber_id",fiber.id},{"annotation",nlohmann::json::parse(fiber.annotation)}}}}}}).dump();
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
}
