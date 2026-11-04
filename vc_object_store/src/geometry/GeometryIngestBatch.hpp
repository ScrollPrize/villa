#pragma once
#include "vc_object_store/objects/GeometryIngest.hpp"
namespace vc::objects::detail {
int64_t ingestGeometryBatch(ObjectStore&, WriteTxn&, std::vector<GeometrySource>,
    const std::string& coordinateContext, bool registeredInitial);
}
