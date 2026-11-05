#pragma once
#include "vc_object_store/objects/GeometryIngest.hpp"
namespace vc::objects::detail {
// The canonical form of a source's UUID; an invalid one rejects the source.
std::string sourceUuid(const std::string&);
int64_t ingestGeometryBatch(ObjectStore&, WriteTxn&, std::vector<GeometrySource>,
    const std::string& coordinateContext, bool registeredInitial);
}
