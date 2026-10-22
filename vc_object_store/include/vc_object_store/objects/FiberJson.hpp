#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"

namespace vc::objects {
// Native wire/file codec only. Identity, transactions, history and relationships
// remain ObjectStore operations. Source annotations contain no coordinate arrays.
struct FiberDocument {
    ObjectRecord object;
    GeometryData geometry;
    std::string source;
};
FiberDocument decodeFiberJson(const std::string& json);
// Encode already pinned geometry and annotations; does not read moving heads.
std::string encodeFiberJson(const std::string& objectUuid, const GeometryData&,
                            const std::string& sourceAnnotations);
// Materialize a bounded, immutable native geometry/annotation document from a
// selected object revision. Relationship topology is read separately by callers.
std::string fiberRevisionJson(const ObjectStore&, const Txn&, const std::string& objectRevision,
                             uint64_t byteBudget = 64 * 1024 * 1024);
}
