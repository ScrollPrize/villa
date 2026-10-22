#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"

namespace vc::objects {
// Native wire/file codec only. Identity, transactions, history and relationships
// remain ObjectStore operations. Source annotations contain no coordinate arrays.
size_t fiberBranchControlIndex(const std::vector<int64_t>& controlIds, size_t controlCount, const Anchor&);

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
// Export an explicitly pinned fiber graph. Empty roots exports the selection;
// otherwise exports the connected components containing roots. Applicable
// fiber_branch edges become reciprocal native branch entries, addressed by
// UUID.json. Names remain labels. Include retired endpoints in the selection to
// resolve lifecycle applicability; they are never exported. The byte budget
// applies to each fiber.
std::map<std::string, std::string> fiberNetworkJson(
    const ObjectStore&, const Txn&, const RevisionSelection&,
    const std::vector<std::string>& roots = {}, uint64_t byteBudget = 64 * 1024 * 1024);
}
