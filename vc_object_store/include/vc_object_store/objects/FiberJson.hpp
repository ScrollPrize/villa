#pragma once
#include "vc_object_store/objects/GeometryIngest.hpp"
#include "vc_object_store/objects/StoreReader.hpp"

namespace vc::objects {
// Native wire/file codec only. Identity, transactions, history and relationships
// remain ObjectStore operations. Source annotations contain no coordinate arrays.
size_t fiberBranchControlIndex(const std::vector<int64_t>& controlIds, const Anchor&);

GeometrySource decodeFiberJson(const std::string& json);
// The fibers outside a prepared fiber batch that its branches link to: the
// existing endpoints a checked ingestGeometryBatch caller must pin.
std::vector<std::string> fiberBatchEndpoints(const std::vector<GeometrySource>&);
// Encode already pinned geometry and annotations; does not read moving heads.
std::string encodeFiberJson(const std::string& objectUuid, const GeometryData&,
                            const std::string& sourceAnnotations);
// Materialize a bounded, immutable native geometry/annotation document from a
// numbered fiber version. Relationship topology is read separately by callers.
std::string fiberVersionJson(const StoreReader&, const ObjectVersion&,
                             uint64_t byteBudget = 64 * 1024 * 1024);
// Export a fiber graph pinned at numbered versions (one per object; a missing,
// deleted or non-fiber version fails). Empty roots exports the selection;
// otherwise exports the connected components containing roots. Applicable
// fiber_branch edges become reciprocal native branch entries, addressed by
// UUID.json. Names remain labels. Include retired endpoints in the selection to
// resolve lifecycle applicability; they are never exported. The byte budget
// applies to each fiber.
std::map<std::string, std::string> fiberNetworkJson(
    const StoreReader&, const std::vector<ObjectVersion>&,
    const std::vector<std::string>& roots = {}, uint64_t byteBudget = 64 * 1024 * 1024);
}
