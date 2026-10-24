#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"

namespace vc::objects {
// Reconstruct native files from immutable geometry and current DB annotations/topology.
// Explicit revisions must belong to the object; native edges must bind that revision.
// Returns the exported revision UUID. No database or source-file writes occur.
// The parent directory must exist. Existing destinations require overwrite;
// directories are replaced as a whole, never merged. Publication is atomic
// (directory publication requires Linux renameat2 support).
std::string exportGeometry(ObjectStore& db, const std::string& objectUuid,
    const std::filesystem::path& destination,
    const std::optional<std::string>& revisionUuid = std::nullopt, bool overwrite = false);
} // namespace vc::objects
