#pragma once
#include "vc/core/graph/GraphDb.hpp"

namespace vc::graph {
// Reconstruct one object's native file/directory from an immutable revision.
// Defaults to the current revision; explicit revisions must belong to the object.
// Returns the exported revision UUID. No database or source-file writes occur.
// The parent directory must exist. Existing destinations require overwrite;
// directories are replaced as a whole, never merged. Publication is atomic
// (directory publication requires Linux renameat2 support).
std::string exportGeometry(GraphDb& db, const std::string& objectUuid,
    const std::filesystem::path& destination,
    const std::optional<std::string>& revisionUuid = std::nullopt, bool overwrite = false);
} // namespace vc::graph
