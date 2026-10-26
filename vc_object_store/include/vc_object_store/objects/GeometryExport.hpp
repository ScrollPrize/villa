#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"

namespace vc::objects {
// Reconstruct native files from an immutable object revision (default: current).
// Explicit pins select metadata and topology at that object revision's publication.
// Returns its geometry revision UUID. No database or source-file writes occur.
// The parent directory must exist. Existing destinations require overwrite;
// directories are replaced as a whole, never merged. Publication is atomic
// (directory publication requires Linux renameat2 support).
std::string exportGeometry(ObjectStore& db, const std::string& objectUuid,
    const std::filesystem::path& destination,
    const std::optional<std::string>& revisionUuid = std::nullopt, bool overwrite = false);
} // namespace vc::objects
