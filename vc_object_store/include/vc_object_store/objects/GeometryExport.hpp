#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"

namespace vc::objects {
// Reconstruct native files from an immutable object revision (default: current).
// Explicit pins select metadata and topology at that object revision's publication.
// Returns its geometry revision UUID. No database or source-file writes occur.
// The parent directory must exist. Existing destinations require overwrite;
// directories are replaced as a whole, never merged. Publication is atomic
// (directory publication uses Linux renameat2 or macOS renamex_np). validOnly
// writes a surface's vertices that its `valid` channel marks invalid as missing
// (the sentinel) and no mask.tif.
std::string exportGeometry(ObjectStore& db, const std::string& objectUuid,
    const std::filesystem::path& destination,
    const std::optional<std::string>& revisionUuid = std::nullopt, bool overwrite = false, bool validOnly = false);
// The same from the caller's read transaction (e.g. inside ObjectStoreSession::read),
// so read-only stores export too and the caller can report the same snapshot.
std::string exportGeometry(const ObjectStore& db, const ReadTxn& txn, const std::string& objectUuid,
    const std::filesystem::path& destination,
    const std::optional<std::string>& revisionUuid = std::nullopt, bool overwrite = false, bool validOnly = false);
} // namespace vc::objects
