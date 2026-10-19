#pragma once
#include "vc/core/objects/ObjectStore.hpp"
#include "vc/core/objects/import/ImportReport.hpp"
#include "vc/core/util/Progress.hpp"

namespace vc::objects {
// Scans fiberDir for vc3d_fiber JSON files (one object per file, kind
// "fiber") and imports the edges their branch entries assert. Branch
// targets are resolved by file name within the scanned set; unresolved
// and one-sided entries are reported, never silently dropped. Runs in
// the caller's WriteTxn. Idempotent: re-importing an unchanged file
// (same uuid + same source content) is a no-op for that fiber.
ImportReport importFibers(ObjectStore& db, WriteTxn& txn, const std::filesystem::path& fiberDir,
                         const util::ProgressCallback& progress = {});

} // namespace vc::objects
