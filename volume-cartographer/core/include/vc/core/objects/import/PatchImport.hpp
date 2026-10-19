#pragma once
#include "vc/core/objects/ObjectStore.hpp"
#include "vc/core/objects/import/ImportReport.hpp"
#include "vc/core/util/Progress.hpp"

namespace vc::objects {
// Registers each immediate subdirectory of segmentsDir that contains a
// meta.json with a "uuid" as a kind="patch" object. Reads nothing else
// from segment directories (in particular, never overlapping.json)
// and creates no edges.
ImportReport importPatches(ObjectStore& db, WriteTxn& txn, const std::filesystem::path& segmentsDir,
                         const util::ProgressCallback& progress = {});

} // namespace vc::objects
