#pragma once
#include "vc/core/objects/ObjectStore.hpp"
#include "vc/core/objects/import/ImportReport.hpp"
#include "vc/core/util/Progress.hpp"

namespace vc::objects {
// Registers tracks from a JSON interchange file: an array of objects
// with dataset_ref + external_key (the track identity) and optional
// name, family, point_count, length, bbox (6 numbers), tags, meta.
// This is the registration path for track datasets produced outside
// this repo (e.g. Spiral): the external producer emits this list, the
// object store mints and remembers a UUID per (dataset_ref, external_key).
// Idempotent: an unchanged identity keeps its UUID and generation; a
// changed entry updates in place. Geometry stays in the external
// dataset; only metadata lands here.
ImportReport importTrackList(ObjectStore& db, WriteTxn& txn, const std::filesystem::path& trackListJson,
                         const util::ProgressCallback& progress = {});

} // namespace vc::objects
