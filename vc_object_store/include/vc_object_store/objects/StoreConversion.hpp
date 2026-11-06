#pragma once
#include <filesystem>
#include <cstdint>
#include "vc_object_store/objects/ObjectStore.hpp"

namespace vc::objects {
struct ConversionReport {
    int64_t tables = 0;
    int64_t rows = 0;
    int64_t bytes = 0;
};
// Offline maintenance only. Acquires SQLite exclusive access, builds a sibling,
// checks every pre-existing table/row/value and SQLite integrity, then replaces
// the source atomically. All original history and the logical store UUID survive.
// A conversion error before publication leaves the original data intact.
ConversionReport convertV1ToV2(const std::filesystem::path&);
// Explicit offline policy transition; never an open option. History removal
// materializes current geometry before dropping old states. Sealed stores and
// stale historical relationship pins must be exported/revalidated first.
// Materialization issues fresh current geometry tokens and remaps native
// relationship/bounds references; callers must reload geometry handles. Unknown
// extension dependencies on removed history cause rejection, never silent loss.
void changeHistoryMode(const std::filesystem::path&, HistoryMode);
}
