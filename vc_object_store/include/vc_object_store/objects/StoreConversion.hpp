#pragma once
#include <filesystem>
#include <cstdint>
#include "vc_object_store/objects/ObjectStore.hpp"

namespace vc::objects {
// Schema conversion is provided separately by migration/ConvertStore.hpp.
// Explicit offline policy transition; never an open option. History removal
// materializes current geometry before dropping old states. Sealed stores and
// stale historical relationship pins must be exported/revalidated first.
// Materialization issues fresh current geometry tokens and remaps native
// relationship/bounds references; callers must reload geometry handles. Unknown
// extension dependencies on removed history cause rejection, never silent loss.
void changeHistoryMode(const std::filesystem::path&, HistoryMode);
}
