#pragma once
#include "vc_object_store/objects/Extensions.hpp"
#include <sqlite3.h>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace vc::objects { class ObjectStore; class WriteTxn; }
namespace vc::objects::detail {
inline constexpr const char* extensionManifestKey = "vcobj_extensions";
inline bool mayContainExtensionManifest(const std::string& meta) {
    return meta.find(extensionManifestKey) != std::string::npos || meta.find('\\') != std::string::npos;
}
// Initial construction accepts typed extension rows, never detached snapshot
// pointers. No database lookup or JSON parse for the ordinary bulk-import case.
inline void rejectInitialExtensionMetadata(const std::string& meta) {
    if (mayContainExtensionManifest(meta) &&
        nlohmann::json::parse(meta).contains(extensionManifestKey))
        throw std::invalid_argument("extension: initial imports cannot supply reserved snapshot metadata");
}
// The reserved descriptor is changed only together with its typed current rows.
void checkExtensionMetadata(const std::string& previousMeta, const std::string& replacementMeta);
void restoreCurrentExtensionRows(sqlite3*, const char* ownerTable, const std::string& ownerUuid,
                                 const std::string& sourceMeta);
void clearCurrentExtensionRows(sqlite3*, const char* ownerTable, const std::string& ownerUuid);
// Current snapshots, not discarded historical versions, survive history removal.
std::vector<std::string> currentExtensionChunks(sqlite3*);
// Localize manual fields with their owner into an overlay. Immutable result
// tables remain in the referenced store and need no copied rows.
void materializeExtensionOwner(sqlite3* sourceDb, ObjectStore& target, WriteTxn&,
                               const char* ownerTable, const std::string& ownerUuid,
                               const std::string& sourceMeta);
} // namespace vc::objects::detail
