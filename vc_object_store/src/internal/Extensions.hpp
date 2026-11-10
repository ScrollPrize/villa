#pragma once
#include "vc_object_store/objects/Extensions.hpp"
#include <sqlite3.h>

namespace vc::objects::detail {
bool hasExtensionTables(sqlite3*);
ExtensionTable extensionDefinition(const ExtensionTableInfo&);
void requireCompatibleExtension(const ExtensionTableInfo&,const ExtensionTableInfo&);
std::vector<ExtensionTableInfo> extensionTables(sqlite3*);
std::optional<ExtensionTableInfo> extensionTable(sqlite3*, const std::string&, bool includeMetadata = true);
void createExtensionTable(sqlite3*, const ExtensionTable&);
void appendExtensionRows(sqlite3*, const std::string&, std::span<const ExtensionRow>);
void replaceCurrentExtensionRows(sqlite3*, const std::string&, const std::string& ownerUuid, std::span<const ExtensionRow>);
void appendInitialExtensions(sqlite3*, std::span<const InitialExtensionRow* const>);
void scanExtensionRows(sqlite3*, const std::string&, const std::function<bool(const ExtensionRow&)>&,
                       std::span<const ExtensionValue> keyPrefix);
void rejectExtensionLoss(sqlite3*, const char* operation);
} // namespace vc::objects::detail
