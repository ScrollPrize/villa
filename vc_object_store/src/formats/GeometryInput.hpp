#pragma once
#include "vc_object_store/objects/GeometryIngest.hpp"
#include "vc_object_store/PackedTrackStore.hpp"
#include <nlohmann/json.hpp>
namespace vc::objects::formats {
// Decoded geometry and source annotations/topology. No database access here;
// ingestion resolves identities and publishes the batch in checked transactions.
using Prepared = GeometrySource;
nlohmann::json readJson(const std::filesystem::path&);
ObjectRecord sourceObject(const nlohmann::json&, const std::string&, const std::filesystem::path&);
// Sources written before objects carried UUIDs keep their names in "uuid" (tifxyz) or
// have none (PCL collections, version-3 fibers). A canonical "uuid" is always kept.
// Otherwise, with a namespace, set "uuid" to uuid5(namespace, "<kind>:<key>") and record
// {namespace, name} as "uuid_derivation"; without one, leave the document to fail as before.
void deriveSourceUuid(nlohmann::json& root, const std::string& kind, const std::string& key,
    const std::optional<std::string>& uuidNamespace);
Prepared fiber(const std::filesystem::path&);
std::vector<Prepared> fibers(const std::filesystem::path&, const std::vector<std::string>& available,
    const std::optional<std::string>& uuidNamespace = std::nullopt);
Prepared patch(const std::filesystem::path&, const std::optional<std::string>& uuidNamespace = std::nullopt);
std::vector<Prepared> collections(const std::filesystem::path&,
    const std::optional<std::string>& uuidNamespace = std::nullopt);
Prepared track(const PackedTrackStore&, size_t row, ObjectRecord);
} // namespace vc::objects::formats
