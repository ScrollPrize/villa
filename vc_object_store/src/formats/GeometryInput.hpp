#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/PackedTrackStore.hpp"
#include <nlohmann/json.hpp>
namespace vc::objects::formats {
// Decoded geometry and source annotations/topology. No database access here;
// ingestion resolves identities and publishes the batch in checked transactions.
struct Prepared {
    ObjectRecord object;
    GeometryData geometry;
    nlohmann::json source;
};
nlohmann::json readJson(const std::filesystem::path&);
ObjectRecord sourceObject(const nlohmann::json&, const std::string&, const std::filesystem::path&);
Prepared fiber(const std::filesystem::path&);
Prepared patch(const std::filesystem::path&);
std::vector<Prepared> collections(const std::filesystem::path&);
Prepared track(const PackedTrackStore&, size_t row, ObjectRecord);
} // namespace vc::objects::formats
