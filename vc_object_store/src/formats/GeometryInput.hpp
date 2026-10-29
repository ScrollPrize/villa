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
Prepared fiber(const std::filesystem::path&);
std::vector<Prepared> fibers(const std::filesystem::path&, const std::vector<std::string>& available);
Prepared surface(const std::filesystem::path&);
// The same without the source's identity (meta.json uuid/name): its geometry,
// grid detail and source meta only.
Prepared surfaceGeometry(const std::filesystem::path&);
// One surface channel TIFF with its exact samples (little-endian), unchecked against a grid.
GeometryArray surfaceChannel(const std::string& name, const std::filesystem::path&);
std::vector<Prepared> collections(const std::filesystem::path&);
Prepared umbilicus(const std::filesystem::path&);
Prepared track(const PackedTrackStore&, size_t row, ObjectRecord);
} // namespace vc::objects::formats
