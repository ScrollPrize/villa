#pragma once
#include "internal/Database.hpp"
namespace vc::objects::detail {
void createEndpointSchema(sqlite3*);
void createEdgeIdentityIndex(sqlite3*);
void createGeometryChunkTable(sqlite3*, const std::string& name = "geometry_chunks");
void createGeometryChunkGuards(sqlite3*);
void createSchema(sqlite3*, bool history = true, const std::optional<std::string>& storeUuid = std::nullopt, bool immutableInitial = false);
}
