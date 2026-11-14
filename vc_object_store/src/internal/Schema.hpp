#pragma once
#include "internal/Database.hpp"
namespace vc::objects::detail {
void createGeometryChunkTable(sqlite3*, const std::string& name = "geometry_chunks");
void createSchema(sqlite3*, bool history = true, const std::optional<std::string>& storeUuid = std::nullopt, bool immutableInitial = false,
                  const OpenOptions& options = {});
}
