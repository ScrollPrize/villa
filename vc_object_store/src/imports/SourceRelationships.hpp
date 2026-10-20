#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include <nlohmann/json.hpp>
namespace vc::objects::detail {
struct SourceRecord { ObjectRecord object; nlohmann::json source; };
// Endpoints must already exist. Resolves source labels to UUIDs once, imports
// canonical anchored edges for newly imported objects. Never reconciles old imports.
int64_t importSourceRelationships(ObjectStore&, WriteTxn&, const std::vector<SourceRecord>&);
}
