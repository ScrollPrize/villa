#pragma once
#include "vc_object_store/objects/Transactions.hpp"
#include <nlohmann/json.hpp>
#include <map>

namespace vc::objects {
struct ManifestReferences {
    std::map<std::string,ObjectVersion> objects;
    std::map<std::string,RelationshipVersion> relationships;
};
// Saved input entries carry uuid, complete reference and optional version.
// Validate all membership before filtering; this does not read old payloads.
ManifestReferences manifestReferences(const nlohmann::json& manifest);
}
