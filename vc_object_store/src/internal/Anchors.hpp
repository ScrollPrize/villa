#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include <nlohmann/json.hpp>

namespace vc::objects::detail {
void validateAnchor(const Anchor&);
void validateEdgeAnchors(const EdgeRecord&);
nlohmann::json dependenciesToJson(const Anchor&);
void dependenciesFromJson(Anchor&, const nlohmann::json&);
nlohmann::json anchorToJson(const Anchor&);
Anchor anchorFromJson(const nlohmann::json&);
void swapEndpoints(EdgeRecord&);
// Logical order is stable across databases; SQL storage may use row-ID order.
void canonicalizeEdge(EdgeRecord&);
std::string edgeIdentity(const EdgeRecord&);
}
