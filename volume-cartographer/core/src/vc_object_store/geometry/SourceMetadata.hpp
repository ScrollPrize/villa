#pragma once
#include "vc/core/objects/ObjectStore.hpp"
#include <nlohmann/json.hpp>
namespace vc::objects::detail {
// Current annotations and topology, combined with immutable format attributes.
nlohmann::json nativeMetadata(ObjectStore&, const Txn&, const ObjectRecord&, const GeometryRevision&);
}
