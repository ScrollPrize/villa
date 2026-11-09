#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include <nlohmann/json.hpp>
namespace vc::objects::detail {
// Annotations and topology from selected historical revisions, or from the
// transaction's current snapshot when no historical selection is provided.
nlohmann::json nativeMetadata(const ObjectStore&, const Txn&, const ObjectRecord&, const GeometryRevision&,
                              const std::optional<RevisionSelection>&);
}
