#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include <nlohmann/json.hpp>
namespace vc::objects::detail {
// Annotations and topology from the selected object revisions.
nlohmann::json nativeMetadata(ObjectStore&, const Txn&, const ObjectRecord&, const GeometryRevision&, const RevisionSelection&);
}
