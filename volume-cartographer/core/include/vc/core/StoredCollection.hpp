#pragma once
// Point collections (PointCollections' collection JSON) as the object store
// keeps them (vc::objects::AttachedCollection): the collection's points with
// their ids, one annotation per point (the point's other fields), and the rest
// of the collection as the object's meta.source. Corrections attached to a
// surface and a volume's own collections use the same form.
#include <nlohmann/json.hpp>
#include <vc_object_store/objects/AttachedDocument.hpp>

namespace vc {

// From to_json(PointCollections::Collection) form.
vc::objects::AttachedCollection storedCollection(const nlohmann::json& collection);
// Back to that form (uuid, name, points by id and the rest).
nlohmann::json collectionJson(const vc::objects::AttachedCollection& stored);

} // namespace vc
