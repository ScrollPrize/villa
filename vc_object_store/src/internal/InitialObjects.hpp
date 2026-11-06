#pragma once
#include "vc_object_store/objects/ObjectCollection.hpp"

namespace vc::objects::detail {
// Insertion only, history off. Caller owns a savepoint spanning all records,
// allocator changes and prepared geometry metadata. Existing edges are allowed
// for privately adding immutable reference endpoints during initial edge import.
std::vector<std::pair<std::string,int64_t>> insertInitialObjects(
    ObjectStore&,WriteTxn&,std::span<const InitialObjectRecord>);
}
