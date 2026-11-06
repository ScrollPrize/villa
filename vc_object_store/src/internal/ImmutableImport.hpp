#pragma once
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "internal/InitialEndpoints.hpp"
struct sqlite3;

namespace vc::objects::detail {
// Private construction access. Published immutable stores can only be opened
// readonly; this access never reopens a pre-existing database for writing.
inline constexpr const char* immutableImportFeature="immutable-initial-v1";
bool immutableInitial(sqlite3*);
struct InitialCollectionAccess {
    static ObjectCollection create(const std::filesystem::path&,
        const std::vector<std::filesystem::path>&,const OpenOptions&);
    static void setEndpoints(ObjectCollection&,std::shared_ptr<const InitialEndpointDirectory>);
};
}
