#pragma once
#include "vc_object_store/objects/ObjectCollection.hpp"
struct sqlite3;

namespace vc::objects::detail {
// Private construction access. Published immutable stores can only be opened
// readonly; this access never reopens a pre-existing database for writing.
inline constexpr const char* immutableImportFeature="immutable-initial-v1";
bool immutableInitial(sqlite3*);
struct InitialCollectionAccess {
    static ObjectCollection create(const std::filesystem::path&,
        const std::vector<std::filesystem::path>&,const OpenOptions&);
    static CollectionMember sealShard(ObjectCollection&,CollectionTxn&);
    static sqlite3* connection(ObjectCollection&,CollectionTxn&);
    // Builder-owned private destination and already completed shard descriptors.
    // Writes routing directly; never opens or validates a collection/shard.
    static void writeManifest(const std::filesystem::path&,
        const std::vector<CollectionMember>&,const OpenOptions&,bool uuidHashRouting);
};
}
