#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/Uuid.hpp"
#include <set>
#include <stdexcept>

namespace vc::objects {
std::filesystem::path resolveStorePosition(const ObjectCollection& collection,
                                         const CollectionTxn& txn, const ExtensionSource& source)
{
    (void)collection.uuid(txn); // Reject closed or foreign transactions first.
    const auto store=util::canonicalUuid(source.storeUuid);
    const auto generation=util::canonicalUuid(source.generation);
    if(source.commitSequence<0)throw std::invalid_argument("store position has a negative commit sequence");
    std::set<std::filesystem::path> roots;
    for(const auto& member:collection.members()) {
        if(member.partition || member.storeUuid!=store || member.generation!=generation ||
           member.commitSequence!=source.commitSequence)continue;
        auto path=member.locator;
        if(isPackedDatabasePath(path)) {
            const auto container=packedContainerPath(path);
            if(packedDatabaseUuid(container)!=member.databaseUuid)
                throw std::invalid_argument("store position names a packed member, not a public root");
            path=container;
        }
        roots.insert(std::filesystem::canonical(path));
    }
    if(roots.empty())throw std::invalid_argument("store position is unknown or no longer matches the recorded snapshot");
    if(roots.size()!=1)throw std::invalid_argument("store position resolves to ambiguous public roots");
    return *roots.begin();
}
}
