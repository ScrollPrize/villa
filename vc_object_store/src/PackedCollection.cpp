#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "internal/Database.hpp"
#include "internal/DatabaseCopy.hpp"

namespace vc::objects {
void packObjectCollection(const std::filesystem::path& source,
                          const std::filesystem::path& destination,
                          const PackedStoreOptions& options)
{
    auto collection = ObjectCollection::open(source, OpenOptions{.readOnly=true});
    auto snapshot = collection.read();
    if (!collection.provisioning(snapshot).sealed)
        throw std::invalid_argument("packing requires a sealed collection root");
    const auto members = collection.members();
    if (members.empty()) throw std::runtime_error("cannot pack an empty collection directory");
    PackedStoreBuilder builder(destination, options);
    for (const auto& member : members) builder.importDatabase(member.databaseUuid, member.locator);
    for (const auto& member : members) {
        using namespace detail;
        std::unique_ptr<sqlite3, decltype(&sqlite3_close_v2)> db(
            openConnection(builder.databasePath(member.databaseUuid), SQLITE_OPEN_READWRITE, {}, false), sqlite3_close_v2);
        exec(db.get(), "BEGIN IMMEDIATE");
        try {
            // External locators are relocated by the shared snapshot importer.
            relocateCollectionMembers(db.get(),[&](const std::string& target) {
                builder.databasePath(target); // Fail if dependency closure is incomplete.
                return packedMemberReference(target);
            });
            exec(db.get(), "UPDATE store_write_state SET dirty=0 WHERE singleton=1; COMMIT");
        } catch (...) { exec(db.get(), "ROLLBACK"); throw; }
    }
    // Resolve all rewritten store_references and validate pins/ranges before publication.
    { auto checked = ObjectCollection::open(builder.databasePath(members.front().databaseUuid), OpenOptions{.readOnly=true}); }
    builder.finalize(members.front().databaseUuid);
}
} // namespace vc::objects
