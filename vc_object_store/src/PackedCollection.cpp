#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "internal/Database.hpp"

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
        Statement table(db.get(), "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='collection_members'");
        if (!table.row()) continue;
        exec(db.get(), "BEGIN IMMEDIATE");
        try {
            // This is physical relocation of a private snapshot, not an object
            // edit. UUIDs, lineage/commit pins and retained history remain intact.
            Statement refs(db.get(), "SELECT database_uuid FROM collection_members ORDER BY ordinal");
            std::vector<std::string> targets;
            while (refs.row()) targets.push_back(formatUuid(refs.blob16(0)));
            for (const auto& target : targets) {
                builder.databasePath(target); // Fail if dependency closure is incomplete.
                execute(db.get(), "UPDATE collection_members SET locator=? WHERE database_uuid=?",
                        packedMemberReference(target), parseUuid(target));
            }
            exec(db.get(), "UPDATE store_write_state SET dirty=0 WHERE singleton=1; COMMIT");
        } catch (...) { exec(db.get(), "ROLLBACK"); throw; }
    }
    // Resolve all rewritten store_references and validate pins/ranges before publication.
    { auto checked = ObjectCollection::open(builder.databasePath(members.front().databaseUuid), OpenOptions{.readOnly=true}); }
    builder.finalize(members.front().databaseUuid);
}
} // namespace vc::objects
