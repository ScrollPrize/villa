// A revision saved before an object was updated names a version that a store without history no longer retains: selecting its
// relationships must be refused with an error, never dereference an unresolved version (review of the selected-graph read).
#include "TestSupport.hpp"
#include <iostream>

using namespace vc::objects;

int main() try
{
    TemporaryDirectory directory;
    OpenOptions options; options.author = "stale-revision-test"; options.history = HistoryMode::Off;
    auto store = ObjectStore::create(directory.path / "stale.vcobj", options);
    std::string uuid;
    {
        auto txn = store.write();
        ObjectRecord record; record.kind = "fiber"; record.meta = R"({"coordinates":{"volume_uuid":"a","scale_factor":1}})";
        uuid = store.upsertObject(txn, record);
        txn.commit();
    }
    std::optional<ObjectRevision> saved;
    {
        auto txn = store.read();
        saved = store.currentObjectRevision(txn, uuid);
        require(saved.has_value(), "the object is readable");
        // While it is current, its relationships are selectable.
        (void)store.relationshipsAt(txn, std::vector<ObjectRevision>{*saved});
    }
    {
        auto txn = store.write();
        store.setMeta(txn, uuid, "note", "true");       // history off: the saved version is not retained any more
        txn.commit();
    }
    {
        auto txn = store.read();
        const auto current = store.currentObjectRevision(txn, uuid);
        require(current && current->version != saved->version, "the update made a new version");
        mustFail([&] { (void)store.relationshipsAt(txn, std::vector<ObjectRevision>{*saved}); });
        (void)store.relationshipsAt(txn, std::vector<ObjectRevision>{*current});   // the current revision still works
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
}
