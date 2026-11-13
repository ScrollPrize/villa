#pragma once
#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/client/HostedStore.hpp"
#include <filesystem>
#include <stdexcept>
#include <source_location>

inline void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
template<class F> void mustFail(F&& callback, const std::source_location location=std::source_location::current())
{
    try { callback(); } catch (const std::exception&) { return; }
    throw std::runtime_error(std::string("expected operation to fail at ")+location.file_name()+":"+std::to_string(location.line()));
}
struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() / vc::objects::util::newUuid();
    TemporaryDirectory() { std::filesystem::create_directory(path); }
    ~TemporaryDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};

// Fixture construction only: deliberately stale/future counters are needed to
// test submission conflicts. Production code must retain its actual loaded ref
// or use the checked current-owner selector, never manufacture one this way.
inline vc::objects::RecordSelection testNumberedGuards(const vc::objects::StoreReader& reader,
    const vc::objects::VersionSelection& numbers, bool relationships = false)
{
    vc::objects::RecordSelection result;
    for (const auto& [uuid, number] : numbers) {
        if (!number) { result[uuid] = std::nullopt; continue; }
        if (relationships) {
            vc::objects::EdgeSelection query; query.uuids = {uuid}; query.includeRetired = true;
            const auto edges = reader.selectEdges(query);
            auto ref = edges.empty() ? vc::objects::RelationshipVersion{reader.status().storeUuid, reader.status().generation, uuid, *number}
                                     : edges.front().reference();
            ref.version = *number; result[uuid] = ref;
        } else {
            const auto current = reader.currentObject(uuid, true);
            auto ref = current ? current->reference() : vc::objects::ObjectVersion{reader.status().storeUuid, reader.status().generation, uuid, *number};
            ref.version = *number; result[uuid] = ref;
        }
    }
    return result;
}
inline vc::objects::RecordSelection testNumberedGuards(vc::objects::ObjectStore& store,
    const vc::objects::VersionSelection& numbers, bool relationships = false)
{
    const auto read = store.read();
    return testNumberedGuards(vc::objects::LocalStoreReader(store, read), numbers, relationships);
}
inline vc::objects::RecordSelection testNumberedGuards(vc::objects::HostedStore& store,
    const vc::objects::VersionSelection& numbers, bool relationships = false)
{
    vc::objects::RecordSelection result;
    for (const auto& [uuid, number] : numbers) {
        if (!number) { result[uuid] = std::nullopt; continue; }
        if (relationships) {
            vc::objects::EdgeSelection query; query.uuids = {uuid}; query.includeRetired = true; query.limit = 1;
            const auto edges = store.selectEdges(query);
            auto ref = edges.empty() ? vc::objects::RelationshipVersion{store.status().storeUuid, store.status().generation, uuid, *number}
                                     : edges.front().reference();
            ref.version = *number; result[uuid] = ref;
            continue;
        }
        const auto current = store.currentObjectRevision(uuid);
        auto ref = current ? current->reference() : vc::objects::ObjectVersion{store.status().storeUuid, store.status().generation, uuid, *number};
        ref.version = *number;
        result[uuid] = ref;
    }
    return result;
}
inline vc::objects::RecordSelection testNumberedGuards(vc::objects::ObjectStoreSession& session,
    const vc::objects::AttachmentIdentity& attachment, const vc::objects::VersionSelection& numbers, bool relationships = false)
{
    vc::objects::RecordSelection result;
    session.readStore(attachment, [&](const vc::objects::StoreReader& reader, const auto&) { result = testNumberedGuards(reader, numbers, relationships); });
    return result;
}
