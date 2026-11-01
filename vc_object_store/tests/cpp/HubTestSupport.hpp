#pragma once
// Shared by the hub tests over local paths and over SFTP.
#include "vc_object_store/objects/ObjectStore.hpp"
#include <nlohmann/json.hpp>
#include <string>

inline vc::objects::OpenOptions as(const std::string& author)
{
    vc::objects::OpenOptions options; options.author = author; return options;
}

inline std::string meta(vc::objects::ObjectStore& store, const std::string& uuid)
{
    auto txn = store.read();
    const auto object = store.objectByUuid(txn, uuid);
    return nlohmann::json::parse(object ? object->meta : store.edgeByUuid(txn, uuid)->meta).dump();
}

inline void setNote(vc::objects::ObjectStore& store, const std::string& uuid, const std::string& value)
{
    auto txn = store.write();
    store.setMeta(txn, uuid, "note", nlohmann::json(value).dump());
    txn.commit();
}
