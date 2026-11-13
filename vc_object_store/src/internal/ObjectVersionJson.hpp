#pragma once
#include "vc_object_store/objects/Transactions.hpp"
#include "vc_object_store/Uuid.hpp"
#include "internal/JsonFields.hpp"
#include <nlohmann/json.hpp>
#include <limits>

namespace vc::objects::detail {
// One complete-reference field codec for every consumer.
inline nlohmann::json referenceJson(const std::string& store, const std::string& generation,
    const char* key, const std::string& uuid, int64_t version)
{
    if (version <= 0) throw std::invalid_argument("version must be a positive integer");
    return {{"store", util::canonicalUuid(store)}, {"generation", util::canonicalUuid(generation)},
            {key, util::canonicalUuid(uuid)}, {"version", version}};
}
inline nlohmann::json objectVersionJson(const ObjectVersion& ref)
{ return referenceJson(ref.storeUuid, ref.generation, "object", ref.objectUuid, ref.version); }
inline int64_t versionNumberFrom(const nlohmann::json& number)
{
    if (!number.is_number_integer() ||
        (number.is_number_unsigned() && number.get<uint64_t>() > uint64_t(std::numeric_limits<int64_t>::max())) ||
        number.get<int64_t>() <= 0)
        throw std::invalid_argument("version must be a positive integer");
    return number.get<int64_t>();
}
template<class Reference>
Reference referenceFrom(const nlohmann::json& value, const char* key)
{
    if (!value.is_object()) throw std::invalid_argument("version reference must be a JSON object");
    for (const auto& item : value.items())
        if (item.key() != "store" && item.key() != "generation" && item.key() != key && item.key() != "version")
            throw std::invalid_argument("version reference: unknown key " + item.key());
    const auto version = versionNumberFrom(value.at("version"));
    return {util::canonicalUuid(value.at("store").get<std::string>()), util::canonicalUuid(value.at("generation").get<std::string>()),
            util::canonicalUuid(value.at(key).get<std::string>()), version};
}
inline ObjectVersion objectVersionFrom(const nlohmann::json& value)
{ return referenceFrom<ObjectVersion>(value, "object"); }
inline RelationshipVersion relationshipVersionFrom(const nlohmann::json& value)
{ return referenceFrom<RelationshipVersion>(value, "relationship"); }
inline nlohmann::json relationshipVersionJson(const RelationshipVersion& ref)
{ return referenceJson(ref.storeUuid, ref.generation, "relationship", ref.relationshipUuid, ref.version); }
inline ObjectVersion canonicalObjectVersion(ObjectVersion ref)
{
    ref.storeUuid = util::canonicalUuid(ref.storeUuid);
    ref.generation = util::canonicalUuid(ref.generation);
    ref.objectUuid = util::canonicalUuid(ref.objectUuid);
    return ref;
}
inline nlohmann::json objectVersionsJson(const std::vector<ObjectVersion>& refs)
{
    auto result = nlohmann::json::array();
    for (const auto& ref : refs) result.push_back(objectVersionJson(ref));
    return result;
}
inline std::vector<ObjectVersion> objectVersionsFrom(const nlohmann::json& refs)
{
    std::vector<ObjectVersion> result;
    for (const auto& ref : jsonArray(refs, "object references")) result.push_back(objectVersionFrom(ref));
    return result;
}
// StoreStatus also carries the position of a read snapshot. FeedCursor is only
// for change-feed pagination, never a second spelling of that position.
inline nlohmann::json storePositionJson(const StoreStatus& status)
{ return {{"store", status.storeUuid}, {"generation", status.generation}, {"commit_sequence", status.commitSequence}}; }
inline int64_t commitSequenceFrom(const nlohmann::json& sequence)
{
    if (!sequence.is_number_integer() || (sequence.is_number_unsigned() && sequence.get<uint64_t>() > uint64_t(INT64_MAX)) || sequence.get<int64_t>() < 0)
        throw std::invalid_argument("commit_sequence must be a nonnegative int64");
    return sequence.get<int64_t>();
}
inline StoreStatus storePositionFrom(const nlohmann::json& value)
{
    return {util::canonicalUuid(value.at("store").get<std::string>()), util::canonicalUuid(value.at("generation").get<std::string>()), commitSequenceFrom(value.at("commit_sequence"))};
}
inline nlohmann::json storeStatusJson(const StoreStatus& status)
{
    auto value = storePositionJson(status);
    value.update({{"feed_start", status.feedStart}, {"history_enabled", status.historyEnabled}, {"extensions_present", status.extensionsPresent}});
    return value;
}
inline StoreStatus storeStatusFrom(const nlohmann::json& value)
{
    auto status = storePositionFrom(value);
    status.feedStart = value.at("feed_start").get<int64_t>();
    status.historyEnabled = value.at("history_enabled").get<bool>();
    status.extensionsPresent = value.at("extensions_present").get<bool>();
    return status;
}
}
