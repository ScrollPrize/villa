#pragma once
#include "vc_object_store/objects/Transactions.hpp"
#include "vc_object_store/Uuid.hpp"
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
template<class Reference>
Reference referenceFrom(const nlohmann::json& value, const char* key)
{
    if (!value.is_object()) throw std::invalid_argument("version reference must be a JSON object");
    for (const auto& item : value.items())
        if (item.key() != "store" && item.key() != "generation" && item.key() != key && item.key() != "version")
            throw std::invalid_argument("version reference: unknown key " + item.key());
    const auto& number = value.at("version");
    if (!number.is_number_integer() ||
        (number.is_number_unsigned() && number.get<uint64_t>() > uint64_t(std::numeric_limits<int64_t>::max())) ||
        number.get<int64_t>() <= 0)
        throw std::invalid_argument("version must be a positive integer");
    return {util::canonicalUuid(value.at("store").get<std::string>()), util::canonicalUuid(value.at("generation").get<std::string>()),
            util::canonicalUuid(value.at(key).get<std::string>()), number.get<int64_t>()};
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
}
