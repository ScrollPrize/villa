#pragma once
#include "vc_object_store/objects/Transactions.hpp"
#include "vc_object_store/Uuid.hpp"
#include <nlohmann/json.hpp>

namespace vc::objects::detail {
// One field codec; enclosing legacy formats retain their own validation policy.
inline nlohmann::json objectVersionJson(const ObjectVersion& ref)
{
    return {{"store", ref.storeUuid}, {"generation", ref.generation},
            {"object", ref.objectUuid}, {"version", ref.version}};
}
inline ObjectVersion objectVersionFrom(const nlohmann::json& value, bool strictKeys = true)
{
    if (!value.is_object()) throw std::invalid_argument("object version must be a JSON object");
    if (strictKeys)
        for (const auto& item : value.items())
            if (item.key() != "store" && item.key() != "generation" && item.key() != "object" && item.key() != "version")
                throw std::invalid_argument("object version: unknown key " + item.key());
    return {value.at("store").get<std::string>(), value.at("generation").get<std::string>(),
            value.at("object").get<std::string>(), value.at("version").get<int64_t>()};
}
inline ObjectVersion canonicalObjectVersion(ObjectVersion ref)
{
    ref.storeUuid = util::canonicalUuid(ref.storeUuid);
    ref.generation = util::canonicalUuid(ref.generation);
    ref.objectUuid = util::canonicalUuid(ref.objectUuid);
    return ref;
}
}
