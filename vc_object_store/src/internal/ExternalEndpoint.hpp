#pragma once
#include "internal/Database.hpp"
#include "geometry/GeometryToken.hpp"

namespace vc::objects::detail {
struct ExternalEndpointToken {std::string ownerUuid,objectUuid;int64_t generation;};
inline std::string externalEndpointToken(const std::string& owner,const std::string& object,int64_t generation)
{
    auto ownerHex=formatUuid(parseUuid(owner));ownerHex.erase(std::remove(ownerHex.begin(),ownerHex.end(),'-'),ownerHex.end());
    return "external:"+ownerHex+":"+currentGeometryToken(object,generation).substr(8);
}
inline std::optional<ExternalEndpointToken> parseExternalEndpointToken(const std::string& token)
{
    if(!token.starts_with("external:"))return {};
    if(token.size()<9+32+1||token[41]!=':')throw std::invalid_argument("invalid external endpoint token");
    const auto owner=parseCurrentGeometryToken("current:"+token.substr(9,32)+":1");
    const auto object=parseCurrentGeometryToken("current:"+token.substr(42));
    return ExternalEndpointToken{owner->objectUuid,object->objectUuid,object->generation};
}
inline std::optional<std::pair<std::string,std::string>> externalEndpointKey(sqlite3* db,const std::string& input,bool includeOverridden=false)
{
    const auto bytes=parseUuid(input);
    const auto uuid=formatUuid(bytes);
    Statement local(db,"SELECT 1 FROM objects WHERE uuid=?",bytes);
    const bool owned=local.row();
    if(owned&&!includeOverridden)return {};
    const auto found=activeStatementPools.find(db);
    if(found!=activeStatementPools.end()&&found->second->externalEndpointOwner) {
        if(const auto owner=found->second->externalEndpointOwner(uuid))return std::pair{*owner,uuid};
        return {};
    }
    // A standalone local endpoint does not become external merely because an
    // edge references it. includeOverridden only probes an available linked
    // source; absent a collection resolver, its own row is authoritative.
    if(owned)return {};
    Statement referenced(db,"SELECT 1 FROM edges WHERE deleted=0 AND (a_uuid=?1 OR b_uuid=?1) LIMIT 1",bytes);
    if(referenced.row())throw std::runtime_error("external endpoint requires a pinned collection transaction");
    return {};
}
inline StatementPool& externalEndpointPool(sqlite3* db)
{
    const auto it=activeStatementPools.find(db);
    if(it==activeStatementPools.end()||!it->second->externalEndpointRecord||!it->second->externalEndpointVersion)
        throw std::runtime_error("external endpoint requires a pinned collection transaction");
    return *it->second;
}
inline std::string externalEndpointVersion(sqlite3* db,const std::string& endpoint)
{
    const auto key=externalEndpointKey(db,endpoint,true);
    if(!key)throw std::invalid_argument("external endpoint key missing");
    const auto version=externalEndpointPool(db).externalEndpointVersion(endpoint);
    const auto parsed=parseExternalEndpointToken(version);
    if(!parsed||parsed->ownerUuid!=key->first||parsed->objectUuid!=key->second)
        throw std::runtime_error("external endpoint resolver identity mismatch");
    return externalEndpointToken(parsed->ownerUuid,parsed->objectUuid,parsed->generation);
}
inline ObjectRecord externalEndpointRecord(sqlite3* db,const std::string& endpoint)
{
    if(!externalEndpointKey(db,endpoint,true))throw std::invalid_argument("external endpoint key missing");
    auto record=externalEndpointPool(db).externalEndpointRecord(endpoint);
    if(record.uuid!=formatUuid(parseUuid(endpoint)))throw std::runtime_error("external endpoint resolver key mismatch");
    return record;
}
inline std::optional<std::string> externalEndpointObject(sqlite3* db,const std::string& token)
{
    const auto parsed=parseExternalEndpointToken(token);if(!parsed)return {};
    const auto endpoint=parsed->objectUuid;
    if(externalEndpointVersion(db,endpoint)!=externalEndpointToken(parsed->ownerUuid,parsed->objectUuid,parsed->generation))
        throw std::runtime_error("external endpoint generation changed");
    return endpoint;
}
}
