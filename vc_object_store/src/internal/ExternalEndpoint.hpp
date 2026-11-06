#pragma once
#include "internal/Database.hpp"
#include "geometry/GeometryToken.hpp"

namespace vc::objects::detail {
class ExternalEndpointPromotion {
    StatementPool& pool_;std::string previous_;
public:
    ExternalEndpointPromotion(sqlite3* db,const std::string& alias):pool_(*activeStatementPools.at(db)),previous_(pool_.promotedEndpointUuid){pool_.promotedEndpointUuid=formatUuid(parseUuid(alias));}
    ~ExternalEndpointPromotion(){pool_.promotedEndpointUuid=std::move(previous_);}
    ExternalEndpointPromotion(const ExternalEndpointPromotion&)=delete;
    ExternalEndpointPromotion& operator=(const ExternalEndpointPromotion&)=delete;
};
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
inline std::optional<std::pair<std::string,std::string>> externalEndpointKey(sqlite3* db,const std::string& alias,bool includeOverridden=false)
{
    Statement row(db,"SELECT owner_uuid,object_uuid FROM external_object_keys WHERE uuid=? AND (? OR NOT EXISTS(SELECT 1 FROM objects WHERE uuid=external_object_keys.uuid))",parseUuid(alias),int64_t(includeOverridden));
    if(!row.row())return {};
    return std::pair{formatUuid(row.blob16(0)),formatUuid(row.blob16(1))};
}
inline StatementPool& externalEndpointPool(sqlite3* db)
{
    const auto it=activeStatementPools.find(db);
    if(it==activeStatementPools.end()||!it->second->externalEndpointRecord||!it->second->externalEndpointVersion)
        throw std::runtime_error("external endpoint requires a pinned collection transaction");
    return *it->second;
}
inline std::string externalEndpointVersion(sqlite3* db,const std::string& alias)
{
    const auto key=externalEndpointKey(db,alias,true);
    if(!key)throw std::invalid_argument("external endpoint key missing");
    const auto version=externalEndpointPool(db).externalEndpointVersion(alias);
    const auto parsed=parseExternalEndpointToken(version);
    if(!parsed||parsed->ownerUuid!=key->first||parsed->objectUuid!=key->second)
        throw std::runtime_error("external endpoint resolver identity mismatch");
    return externalEndpointToken(parsed->ownerUuid,parsed->objectUuid,parsed->generation);
}
inline ObjectRecord externalEndpointRecord(sqlite3* db,const std::string& alias)
{
    if(!externalEndpointKey(db,alias,true))throw std::invalid_argument("external endpoint key missing");
    auto record=externalEndpointPool(db).externalEndpointRecord(alias);
    if(record.uuid!=formatUuid(parseUuid(alias)))throw std::runtime_error("external endpoint resolver alias mismatch");
    return record;
}
inline std::optional<std::string> externalEndpointAlias(sqlite3* db,const std::string& token)
{
    const auto parsed=parseExternalEndpointToken(token);if(!parsed)return {};
    Statement row(db,"SELECT uuid FROM external_object_keys WHERE owner_uuid=? AND object_uuid=?",parseUuid(parsed->ownerUuid),parseUuid(parsed->objectUuid));
    if(!row.row())throw std::runtime_error("external endpoint token is not linked by this store");
    const auto alias=formatUuid(row.blob16(0));
    if(externalEndpointVersion(db,alias)!=externalEndpointToken(parsed->ownerUuid,parsed->objectUuid,parsed->generation))
        throw std::runtime_error("external endpoint generation changed");
    return alias;
}
}
