#pragma once
#include "internal/RecordSql.hpp"
#include "internal/EdgeSql.hpp"
#include "internal/Versions.hpp"
#include "internal/ReferenceResolver.hpp"

namespace vc::objects::detail {
struct ResolvedObject {
    sqlite3* source;
    ObjectRecord record;
};
// Inherited lookup is for an overlay's before-state only; ordinary consumers
// use visible lookup, where a tombstone stops fallback.
// includeDeleted returns the winning tombstone's owner, never an older live row.
inline sqlite3* objectSource(sqlite3* db,const std::string& uuid,bool inherited=false,bool includeDeleted=false)
{
    if(const auto pool=activeStatementPools.find(db);pool!=activeStatementPools.end()&&pool->second->objectSource)
        return pool->second->objectSource(uuid,inherited,includeDeleted);
    if(inherited)return nullptr;
    Statement row(db,"SELECT state FROM objects WHERE uuid=?",parseUuid(uuid));
    return row.row()&&(includeDeleted||row.i64(0)!=2)?db:nullptr;
}
inline std::optional<ResolvedObject> resolveObject(sqlite3* db,const std::string& uuid,bool inherited=false)
{
    auto* source=objectSource(db,uuid,inherited);
    if(!source)return std::nullopt;
    Statement row(source,std::string("SELECT ")+kObjectCols+" FROM objects WHERE uuid=? AND state<>2",parseUuid(uuid));
    if(!row.row())return std::nullopt;
    return ResolvedObject{source,readObjectRow(source,row)};
}
inline sqlite3* edgeSource(sqlite3* db,const std::string& uuid,bool inherited=false,bool includeDeleted=false)
{
    if(const auto pool=activeStatementPools.find(db);pool!=activeStatementPools.end()&&pool->second->edgeSource)
        return pool->second->edgeSource(uuid,inherited,includeDeleted);
    if(inherited)return nullptr;
    Statement row(db,"SELECT state FROM edges WHERE uuid=?",parseUuid(uuid));
    return row.row()&&(includeDeleted||row.i64(0)!=2)?db:nullptr;
}
struct ResolvedEdge {sqlite3* source;EdgeRecord record;};
inline std::optional<ResolvedEdge> resolveEdge(sqlite3* db,const std::string& uuid)
{
    auto* source=edgeSource(db,uuid);
    if(!source)return {};
    Statement row(source,std::string("SELECT ")+kEdgeCols+" FROM edges WHERE uuid=? AND state=0",parseUuid(uuid));
    return row.row()?std::optional(ResolvedEdge{source,readEdgeRow(source,row)}):std::nullopt;
}
inline std::vector<sqlite3*> querySources(sqlite3* db)
{
    const auto pool=activeStatementPools.find(db);
    return pool!=activeStatementPools.end()&&pool->second->querySources?pool->second->querySources():std::vector{db};
}
// Enumerate incident visible records once, regardless of physical partition.
inline std::vector<EdgeRecord> incidentEdges(sqlite3* db,const std::string& uuid)
{
    std::vector<EdgeRecord> result;
    for(auto* source:querySources(db)){
        Statement rows(source,std::string("SELECT ")+kEdgeCols+
            " FROM edges WHERE state=0 AND (a_uuid=?1 OR b_uuid=?1) ORDER BY id",parseUuid(uuid));
        while(rows.row()){
            auto edge=readEdgeRow(source,rows);
            if(edgeSource(db,edge.uuid)==source)result.push_back(std::move(edge));
        }
    }
    return result;
}
inline ObjectVersion objectVersion(const ResolvedObject& object)
{
    return {storeUuidText(object.source),object.record.versionGeneration,object.record.uuid,object.record.version.value_or(0)};
}
inline std::array<uint8_t,56> versionReferenceBytes(const ObjectVersion& ref)
{
    if(ref.version<=0)throw std::invalid_argument("object reference: invalid version");
    std::array<uint8_t,56> bytes{};
    const std::array<std::string,3> ids{ref.storeUuid,ref.generation,ref.objectUuid};
    for(size_t i=0;i<ids.size();++i){const auto id=parseUuid(ids[i]);std::copy(id.begin(),id.end(),bytes.begin()+16*i);}
    for(size_t i=0;i<8;++i)bytes[48+i]=uint64_t(ref.version)>>(8*i);
    return bytes;
}
inline ObjectVersion versionReferenceFromBytes(std::span<const uint8_t> bytes)
{
    if(bytes.size()!=56)throw std::invalid_argument("object reference: invalid binary size");
    std::array<std::string,3> ids;
    for(size_t i=0;i<ids.size();++i){std::array<uint8_t,16> id;std::copy_n(bytes.begin()+16*i,16,id.begin());ids[i]=formatUuid(id);}
    uint64_t version=0;for(size_t i=0;i<8;++i)version|=uint64_t(bytes[48+i])<<(8*i);
    if(version==0||version>INT64_MAX)throw std::invalid_argument("object reference: invalid binary version");
    return {ids[0],ids[1],ids[2],int64_t(version)};
}
struct ResolvedObjectVersion {sqlite3* source;ObjectRevision revision;};
std::optional<ResolvedObjectVersion> resolveObjectVersion(sqlite3*,const ObjectVersion&);
std::optional<ResolvedObjectVersion> resolveObjectRevision(sqlite3*,const std::string&);
inline std::optional<ResolvedObjectVersion> resolveObjectRevisionKey(sqlite3* db, const ObjectRevisionKey& key)
{
    if (const auto* version = std::get_if<ObjectVersion>(&key)) return resolveObjectVersion(db, *version);
    return resolveObjectRevision(db, std::get<std::string>(key));
}
}
