#pragma once
#include "TestSupport.hpp"
#include "internal/Database.hpp"

// Audit persisted UUID fields through SQL as well as public string round trips.
inline void requireBinaryUuidStorage(const vc::objects::ObjectStore& store,const vc::objects::Txn& txn)
{
    using namespace vc::objects;
    using namespace vc::objects::detail;
    auto* db=ObjectStoreImplAccess::conn(txn);
    Statement tables(db,"SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'");
    while(tables.row()){
        const auto table=tables.text(0);
        Statement columns(db,"SELECT name,type FROM pragma_table_xinfo(?)",table);
        while(columns.row()){
            const auto column=columns.text(0);
            const bool reference=table=="relationship_revisions"&&(column=="a_revision"||column=="b_revision"||column=="a_geometry"||column=="b_geometry");
            if(column!="uuid"&&!column.ends_with("_uuid")&&!reference&&
                !(table=="hub_leases"&&column=="holder")&&!(table=="hub_upstream"&&column=="hub_replica"))continue;
            if(columns.text(1)!="BLOB")throw std::runtime_error("UUID column must declare BLOB: "+table+"."+column);
            const bool tagged=reference||(table=="history_expected"&&column=="revision_uuid");
            // All names originate from our schema. Quote anyway so the audit
            // remains valid for an extension containing punctuation.
            const auto quote=[](const std::string& s){std::string q="\"";for(char c:s){q+=c;if(c=='\"')q+=c;}return q+'\"';};
            const auto name=quote(column);
            Statement invalid(db,"SELECT count(*) FROM "+quote(table)+" WHERE "+name+" IS NOT NULL AND NOT "
                "(typeof("+name+")='blob' AND length("+name+")=16"+
                (tagged?" OR (typeof("+name+")='text' AND ("+name+" GLOB 'external:*' OR "+name+" GLOB 'current:*'))":"")+")");
            if(!invalid.row()||invalid.i64(0)!=0)throw std::runtime_error("UUID must use BLOB16 storage: "+table+"."+column);
        }
    }
    (void)store;
}
