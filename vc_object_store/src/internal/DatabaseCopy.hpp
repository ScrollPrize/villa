#pragma once
#include "internal/Sqlite.hpp"
#include "internal/RecordOverrides.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include <functional>

namespace vc::objects::detail {
inline void backupSqliteSnapshot(sqlite3* source, sqlite3* destination)
{
    auto* backup=sqlite3_backup_init(destination,"main",source,"main");
    if(!backup)fail(destination,"initialize database copy");
    const int step=sqlite3_backup_step(backup,-1), finished=sqlite3_backup_finish(backup);
    if(step!=SQLITE_DONE)check(step,destination,"copy database snapshot");
    check(finished,destination,"finish database copy");
}
// Copy the caller's pinned SQLite snapshot, not a separately opened connection.
// Used by ordinary collection exports and packed imports alike.
inline void copySqliteSnapshot(sqlite3* source, const std::filesystem::path& destination)
{
    sqlite3* raw=nullptr;
    const int opened=sqlite3_open_v2(destination.string().c_str(), &raw,
        SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE, isPackedDatabasePath(destination)?packedStoreVfs():nullptr);
    std::unique_ptr<sqlite3,decltype(&sqlite3_close_v2)> output(raw,sqlite3_close_v2);
    check(opened,raw,"open database copy destination");
    backupSqliteSnapshot(source,raw);
    check(sqlite3_close(raw),raw,"close database copy destination");
    output.release();
}

inline void relocateCollectionMembers(sqlite3* db,
    const std::function<std::string(const std::string&)>& locator)
{
    Statement table(db,"SELECT 1 FROM sqlite_schema WHERE type='table' AND name='collection_members'");
    if(!table.row())return;
    Statement refs(db,"SELECT database_uuid FROM collection_members ORDER BY ordinal");
    std::vector<std::string> targets;
    while(refs.row())targets.push_back(refs.uuid(0));
    for(const auto& target:targets)
        execute(db,"UPDATE collection_members SET locator=? WHERE database_uuid=?",locator(target),parseUuid(target));
}
}
