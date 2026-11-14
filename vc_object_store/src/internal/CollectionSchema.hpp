#pragma once
#include "internal/Sqlite.hpp"

namespace vc::objects::detail {
// Shared physical schema for normal collection construction and the explicit
// converter. Numeric reservations apply only to object rows, never edges.
inline void createCollectionTables(sqlite3* db)
{
    exec(db,R"SQL(
CREATE TABLE collection_members(
 ordinal INTEGER PRIMARY KEY, database_uuid BLOB NOT NULL UNIQUE CHECK(length(database_uuid)=16),
 store_uuid BLOB NOT NULL CHECK(length(store_uuid)=16), generation BLOB NOT NULL CHECK(length(generation)=16),
 commit_sequence INTEGER NOT NULL CHECK(commit_sequence>0),
 locator TEXT NOT NULL, partition INTEGER NOT NULL CHECK(partition IN (0,1)),
 object_low INTEGER NOT NULL, object_high INTEGER NOT NULL
);
CREATE TABLE collection_routes(
 key_space TEXT NOT NULL CHECK(key_space IN ('objects','objects.uuid','edges.uuid')),
 kind INTEGER NOT NULL CHECK(kind BETWEEN 0 AND 4),
 lower_bound INTEGER NOT NULL CHECK(lower_bound>0), upper_bound INTEGER NOT NULL CHECK(upper_bound>=lower_bound),
 divisor INTEGER NOT NULL CHECK(divisor>0), remainder INTEGER NOT NULL CHECK(remainder>=0 AND remainder<divisor),
 shift INTEGER NOT NULL CHECK(shift BETWEEN 0 AND 62),
 database_uuid BLOB NOT NULL REFERENCES collection_members(database_uuid),
 PRIMARY KEY(key_space,lower_bound,upper_bound,remainder)
) WITHOUT ROWID;
)SQL");
}
} // namespace vc::objects::detail
