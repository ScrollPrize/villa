#pragma once
#include "internal/Database.hpp"

namespace vc::objects::detail {
// Resolve an immutable reference in the transaction's pinned store scope.
// The predicate is a storage lookup, never a fallback to another version.
// Consumers use the same lookup whether the reference is local or borrowed.
inline sqlite3* resolveReference(sqlite3* db, const std::function<bool(sqlite3*)>& matches)
{
    if (matches(db)) return db;
    const auto pool = activeStatementPools.find(db);
    return pool != activeStatementPools.end() && pool->second->referenceSource
        ? pool->second->referenceSource(matches) : nullptr;
}
}
