#pragma once
#include "internal/RecordSql.hpp"

namespace vc::objects::detail {
// Ordinary UUID-ordered selection, in one composed snapshot. Physical SQL
// cursors stay open for the scan; hydration is bounded and batched per source.
// Volume/geometry-measured ordering/filtering is handled by its separate query path.
void scanVisibleObjects(sqlite3*, const ObjectSelection&, const ObjectFields&,
                        const std::function<bool(ObjectRecord&&)>&);
}
