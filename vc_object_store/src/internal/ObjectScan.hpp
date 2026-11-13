#pragma once
#include "internal/RecordSql.hpp"

namespace vc::objects::detail {
// Ordinary UUID-ordered selection, in one composed snapshot. Physical SQL
// cursors stay open for the scan; hydration is bounded and batched per source.
// Volume/geometry-measured ordering/filtering is handled by its separate query path.
void scanVisibleObjects(sqlite3*, const ObjectSelection&, const ObjectFields&,
                        const std::function<bool(ObjectRecord&&)>&);
// Same selection, overlay visibility and UUID ordering as ordinary records,
// reading only complete publication identity columns (no content hydration).
void scanVisibleObjectReferences(sqlite3*,const ObjectSelection&,
                                const std::function<bool(ObjectVersion&&)>&);
// Same cursor/hydration implementation, with the geometry head read from the
// selected row. An omitted descriptor is an absent head, never a base lookup.
struct SelectedGeometryHead {
    sqlite3* source;
    std::string reference,manifest;
};
void scanVisibleObjectRows(sqlite3*, const ObjectSelection&, const ObjectFields&,
    const std::function<bool(ObjectRecord&&,std::optional<SelectedGeometryHead>)>&,
    bool includeGeometryHeads=true);
}
