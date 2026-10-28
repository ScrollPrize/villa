#pragma once
#include "sync/Baselines.hpp"
#include "geometry/Dependencies.hpp"
#include <functional>

// Installing current-state sync records (syncRecord JSON) into one store. Shared
// by two-store synchronization and hosted hub pushes/pulls.
namespace vc::objects::detail {
inline Json recordAt(const Json& state, const std::string& id) { return state.value(id, Json()); }
inline Json recordHead(const Json& record)
{
    return record.is_object() ? record.value("geometry", Json::object()).value("revision_uuid", Json()) : Json();
}
inline bool isObjectRecord(const Json& record) { return record.is_object() && record.at("record_type") == "objects"; }

// Every relationship touching an object, including dormant ones.
inline void incidentEdges(sqlite3* db, const std::string& object, std::set<std::string>& ids)
{
    Statement edges(db, "SELECT uuid FROM edges WHERE a_id=(SELECT id FROM objects WHERE uuid=?1) "
                        "OR b_id=(SELECT id FROM objects WHERE uuid=?1)", parseUuid(object));
    while (edges.row()) ids.insert(formatUuid(edges.blob16(0)));
}

// Remove, delete and upsert records that differ between before and merged.
// Run before copying geometry; applyHeadsAndEdges finishes the installation.
void applyObjects(ObjectStore&, WriteTxn&, const Json& before, const Json& merged);
void applyHeadsAndEdges(ObjectStore&, WriteTxn&, const Json& before, const Json& merged);

// Geometry for a revision UUID (JSON string or null) from whichever store holds it.
using DependencyLookup = std::function<std::optional<DependencyGeometry>(const Json& revision)>;
// Relationships in merged bound to geometry other than their endpoint's merged
// head, where the change touches their anchors or cannot be checked. Every
// endpoint of a merged relationship must itself be in merged.
std::vector<std::string> undecidedRelationships(const Json& merged, const DependencyLookup&);
}
