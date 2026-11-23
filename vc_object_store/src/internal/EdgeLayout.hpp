#pragma once
#include "internal/Sqlite.hpp"
#include <set>

namespace vc::objects::detail {
struct EdgeColumnSpec {const char* name;const char* declaration;const char* fallback;};
std::span<const EdgeColumnSpec> edgeColumnSpecs();
std::set<std::string> selectedEdgeColumns(const OpenOptions&);
std::set<std::string> edgeColumns(sqlite3*);
bool hasEdgeColumn(sqlite3*,const std::string&);
void requireEdgeColumns(sqlite3*,std::initializer_list<const char*>);
void createEdgeSchema(sqlite3*,const OpenOptions&,bool immutable);
void requireAdjacencyIndex(sqlite3*);
// Logical read expressions supply only documented defaults for intentionally
// absent optional fields, never for a missing required key column.
std::string edgeReadColumn(sqlite3*,const std::string&,const std::string& prefix={});
std::string edgeReadSource(sqlite3*);
} // namespace vc::objects::detail
