#pragma once
#include "internal/Anchors.hpp"
#include "internal/RecordSql.hpp"
namespace vc::objects::detail {
// Shared semantics for SQL current queries and immutable relationship records.
std::string fiberAdjacencySql(std::optional<bool> adjacent);
bool matchesFiberAdjacency(const EdgeRecord&, std::optional<bool> adjacent);
void validateEdgeSelection(const EdgeSelection&);
bool matchesEdgeSelection(const EdgeRecord&, const EdgeSelection&);
extern const char* kEdgeCols;
std::unique_ptr<Statement> edgeSelectionQuery(sqlite3* db, const EdgeSelection& selection,
    const std::string& columns, const std::optional<std::vector<int64_t>>& endpointIds = std::nullopt);
std::string projectedEdgeColumns(const ObjectFields& fields);

Anchor readAnchor(const Statement& s, int firstCol);

void bindAnchor(Statement& s, int firstIdx, const Anchor& a);

std::string uuidTextByObjectId(sqlite3* db, int64_t id);

EdgeRecord readEdgeRow(sqlite3* db, const Statement& s, bool details = true,
                       const ObjectFields& fields = std::nullopt);
std::vector<EdgeRecord> readEdgeRows(sqlite3* db, Statement& rows, const ObjectFields& fields = std::nullopt);
std::vector<int64_t> componentIds(sqlite3* db, const std::string& start,
                                 const std::vector<std::string>& kinds, bool includeRetired = false,
                                 std::optional<bool> adjacent = std::nullopt);

std::string addEdgeImpl(sqlite3* db, const std::string& author, const std::string& host, const EdgeRecord& input,
                        bool preserveRecord = false, const char* importOperation = "import");


} // namespace vc::objects::detail
