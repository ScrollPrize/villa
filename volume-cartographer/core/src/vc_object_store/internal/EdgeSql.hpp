#pragma once
#include "internal/Anchors.hpp"
#include "internal/RecordSql.hpp"
namespace vc::objects::detail {
extern const char* kEdgeCols;

Anchor readAnchor(const Statement& s, int firstCol);

void bindAnchor(Statement& s, int firstIdx, const Anchor& a);

std::string uuidTextByObjectId(sqlite3* db, int64_t id);

EdgeRecord readEdgeRow(sqlite3* db, const Statement& s, bool details = true);
std::vector<EdgeRecord> readEdgeRows(sqlite3* db, Statement& rows);

std::string addEdgeImpl(sqlite3* db, const std::string& author, const std::string& host, const EdgeRecord& input,
                        bool preserveRecord = false, const char* importOperation = "import");


} // namespace vc::objects::detail
