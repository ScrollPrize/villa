#pragma once
#include "internal/Database.hpp"
#include "internal/Spatial.hpp"

namespace vc::objects::detail {
extern const char* kObjectCols;

using ObjectFields = std::optional<std::set<std::string>>;
bool wants(const ObjectFields& fields, const std::string& name);
std::string projectedObjectColumns(const ObjectFields& fields);

void appendJournal(sqlite3* db, const std::string& author, const std::string& hostname,
                   const std::array<unsigned char, 16>& uuid, int64_t generation, const char* changeKind);

void readCommon(RecordCommon& rec, const Statement& s, int uuidCol, int kindCol, int genCol);

std::optional<std::array<double, 6>> readBbox(const Statement& s, int firstCol);

void bindBbox(Statement& s, int firstIdx, const std::optional<std::array<double, 6>>& box);

FiberDetail readFiberDetail(const Statement& d);

PatchDetail readPatchDetail(const Statement& d);

TrackDetail readTrackDetail(const Statement& d);

ObjectRecord readObjectRow(sqlite3* db, const Statement& s, bool details = true);

std::vector<ObjectRecord> readObjectRows(sqlite3* db, Statement& rows, const ObjectFields& fields = std::nullopt);

std::optional<int64_t> objectIdByUuid(sqlite3* db, const std::array<unsigned char, 16>& uuid);

void writeObjectDetail(sqlite3* db, int64_t id, const ObjectRecord& rec);

void writeTags(sqlite3* db, const char* table, const char* idColumn, int64_t id, const std::vector<std::string>& tags);

void restoreCommon(sqlite3* db, const char* table, const RecordCommon& rec);

std::string upsertObjectImpl(sqlite3* db, const std::string& author, const std::string& host, const ObjectRecord& input,
                             bool preserveRecord = false, const char* importOperation = "import");

std::optional<int64_t> edgeIdByUuid(sqlite3* db, const std::array<unsigned char, 16>& uuid);

// Shared content-mutation stamp for setMeta/removeMeta: bumps
// generation, restamps changed_*, resets approval, journals.
void stampContentChange(sqlite3* db, const std::string& author, const std::string& host, const char* table,
                        int64_t id, const std::array<unsigned char, 16>& uuid, const char* changeKind);

std::string jsonPathForKey(const std::string& key);

std::optional<std::string> geometryHead(sqlite3* db, const std::string& objectUuid);
} // namespace vc::objects::detail
