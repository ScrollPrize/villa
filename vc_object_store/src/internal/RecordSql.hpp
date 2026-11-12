#pragma once
#include "internal/Database.hpp"
#include "internal/Spatial.hpp"

namespace vc::objects::detail {
extern const char* kObjectCols;
// Stage one caller-supplied object ID for native insertion. Edges use UUIDs.
class ExplicitRecordId {
public:
    ExplicitRecordId(sqlite3*, const char* keySpace, const std::string& uuid, int64_t id);
    void finish();
private:
    sqlite3* db_;
    std::string keySpace_;
    int64_t id_;
    std::optional<int64_t> next_;
    Savepoint save_;
};
void readLifecycle(sqlite3*, int64_t id, ObjectRecord&);
void writeLifecycle(sqlite3*, int64_t id, const ObjectRecord&);
std::string activeObjectSql(const std::string& id);
std::string activeEdgeSql();
std::pair<std::string, std::string> lifecycleBinding(sqlite3*, const EdgeRecord&);
bool lifecycleBindingMatches(sqlite3*, const EdgeRecord&);
void bindLifecycle(sqlite3*, const EdgeRecord&, const std::string& aEpoch, const std::string& bEpoch);
struct FrameCursor {
    double tieLow = 0, tieHigh = 0;
    double stored = 0;
    std::string uuid;
};
struct FrameFilter {
    std::optional<FrameCursor> cursor;
    std::optional<std::string> unboundVolume;
};
std::unique_ptr<Statement> objectSelectionQuery(sqlite3* db, const ObjectSelection& selection,
                                               const std::string& columns, const FrameFilter* frame = nullptr);
using ObjectFields = std::optional<std::set<std::string>>;
bool wants(const ObjectFields& fields, const std::string& name);
std::string projectedObjectColumns(const ObjectFields& fields);
void appendJournal(sqlite3* db, const std::string& author, const std::string& hostname,
                   const std::array<unsigned char, 16>& uuid, int64_t commitSequence, const char* changeKind,
                   const std::string& deletedKind = {});
void readCommon(RecordCommon& rec, const Statement& s, int uuidCol, int kindCol, int genCol);
std::optional<std::array<double, 6>> readBbox(const Statement& s, int firstCol);
void bindBbox(Statement& s, int firstIdx, const std::optional<std::array<double, 6>>& box);
ObjectRecord readObjectRow(sqlite3* db, const Statement& s, bool details = true,
                           const std::string* initialGeneration = nullptr, bool lifecycle = true);
std::vector<ObjectRecord> readObjectRows(sqlite3* db, Statement& rows, const ObjectFields& fields = std::nullopt);
std::optional<int64_t> objectIdByUuid(sqlite3* db, const std::array<unsigned char, 16>& uuid);
void writeObjectDetail(sqlite3* db, int64_t id, const ObjectRecord& rec);
void insertObjectDetail(sqlite3* db, int64_t id, const ObjectRecord& rec);
ObjectRecord prepareObjectRecord(const ObjectRecord&);
void normalizeObjectCoordinates(ObjectRecord&,bool addDefaultCoordinates,sqlite3* existingStore=nullptr);
void normalizeObjectCoordinates(nlohmann::json& meta,const std::string& kind,bool addDefaultCoordinates);
void insertObjectRow(sqlite3*,const ObjectRecord&,int64_t id,int64_t commitSequence,
    const std::string& author,const std::string& host,const std::string& time,bool preserveRecord=false,
    const std::string* geometryManifest=nullptr);
void writeTags(sqlite3* db, const char* table, const char* idColumn, int64_t id, const std::vector<std::string>& tags);
void insertTags(sqlite3* db, const char* table, const char* idColumn, int64_t id, const std::vector<std::string>& tags);
void restoreCommon(sqlite3* db, const char* table, const RecordCommon& rec);
std::string upsertObjectImpl(sqlite3* db, const std::string& author, const std::string& host, const ObjectRecord& input,
                             bool preserveRecord = false, const char* importOperation = "import",
                             bool restoringExtensions = false);
// Shared stamp; the numeric ID is used only by object callers. Edge writes
// address the UUID directly and never allocate or store a numeric edge ID.
void stampContentChange(sqlite3* db, const std::string& author, const std::string& host, const char* table,
                        int64_t id, const std::array<unsigned char, 16>& uuid, const char* changeKind);
std::string jsonPathForKey(const std::string& key);
void validateGeometrySourceMetadata(sqlite3*, const std::string& objectUuid, const nlohmann::json& source);
std::optional<std::string> geometryHead(sqlite3* db, const std::string& objectUuid);
} // namespace vc::objects::detail
