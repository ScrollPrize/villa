#include "internal/RecordOverrides.hpp"
#include "internal/ExtensionHistory.hpp"
#include "internal/History.hpp"
#include "vc_object_store/objects/PackedStore.hpp"

namespace vc::objects::detail {
namespace {
std::filesystem::path externalDirectory(const std::filesystem::path& path)
{
    return std::filesystem::absolute(isPackedDatabasePath(path)?packedContainerPath(path):path).lexically_normal().parent_path();
}
}
std::string rebaseExternalLocator(const std::string& locator, const std::filesystem::path& source,
                                 const std::filesystem::path& destination)
{
    namespace fs = std::filesystem;
    if(locator.empty() || locator.find("://") != std::string::npos || !fs::path(locator).is_relative())return locator;
    const auto from=externalDirectory(source), to=externalDirectory(destination);
    if(from==to)return locator;
    const auto absolute=(from/locator).lexically_normal();
    const auto relative=absolute.lexically_relative(to);
    return (relative.empty()?absolute:relative).generic_string();
}

void relocateExternalLocators(sqlite3* db, const std::filesystem::path& source,
                              const std::filesystem::path& destination)
{
    if(externalDirectory(source)==externalDirectory(destination))return;
    // This connection owns an unpublished copy. Mutating only physical hints
    // must not trigger history, version increments or sealed-store guards.
    struct Triggers {
        sqlite3* db; int enabled=0;
        explicit Triggers(sqlite3* value):db(value) {
            check(sqlite3_db_config(db,SQLITE_DBCONFIG_ENABLE_TRIGGER,-1,&enabled),db,"read trigger policy");
            check(sqlite3_db_config(db,SQLITE_DBCONFIG_ENABLE_TRIGGER,0,nullptr),db,"suspend export triggers");
        }
        ~Triggers(){sqlite3_db_config(db,SQLITE_DBCONFIG_ENABLE_TRIGGER,enabled,nullptr);}
    } triggers(db);
    const auto rebase=[&](const std::string& value){return rebaseExternalLocator(value,source,destination);};
    for(const auto* table:{"objects","tracks"}) {
        const std::string column=std::string_view(table)=="objects"?"storage_ref":"dataset_ref";
        const std::string key=std::string_view(table)=="objects"?"id":"object_id";
        Statement rows(db,"SELECT "+key+","+column+" FROM "+table+
            (std::string_view(table)=="objects"?" WHERE storage_kind='file'":"")+" ORDER BY "+key);
        Statement update(db,"UPDATE "+std::string(table)+" SET "+column+"=? WHERE "+key+"=?");
        while(rows.row()) {const auto old=rows.text(1), value=rebase(old);if(value!=old)update.run(value,rows.i64(0));}
    }
    Statement history(db,"SELECT 1 FROM sqlite_schema WHERE type='table' AND name='object_revisions'");
    if(!history.row())return;
    using Json=nlohmann::json;
    transformArchivedObjectRecords(db,[&](const std::string&,Json& record) {
        const auto field=[&](Json& object,const char* name) {
            const auto it=object.find(name);
            if(it!=object.end()&&it->is_string())*it=rebase(it->get<std::string>());
        };
        if(record.value("storage_kind",std::string())=="file")field(record,"storage_ref");
        if(record.contains("track")&&record["track"].is_object())field(record["track"],"dataset_ref");
    });
}

void prepareOverrideReferences(sqlite3* source,sqlite3* db,ObjectRecord& record)
{
    if(source==db)return;
    const auto rebase = [&](std::string& locator) {
        locator=rebaseExternalLocator(locator,sqlite3_db_filename(source,"main"),sqlite3_db_filename(db,"main"));
    };
    if(record.storageKind=="file") rebase(record.storageRef);
    if(record.track) rebase(record.track->datasetRef);
    referenceExtensionSchemas(source,db,"objects",record.meta);
}
bool createRecordOverride(sqlite3* db,const char* table,const std::string& uuid)
{
    const bool object=std::string_view(table)=="objects";
    if(!object&&std::string_view(table)!="edges")throw std::invalid_argument("invalid override kind");
    auto* source=object?objectSource(db,uuid):edgeSource(db,uuid);
    if(!source||source==db)return false;
    Savepoint save(db);
    const auto id=allocateRecordId(db,table);
    RecordCommon common;
    if(object){
        auto record=resolveObject(db,uuid)->record;common=record;
        if (record.createdAt.size() == 20 && preciseObjectTimestamp(record.kind))
            record.createdAt = refineLegacyCreationStamp(record.createdAt, retainedCreationPublication(source, uuid));
        prepareOverrideReferences(source,db,record);
        Statement descriptor(source,"SELECT geometry_manifest FROM objects WHERE uuid=?",parseUuid(uuid));
        descriptor.row();const auto text=descriptor.optText(0);
        insertObjectRow(db,record,id,record.commitSequence,record.createdBy,record.createdHost,record.createdAt,true,text?&*text:nullptr);
        insertObjectDetail(db,id,record);insertTags(db,"object_tags","object_id",id,record.tags);
        inheritSpatialBounds(source,db,record,id);
        execute(db,"UPDATE objects SET state=?,lifecycle_epoch=? WHERE uuid=?",int64_t(record.retired),record.lifecycleEpoch,parseUuid(uuid));
    }else{
        const auto record=resolveEdge(db,uuid)->record;common=record;
        referenceExtensionSchemas(source,db,table,record.meta);
        insertEdgeRow(db,record,id,record.commitSequence,record.createdBy,record.createdHost,record.createdAt,true);
    }
    // Copying a before-state is not a publication. Keep its stamps and
    // version; the ordinary mutation advances them once in this transaction.
    execute(db,std::string("UPDATE ")+table+" SET version=?,version_generation=?,changed_by=?,changed_at=?,changed_host=? WHERE uuid=?",
        common.version.value_or(1),parseUuid(common.versionGeneration),common.changedBy,common.changedAt,common.changedHost,parseUuid(uuid));
    save.release();return true;
}
}
