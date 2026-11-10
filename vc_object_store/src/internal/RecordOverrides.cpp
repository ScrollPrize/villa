#include "internal/RecordOverrides.hpp"
#include "internal/ExtensionHistory.hpp"
#include "internal/History.hpp"
#include "vc_object_store/objects/PackedStore.hpp"

namespace vc::objects::detail {
void prepareOverrideReferences(sqlite3* source,sqlite3* db,ObjectRecord& record)
{
    if(source==db)return;
    const auto rebase = [&](std::string& locator) {
      if(!locator.empty()&&locator.find("://")==std::string::npos&&std::filesystem::path(locator).is_relative()){
        auto path=std::filesystem::path(sqlite3_db_filename(source,"main"));
        if(isPackedDatabasePath(path))path=packedContainerPath(path);
        auto destination=std::filesystem::path(sqlite3_db_filename(db,"main"));
        if(isPackedDatabasePath(destination))destination=packedContainerPath(destination);
        const auto absolute=std::filesystem::absolute(path.parent_path()/locator).lexically_normal();
        const auto relative=absolute.lexically_relative(std::filesystem::absolute(destination).parent_path());
        locator=(relative.empty()?absolute:relative).generic_string();
      }
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
