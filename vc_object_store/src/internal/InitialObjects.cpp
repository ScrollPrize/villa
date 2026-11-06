#include "internal/InitialObjects.hpp"
#include "internal/RecordSql.hpp"
#include "geometry/InitialGeometry.hpp"
#include "geometry/GeometryInternal.hpp"
#include "internal/ImmutableImport.hpp"
#include "imports/SourceIdentityIndex.hpp"

namespace vc::objects::detail {
std::vector<std::pair<std::string,int64_t>> insertInitialObjects(
    ObjectStore& store,WriteTxn& txn,std::span<const InitialObjectRecord> input)
{
    auto* db=ObjectStoreImplAccess::conn(txn);
    if(historyEnabled(db))throw std::invalid_argument("initial objects require history-off storage");
    if(input.empty())return {};
    GeometryPhase phase("native.initial_objects");
    int64_t lower,upper;std::optional<int64_t> next;
    {Statement range(db,"SELECT lower_bound,upper_bound,next_id FROM key_allocators WHERE name='objects'");
        if(!range.row())throw std::runtime_error("missing object ID allocation range");
        lower=range.i64(0);upper=range.i64(1);if(!range.isNull(2))next=range.i64(2);}
    std::vector<int64_t> ids;ids.reserve(input.size());
    for(const auto& value:input){
        if(!value.id&&!next)throw std::runtime_error("object store: reserved objects ID range exhausted");
        const auto id=value.id?*value.id:*next;
        if(id<lower||id>upper)throw std::invalid_argument("object store: explicit ID outside reservation");
        ids.push_back(id);
        if(next&&id>=*next)next=id==upper?std::optional<int64_t>{}:std::optional<int64_t>{id+1};
    }
    execute(db,"UPDATE key_allocators SET next_id=? WHERE name='objects'",next);
    const auto generation=writeGeneration(db);const auto time=nowIsoUtc(),author=store.author(),host=store.hostname();
    std::vector<std::pair<std::string,int64_t>> result;result.reserve(input.size());
    for(size_t i=0;i<input.size();++i){
        const auto& value=input[i];auto rec=prepareObjectRecord(value.record);
        normalizeObjectCoordinates(rec,true);
        std::optional<PreparedInitialGeometry> geometry;
        if(value.geometry){geometry=prepareInitialGeometry(db,rec,*value.geometry);rec=geometry->record;}
        else if(rec.storageKind=="sqlite")
            throw std::invalid_argument("initial object managed storage requires supplied geometry");
        const auto id=ids[i];
        insertObjectRow(db,rec,id,generation,author,host,time);
        insertObjectDetail(db,id,rec);
        insertTags(db,"object_tags","object_id",id,rec.tags);
        // Active/default state is stored directly in the object row.
        if(rec.retired||!rec.lifecycleEpoch.empty())writeLifecycle(db,id,rec);
        if(geometry)writeInitialGeometry(store,txn,*geometry,*value.geometry,id);
        else setSpatialBounds(db,id,recordBounds(rec));
        if(immutableInitial(db)&&rec.kind=="point_collection")
            markSourceIdentityDirty(db,rec.uuid);
        result.emplace_back(rec.uuid,id);
    }
    return result;
}
}
