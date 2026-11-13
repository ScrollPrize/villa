#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/InitialImport.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/objects/StoreConversion.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "internal/Database.hpp"
#include "internal/CollectionReads.hpp"
#include "internal/RecordSql.hpp"
#include "internal/RecordResolver.hpp"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <string_view>
#include <limits>
using namespace vc::objects;
using namespace vc::objects::detail;
ObjectRecord fiber(){ObjectRecord r;r.kind="fiber";r.fiber.emplace();return r;}

void emptyRoutedSelectionValidation()
{
    TemporaryDirectory dir;InitialImportOptions options;options.store.author="empty-query-validation";
    options.shards.resize(2);options.shards[0].objectIds={1,100};options.shards[1].objectIds={101,200};
    const auto initial=importInitialCollection(dir.path/"empty",options,[](size_t,InitialShardImport&){});
    auto store=ObjectStore::open(initial.path,OpenOptions{.readOnly=true});auto read=store.read();
    ObjectSelection valid;valid.kind="fiber";valid.uuids={util::newUuid()};
    require(store.selectObjects(read,valid).empty(),"a valid explicit selection on sealed empty partitions has no matches");
    const auto rejects=[&](const ObjectSelection& selection,const char* message){
        bool rejected=false;
        try{(void)store.selectObjects(read,selection);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,message);
    };
    auto invalid=valid;invalid.kind="surface";invalid.family="H";
    rejects(invalid,"typed catalog selection is validated even when route metadata eliminates every physical query");
    invalid=valid;invalid.afterUuid=util::newUuid();invalid.offset=1;
    rejects(invalid,"cursor and offset cannot be combined on an empty routed selection");
    invalid=valid;invalid.afterUuid="not-a-uuid";
    bool malformedCursorRejected=false;
    try{(void)store.selectObjects(read,invalid);}catch(const std::runtime_error& error){
        malformedCursorRejected=std::string_view(error.what()).find("not a UUID")!=std::string_view::npos;
    }
    require(malformedCursorRejected,"cursor UUID syntax retains its runtime-error contract independently of matching physical rows");
    invalid=valid;invalid.minLength=std::numeric_limits<double>::quiet_NaN();
    rejects(invalid,"nonfinite typed query limits are rejected before empty-result short circuits");
    invalid=valid;invalid.afterLength=1.;
    rejects(invalid,"length cursors require the same ordering policy on empty routed selections");
}

void routedExplicitSelection()
{
    TemporaryDirectory dir;InitialImportOptions options;options.store.author="routed-selection";
    constexpr size_t shards=8,count=327;options.shards.resize(shards);
    for(size_t i=0;i<shards;++i)options.shards[i].objectIds={int64_t(i*1000+1),int64_t((i+1)*1000)};
    std::vector<std::string> ids;
    for(size_t i=0;i<count;++i)ids.push_back(util::nameUuid("routed-selection-"+std::to_string(i)));
    std::sort(ids.begin(),ids.end());
    const auto imported=importInitialCollection(dir.path/"source",options,[&](size_t shard,InitialShardImport& writer){
        std::vector<InitialObjectRecord> rows;
        for(const auto& id:ids)if(uuidShardIndex(id,shards)==shard){InitialObjectRecord row;row.record.uuid=id;row.record.kind="note";rows.push_back(row);}
        writer.appendObjects(rows);
    });
    auto store=ObjectStore::open(imported.path,OpenOptions{.readOnly=true});auto read=store.read();
    struct Work {
        size_t idProbes=0,uuidLiterals=0;std::vector<sqlite3*> sources;
        explicit Work(sqlite3* root):sources(querySources(root)){
            for(auto* source:sources)sqlite3_trace_v2(source,SQLITE_TRACE_STMT,[](unsigned,void* state,void* query,void*){
                auto& work=*static_cast<Work*>(state);const std::string_view sql=sqlite3_sql(static_cast<sqlite3_stmt*>(query));
                if(sql.starts_with("SELECT id FROM objects WHERE uuid="))++work.idProbes;
                const auto clause=sql.find(" AND o.uuid IN (");
                if(clause!=std::string_view::npos){
                    const auto end=sql.find(')',clause);
                    for(auto at=sql.find("X'",clause);at!=std::string_view::npos&&at<end;at=sql.find("X'",at+2))++work.uuidLiterals;
                }
                return 0;
            },this);
        }
        ~Work(){for(auto* source:sources)sqlite3_trace_v2(source,0,nullptr,nullptr);}
    } work(ObjectStoreImplAccess::conn(read));
    ObjectSelection selection;selection.kind="note";
    for(size_t i=0;i<ids.size();i+=2)selection.uuids.push_back(ids[i]);
    selection.uuids.push_back(util::nameUuid("absent-routed-selection"));
    const auto rows=store.selectObjects(read,selection,std::vector<std::string>{});
    require(rows.size()==(ids.size()+1)/2,"explicit routed UUID selection retains all existing requested records");
    require(work.idProbes==0,"explicit UUID selection never reads back physical row IDs");
    require(work.uuidLiterals==selection.uuids.size(),"requested UUID predicates are routed once, not repeated on every physical shard");
}

void compactReferenceSelectionPlan()
{
    TemporaryDirectory dir;OpenOptions options;options.author="reference-plan";options.history=HistoryMode::Off;options.spatialIndex=false;
    auto store=ObjectStore::create(dir.path/"local",options);
    std::vector<std::string> ids{util::newUuid(),util::newUuid(),util::newUuid()};
    {auto write=store.write();for(const auto& id:ids){ObjectRecord record;record.uuid=id;record.kind="note";store.upsertObject(write,record);}write.commit();}
    {auto write=store.writeNumbered(VersionSelection{{ids[1],1},{ids[2],1}});
        store.retireObject(write,ids[1]);store.deleteObject(write,ids[2]);write.commit();}
    auto read=store.read();auto* db=ObjectStoreImplAccess::conn(read);
    for(const bool retired:{false,true}){
        ObjectSelection selection;selection.kind="note";selection.includeRetired=retired;
        auto query=objectSelectionQuery(db,selection,"o.uuid,o.version,o.version_generation");
        Statement plan(db,std::string("EXPLAIN QUERY PLAN ")+sqlite3_sql(query->value));
        while(plan.row())require(plan.text(3).find("CORRELATED")==std::string::npos,
            "ordinary local-row lifecycle selection does not repeat a correlated self-lookup");
        const auto rows=store.selectObjects(read,selection,std::vector<std::string>{});
        require(rows.size()==(retired?2:1)&&store.objectCount(read,"note",retired)==int64_t(rows.size()),
            "compact identity/count selections preserve active, retired and deleted lifecycle semantics");
        std::vector<ObjectVersion> references;for(const auto& row:rows)references.push_back(row.reference());
        require(store.selectedReferenceDigest(read,selection,"test v1")==objectReferencesDigest(references,"test v1"),
            "compact identity scanning shares exact generation and reference encoding with full selected records");
    }
}

void interleavedPartitionScan()
{
    TemporaryDirectory dir;OpenOptions options;options.author="heap-scan";options.history=HistoryMode::Off;options.spatialIndex=false;
    options.creationStoreUuid=util::newUuid();options.creationGeneration=util::newUuid();
    constexpr size_t partitions=12,count=203;
    std::vector<std::string> ids;
    for(size_t i=0;i<count;++i)ids.push_back(util::nameUuid("interleaved-heap-"+std::to_string(i)));
    std::sort(ids.begin(),ids.end());
    std::vector<std::filesystem::path> paths;
    for(size_t shard=0;shard<partitions;++shard){
        auto opt=options;opt.objectIdRange={int64_t(1+shard*1000),int64_t((shard+1)*1000)};
        const auto path=dir.path/("part-"+std::to_string(shard));paths.push_back(path);
        auto store=ObjectStore::create(path,opt);auto txn=store.write();
        for(size_t i=shard;i<count;i+=partitions){ObjectRecord row;row.uuid=ids[i];row.kind="note";store.upsertObject(txn,row);}
        store.seal(txn);txn.commit();
    }
    const auto root=dir.path/"root";
    {auto store=ObjectCollection::publishShards(root,paths,options);}
    auto store=ObjectStore::open(root,OpenOptions{.readOnly=true});auto txn=store.read();
    ObjectSelection selection;selection.kind="note";
    const auto all=store.selectObjects(txn,selection,std::vector<std::string>{});
    std::vector<std::string> found;for(const auto& row:all)found.push_back(row.uuid);
    require(found==ids,"heap scan interleaves indexed partitions in global UUID order, including uneven exhausted cursors");
    selection.afterUuid=ids[95];selection.limit=19;
    const auto page=store.selectObjects(txn,selection,std::vector<std::string>{});
    require(page.size()==19&&page.front().uuid==ids[96]&&page.back().uuid==ids[114],"heap scan retains cursor pagination across partitions");
}

void boundedVisibleScan()
{
    TemporaryDirectory dir;OpenOptions options;options.author="scan-test";options.history=HistoryMode::Off;options.spatialIndex=false;
    constexpr size_t count=5000;
    std::vector<std::string> ids;for(size_t i=0;i<count;++i)ids.push_back(util::nameUuid("bounded-scan-"+std::to_string(i)));
    std::sort(ids.begin(),ids.end());
    const auto base=dir.path/"base",early=dir.path/"early",late=dir.path/"late",root=dir.path/"root";
    std::string baseOwner,lateOwner,lateGeneration;int64_t lateCommit=0;
    {auto store=ObjectStore::create(base,options);auto txn=store.write();baseOwner=store.uuid(txn);
        for(const auto& id:ids){ObjectRecord record;record.uuid=id;record.kind="note";record.tags={"eligible"};store.upsertObject(txn,record);}
        store.seal(txn);txn.commit();}
    auto lateOptions=options;lateOptions.objectIdRange={101,200};
    {auto store=ObjectCollection::create(late,{base},lateOptions);auto txn=store.write();lateOwner=store.uuid(txn);
        auto changed=store.object(txn,{{},ids[0]})->record;changed.name="late override";changed.tags.clear();
        store.replaceObject(txn,{{},ids[0]},changed);store.deleteObject(txn,{{},ids[1]});store.seal(txn);txn.commit();
        auto read=store.read();const auto p=store.provisioning(read);lateGeneration=p.generation;lateCommit=p.commitSequence;}
    // Both partitions form one logical layer. An earlier partition's base
    // must not win over a later partition's override or deletion.
    auto earlyOptions=options;earlyOptions.objectIdRange={1,100};
    earlyOptions.creationStoreUuid=lateOwner;earlyOptions.creationGeneration=lateGeneration;
    {auto store=ObjectCollection::create(early,{base},earlyOptions);auto txn=store.write();store.seal(txn);txn.commit();
        auto read=store.read();require(store.provisioning(read).commitSequence==lateCommit,"partition fixtures share the publication pin");}
    {auto store=ObjectCollection::publishShards(root,{early,late},options);}
    auto store=ObjectCollection::open(root,OpenOptions{.readOnly=true});auto txn=store.read();
    struct RowCount {
        std::vector<sqlite3*> sources;size_t count=0,stateProbes=0;
        RowCount(const ObjectCollection& store,const CollectionTxn& txn) {
            for(const auto& source:CollectionScanAccess::partitions(store,txn)) {
                sources.push_back(source.db);
                sqlite3_trace_v2(source.db,SQLITE_TRACE_ROW|SQLITE_TRACE_STMT,[](unsigned event,void* context,void* statement,void*){
                    auto* query=static_cast<sqlite3_stmt*>(statement);
                    if(event==SQLITE_TRACE_STMT){
                        if(std::string_view(sqlite3_sql(query)).starts_with("SELECT state FROM objects WHERE uuid="))
                            ++static_cast<RowCount*>(context)->stateProbes;
                    }else if(sqlite3_column_count(query)>=23&&std::string_view(sqlite3_column_name(query,0))=="id"&&
                       std::string_view(sqlite3_column_name(query,1))=="uuid")++static_cast<RowCount*>(context)->count;
                    return 0;
                },this);
            }
        }
        ~RowCount(){for(auto* source:sources)sqlite3_trace_v2(source,0,nullptr,nullptr);}
    } work(store,txn);
    size_t seen=0;
    store.scanObjects(txn,[&](const CollectionObject& row){
        require(row.record.uuid==ids[0]&&row.record.name=="late override"&&row.key.storeUuid==lateOwner,
                "early scan sees the late-sibling override and its actual owner");++seen;return false;
    },"note");
    require(seen==1&&work.count<=260,"early-stop scan decodes only one bounded batch, not the whole store");
    work.count=0;work.stateProbes=0;std::vector<std::string> found;
    store.scanObjects(txn,[&](const CollectionObject& row){
        require(row.record.versionStoreUuid==row.key.storeUuid,"streamed record retains its selected owner reference");
        require(row.key.storeUuid==(row.record.uuid==ids[0]?lateOwner:baseOwner),"source ownership does not become root ownership");
        found.push_back(row.record.uuid);return true;
    },"note");
    auto expected=ids;expected.erase(expected.begin()+1);
    require(found==expected,"one-pass scan is UUID ordered and suppresses tombstones and shadowed duplicates");
    require(work.count<=count+4,"one-pass scan neither rescans pages nor rereads emitted full records");
    require(work.stateProbes<=count,"sealed empty layers and already selected rows need no repeated existence probes");
    txn.commit();
    auto ordinary=ObjectStore::open(root,OpenOptions{.readOnly=true});auto read=ordinary.read();
    ObjectSelection selection;selection.kind="note";selection.tag="eligible";selection.limit=1;
    const auto selected=ordinary.selectObjects(read,selection,std::vector<std::string>{"name"});
    require(selected.size()==1&&selected[0].uuid==ids[2],"filtered upper overrides and tombstones still suppress matching lower records");
    selection.uuids={ids[0],ids[1],ids[2]};
    const auto exact=ordinary.selectObjects(read,selection,std::vector<std::string>{"name"});
    require(exact.size()==1&&exact[0].uuid==ids[2],"explicit UUID routing preserves filtered overrides, inherited matches and tombstones");
    selection.uuids.clear();
    selection.tag.reset();selection.afterUuid=ids[4095];selection.limit=3;
    const auto page=ordinary.selectObjects(read,selection,std::vector<std::string>{});
    require(page.size()==3&&page[0].uuid==ids[4096]&&page[2].uuid==ids[4098],"cursor pagination seeks directly past its UUID boundary");
    read.commit();
    auto mutableStore=ObjectCollection::create(dir.path/"mutable",{root},options);
    {auto before=mutableStore.read();require(!mutableStore.object(before,{{},util::nameUuid("new-local")}),"new UUID begins absent");}
    {auto write=mutableStore.write();ObjectRecord record;record.uuid=util::nameUuid("new-local");record.kind="note";
        mutableStore.upsertObject(write,record);write.commit();}
    {auto after=mutableStore.read();require(bool(mutableStore.object(after,{{},util::nameUuid("new-local")})),
        "a mutable initially empty layer still exposes later insertions");}
}

void metadataOverrides()
{
    TemporaryDirectory dir;
    for (const auto baseHistory : {HistoryMode::Off, HistoryMode::On})
    for (const auto overlayHistory : {HistoryMode::Off, HistoryMode::On}) {
        const auto root = dir.path / (std::to_string(int(baseHistory)) + "-" + std::to_string(int(overlayHistory)));
        std::filesystem::create_directories(root / "source");
        std::filesystem::create_directories(root / "derived");
        const auto basePath = root / "source/base.vcobj", overlayPath = root / "derived/overlay.vcobj";
        OpenOptions options; options.history = baseHistory; options.author = "test";
        ObjectKey managed;
        std::vector<ObjectKey> external;
        std::vector<std::pair<std::string, std::string>> locations{
            {"file", "data/fiber.obj"}, {"file", (root / "absolute.obj").string()},
            {"dataset", "s3://bucket/volume"}, {"url", "https://example.org/fiber"}};
        std::string reference, checksum, descriptor;
        ObjectVersion baseVersion, metadataVersion;
        std::vector<uint8_t> bytes;
        {
            auto base = ObjectStore::create(basePath, options); auto w = base.write();
            managed = {base.uuid(w), base.upsertObject(w, fiber())};
            vc::geometry::Fiber shape; shape.linePoints = {{0, 1, 2}, {3, 4, 5}};
            reference = base.putGeometry(w, managed.objectUuid, vc::geometry::encode(shape));
            for (const auto& [kind, location] : locations) {
                auto record = fiber(); record.storageKind = kind; record.storageRef = location; record.dataChecksum = "external-checksum";
                external.push_back({base.uuid(w), base.upsertObject(w, record)});
            }
            w.commit();
            { auto r = base.read(); bytes = base.readGeometryArray(r, reference, "line_points"); checksum = base.currentGeometry(r, managed.objectUuid)->contentHash;
                Statement stored(ObjectStoreImplAccess::conn(r),"SELECT geometry_manifest FROM objects WHERE uuid=?",parseUuid(managed.objectUuid));
                stored.row(); descriptor=stored.text(0);
                baseVersion=base.currentObjectRevision(r,managed.objectUuid)->reference(); }
            auto seal = base.write(); base.seal(seal); seal.commit();
        }
        options.history = overlayHistory;
        {
            auto overlay = ObjectCollection::create(overlayPath, {basePath}, options);
            { auto w = overlay.write(); overlay.setMeta(w, managed, "discard", "true"); }
            { auto w = overlay.write();
                overlay.setMeta(w, managed, "label", "1");
                auto replacement = overlay.object(w, external.front())->record;
                replacement.meta = "{\"label\":1}";
                overlay.replaceObject(w, external.front(), replacement);
                for (size_t i = 1; i < external.size(); ++i) overlay.setMeta(w, external[i], "label", "1");
                w.commit(); }
            { auto w = overlay.write(); overlay.setMeta(w, managed, "label", "2"); w.commit(); }
            { auto w = overlay.write();
                auto replacement = overlay.object(w, external.front())->record;
                replacement.name = "renamed"; overlay.replaceObject(w, external.front(), replacement); w.commit(); }
            { auto r = overlay.read();
                const auto object=overlay.object(r,managed)->record;
                metadataVersion={overlay.uuid(r),object.versionGeneration,managed.objectUuid,*object.version};
                require(overlay.readGeometryArrayAtVersion(r,baseVersion,"line_points")==bytes&&
                    overlay.readGeometryArrayAtVersion(r,metadataVersion,"line_points")==bytes,
                    "base and metadata-only override versions use the same geometry resolution");
                require(overlay.currentGeometry(r, managed)->uuid == reference, "metadata edits preserve exact inherited geometry identity");
                require(overlay.currentGeometry(r, managed)->contentHash == checksum, "metadata edits preserve geometry checksum");
                require(overlay.readGeometryArray(r, managed, "line_points") == bytes, "borrowed geometry stays readable");
                require(overlay.object(r, managed)->record.meta.find("discard") == std::string::npos, "rollback preserves inherited metadata");
                for (size_t i = 0; i < external.size(); ++i) {
                    const auto record = overlay.object(r, external[i])->record;
                    require(record.storageKind == locations[i].first && record.dataChecksum == "external-checksum", "external kind and checksum retained");
                    const auto expected = i == 0 ? std::string("../source/data/fiber.obj") : locations[i].second;
                    require(record.storageRef == expected, "relative file reference rebased; opaque and absolute references unchanged");
                }
            }
        }
        {
            auto raw = ObjectStore::open(overlayPath, OpenOptions{.physicalPartitionOnly = true, .readOnly = true}); auto r = raw.read();
            auto* sql = ObjectStoreImplAccess::conn(r);
            Statement payloads(sql, "SELECT count(*) FROM geometry_chunks"); payloads.row();
            require(payloads.i64(0) == 0, "metadata override copies no geometry bytes");
            Statement descriptors(sql, "SELECT geometry_manifest FROM objects WHERE uuid=?",parseUuid(managed.objectUuid));
            require(descriptors.row()&&descriptors.text(0)==descriptor, "metadata override copies the complete descriptor verbatim");
            if (overlayHistory == HistoryMode::On) {
                Statement archives(sql, "SELECT count(*) FROM geometry_revisions"); archives.row();
                require(archives.i64(0) == 0, "metadata history does not create geometry revisions");
                const auto history = raw.objectHistory(r, managed.objectUuid);
                require(history.size() == 2, "each actual metadata edit is historized");
                require(history.front().record.meta.find("\"label\":1") != std::string::npos, "old metadata is unchanged");
                for (const auto& version : history) require(version.geometryRevision == reference, "history preserves borrowed reference across history modes");
                Statement pin(sql,"SELECT geometry_uuid FROM object_revisions WHERE object_uuid=?",parseUuid(managed.objectUuid));
                while(pin.row())require(!pin.isNull(0),"borrowed references are stored directly, never replaced with null");
                Statement keys(sql,"PRAGMA foreign_key_list(object_revisions)");
                while(keys.row())require(keys.text(3)!="geometry_uuid","geometry references must not require local archive rows");
            }
        }
        // Explicit history policy conversion may change geometry tokens, but
        // must preserve the geometry described by the copied metadata.
        for (const auto mode : {HistoryMode::On, HistoryMode::Off, overlayHistory}) {
            changeHistoryMode(overlayPath, mode);
            auto converted = ObjectCollection::open(overlayPath, options); auto r = converted.read();
            require(converted.readGeometryArray(r, managed, "line_points") == bytes,
                "history policy conversion preserves borrowed geometry");
            reference=converted.currentGeometry(r,managed)->uuid;
        }
        {
            auto overlay = ObjectCollection::open(overlayPath, options);
            { auto r = overlay.read(); require(overlay.readGeometryArray(r, managed, "line_points") == bytes, "reopened overlay resolves borrowed geometry"); }
            { auto w = overlay.write(); overlay.seal(w); w.commit(); }
            auto child = ObjectCollection::create(root / "child.vcobj", {overlayPath}, options);
            { auto w = child.write(); child.setMeta(w, managed, "label", "3"); w.commit(); }
            ObjectVersion beforeEdit;
            { auto r = child.read(); require(child.currentGeometry(r, managed)->uuid == reference && child.readGeometryArray(r, managed, "line_points") == bytes,
                "nested metadata override retains original reference");
                const auto record=child.object(r,managed)->record;
                beforeEdit={child.uuid(r),record.versionGeneration,managed.objectUuid,*record.version}; }
            GeometryArrayEdit edit{"line_points", 0, std::vector<uint8_t>(sizeof(double))};
            double value = 12; std::memcpy(edit.bytes.data(), &value, sizeof(value));
            { auto w = child.write(); child.editGeometry(w, managed, {edit}, reference); w.commit(); }
            auto edited = bytes; std::memcpy(edited.data(), &value, sizeof(value));
            { auto r = child.read(); require(child.readGeometryArray(r, managed, "line_points") == edited, "explicit geometry edit still works after metadata overrides"); }
            if(overlayHistory==HistoryMode::On) {
                auto r=child.read();
                require(child.readGeometryArrayAtVersion(r,beforeEdit,"line_points")==bytes,
                    "historical geometry follows unchanged reference after local edit");
                require(child.currentGeometry(r,managed)->parentUuid==reference,
                    "sparse edit keeps actual referenced parent, without a synthetic local base");
                auto local=ObjectStore::open(root/"child.vcobj",OpenOptions{.physicalPartitionOnly=true,.readOnly=true});
                auto read=local.read();Statement count(ObjectStoreImplAccess::conn(read),"SELECT count(*) FROM geometry_revisions");
                require(count.row()&&count.i64(0)==1,"one real edit creates exactly one geometry revision");
            }
            { auto r = overlay.read(); require(overlay.readGeometryArray(r, managed, "line_points") == bytes, "child geometry edit leaves base untouched"); }
        }
    }
}

void metadataOverridesOfDeltaGeometry()
{
    TemporaryDirectory dir;
    const auto basePath=dir.path/"delta-base.vcobj";
    OpenOptions options; options.author="test";
    ObjectKey key;
    std::string token,descriptor;
    std::vector<uint8_t> expected;
    GeometryArrayEdit change{"line_points",0,std::vector<uint8_t>(sizeof(double))};
    double value=15; std::memcpy(change.bytes.data(),&value,sizeof(value));
    {
        auto base=ObjectStore::create(basePath,options);
        { auto w=base.write(); key={base.uuid(w),base.upsertObject(w,fiber())};
            vc::geometry::Fiber shape; for(int i=0;i<100000;++i)shape.linePoints.push_back({double(i),2,3});
            token=base.putGeometry(w,key.objectUuid,vc::geometry::encode(shape)); w.commit(); }
        { auto w=base.write(); token=base.editGeometry(w,key.objectUuid,{change},token); w.commit(); }
        { auto r=base.read(); expected=base.readGeometryArray(r,token,"line_points");
            Statement d(ObjectStoreImplAccess::conn(r),"SELECT geometry_manifest FROM objects WHERE uuid=?",parseUuid(key.objectUuid));d.row();descriptor=d.text(0);
            Statement delta(ObjectStoreImplAccess::conn(r),"SELECT count(*) FROM geometry_deltas");delta.row();require(delta.i64(0)>0,"fixture must contain an archived geometry delta"); }
        auto w=base.write();base.seal(w);w.commit();
    }
    for(const auto mode:{HistoryMode::Off,HistoryMode::On}){
        options.history=mode;
        const auto path=dir.path/("delta-overlay-"+std::to_string(int(mode)));
        auto overlay=ObjectCollection::create(path,{basePath},options);
        { auto w=overlay.write();overlay.setMeta(w,key,"classification","\"H\"");w.commit(); }
        { auto r=overlay.read();require(overlay.currentGeometry(r,key)->uuid==token,"metadata edit preserves delta reference");
            require(overlay.readGeometryArray(r,key,"line_points")==expected,"copied descriptor retains readable delta-backed payloads"); }
        { auto local=ObjectStore::open(path,OpenOptions{.physicalPartitionOnly=true,.readOnly=true});auto r=local.read();auto* db=ObjectStoreImplAccess::conn(r);
            Statement d(db,"SELECT geometry_manifest FROM objects WHERE uuid=?",parseUuid(key.objectUuid));d.row();require(d.text(0)==descriptor,"delta-backed descriptor is copied verbatim");
            Statement chunks(db,"SELECT count(*) FROM geometry_chunks");chunks.row();require(chunks.i64(0)==0,"metadata edit does not materialize delta payloads");
            if(mode==HistoryMode::On){Statement revisions(db,"SELECT count(*) FROM geometry_revisions");revisions.row();require(revisions.i64(0)==0,"metadata edit does not create a geometry snapshot");} }
        change.elementOffset=240000;value=30;std::memcpy(change.bytes.data(),&value,sizeof(value));
        { auto w=overlay.write();overlay.editGeometry(w,key,{change},token);w.commit(); }
        auto edited=expected;std::memcpy(edited.data()+change.elementOffset*sizeof(double),&value,sizeof(value));
        { auto r=overlay.read();require(overlay.readGeometryArray(r,key,"line_points")==edited,"explicit delta-backed geometry edit still works"); }
        if(mode==HistoryMode::On) {
            std::string first;
            {auto r=overlay.read();first=overlay.currentGeometry(r,key)->uuid;require(overlay.currentGeometry(r,key)->parentUuid==token,"edit references source delta directly");}
            change.elementOffset=150000;value=47;std::memcpy(change.bytes.data(),&value,sizeof(value));
            {auto w=overlay.write();overlay.editGeometry(w,key,{change},first);w.commit();}
            std::memcpy(edited.data()+change.elementOffset*sizeof(double),&value,sizeof(value));
            {auto r=overlay.read();require(overlay.readGeometryArray(r,key,"line_points")==edited,"multiple local deltas retain unchanged source-delta chunks");}
            auto local=ObjectStore::open(path,OpenOptions{.physicalPartitionOnly=true,.readOnly=true});auto r=local.read();
            Statement count(ObjectStoreImplAccess::conn(r),"SELECT count(*) FROM geometry_chunks");
            require(count.row()&&count.i64(0)==0,"sparse history-on edits do not copy inherited payloads");
        }
    }
}
int main()try{
    emptyRoutedSelectionValidation();
    routedExplicitSelection();
    compactReferenceSelectionPlan();
    interleavedPartitionScan();
    boundedVisibleScan();
    metadataOverrides();
    metadataOverridesOfDeltaGeometry();
    TemporaryDirectory dir;
    auto options=OpenOptions{.history=HistoryMode::Off,.spatialIndex=false,.author="test"};
    const auto basePath=dir.path/"base.vcobj",overlayPath=dir.path/"overlay.vcobj";
    ObjectKey a,b;std::vector<uint8_t> geometry;
    {
        auto base=ObjectStore::create(basePath,options);auto w=base.write();
        a={base.uuid(w),base.upsertObject(w,fiber())};b={base.uuid(w),base.upsertObject(w,fiber())};
        vc::geometry::Fiber shape;for(int i=0;i<60000;++i)shape.linePoints.push_back({double(i),2,3});
        auto rev=base.putGeometry(w,a.objectUuid,vc::geometry::encode(shape));
        w.commit();auto read=base.read();geometry=base.readGeometryArray(read,rev,"line_points");read.commit();
        mustFail([&]{ObjectCollection::create(overlayPath,{basePath},options);});
        auto seal=base.write();base.seal(seal);seal.commit();
    }
    auto overlay=ObjectCollection::create(overlayPath,{basePath},options);
    {auto native=ObjectStore::open(overlayPath,OpenOptions{.readOnly=true});auto r=native.read();
     require(native.objectCount(r,"fiber")==2,"ordinary store API opens logical collections");}
    {
        auto r=overlay.read();require(overlay.readGeometryArray(r,a,"line_points")==geometry,"inherited geometry exact");
        require(overlay.provisioning(r).commitSequence>overlay.members()[1].commitSequence,"derived commit sequence above parent");
        require(overlay.objectById(r,a.storeUuid,1)->key==a,"numeric base routing");
    }
    ObjectKey newFiber,edge;
    {
        auto w=overlay.write();overlay.setMeta(w,a,"classification","\"H\"");newFiber=overlay.upsertObject(w,fiber());
        EdgeRecord e;e.kind="perpendicular";e.aAnchor.wholeGeometry=true;e.bAnchor.wholeGeometry=true;
        edge=overlay.addEdge(w,a,newFiber,e);w.commit();
        auto r=overlay.read();require(overlay.edgesOf(r,a).size()==1,"cross store adjacency");
        const auto bulk=overlay.edgesByKind(r,"perpendicular");
        require(bulk.size()==1&&bulk[0].key==edge&&bulk[0].record.version==overlay.edge(r,edge)->record.version,
            "bulk edge query agrees with resolved single lookup");
        require(bulk[0].record.versionGeneration==overlay.edge(r,edge)->record.versionGeneration,
            "bulk edge query retains implicit initial generation");
        require(overlay.edgesOf(r,newFiber)[0].key==edge,"reverse adjacency");
        require(overlay.object(r,a)->record.meta.find("H")!=std::string::npos,"metadata override");
        require(overlay.readGeometryArray(r,a,"line_points")==geometry,"metadata override borrows chunks");
        size_t count=0;overlay.scanObjects(r,[&](auto&){++count;return true;});require(count==3,"references do not duplicate objects");
        count=0;overlay.scanObjects(r,[&](const auto& row){
            const auto resolved=overlay.object(r,row.key);
            require(resolved&&row.record.meta==resolved->record.meta&&row.record.version==resolved->record.version&&
                row.record.versionGeneration==resolved->record.versionGeneration&&row.record.lifecycleEpoch==resolved->record.lifecycleEpoch,
                "bulk object values equal resolved lookup");
            ++count;return true;},"fiber");require(count==3,"kind-filtered references do not duplicate objects");
        count=0;overlay.scanObjects(r,[&](auto&){++count;return true;},"volume_transform");require(count==0,"unrelated fiber records excluded by kind");
    }
    {
        auto base=ObjectStore::open(basePath,OpenOptions{.readOnly=true});auto r=base.read();
        require(base.objectByUuid(r,a.objectUuid)->meta.find("classification")==std::string::npos,"base not modified");
        auto w=overlay.write();overlay.setMeta(w,a,"rollback","true");
    }
    {
        auto r=overlay.read();require(overlay.object(r,a)->record.meta.find("rollback")==std::string::npos,"overlay rollback");
        auto w=overlay.write();mustFail([&]{overlay.deleteObject(w,a);});overlay.seal(w);w.commit();
    }
    auto child=ObjectCollection::create(dir.path/"child.vcobj",{overlayPath},options);
    {
        auto w=child.write();child.setMeta(w,a,"classification","\"V\"");child.removeEdge(w,edge);child.deleteObject(w,a);w.commit();
        auto r=child.read();require(!child.object(r,a),"inherited deletion tombstone");require(child.edgesOf(r,newFiber).empty(),"inherited edge tombstone");
        require(child.edgesByKind(r,"perpendicular").empty(),"bulk edge scan honors inherited deletion tombstones");
        size_t count=0;child.scanObjects(r,[&](auto&){++count;return true;});require(count==2,"deleted inherited geometry not enumerated");
        count=0;child.scanObjects(r,[&](auto&){++count;return true;},"fiber");require(count==2,"kind-filtered scan honors inherited deletion tombstones");
    }
    {
        auto r=overlay.read();require(overlay.object(r,a).has_value()&&overlay.edgesOf(r,a).size()==1,"child changes do not change sealed parent");
    }
    {
        auto raw=ObjectStore::open(dir.path/"child.vcobj",OpenOptions{.physicalPartitionOnly=true,.readOnly=true});
        auto r=raw.read();auto* sql=ObjectStoreImplAccess::conn(r);
        Statement object(sql,"SELECT state FROM objects WHERE uuid=?",parseUuid(a.objectUuid));
        require(object.row()&&object.i64(0)==2,"inherited object deletion must use its original UUID row");
        Statement relation(sql,"SELECT state FROM edges WHERE uuid=?",parseUuid(edge.objectUuid));
        require(relation.row()&&relation.i64(0)==2,"inherited edge deletion must use its original UUID row");
        Statement redundant(sql,"SELECT count(*) FROM sqlite_schema WHERE name IN "
            "('collection_refs','collection_edge_tombstones','external_object_keys','endpoint_keys')");
        require(redundant.row()&&redundant.i64(0)==0,"overlay must not add alias or tombstone tables");
    }
    for(const auto history:{HistoryMode::Off,HistoryMode::On}){
        auto opt=options;opt.history=history;auto file=dir.path/(history==HistoryMode::Off?"geometry-off.vcobj":"geometry-on.vcobj");
        auto c=ObjectCollection::create(file,{basePath},opt);
        std::string initial;{auto r=c.read();initial=c.currentGeometry(r,a)->uuid;}
        GeometryArrayEdit change{"line_points",0,std::vector<uint8_t>(8)};double value=123;std::memcpy(change.bytes.data(),&value,8);
        {auto w=c.write();c.editGeometry(w,a,{change},initial);w.commit();}
        auto expected=geometry;std::memcpy(expected.data(),&value,8);
        {auto r=c.read();require(c.readGeometryArray(r,a,"line_points")==expected,"store_reference sparse geometry edit exact");}
        {auto raw=ObjectStore::open(file,OpenOptions{.physicalPartitionOnly=true,.readOnly=true});auto r=raw.read();
            Statement bytes(ObjectStoreImplAccess::conn(r),"SELECT count(*) FROM geometry_chunks WHERE data IS NULL");bytes.row();require(bytes.i64(0)==0,"referenced chunks create no placeholder rows");
            mustFail([&]{auto records=raw.objectsByKind(r,"fiber");auto rev=raw.currentGeometry(r,records[0].uuid);raw.readGeometryArray(r,rev->uuid,"line_points");});}
        {auto w=c.write();c.seal(w);w.commit();}
        auto deeper=ObjectCollection::create(dir.path/(history==HistoryMode::Off?"deeper-off.vcobj":"deeper-on.vcobj"),{file},opt);
        {auto r=deeper.read();require(deeper.readGeometryArray(r,a,"line_points")==expected,"referenced geometry resolves through deeper overlay");}
    }
    {
        const auto file=dir.path/"store-reference-policy.vcobj";auto expected=geometry;double value=111;std::memcpy(expected.data(),&value,8);
        {auto c=ObjectCollection::create(file,{basePath},options);std::string token;{auto r=c.read();token=c.currentGeometry(r,a)->uuid;}
            GeometryArrayEdit change{"line_points",0,std::vector<uint8_t>(8)};std::memcpy(change.bytes.data(),&value,8);
            auto w=c.write();c.editGeometry(w,a,{change},token);w.commit();}
        for(auto mode:{HistoryMode::On,HistoryMode::Off}){
            changeHistoryMode(file,mode);auto c=ObjectCollection::open(file,OpenOptions{.readOnly=true});auto r=c.read();
            require(c.readGeometryArray(r,a,"line_points")==expected,"history policy preserves referenced geometry");
            require(c.provisioning(r).history==mode,"store_reference history policy persisted");
        }
    }
    // Two independent workers reserve disjoint ranges under one publication.
    const auto logical=util::newUuid(),lineage=util::newUuid();std::vector<std::filesystem::path> shards;std::vector<ObjectKey> keys;
    for(int i=0;i<2;++i){auto path=dir.path/("shard"+std::to_string(i)+".vcobj");shards.push_back(path);
        auto opt=options;opt.creationStoreUuid=logical;opt.creationGeneration=lineage;opt.initialCommitSequence=9;opt.objectIdRange={1+i*100,100+i*100};
        auto db=ObjectStore::create(path,opt);auto w=db.write();keys.push_back({logical,db.upsertObject(w,fiber())});db.seal(w);w.commit();}
    auto collection=ObjectCollection::publishShards(dir.path/"sharded.vcobj",shards,options);
    {
        auto r=collection.read();require(collection.provisioning(r).commitSequence==10&&collection.provisioning(r).generation==lineage,"common reserved publication lineage and commit sequence");
        require(collection.objectById(r,logical,101)->key==keys[1],"numeric range routes to second shard");
        require(!collection.objectById(r,logical,99),"reserved unpopulated ID remains absent");
        size_t count=0;collection.scanObjects(r,[&](auto&){++count;return true;});require(count==2,"all partitions enumerated");
    }
    mustFail([&]{ObjectCollection::publishShards(dir.path/"overlap.vcobj",{shards[0],shards[0]},options);});
    require(!std::filesystem::exists(dir.path/"overlap.vcobj"),"invalid shard publication invisible");
    for(auto mode:{RoutingKind::Modulo,RoutingKind::Bits,RoutingKind::HashV1}){
        const auto owner=util::newUuid(),generation=util::newUuid();std::vector<std::filesystem::path> parts;std::vector<KeyRoute> rules;std::map<int64_t,ObjectKey> ids,edgeIds;
        for(int partition=0;partition<2;++partition){auto p=dir.path/("route-"+std::to_string(int(mode))+"-"+std::to_string(partition));parts.push_back(p);
            auto opt=options;opt.creationStoreUuid=owner;opt.creationGeneration=generation;opt.objectIdRange={1,100};
            auto db=ObjectStore::create(p,opt);auto w=db.write();const auto physical=db.databaseIdentity(w).uuid;
            KeyRoute rule{"objects",mode,1,100,2,uint64_t(partition),mode==RoutingKind::Bits?1u:0u,physical};
            std::vector<std::string> endpoints;
            for(int64_t id=1;id<=12;++id)if(rule.matches(id)){
                const auto object=db.insertObjectWithId(w,fiber(),id);ids.emplace(id,ObjectKey{owner,object});endpoints.push_back(object);}
            require(endpoints.size()>=2,"custom edge route fixture needs two endpoints");
            for(int64_t id=1;id<=12;++id)if(rule.matches(id)){
                EdgeRecord edge;edge.uuid=util::newUuid();edge.kind="declared_reference";edge.aUuid=endpoints[0];edge.bUuid=endpoints[1];
                edgeIds.emplace(id,ObjectKey{owner,db.addEdge(w,edge)});
                mustFail([&]{db.addEdge(w,edge);});}
            rules.push_back(rule);db.seal(w);w.commit();}
        auto c=ObjectCollection::publishShards(dir.path/("routed-"+std::to_string(int(mode))),parts,options,rules);
        auto r=c.read();for(auto& [id,key]:ids)require(c.objectById(r,owner,id)->key==key,"custom routing exact");
        require(c.edgesByKind(r,"declared_reference").size()==edgeIds.size(),"bulk relation scan includes every shard");
        for(auto& [id,key]:edgeIds)require(c.edge(r,key)->key==key,"UUID edges remain exact beside custom object routing");
    }
    // Same manifest/routing API inside the packed VFS; no absolute staging refs.
    const auto packedPath=dir.path/"packed.vcobj";const auto packedOwner=util::newUuid(),packedLineage=util::newUuid();
    {
        PackedStoreBuilder builder(packedPath);std::vector<std::filesystem::path> paths;
        for(int i=0;i<2;++i){auto p=builder.addDatabase(util::newUuid());paths.push_back(p);auto opt=options;opt.creationStoreUuid=packedOwner;opt.creationGeneration=packedLineage;
            opt.objectIdRange={1+i*100,100+i*100};
            auto db=ObjectStore::create(p,opt);auto w=db.write();db.upsertObject(w,fiber());db.seal(w);w.commit();}
        const auto root=util::newUuid();auto rootPath=builder.addDatabase(root);
        {auto c=ObjectCollection::publishShards(rootPath,paths,options);auto r=c.read();size_t n=0;c.scanObjects(r,[&](auto&){++n;return true;});require(n==2,"unpublished packed routing");}
        builder.finalize(root);
    }
    {
        auto c=ObjectCollection::open(packedPath,OpenOptions{.readOnly=true});auto r=c.read();size_t n=0;c.scanObjects(r,[&](auto&){++n;return true;});require(n==2,"published packed routing");
    }
    std::filesystem::rename(packedPath,dir.path/"moved.vcobj");
    {
        auto c=ObjectCollection::open(dir.path/"moved.vcobj",OpenOptions{.readOnly=true});auto r=c.read();require(c.objectById(r,packedOwner,101).has_value(),"packed references survive move");
    }
    // Identity mismatch must never silently resolve to the wrong file.
    const auto brokenPath=dir.path/"broken.vcobj";
    {auto c=ObjectCollection::create(brokenPath,{basePath},options);}
    {auto db=ObjectStore::open(brokenPath,OpenOptions{.physicalPartitionOnly=true,.author="test"});auto w=db.write();
        execute(ObjectStoreImplAccess::conn(w),"UPDATE collection_members SET commit_sequence=commit_sequence+1");w.commit();}
    mustFail([&]{ObjectCollection::open(brokenPath,OpenOptions{.readOnly=true});});
    const auto brokenLineage=dir.path/"broken-lineage.vcobj";
    {auto c=ObjectCollection::create(brokenLineage,{basePath},options);}
    {auto db=ObjectStore::open(brokenLineage,OpenOptions{.physicalPartitionOnly=true,.author="test"});auto w=db.write();
        execute(ObjectStoreImplAccess::conn(w),"UPDATE collection_members SET generation=?",parseUuid(util::newUuid()));w.commit();}
    mustFail([&]{ObjectCollection::open(brokenLineage,OpenOptions{.readOnly=true});});
    std::cout<<"collections passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
