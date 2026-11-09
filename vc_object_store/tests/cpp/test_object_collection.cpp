#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/objects/StoreConversion.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "internal/Database.hpp"
#include <iostream>
#include <cstring>
using namespace vc::objects;
using namespace vc::objects::detail;
ObjectRecord fiber(){ObjectRecord r;r.kind="fiber";r.fiber.emplace();return r;}

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
                baseVersion=base.versionReference(r,*base.currentObjectRevision(r,managed.objectUuid)); }
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
    mustFail([&]{ObjectStore::open(overlayPath,OpenOptions{.readOnly=true});});
    {
        auto r=overlay.read();require(overlay.readGeometryArray(r,a,"line_points")==geometry,"inherited geometry exact");
        require(overlay.provisioning(r).commitSequence>overlay.members()[1].commitSequence,"derived commit sequence above parent");
        require(overlay.objectById(r,a.storeUuid,1)->key==a,"numeric base routing");
    }
    ObjectKey newFiber,edge;
    {
        auto w=overlay.write();overlay.setMeta(w,a,"classification","\"H\"");newFiber=overlay.upsertObject(w,fiber());
        EdgeRecord e;e.kind="fiber_constraint";e.aAnchor.wholeGeometry=true;e.bAnchor.wholeGeometry=true;
        edge=overlay.addEdge(w,a,newFiber,e);w.commit();
        auto r=overlay.read();require(overlay.edgesOf(r,a).size()==1,"cross store adjacency");
        require(overlay.edgesOf(r,newFiber)[0].key==edge,"reverse adjacency");
        require(overlay.object(r,a)->record.meta.find("H")!=std::string::npos,"metadata override");
        require(overlay.readGeometryArray(r,a,"line_points")==geometry,"metadata override borrows chunks");
        size_t count=0;overlay.scanObjects(r,[&](auto&){++count;return true;});require(count==3,"references do not duplicate objects");
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
        size_t count=0;child.scanObjects(r,[&](auto&){++count;return true;});require(count==2,"deleted inherited geometry not enumerated");
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
        auto opt=options;opt.creationStoreUuid=logical;opt.creationGeneration=lineage;opt.initialCommitSequence=9;opt.objectIdRange={1+i*100,100+i*100};opt.edgeIdRange=opt.objectIdRange;
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
            auto opt=options;opt.creationStoreUuid=owner;opt.creationGeneration=generation;opt.objectIdRange={1,100};opt.edgeIdRange={1,100};
            auto db=ObjectStore::create(p,opt);auto w=db.write();const auto physical=db.databaseIdentity(w).uuid;
            KeyRoute rule{"objects",mode,1,100,2,uint64_t(partition),mode==RoutingKind::Bits?1u:0u,physical};
            std::vector<std::string> endpoints;
            for(int64_t id=1;id<=12;++id)if(rule.matches(id)){
                const auto object=db.insertObjectWithId(w,fiber(),id);ids.emplace(id,ObjectKey{owner,object});endpoints.push_back(object);}
            require(endpoints.size()>=2,"custom edge route fixture needs two endpoints");
            for(int64_t id=1;id<=12;++id)if(rule.matches(id)){
                EdgeRecord edge;edge.kind="routed";edge.aUuid=endpoints[0];edge.bUuid=endpoints[1];
                edgeIds.emplace(id,ObjectKey{owner,db.insertEdgeWithId(w,edge,id)});
                mustFail([&]{db.insertEdgeWithId(w,edge,id);});}
            rules.push_back(rule);rule.keySpace="edges";rules.push_back(rule);db.seal(w);w.commit();}
        auto c=ObjectCollection::publishShards(dir.path/("routed-"+std::to_string(int(mode))),parts,options,rules);
        auto r=c.read();for(auto& [id,key]:ids)require(c.objectById(r,owner,id)->key==key,"custom routing exact");
        for(auto& [id,key]:edgeIds)require(c.edgeById(r,owner,id)->key==key,"custom edge routing exact");
    }
    // Same manifest/routing API inside the packed VFS; no absolute staging refs.
    const auto packedPath=dir.path/"packed.vcobj";const auto packedOwner=util::newUuid(),packedLineage=util::newUuid();
    {
        PackedStoreBuilder builder(packedPath);std::vector<std::filesystem::path> paths;
        for(int i=0;i<2;++i){auto p=builder.addDatabase(util::newUuid());paths.push_back(p);auto opt=options;opt.creationStoreUuid=packedOwner;opt.creationGeneration=packedLineage;
            opt.objectIdRange={1+i*100,100+i*100};opt.edgeIdRange=opt.objectIdRange;
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
