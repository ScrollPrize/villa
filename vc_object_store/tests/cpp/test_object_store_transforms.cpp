#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
#include "vc_object_store/objects/VolumeUmbilicus.hpp"
#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/geometry/ArrayBytes.hpp"
#include <iostream>
using namespace vc::objects;
int main() try {
    TemporaryDirectory directory;OpenOptions options;options.author="transform-test";
    auto db=ObjectStore::create(directory.path/"transforms.vcobj",options);
    std::string fiber,track,surface,points,umbilicus,registration;
    VolumeAffine matrix;matrix(0,0)=2;matrix(0,1)=.5;matrix(0,3)=10;matrix(2,3)=20;
    {
        auto txn=db.write();
        registration=db.registerVolumeTransform(txn,"a","b",matrix,"scan registration");
        bool pending=false;
        try { (void)db.volumeTransforms(txn); } catch(const std::runtime_error&) { pending=true; }
        require(pending,"new registrations must have a published revision before resolving views");
        auto add=[&](const std::string& kind,const vc::geometry::Geometry& geometry){
            ObjectRecord r;r.kind=kind;r.meta=R"({"coordinates":{"volume_uuid":"a","scale_factor":1}})";
            auto id=db.upsertObject(txn,r);db.putGeometry(txn,id,vc::geometry::encode(geometry));return id;
        };
        vc::geometry::Fiber f;f.controlPoints={{0,0,0},{3,0,0}};fiber=add("fiber",f);
        vc::geometry::Track t;t.points={{0,0,0},{3,0,0}};track=add("track",t);
        vc::geometry::PointCollection pcl;pcl.ids={17};pcl.points={{1,2,3}};points=add("point_collection",pcl);
        vc::geometry::Umbilicus u;u.points={{1,2,3},{2,3,4}};umbilicus=add("umbilicus",u);
        vc::geometry::Surface s;s.rows=s.columns=2;s.points={{0,0,1},{2,0,1},{0,2,1},{2,2,1}};surface=add("surface",s);
        txn.commit();
    }
    {
        auto txn=db.read();ObjectSelection select;select.kind="fiber";select.volume="b";select.minLength=5.5;
        require(db.selectObjects(txn,select).size()==1,"target-space length includes registered fiber");
        select.minLength=6.5;require(db.selectObjects(txn,select).empty(),"length threshold in target space");
        select.minLength.reset();select.orderBy="length_desc";
        const auto ordered=db.selectObjects(txn,select);
        require(ordered.size()==1&&ordered.front().volumeLength==db.geometryLengthInVolume(txn,fiber,"b"),
                "registered rows carry the target length they were ordered by");
        SurfaceProximityQuery query;query.volume="b";query.point={12.5,1,21.5};query.tolerance=.51;
        const auto hits=db.nearestSurfaces(txn,query);require(hits.size()==1,"registered nearest surface");
        require(std::abs(hits.front().distance-.5)<1e-10,"target-space Euclidean distance");
        // At half resolution (a pyramid level) the query and hit are in half-scale voxels.
        query.targetScale=.5;query.point={6.25,.5,10.75};query.tolerance=.26;
        const auto half=db.nearestSurfaces(txn,query);
        require(half.size()==1&&std::abs(half.front().distance-.25)<1e-10,"proximity at a target scale");
        RegionOptions region;region.volume="b";region.kind="fiber";
        RegionSlice slice;slice.origin={13,0,20};slice.localBounds={-4,-1,-1,4,1,1};
        const auto runs=db.geometryRegion(txn,slice,region);require(runs.complete()&&runs.runs.size()==1,"registered region intersection");
        require(runs.runs.front().points.front()==std::array<double,3>{10,0,20},"runs use target coordinates");
        region.targetScale=.5;slice.origin={6.5,0,10};slice.localBounds={-2,-.5,-.5,2,.5,.5};
        const auto halfRuns=db.geometryRegion(txn,slice,region);
        require(halfRuns.complete()&&halfRuns.runs.size()==1&&halfRuns.runs.front().points.front()==std::array<double,3>{5,0,10},
                "regions at a target scale");
        // A registered surface cut by x'=13 and clipped to y in [0.5,1.5]: both
        // run ends are interpolated inside the mesh, already in target space.
        RegionOptions surfaces;surfaces.volume="b";surfaces.kind="surface";
        RegionSlice cut;cut.origin={13,1,21};cut.u={0,1,0};cut.v={0,0,1};cut.normal={1,0,0};cut.localBounds={-.5,-1,-1,.5,1,1};
        auto clippedEnds=[&](double scale){
            const auto found=db.geometryRegion(txn,cut,surfaces);
            require(found.complete()&&found.runs.size()==1&&found.runs.front().points.size()>=2,"registered surface cut");
            for(const auto& p:{found.runs.front().points.front(),found.runs.front().points.back()})
                require(std::abs(p[0]-13*scale)<1e-9&&std::abs(p[2]-21*scale)<1e-9&&
                        (std::abs(p[1]-.5*scale)<1e-9||std::abs(p[1]-1.5*scale)<1e-9),"clipped surface run ends are mapped once");
        };
        clippedEnds(1);
        surfaces.targetScale=.5;cut.origin={6.5,.5,10.5};cut.localBounds={-.25,-.5,-.5,.25,.5,.5};
        clippedEnds(.5);
    }
    std::string objectRevision,geometryRevision;
    std::vector<GeometryArrayEdit> changes;
    {
        auto snapshot=db.geometrySnapshot();const auto view=snapshot.track(track).viewTransformedAt("b");
        require(snapshot.pointCollection(points).viewTransformedAt("b").points("points",0,1).front()==std::array<double,3>{13,2,23},"point collections share transformed accessor");
        require(snapshot.umbilicus(umbilicus).viewTransformedAt("b").points("points",0,1).front()==std::array<double,3>{13,2,23},"umbilici share transformed accessor");
        require(snapshot.fiber(fiber).viewTransformedAt("b").points("control_points",0,2).back()==std::array<double,3>{16,0,20},"fibers share transformed accessor");
        {
            // A whole edited fiber back to canonical coordinates: shown points keep
            // their exact stored values by ID; moved and inserted ones map back.
            const auto fiberView=snapshot.fiber(fiber);const auto shown=fiberView.viewTransformedAt("b");
            const auto ids=fiberView.controlPointIds(0,2);
            vc::geometry::Fiber edited;
            edited.controlPoints={shown.points("control_points",0,2)[1],{12,0,20},{14,1,21}};
            edited.controlPointIds={ids[1],ids[0],ids[1]+100};
            const auto canonical=shown.canonicalFiber(edited);
            require(canonical.controlPoints[0]==std::array<double,3>{3,0,0},"unchanged control point keeps its canonical value");
            require(canonical.controlPoints[1]==std::array<double,3>{1,0,0},"moved control point maps through the inverse");
            const auto back=matrix.apply(canonical.controlPoints[2]);
            require(std::abs(back[0]-14)<1e-12&&std::abs(back[1]-1)<1e-12&&std::abs(back[2]-21)<1e-12,"inserted control point maps back");
            require(canonical.controlPointIds==edited.controlPointIds,"control point IDs are kept");
        }
        {
            // The same for a point collection, matched by stable point ID.
            const auto collectionView=snapshot.pointCollection(points);const auto shown=collectionView.viewTransformedAt("b");
            vc::geometry::PointCollection edited;
            edited.ids={collectionView.ids(0,1).front(),99};
            edited.points={shown.points("points",0,1).front(),{14,1,21}};
            const auto canonical=shown.canonicalPointCollection(edited);
            require(canonical.points[0]==collectionView.points(0,1).front(),"unchanged point keeps its canonical value");
            const auto back=matrix.apply(canonical.points[1]);
            require(std::abs(back[0]-14)<1e-12&&std::abs(back[1]-1)<1e-12&&std::abs(back[2]-21)<1e-12,"new point maps back");
            require(canonical.ids==edited.ids,"point IDs are kept");
        }
        require(snapshot.pointCollection(points).ids(0,1).front()==17,"stable point IDs remain canonical");
        require(snapshot.track(track).points(1,1).front()[0]==3,"requesting view never changes canonical geometry");
        bool unavailable=false;try{snapshot.track(track).viewTransformedAt("missing");}catch(const std::exception&){unavailable=true;}
        require(unavailable,"unavailable target is explicit");
        require(view.points("points",0,2).back()==std::array<double,3>{16,0,20},"floating track view");
        require(view.coordinateEdits("points",0,view.points("points",0,2)).empty(),"unchanged positions remain byte-identical");
        changes=view.coordinateEdits("points",1,{{17,0,20}}); // inverse x=3.5 -> 4
        objectRevision=view.objectRevision();geometryRevision=view.revision().uuid;
        const auto tile=snapshot.viewTransformedAt(surface,"b").readTile(0,0,1,2);
        require(tile.points.size()==2&&tile.validity[0],"bounded surface tile");
        const auto surfaceView=snapshot.surface(surface).viewTransformedAt("b");
        require(surfaceView.surfaceEdits(0,0,1,2,{{10,0,21},{14,0,21}}).empty(),"surface no-op keeps canonical bytes");
        const auto surfaceChanges=surfaceView.surfaceEdits(0,0,1,1,{{12,0,21}});
        require(surfaceChanges.size()==1,"surface inverse edits supplied by store");
        const auto changedPoint=vc::objects::xyzFromBytes<float>(surfaceChanges.front().bytes).front();
        require(changedPoint==std::array<float,3>{1,0,1},"surface inverse uses registration");
        snapshot.close();
        bool closed=false;try{view.points("points",0,1);}catch(const std::exception&){closed=true;}require(closed,"snapshot lifetime retained");
    }
    {
        auto txn=db.write({{track,objectRevision}});db.editGeometry(txn,track,changes,geometryRevision);txn.commit();
        auto snapshot=db.geometrySnapshot();require(snapshot.track(track).points(1,1).front()[0]==4,"canonical track rounding");
    }
    {
        auto txn=db.write();db.registerVolumeTransform(txn,"a","b",VolumeAffine::scale(3),"alternative");txn.commit();
        auto read=db.read();const auto context=db.volumeTransforms(read);const auto conflict=context.resolve("a","b");
        require(conflict.status==vcr::catalog::TransformStatus::Ambiguous,"conflicting store registrations");
        bool ambiguous=false;try{GeometrySnapshot(db,read).surface(surface).viewTransformedAt("b");}catch(const std::exception&){ambiguous=true;}
        require(ambiguous,"ambiguous target cannot silently choose a view");
        require(!context.toVolume(nlohmann::json::parse(db.objectByUuid(read,fiber)->meta),"b"),"ambiguous object stays unavailable");
    }
    {
        // A choice settles the ambiguity; choosing again updates the same record.
        std::vector<std::vector<std::string>> paths;
        {auto read=db.read();for(const auto& c:db.volumeTransforms(read).graph.resolve("a","b").candidates)paths.push_back(c.path);}
        require(paths.size()==2,"two candidate routes");
        const auto choice=vc::objects::util::newUuid();
        {auto txn=db.write({{choice,std::nullopt}});require(db.chooseVolumeTransform(txn,"a","b",paths[1],choice)==choice,"choice takes the given uuid");txn.commit();}
        {auto read=db.read();require(db.volumeTransforms(read).resolve("a","b").status==vcr::catalog::TransformStatus::Resolved,"a choice resolves the pair");}
        {auto read=db.read();const auto head=db.currentObjectRevision(read,choice)->uuid;
         auto txn=db.write({{choice,head}});require(db.chooseVolumeTransform(txn,"a","b",paths[0])==choice,"choosing again updates the choice");txn.commit();}
        auto read=db.read();
        require(db.volumeTransforms(read).resolve("a","b").resolved().path==paths[0],"the latest choice wins");
    }
    {
        // A volume's one umbilicus, read in its voxels; importing another replaces it.
        ObjectRecord first;first.kind="umbilicus";first.name="first";first.uuid=vc::objects::util::newUuid();
        first.meta=R"({"coordinates":{"volume_uuid":"c","scale_factor":2}})";
        vc::geometry::Umbilicus u;u.points={{2,4,6},{4,6,8}};
        {auto txn=db.write({{first.uuid,std::nullopt}});setVolumeUmbilicus(db,txn,"c",first,u);txn.commit();}
        {
            auto read=db.read();const auto found=volumeUmbilicus(db,read,"c");
            require(found&&found->uuid==first.uuid&&found->points.front()==std::array<double,3>{1,2,3},
                    "the volume's umbilicus in its voxels");
            require(!volumeUmbilicus(db,read,"d"),"a volume without one has none");
        }
        auto second=first;second.uuid=vc::objects::util::newUuid();second.name="second";
        RevisionSelection pins;{auto read=db.read();pins=volumeUmbilicusRevisions(db,read,"c");}
        require(pins.size()==1&&pins.contains(first.uuid),"the replacement pins the umbilicus it replaces");
        pins[second.uuid]=std::nullopt;
        {auto txn=db.write(pins);setVolumeUmbilicus(db,txn,"c",second,u);txn.commit();}
        {
            auto read=db.read();
            require(volumeUmbilicus(db,read,"c")->uuid==second.uuid&&db.objectByUuid(read,first.uuid)->retired,
                    "importing replaces the volume's umbilicus");
        }
        auto elsewhere=first;elsewhere.uuid=vc::objects::util::newUuid();
        elsewhere.meta=R"({"coordinates":{"volume_uuid":"d","scale_factor":1}})";
        bool refused=false;
        try{auto txn=db.write({{elsewhere.uuid,std::nullopt}});setVolumeUmbilicus(db,txn,"c",elsewhere,u);}
        catch(const std::invalid_argument&){refused=true;}
        require(refused,"an umbilicus bound elsewhere is not made a volume's umbilicus");
        auto extra=first;extra.uuid=vc::objects::util::newUuid();
        {auto txn=db.write();db.upsertObject(txn,extra);db.putGeometry(txn,extra.uuid,vc::geometry::encode(u));txn.commit();}
        bool several=false;
        try{auto read=db.read();(void)volumeUmbilicus(db,read,"c");}catch(const std::runtime_error&){several=true;}
        require(several,"several umbilici for one volume are refused");
    }
    {
        // Binding declares a frame once; a conflicting declaration is refused.
        std::string loose;
        {auto txn=db.write();ObjectRecord r;r.kind="fiber";loose=db.upsertObject(txn,r);txn.commit();}
        const nlohmann::json at{{"volume_uuid","c"},{"scale_factor",2}};
        {auto txn=db.write();db.bindGeometryCoordinates(txn,loose,at,{{"shape_zyx",{1,2,3}}});txn.commit();}
        {auto txn=db.write();db.bindGeometryCoordinates(txn,loose,at);txn.commit();}
        auto read=db.read();const auto meta=nlohmann::json::parse(db.objectByUuid(read,loose)->meta);
        require(meta["coordinates"]==at&&meta["coordinate_binding"]["shape_zyx"].size()==3,"bound coordinates and binding");
        bool conflict=false;
        try{auto txn=db.write();db.bindGeometryCoordinates(txn,loose,{{"volume_uuid","c"},{"scale_factor",4}});}
        catch(const std::invalid_argument&){conflict=true;}
        require(conflict,"a different frame is refused");
    }
    std::cout<<"object volume transformation checks passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
