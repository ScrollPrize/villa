#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
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
        SurfaceProximityQuery query;query.volume="b";query.point={12.5,1,21.5};query.tolerance=.51;
        const auto hits=db.nearestSurfaces(txn,query);require(hits.size()==1,"registered nearest surface");
        require(std::abs(hits.front().distance-.5)<1e-10,"target-space Euclidean distance");
        RegionOptions region;region.volume="b";region.kind="fiber";
        RegionSlice slice;slice.origin={13,0,20};slice.localBounds={-4,-1,-1,4,1,1};
        const auto runs=db.geometryRegion(txn,slice,region);require(runs.complete()&&runs.runs.size()==1,"registered region intersection");
        require(runs.runs.front().points.front()==std::array<double,3>{10,0,20},"runs use target coordinates");
    }
    std::string objectRevision,geometryRevision;
    std::vector<GeometryArrayEdit> changes;
    {
        auto snapshot=db.geometrySnapshot();const auto view=snapshot.track(track).viewTransformedAt("b");
        require(snapshot.pointCollection(points).viewTransformedAt("b").points("points",0,1).front()==std::array<double,3>{13,2,23},"point collections share transformed accessor");
        require(snapshot.umbilicus(umbilicus).viewTransformedAt("b").points("points",0,1).front()==std::array<double,3>{13,2,23},"umbilici share transformed accessor");
        require(snapshot.fiber(fiber).viewTransformedAt("b").points("control_points",0,2).back()==std::array<double,3>{16,0,20},"fibers share transformed accessor");
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
    std::cout<<"object volume transformation checks passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
