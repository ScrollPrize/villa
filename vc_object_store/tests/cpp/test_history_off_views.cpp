#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/objects/VolumeTransforms.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
#include <iostream>

using namespace vc::objects;

int main() try
{
    TemporaryDirectory directory;
    OpenOptions options; options.author="direct-test"; options.history=HistoryMode::Off;
    const auto path=directory.path/"direct.vcobj";
    auto store=ObjectStore::create(path,options);
    std::string fiber;
    {
        auto txn=store.write();
        ObjectRecord record;record.kind="fiber";
        record.meta=R"({"coordinates":{"volume_uuid":"a","scale_factor":1}})";
        fiber=store.upsertObject(txn,record);
        vc::geometry::Fiber data;data.linePoints={{0,0,0},{20,0,0}};
        store.putGeometry(txn,fiber,vc::geometry::encode(data));
        store.registerVolumeTransform(txn,"a","b",VolumeAffine::scale(2),"test");
        mustFail([&]{store.volumeTransforms(txn);});
        txn.commit();
    }
    {
        GeometrySnapshot snapshot(store);
        const auto view=snapshot.fiber(fiber);
        require(view.objectRevision().empty()&&view.objectGeneration()>0,"current view uses generation without synthetic history");
        require(view.controlPointCount()==0&&view.linePointCount()==2,"line-only fiber has no invented controls");
        require(view.load().linePoints.back()[0]==20,"typed current load works without history");
        const auto shown=view.viewTransformedAt("b");
        require(shown.points("line_points",0,2).back()[0]==40,"current transforms work without object revisions");
        {
            auto txn=store.write();shown.validateForWrite(store,txn);
            store.setMeta(txn,fiber,"note","true");txn.commit();
        }
        auto current=store.read();
        mustFail([&]{shown.validateForWrite(store,current);});
    }
    ObjectStoreSession session;
    const auto attached=session.attach(path,options);
    require(session.writable(attached),"history-off session attaches writable");
    session.takeEvents();
    PinnedObject pin;
    session.read(attached,[&](const auto& db,const auto& txn,const auto&){
        pin={attached,{attached.storeUuid,fiber},"",db.currentGeometry(txn,fiber)->uuid,db.objectByUuid(txn,fiber)->generation};
    });
    {const auto lease=session.holdForEditing(pin);}
    int64_t generation=0;
    session.read(attached,[&](const auto& db,const auto& txn,const auto&){generation=db.provisioning(txn).generation;});
    session.actAtGeneration(attached,generation,[&](auto& db,auto& txn){db.setMeta(txn,fiber,"direct","true");});
    mustFail([&]{session.holdForEditing(pin);});
    auto events=session.takeEvents();
    require(events.size()==1&&events[0].fullInvalidation&&events[0].changes.empty()&&events[0].storeGeneration==generation+1,
            "current-only commits emit generation invalidation without history journal");
    require(events[0].touches({"fiber"})&&events[0].touches({"surface"}),"whole-store invalidation reaches every cache");
    bool invoked=false;
    mustFail([&]{session.actAtGeneration(attached,generation,[&](auto&,auto&){invoked=true;});});
    require(!invoked&&session.takeEvents().empty(),"stale generation rejects before callback without commit");
    session.editRecords(attached,{fiber},[](auto& record){record.name="renamed";});
    session.read(attached,[&](const auto& db,const auto& txn,const auto&){require(db.objectByUuid(txn,fiber)->name=="renamed","ordinary current record edit works");});
    registerVolumeTransform(session,attached,"b","c",VolumeAffine::scale(3),"new");
    std::vector<std::string> route;
    session.read(attached,[&](const auto& db,const auto& txn,const auto&){route=db.volumeTransforms(txn).resolve("b","c").resolved().path;});
    chooseVolumeTransform(session,attached,"b","c",route);
    chooseVolumeTransform(session,attached,"b","c",route);
    session.read(attached,[&](const auto& db,const auto& txn,const auto&){
        require(GeometrySnapshot(db,txn).fiber(fiber).viewTransformedAt("c").points("line_points",0,2).back()[0]==120,
                "session registrations and choice updates are current-only");
    });
    const auto copied=session.copyObject(attached,attached,fiber,"copy");
    session.read(attached,[&](const auto& db,const auto& txn,const auto&){
        require(GeometrySnapshot(db,txn).fiber(copied).load().linePoints.back()[0]==20,"session copies current geometry without history");
    });
    session.takeEvents();
    {auto txn=store.write();store.setMeta(txn,fiber,"external","true");txn.commit();}
    events=session.takeEvents();
    require(events.size()==1&&events[0].fullInvalidation,"external current-only changes are observed");
    std::string remote;
    session.read(attached,[&](const auto& db,const auto& txn,const auto&){generation=db.provisioning(txn).generation;});
    session.actAtGeneration(attached,generation,[&](auto& db,auto& txn){
        ObjectRecord record;record.kind="fiber";record.meta=R"({"remote_test":true})";
        remote=db.upsertObject(txn,record);
    });
    session.setGeometryFetcher("remote_test",[](auto& db,const std::string& uuid,const ObjectStoreSession::FetchWrite& write){
        int64_t expected;
        {auto txn=db.read();expected=db.provisioning(txn).generation;}
        write.atGeneration(expected,[&](auto& target,auto& txn){
            vc::geometry::Fiber data;data.linePoints={{0,0,0},{7,0,0}};
            target.putGeometry(txn,uuid,vc::geometry::encode(data));
        });
    });
    session.ensureGeometry(attached,remote);
    session.read(attached,[&](const auto& db,const auto& txn,const auto&){
        require(GeometrySnapshot(db,txn).fiber(remote).linePoints(0,2).back()[0]==7,"current-only async geometry fetch uses generation precondition");
    });
    return 0;
} catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
}
