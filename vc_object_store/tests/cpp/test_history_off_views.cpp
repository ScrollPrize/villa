#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/objects/VolumeTransforms.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
#include "vc_object_store/geometry/ArrayBytes.hpp"
#include <iostream>

using namespace vc::objects;

int main() try
{
    TemporaryDirectory directory;
    OpenOptions options; options.author="direct-test"; options.history=HistoryMode::Off;
    const auto path=directory.path/"direct.vcobj";
    auto store=ObjectStore::create(path,options);
    std::string fiber, registration;
    {
        auto txn=store.write();
        ObjectRecord record;record.kind="fiber";
        record.meta=R"({"coordinates":{"volume_uuid":"a","scale_factor":1}})";
        fiber=store.upsertObject(txn,record);
        vc::geometry::Fiber data;data.linePoints={{0,0,0},{20,0,0}};
        store.putGeometry(txn,fiber,vc::geometry::encode(data));
        registration=store.registerVolumeTransform(txn,"a","b",VolumeAffine::scale(2),"test");
        require(store.volumeTransforms(txn).resolve("a","b").resolved().forward.apply({1,0,0})[0]==2,
                "current-only registration is available inside its write transaction");
        txn.commit();
    }
    {
        GeometrySnapshot snapshot(store);
        const auto view=snapshot.fiber(fiber);
        require(view.version()==1,"current view uses object version without synthetic history");
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
    {
        auto txn=store.write();
        const auto change=[](double value){return GeometryArrayEdit{"line_points",3,detail::scalarBytes<double>(std::array<double,1>{value})};};
        const auto token=store.editGeometry(txn,fiber,{change(21)},store.currentGeometry(txn,fiber)->uuid);
        GeometrySnapshot borrowed(store,txn);
        const auto previous=borrowed.fiber(fiber);
        require(previous.load().linePoints.back()[0]==21,"writer view reads current geometry");
        require(store.geometryLengthInVolume(txn,fiber,"b")==42,"writer transformed length before replacement");
        require(store.editGeometry(txn,fiber,{change(22)},token)==token,
                "current geometry identity is object plus object version");
        mustFail([&]{previous.load();});
        require(borrowed.fiber(fiber).load().linePoints.back()[0]==22,"new writer view sees replacement at same version");
        require(store.geometryLengthInVolume(txn,fiber,"b")==44,"writer transformed length does not cache same-token old geometry");
        // Roll back to preserve the fixture used by the session checks below.
    }
    ObjectStoreSession session;
    const auto attached=session.attach(path,options);
    require(session.writable(attached),"history-off session attaches writable");
    session.takeEvents();
    PinnedObject pin;
    int64_t sequence=0;
    session.readStore(attached,[&](const StoreReader& reader,const auto&){
        const auto state=reader.status();
        pin=pinOf(attached,*reader.currentObject(fiber),state.generation);
        sequence=state.commitSequence;
    });
    session.requireCurrent(pin);
    auto request=session.newRequest(attached);
    request.expectedObjects[fiber]=pin.version;
    request.operations.push_back(SetMetaOp{fiber,"direct","true"});
    session.submit(attached,request);
    mustFail([&]{session.requireCurrent(pin);});
    auto events=session.takeEvents();
    require(events.size()==1&&events[0].revalidate&&events[0].changes.empty()&&events[0].journalPosition==sequence+1,
            "current-only commits invalidate without a fabricated history journal");
    require(events[0].touches({"fiber"})&&events[0].touches({"surface"}),"whole-store invalidation reaches every cache");
    auto stale=session.newRequest(attached);
    stale.expectedObjects[fiber]=pin.version;
    stale.operations.push_back(SetMetaOp{fiber,"stale","true"});
    mustFail([&]{session.submit(attached,stale);});
    require(session.takeEvents().empty(),"stale record version rejects without a commit");
    session.editRecords(attached,{fiber},[](auto& record){record.name="renamed";});
    session.readStore(attached,[&](const StoreReader& reader,const auto&){
        require(reader.currentObject(fiber)->record.name=="renamed","ordinary current record edit works");
    });
    registerVolumeTransform(session,attached,"b","c",VolumeAffine::scale(3),"new");
    std::vector<std::string> route;
    session.readStore(attached,[&](const StoreReader& reader,const auto&){route=reader.volumeTransforms().resolve("b","c").resolved().path;});
    chooseVolumeTransform(session,attached,"b","c",route);
    chooseVolumeTransform(session,attached,"b","c",route);
    session.readStore(attached,[&](const StoreReader& reader,const auto&){
        require(GeometrySnapshot(reader).fiber(fiber).viewTransformedAt("c").points("line_points",0,2).back()[0]==120,
                "session registrations and choice updates are current-only");
    });
    const auto copied=session.copyObject(attached,attached,fiber,"copy");
    session.readStore(attached,[&](const StoreReader& reader,const auto&){
        require(GeometrySnapshot(reader).fiber(copied).load().linePoints.back()[0]==20,"session copies current geometry without history");
    });
    session.takeEvents();
    {auto txn=store.write();store.setMeta(txn,fiber,"external","true");txn.commit();}
    events=session.takeEvents();
    require(events.size()==1&&events[0].revalidate,"external current-only changes are observed");
    const auto remote=util::newUuid();
    request=session.newRequest(attached);
    request.expectedObjects[remote]=std::nullopt;
    ObjectRecord record;record.uuid=remote;record.kind="fiber";record.meta=R"({"remote_test":true})";
    request.operations.push_back(CreateObjectOp{objectRecordJson(record)});
    session.submit(attached,request);
    session.setGeometryFetcher("remote_test",[](auto& db,const std::string& uuid,const ObjectStoreSession::FetchWrite& write){
        VersionSelection expected;
        {auto txn=db.read();expected[uuid]=db.currentVersion(txn,uuid);}
        write(expected,[&](auto& target,auto& txn){
            vc::geometry::Fiber data;data.linePoints={{0,0,0},{7,0,0}};
            target.putGeometry(txn,uuid,vc::geometry::encode(data));
        });
    });
    session.ensureGeometry(attached,remote);
    session.readStore(attached,[&](const StoreReader& reader,const auto&){
        require(GeometrySnapshot(reader).fiber(remote).linePoints(0,2).back()[0]==7,"current-only async fetch uses ordinary version checks");
    });
    std::vector<std::string> previousRoute;
    {auto txn=store.read();previousRoute=store.volumeTransforms(txn).resolve("a","b").resolved().path;}
    {auto txn=store.write();store.registerVolumeTransform(txn,"a","b",VolumeAffine::scale(4),"changed",registration);txn.commit();}
    {auto txn=store.read();
     const auto updated=store.volumeTransforms(txn).resolve("a","b").resolved();
     require(updated.path!=previousRoute,"current-only registration edits change route version evidence");}
    return 0;
} catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
}
