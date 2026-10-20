#define VC_TEST_MAIN
#include "vc_test.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/geometry/SegmentClip.hpp"
#include "vc_object_store/Uuid.hpp"
#include "internal/ReadRequest.hpp"
#include "geometry/GeometryReader.hpp"
#include <future>
using namespace vc::objects;
namespace {
struct Fixture {
    std::filesystem::path path=std::filesystem::temp_directory_path()/("vc-region-"+util::newUuid());
    ObjectStore store=ObjectStore::create(path,OpenOptions{.author="region test"});
    std::string id, revision;
    Fixture() {
        auto w=store.write();ObjectRecord o;o.kind="fiber";o.fiber=FiberDetail{};
        id=store.upsertObject(w,o);
        vc::geometry::Fiber f;
        for(int i=0;i<6000;++i)f.controlPoints.push_back({double(i),0,0});
        revision=store.putGeometry(w,id,vc::geometry::encode(f));w.commit();
    }
    void indexed() {
        RevisionSelection expected;
        {auto r=store.read();expected={{id,store.currentObjectRevision(r,id)->uuid}};}
        auto w=store.write(expected);GeometrySequenceEdit edit;edit.array="control_points";edit.start=3000;
        revision=store.editGeometrySequence(w,id,edit,revision);w.commit();
    }
    ~Fixture() {for(auto suffix:{"","-wal","-shm"}){std::error_code e;std::filesystem::remove(path.string()+suffix,e);}}
};
FiberSlice all() {FiberSlice s;s.localBounds={-1,-1,-1,7000,1,1};return s;}
}
TEST_CASE("cancellation reaches SQL tree payload verification and clipping and leaves snapshot usable") {
    Fixture f;f.indexed();auto r=f.store.read();
    const std::vector<FiberRegionCancel> cancellations{
        [](const auto& s){return s.sqlSteps>=10;},
        [](const auto& s){return s.candidateVisits>=1;},
        [](const auto& s){return s.treeVisits>=2;},
        [](const auto& s){return s.payloadBytes>=10000;},
        [](const auto& s){return s.verifiedBytes>=10000;},
        [](const auto& s){return s.segmentTests>=5;},
    };
    for(const auto& cancel:cancellations) {
        const auto partial=f.store.fiberRegion(r,all(),{},cancel);
        CHECK(partial.stop==FiberRegionStop::Cancelled);
        CHECK_FALSE(partial.complete());
        const auto complete=f.store.fiberRegion(r,all());
        REQUIRE(complete.complete());REQUIRE(complete.runs.size()==1);
        CHECK(complete.runs[0].points.size()==6000);
    }
    CHECK_THROWS_WITH_AS(f.store.fiberRegion(r,all(),{},[](const auto&)->bool{throw std::runtime_error("callback failure");}),doctest::Contains("callback failure"),std::runtime_error);
    CHECK(f.store.fiberRegion(r,all()).complete());
    r.commit();CHECK_THROWS(f.store.fiberRegion(r,all()));
}
TEST_CASE("strict allowances count cache hits rejected segments and processing separately") {
    Fixture f;auto r=f.store.read();
    FiberRegionOptions o;o.candidateWorkBudget=10;
    auto limited=f.store.fiberRegion(r,all(),o);
    CHECK(limited.stop==FiberRegionStop::CandidateBudget);
    CHECK(limited.stats.candidateWork==10);
    CHECK(limited.stats.candidateVisits+limited.stats.treeVisits+limited.stats.segmentTests==10);
    o=FiberRegionOptions{};o.sqlStepBudget=150;
    limited=f.store.fiberRegion(r,all(),o);
    CHECK(limited.stop==FiberRegionStop::SqlBudget);CHECK(limited.stats.sqlSteps==150);
    o=FiberRegionOptions{};o.processingByteBudget=100000;
    limited=f.store.fiberRegion(r,all(),o);
    CHECK(limited.stop==FiberRegionStop::ProcessingBudget);CHECK(limited.stats.processingBytes<=100000);
    const auto complete=f.store.fiberRegion(r,all());
    REQUIRE(complete.complete());CHECK(complete.stats.cacheBytes>0);
    CHECK(complete.stats.processingBytes==complete.stats.payloadBytes+complete.stats.verifiedBytes+
        complete.stats.metadataBytes+complete.stats.reconstructedBytes+complete.stats.cacheBytes);
}
TEST_CASE("bounds filtering metadata and geometry retain a snapshot across a concurrent commit") {
    Fixture f;auto read=f.store.read();
    const auto before=f.store.fiberRegion(read,all());
    REQUIRE(before.runs.size()==1);
    auto writer=std::async(std::launch::async,[&]{
        auto store=ObjectStore::open(f.path,OpenOptions{.author="other"});auto w=store.write();
        auto o=*store.objectByUuid(w,f.id);o.fiber->family="V";store.upsertObject(w,o);
        vc::geometry::Fiber fiber;fiber.controlPoints={{10000,0,0},{10001,0,0}};
        store.putGeometry(w,f.id,vc::geometry::encode(fiber),f.revision);w.commit();
    });
    writer.get();
    const auto after=f.store.fiberRegion(read,all());
    REQUIRE(after.runs.size()==1);CHECK(after.runs[0].revisionUuid==f.revision);
    CHECK(after.runs[0].family=="");CHECK(after.runs[0].points==before.runs[0].points);
    read.commit();auto fresh=f.store.read();CHECK(f.store.fiberRegion(fresh,all()).runs.empty());
}
TEST_CASE("public clipping primitive handles tangent zero-length and crossing segments") {
    const std::array<double,6> box{-1,-1,-1,1,1,1};double first,last;
    CHECK(vc::geometry::clipSegment({-2,0,0},{2,0,0},box,first,last));
    CHECK(first==0.25);CHECK(last==0.75);
    CHECK(vc::geometry::clipSegment({1,1,1},{1,1,1},box,first,last));
    CHECK_FALSE(vc::geometry::clipSegment({2,2,2},{2,2,2},box,first,last));
}
TEST_CASE("priority and region cursors stream without SQLite sort operations") {
    Fixture f;
    {auto w=f.store.write();
        for(int i=0;i<20;++i) {
            ObjectRecord o;o.kind="fiber";auto id=f.store.upsertObject(w,o);
            vc::geometry::Fiber fiber;fiber.controlPoints={{10000,0,0},{10001,0,0}};
            f.store.putGeometry(w,id,vc::geometry::encode(fiber));
        }
        w.commit();}
    auto r=f.store.read();auto* db=ObjectStoreImplAccess::conn(r);int sorts=0;
    struct TraceGuard {sqlite3* db;~TraceGuard(){sqlite3_trace_v2(db,0,nullptr,nullptr);}} guard{db};
    sqlite3_trace_v2(db,SQLITE_TRACE_PROFILE,[](unsigned,void* context,void* statement,void*) {
        *static_cast<int*>(context)+=sqlite3_stmt_status(static_cast<sqlite3_stmt*>(statement),SQLITE_STMTSTATUS_SORT,1);
        return 0;
    },&sorts);
    const auto result=f.store.fiberRegion(r,all());
    CHECK(result.complete());CHECK(result.intersectingFibers==1);CHECK(sorts==0);
}
