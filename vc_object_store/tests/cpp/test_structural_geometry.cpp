#define VC_TEST_MAIN
#include "vc_test.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/Uuid.hpp"
#include "geometry/GeometryReader.hpp"
#include "geometry/ArrayBytes.hpp"
using namespace vc::objects;
namespace {
struct Fixture {
    std::filesystem::path path=std::filesystem::temp_directory_path()/("vc-structure-"+util::newUuid());
    ObjectStore store=ObjectStore::create(path,OpenOptions{.author="test"});
    ~Fixture() { for (auto suffix:{"","-wal","-shm"}) { std::error_code e; std::filesystem::remove(path.string()+suffix,e); } }
    RevisionSelection expected(const std::string& object) { auto r=store.read(); return {{object,store.currentObjectRevision(r,object)->uuid}}; }
    int64_t scalar(const char* sql) {auto r=store.read();detail::Statement s(ObjectStoreImplAccess::conn(r),sql);s.row();return s.i64(0);}
};
}
TEST_CASE("structural snapshots restart compact delta history on a million-point surface") {
    Fixture f;
    constexpr int64_t size=1000;
    std::vector<float> points(size*size*3);for(size_t i=0;i<points.size();++i)points[i]=float(i+1);
    GeometryData data;data.kind="patch";data.attributes=R"({"scale":[1,1]})";
    data.arrays={detail::scalarArray<float>("points","<f4",{size,size,3},points)};
    auto w=f.store.write();ObjectRecord object;object.kind=data.kind;
    const auto id=f.store.upsertObject(w,object);auto revision=f.store.putGeometry(w,id,data);w.commit();const auto first=revision;
    GeometrySurfaceResize resize{0,0,size,size+1,{detail::scalarArray<float>("points","<f4",{1,1,3},std::array<float,3>{-1,-1,-1})}};
    {auto tx=f.store.write(f.expected(id));revision=f.store.resizeSurface(tx,id,resize,revision);tx.commit();}
    resize.columns=size;
    {auto tx=f.store.write(f.expected(id));revision=f.store.resizeSurface(tx,id,resize,revision);tx.commit();}
    const auto base=revision;const auto chunks=f.scalar("SELECT count(*) FROM geometry_chunks");
    f.store.checkpoint();const auto initialBytes=std::filesystem::file_size(f.path);
    std::string middle;std::vector<uint8_t> middleBytes;
    for(int i=0;i<1000;++i) {
        const auto point=(i*7919LL)%(size*size);
        const auto values=std::array<float,3>{float(-i-2),float(i+2),float(i+3)};
        auto tx=f.store.write();revision=f.store.editGeometry(tx,id,{{"points",point*3,detail::scalarBytes<float>(values)}},revision);tx.commit();
        std::copy(values.begin(),values.end(),points.begin()+point*3);
        if(i==499){middle=revision;middleBytes=detail::scalarBytes<float>(points);}
    }
    CHECK(f.scalar("SELECT count(*) FROM geometry_chunks")==chunks);
    CHECK(f.scalar("SELECT count(*) FROM geometry_deltas")==1000);
    CHECK(f.scalar("SELECT sum(length(data)) FROM geometry_deltas")<1024*1024);
    f.store.checkpoint();CHECK(std::filesystem::file_size(f.path)-initialBytes<32*1024*1024);
    {auto tx=f.store.read();detail::GeometryReader reader(tx,revision);
        CHECK(reader.manifest().at("arrays").at("points").contains("chunks"));
        CHECK(f.store.readGeometryArray(tx,revision,"points")==detail::scalarBytes<float>(points));
        CHECK(f.store.readGeometryArray(tx,middle,"points")==middleBytes);
        CHECK(f.store.readGeometryArray(tx,base,"points")==data.arrays[0].bytes);
        CHECK(f.store.readGeometryArray(tx,first,"points")==data.arrays[0].bytes);}
    // A checkpoint after a long delta chain preserves it and starts a new base.
    {auto tx=f.store.write(f.expected(id));revision=f.store.resizeSurface(tx,id,resize,revision);tx.commit();}
    const auto checkpoint=revision;
    {auto tx=f.store.write();revision=f.store.editGeometry(tx,id,{{"points",0,detail::scalarBytes<float>(std::array<float,1>{42})}},revision);tx.commit();}
    {auto tx=f.store.read();detail::Statement s(ObjectStoreImplAccess::conn(tx),"SELECT manifest FROM geometry_revisions WHERE uuid=?");s.bindBlob(1,detail::parseUuid(revision));s.row();
        CHECK(nlohmann::json::parse(s.text(0)).at("delta").at("base")==checkpoint);
        CHECK(f.store.readGeometryArray(tx,checkpoint,"points")==detail::scalarBytes<float>(points));}
}
TEST_CASE("sequence snapshots reuse complete chunks and preserve stable identities through deltas") {
    Fixture f;constexpr int64_t count=100000;
    std::vector<double> xyz(count*3,1);std::vector<int64_t> ids(count);for(int64_t i=0;i<count;++i){ids[i]=i;xyz[i*3]=double(i);}
    GeometryData data;data.kind="point_collection";
    data.arrays={detail::scalarArray<double>("points","<f8",{count,3},xyz),detail::scalarArray<int64_t>("point_ids","<i8",{count},ids)};
    auto w=f.store.write();ObjectRecord object;object.kind=data.kind;const auto id=f.store.upsertObject(w,object);
    auto revision=f.store.putGeometry(w,id,data);w.commit();
    GeometrySequenceEdit edit;edit.array="points";edit.start=count;
    GeometrySequencePart part;part.count=1;part.ids={count};part.bytes=detail::scalarBytes<double>(std::array<double,3>{-5,2,3});edit.parts={part};
    detail::GeometryWork work;
    {detail::GeometryWorkScope scope(work);auto tx=f.store.write(f.expected(id));revision=f.store.editGeometrySequence(tx,id,edit,revision);tx.commit();}
    CHECK(work.writtenBytes["points"]==xyz.size()*8%(1024*1024)+24);
    const auto snapshot=revision;
    {auto tx=f.store.write();revision=f.store.editGeometry(tx,id,{{"points",count*3,detail::scalarBytes<double>(std::array<double,1>{-7})}},revision);tx.commit();}
    {auto tx=f.store.read();CHECK(f.store.readGeometryArray(tx,snapshot,"points",count*24,24)==part.bytes);
        CHECK(f.store.readGeometryArray(tx,revision,"points",count*24,8)==detail::scalarBytes<double>(std::array<double,1>{-7}));
        ObjectSelection selection;selection.bbox=std::array<double,6>{-8,1,2,-6,3,4};CHECK_FALSE(f.store.spatialRegions(tx,selection).empty());}
    CHECK(f.scalar("SELECT count(*) FROM geometry_deltas")==1);
}
