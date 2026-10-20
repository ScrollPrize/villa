#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/Uuid.hpp"
#include "geometry/GeometryInternal.hpp"
#include "geometry/ArrayBytes.hpp"
using namespace vc::objects;
namespace {
struct Fixture {
    std::filesystem::path path=std::filesystem::temp_directory_path()/("vc-structure-"+util::newUuid());
    ObjectStore store=ObjectStore::create(path,OpenOptions{.author="test"});
    ~Fixture() { for (auto suffix:{"","-wal","-shm"}) { std::error_code e; std::filesystem::remove(path.string()+suffix,e); } }
    RevisionSelection expected(const std::string& object) { auto r=store.read(); return {{object,store.currentObjectRevision(r,object)->uuid}}; }
};
}
TEST_CASE("structural append reuses coordinate payloads and compaction is bounded") {
    Fixture f;
    constexpr int64_t count=500000;
    std::vector<double> xyz(count*3,1); std::vector<int64_t> ids(count);
    for (int64_t i=0;i<count;++i) { ids[i]=i; xyz[3*i]=double(i); }
    GeometryData data; data.kind="point_collection";
    data.arrays={detail::scalarArray<double>("points","<f8",{count,3},xyz),detail::scalarArray<int64_t>("point_ids","<i8",{count},ids)};
    auto write=f.store.write(); ObjectRecord object; object.kind=data.kind;
    const auto id=f.store.upsertObject(write,object); auto revision=f.store.putGeometry(write,id,data); write.commit();
    GeometrySequenceEdit edit; edit.array="points"; edit.start=count;
    // Explicit first structural conversion establishes identity/annotation state
    // and block bounds. Measure subsequent appends, not that one-time cost.
    { auto txn=f.store.write(f.expected(id)); revision=f.store.editGeometrySequence(txn,id,edit,revision); txn.commit(); }
    GeometrySequencePart part; part.count=1; part.ids={count}; part.bytes=detail::scalarBytes<double>(std::array<double,3>{-5,2,3}); edit.parts={part};
    detail::GeometryWork work;
    { detail::GeometryWorkScope scope(work); auto txn=f.store.write(f.expected(id)); revision=f.store.editGeometrySequence(txn,id,edit,revision); txn.commit(); }
    CHECK(work.writtenBytes["points"]==24);
    CHECK(work.hashedBytes["points"]<4*1024*1024);
    CHECK(work.readBytes["points"]<4*1024*1024);
    CHECK(work.writtenBytes["point_ids"]<=4096);
    CHECK(work.identityRows<=4);
    CHECK(work.manifestBytesWritten<8192);
    CHECK(work.manifestEntries<160);
    CHECK(work.spatialRows<=2);
    CHECK(work.readBytes["@tree"]<256*1024);
    auto txn=f.store.read();
    CHECK(f.store.readGeometryArray(txn,revision,"points",count*24,24)==part.bytes);
    ObjectSelection selection; selection.bbox=std::array<double,6>{-6,1,2,-4,3,4};
    const auto regions=f.store.spatialRegions(txn,selection);
    CHECK_FALSE(regions.empty());
}
TEST_CASE("surface second growth touches boundary tiles only") {
    Fixture f;
    constexpr int64_t size=512;
    GeometryData data; data.kind="patch"; data.attributes=R"({"scale":[1,1]})";
    data.arrays={detail::scalarArray<float>("points","<f4",{size,size,3},std::vector<float>(size*size*3,1))};
    auto write=f.store.write(); ObjectRecord object; object.kind=data.kind;
    const auto id=f.store.upsertObject(write,object); auto revision=f.store.putGeometry(write,id,data,{}, {},GeometryStorageOptions{32}); write.commit();
    GeometrySurfaceResize resize{0,0,size,size+1,{detail::scalarArray<float>("points","<f4",{1,1,3},std::array<float,3>{-1,-1,-1})}};
    { auto txn=f.store.write(f.expected(id)); revision=f.store.resizeSurface(txn,id,resize,revision); txn.commit(); }
    resize.columns++;
    detail::GeometryWork work;
    { detail::GeometryWorkScope scope(work); auto txn=f.store.write(f.expected(id)); revision=f.store.resizeSurface(txn,id,resize,revision); txn.commit(); }
    CHECK(work.readBytes["points"]<size*32*12);
    CHECK(work.hashedBytes["points"]<size*32*12);
    auto txn=f.store.read();
    CHECK(f.store.readGeometryRectangle(txn,revision,"points",0,0,1,1)==detail::scalarBytes<float>(std::array<float,3>{1,1,1}));
}
TEST_CASE("indexed insertion scales with paths, annotations and affected relationships") {
    for(const int64_t count:{1000,100000}) {
        Fixture f;GeometryData data;data.kind="point_collection";
        std::vector<double> xyz(count*3,1);std::vector<int64_t> ids(count);
        nlohmann::json attrs=nlohmann::json::array();
        for(int64_t i=0;i<count;++i){ids[i]=i;xyz[i*3]=double(i);attrs.push_back({{"label",std::to_string(i)}});}
        data.arrays={detail::scalarArray<double>("points","<f8",{count,3},xyz),detail::scalarArray<int64_t>("point_ids","<i8",{count},ids)};
        auto write=f.store.write();ObjectRecord object;object.kind=data.kind;object.meta=nlohmann::json{{"source",{{"point_attributes",attrs}}}}.dump();
        const auto id=f.store.upsertObject(write,object);auto revision=f.store.putGeometry(write,id,data);write.commit();
        GeometrySequenceEdit edit;edit.array="points";edit.start=count/2;
        {auto txn=f.store.write(f.expected(id));revision=f.store.editGeometrySequence(txn,id,edit,revision);txn.commit();}
        {auto txn=f.store.write();ObjectRecord note;note.kind="note";const auto peer=f.store.upsertObject(txn,note);
            for(int64_t i=0;i<128;++i) {EdgeRecord edge;edge.kind="connected";edge.aUuid=id;edge.bUuid=peer;edge.aAnchor.points={{"point_ids",{i}}};edge.bAnchor.wholeGeometry=true;f.store.addEdge(txn,edge);}txn.commit();}
        GeometrySequencePart part;part.count=1;part.ids={count};part.bytes=detail::scalarBytes<double>(std::array<double,3>{-1,2,3});part.annotations=R"([{"label":"new"}])";edit.parts={part};
        detail::GeometryWork work;
        {detail::GeometryWorkScope scope(work);auto txn=f.store.write(f.expected(id));revision=f.store.editGeometrySequence(txn,id,edit,revision);txn.commit();}
        CHECK(work.identityRows<=4);
        CHECK(work.relationshipRows==0); // 128 unchanged ID links are not visited.
        CHECK(work.spatialRows<=2);
        CHECK(work.manifestBytesRead<64*1024);
        CHECK(work.manifestBytesWritten<8192);
        CHECK(work.manifestEntries<200);
        CHECK(work.readBytes["point_ids"]<128*1024);
        CHECK(work.readBytes["points"]<512*1024);
        CHECK(work.readBytes["@tree"]<256*1024);
        CHECK(work.metadataBytes<1024);
        auto txn=f.store.read();CHECK(f.store.readGeometryArray(txn,revision,"points",count/2*24,24)==part.bytes);
    }
}
TEST_CASE("right growth of a long one-row grid does not visit the unchanged width") {
    for(const int64_t width:{128,16384}) {
        Fixture f;GeometryData data;data.kind="patch";data.attributes=R"({"scale":[1,1]})";
        data.arrays={detail::scalarArray<float>("points","<f4",{1,width,3},std::vector<float>(width*3,1))};
        auto write=f.store.write();ObjectRecord object;object.kind=data.kind;const auto id=f.store.upsertObject(write,object);
        auto revision=f.store.putGeometry(write,id,data,{}, {},GeometryStorageOptions{32});write.commit();
        GeometrySurfaceResize resize{0,0,1,width,{detail::scalarArray<float>("points","<f4",{1,1,3},std::array<float,3>{2,2,2})}};
        {auto txn=f.store.write(f.expected(id));revision=f.store.resizeSurface(txn,id,resize,revision);txn.commit();}
        ++resize.columns;detail::GeometryWork work;
        {detail::GeometryWorkScope scope(work);auto txn=f.store.write(f.expected(id));revision=f.store.resizeSurface(txn,id,resize,revision);txn.commit();}
        CHECK(work.spatialRows<=2);
        CHECK(work.manifestBytesWritten<4096);
        CHECK(work.manifestEntries<100);
        CHECK(work.readBytes["@tree"]<128*1024);
        CHECK(work.readBytes["points"]<4096);
    }
}
