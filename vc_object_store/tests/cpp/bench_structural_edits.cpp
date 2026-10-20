// Synthetic scaling/SQL-profile harness. No external datasets or dependencies.
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/Uuid.hpp"
#include "geometry/SharedGeometry.hpp"
#include "geometry/ArrayBytes.hpp"
#include <chrono>
#include <iostream>
#include <numeric>

using namespace vc::objects;
using Json=nlohmann::json;
namespace {
struct Profile {
    uint64_t steps=0;
    std::map<std::string,uint64_t> ns;
    static int callback(unsigned,void* context,void* statement,void* duration) {
        auto& self=*static_cast<Profile*>(context); auto* s=static_cast<sqlite3_stmt*>(statement);
        self.steps+=sqlite3_stmt_status(s,SQLITE_STMTSTATUS_VM_STEP,1);
        self.ns[sqlite3_sql(s)]+=*static_cast<sqlite3_uint64*>(duration); return 0;
    }
};
struct Store {
    std::filesystem::path path=std::filesystem::temp_directory_path()/("vc-structure-bench-"+util::newUuid());
    ObjectStore db=ObjectStore::create(path,OpenOptions{.author="benchmark"});
    ~Store() { for (auto suffix:{"","-wal","-shm"}) { std::error_code e; std::filesystem::remove(path.string()+suffix,e); } }
    RevisionSelection heads(const std::vector<std::string>& ids) {
        RevisionSelection out; auto r=db.read(); for (const auto& id:ids) out[id]=db.currentObjectRevision(r,id)->uuid; return out;
    }
};
struct Sample { double ms; detail::GeometryWork work; Profile sql; };
template<class F> Sample measure(Store& store,const std::vector<std::string>& ids,F&& edit) {
    // Include expected-head reads and the entire checked commit, including
    // metadata/history publication, dependencies and SQLite durability.
    Sample result; const auto start=std::chrono::steady_clock::now();
    detail::GeometryWorkScope scope(result.work);
    auto txn=store.db.write(store.heads(ids)); auto* connection=ObjectStoreImplAccess::conn(txn);
    sqlite3_trace_v2(connection,SQLITE_TRACE_PROFILE,Profile::callback,&result.sql);
    try { edit(txn); txn.commit(); }
    catch (...) { sqlite3_trace_v2(connection,0,nullptr,nullptr); throw; }
    sqlite3_trace_v2(connection,0,nullptr,nullptr);
    result.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(); return result;
}
void emit(const std::string& operation,int64_t size,bool structural,int links,const std::vector<Sample>& samples) {
    std::vector<double> times; double sum=0; Json counters=Json::array(); std::map<std::string,uint64_t> sql;
    for (const auto& s:samples) {
        times.push_back(s.ms); sum+=s.ms;
        counters.push_back({{"phase_ns",s.work.phaseNanoseconds},{"read",s.work.readBytes},{"hashed",s.work.hashedBytes},{"written",s.work.writtenBytes},
            {"manifest_bytes_read",s.work.manifestBytesRead},{"manifest_bytes_written",s.work.manifestBytesWritten},{"metadata_bytes",s.work.metadataBytes},{"manifest_entries",s.work.manifestEntries},{"identity_rows",s.work.identityRows},{"relationship_rows",s.work.relationshipRows},
            {"spatial_rows",s.work.spatialRows},{"compactions",s.work.compactions},{"sql_vm_steps",s.sql.steps}});
        for (const auto& [query,time]:s.sql.ns) sql[query]+=time;
    }
    std::sort(times.begin(),times.end());
    std::vector<std::pair<uint64_t,std::string>> hot; for (const auto& [query,time]:sql) hot.emplace_back(time,query);
    std::sort(hot.rbegin(),hot.rend()); Json profile=Json::array();
    for (size_t i=0;i<std::min<size_t>(5,hot.size());++i) profile.push_back({{"ms",hot[i].first/1e6},{"sql",hot[i].second}});
    std::cout<<Json{{"operation",operation},{"size",size},{"structural",structural},{"links",links},{"iterations",samples.size()},
        {"mean_ms",sum/times.size()},{"min_ms",times.front()},{"median_ms",times[times.size()/2]},{"max_ms",times.back()},
        {"work",counters},{"sql_profile",profile}}.dump()<<'\n';
}
void sequence(int64_t count,bool structural,bool insert,int links,int iterations) {
    Store store; GeometryData data; data.kind="point_collection";
    std::vector<double> points(count*3,1); std::vector<int64_t> ids(count); std::iota(ids.begin(),ids.end(),0);
    for (int64_t i=0;i<count;++i) points[i*3]=double(i);
    auto geometry=[&] { data.arrays={detail::scalarArray<double>("points","<f8",{int64_t(ids.size()),3},points),
        detail::scalarArray<int64_t>("point_ids","<i8",{int64_t(ids.size())},ids)}; };
    geometry(); auto write=store.db.write(); ObjectRecord object; object.kind=data.kind;
    const auto uuid=store.db.upsertObject(write,object); auto revision=store.db.putGeometry(write,uuid,data); write.commit();
    std::vector<std::string> endpoints{uuid};
    if (structural) {
        const auto conversion=measure(store,endpoints,[&](WriteTxn& txn) {GeometrySequenceEdit edit;edit.array="points";revision=store.db.editGeometrySequence(txn,uuid,edit,revision);});
        emit("sequence_initialize",count,true,0,{conversion});
    }
    { auto txn=store.db.write(); ObjectRecord note; note.kind="note";
      for (int i=0;i<links;++i) { const auto peer=store.db.upsertObject(txn,note); endpoints.push_back(peer);
        EdgeRecord e; e.kind="connected"; e.aUuid=uuid; e.bUuid=peer;
        e.aAnchor.points={{"point_ids",{i}}}; e.bAnchor.wholeGeometry=true; store.db.addEdge(txn,e); } txn.commit(); }
    std::vector<Sample> samples;
    for (int i=0;i<iterations;++i) {
        const int64_t at=insert ? count/2 : count+i;
        samples.push_back(measure(store,endpoints,[&](WriteTxn& txn) {
            if (structural) { GeometrySequenceEdit edit; edit.array="points"; edit.start=at;
                GeometrySequencePart part; part.count=1; part.ids={count+i}; part.bytes=detail::scalarBytes<double>(std::array<double,3>{double(count+i),2,3});
                edit.parts={part}; revision=store.db.editGeometrySequence(txn,uuid,edit,revision);
            } else { ids.insert(ids.begin()+at,count+i); points.insert(points.begin()+at*3,{double(count+i),2,3}); geometry(); revision=store.db.putGeometry(txn,uuid,data,revision); }
        }));
    }
    emit(insert ? "insert" : "append",count,structural,links,samples);
}
void surface(int64_t size,bool structural,int links,int iterations) {
    Store store; GeometryData data; data.kind="patch"; data.attributes=R"({"scale":[1,1]})";
    const int64_t rows=size; int64_t cols=size;
    std::vector<float> xyz(rows*cols*3,1); std::vector<uint16_t> channel(rows*cols,5);
    auto geometry=[&] { data.arrays={detail::scalarArray<float>("points","<f4",{rows,cols,3},xyz),detail::scalarArray<uint16_t>("generations","<u2",{rows,cols},channel)}; };
    geometry(); auto write=store.db.write(); ObjectRecord object; object.kind=data.kind;
    const auto uuid=store.db.upsertObject(write,object); auto revision=store.db.putGeometry(write,uuid,data,{}, {},GeometryStorageOptions{32}); write.commit();
    std::vector<std::string> endpoints{uuid};
    GeometrySurfaceResize resize{0,0,rows,cols,{detail::scalarArray<float>("points","<f4",{1,1,3},std::array<float,3>{1,1,1}),
        detail::scalarArray<uint16_t>("generations","<u2",{1,1},std::array<uint16_t,1>{5})}};
    if (structural) {
        const auto conversion=measure(store,endpoints,[&](WriteTxn& txn) {revision=store.db.resizeSurface(txn,uuid,resize,revision);});
        emit("surface_initialize",size,true,0,{conversion});
    }
    { auto txn=store.db.write(); ObjectRecord note; note.kind="note";
      for (int i=0;i<links;++i) { const auto peer=store.db.upsertObject(txn,note); endpoints.push_back(peer);
        EdgeRecord e; e.kind="connected"; e.aUuid=uuid; e.bUuid=peer;
        e.aAnchor.points={{"points",{i,0}}}; e.bAnchor.wholeGeometry=true; store.db.addEdge(txn,e); } txn.commit(); }
    std::vector<Sample> samples;
    for (int i=0;i<iterations;++i) {
        // Positional patch anchors conservatively require an explicit decision
        // on any shape change, even when the referenced pixel stays in place.
        std::vector<GeometryEdgeEdit> decisions;
        { auto txn=store.db.read(); for (const auto& e:store.db.edgesOf(txn,uuid,true)) {
            GeometryEdgeEdit d; d.edgeUuid=e.uuid; d.expectedGeneration=e.generation; d.action="retain";
            if (auto r=store.db.currentGeometry(txn,e.aUuid)) d.aRevision=r->uuid;
            if (auto r=store.db.currentGeometry(txn,e.bUuid)) d.bRevision=r->uuid; decisions.push_back(d);
        } }
        samples.push_back(measure(store,endpoints,[&](WriteTxn& txn) {
            ++cols;
            if (structural) { resize.columns=cols; revision=store.db.resizeSurface(txn,uuid,resize,revision,decisions); }
            else { xyz.resize(rows*cols*3,1); channel.resize(rows*cols,5); geometry(); revision=store.db.putGeometry(txn,uuid,data,revision,decisions,GeometryStorageOptions{32}); }
        }));
    }
    emit("right_growth",size,structural,links,samples);
}
}
int main(int argc,char** argv) {
    try {
        const int iterations=argc>1 ? std::stoi(argv[1]) : 3;
        if (iterations<=0) throw std::invalid_argument("iterations must be positive");
        for (bool structural:{false,true}) for (int links:{0,8}) {
            for (int64_t size:{1000,10000,100000}) for (bool insert:{false,true}) sequence(size,structural,insert,links,iterations);
            for (int64_t size:{128,512,1024}) surface(size,structural,links,iterations);
        }
        for (bool structural:{false,true}) sequence(1000,structural,false,0,80);
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
