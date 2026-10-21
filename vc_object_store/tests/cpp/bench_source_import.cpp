// Synthetic import scaling, payload-work checks and scoped native profiling.
#include "vc_object_store/objects/GeometryIngest.hpp"
#include "vc_object_store/Uuid.hpp"
#include "geometry/GeometryInternal.hpp"
#include "geometry/ArrayBytes.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
using namespace vc::objects;
using Json=nlohmann::json;
int main(int argc,char** argv) {
    const int iterations=argc>1?std::stoi(argv[1]):5;
    if(iterations<1) return 2;
    for(int collections:{0,20,100}) for(bool linked:{false,true}) {
        const auto folder=std::filesystem::temp_directory_path()/("vc-source-bench-"+util::newUuid());
        std::filesystem::create_directories(folder);
        struct Cleanup { std::filesystem::path path; ~Cleanup() { std::error_code error; std::filesystem::remove_all(path,error); } } cleanup{folder};
        auto db=ObjectStore::create(folder/"store.vcobj",OpenOptions{.author="benchmark"});
        const auto file=folder/"source.json";
        auto save=[&](const Json& entries) { std::ofstream(file)<<Json{{"vc_pointcollections_json_version","1"},{"collections",entries}}; };
        Json initial=Json::object();
        for(int i=0;i<collections;++i) {
            Json points=Json::object();
            for(int j=0;j<1000;++j) points[std::to_string(i*1000+j)]={{"p",{j,2,3}}};
            initial[std::to_string(i)]={{"uuid",util::newUuid()},{"points",points}};
        }
        if(collections) { save(initial); const auto r=ingestGeometrySource(db,"point_collection",file); if(!r.problems.empty()) throw std::runtime_error(r.problems[0]); }
        Json samples=Json::array(); std::vector<double> times;
        for(int i=0;i<iterations;++i) {
            Json p={{"p",{1,2,3}}}; if(linked && collections) p["links"]={0};
            save(Json{{std::to_string(collections+i),{{"uuid",util::newUuid()},{"points",{{std::to_string((collections+i)*1000),p}}}}}});
            detail::GeometryWork work;
            const auto start=std::chrono::steady_clock::now();
            GeometryIngestReport report;
            { detail::GeometryWorkScope scope(work); report=ingestGeometrySource(db,"point_collection",file); }
            const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(); times.push_back(ms);
            if(!report.problems.empty()) throw std::runtime_error(report.problems[0]);
            // One incoming point and a target chunk, not all 100,000 unrelated points/IDs.
            if(work.readBytes["points"]>100000 || work.readBytes["point_ids"]>10000 || work.identityRows>20)
                throw std::runtime_error("import exceeded bounded identity/payload work budget");
            samples.push_back({{"ms",ms},{"read_bytes",work.readBytes},{"identity_rows",work.identityRows},{"phase_ns",work.phaseNanoseconds}});
        }
        if(collections==100 && !linked) {
            const auto object=initial.at("0").at("uuid").get<std::string>();
            std::string revision;
            auto expected=[&] { auto r=db.read(); revision=db.currentGeometry(r,object)->uuid; return RevisionSelection{{object,db.currentObjectRevision(r,object)->uuid}}; };
            GeometrySequenceEdit edit; edit.array="points"; edit.start=1000;
            { auto txn=db.write(expected()); db.editGeometrySequence(txn,object,edit,revision); txn.commit(); }
            GeometrySequencePart part; part.count=1; part.ids={1000000};
            part.bytes=detail::scalarBytes<double>(std::array<double,3>{10,2,3}); edit.parts={part};
            detail::GeometryWork work;
            { detail::GeometryWorkScope scope(work); auto txn=db.write(expected()); db.editGeometrySequence(txn,object,edit,revision); txn.commit(); }
            // A structural snapshot scans its 1,000-point collection, not the whole document.
            if(work.identityRows>5000 || work.readBytes["point_ids"]>65536)
                throw std::runtime_error("structural snapshot scanned unrelated source identities");
        }
        std::sort(times.begin(),times.end());
        std::cout<<Json{{"existing_collections",collections},{"existing_points",collections*1000},{"linked",linked},{"iterations",iterations},
            {"mean_ms",std::accumulate(times.begin(),times.end(),0.0)/iterations},{"min_ms",times.front()},{"median_ms",times[times.size()/2]},{"max_ms",times.back()},
            {"samples",samples}}.dump()<<'\n';
    }
}
