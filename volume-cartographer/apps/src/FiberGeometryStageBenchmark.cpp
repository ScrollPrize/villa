#include "FiberGeometryStageBenchmark.hpp"
#include <openssl/evp.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

nlohmann::json benchmarkFiberGeometryStage(
    const std::vector<vc::fiber_tracer::FiberletCropTraceLine>& lines,
    vc::fiber_tracer::FiberTraceConstraintConfig options,const nlohmann::json& cfg)
{
    using namespace vc::fiber_tracer;
    using Clock=std::chrono::steady_clock;using Json=nlohmann::json;
    const auto seconds=[](auto start){return std::chrono::duration<double>(Clock::now()-start).count();};
    Json runs=Json::array();std::string reference;
    for(const auto& schedule:cfg.value("geometry_schedules",std::vector<std::string>{"global","pair"})) {
        if(schedule!="global"&&schedule!="pair")throw std::invalid_argument("Unknown geometry schedule");
        options.pairOwnedGeometry=schedule=="pair";
        FiberTraceConstraintStream stream;
        const auto solving=Clock::now();const auto cpuStart=std::clock();
        double geometrySeconds=0.,geometryCpu=0.,verificationSeconds=0.;
        size_t events=0,parallel=0;std::string digest;
        stream.geometryReady=[&](auto,auto positions,const auto&){events=positions.size();};
        stream.inspectGeometry=[&](std::span<const FiberTraceConstraint* const> records) {
            const auto begin=Clock::now();
            for(auto* c:records)parallel+=c->parallelScore>c->perpendicularScore;
            if(cfg.value("verify_geometry",true)) {
                std::vector<const FiberTraceConstraint*> ordered(records.begin(),records.end());
                std::sort(ordered.begin(),ordered.end(),[](auto* a,auto* b) {
                    return std::tie(a->pieceA,a->pieceB,a->arcABaseVoxels,a->arcBBaseVoxels,a->parallelScore,a->perpendicularScore)<
                        std::tie(b->pieceA,b->pieceB,b->arcABaseVoxels,b->arcBBaseVoxels,b->parallelScore,b->perpendicularScore);
                });
                std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> hash(EVP_MD_CTX_new(),EVP_MD_CTX_free);
                if(!hash||EVP_DigestInit_ex(hash.get(),EVP_sha256(),nullptr)!=1)throw std::runtime_error("SHA256 initialization failed");
                for(const auto* c:ordered) {
                    const std::array<double,17> numeric{c->arcABaseVoxels,c->arcBBaseVoxels,
                        c->pointABaseXYZ[0],c->pointABaseXYZ[1],c->pointABaseXYZ[2],
                        c->pointBBaseXYZ[0],c->pointBBaseXYZ[1],c->pointBBaseXYZ[2],
                        c->closestDistanceBaseVoxels,c->parallelScore,c->perpendicularScore,c->evidenceWeight,
                        c->supportABegin,c->supportAEnd,c->supportBBegin,c->supportBEnd,double(c->localOrientation)};
                    // Explicit little-endian scalar fields, never struct padding.
                    std::array<unsigned char,19*8> bytes{};
                    const auto put=[&](size_t slot,uint64_t value){for(size_t k=0;k<8;++k)bytes[slot*8+k]=static_cast<unsigned char>(value>>(8*k));};
                    put(0,c->pieceA);put(1,c->pieceB);
                    for(size_t k=0;k<numeric.size();++k)put(k+2,std::bit_cast<uint64_t>(numeric[k]));
                    if(EVP_DigestUpdate(hash.get(),bytes.data(),bytes.size())!=1)throw std::runtime_error("SHA256 update failed");
                }
                std::array<unsigned char,32> bytes;unsigned count=0;
                if(EVP_DigestFinal_ex(hash.get(),bytes.data(),&count)!=1||count!=bytes.size())throw std::runtime_error("SHA256 finalization failed");
                std::ostringstream out;for(auto value:bytes)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(value);
                digest=out.str();
                if(reference.empty())reference=digest;
                if(digest!=reference)throw std::runtime_error("Normalized geometry differs between schedules");
            }
            verificationSeconds=seconds(begin);
        };
        std::cout<<"GEOMETRY_BEGIN "<<schedule<<std::endl;
        const auto report=extractFiberTraceConstraints(lines,options,
            [](const auto&,const auto&,double)->double{throw std::logic_error("Geometry-only benchmark attempted prediction sampling");},
            {},{},nullptr,[&](const auto& p) {
                if(p.completed==0 || p.completed==p.total)
                    std::cout<<"GEOMETRY_PROGRESS "<<Json{{"schedule",schedule},{"phase",p.phase},
                        {"completed",p.completed},{"total",p.total},{"seconds",seconds(solving)}}.dump()<<std::endl;
                if(p.phase=="geometry_complete") {
                    geometrySeconds=seconds(solving);geometryCpu=double(std::clock()-cpuStart)/CLOCKS_PER_SEC;
                }
            },stream);
        Json row={{"schedule",schedule},{"threads",options.parallelThreads},{"geometry_seconds",geometrySeconds},
            {"geometry_cpu_seconds",geometryCpu},{"effective_parallelism",geometryCpu/std::max(geometrySeconds,1e-9)},
            {"verification_seconds",verificationSeconds},{"teardown_seconds",seconds(solving)-geometrySeconds-verificationSeconds},
            {"initial_candidates",report.measuredCandidates},{"refined_candidates",report.localRefinedCandidates},
            {"events",events},{"parallel",parallel},{"perpendicular",events-parallel},{"geometry_sha256",digest},
            {"volume_reads",0},{"import_rows",0}};
        runs.push_back(row);std::cout<<"GEOMETRY_CASE "<<row.dump()<<std::endl;
    }
    return runs;
}
