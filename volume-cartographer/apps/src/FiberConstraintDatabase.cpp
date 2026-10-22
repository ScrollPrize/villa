#include "FiberConstraintDatabase.hpp"
#include "vc/fiber_tracer/FiberNormalOrientation.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"
#include "vc/core/util/AtomicFile.hpp"
#include <sqlite3.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <unordered_map>
#include <limits>
#include <fstream>
#include <array>
#include <bit>

namespace {
using Json = nlohmann::json;
Json point(const cv::Vec3d& p) { return {p[0],p[1],p[2]}; }
Json measurement(const vc::fiber_tracer::FiberTraceConstraint& c) {
    Json j;
#define FIELD(x) j[#x]=c.x
    FIELD(pieceA); FIELD(pieceB); FIELD(arcABaseVoxels); FIELD(arcBBaseVoxels);
    j["pointABaseXYZ"]=point(c.pointABaseXYZ); j["pointBBaseXYZ"]=point(c.pointBBaseXYZ);
    FIELD(closestDistanceBaseVoxels); FIELD(parallelScore); FIELD(perpendicularScore);
    FIELD(windingDistance); FIELD(hardContinuity); FIELD(artificialTileSeam);
    FIELD(allowMeasurementDisable); FIELD(evidenceWeight); FIELD(localEvent); FIELD(localOrientation);
    FIELD(supportABegin); FIELD(supportAEnd); FIELD(supportBBegin); FIELD(supportBEnd);
    FIELD(parallelWindingDistance); FIELD(parallelCorrespondenceSamples);
    FIELD(parallelMeanAdvanceResidualFraction); FIELD(parallelMeanConnectorTangentAbsDot);
    FIELD(parallelMeanConnectorLengthChangeFraction); FIELD(parallelMeanConnectorDirectionChange);
    FIELD(parallelLimitHitFraction);
#undef FIELD
#define OPTIONAL(x) j[#x]=c.x ? Json(*c.x) : Json(nullptr)
    OPTIONAL(signedWindingDelta); OPTIONAL(perpendicularNormalAlignment); OPTIONAL(windingNormalComponent);
    OPTIONAL(signedParallelWindingDelta); OPTIONAL(parallelNormalAlignment);
#undef OPTIONAL
    return j;
}
}

void verifyFiberConstraintDatabaseMeasurements(const std::filesystem::path& path,
    const vc::fiber_tracer::FiberTraceConstraintReport& report)
{
    sqlite3* raw=nullptr;
    const auto name=path.u8string();
    const int opened=sqlite3_open_v2(reinterpret_cast<const char*>(name.c_str()),&raw,SQLITE_OPEN_READONLY,nullptr);
    std::unique_ptr<sqlite3,decltype(&sqlite3_close)> db(raw,sqlite3_close);
    if(opened!=SQLITE_OK)throw std::runtime_error("Cannot open comparison database");
    sqlite3_stmt* stmt=nullptr;
    if(sqlite3_prepare_v2(db.get(),"SELECT id,measurement FROM constraints ORDER BY id",-1,&stmt,nullptr)!=SQLITE_OK)
        throw std::runtime_error(sqlite3_errmsg(db.get()));
    std::unique_ptr<sqlite3_stmt,decltype(&sqlite3_finalize)> query(stmt,sqlite3_finalize);
    for(size_t i=0;i<report.constraints.size();++i) {
        if(sqlite3_step(stmt)!=SQLITE_ROW || sqlite3_column_int64(stmt,0)!=static_cast<int64_t>(i) ||
            measurement(report.constraints[i]).dump()!=reinterpret_cast<const char*>(sqlite3_column_text(stmt,1)))
            throw std::runtime_error("Constraint audit differs from saved measurement at index "+std::to_string(i));
    }
    if(sqlite3_step(stmt)!=SQLITE_DONE)throw std::runtime_error("Constraint comparison row count differs");
}

nlohmann::json writeFiberConstraintDatabase(
    const std::filesystem::path& output, const std::vector<int64_t>& sourceIds,
    const vc::fiber_tracer::FiberTraceConstraintReport& report,
    const vc::lasagna::LasagnaNormalSampler& normals, int threads, const Json& metadata)
{
    const auto start=std::chrono::steady_clock::now();
    auto temporary=output;
    temporary += ".partial";
    if(std::filesystem::exists(output) || std::filesystem::exists(temporary))
        throw std::runtime_error("Refusing to overwrite constraint database or partial file");
    std::filesystem::create_directories(output.parent_path());
    sqlite3* raw=nullptr;
    const auto name=temporary.u8string();
    const int opened=sqlite3_open_v2(reinterpret_cast<const char*>(name.c_str()),&raw,SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE,nullptr);
    std::unique_ptr<sqlite3,decltype(&sqlite3_close)> db(raw,sqlite3_close);
    if(opened!=SQLITE_OK)throw std::runtime_error("Cannot create constraint database");
    const auto check=[&](int rc){if(rc!=SQLITE_OK && rc!=SQLITE_DONE)throw std::runtime_error(sqlite3_errmsg(db.get()));};
    const auto exec=[&](const char* sql){check(sqlite3_exec(db.get(),sql,nullptr,nullptr,nullptr));};
    using Statement=std::unique_ptr<sqlite3_stmt,decltype(&sqlite3_finalize)>;
    const auto prepare=[&](const char* sql){sqlite3_stmt* p=nullptr;check(sqlite3_prepare_v2(db.get(),sql,-1,&p,nullptr));return Statement(p,sqlite3_finalize);};
    const auto text=[&](sqlite3_stmt* p,int col,const std::string& value){check(sqlite3_bind_text(p,col,value.c_str(),int(value.size()),SQLITE_TRANSIENT));};
    exec("PRAGMA application_id=1447249475; PRAGMA user_version=1; PRAGMA foreign_keys=ON; BEGIN IMMEDIATE;"
         "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);"
         "CREATE TABLE fibers(id INTEGER PRIMARY KEY,source_index INTEGER UNIQUE NOT NULL,length_base REAL NOT NULL);"
         "CREATE TABLE constraints(id INTEGER PRIMARY KEY,fiber_a INTEGER NOT NULL REFERENCES fibers(id),"
         "fiber_b INTEGER NOT NULL REFERENCES fibers(id),arc_a REAL NOT NULL,arc_b REAL NOT NULL,"
         "ax REAL NOT NULL,ay REAL NOT NULL,az REAL NOT NULL,bx REAL NOT NULL,by REAL NOT NULL,bz REAL NOT NULL,"
         "parallel_score REAL NOT NULL,perpendicular_score REAL NOT NULL,winding_distance REAL NOT NULL,"
         "parallel_winding_distance REAL NOT NULL,evidence_weight REAL NOT NULL,"
         "normal_a TEXT NOT NULL,normal_b TEXT NOT NULL,measurement TEXT NOT NULL);");
    auto meta=prepare("INSERT INTO metadata VALUES(?,?)");
    const auto putMeta=[&](const char* key,const Json& value){text(meta.get(),1,key);text(meta.get(),2,value.dump());check(sqlite3_step(meta.get()));check(sqlite3_reset(meta.get()));};
    putMeta("format","fiber-constraints"); putMeta("schema_version",1); putMeta("provenance",metadata);
    putMeta("normal_semantics","unoriented sheet-normal axes sampled at exact constraint base points; invalid values are null");
    auto fiber=prepare("INSERT INTO fibers VALUES(?,?,?)");
    if(report.pieces.size()!=sourceIds.size())throw std::runtime_error("Export requires whole source fibers");
    std::vector<size_t> degree(sourceIds.size());
    for(const auto& p:report.pieces) {
        check(sqlite3_bind_int64(fiber.get(),1,sourceIds.at(p.traceIndex)));
        check(sqlite3_bind_int64(fiber.get(),2,p.traceIndex));
        check(sqlite3_bind_double(fiber.get(),3,p.endArcBaseVoxels-p.beginArcBaseVoxels));
        check(sqlite3_step(fiber.get()));check(sqlite3_reset(fiber.get()));
    }
    auto insert=prepare("INSERT INTO constraints VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
    size_t invalidNormals=0,parallel=0;
    double samplingSeconds=0.;
    for(size_t begin=0;begin<report.constraints.size();begin+=8192) {
        const size_t end=std::min(begin+8192,report.constraints.size());
        std::vector<cv::Vec3d> points; points.reserve(2*(end-begin));
        for(size_t i=begin;i<end;++i){points.push_back(report.constraints[i].pointABaseXYZ);points.push_back(report.constraints[i].pointBBaseXYZ);}
        std::vector<vc::lasagna::NormalSampleWithDerivative> samples;
        const auto samplingStart=std::chrono::steady_clock::now();
        (void)normals.sampleNormalBatch(points,false,threads,samples);
        samplingSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-samplingStart).count();
        if(samples.size()!=points.size())throw std::runtime_error("Normal batch size mismatch");
        for(size_t i=begin;i<end;++i) {
            const auto& c=report.constraints[i];
            const auto a=report.pieces.at(c.pieceA).traceIndex,b=report.pieces.at(c.pieceB).traceIndex;
            ++degree.at(a);++degree.at(b);parallel+=c.parallelScore>c.perpendicularScore;
            check(sqlite3_bind_int64(insert.get(),1,i));check(sqlite3_bind_int64(insert.get(),2,sourceIds.at(a)));check(sqlite3_bind_int64(insert.get(),3,sourceIds.at(b)));
            const double values[]={c.arcABaseVoxels,c.arcBBaseVoxels,c.pointABaseXYZ[0],c.pointABaseXYZ[1],c.pointABaseXYZ[2],c.pointBBaseXYZ[0],c.pointBBaseXYZ[1],c.pointBBaseXYZ[2],c.parallelScore,c.perpendicularScore,c.windingDistance,c.parallelWindingDistance,c.evidenceWeight};
            for(int k=0;k<13;++k)check(sqlite3_bind_double(insert.get(),4+k,values[k]));
            for(int side=0;side<2;++side) {
                const auto& n=samples[2*(i-begin)+side].sample;invalidNormals+=!n.valid;
                text(insert.get(),17+side,Json{{"valid",n.valid},{"xyz",n.valid?point(n.normal):Json(nullptr)},{"reason",n.reason}}.dump());
            }
            text(insert.get(),19,measurement(c).dump());check(sqlite3_step(insert.get()));check(sqlite3_reset(insert.get()));
        }
        if(begin%262144==0)std::cout<<"export constraints "<<end<<"/"<<report.constraints.size()<<std::endl;
    }
    std::sort(degree.begin(),degree.end());
    const auto quantile=[&](double p){return degree.empty()?0.:double(degree[size_t(p*(degree.size()-1))]);};
    Json stats={{"fibers",sourceIds.size()},{"constraints",report.constraints.size()},
        {"parallel",parallel},{"orthogonal",report.constraints.size()-parallel},{"invalid_endpoint_normals",invalidNormals},
        {"isolated_fibers",std::count(degree.begin(),degree.end(),size_t(0))},
        {"mean_degree",sourceIds.empty()?0.:2.*report.constraints.size()/sourceIds.size()},
        {"median_degree",quantile(.5)},{"p95_degree",quantile(.95)},{"max_degree",quantile(1)},
        {"endpoint_normal_sampling_seconds",samplingSeconds}};
    putMeta("statistics",stats);putMeta("complete",true);
    exec("CREATE INDEX constraint_a ON constraints(fiber_a); CREATE INDEX constraint_b ON constraints(fiber_b); COMMIT;");
    auto integrity=prepare("PRAGMA quick_check");
    if(sqlite3_step(integrity.get())!=SQLITE_ROW || std::string(reinterpret_cast<const char*>(sqlite3_column_text(integrity.get(),0)))!="ok")throw std::runtime_error("Constraint database integrity check failed");
    auto fk=prepare("PRAGMA foreign_key_check");if(sqlite3_step(fk.get())!=SQLITE_DONE)throw std::runtime_error("Constraint database foreign key failure");
    fk.reset();integrity.reset();insert.reset();fiber.reset();meta.reset();db.reset();
    std::filesystem::rename(temporary,output);
    stats["export_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    stats["database_bytes"]=std::filesystem::file_size(output);
    return stats;
}

nlohmann::json orientFiberConstraintDatabase(
    const std::filesystem::path& input, const std::filesystem::path& output,
    const std::vector<int64_t>& sourceIds,
    const std::vector<vc::fiber_tracer::FiberletCropTraceLine>& lines,
    const vc::lasagna::LasagnaNormalSampler& normals, int threads,
    double sampleSpacing, const Json& provenance, const vc::core::util::Umbilicus* priorUmbilicus,
    bool exportNormalCurves)
{
    using namespace vc::fiber_tracer;
    using Clock=std::chrono::steady_clock;
    const auto started=Clock::now();
    if(!std::isfinite(sampleSpacing) || sampleSpacing<=0. || threads<1 || lines.size()!=sourceIds.size())
        throw std::invalid_argument("Invalid normal orientation sampling configuration");
    auto partial=output;
    partial += ".partial";
    if(std::filesystem::exists(output) || std::filesystem::exists(partial))
        throw std::runtime_error("Refusing to overwrite oriented database or partial output");
    using Database=std::unique_ptr<sqlite3,decltype(&sqlite3_close)>;
    using Statement=std::unique_ptr<sqlite3_stmt,decltype(&sqlite3_finalize)>;
    const auto open=[](const auto& path,int flags) {
        const auto name=path.u8string();
        sqlite3* raw=nullptr; const int rc=sqlite3_open_v2(reinterpret_cast<const char*>(name.c_str()),&raw,flags,nullptr);
        Database db(raw,sqlite3_close);
        if(rc!=SQLITE_OK)throw std::runtime_error("Cannot open constraint database: "+path.string());
        return db;
    };
    auto src=open(input,SQLITE_OPEN_READONLY);
    if(sqlite3_exec(src.get(),"BEGIN",nullptr,nullptr,nullptr)!=SQLITE_OK)
        throw std::runtime_error("Cannot acquire source database snapshot");
    const auto check=[](sqlite3* db,int rc) {
        if(rc!=SQLITE_OK && rc!=SQLITE_DONE)throw std::runtime_error(sqlite3_errmsg(db));
    };
    const auto prepare=[&](sqlite3* db,const char* sql) {
        sqlite3_stmt* raw=nullptr;check(db,sqlite3_prepare_v2(db,sql,-1,&raw,nullptr));
        return Statement(raw,sqlite3_finalize);
    };
    const auto metaValue=[&](const char* key) {
        auto s=prepare(src.get(),"SELECT value FROM metadata WHERE key=?");
        check(src.get(),sqlite3_bind_text(s.get(),1,key,-1,SQLITE_STATIC));
        if(sqlite3_step(s.get())!=SQLITE_ROW)throw std::runtime_error("Missing source metadata");
        return Json::parse(reinterpret_cast<const char*>(sqlite3_column_text(s.get(),0)));
    };
    if(metaValue("format")!="fiber-constraints" || metaValue("schema_version")!=((priorUmbilicus || exportNormalCurves)?2:1) || metaValue("complete")!=true)
        throw std::runtime_error("Orientation requires schema1; prior extraction requires oriented schema2");
    if((priorUmbilicus || exportNormalCurves) && metaValue("normal_orientation").at("sample_spacing_base")!=sampleSpacing)
        throw std::runtime_error("Prior sweep spacing must match baked orientation");
    const auto original=metaValue("provenance");
    for(const char* key:{"afv_uuid","normal_manifest","normal_provenance"})
        if(original.at(key)!=provenance.at(key))throw std::runtime_error(std::string("Source mismatch: ")+key);

    struct Endpoint { size_t slot; double arc; cv::Vec3d point; };
    std::vector<std::vector<Endpoint>> endpoints(sourceIds.size());
    std::unordered_map<int64_t,size_t> index;
    std::vector<double> lengths(sourceIds.size());
    for(size_t i=0;i<sourceIds.size();++i) {
        if(!index.emplace(sourceIds[i],i).second)throw std::runtime_error("Duplicate source fiber");
        lengths[i]=makePolylineArcGeometry(lines[i].pointsBaseXYZ).length();
    }
    auto fiberQuery=prepare(src.get(),"SELECT id,source_index,length_base FROM fibers ORDER BY source_index");
    size_t fiberCount=0; int rc;
    while((rc=sqlite3_step(fiberQuery.get()))==SQLITE_ROW) {
        if(fiberCount>=sourceIds.size() || sqlite3_column_int64(fiberQuery.get(),0)!=sourceIds[fiberCount] ||
           sqlite3_column_int64(fiberQuery.get(),1)!=static_cast<int64_t>(fiberCount) ||
           std::abs(sqlite3_column_double(fiberQuery.get(),2)-lengths[fiberCount])>1e-6)
            throw std::runtime_error("Source fiber identity/order/length differs");
        ++fiberCount;
    }
    check(src.get(),rc);
    if(fiberCount!=sourceIds.size())throw std::runtime_error("Source fiber count differs");
    auto query=prepare(src.get(),"SELECT id,fiber_a,fiber_b,arc_a,arc_b,ax,ay,az,bx,by,bz FROM constraints ORDER BY id");
    size_t events=0;
    while((rc=sqlite3_step(query.get()))==SQLITE_ROW) {
        if(sqlite3_column_int64(query.get(),0)!=static_cast<int64_t>(events))throw std::runtime_error("Non-contiguous constraint IDs");
        for(int side=0;side<2;++side) {
            const auto i=index.at(sqlite3_column_int64(query.get(),1+side));
            const double arc=sqlite3_column_double(query.get(),3+side);
            if(!std::isfinite(arc) || arc<0 || arc>lengths[i]+1e-6)throw std::runtime_error("Invalid constraint arc");
            const int c=5+side*3;
            endpoints[i].push_back({2*events+side,arc,{sqlite3_column_double(query.get(),c),
                sqlite3_column_double(query.get(),c+1),sqlite3_column_double(query.get(),c+2)}});
        }
        ++events;
    }
    check(src.get(),rc);query.reset();fiberQuery.reset();
    std::vector<cv::Vec3d> baked(2*events);
    std::vector<double> priorSum(lines.size(),0.),priorWeight(lines.size(),0.);
    std::vector<unsigned char> valid(2*events,0);
    size_t samplesTotal=0,flips=0,endpointFlips=0,invalid=0,invalidEndpoints=0,invalidFibers=0;
    double sampleSeconds=0.;
    const auto sweepStarted=Clock::now();
    std::ofstream curveStream;
    Json curveIndex=Json::array();size_t curveSamples=0;
    if(exportNormalCurves) {
        if(priorUmbilicus)throw std::invalid_argument("Choose curve export OR outward priors");
        // Portable explicit little-endian float64 payload, regular arc + normal
        // + validity. Geometry remains in the original AFV, not this cache.
        std::filesystem::create_directories(partial);
        curveStream.open(partial/"normals.bin",std::ios::binary);
        if(!curveStream)throw std::runtime_error("Cannot create normal curve payload");
    }
    for(size_t begin=0;begin<lines.size();begin+=256) {
        struct Sample { double arc; cv::Vec3d point; size_t slot; double weight=0.; };
        const size_t end=std::min(begin+256,lines.size()), regular=std::numeric_limits<size_t>::max();
        std::vector<Sample> ordered;std::vector<size_t> offsets{0};
        for(size_t i=begin;i<end;++i) {
            const auto geometry=makePolylineArcGeometry(lines[i].pointsBaseXYZ);
            const size_t first=ordered.size();
            for(double arc=0;arc<geometry.length();arc+=sampleSpacing)
                ordered.push_back({arc,samplePolylineArc(geometry,arc).point,regular});
            ordered.push_back({geometry.length(),samplePolylineArc(geometry,geometry.length()).point,regular});
            for(size_t j=first;j<ordered.size();++j) {
                const double left=j==first?ordered[j].arc:ordered[j-1].arc;
                const double right=j+1==ordered.size()?ordered[j].arc:ordered[j+1].arc;
                ordered[j].weight=(right-left)*.5;
            }
            for(const auto& e:endpoints[i])ordered.push_back({e.arc,e.point,e.slot});
            std::stable_sort(ordered.begin()+first,ordered.end(),[](const auto& a,const auto& b){return a.arc<b.arc;});
            offsets.push_back(ordered.size());
        }
        std::vector<cv::Vec3d> points;points.reserve(ordered.size());
        for(const auto& s:ordered)points.push_back(s.point);
        std::vector<vc::lasagna::NormalSampleWithDerivative> sampled;
        const auto sampleStarted=Clock::now();
        (void)normals.sampleNormalBatch(points,false,threads,sampled);
        sampleSeconds+=std::chrono::duration<double>(Clock::now()-sampleStarted).count();
        if(sampled.size()!=ordered.size())throw std::runtime_error("Normal batch size differs");
        for(size_t f=0;f+1<offsets.size();++f) {
            FiberNormalOrientationSweep sweep;size_t good=0;
            const auto firstSample=curveSamples;
            for(size_t j=offsets[f];j<offsets[f+1];++j) {
                auto n=sampled[j].sample.normal;const bool ok=sampled[j].sample.valid;
                if(ok && (!std::isfinite(n.dot(n)) || std::abs(n.dot(n)-1.)>1e-6))
                    throw std::runtime_error("Non-unit valid normal");
                const bool flipped=sweep.orient(n,ok);
                flips+=flipped;invalid+=!ok;good+=ok;
                if(priorUmbilicus && ok && ordered[j].weight>0.) {
                    const cv::Vec3f p(ordered[j].point);
                    const cv::Vec3d radial(-priorUmbilicus->vector_to_umbilicus(p));
                    const double r=cv::norm(radial);
                    if(r>0.) {
                        priorSum[begin+f]+=ordered[j].weight*n.dot(radial/r);
                        priorWeight[begin+f]+=ordered[j].weight;
                    }
                }
                if(ordered[j].slot!=regular) {
                    const auto slot=ordered[j].slot;baked[slot]=n;valid[slot]=ok;
                    endpointFlips+=flipped;invalidEndpoints+=!ok;
                }
                else if(exportNormalCurves) {
                    std::array<double,5> record{ordered[j].arc,n[0],n[1],n[2],ok?1.:0.};
                    for(double value:record) {
                        auto bytes=std::bit_cast<std::array<char,8>>(value);
                        if constexpr(std::endian::native!=std::endian::little)std::reverse(bytes.begin(),bytes.end());
                        curveStream.write(bytes.data(),bytes.size());
                    }
                    ++curveSamples;
                }
            }
            if(exportNormalCurves)curveIndex.push_back({sourceIds[begin+f],firstSample,curveSamples-firstSample});
            invalidFibers+=good==0;
        }
        samplesTotal+=ordered.size();
        if(begin%8192==0 || end==lines.size())std::cout<<"orient normals fibers="<<end<<"/"<<lines.size()
            <<" samples="<<samplesTotal<<" seconds="<<std::chrono::duration<double>(Clock::now()-sweepStarted).count()<<std::endl;
    }
    const double sweepSeconds=std::chrono::duration<double>(Clock::now()-sweepStarted).count();
    if(priorUmbilicus || exportNormalCurves) {
        // Verify the gauge against ALL stored endpoint normals before exporting priors.
        query=prepare(src.get(),"SELECT normal_a,normal_b FROM constraints ORDER BY id");
        size_t count=0;
        while((rc=sqlite3_step(query.get()))==SQLITE_ROW) {
            for(int side=0;side<2;++side) {
                const auto slot=2*count+side;
                const auto n=Json::parse(reinterpret_cast<const char*>(sqlite3_column_text(query.get(),side)));
                if(n.at("valid").get<bool>()!=bool(valid[slot]))throw std::runtime_error("Prior endpoint validity mismatch");
                if(valid[slot]) {
                    const auto& p=n.at("xyz");const cv::Vec3d old{p[0].get<double>(),p[1].get<double>(),p[2].get<double>()};
                    if(cv::norm(old-baked[slot])>1e-12)throw std::runtime_error("Prior sweep differs from baked normal gauge");
                }
            }
            ++count;
        }
        check(src.get(),rc);if(count!=events)throw std::runtime_error("Prior endpoint count mismatch");
        if(exportNormalCurves) {
            curveStream.close();if(!curveStream)throw std::runtime_error("Normal curve write failed");
            Json stats={{"fibers",fiberCount},{"checked_endpoints",2*events},{"samples",curveSamples},
                {"invalid_samples",invalid},{"sample_spacing_base",sampleSpacing},
                {"sampling_seconds",sampleSeconds},{"sweep_seconds",sweepSeconds},
                {"total_seconds",std::chrono::duration<double>(Clock::now()-started).count()}};
            Json doc={{"format","fiber-normal-curves"},{"version",1},{"dtype","<f8"},
                {"columns",{"arc","nx","ny","nz","valid"}},
                {"index",curveIndex},{"statistics",stats},{"provenance",provenance},
                {"normal_convention","fiber-local baked gauge; apply stage1 normal_sign"}};
            vc::core::util::ephemeralWriteString(partial/"index.json",doc.dump()+"\n");
            std::filesystem::rename(partial,output);return stats;
        }
        std::vector<double> agreement(lines.size());
        for(size_t i=0;i<lines.size();++i)agreement[i]=priorWeight[i]>0?priorSum[i]/priorWeight[i]:0.;
        Json stats={{"fibers",fiberCount},{"checked_endpoints",2*events},{"sample_spacing_base",sampleSpacing},
            {"invalid_samples",invalid},{"sweep_seconds",sweepSeconds},
            {"total_seconds",std::chrono::duration<double>(Clock::now()-started).count()}};
        Json document={{"format","fiber-outward-priors"},{"version",1},{"source_database",input.string()},
            {"source_database_sha256",provenance.at("config").at("source_database_sha256")},
            {"umbilicus_sha256",provenance.at("config").at("umbilicus_sha256")},
            {"source_ids",sourceIds},{"agreement",agreement},{"valid_arc_weight",priorWeight},
            {"provenance",provenance},{"statistics",stats}};
        std::filesystem::create_directories(output.parent_path());
        vc::core::util::ephemeralWriteString(output,document.dump()+"\n");
        return stats;
    }
    std::filesystem::create_directories(output.parent_path());
    auto dst=open(partial,SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE);
    std::cout<<"orientation: cloning original database"<<std::endl;
    sqlite3_backup* backup=sqlite3_backup_init(dst.get(),"main",src.get(),"main");
    if(!backup)throw std::runtime_error(sqlite3_errmsg(dst.get()));
    const int copyRc=sqlite3_backup_step(backup,-1);const int finishRc=sqlite3_backup_finish(backup);
    check(dst.get(),copyRc);check(dst.get(),finishRc);
    const auto exec=[&](const char* sql){check(dst.get(),sqlite3_exec(dst.get(),sql,nullptr,nullptr,nullptr));};
    exec("PRAGMA foreign_keys=ON; BEGIN IMMEDIATE; ALTER TABLE fibers ADD COLUMN normal_sign INTEGER NOT NULL DEFAULT 1 CHECK(normal_sign IN(-1,1)); PRAGMA user_version=2;");
    auto update=prepare(dst.get(),"UPDATE constraints SET normal_a=?,normal_b=? WHERE id=?");
    query=prepare(src.get(),"SELECT id,normal_a,normal_b FROM constraints ORDER BY id");
    size_t count=0;
    while((rc=sqlite3_step(query.get()))==SQLITE_ROW) {
        for(int side=0;side<2;++side) {
            const auto slot=2*count+side;
            auto n=Json::parse(reinterpret_cast<const char*>(sqlite3_column_text(query.get(),1+side)));
            if(n.at("valid").get<bool>()!=bool(valid[slot]))throw std::runtime_error("Endpoint validity changed");
            if(valid[slot]) {
                const auto& p=n.at("xyz");const cv::Vec3d old{p[0].get<double>(),p[1].get<double>(),p[2].get<double>()};
                if(std::min(cv::norm(old-baked[slot]),cv::norm(old+baked[slot]))>1e-12)
                    throw std::runtime_error("Endpoint normal axis changed, not merely sign");
                n["xyz"]=point(baked[slot]);
            }
            const auto str=n.dump();check(dst.get(),sqlite3_bind_text(update.get(),side+1,str.c_str(),int(str.size()),SQLITE_TRANSIENT));
        }
        check(dst.get(),sqlite3_bind_int64(update.get(),3,count));check(dst.get(),sqlite3_step(update.get()));
        check(dst.get(),sqlite3_reset(update.get()));++count;
        if(count%262144==0)std::cout<<"oriented endpoint publication "<<count<<"/"<<events<<std::endl;
    }
    check(src.get(),rc);
    if(count!=events)throw std::runtime_error("Endpoint count changed");
    Json stats={{"fibers",fiberCount},{"constraints",events},{"samples",samplesTotal},{"flipped_samples",flips},
        {"flipped_endpoints",endpointFlips},{"invalid_samples",invalid},{"invalid_endpoints",invalidEndpoints},
        {"fibers_without_valid_normals",invalidFibers},{"sample_spacing_base",sampleSpacing},
        {"sampling_seconds",sampleSeconds},{"sweep_seconds",sweepSeconds},
        {"source_database",input.string()},{"cross_fiber_alignment",false}};
    auto meta=prepare(dst.get(),"INSERT OR REPLACE INTO metadata VALUES(?,?)");
    const auto put=[&](const char* key,const Json& value) {
        const auto str=value.dump();check(dst.get(),sqlite3_bind_text(meta.get(),1,key,-1,SQLITE_STATIC));
        check(dst.get(),sqlite3_bind_text(meta.get(),2,str.c_str(),int(str.size()),SQLITE_TRANSIENT));
        check(dst.get(),sqlite3_step(meta.get()));check(dst.get(),sqlite3_reset(meta.get()));
    };
    put("schema_version",2);put("normal_orientation",stats);put("normal_orientation_provenance",provenance);
    put("normal_semantics","fiber-local signed unit sheet normals; effective normal=normal_sign*baked normal; first valid sample anchors gauge; invalid gaps retain last valid orientation; no cross-fiber alignment");
    exec("COMMIT;");
    auto integrity=prepare(dst.get(),"PRAGMA quick_check");
    if(sqlite3_step(integrity.get())!=SQLITE_ROW || std::string(reinterpret_cast<const char*>(sqlite3_column_text(integrity.get(),0)))!="ok")
        throw std::runtime_error("Oriented database integrity failure");
    auto fk=prepare(dst.get(),"PRAGMA foreign_key_check");
    if(sqlite3_step(fk.get())!=SQLITE_DONE)throw std::runtime_error("Oriented database foreign-key failure");
    fk.reset();integrity.reset();meta.reset();update.reset();query.reset();dst.reset();src.reset();
    std::filesystem::rename(partial,output);
    stats["total_seconds"]=std::chrono::duration<double>(Clock::now()-started).count();
    stats["database_bytes"]=std::filesystem::file_size(output);
    return stats;
}
