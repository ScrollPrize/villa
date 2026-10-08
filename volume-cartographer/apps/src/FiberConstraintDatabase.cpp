#include "FiberConstraintDatabase.hpp"
#include <sqlite3.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>

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
    const int opened=sqlite3_open_v2(path.c_str(),&raw,SQLITE_OPEN_READONLY,nullptr);
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
    const auto temporary=std::filesystem::path(output.string()+".partial");
    if(std::filesystem::exists(output) || std::filesystem::exists(temporary))
        throw std::runtime_error("Refusing to overwrite constraint database or partial file");
    std::filesystem::create_directories(output.parent_path());
    sqlite3* raw=nullptr;
    const int opened=sqlite3_open_v2(temporary.c_str(),&raw,SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE,nullptr);
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
