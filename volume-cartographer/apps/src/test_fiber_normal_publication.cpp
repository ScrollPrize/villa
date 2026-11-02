#include "FiberConstraintDatabase.hpp"
#include <sqlite3.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

using Json=nlohmann::json;
using Db=std::unique_ptr<sqlite3,decltype(&sqlite3_close)>;
using Stmt=std::unique_ptr<sqlite3_stmt,decltype(&sqlite3_finalize)>;
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
void check(sqlite3* db,int rc) {if(rc!=SQLITE_OK && rc!=SQLITE_DONE)throw std::runtime_error(sqlite3_errmsg(db));}
Db database() {sqlite3* raw=nullptr;check(raw,sqlite3_open(":memory:",&raw));return Db(raw,sqlite3_close);}
void exec(sqlite3* db,const char* sql) {check(db,sqlite3_exec(db,sql,nullptr,nullptr,nullptr));}
Stmt stmt(sqlite3* db,const char* sql) {sqlite3_stmt* raw=nullptr;check(db,sqlite3_prepare_v2(db,sql,-1,&raw,nullptr));return Stmt(raw,sqlite3_finalize);}
std::string text(sqlite3_stmt* s,int col) {return reinterpret_cast<const char*>(sqlite3_column_text(s,col));}
void bindText(sqlite3* db,sqlite3_stmt* s,int col,const std::string& v) {check(db,sqlite3_bind_text(s,col,v.c_str(),int(v.size()),SQLITE_TRANSIENT));}
Db clone(sqlite3* src) {
    auto dst=database();auto* backup=sqlite3_backup_init(dst.get(),"main",src,"main");
    require(backup,"backup init");check(dst.get(),sqlite3_backup_step(backup,-1));check(dst.get(),sqlite3_backup_finish(backup));return dst;
}

// Frozen scalar publication oracle: same parse/validate/serialize/UPDATE loop as
// the previous implementation, including unchanged rows and signed-zero output.
void reference(sqlite3* src,sqlite3* dst,const std::vector<cv::Vec3d>& baked,const std::vector<unsigned char>& valid) {
    auto q=stmt(src,"SELECT id,normal_a,normal_b FROM constraints ORDER BY id");
    auto u=stmt(dst,"UPDATE constraints SET normal_a=?,normal_b=? WHERE id=?");
    size_t count=0;int rc;
    while((rc=sqlite3_step(q.get()))==SQLITE_ROW) {
        for(int side=0;side<2;++side) {
            const size_t slot=2*count+side;auto n=Json::parse(text(q.get(),1+side));
            require(n.at("valid").get<bool>()==bool(valid[slot]),"valid mismatch");
            if(valid[slot]) {
                const auto& p=n.at("xyz");cv::Vec3d old{p[0].get<double>(),p[1].get<double>(),p[2].get<double>()};
                require(std::min(cv::norm(old-baked[slot]),cv::norm(old+baked[slot]))<=1e-12,"axis mismatch");
                n["xyz"]=Json{baked[slot][0],baked[slot][1],baked[slot][2]};
            }
            bindText(dst,u.get(),side+1,n.dump());
        }
        check(dst,sqlite3_bind_int64(u.get(),3,count));check(dst,sqlite3_step(u.get()));check(dst,sqlite3_reset(u.get()));++count;
    }
    check(src,rc);require(count*2==baked.size(),"count mismatch");
}

int main(int argc,char** argv) {
    try {
        const bool benchmark=argc>1 && std::string(argv[1])=="--benchmark";
        const size_t rows=benchmark && argc>2?std::stoull(argv[2]):65539;
        const int threads=benchmark && argc>3?std::stoi(argv[3]):4;
        const int batch=benchmark && argc>4?std::stoi(argv[4]):32;
        auto src=database();exec(src.get(),"CREATE TABLE constraints(id INTEGER PRIMARY KEY,normal_a TEXT NOT NULL,normal_b TEXT NOT NULL,measurement TEXT NOT NULL); BEGIN;");
        auto insert=stmt(src.get(),"INSERT INTO constraints VALUES(?,?,?,?)");
        std::vector<cv::Vec3d> baked(2*rows);std::vector<unsigned char> valid(2*rows);
        size_t expectedChanged=0;
        for(size_t i=0;i<rows;++i) {
            check(src.get(),sqlite3_bind_int64(insert.get(),1,i));bool changed=false;
            for(int side=0;side<2;++side) {
                const size_t slot=2*i+side;valid[slot]=(i+side)%11!=0;
                const cv::Vec3d normal{.6,0.,.8};baked[slot]=normal;
                if(i%4<2 && valid[slot]) {baked[slot]=-normal;changed=true;}
                const Json n={{"valid",bool(valid[slot])},{"xyz",valid[slot]?Json{normal[0],normal[1],normal[2]}:Json(nullptr)},
                    {"reason",valid[slot]?"":"missing sample"},{"preserve",Json{{"nested",i%7}}}};
                bindText(src.get(),insert.get(),side+2,n.dump());
            }
            expectedChanged+=changed;bindText(src.get(),insert.get(),4,"untouched measurement");
            check(src.get(),sqlite3_step(insert.get()));check(src.get(),sqlite3_reset(insert.get()));
        }
        insert.reset();exec(src.get(),"COMMIT");
        for(int repeat=0;repeat<(benchmark?3:1);++repeat) {
            auto legacy=clone(src.get()),fast=clone(src.get());
            exec(legacy.get(),"BEGIN");exec(fast.get(),"BEGIN");
            const auto start=std::chrono::steady_clock::now();reference(src.get(),legacy.get(),baked,valid);
            const double oldSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            const auto stats=updateFiberConstraintNormals(src.get(),fast.get(),baked,valid,threads,batch);
            exec(legacy.get(),"COMMIT");exec(fast.get(),"COMMIT");
            require(stats.at("scanned")==rows && stats.at("updated")==expectedChanged,"publication counts");
            auto a=stmt(legacy.get(),"SELECT id,normal_a,normal_b,measurement FROM constraints ORDER BY id");
            auto b=stmt(fast.get(),"SELECT id,normal_a,normal_b,measurement FROM constraints ORDER BY id");
            size_t count=0;
            while(sqlite3_step(a.get())==SQLITE_ROW) {
                require(sqlite3_step(b.get())==SQLITE_ROW,"missing fast row");
                for(int col=0;col<4;++col)require(text(a.get(),col)==text(b.get(),col),"scalar/batched output differs");
                ++count;
            }
            require(count==rows && sqlite3_step(b.get())==SQLITE_DONE,"extra rows");
            std::cout<<Json{{"repeat",repeat},{"rows",rows},{"threads",threads},{"legacy_seconds",oldSeconds},{"fast",stats},{"exact_match",true}}.dump()<<std::endl;
        }
        if(!benchmark) {
            auto dst=clone(src.get());auto bad=baked;bad[2]={1.,0.,0.};
            bool threw=false;try {updateFiberConstraintNormals(src.get(),dst.get(),bad,valid,threads);}catch(const std::exception&){threw=true;}
            require(threw,"axis mismatch not rejected");
            auto wrongValid=valid;wrongValid[0]=!wrongValid[0];threw=false;
            try {updateFiberConstraintNormals(src.get(),dst.get(),baked,wrongValid,threads);}catch(const std::exception&){threw=true;}
            require(threw,"validity mismatch not rejected");
            auto missing=clone(src.get());exec(missing.get(),"DELETE FROM constraints WHERE id=0");threw=false;
            try {updateFiberConstraintNormals(missing.get(),dst.get(),baked,valid,threads);}catch(const std::exception&){threw=true;}
            require(threw,"noncontiguous IDs not rejected");
            auto absent=database();exec(absent.get(),"CREATE TABLE constraints(id INTEGER PRIMARY KEY,normal_a TEXT,normal_b TEXT)");threw=false;
            try {updateFiberConstraintNormals(src.get(),absent.get(),baked,valid,threads);}catch(const std::exception&){threw=true;}
            require(threw,"missing destination rows not rejected");
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
    return 0;
}
