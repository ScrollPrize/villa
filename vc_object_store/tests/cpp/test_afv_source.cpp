#include "TestSupport.hpp"
#include "vc_object_store/objects/AfvSource.hpp"
#include "internal/Sqlite.hpp"
#include <nlohmann/json.hpp>
#include <bit>
#include <cmath>

using namespace vc::objects;
using Json=nlohmann::json;
void fixture(const std::filesystem::path& path)
{
    sqlite3* db=nullptr; detail::check(sqlite3_open(path.string().c_str(),&db),db);
    try {
        detail::exec(db,"PRAGMA application_id=1447249475;PRAGMA user_version=1;"
            "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);"
            "CREATE TABLE fibers(id INTEGER PRIMARY KEY,name TEXT,family TEXT,point_count INTEGER,length REAL,"
            "min_x REAL,max_x REAL,min_y REAL,max_y REAL,min_z REAL,max_z REAL,annotation TEXT);"
            "CREATE TABLE blocks(fiber_id INTEGER,first_segment INTEGER,points BLOB,UNIQUE(fiber_id,first_segment));");
        const Json meta={{"uuid","1482bcc8-f93e-4c3b-b380-2ddf369318f4"},{"complete",true},{"fiber_count",2},
            {"point_count",262},{"frame",Json::object()},{"root",{{"coordinate_order","XYZ"},{"coordinate_units","voxel"}}}};
        for (const auto& [key,value]:meta.items()) detail::execute(db,"INSERT INTO metadata VALUES(?,?)",key,value.dump());
        for (int64_t id:{0,9}) {
            const int64_t count=id==0?260:2;
            detail::execute(db,"INSERT INTO fibers VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",id,std::string("fiber"),std::string("H"),count,
                7.25,0.,double(count-1),0.,0.,0.,0.,std::string("{}"));
            for (int64_t start=0;start<count-1;start+=256) {
                std::vector<uint8_t> bytes;
                for (int64_t p=start;p<std::min(start+257,count);++p) for (int c=0;c<3;++c) {
                    const auto bits=std::bit_cast<uint64_t>(c==0?double(p):0.);
                    for (int b=0;b<8;++b) bytes.push_back(static_cast<uint8_t>(bits>>(b*8)));
                }
                detail::execute(db,"INSERT INTO blocks VALUES(?,?,?)",id,start,std::span<const uint8_t>(bytes));
            }
        }
    } catch (...) { sqlite3_close(db); throw; }
    sqlite3_close(db);
}
int main()
{
    TemporaryDirectory dir; const auto good=dir.path/"source.afv"; fixture(good);
    {
        AfvSource source(good); AfvFiber f;
        require(source.fiberCount()==2 && source.pointCount()==262,"metadata counts");
        require(source.idRanges(2)==std::vector<std::array<int64_t,2>>{{0,9},{9,10}},"balanced sparse intervals");
        require(source.idRanges(2,1)==std::vector<std::array<int64_t,2>>{{0,1}},"limited interval");
        require(source.next(f) && f.id==0 && f.xyz.size()==260*24 && f.detail.controlPointCount==0,"multi-block fiber");
        require(source.next(f) && f.id==9 && f.xyz.size()==48,"sparse IDs");
        require(!source.next(f),"EOF");
    }
    { AfvSource source(good,1,10); AfvFiber f; require(source.next(f)&&f.id==9&&!source.next(f),"bounded sparse reader"); }
    {
        AfvSource source(good);AfvFiber f;
        require(source.next(f)&&f.id==0,"range cursor initial scan");const auto bytes=f.xyz;
        source.selectRange(9,10);require(source.next(f)&&f.id==9&&!source.next(f),"seek abandons unfinished scan");
        source.selectRange(1,9);require(!source.next(f)&&!source.next(f),"empty reused range remains exhausted");
        source.selectRange(0,1);mustFail([&]{source.selectRange(-1,1);});mustFail([&]{source.selectRange(10,9);});
        require(source.next(f)&&f.id==0&&f.xyz==bytes&&!source.next(f),"invalid range leaves selected scan intact");
        source.selectRange(0,INT64_MAX);int count=0;while(source.next(f))++count;
        require(count==2,"full rescan resets metadata total accounting");
    }
    {
        const auto snapshot=dir.path/"snapshot.afv";std::filesystem::copy_file(good,snapshot);
        sqlite3* writer=nullptr;detail::check(sqlite3_open(snapshot.string().c_str(),&writer),writer);
        detail::exec(writer,"PRAGMA journal_mode=WAL");
        {
            AfvSource source(snapshot,0,0);AfvFiber f;
            detail::exec(writer,"UPDATE fibers SET annotation='{\"new\":true}' WHERE id=9");
            source.selectRange(9,10);require(source.next(f)&&f.annotation=="{}","range changes retain original validated snapshot");
        }
        sqlite3_close(writer);
        AfvSource changed(snapshot,9,10);AfvFiber f;
        require(changed.next(f)&&Json::parse(f.annotation).at("new")==true,"new reader sees committed source update");
    }
    {
        const auto counts=dir.path/"range-counts.afv";std::filesystem::copy_file(good,counts);
        sqlite3* writer=nullptr;detail::check(sqlite3_open(counts.string().c_str(),&writer),writer);
        detail::exec(writer,"UPDATE metadata SET value='263' WHERE key='point_count'");sqlite3_close(writer);
        AfvSource source(counts);AfvFiber f;require(source.next(f),"unfinished scan before range selection");
        source.selectRange(9,10);require(source.next(f)&&!source.next(f),"bounded scan checks its fibers, not full totals");
        source.selectRange(0,INT64_MAX);mustFail([&]{while(source.next(f)){};});
    }
    for (const auto* mutation:{"UPDATE metadata SET value='263' WHERE key='point_count'",
        "UPDATE blocks SET first_segment=255 WHERE first_segment=256",
        "DELETE FROM blocks WHERE fiber_id=9", "UPDATE blocks SET points=zeroblob(25) WHERE fiber_id=0 AND first_segment=0",
        "UPDATE blocks SET points=zeroblob(length(points)) WHERE first_segment=256",
        "UPDATE fibers SET max_x=max_x+1 WHERE id=0"}) {
        const auto bad=dir.path/(util::newUuid()+".afv"); std::filesystem::copy_file(good,bad);
        sqlite3* db=nullptr; detail::check(sqlite3_open(bad.string().c_str(),&db),db); detail::exec(db,mutation); sqlite3_close(db);
        mustFail([&]{AfvSource source(bad);AfvFiber f;while(source.next(f)){};});
    }
    const auto afc=dir.path/"constraints.sqlite";
    sqlite3* db=nullptr;detail::check(sqlite3_open(afc.string().c_str(),&db),db);
    detail::exec(db,"CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);"
        "CREATE TABLE constraints(id INTEGER PRIMARY KEY,fiber_a INTEGER,fiber_b INTEGER,arc_a REAL,arc_b REAL,"
        "ax REAL,ay REAL,az REAL,bx REAL,by REAL,bz REAL,parallel_score REAL,perpendicular_score REAL,"
        "winding_distance REAL,parallel_winding_distance REAL,evidence_weight REAL,normal_a TEXT,normal_b TEXT,measurement TEXT);"
        "INSERT INTO constraints VALUES(0,0,9,1.25,2.5,1,2,3,4,5,6,0,1,.75,.25,.9,"
        "'{\"valid\":true,\"xyz\":[1,0,0]}','{\"valid\":false,\"xyz\":null}','{\"custom\":7}');");
    const Json metadata={{"format","fiber-constraints"},{"schema_version",1},{"complete",true},
        {"provenance",{{"afv_uuid","1482bcc8-f93e-4c3b-b380-2ddf369318f4"}}},{"statistics",{{"constraints",1}}}};
    for(const auto& [key,value]:metadata.items())detail::execute(db,"INSERT INTO metadata VALUES(?,?)",key,value.dump());
    sqlite3_close(db);
    {
        AfcSource source(afc,"1482bcc8-f93e-4c3b-b380-2ddf369318f4");EdgeRecord edge;
        require(source.next(edge)&&edge.kind=="perpendicular"&&!edge.aAnchor.direction,"one perpendicular with normals not tangents");
        const auto values=Json::parse(edge.meta).at("afc");
        require(values.at("arc_a")==1.25&&values.at("winding_distance")==.75&&values.at("measurement").at("custom")==7,"unsupported values retained");
        require(edge.aAnchor.position==std::array<double,3>{1,2,3}&&!source.next(edge),"native anchors and EOF");
        require(!source.next(edge)&&!source.next(edge),"AFC EOF remains terminal across producer batches");
    }
    {AfcSource source(afc,"1482bcc8-f93e-4c3b-b380-2ddf369318f4",5);EdgeRecord edge;require(!source.next(edge),"subset excludes missing endpoint");}
    {AfcSource source(afc,"1482bcc8-f93e-4c3b-b380-2ddf369318f4");
        require(source.idRanges(8)==std::vector<std::array<int64_t,2>>{{0,1}},"AFC sparse partition bounds");}
    {AfcSource source(afc,"1482bcc8-f93e-4c3b-b380-2ddf369318f4",INT64_MAX,1,10);EdgeRecord edge;
        require(!source.next(edge)&&!source.next(edge),"empty AFC ID range remains terminal");}
    detail::check(sqlite3_open(afc.string().c_str(),&db),db);
    detail::exec(db,"UPDATE constraints SET parallel_winding_distance=NULL");sqlite3_close(db);
    {AfcSource source(afc,"1482bcc8-f93e-4c3b-b380-2ddf369318f4");EdgeRecord edge;
        require(source.next(edge)&&Json::parse(edge.meta).at("afc").at("parallel_winding_distance").is_null(),"optional SQL null is not silently zero");}
    mustFail([&]{AfcSource source(afc,util::newUuid());});
}
