#include "TestSupport.hpp"
#include "vc_object_store/objects/AfvSource.hpp"
#include "vc_object_store/objects/FiberImportExtension.hpp"
#include "internal/Sqlite.hpp"
#include "imports/AfvRecords.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <cmath>

using namespace vc::objects;
using Json=nlohmann::json;
const ExtensionValue& field(const AfcConstraint& constraint, const std::string& name)
{
    static const auto spec = fiberConstraintExtension({"1482bcc8-f93e-4c3b-b380-2ddf369318f4"});
    require(constraint.measurement.table == spec.name && constraint.measurement.values.size() == spec.columns.size(),
            "AFC row follows the typed fiber constraint extension schema");
    for (size_t i = 0; i < spec.columns.size(); ++i)
        if (spec.columns[i].name == name) return constraint.measurement.values[i];
    throw std::runtime_error("missing test extension column: " + name);
}
bool isNull(const AfcConstraint& constraint, const std::string& name)
{
    return std::holds_alternative<std::monostate>(field(constraint, name));
}
void compactProvisioning()
{
    const auto uuid="1482bcc8-f93e-4c3b-b380-2ddf369318f4";
    FiberSourceInfo source{uuid,std::string(64,'A'),2,262};
    const auto fibers=fiberSourceExtension(source);
    require(fibers.name=="ext_fiber_graph_solve_sources"&&fibers.extension==kFiberGraphSolveExtension,
            "fiber source rows belong to the shared graph-solve extension family");
    require(Json::parse(fibers.provisioning)==Json({{"source_uuid",uuid},{"source_sha256",std::string(64,'a')},
        {"fiber_count",2},{"point_count",262}}),"fiber provisioning is constructed only from explicit scalar source values");
    FiberConstraintSourceInfo input{uuid,{},17};
    const auto constraints=fiberConstraintExtension(input);
    require(constraints.name=="ext_fiber_graph_solve_constraints"&&constraints.extension==kFiberGraphSolveExtension,
            "one typed measurement row uses the graph-solve constraint table");
    require(Json::parse(constraints.provisioning)==Json({{"fiber_source_uuid",uuid},{"constraint_count",17},
        {"length_units","base_voxels"},{"winding_units","turns"},{"endpoint_order","source"}}),
        "constraint provenance does not pretend the parent fiber UUID identifies the AFC file");
    require(fibers.basedOn.empty()&&constraints.basedOn.empty(),"external files do not fabricate vcobj source commits");
    require(Json::parse(fiberSourceExtension({uuid}).provisioning)==Json({{"source_uuid",uuid}}),
            "absent source statistics and checksums are omitted instead of invented");
    source.fiberCount=-1;mustFail([&]{fiberSourceExtension(source);});source.fiberCount=2;
    source.sha256="not-a-checksum";mustFail([&]{fiberSourceExtension(source);});
    input.fiberSourceUuid="not-a-uuid";mustFail([&]{fiberConstraintExtension(input);});
}
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
    compactProvisioning();
    require(util::nameUuid("6ba7b810-9dad-11d1-80b4-00c04fd430c8","www.widgets.com")=="21f7f8de-8051-5b89-8680-0195ef798b6a","RFC UUIDv5 value");
    require(util::nameUuid(detail::parseUuid("6ba7b810-9dad-11d1-80b4-00c04fd430c8"),"www.widgets.com")=="21f7f8de-8051-5b89-8680-0195ef798b6a","binary namespace UUIDv5 value");
    TemporaryDirectory dir; const auto good=dir.path/"source.afv"; fixture(good);
    {
        AfvSource source(good); AfvFiber f;
        source.setCacheBytes(128*1024*1024);mustFail([&]{source.setCacheBytes(0);});
        require(source.setMmapBytes(16*1024*1024)<=16*1024*1024,"source mapping exceeds configured budget");
        require(source.fiberCount()==2 && source.pointCount()==262,"metadata counts");
        require(source.idRanges(2)==std::vector<std::array<int64_t,2>>{{0,9},{9,10}},"balanced sparse intervals");
        require(source.idRanges(2,1)==std::vector<std::array<int64_t,2>>{{0,1}},"limited interval");
        require(source.workRanges(100,100)==std::vector<std::array<int64_t,2>>{{0,9},{9,10}},"point-budgeted long fiber stays whole");
        require(source.workRanges(1000,1)==source.idRanges(2),"row-budgeted chunks cover sparse IDs");
        require(source.workRanges(1000,100,1)==source.idRanges(1,1),"work chunks preserve prefix selection");
        mustFail([&]{source.workRanges(0,100);});
        require(source.next(f) && f.id==0 && f.xyz.size()==260*24 && f.detail.controlPointCount==0,"multi-block fiber");
        const Json coordinates={{"volume_uuid","test-volume"},{"scale_factor",1.}};
        const auto record=imports::afvObjectRecord(f,source.uuid(),coordinates);
        require(record.uuid==util::nameUuid(source.uuid(),"fiber:0")&&record.kind=="fiber"&&record.formatVersion==4,
            "shared AFV record identity and format");
        const auto meta=Json::parse(record.meta);
        require(meta==Json({{"coordinates",coordinates}}),"AFV records retain coordinates without duplicated source JSON");
        const auto sourceRow=fiberSourceRow(record.uuid,f.id,f.annotation);
        require(sourceRow.table==kFiberSourceTable&&sourceRow.values==ExtensionRow{
            extensionUuid(record.uuid),int64_t(0),std::string("{}")},"AFV source identity and annotation use typed extension rows");
        const auto original=f.xyz;const auto geometry=imports::afvGeometry(f);
        require(geometry.kind=="fiber"&&geometry.arrays.size()==3&&geometry.arrays[0].bytes.empty()&&geometry.arrays[1].bytes.empty()&&
            geometry.arrays[2].name=="line_points"&&geometry.arrays[2].bytes==original,"shared AFV geometry preserves exact points without controls");
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
        {"provenance",{{"afv_uuid","1482BCC8-F93E-4C3B-B380-2DDF369318F4"}}},{"statistics",{{"constraints",1}}}};
    for(const auto& [key,value]:metadata.items())detail::execute(db,"INSERT INTO metadata VALUES(?,?)",key,value.dump());
    sqlite3_close(db);
    {
        AfcSource source(afc,"1482bcc8-f93e-4c3b-b380-2ddf369318f4");AfcConstraint constraint;
        source.setCacheBytes(128*1024*1024);mustFail([&]{source.setCacheBytes(0);});
        require(source.setMmapBytes(16*1024*1024)<=16*1024*1024,"constraint mapping exceeds configured budget");
        require(source.next(constraint)&&constraint.edge.kind=="perpendicular"&&!constraint.edge.aAnchor.direction,
                "one perpendicular with normals not tangents");
        require(constraint.sourceId==0&&constraint.sourceFiberA==0&&constraint.sourceFiberB==9,
                "source IDs remain available without decoding JSON metadata");
        require(constraint.edge.meta=="{}","ordinary constraint measurements are not duplicated in edge JSON");
        require(extensionUuid(field(constraint,"edge_uuid"))==constraint.edge.uuid&&
                std::get<int64_t>(field(constraint,"source_id"))==0&&
                std::get<int64_t>(field(constraint,"source_fiber_a"))==0&&
                std::get<int64_t>(field(constraint,"source_fiber_b"))==9,"typed source identities");
        require(std::get<double>(field(constraint,"arc_a"))==1.25&&
                std::get<double>(field(constraint,"arc_b"))==2.5&&
                std::get<double>(field(constraint,"las_winding"))==.75&&
                std::get<double>(field(constraint,"parallel_las_winding"))==.25,
                "fractional winding measurements and arc positions remain typed without integer expansion");
        require(Json::parse(std::get<std::string>(field(constraint,"extra"))).at("measurement").at("custom")==7,
                "unknown source properties remain in the extension's explicit extra field");
        require(std::get<int64_t>(field(constraint,"normal_a_valid"))==1&&
                std::get<double>(field(constraint,"normal_a_x"))==1&&
                std::get<double>(field(constraint,"normal_a_y"))==0&&
                std::get<double>(field(constraint,"normal_a_z"))==0&&
                std::get<int64_t>(field(constraint,"normal_b_valid"))==0&&
                isNull(constraint,"normal_b_x")&&isNull(constraint,"normal_b_y")&&isNull(constraint,"normal_b_z"),
                "valid and invalid source normals retain their typed validity and coordinates");
        const auto sourceA=util::nameUuid("1482bcc8-f93e-4c3b-b380-2ddf369318f4","fiber:0");
        const auto sourceB=util::nameUuid("1482bcc8-f93e-4c3b-b380-2ddf369318f4","fiber:9");
        const bool reversed=std::get<int64_t>(field(constraint,"source_reversed"))!=0;
        require(reversed==(sourceB<sourceA)&&constraint.edge.aUuid==std::min(sourceA,sourceB)&&
                constraint.edge.bUuid==std::max(sourceA,sourceB),"source orientation is explicitly mapped to canonical core endpoints");
        const auto& sourceAAnchor=reversed?constraint.edge.bAnchor:constraint.edge.aAnchor;
        const auto& sourceBAnchor=reversed?constraint.edge.aAnchor:constraint.edge.bAnchor;
        require(sourceAAnchor.position==std::array<double,3>{1,2,3}&&
                sourceBAnchor.position==std::array<double,3>{4,5,6}&&!source.next(constraint),"native anchors and EOF");
        require(!source.next(constraint)&&!source.next(constraint),"AFC EOF remains terminal across producer batches");
        const auto before=constraint;source.selectRange(0,1);
        require(source.next(constraint)&&constraint.edge.uuid==before.edge.uuid&&constraint.measurement.values==before.measurement.values&&
                !source.next(constraint),"AFC reuses cursor with identical identity and typed values");
        source.selectRange(1,1);require(!source.next(constraint),"AFC empty task");
        mustFail([&]{source.selectRange(2,1);});
    }
    {AfcSource source(afc,"1482bcc8-f93e-4c3b-b380-2ddf369318f4",5);AfcConstraint constraint;require(!source.next(constraint),"subset excludes missing endpoint");}
    {AfcSource source(afc,"1482bcc8-f93e-4c3b-b380-2ddf369318f4");
        require(source.idRanges(8)==std::vector<std::array<int64_t,2>>{{0,1}},"AFC sparse partition bounds");}
    {AfcSource source(afc,"1482bcc8-f93e-4c3b-b380-2ddf369318f4",INT64_MAX,1,10);AfcConstraint constraint;
        require(!source.next(constraint)&&!source.next(constraint),"empty AFC ID range remains terminal");}
    {
        const auto reused=dir.path/"reuse.sqlite";std::filesystem::copy_file(afc,reused);
        detail::check(sqlite3_open(reused.string().c_str(),&db),db);
        detail::exec(db,"INSERT INTO constraints SELECT 10,fiber_a,fiber_b,NULL,NULL,ax,ay,az,bx,by,bz,"
            "1,0,NULL,NULL,NULL,NULL,NULL,'{\"different\":true}' FROM constraints;"
            "UPDATE metadata SET value='{\"constraints\":2}' WHERE key='statistics';");sqlite3_close(db);
        AfcSource source(reused,"1482bcc8-f93e-4c3b-b380-2ddf369318f4");AfcConstraint constraint;
        require(source.next(constraint),"first reusable typed row");const auto first=constraint.measurement.values;
        require(source.next(constraint)&&constraint.edge.kind=="parallel","second reusable typed row");
        require(isNull(constraint,"arc_a")&&isNull(constraint,"arc_b")&&isNull(constraint,"normal_a_valid")&&
                isNull(constraint,"normal_a_x")&&isNull(constraint,"normal_b_valid")&&isNull(constraint,"normal_b_x")&&
                Json::parse(std::get<std::string>(field(constraint,"extra"))).at("measurement")==Json({{"different",true}}),
                "typed optional fields and extra properties do not leak between reused rows");
        require(!source.next(constraint),"reused typed row EOF");source.selectRange(0,1);
        require(source.next(constraint)&&constraint.measurement.values==first,"reused row changes canonical typed values");
    }
    detail::check(sqlite3_open(afc.string().c_str(),&db),db);
    detail::exec(db,"UPDATE constraints SET parallel_winding_distance=NULL");sqlite3_close(db);
    {AfcSource source(afc,"1482bcc8-f93e-4c3b-b380-2ddf369318f4");AfcConstraint constraint;
        require(source.next(constraint)&&isNull(constraint,"parallel_las_winding"),"optional SQL null is not silently zero");}
    mustFail([&]{AfcSource source(afc,util::newUuid());});
}
