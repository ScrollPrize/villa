#include "vc_object_store/objects/AfvSource.hpp"
#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/objects/FiberImportExtension.hpp"
#include "internal/Anchors.hpp"
#include "internal/Sqlite.hpp"
#include "geometry/GeometryInternal.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <bit>
#include <cstring>
#include <limits>
#include <set>

namespace vc::objects {
namespace {
void sourceCache(sqlite3* db,size_t bytes){
    const auto kib=bytes/1024+(bytes%1024!=0);
    if(!kib||kib>size_t(std::numeric_limits<int>::max()))throw std::invalid_argument("source SQLite cache budget outside range");
    detail::exec(db,("PRAGMA cache_size=-"+std::to_string(kib)).c_str());
}
size_t sourceMmap(sqlite3* db,size_t bytes){
    if(bytes>uint64_t(INT64_MAX))throw std::invalid_argument("source SQLite mapping budget outside range");
    detail::Statement setting(db,"PRAGMA mmap_size="+std::to_string(bytes));
    return setting.row()?size_t(std::max(int64_t(0),setting.i64(0))):0;
}
}
struct AfvSource::Impl {
    sqlite3* db = nullptr;
    std::unique_ptr<detail::Statement> rows;
    std::string metadata, uuid;
    int64_t fibers = 0, points = 0, readFibers = 0, readPoints = 0;
    bool row = false, full = false;
    ~Impl() { rows.reset(); if (db) sqlite3_close(db); }
};

AfvSource::AfvSource(const std::filesystem::path& path, int64_t firstId, int64_t pastLastId)
    : impl_(std::make_unique<Impl>())
{
    if (firstId < 0 || pastLastId < firstId) throw std::invalid_argument("invalid AFV ID interval");
    auto& p = *impl_;
    // A source cursor belongs to one reader, including all of its SQLite calls.
    const auto rc = sqlite3_open_v2(path.string().c_str(), &p.db, SQLITE_OPEN_READONLY|SQLITE_OPEN_NOMUTEX, nullptr);
    detail::check(rc, p.db, "open AFV source");
    detail::exec(p.db, "PRAGMA query_only=ON; PRAGMA cache_size=-32768; BEGIN;");
    if (detail::pragmaInt(p.db, "application_id") != 0x56434643 || detail::pragmaInt(p.db, "user_version") != 1)
        throw std::invalid_argument("expected AFV application ID and format version 1");
    nlohmann::json meta = nlohmann::json::object();
    detail::Statement m(p.db, "SELECT key,value FROM metadata");
    while (m.row()) meta[m.text(0)] = nlohmann::json::parse(m.text(1));
    if (!meta.at("complete").is_boolean() || !meta.at("complete").get<bool>())
        throw std::invalid_argument("AFV is incomplete");
    p.uuid = util::canonicalUuid(meta.at("uuid").get<std::string>());
    for (const auto* key : {"fiber_count", "point_count"})
        if (!meta.at(key).is_number_integer() || meta.at(key).get<int64_t>() < 0)
            throw std::invalid_argument("invalid AFV declared counts");
    p.fibers = meta.at("fiber_count").get<int64_t>(); p.points = meta.at("point_count").get<int64_t>();
    for (const auto* key : {"frame", "root"}) {
        const auto& frame = meta.at(key);
        if (frame.value("vc_open_data_source_coordinate_level", 0) != 0 ||
            frame.value("vc_open_data_source_coordinate_scale_factor", 1.) != 1.)
            throw std::invalid_argument("only native L0 AFV coordinates are supported");
    }
    const auto& root = meta.at("root");
    if (root.value("scale", 1.) != 1. || root.value("coordinate_order", "") != "XYZ" ||
        root.value("coordinate_units", "") != "voxel")
        throw std::invalid_argument("AFV must declare native XYZ voxel coordinates");
    p.metadata = meta.dump();
    p.rows = std::make_unique<detail::Statement>(p.db,
        "SELECT f.id,f.name,f.family,f.point_count,f.length,"
        "f.min_x,f.min_y,f.min_z,f.max_x,f.max_y,f.max_z,f.annotation,"
        "b.first_segment,b.points FROM fibers f LEFT JOIN blocks b ON b.fiber_id=f.id "
        "WHERE f.id>=? AND f.id<? ORDER BY f.id,b.first_segment");
    selectRange(firstId,pastLastId);
}
void AfvSource::selectRange(int64_t firstId,int64_t pastLastId)
{
    if(firstId<0||pastLastId<firstId)throw std::invalid_argument("invalid AFV ID interval");
    auto& p=*impl_;
    p.rows->reset();p.row=false;p.readFibers=0;p.readPoints=0;
    p.full=firstId==0&&pastLastId==INT64_MAX;
    p.rows->bind(1,firstId);p.rows->bind(2,pastLastId);
    p.row = p.rows->row();
}
AfvSource::~AfvSource() = default;
const std::string& AfvSource::metadataJson() const { return impl_->metadata; }
const std::string& AfvSource::uuid() const { return impl_->uuid; }
int64_t AfvSource::fiberCount() const { return impl_->fibers; }
int64_t AfvSource::pointCount() const { return impl_->points; }
void AfvSource::setCacheBytes(size_t bytes){sourceCache(impl_->db,bytes);}
size_t AfvSource::setMmapBytes(size_t bytes){return sourceMmap(impl_->db,bytes);}
std::vector<std::array<int64_t,2>> AfvSource::idRanges(size_t partitions,int64_t limit) const
{
    if(!partitions || limit<0)throw std::invalid_argument("invalid AFV partition count or limit");
    detail::Statement s(impl_->db,"SELECT id FROM fibers ORDER BY id LIMIT ?",limit?limit:int64_t(-1));
    std::vector<int64_t> ids;
    while(s.row()){
        const auto id=s.i64(0);
        if(id<0 || id>INT64_MAX-3)throw std::invalid_argument("AFV ID outside import reservation range");
        ids.push_back(id);
    }
    if(ids.size()!=static_cast<size_t>(limit?std::min(limit,impl_->fibers):impl_->fibers))
        throw std::invalid_argument("AFV fiber catalog count mismatch");
    partitions=std::min(partitions,ids.size());
    std::vector<std::array<int64_t,2>> ranges;
    for(size_t i=0;i<partitions;++i)ranges.push_back({i?ids[ids.size()*i/partitions]:0,
        i+1==partitions?ids.back()+1:ids[ids.size()*(i+1)/partitions]});
    return ranges;
}
std::vector<std::array<int64_t,2>> AfvSource::workRanges(size_t targetPoints,size_t maxFibers,int64_t limit) const
{
    if(!targetPoints||!maxFibers||limit<0)throw std::invalid_argument("invalid AFV work chunk budget");
    detail::Statement rows(impl_->db,"SELECT id,point_count FROM fibers ORDER BY id LIMIT ?",limit?limit:int64_t(-1));
    std::vector<std::array<int64_t,2>> result;int64_t start=0,last=-1,count=0;uint64_t points=0;size_t fibers=0;
    while(rows.row()) {
        const auto id=rows.i64(0),n=rows.i64(1);
        if(id<0||id>INT64_MAX-3||n<2)throw std::invalid_argument("invalid AFV task catalog entry");
        if(fibers&&(points>=targetPoints||fibers>=maxFibers)){result.push_back({start,id});start=id;points=0;fibers=0;}
        points+=uint64_t(n);++fibers;++count;last=id;
    }
    if(count!=(limit?std::min(limit,impl_->fibers):impl_->fibers))throw std::invalid_argument("AFV fiber catalog count mismatch");
    if(fibers)result.push_back({start,last+1});
    return result;
}
bool AfvSource::next(AfvFiber& fiber)
{
    auto& p = *impl_;
    if (!p.row) {
        if (p.full && (p.readFibers != p.fibers || p.readPoints != p.points))
            throw std::invalid_argument("AFV catalog totals do not match collection metadata");
        return false;
    }
    auto& s = *p.rows;
    fiber = {}; fiber.id = s.i64(0); fiber.name = s.text(1); fiber.detail.family = s.text(2);
    fiber.detail.linePointCount = s.i64(3); fiber.detail.length = s.f64(4);
    if (fiber.id < 0 || fiber.detail.linePointCount < 2 || !std::isfinite(fiber.detail.length) || fiber.detail.length < 0)
        throw std::invalid_argument("invalid AFV fiber catalog entry");
    std::array<double,6> bounds{};
    for (int i=0; i<6; ++i) bounds[i] = s.f64(i+5);
    fiber.detail.bbox = bounds; fiber.annotation = s.text(11);
    const auto annotation = nlohmann::json::parse(fiber.annotation);
    if (!annotation.is_object()) throw std::invalid_argument("AFV annotation must be an object");
    if (static_cast<uint64_t>(fiber.detail.linePointCount) > std::numeric_limits<size_t>::max()/24)
        throw std::invalid_argument("AFV fiber is too large");
    fiber.xyz.reserve(static_cast<size_t>(fiber.detail.linePointCount)*24);
    int64_t nextSegment = 0;
    do {
        const auto bytes = sqlite3_column_bytes(s.value, 13);
        const auto* blob = static_cast<const uint8_t*>(sqlite3_column_blob(s.value, 13));
        if (s.isNull(12) || !blob || bytes%24 || bytes<48 || bytes>257*24 || s.i64(12)!=nextSegment)
            throw std::invalid_argument("invalid or discontinuous AFV block");
        const size_t skip = fiber.xyz.empty() ? 0 : 24;
        if (skip && std::memcmp(fiber.xyz.data()+fiber.xyz.size()-24, blob, 24))
            throw std::invalid_argument("AFV block endpoints differ");
        fiber.xyz.insert(fiber.xyz.end(), blob+skip, blob+bytes);
        nextSegment += bytes/24-1;
        if (nextSegment >= fiber.detail.linePointCount)
            throw std::invalid_argument("AFV block exceeds declared point count");
        p.row = s.row();
    } while (p.row && s.i64(0) == fiber.id);
    if (nextSegment+1 != fiber.detail.linePointCount) throw std::invalid_argument("AFV point count mismatch");
    std::array<double,6> actual{INFINITY,INFINITY,INFINITY,-INFINITY,-INFINITY,-INFINITY};
    for (size_t i=0; i<fiber.xyz.size()/8; ++i) {
        uint64_t bits = 0;
        for (int b=0; b<8; ++b) bits |= uint64_t(fiber.xyz[i*8+b]) << (8*b);
        const double v = std::bit_cast<double>(bits);
        if (!std::isfinite(v)) throw std::invalid_argument("non-finite AFV point");
        actual[i%3] = std::min(actual[i%3],v); actual[3+i%3] = std::max(actual[3+i%3],v);
    }
    if (actual != bounds) throw std::invalid_argument("AFV bounds mismatch");
    ++p.readFibers; p.readPoints += fiber.detail.linePointCount;
    return true;
}

struct AfcSource::Impl {
    sqlite3* db=nullptr;
    std::unique_ptr<detail::Statement> rows;
    std::string metadata,uuid;
    std::array<unsigned char,16> uuidBytes;
    int64_t count=0,read=0,fiberEnd=INT64_MAX;
    bool full=false,finished=false;
    ~Impl(){rows.reset();if(db)sqlite3_close(db);}
};
AfcSource::AfcSource(const std::filesystem::path& path,const std::string& afvUuid,int64_t pastLastFiberId,int64_t firstConstraintId,int64_t pastLastConstraintId)
    : impl_(std::make_unique<Impl>())
{
    if(firstConstraintId<0 || pastLastConstraintId<=firstConstraintId)throw std::invalid_argument("invalid AFC ID range");
    auto& p=*impl_; p.uuid=util::canonicalUuid(afvUuid);p.fiberEnd=pastLastFiberId;p.full=pastLastFiberId==INT64_MAX && firstConstraintId==0 && pastLastConstraintId==INT64_MAX;
    p.uuidBytes=detail::parseUuid(p.uuid);
    detail::check(sqlite3_open_v2(path.string().c_str(),&p.db,SQLITE_OPEN_READONLY|SQLITE_OPEN_NOMUTEX,nullptr),p.db,"open AFC source");
    detail::exec(p.db,"PRAGMA query_only=ON;PRAGMA cache_size=-32768;BEGIN;");
    nlohmann::json meta=nlohmann::json::object();
    detail::Statement m(p.db,"SELECT key,value FROM metadata");
    while(m.row())meta[m.text(0)]=nlohmann::json::parse(m.text(1));
    if(meta.value("format","")!="fiber-constraints" || meta.value("schema_version",0)!=1 ||
        !meta.value("complete",false) || util::canonicalUuid(meta.at("provenance").at("afv_uuid").get<std::string>())!=p.uuid)
        throw std::invalid_argument("AFC format, completeness or AFV identity mismatch");
    p.count=meta.at("statistics").at("constraints").get<int64_t>();
    p.metadata=meta.dump();
    // This reader intentionally accepts the original integer-ID AFV contract,
    // not the newer UUID-object-source variant, which needs no AFV remapping.
    std::set<std::string> columns;
    detail::Statement types(p.db,"PRAGMA table_info(constraints)");
    while(types.row()) {
        columns.insert(types.text(1));
        if((types.text(1)=="fiber_a" || types.text(1)=="fiber_b") && types.text(2)!="INTEGER")
            throw std::invalid_argument("AFC source must use AFV integer fiber IDs");
    }
    if(columns!=std::set<std::string>{"id","fiber_a","fiber_b","arc_a","arc_b","ax","ay","az","bx","by","bz",
        "parallel_score","perpendicular_score","winding_distance","parallel_winding_distance","evidence_weight","normal_a","normal_b","measurement"})
        throw std::invalid_argument("unsupported AFC columns: refusing to drop unknown fields");
    p.rows=std::make_unique<detail::Statement>(p.db,
        "SELECT id,fiber_a,fiber_b,arc_a,arc_b,ax,ay,az,bx,by,bz,parallel_score,perpendicular_score,"
        "winding_distance,parallel_winding_distance,evidence_weight,normal_a,normal_b,measurement "
        "FROM constraints WHERE fiber_a<? AND fiber_b<? AND id>=? AND id<? ORDER BY id");
    selectRange(firstConstraintId,pastLastConstraintId);
}
void AfcSource::selectRange(int64_t firstId,int64_t pastLastId)
{
    if(firstId<0||pastLastId<firstId)throw std::invalid_argument("invalid AFC ID range");
    auto& p=*impl_;p.rows->reset();p.read=0;p.finished=false;
    p.full=p.fiberEnd==INT64_MAX&&firstId==0&&pastLastId==INT64_MAX;
    p.rows->bindAll(p.fiberEnd,p.fiberEnd,firstId,pastLastId);
}
AfcSource::~AfcSource()=default;
const std::string& AfcSource::metadataJson()const{return impl_->metadata;}
int64_t AfcSource::constraintCount()const{return impl_->count;}
void AfcSource::setCacheBytes(size_t bytes){sourceCache(impl_->db,bytes);}
size_t AfcSource::setMmapBytes(size_t bytes){return sourceMmap(impl_->db,bytes);}
std::vector<std::array<int64_t,2>> AfcSource::idRanges(size_t partitions,int64_t limit)const
{
    if(partitions<1 || limit<0)throw std::invalid_argument("invalid AFC partition count/limit");
    // Full-volume partitioning needs only the two B-tree endpoints, not a
    // min/max aggregate that scans every large constraint record first.
    std::unique_ptr<detail::Statement> filtered;
    if(impl_->fiberEnd!=INT64_MAX)filtered=std::make_unique<detail::Statement>(impl_->db,
        "SELECT min(id),max(id) FROM (SELECT id FROM constraints WHERE fiber_a<? AND fiber_b<? ORDER BY id LIMIT ?)",
        impl_->fiberEnd,impl_->fiberEnd,limit?limit:int64_t(-1));
    else filtered=std::make_unique<detail::Statement>(impl_->db,limit?
        "SELECT min(id),max(id) FROM (SELECT id FROM constraints ORDER BY id LIMIT ?)":
        "SELECT (SELECT id FROM constraints ORDER BY id LIMIT 1),(SELECT id FROM constraints ORDER BY id DESC LIMIT 1) WHERE ?=0",limit);
    auto& bounds=*filtered;
    if(!bounds.row() || sqlite3_column_type(bounds.value,0)==SQLITE_NULL)return {};
    const auto first=bounds.i64(0),last=bounds.i64(1);
    if(first<0 || last==INT64_MAX)throw std::invalid_argument("AFC IDs cannot be partitioned safely");
    const auto span=last-first+1;const auto n=std::min<uint64_t>(partitions,uint64_t(span));
    std::vector<std::array<int64_t,2>> result;
    // Equal numeric ranges do not assume dense IDs; the imported aggregate is
    // checked against the declared count/explicit prefix limit after reading.
    for(uint64_t i=0;i<n;++i)result.push_back({first+int64_t(i*(span/int64_t(n))+std::min<uint64_t>(i,span%int64_t(n))),
        first+int64_t((i+1)*(span/int64_t(n))+std::min<uint64_t>(i+1,span%int64_t(n)))});
    return result;
}
bool AfcSource::next(AfcConstraint& constraint)
{
    auto& p=*impl_;auto& s=*p.rows;
    if(p.finished)return false;
    detail::GeometryPhase total("source.afc.decode");bool hasRow;
    {detail::GeometryPhase step("source.afc.sqlite_step");hasRow=s.row();}
    if(!hasRow){
        p.finished=true;
        if(p.full && p.read!=p.count)throw std::invalid_argument("AFC declared count mismatch");
        return false;
    }
    for(int col:{0,1,2})if(sqlite3_column_type(s.value,col)!=SQLITE_INTEGER || s.i64(col)<0)
        throw std::invalid_argument("missing or invalid AFC constraint/fiber ID");
    constraint={};auto& edge=constraint.edge;
    constraint.sourceId=s.i64(0);constraint.sourceFiberA=s.i64(1);constraint.sourceFiberB=s.i64(2);
    {detail::GeometryPhase identity("source.afc.uuid");
    edge={};edge.uuid=util::nameUuid(p.uuidBytes,"constraint:"+std::to_string(s.i64(0)));
    edge.aUuid=util::nameUuid(p.uuidBytes,"fiber:"+std::to_string(s.i64(1)));
    edge.bUuid=util::nameUuid(p.uuidBytes,"fiber:"+std::to_string(s.i64(2)));
    }
    for(int col:{5,6,7,8,9,10,11,12})
        if(sqlite3_column_type(s.value,col)==SQLITE_NULL || !std::isfinite(s.f64(col)))
            throw std::invalid_argument("missing or non-finite AFC position/classification score");
    edge.kind=s.f64(11)>s.f64(12)?"parallel":"perpendicular";
    edge.aAnchor.position=std::array<double,3>{s.f64(5),s.f64(6),s.f64(7)};
    edge.bAnchor.position=std::array<double,3>{s.f64(8),s.f64(9),s.f64(10)};
    edge.aAnchor.wholeGeometry=true;edge.bAnchor.wholeGeometry=true;
    std::array<std::optional<double>,7> values;size_t index=0;
    for(int col:{3,4,11,12,13,14,15}){
        if(sqlite3_column_type(s.value,col)!=SQLITE_NULL){
            if(!std::isfinite(s.f64(col)))throw std::invalid_argument("non-finite AFC scalar");values[index]=s.f64(col);}
        ++index;
    }
    std::array<nlohmann::json,3> fields;
    {detail::GeometryPhase parse("source.afc.json_parse");
    for(int col=16;col<=18;++col)fields[col-16]=
        sqlite3_column_type(s.value,col)==SQLITE_NULL?nlohmann::json(nullptr):nlohmann::json::parse(s.text(col));
    }
    auto& measurement=fields[2];
    for(const auto& [name,position]:{std::pair{"pointABaseXYZ",edge.aAnchor.position},std::pair{"pointBBaseXYZ",edge.bAnchor.position}})
        if(measurement.is_object()&&measurement.contains(name)&&measurement.at(name)==nlohmann::json(*position))measurement.erase(name);
    const bool reversed=detail::canonicalizeEdge(edge);
    constraint.measurement=fiberConstraintRow(edge.uuid,constraint.sourceId,constraint.sourceFiberA,constraint.sourceFiberB,reversed,values,fields[0],fields[1],measurement);
    ++p.read;return true;
}
}
