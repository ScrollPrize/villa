#include "vc_object_store/objects/FiberImportExtension.hpp"
#include "vc_object_store/Uuid.hpp"
#include <vc_reader/compact_normal.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <bit>
#include <limits>

namespace vc::objects {
namespace {
using Json=nlohmann::json;
ExtensionValue scalar(const Json& v,ExtensionType type){
    if(v.is_null())return std::monostate{};
    if(type==ExtensionType::Integer){
        if(v.is_boolean())return int64_t(v.get<bool>());
        if(!v.is_number_integer())throw std::invalid_argument("fiber extension: expected integer diagnostic");
        return v.get<int64_t>();}
    if(!v.is_number()||!std::isfinite(v.get<double>()))throw std::invalid_argument("fiber extension: expected finite diagnostic");
    return v.get<double>();
}
void checksum(Json& provisioning,const std::optional<std::string>& sha256){
    if(!sha256)return;
    if(sha256->size()!=64||!std::all_of(sha256->begin(),sha256->end(),[](unsigned char c){
        return (c>='0'&&c<='9')||(c>='a'&&c<='f')||(c>='A'&&c<='F');}))
        throw std::invalid_argument("fiber extension: source SHA-256 must contain 64 hex digits");
    auto value=*sha256;
    for(auto& c:value)if(c>='A'&&c<='F')c=char(c-'A'+'a');
    provisioning["source_sha256"]=std::move(value);
}
void count(Json& provisioning,const char* name,const std::optional<int64_t>& value){
    if(!value)return;
    if(*value<0)throw std::invalid_argument("fiber extension: source count must not be negative");
    provisioning[name]=*value;
}
}
std::vector<uint8_t> encodeFiberArc(double value){
    const float f=static_cast<float>(value);
    if(!std::isfinite(value)||value<0.||!std::isfinite(f))throw std::invalid_argument("fiber arc must be finite, nonnegative float32");
    const auto bits=std::bit_cast<uint32_t>(f);std::vector<uint8_t> bytes(4);
    for(unsigned i=0;i<4;++i)bytes[i]=uint8_t(bits>>(8*i));return bytes;
}
double decodeFiberArc(std::span<const uint8_t> bytes){
    if(bytes.size()!=4)throw std::invalid_argument("fiber arc must be a four-byte float32 BLOB");
    uint32_t bits=0;for(unsigned i=0;i<4;++i)bits|=uint32_t(bytes[i])<<(8*i);
    const auto f=std::bit_cast<float>(bits);
    if(!std::isfinite(f)||f<0.f)throw std::invalid_argument("invalid encoded fiber arc");return f;
}
std::vector<uint8_t> encodeFiberDirection(const std::array<double,3>& direction){
    const auto bytes=vcr::normal::encodeSignedCompact(direction);return {bytes.begin(),bytes.end()};
}
std::array<double,3> decodeFiberDirection(std::span<const uint8_t> bytes){
    if(bytes.size()!=3)throw std::invalid_argument("fiber direction must be a three-byte BLOB");
    return vcr::normal::decodeSignedCompact({bytes[0],bytes[1],bytes[2]});
}
std::optional<uint8_t> encodeFiberWindingDistance(double distance){
    if(!std::isfinite(distance)||distance<0.)throw std::invalid_argument("fiber winding distance must be finite and nonnegative");
    if(distance>kMaximumCompactWindingDistance)return {};
    return uint8_t(std::lround(distance*16.)+1);
}
double decodeFiberWindingDistance(std::span<const uint8_t> bytes){
    if(bytes.size()!=1||bytes[0]==0)throw std::invalid_argument("invalid encoded fiber winding distance");
    return (unsigned(bytes[0])-1)/16.;
}
ExtensionTable fiberSourceExtension(const FiberSourceInfo& source){
    ExtensionTable spec;spec.name=kFiberSourceTable;spec.extension=kFiberGraphSolveExtension;spec.keySpace="objects";
    Json provisioning=Json::object();
    if(!source.uuid.empty())provisioning["source_uuid"]=util::canonicalUuid(source.uuid);
    checksum(provisioning,source.sha256);count(provisioning,"fiber_count",source.fiberCount);count(provisioning,"point_count",source.pointCount);
    spec.provisioning=provisioning.dump();
    spec.columns={{"fiber_uuid",ExtensionType::Blob},{"source_id",ExtensionType::Integer},{"annotation",ExtensionType::Text}};
    spec.primaryKey={"fiber_uuid"};spec.indexes={{{"source_id"},false}};return spec;
}
ExtensionTable fiberConstraintExtension(const FiberConstraintSourceInfo& source,std::span<const ExtensionSource> basedOn){
    ExtensionTable spec;spec.name=kFiberConstraintTable;spec.extension=kFiberGraphSolveExtension;spec.keySpace="edges";
    spec.basedOn.assign(basedOn.begin(),basedOn.end());
    spec.formatVersion=kFiberConstraintFormat;
    Json provisioning={{"length_units","base_voxels"},{"winding_units","turns"},{"endpoint_order","canonical"},
        {"arc_encoding","float32-le"},{"direction_encoding","lasagna-xy-u8-hemisphere-i8"},
        {"connector_direction","a_to_b"},{"winding_scale",16},{"winding_code_offset",1},{"winding_invalid_code",0}};
    if(!source.fiberSourceUuid.empty())provisioning["fiber_source_uuid"]=util::canonicalUuid(source.fiberSourceUuid);
    if(source.coordinateSpace){
        if(source.coordinateSpace->empty())throw std::invalid_argument("fiber extension: empty measurement frame");
        provisioning["coordinate_space"]=*source.coordinateSpace;
    }
    checksum(provisioning,source.sha256);count(provisioning,"constraint_count",source.constraintCount);
    spec.provisioning=provisioning.dump();
    spec.columns={{"edge_uuid",ExtensionType::Blob},{"arc_a",ExtensionType::Blob},{"arc_b",ExtensionType::Blob},
        {"normal_a",ExtensionType::Blob},{"normal_b",ExtensionType::Blob},{"connector",ExtensionType::Blob},
        {"winding_distance",ExtensionType::Blob}};
    spec.columns.push_back({"extra",ExtensionType::Text,true});
    spec.primaryKey={"edge_uuid"};
    return spec;
}
InitialExtensionRow fiberSourceRow(const std::string& uuid,int64_t sourceId,const std::string& annotation){
    if(!Json::parse(annotation).is_object())throw std::invalid_argument("fiber extension: annotation must be an object");
    return {kFiberSourceTable,{extensionUuid(uuid),sourceId,annotation}};
}
InitialExtensionRow fiberConstraintRow(const std::string& uuid,int64_t id,int64_t a,int64_t b,bool reversed,
    FiberConstraintData data){
    (void)id;(void)a;(void)b;
    if(!data.values[0]||!data.values[1])throw std::invalid_argument("fiber measurement requires both arc positions");
    const bool parallel=data.values[2].value_or(0.)>data.values[3].value_or(0.);
    const auto distance=parallel?data.values[5]:data.values[4];
    if(!distance)throw std::invalid_argument("fiber measurement requires winding distance");
    const auto code=encodeFiberWindingDistance(*distance);
    if(!code)throw std::out_of_range("fiber measurement exceeds 15.875 windings; skip it before insertion");
    if(!data.measuredPoints)throw std::invalid_argument("fiber measurement construction requires source positions for A-to-B direction");
    std::array<double,3> connector{};
    for(size_t i=0;i<3;++i)connector[i]=(*data.measuredPoints)[1][i]-(*data.measuredPoints)[0][i];
    auto arcs=std::array<double,2>{*data.values[0],*data.values[1]};
    const auto normal=[](const FiberConstraintNormal& n){std::array<double,3> out{};
        if(n.valid.value_or(0))for(size_t i=0;i<3;++i){if(!n.xyz[i])throw std::invalid_argument("valid normal misses a component");out[i]=*n.xyz[i];}
        return out;};
    std::array<std::array<double,3>,2> normals{normal(data.normals[0]),normal(data.normals[1])};
    if(reversed){std::swap(arcs[0],arcs[1]);std::swap(normals[0],normals[1]);for(auto& x:connector)x=-x;}
    InitialExtensionRow out{kFiberConstraintTable,{extensionUuid(uuid),encodeFiberArc(arcs[0]),encodeFiberArc(arcs[1]),
        encodeFiberDirection(normals[0]),encodeFiberDirection(normals[1]),encodeFiberDirection(connector),std::vector<uint8_t>{*code},std::monostate{}}};
    if(data.extra){auto value=Json::parse(*data.extra);if(!value.is_object())throw std::invalid_argument("fiber extra must be a JSON object");if(!value.empty())out.values.back()=value.dump();}
    return out;
}
InitialExtensionRow fiberConstraintRow(const std::string& uuid,int64_t id,int64_t a,int64_t b,bool reversed,
    const std::array<std::optional<double>,7>& values,const Json& normalA,const Json& normalB,const Json& measurement,
    const std::optional<std::array<std::array<double,3>,2>>& measuredPoints){
    FiberConstraintData data;data.values=values;data.measuredPoints=measuredPoints;
    size_t endpoint=0;
    for(const auto& [key,normal]:{std::pair{"normal_a",normalA},std::pair{"normal_b",normalB}}){
        auto& typed=data.normals[endpoint++];
        if(normal.is_null())continue;
        if(!normal.is_object())throw std::invalid_argument("fiber extension: normal must be object or null");
        auto valid=scalar(normal.value("valid",Json()),ExtensionType::Integer);
        if(!std::holds_alternative<std::monostate>(valid))typed.valid=std::get<int64_t>(valid);
        const auto xyz=normal.value("xyz",Json());
        if(!xyz.is_null()&&(!xyz.is_array()||xyz.size()!=3))throw std::invalid_argument("fiber extension: invalid normal coordinates");
        if(!xyz.is_null()){
            for(int i=0;i<3;++i){
                auto value=scalar(xyz[i],ExtensionType::Real);
                if(!std::holds_alternative<std::monostate>(value))typed.xyz[i]=std::get<double>(value);
            }
        }
        (void)key;
    }
    if(!measurement.is_null()&&!measurement.is_object())throw std::invalid_argument("fiber extension: measurement must be an object or null");
    return fiberConstraintRow(uuid,id,a,b,reversed,std::move(data));
}
} // namespace vc::objects
