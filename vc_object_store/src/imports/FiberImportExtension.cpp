#include "vc_object_store/objects/FiberImportExtension.hpp"
#include "vc_object_store/Uuid.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc::objects {
namespace {
using Json=nlohmann::json;
struct Field {const char* name;const char* source;ExtensionType type;};
constexpr Field diagnostics[]={
    {"piece_a","pieceA",ExtensionType::Integer},{"piece_b","pieceB",ExtensionType::Integer},
    {"closest_distance","closestDistanceBaseVoxels",ExtensionType::Real},
    {"hard_continuity","hardContinuity",ExtensionType::Integer},{"artificial_tile_seam","artificialTileSeam",ExtensionType::Integer},
    {"allow_measurement_disable","allowMeasurementDisable",ExtensionType::Integer},{"local_event","localEvent",ExtensionType::Integer},
    {"local_orientation","localOrientation",ExtensionType::Integer},
    {"support_a_begin","supportABegin",ExtensionType::Real},{"support_a_end","supportAEnd",ExtensionType::Real},
    {"support_b_begin","supportBBegin",ExtensionType::Real},{"support_b_end","supportBEnd",ExtensionType::Real},
    {"parallel_correspondence_samples","parallelCorrespondenceSamples",ExtensionType::Integer},
    {"parallel_advance_residual","parallelMeanAdvanceResidualFraction",ExtensionType::Real},
    {"parallel_connector_tangent_dot","parallelMeanConnectorTangentAbsDot",ExtensionType::Real},
    {"parallel_connector_length_change","parallelMeanConnectorLengthChangeFraction",ExtensionType::Real},
    {"parallel_connector_direction_change","parallelMeanConnectorDirectionChange",ExtensionType::Real},
    {"parallel_limit_hit_fraction","parallelLimitHitFraction",ExtensionType::Real},
    {"signed_winding_delta","signedWindingDelta",ExtensionType::Real},
    {"perpendicular_normal_alignment","perpendicularNormalAlignment",ExtensionType::Real},
    {"winding_normal_component","windingNormalComponent",ExtensionType::Real},
    {"signed_parallel_winding_delta","signedParallelWindingDelta",ExtensionType::Real},
    {"parallel_normal_alignment","parallelNormalAlignment",ExtensionType::Real}
};
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
    Json provisioning={{"length_units","base_voxels"},{"winding_units","turns"},{"endpoint_order","source"}};
    if(!source.fiberSourceUuid.empty())provisioning["fiber_source_uuid"]=util::canonicalUuid(source.fiberSourceUuid);
    checksum(provisioning,source.sha256);count(provisioning,"constraint_count",source.constraintCount);
    spec.provisioning=provisioning.dump();
    spec.columns={{"edge_uuid",ExtensionType::Blob},{"source_id",ExtensionType::Integer},
        {"source_fiber_a",ExtensionType::Integer},{"source_fiber_b",ExtensionType::Integer},{"source_reversed",ExtensionType::Integer}};
    for(const auto* name:{"arc_a","arc_b","parallel_score","perpendicular_score","las_winding","parallel_las_winding","evidence_weight"})
        spec.columns.push_back({name,ExtensionType::Real,true});
    for(const auto* endpoint:{"a","b"}){
        const auto prefix=std::string("normal_")+endpoint;
        spec.columns.push_back({prefix+"_valid",ExtensionType::Integer,true});
        for(const auto* axis:{"x","y","z"})spec.columns.push_back({prefix+"_"+axis,ExtensionType::Real,true});
        spec.columns.push_back({prefix+"_reason",ExtensionType::Text,true});
    }
    for(const auto& field:diagnostics)spec.columns.push_back({field.name,field.type,true});
    spec.columns.push_back({"extra",ExtensionType::Text,true});
    spec.primaryKey={"edge_uuid"};spec.indexes={{{"source_id"},false},{{"las_winding"},false}};
    return spec;
}
InitialExtensionRow fiberSourceRow(const std::string& uuid,int64_t sourceId,const std::string& annotation){
    if(!Json::parse(annotation).is_object())throw std::invalid_argument("fiber extension: annotation must be an object");
    return {kFiberSourceTable,{extensionUuid(uuid),sourceId,annotation}};
}
InitialExtensionRow fiberConstraintRow(const std::string& uuid,int64_t id,int64_t a,int64_t b,bool reversed,
    const std::array<std::optional<double>,7>& values,const Json& normalA,const Json& normalB,const Json& measurement){
    InitialExtensionRow out{kFiberConstraintTable,{extensionUuid(uuid),id,a,b,int64_t(reversed)}};
    auto& row=out.values;Json extra=Json::object();
    for(const auto& value:values)row.emplace_back(value?ExtensionValue(*value):ExtensionValue(std::monostate{}));
    for(const auto& [key,normal]:{std::pair{"normal_a",normalA},std::pair{"normal_b",normalB}}){
        if(normal.is_null()){for(int i=0;i<5;++i)row.emplace_back(std::monostate{});continue;}
        if(!normal.is_object())throw std::invalid_argument("fiber extension: normal must be object or null");
        auto remaining=normal;
        row.push_back(scalar(normal.value("valid",Json()),ExtensionType::Integer));remaining.erase("valid");
        const auto xyz=normal.value("xyz",Json());
        if(!xyz.is_null()&&(!xyz.is_array()||xyz.size()!=3))throw std::invalid_argument("fiber extension: invalid normal coordinates");
        for(int i=0;i<3;++i)row.push_back(scalar(xyz.is_null()?Json():xyz[i],ExtensionType::Real));remaining.erase("xyz");
        const auto reason=normal.value("reason",Json());
        row.push_back(reason.is_null()?ExtensionValue(std::monostate{}):ExtensionValue(reason.get<std::string>()));remaining.erase("reason");
        if(!remaining.empty())extra[key]=std::move(remaining);
    }
    if(!measurement.is_null()&&!measurement.is_object())throw std::invalid_argument("fiber extension: measurement must be an object or null");
    auto remaining=measurement.is_null()?Json::object():measurement;
    for(const auto& field:diagnostics){row.push_back(scalar(remaining.value(field.source,Json()),field.type));remaining.erase(field.source);}
    // Remove repeated scalar measurements only if equal: conflicting diagnostic
    // copies remain explicitly visible in extra rather than being discarded.
    const char* repeated[]={"arcABaseVoxels","arcBBaseVoxels","parallelScore","perpendicularScore","windingDistance","parallelWindingDistance","evidenceWeight"};
    for(size_t i=0;i<values.size();++i)if(remaining.contains(repeated[i])&&remaining.at(repeated[i])==(values[i]?Json(*values[i]):Json()))remaining.erase(repeated[i]);
    // XYZ duplicates are removed by the source decoder after comparison with
    // native anchors; arbitrary extra fields remain losslessly available.
    if(!remaining.empty())extra["measurement"]=std::move(remaining);
    row.emplace_back(extra.empty()?ExtensionValue(std::monostate{}):ExtensionValue(extra.dump()));return out;
}
} // namespace vc::objects
