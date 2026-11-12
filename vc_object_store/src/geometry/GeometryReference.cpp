#include "geometry/GeometryReference.hpp"
#include "geometry/GeometryDelta.hpp"
#include "internal/ReferenceResolver.hpp"

namespace vc::objects::detail {
GeometryRecord decodeGeometryRecord(const Statement& s,int c)
{
    GeometryRecord record;
    record.source = sqlite3_db_handle(s.value);
    auto& r = record.revision;
    r.uuid = readGeometryTokenColumn(s,c); r.objectUuid = s.uuid(c+1);
    if (!s.isNull(c+2)) r.parentUuid = readGeometryTokenColumn(s,c+2);
    r.contentHash = sqlite3_column_type(s.value,c+3)==SQLITE_TEXT ? s.text(c+3) : readDigest(s,c+3);
    record.manifest = s.text(c+4);
    record.descriptor = loadGeometryManifest(record.source,r.uuid,record.manifest,r.contentHash);
    const auto& manifest = record.descriptor;
    r.gridOrigin = manifest.value("grid_origin",std::array<int64_t,2>{0,0});
    r.kind = manifest.at("kind"); r.formatVersion = manifest.at("format_version");
    r.coordinateSystem = manifest.at("coordinate_system"); r.attributes = manifest.at("attributes").dump();
    r.objectVersion = s.i64(c+5); r.createdBy = s.text(c+6); r.createdAt = s.text(c+7); r.createdHost = s.text(c+8);
    for (const auto& [name,a] : manifest.at("arrays").items())
        r.arrays.push_back({name,a.at("dtype"),a.at("shape").get<std::vector<int64_t>>(),a.at("byte_length")});
    return record;
}


std::optional<GeometryRecord> resolveGeometryRecord(sqlite3* db,const std::string& reference)
{
    const std::array<std::string,1> references{reference};
    return resolveGeometryRecords(db,references).front();
}
std::vector<std::optional<GeometryRecord>> resolveGeometryRecords(sqlite3* db,std::span<const std::string> references)
{
    std::vector<std::optional<GeometryRecord>> result(references.size());
    size_t missing=references.size();
    // Archives retain parent/delta information. Prefer them throughout the
    // pinned scope; an overlay may hold only a verbatim current descriptor.
    for(bool archived:{true,false}) {
        if(!missing)break;
        resolveReference(db,[&](sqlite3* source){
            if(archived&&!historyEnabled(source))return false;
            std::vector<size_t> indices;
            for(size_t i=0;i<references.size();++i)if(!result[i])indices.push_back(i);
            constexpr size_t window=256;
            for(size_t begin=0;begin<indices.size();begin+=window) {
                const auto count=std::min(window,indices.size()-begin);
                std::string sql="WITH requested(pos,uuid) AS (VALUES ";
                for(size_t j=0;j<count;++j)sql+=(j?",":"")+std::string("(")+std::to_string(indices[begin+j])+",?)";
                sql+=archived?
                    ") SELECT q.pos,r.uuid,r.object_uuid,r.parent_uuid,r.content_hash,r.manifest,r.object_version,r.created_by,r.created_at,r.created_host FROM requested q JOIN geometry_revisions r ON r.uuid=q.uuid":
                    ") SELECT q.pos,r.storage_ref,r.uuid,NULL,r.data_checksum,r.geometry_manifest,r.version,r.changed_by,r.changed_at,r.changed_host FROM requested q JOIN objects r ON r.storage_ref=q.uuid WHERE r.state<>2 AND r.geometry_manifest IS NOT NULL";
                Statement rows(source,sql);
                for(size_t j=0;j<count;++j) {
                    const auto& ref=references[indices[begin+j]];
                    if(archived)rows.bind(int(j+1),parseUuid(ref));
                    else rows.bind(int(j+1),canonicalGeometryToken(ref));
                }
                while(rows.row()) {
                    const auto index=size_t(rows.i64(0));
                    if(result[index])throw std::runtime_error("geometry: duplicate revision identity");
                    result[index]=decodeGeometryRecord(rows,1);
                    result[index]->archived=archived;
                    --missing;
                }
            }
            return !missing;
        });
    }
    return result;
}
}
