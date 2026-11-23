#include "geometry/DependencyIndex.hpp"
#include "internal/EdgeSql.hpp"
#include "internal/RecordResolver.hpp"
namespace vc::objects::detail {
void createGeometryHistoryIndex(sqlite3* db) {
    // Revision allocation and latest-ledger queries must not sort full history.
    // Relationship dependencies themselves are already native edge fields;
    // do not duplicate every edge in a trigger-maintained materialization.
    if (historyEnabled(db)) exec(db,"CREATE INDEX geometry_revision_version ON geometry_revisions(object_uuid,object_version DESC);");
}
std::vector<EdgeRecord> structuralEdgeCandidates(sqlite3* db,const std::string& uuid,const std::string& array,const std::vector<int64_t>& ids) {
    const auto canonical=formatUuid(parseUuid(uuid));
    const auto object=resolveObject(db,canonical);
    if(!object)throw std::invalid_argument("geometry dependencies: unknown object");
    const std::set<int64_t> changed(ids.begin(),ids.end());
    const auto affected=[&](const Anchor& anchor){
        if(anchor.points.empty()||!anchor.channels.empty())return true;
        for(const auto& point:anchor.points){
            if(point.array!="point_ids"&&point.array!="control_point_ids")return true;
            if(point.array==array&&!point.index.empty()&&changed.contains(point.index.front()))return true;
        }
        return false;
    };
    std::vector<EdgeRecord> result;
    for(auto& edge:incidentEdges(db,canonical)){
        if((edge.aUuid==canonical&&affected(edge.aAnchor))||(edge.bUuid==canonical&&affected(edge.bAnchor)))
            result.push_back(std::move(edge));
    }
    return result;
}
}
