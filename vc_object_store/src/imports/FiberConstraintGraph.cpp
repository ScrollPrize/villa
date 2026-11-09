#include "vc_object_store/objects/FiberConstraintGraph.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace vc::objects {
std::vector<FiberConstraintRecord> readFiberConstraintGraph(const ObjectCollection& store,const CollectionTxn& txn)
{
    const auto tables=store.extensionTables(txn);
    const auto schema=std::find_if(tables.begin(),tables.end(),[](const auto& t){return t.name==kFiberConstraintTable;});
    if(schema==tables.end())throw std::invalid_argument("store has no fiber constraint measurements");
    if(schema->formatVersion!=1&&schema->formatVersion!=2)
        throw std::invalid_argument("unsupported fiber constraint format");
    std::unordered_map<std::string,size_t> columns;
    for(size_t i=0;i<schema->columns.size();++i)columns.emplace(schema->columns[i].name,i);
    const auto numeric=[&](const ExtensionRow& r,const std::string& key){
        const auto& v=r.at(columns.at(key));double out;
        if(const auto* x=std::get_if<double>(&v))out=*x;
        else if(const auto* x=std::get_if<int64_t>(&v))out=static_cast<double>(*x);
        else throw std::invalid_argument("missing numeric constraint field: "+key);
        if(!std::isfinite(out))throw std::invalid_argument("nonfinite constraint field: "+key);
        return out;
    };
    std::unordered_map<std::string,EdgeRecord> edges;
    for(const auto* kind:{"parallel","perpendicular"})for(auto& e:store.edgesByKind(txn,kind))
        edges.emplace(e.record.uuid,std::move(e.record));
    std::vector<FiberConstraintRecord> result;result.reserve(edges.size());
    store.scanExtensionRows(txn,kFiberConstraintTable,[&](const ExtensionRow& row){
        const auto uuid=extensionUuid(row.at(columns.at("edge_uuid")));
        const auto found=edges.find(uuid);
        if(found==edges.end())throw std::invalid_argument("measurement references missing/wrong-kind edge: "+uuid);
        FiberConstraintRecord r;r.edge=std::move(found->second);edges.erase(found);
        r.measurement={kFiberConstraintTable,row};
        r.sourceId=std::get<int64_t>(row.at(columns.at("source_id")));
        const bool reversed=std::get<int64_t>(row.at(columns.at("source_reversed")))!=0;
        r.a=reversed?r.edge.bUuid:r.edge.aUuid;r.b=reversed?r.edge.aUuid:r.edge.bUuid;
        const auto& aa=reversed?r.edge.bAnchor:r.edge.aAnchor;
        const auto& ba=reversed?r.edge.aAnchor:r.edge.bAnchor;
        if(!aa.position||!ba.position)throw std::invalid_argument("measured constraint has no endpoint positions");
        if(schema->formatVersion==1){r.pointA=*aa.position;r.pointB=*ba.position;}
        else for(size_t i=0;i<3;++i){const std::string axis(1,"xyz"[i]);
            r.pointA[i]=numeric(row,"point_a_"+axis);r.pointB[i]=numeric(row,"point_b_"+axis);
        }
        r.arcA=numeric(row,"arc_a");r.arcB=numeric(row,"arc_b");
        r.parallelScore=numeric(row,"parallel_score");r.perpendicularScore=numeric(row,"perpendicular_score");
        r.windingDistance=numeric(row,"las_winding");r.parallelWindingDistance=numeric(row,"parallel_las_winding");
        r.evidenceWeight=numeric(row,"evidence_weight");
        r.validA=numeric(row,"normal_a_valid")!=0;r.validB=numeric(row,"normal_b_valid")!=0;
        for(size_t i=0;i<3;++i){const std::string axis(1,"xyz"[i]);
            if(r.validA)r.normalA[i]=numeric(row,"normal_a_"+axis);
            if(r.validB)r.normalB[i]=numeric(row,"normal_b_"+axis);}
        result.push_back(std::move(r));return true;
    });
    std::sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.sourceId<b.sourceId;});
    for(size_t i=1;i<result.size();++i)if(result[i-1].sourceId==result[i].sourceId)
        throw std::invalid_argument("constraint source IDs collide: select one extracted graph");
    return result;
}
} // namespace vc::objects
