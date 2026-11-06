#pragma once
#include <map>
#include <string>

namespace vc::objects::detail {
// Retained history triggers whose endpoint columns changed in v2. Conversion
// recognizes only these exact native definitions and installs the replacement
// from the native schema, rather than rewriting arbitrary extension SQL.
inline std::map<std::string,std::string> legacyEndpointHistoryTriggers()
{
    std::map<std::string,std::string> result;
    for(const auto* operation:{"INSERT","UPDATE","DELETE"}) {
        const std::string row=std::string(operation)=="DELETE"?"old":"new";
        const auto suffix=std::string("_history_")+operation;
        result["edges"+suffix]="CREATE TRIGGER edges"+suffix+" AFTER "+operation+" ON edges BEGIN "
            "INSERT INTO history_dirty_edges VALUES("+row+".uuid) ON CONFLICT(uuid) DO NOTHING;"
            "INSERT INTO history_dirty_objects SELECT uuid FROM objects WHERE id IN ("+row+".a_id,"+row+".b_id) ON CONFLICT(uuid) DO NOTHING; END";
        result["edge_tags"+suffix]="CREATE TRIGGER edge_tags"+suffix+" AFTER "+operation+" ON edge_tags BEGIN "
            "INSERT INTO history_dirty_edges SELECT uuid FROM edges WHERE id="+row+".edge_id ON CONFLICT(uuid) DO NOTHING;"
            "INSERT INTO history_dirty_objects SELECT o.uuid FROM objects o JOIN edges e ON o.id=e.a_id OR o.id=e.b_id "
            "WHERE e.id="+row+".edge_id ON CONFLICT(uuid) DO NOTHING; END";
    }
    return result;
}
// Schema-17 migration specification only. These definitions are not installed
// by native v2 provisioning. Keeping the exact old definitions lets conversion
// remove native bookkeeping while rejecting an extension using the same name.
inline std::map<std::string,std::string> legacyCurrentTriggers()
{
    std::map<std::string,std::string> result;
    for(const auto* operation:{"INSERT","UPDATE","DELETE"}) {
        const std::string row=std::string(operation)=="DELETE"?"old":"new";
        const auto headName=std::string("geometry_heads_history_")+operation;
        result[headName]="CREATE TRIGGER "+headName+" AFTER "+operation+" ON geometry_heads BEGIN "
            "INSERT INTO history_dirty_objects VALUES("+row+".object_uuid) ON CONFLICT(uuid) DO NOTHING; END";
        const auto name=std::string("edge_geometry_bindings_history_")+operation;
        result[name]="CREATE TRIGGER "+name+" AFTER "+operation+" ON edge_geometry_bindings BEGIN "
            "INSERT INTO history_dirty_edges SELECT uuid FROM edges WHERE id="+row+".edge_id ON CONFLICT(uuid) DO NOTHING;"
            "INSERT INTO history_dirty_objects SELECT o.uuid FROM objects o JOIN edges e ON o.id=e.a_id OR o.id=e.b_id "
            "WHERE e.id="+row+".edge_id ON CONFLICT(uuid) DO NOTHING; END";
    }
    const auto rows=[](const std::string& row){
        std::string sql;
        for(const auto* endpoint:{"a","b"}) {
            const auto object=row+"."+endpoint+"_id",deps=row+"."+endpoint+"_dependencies";
            sql+="INSERT OR IGNORE INTO geometry_edge_dependencies SELECT "+row+".id,"+object+",'',-1 WHERE "
                "coalesce(json_array_length("+deps+",'$.points'),0)=0 OR coalesce(json_array_length("+deps+",'$.channels'),0)>0 OR "
                "EXISTS(SELECT 1 FROM json_each("+deps+",'$.points') p WHERE json_extract(p.value,'$.array') NOT IN ('point_ids','control_point_ids'));";
            sql+="INSERT OR IGNORE INTO geometry_edge_dependencies SELECT "+row+".id,"+object+",json_extract(p.value,'$.array'),json_extract(p.value,'$.index[0]') "
                "FROM json_each("+deps+",'$.points') p WHERE json_extract(p.value,'$.array') IN ('point_ids','control_point_ids');";
        }
        return sql;
    };
    result["geometry_dependencies_insert"]="CREATE TRIGGER geometry_dependencies_insert AFTER INSERT ON edges BEGIN "+rows("new")+" END";
    result["geometry_dependencies_update"]="CREATE TRIGGER geometry_dependencies_update AFTER UPDATE OF a_id,b_id,a_dependencies,b_dependencies ON edges BEGIN "
        "DELETE FROM geometry_edge_dependencies WHERE edge_id=old.id;"+rows("new")+" END";
    return result;
}
}
