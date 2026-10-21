#include "geometry/DependencyIndex.hpp"
#include "internal/EdgeSql.hpp"
namespace vc::objects::detail {
void createDependencyIndex(sqlite3* db) {
    exec(db,R"SQL(
CREATE TABLE geometry_edge_dependencies(
  edge_id INTEGER NOT NULL REFERENCES edges(id) ON DELETE CASCADE,
  object_id INTEGER NOT NULL REFERENCES objects(id),
  array_name TEXT NOT NULL, point_id INTEGER NOT NULL,
  PRIMARY KEY(object_id,array_name,point_id,edge_id)
) WITHOUT ROWID;
CREATE INDEX geometry_dependencies_edge ON geometry_edge_dependencies(edge_id);
)SQL");
    auto rows=[](const std::string& row) {
        std::string sql;
        for(const auto* endpoint:{"a","b"}) {
            const auto object=row+"."+endpoint+"_id", deps=row+"."+endpoint+"_dependencies";
            // Empty array_name is the conservative global/positional bucket.
            sql+="INSERT OR IGNORE INTO geometry_edge_dependencies SELECT "+row+".id,"+object+",'',-1 WHERE "
                "coalesce(json_array_length("+deps+",'$.points'),0)=0 OR coalesce(json_array_length("+deps+",'$.channels'),0)>0 OR "
                "EXISTS(SELECT 1 FROM json_each("+deps+",'$.points') p WHERE json_extract(p.value,'$.array') NOT IN ('point_ids','control_point_ids'));";
            sql+="INSERT OR IGNORE INTO geometry_edge_dependencies SELECT "+row+".id,"+object+",json_extract(p.value,'$.array'),json_extract(p.value,'$.index[0]') " +
                "FROM json_each("+deps+",'$.points') p WHERE json_extract(p.value,'$.array') IN ('point_ids','control_point_ids');";
        }
        return sql;
    };
    exec(db,("CREATE TRIGGER geometry_dependencies_insert AFTER INSERT ON edges BEGIN "+rows("new")+" END;").c_str());
    exec(db,("CREATE TRIGGER geometry_dependencies_update AFTER UPDATE OF a_id,b_id,a_dependencies,b_dependencies ON edges BEGIN "
        "DELETE FROM geometry_edge_dependencies WHERE edge_id=old.id;"+rows("new")+" END;").c_str());
    // Revision allocation and latest-ledger queries must not sort full history.
    exec(db,"CREATE INDEX geometry_revision_generation ON geometry_revisions(object_uuid,object_generation DESC);");
}
std::vector<EdgeRecord> structuralEdgeCandidates(sqlite3* db,const std::string& uuid,const std::string& array,const std::vector<int64_t>& ids) {
    const auto object=*objectIdByUuid(db,parseUuid(uuid));std::set<int64_t> candidates;
    Statement global(db,"SELECT edge_id FROM geometry_edge_dependencies WHERE object_id=? AND array_name='' AND point_id=-1");global.bind(1,object);
    while(global.row())candidates.insert(global.i64(0));
    Statement points(db,"SELECT edge_id FROM geometry_edge_dependencies WHERE object_id=? AND array_name=? AND point_id=?");
    for(auto id:ids) {points.reset();points.bind(1,object);points.bind(2,array);points.bind(3,id);while(points.row())candidates.insert(points.i64(0));}
    std::vector<EdgeRecord> result;Statement edge(db,std::string("SELECT ")+kEdgeCols+" FROM edges WHERE id=?");
    for(auto id:candidates) {edge.reset();edge.bind(1,id);if(edge.row())result.push_back(readEdgeRow(db,edge));}return result;
}
}
