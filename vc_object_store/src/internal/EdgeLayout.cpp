#include "internal/EdgeLayout.hpp"
#include "internal/Database.hpp"
#include "vc_object_store/objects/EdgeKinds.hpp"
#include <algorithm>

namespace vc::objects::detail {
namespace {
constexpr EdgeColumnSpec columns[]{
    {"uuid","BLOB PRIMARY KEY CHECK(length(uuid)=16)",nullptr},
    {"state","INTEGER NOT NULL DEFAULT 0 CHECK(state=0 OR state=2)","0"},
    {"a_uuid","BLOB NOT NULL CHECK(length(a_uuid)=16)",nullptr},
    {"b_uuid","BLOB NOT NULL CHECK(length(b_uuid)=16)",nullptr},
    {"kind","INTEGER NOT NULL CHECK(typeof(kind)='integer' AND kind>=0 AND kind<10)",nullptr},
    {"commit_sequence","INTEGER NOT NULL DEFAULT 0",nullptr},
    {"version","INTEGER NOT NULL DEFAULT 1 CHECK(version>0)","1"},
    {"version_generation","BLOB CHECK(version_generation IS NULL OR length(version_generation)=16)","NULL"},
    {"created_by","TEXT NOT NULL DEFAULT ''","''"},
    {"created_at","TEXT NOT NULL DEFAULT ''","''"},
    {"created_host","TEXT NOT NULL DEFAULT ''","''"},
    {"changed_by","TEXT NOT NULL DEFAULT ''","''"},
    {"changed_at","TEXT NOT NULL DEFAULT ''","''"},
    {"changed_host","TEXT NOT NULL DEFAULT ''","''"},
    {"approval_state","INTEGER NOT NULL DEFAULT 0 CHECK(approval_state=0 OR approval_state=1 OR approval_state=2)","0"},
    {"approved_by","TEXT","NULL"},{"approved_at","TEXT","NULL"},{"approved_host","TEXT","NULL"},
    {"approved_version","INTEGER","NULL"},
    {"a_anchor_x","REAL","NULL"},{"a_anchor_y","REAL","NULL"},{"a_anchor_z","REAL","NULL"},
    {"a_dir_x","REAL","NULL"},{"a_dir_y","REAL","NULL"},{"a_dir_z","REAL","NULL"},
    {"b_anchor_x","REAL","NULL"},{"b_anchor_y","REAL","NULL"},{"b_anchor_z","REAL","NULL"},
    {"b_dir_x","REAL","NULL"},{"b_dir_y","REAL","NULL"},{"b_dir_z","REAL","NULL"},
    {"meta","TEXT CHECK(meta IS NULL OR meta='' OR (json_valid(meta) AND json_type(meta)='object'))","NULL"},
    {"a_dependencies","TEXT CHECK(a_dependencies IS NULL OR a_dependencies='' OR (json_valid(a_dependencies) AND json_type(a_dependencies)='object'))","NULL"},
    {"b_dependencies","TEXT CHECK(b_dependencies IS NULL OR b_dependencies='' OR (json_valid(b_dependencies) AND json_type(b_dependencies)='object'))","NULL"}
};
}
std::span<const EdgeColumnSpec> edgeColumnSpecs(){return columns;}
std::set<std::string> selectedEdgeColumns(const OpenOptions& options){
    std::set<std::string> result;
    for(const auto& c:columns)if(!options.edgeColumns||!c.fallback)result.insert(c.name);
    if(options.edgeColumns)for(const auto& name:*options.edgeColumns){
        if(std::none_of(std::begin(columns),std::end(columns),[&](const auto& c){return name==c.name;}))
            throw std::invalid_argument("unknown optional edge column: "+name);
        result.insert(name);
    }
    return result;
}
std::set<std::string> edgeColumns(sqlite3* db){
    if(auto p=activeStatementPools.find(db);p!=activeStatementPools.end()&&p->second->edgeColumnNames)
        return *p->second->edgeColumnNames;
    std::set<std::string> names;Statement query(db,"SELECT name FROM pragma_table_info('edges')");
    while(query.row())names.insert(query.text(0));
    for(const auto& c:columns)if(!c.fallback&&!names.contains(c.name))
        throw std::runtime_error("missing required edge column: "+std::string(c.name));
    if(auto p=activeStatementPools.find(db);p!=activeStatementPools.end())p->second->edgeColumnNames=names;
    return names;
}
bool hasEdgeColumn(sqlite3* db,const std::string& name){return edgeColumns(db).contains(name);}
void requireEdgeColumns(sqlite3* db,std::initializer_list<const char*> names){
    auto present=edgeColumns(db);for(const auto* name:names)if(!present.contains(name))
        throw std::runtime_error("edge field is not supported by this table layout: "+std::string(name));
}
void requireAdjacencyIndex(sqlite3* db){
    Statement q(db,"SELECT count(*) FROM sqlite_schema WHERE type='index' AND name IN ('edges_a','edges_b')");q.row();
    if(q.i64(0)!=2)throw std::runtime_error("edge adjacency index is absent; explicitly build it before adjacency queries");
}
void createEdgeSchema(sqlite3* db,const OpenOptions& options,bool immutable){
    const auto selected=selectedEdgeColumns(options);std::string sql="CREATE TABLE edges(";
    bool first=true;for(const auto& c:columns)if(selected.contains(c.name)){
        if(!first)sql+=',';first=false;sql+=std::string(c.name)+' '+c.declaration;
    }
    sql+=",CHECK(a_uuid<>b_uuid";
    if(selected.contains("a_dependencies")&&selected.contains("b_dependencies"))
        sql+=" OR (coalesce(json_array_length(nullif(a_dependencies,''),'$.points'),0)>0 AND coalesce(json_array_length(nullif(b_dependencies,''),'$.points'),0)>0 AND a_dependencies<>b_dependencies)";
    sql+=")) WITHOUT ROWID";exec(db,sql.c_str());
    exec(db,"CREATE TABLE edge_tags(edge_uuid BLOB NOT NULL REFERENCES edges(uuid) ON DELETE CASCADE,tag TEXT NOT NULL,PRIMARY KEY(edge_uuid,tag)) WITHOUT ROWID;CREATE INDEX edge_tags_tag ON edge_tags(tag);");
    if(options.edgeAdjacencyIndex)exec(db,"CREATE INDEX edges_a ON edges(a_uuid,kind);CREATE INDEX edges_b ON edges(b_uuid,kind);");
    if(!immutable){
        exec(db,"CREATE INDEX edges_kind_uuid ON edges(kind,uuid);");
        if(selected.contains("approval_state"))exec(db,"CREATE INDEX edges_kind_approval_uuid ON edges(kind,approval_state,uuid);");
    }
}
std::string edgeReadColumn(sqlite3* db,const std::string& name,const std::string& prefix){
    auto specs=edgeColumnSpecs();auto found=std::find_if(specs.begin(),specs.end(),[&](const auto& c){return name==c.name;});
    if(found==specs.end())throw std::invalid_argument("unknown edge column: "+name);
    std::string expression;
    if(hasEdgeColumn(db,name))expression=prefix+name;
    else if(found->fallback)expression=found->fallback;
    else throw std::runtime_error("missing required edge column: "+name);
    return expression;
}
std::string edgeReadSource(sqlite3* db){
    std::string sql="(SELECT ";bool first=true;for(const auto& c:columns){
        if(!first)sql+=',';first=false;sql+=edgeReadColumn(db,c.name)+" AS "+c.name;
    }
    return sql+" FROM edges)";
}
} // namespace vc::objects::detail
