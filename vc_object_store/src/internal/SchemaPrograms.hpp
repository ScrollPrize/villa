#pragma once
#include "internal/Sqlite.hpp"
#include <set>

namespace vc::objects::detail {
inline std::string quoteSqlIdentifier(const std::string& identifier)
{
    std::string result="\"";for(auto c:identifier){result+=c;if(c=='"')result+='"';}return result+'"';
}
struct SchemaProgram {std::string type,name,owner,sql;};
inline std::vector<SchemaProgram> schemaPrograms(sqlite3* db)
{
    std::vector<SchemaProgram> result;
    Statement rows(db,"SELECT type,name,tbl_name,sql FROM sqlite_schema WHERE type IN ('trigger','view') AND sql IS NOT NULL ORDER BY rowid");
    while(rows.row())result.push_back({rows.text(0),rows.text(1),rows.text(2),rows.text(3)});return result;
}
inline void suspendPrograms(sqlite3* db,const std::vector<SchemaProgram>& programs,bool views)
{
    for(const auto* type:{"trigger","view"})for(const auto& program:programs)
        if(program.type==type&&(views||program.type=="trigger"))exec(db,("DROP "+program.type+" "+quoteSqlIdentifier(program.name)).c_str());
}
inline void restorePrograms(sqlite3* db,const std::vector<SchemaProgram>& programs,bool views,bool validateTriggers=true)
{
    for(const auto* type:{"view","trigger"})for(const auto& program:programs)
        if(program.type==type&&(views||program.type=="trigger"))exec(db,program.sql.c_str());
    std::set<std::string> owners;
    for(const auto& program:programs){
        if(program.type=="trigger")owners.insert(program.owner);
        else{Statement check(db,"SELECT * FROM "+quoteSqlIdentifier(program.name)+" LIMIT 0");}
    }
    if(!validateTriggers)return;
    for(const auto& owner:owners){
        Statement kind(db,"SELECT type FROM sqlite_schema WHERE name=?",owner);
        if(!kind.row()||kind.text(0)!="table")throw std::runtime_error("maintenance: cannot validate extension trigger on non-table "+owner);
        std::string assignments;Statement columns(db,"SELECT name FROM pragma_table_xinfo(?) WHERE hidden=0",owner);
        while(columns.row()){if(!assignments.empty())assignments+=',';assignments+=quoteSqlIdentifier(columns.text(0))+"="+quoteSqlIdentifier(columns.text(0));}
        Statement insert(db,"EXPLAIN INSERT INTO "+quoteSqlIdentifier(owner)+" DEFAULT VALUES");
        Statement update(db,"EXPLAIN UPDATE "+quoteSqlIdentifier(owner)+" SET "+assignments+" WHERE 0");
        Statement erase(db,"EXPLAIN DELETE FROM "+quoteSqlIdentifier(owner)+" WHERE 0");
    }
}
} // namespace vc::objects::detail
