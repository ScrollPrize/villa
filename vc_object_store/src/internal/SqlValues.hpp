#pragma once
#include "internal/Sqlite.hpp"
#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace vc::objects::detail {
// One checked binding path for single-row editing and initial multi-row writes.
// Caller owns transactional rollback; SQL constraints remain enabled unchanged.
template<class Bind>
void writeSqlValues(sqlite3* db,std::string_view prefix,std::string_view suffix,
                    size_t columns,size_t count,size_t maxRows,Bind&& bind)
{
    if(!maxRows||!columns) throw std::invalid_argument("SQL VALUES batch dimensions must be positive");
    const auto variables=sqlite3_limit(db,SQLITE_LIMIT_VARIABLE_NUMBER,-1);
    const auto rows=std::min(maxRows,size_t(variables)/columns);
    if(!rows) throw std::runtime_error("SQLite variable limit cannot hold one record");
    std::string pattern="(";
    for(size_t i=0;i<columns;++i) {if(i)pattern+=',';pattern+='?';}
    pattern+=')';
    for(size_t begin=0;begin<count;) {
        const auto n=std::min(rows,count-begin);
        std::string sql(prefix);sql+=" VALUES";
        for(size_t i=0;i<n;++i) {if(i)sql+=',';sql+=pattern;}
        sql+=suffix;
        Statement statement(db,sql);
        for(size_t i=0;i<n;++i) bind(statement,begin+i,int(i*columns));
        statement.stepDone();begin+=n;
    }
}
}
