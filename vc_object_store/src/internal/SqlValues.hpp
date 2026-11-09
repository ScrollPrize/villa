#pragma once
#include "internal/Sqlite.hpp"
#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace vc::objects::detail {
// Fixed bindings occupy the first parameters; every row only supplies its
// varying values. Used when a whole batch shares provenance/generation.
template<class Pattern,class Fixed,class Bind>
void writeSqlValuesShared(sqlite3* db,std::string_view prefix,size_t shared,size_t columns,
                          size_t count,size_t maxRows,Pattern&& pattern,Fixed&& fixed,Bind&& bind)
{
    if(!maxRows||!columns)throw std::invalid_argument("SQL VALUES batch dimensions must be positive");
    const auto variables=size_t(sqlite3_limit(db,SQLITE_LIMIT_VARIABLE_NUMBER,-1));
    const auto capacity=variables>shared?std::min(maxRows,(variables-shared)/columns):0;
    if(!capacity)throw std::runtime_error("SQLite variable limit cannot hold one record");
    std::unique_ptr<Statement> statement;size_t previous=0;
    for(size_t begin=0;begin<count;){
        const auto n=std::min(capacity,count-begin);
        if(n!=previous){
            std::string sql(prefix);sql+=" VALUES";
            for(size_t i=0;i<n;++i){if(i)sql+=',';sql+=pattern(shared+i*columns);}
            statement=std::make_unique<Statement>(db,sql);previous=n;
            fixed(*statement);
        }else statement->resetKeepingBindings();
        for(size_t i=0;i<n;++i)bind(*statement,begin+i,int(shared+i*columns));
        statement->stepDone();begin+=n;
    }
}
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
    std::unique_ptr<Statement> statement;size_t previousRows=0;
    for(size_t begin=0;begin<count;) {
        const auto n=std::min(rows,count-begin);
        if(n!=previousRows){
            std::string sql(prefix);sql+=" VALUES";
            for(size_t i=0;i<n;++i) {if(i)sql+=',';sql+=pattern;}
            sql+=suffix;statement=std::make_unique<Statement>(db,sql);previousRows=n;
        }else statement->reset();
        for(size_t i=0;i<n;++i) bind(*statement,begin+i,int(i*columns));
        statement->stepDone();begin+=n;
    }
}
}
