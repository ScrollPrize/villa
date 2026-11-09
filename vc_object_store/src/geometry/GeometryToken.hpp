#pragma once
#include "internal/Sqlite.hpp"

namespace vc::objects::detail {
inline std::string canonicalGeometryToken(const std::string& token)
{
    return formatUuid(parseUuid(token));
}
inline std::string readGeometryTokenColumn(const Statement& row,int column)
{
    if(sqlite3_column_type(row.value,column)==SQLITE_TEXT)return canonicalGeometryToken(row.text(column));
    return formatUuid(row.blob16(column));
}
inline void bindGeometryToken(Statement& row,int column,const std::string& token)
{
    row.bind(column,parseUuid(token));
}
}
