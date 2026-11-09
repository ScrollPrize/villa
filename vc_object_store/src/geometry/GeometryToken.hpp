#pragma once
#include "internal/Sqlite.hpp"
#include <algorithm>
#include <charconv>

namespace vc::objects::detail {
struct CurrentGeometryToken {std::string objectUuid;int64_t version;};
inline std::optional<CurrentGeometryToken> parseCurrentGeometryToken(const std::string& token)
{
    constexpr std::string_view prefix="current:";
    if(!token.starts_with(prefix))return {};
    const auto split=token.find(':',prefix.size());
    if(split!=prefix.size()+32)throw std::invalid_argument("invalid current geometry token");
    const auto compact=token.substr(prefix.size(),32);
    const auto uuid=formatUuid(parseUuid(compact.substr(0,8)+"-"+compact.substr(8,4)+"-"+compact.substr(12,4)+"-"+compact.substr(16,4)+"-"+compact.substr(20)));
    int64_t version=0;const auto* first=token.data()+split+1;const auto* last=token.data()+token.size();
    const auto result=std::from_chars(first,last,version);
    if(result.ec!=std::errc{}||result.ptr!=last||version<=0)throw std::invalid_argument("invalid current geometry version");
    return CurrentGeometryToken{uuid,version};
}
inline std::string currentGeometryToken(const std::string& object,int64_t version)
{
    if(version<=0)throw std::invalid_argument("invalid current geometry version");
    auto compact=formatUuid(parseUuid(object));compact.erase(std::remove(compact.begin(),compact.end(),'-'),compact.end());
    return "current:"+compact+":"+std::to_string(version);
}
inline std::string canonicalGeometryToken(const std::string& token)
{
    if(auto current=parseCurrentGeometryToken(token))return currentGeometryToken(current->objectUuid,current->version);
    return formatUuid(parseUuid(token));
}
inline std::string readGeometryTokenColumn(const Statement& row,int column)
{
    if(sqlite3_column_type(row.value,column)==SQLITE_TEXT)return canonicalGeometryToken(row.text(column));
    return formatUuid(row.blob16(column));
}
inline void bindGeometryToken(Statement& row,int column,const std::string& token)
{
    if(auto current=parseCurrentGeometryToken(token))row.bind(column,currentGeometryToken(current->objectUuid,current->version));
    else row.bind(column,parseUuid(token));
}
}
