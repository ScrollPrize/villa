#pragma once
#include "internal/Sqlite.hpp"
#include <stdexcept>

namespace vc::objects::detail {
using Digest = std::array<uint8_t,32>;
inline Digest parseDigest(const std::string& hex)
{
    if(hex.size()!=64)throw std::invalid_argument("SHA-256 digest requires 64 hex digits");
    auto nibble=[](char c)->uint8_t {
        if(c>='0'&&c<='9')return uint8_t(c-'0');
        if(c>='a'&&c<='f')return uint8_t(c-'a'+10);
        if(c>='A'&&c<='F')return uint8_t(c-'A'+10);
        throw std::invalid_argument("invalid SHA-256 hex digit");
    };
    Digest result;for(size_t i=0;i<result.size();++i)result[i]=uint8_t((nibble(hex[2*i])<<4)|nibble(hex[2*i+1]));
    return result;
}
inline std::string formatDigest(std::span<const uint8_t> bytes)
{
    if(bytes.size()!=32)throw std::runtime_error("invalid SHA-256 binary digest length");
    constexpr char hex[]="0123456789abcdef";std::string result(64,'0');
    for(size_t i=0;i<bytes.size();++i){result[2*i]=hex[bytes[i]>>4];result[2*i+1]=hex[bytes[i]&15];}
    return result;
}
inline std::string readDigest(const Statement& row,int column)
{
    if(sqlite3_column_type(row.value,column)!=SQLITE_BLOB||sqlite3_column_bytes(row.value,column)!=32)
        throw std::runtime_error("expected binary SHA-256 digest");
    return formatDigest(std::span(static_cast<const uint8_t*>(sqlite3_column_blob(row.value,column)),size_t(32)));
}
inline void bindDigest(Statement& row,int column,const std::string& hex){row.bind(column,parseDigest(hex));}
}
