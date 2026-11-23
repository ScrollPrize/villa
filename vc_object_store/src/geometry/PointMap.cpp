#include "vc_object_store/geometry/PointMap.hpp"
#include <charconv>
#include <cmath>
#include <map>
#include <stdexcept>

namespace vc::geometry {
int64_t pointIdFromString(std::string_view key)
{
    int64_t id;
    const auto [end,error]=std::from_chars(key.data(),key.data()+key.size(),id);
    if(error!=std::errc{}||end!=key.data()+key.size()||id<0)
        throw std::invalid_argument("invalid numeric point ID");
    return id;
}

AnnotatedPointMap decodePointMap(const nlohmann::json& entries)
{
    if(!entries.is_object())throw std::invalid_argument("points must be an object");
    std::map<int64_t,const nlohmann::json*> ordered;
    for(const auto& [key,value]:entries.items())
        if(!ordered.emplace(pointIdFromString(key),&value).second)
            throw std::invalid_argument("duplicate numeric point ID");
    AnnotatedPointMap result;
    for(const auto& [id,entry]:ordered){
        const auto& p=entry->at("p");
        if(!p.is_array()||p.size()!=3)throw std::invalid_argument("invalid XYZ point");
        Point xyz;
        for(size_t axis=0;axis<3;++axis){
            if(!p[axis].is_number()||!std::isfinite(xyz[axis]=p[axis].get<double>()))
                throw std::invalid_argument("invalid XYZ point");
        }
        result.points.ids.push_back(id);result.points.points.push_back(xyz);
        auto annotation=*entry;annotation.erase("p");
        result.annotations.push_back(std::move(annotation));
    }
    return result;
}
}
