#pragma once
#include "geometry/GeometryReader.hpp"
#include "vc_object_store/geometry/ArrayBytes.hpp"
namespace vc::objects::detail {
inline double transformedLength(sqlite3* db,const std::string& revision,const VolumeAffine& matrix,const std::string& kind) {
    const auto key=revision+matrix.json().dump();
    auto pool=activeStatementPools.find(db);
    if(pool!=activeStatementPools.end()) {
        const auto found=pool->second->transformedLengths.find(key);
        if(found!=pool->second->transformedLengths.end())return found->second;
    }
    GeometryReader reader(db,revision);
    const auto& arrays=reader.manifest().at("arrays");
    const auto name=kind=="fiber"?(arrays.at("line_points").at("shape")[0].get<int64_t>()?"line_points":"control_points"):"points";
    const auto count=arrays.at(name).at("shape")[0].get<int64_t>();
    const auto stride=kind=="track"?12:24;
    double length=0;std::optional<std::array<double,3>> previous;
    for(int64_t start=0;start<count;start+=256) {
        const auto n=std::min<int64_t>(256,count-start);
        const auto bytes=reader.read(name,start*stride,n*stride);
        for(int64_t i=0;i<n;++i) {
            std::array<double,3> p;
            if(stride==24)copyScalars<double>(std::span(bytes).subspan(i*stride,stride),p);
            else {std::array<int32_t,3> q;copyScalars<int32_t>(std::span(bytes).subspan(i*stride,stride),q);for(int k=0;k<3;++k)p[k]=q[k];}
            // Transform displacements directly to avoid cancellation from translation.
            if(previous){auto delta=p;for(int k=0;k<3;++k)delta[k]-=(*previous)[k];delta=matrix.apply(delta,true);length+=std::hypot(delta[0],delta[1],delta[2]);}
            previous=p;
        }
    }
    if(pool!=activeStatementPools.end()) {
        auto& cache=pool->second->transformedLengths;
        if(cache.size()>=1024)cache.clear();
        cache.emplace(key,length);
    }
    return length;
}
}
