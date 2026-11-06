#pragma once
#include "geometry/Dependencies.hpp"

namespace vc::objects::detail {

// Build-private endpoint validation state. Input must have passed the native
// geometry importer. No XYZ payload or serialized endpoint metadata is retained.
// Stable point identities and surface validity are the only per-point state.
struct PreparedDependencies {
    std::string kind;
    std::map<std::string,GeometryArrayInfo> arrays;
    std::map<std::string,std::vector<std::pair<int64_t,int64_t>>> identities;
    std::vector<uint64_t> invalidSurfacePoints;

    bool hasPoint(const PointReference& point) const
    {
        const auto coordinate=dependencyCoordinateArray(kind,point.array);
        if(!coordinate)return false;
        auto index=point.index;
        if(point.array=="point_ids"||point.array=="control_point_ids"){
            if(index.size()!=1||index[0]<0)return false;
            const auto found=identities.find(point.array);
            if(found==identities.end())return false;
            const auto row=dependencyIdentityOffset(found->second,index[0]);
            if(!row)return false;
            index={*row};
        }
        const auto found=arrays.find(*coordinate);
        if(found==arrays.end())return false;
        const auto offset=dependencyPointOffset(found->second,index);
        if(!offset)return false;
        return kind!="surface"||invalidSurfacePoints.empty()||
            !(invalidSurfacePoints.at(size_t(*offset)/64)&(uint64_t(1)<<(size_t(*offset)%64)));
    }
};

inline PreparedDependencies prepareDependencies(const GeometryData& data)
{
    PreparedDependencies result;
    result.kind=data.kind;
    for(const auto& array:data.arrays){
        result.arrays.emplace(array.name,GeometryArrayInfo{array.name,array.dtype,array.shape,int64_t(array.bytes.size())});
        if(array.name=="point_ids"||array.name=="control_point_ids"){
            const auto ids=scalarsFromBytes<int64_t>(array.bytes);
            auto& rows=result.identities[array.name];rows.reserve(ids.size());
            for(size_t i=0;i<ids.size();++i)rows.emplace_back(ids[i],int64_t(i));
            std::sort(rows.begin(),rows.end());
        }
        if(data.kind=="surface"&&array.name=="points"){
            const auto count=array.bytes.size()/(3*sizeof(float));
            if(array.dtype!="<f4"||array.bytes.size()%(3*sizeof(float)))
                throw std::invalid_argument("prepared dependencies require validated surface coordinates");
            for(size_t i=0;i<count;++i){
                std::array<float,3> xyz;
                copyScalars<float>(std::span(array.bytes).subspan(i*3*sizeof(float),3*sizeof(float)),xyz);
                if(!vc::geometry::validSurfacePoint(xyz)){
                    if(result.invalidSurfacePoints.empty())result.invalidSurfacePoints.resize((count+63)/64);
                    result.invalidSurfacePoints[i/64]|=uint64_t(1)<<(i%64);
                }
            }
        }
    }
    return result;
}

inline void validateDependencies(const Anchor& anchor,const PreparedDependencies& geometry)
{
    validateDependencyReferences(anchor,[&](const PointReference& point){return geometry.hasPoint(point);});
}
}
