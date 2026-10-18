#pragma once
#include "common.hpp"
#include "../element_type.hpp"
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
namespace vcr::aligned {
using Triple=std::array<std::int64_t,3>;
struct Plan {
    Grid grid;
    std::vector<std::int64_t> keys;
    Plan(Triple source,Triple shape,Triple output,Triple offset,Triple base,
         std::array<float,3> origin,std::array<float,9> axes,Triple tile,int channels,int order,bool plane,bool plan_tiles) {
        if(channels<=0 || (order!=0 && order!=1)) throw std::invalid_argument("invalid channels/order");
        grid.channels=channels; grid.order=order; grid.plane=plane;
        grid.count=grid.output_count=grid.tile_count=1;
        std::uint64_t tile_keys=1;
        for(int a=0;a<3;++a) {
            if(source[a]<0 || source[a]>(1LL<<40) || shape[a]<0 || shape[a]>(1<<24) || output[a]<0 ||
               offset[a]<0 || offset[a]>output[a] || shape[a]>output[a]-offset[a] || tile[a]<1 || tile[a]>256 ||
               base[a]<-(1LL<<60) || base[a]>(1LL<<60) || !std::isfinite(origin[a]))
                throw std::invalid_argument("invalid aligned grid extent/origin");
            grid.source[a]=source[a]; grid.shape[a]=shape[a]; grid.output[a]=output[a]; grid.offset[a]=offset[a];
            grid.base[a]=base[a]; grid.origin[a]=origin[a]; grid.tile[a]=tile[a];
            grid.tiles[a]=(source[a]+tile[a]-1)/tile[a];
            if(grid.tiles[a] && tile_keys>INT64_MAX/std::uint64_t(grid.tiles[a])) throw std::invalid_argument("tile grid overflow");
            tile_keys*=grid.tiles[a];
            for(auto pair:{std::pair{&grid.count,shape[a]},std::pair{&grid.output_count,output[a]},std::pair{&grid.tile_count,tile[a]}}) {
                if(pair.second && *pair.first>SIZE_MAX/std::size_t(pair.second)) throw std::invalid_argument("shape overflow");
                *pair.first*=pair.second;
            }
        }
        if(grid.count>32768) throw std::invalid_argument("aligned blocks are limited to 32768 voxels");
        if(grid.output_count>SIZE_MAX/std::size_t(channels)/8) throw std::invalid_argument("output overflow");
        for(int a=0;a<9;++a) { if(!std::isfinite(axes[a])) throw std::invalid_argument("nonfinite affine"); grid.axes[a]=axes[a]; }
        if(plane && (base[0]!=0 || source[0]!=1 || tile[0]!=1)) throw std::invalid_argument("invalid plane grid");
        if(plan_tiles) {
            std::unordered_set<std::int64_t> seen;
            for(std::size_t i=0;i<grid.count;++i) {
                const auto c=cell(grid,i);
                for(int k=0;k<c.count;++k) if(c.support[k]) seen.insert(key(grid,c.at[k]));
            }
            keys.assign(seen.begin(),seen.end()); std::sort(keys.begin(),keys.end());
        }
    }
    std::vector<Triple> starts() const {
        std::vector<Triple> result;
        for(auto k:keys) {
            Triple p{}; for(int a=2;a>=0;--a) { p[a]=(k%grid.tiles[a])*grid.tile[a]; k/=grid.tiles[a]; }
            result.push_back(p);
        }
        return result;
    }
    void cpu(const std::vector<const void*>& tiles,const void* fill,ElementType type,void* output,bool* valid,Field f) const {
        auto g=grid; g.field=f;
        type.visit_load([&](auto load) {
            using T=decltype(load(static_cast<const std::uint8_t*>(fill)));
            // Host tile buffers are native-endian and aligned by NumPy.
            std::vector<const T*> ptrs; for(auto p:tiles) ptrs.push_back(static_cast<const T*>(p));
            const T value=load(static_cast<const std::uint8_t*>(fill));
            for(std::size_t i=0;i<grid.count;++i) sample(g,keys.data(),ptrs.data(),keys.size(),value,output,valid,i);
        });
    }
};
#ifdef VC_READER_CUDA
void launch(const Grid&,const std::vector<std::int64_t>&,const std::vector<const void*>&,const void*,ElementType,void*,bool*,int);
void launch_many(const std::vector<Grid>&,const std::vector<std::vector<std::int64_t>>&,
                 const std::vector<std::vector<const void*>>&,const void*,ElementType,void*,bool*,int);
void launch_valid(const Grid&,bool*);
void launch_box(Triple,Triple,Triple,bool*);
void launch_field(float*,bool*,std::size_t,Field);
#endif
}
