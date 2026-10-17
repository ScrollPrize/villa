#pragma once
#include "core.hpp"
namespace vcr::augment::component_detail {
struct GridView {
    std::size_t shape[3]{1,1,1},stride[3]{1,1,1},size; int dim;
    explicit GridView(const Grid& g):size(g.size),dim(int(g.shape.size())) {for(int a=0;a<dim;++a){shape[a]=g.shape[a];stride[a]=g.stride[a];}}
};
// Scan-order components and lexicographic BFS neighbors match the CPU replay
// contract. Scratch arrays live in host or device memory, depending on caller.
template<class T> VCR_HD std::size_t components(const T* input,GridView g,bool full,std::uint8_t* visited,std::size_t* queue,std::size_t* offsets) {
    for(std::size_t i=0;i<g.size;++i)visited[i]=0;
    std::size_t used=0,count=0; offsets[0]=0;
    for(std::size_t i=0;i<g.size;++i) {
        if(visited[i] || input[i]==0)continue;
        std::size_t head=used;queue[used++]=i;visited[i]=1;
        while(head<used) {
            const auto at=queue[head++]; long long point[3]{};
            for(int a=0;a<g.dim;++a)point[a]=at/g.stride[a]%g.shape[a];
            const int neighbors=g.dim==3?27:9;
            for(int j=0;j<neighbors;++j) {
                int code=j,off[3]{},distance=0;
                for(int a=g.dim-1;a>=0;--a){off[a]=code%3-1;code/=3;distance+=off[a]<0?-off[a]:off[a];}
                if(!distance || (!full && distance!=1))continue;
                bool valid=true;std::size_t next=0;
                for(int a=0;a<g.dim;++a){const auto x=point[a]+off[a];if(x<0 || x>=static_cast<long long>(g.shape[a])){valid=false;break;}next+=std::size_t(x)*g.stride[a];}
                if(valid && input[next]!=0 && !visited[next]){visited[next]=1;queue[used++]=next;}
            }
        }
        offsets[++count]=used;
    }
    return count;
}
}
