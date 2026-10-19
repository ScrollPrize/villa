#pragma once
#include "common.hpp"
#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <vector>
#ifdef VC_READER_CUDA
#include "../cuda_buffer.hpp"
#endif
namespace vcr::geometry {
struct Index {
    std::vector<Node> nodes;std::vector<int> order,roots;int root=-1;
};
struct Curves {
    std::vector<float> points,arcs;std::vector<std::int64_t> offsets;std::vector<int> owners;
    Index edges,vertices;
    Curves(const float* p,std::size_t n,const std::int64_t* off,std::size_t m,const float* s){
        if(n>INT32_MAX/2||m<1||m>INT32_MAX/2||off[0]!=0||off[m-1]!=std::int64_t(n))throw std::invalid_argument("invalid packed curve offsets/count");
        offsets.assign(off,off+m);if(n)points.assign(p,p+std::size_t(3)*n);arcs.resize(n);owners.resize(n);
        for(float x:points)if(!std::isfinite(x)||fabsf(x)>1e15f)throw std::invalid_argument("curve coordinates must be finite and within +/-1e15");
        for(std::size_t c=0;c+1<m;++c){
            if(off[c]<0||off[c]>=off[c+1]||off[c+1]>std::int64_t(n))throw std::invalid_argument("each curve must have at least one vertex");
            float arc=0;
            for(int i=int(off[c]);i<off[c+1];++i){
                owners[i]=int(c);
                if(s){arc=s[i];if(!std::isfinite(arc)||(i>off[c]&&(arc<arcs[i-1]||!std::isfinite(arc-arcs[i-1]))))throw std::invalid_argument("arclengths must be finite and nondecreasing with finite segment spans");}
                else if(i>off[c]){V d=vector(p+std::size_t(3)*i)-vector(p+std::size_t(3)*(i-1));arc+=sqrtf(dot(d,d));}
                if(!std::isfinite(arc))throw std::invalid_argument("curve arclength overflow");
                arcs[i]=arc;
            }
        }
        edges=build(false);vertices=build(true);
    }
    Node bounds(int i,bool vertex) const {
        V a=vector(points.data()+std::size_t(3)*i),b=a;
        if(!vertex&&i+1<offsets[owners[i]+1])b=vector(points.data()+std::size_t(3)*(i+1));
        return {{fminf(a.x,b.x),fminf(a.y,b.y),fminf(a.z,b.z)},
                {fmaxf(a.x,b.x),fmaxf(a.y,b.y),fmaxf(a.z,b.z)}};
    }
    static void include(Node& a,const Node& b){
        a.lo={fminf(a.lo.x,b.lo.x),fminf(a.lo.y,b.lo.y),fminf(a.lo.z,b.lo.z)};
        a.hi={fmaxf(a.hi.x,b.hi.x),fmaxf(a.hi.y,b.hi.y),fmaxf(a.hi.z,b.hi.z)};
    }
    template<class Bound> static int tree(Index& idx,std::vector<int>& items,int begin,int end,Bound bound,bool top){
        Node node{{INFINITY,INFINITY,INFINITY},{-INFINITY,-INFINITY,-INFINITY}};
        for(int j=begin;j<end;++j)include(node,bound(items[j]));
        if(top&&end-begin==1)return items[begin];
        int slot=int(idx.nodes.size());idx.nodes.push_back(node);
        if(!top&&end-begin<=8){node.begin=begin;node.end=end;}
        else {
            V extent=node.hi-node.lo;int axis=extent.y>extent.x?1:0;if(extent.z>(axis?extent.y:extent.x))axis=2;
            auto center=[&](int i){Node b=bound(i);return axis==0?b.lo.x+b.hi.x:axis==1?b.lo.y+b.hi.y:b.lo.z+b.hi.z;};
            int mid=begin+(end-begin)/2;
            std::nth_element(items.begin()+begin,items.begin()+mid,items.begin()+end,[&](int a,int b){float x=center(a),y=center(b);return x<y||(x==y&&a<b);});
            node.left=tree(idx,items,begin,mid,bound,top);node.right=tree(idx,items,mid,end,bound,top);
        }
        idx.nodes[slot]=node;return slot;
    }
    Index build(bool vertex)const{
        Index idx;idx.roots.resize(offsets.size()-1);idx.order.reserve(owners.size());
        for(std::size_t c=0;c+1<offsets.size();++c){
            int begin=int(idx.order.size()),end=int(offsets[c+1])-(vertex?0:1);
            if(end==offsets[c])++end;
            for(int i=int(offsets[c]);i<end;++i)idx.order.push_back(i);
            idx.roots[c]=tree(idx,idx.order,begin,int(idx.order.size()),[&](int i){return bounds(i,vertex);},false);
        }
        if(!idx.roots.empty()){auto roots=idx.roots;idx.root=tree(idx,roots,0,int(roots.size()),[&](int i){return idx.nodes[i];},true);}
        return idx;
    }
    CurvesView view(bool vertex=false)const{
        auto& idx=vertex?vertices:edges;
        return {points.data(),arcs.data(),offsets.data(),owners.data(),idx.nodes.data(),idx.order.data(),idx.roots.data(),int(offsets.size()-1),idx.root,vertex};
    }
    std::size_t nbytes()const{
        auto bytes=[](const Index& i){return i.nodes.size()*sizeof(Node)+(i.order.size()+i.roots.size())*sizeof(int);};
        return points.size()*4+arcs.size()*4+offsets.size()*8+owners.size()*4+bytes(edges)+bytes(vertices);
    }
};
struct Voxels {
    VoxelGrid grid;std::vector<std::int64_t> keys;
    Voxels(const float* points,const std::int64_t* batches,std::size_t n,VoxelGrid g):grid(g),keys(n){
        for(std::size_t i=0;i<n;++i)keys[i]=voxel_key(vector(points+std::size_t(3)*i),batches?batches[i]:0,g);
        std::sort(keys.begin(),keys.end());keys.erase(std::unique(keys.begin(),keys.end()),keys.end());
        if(!keys.empty()&&keys.back()==missing_key)keys.pop_back();
    }
};
#ifdef VC_READER_CUDA
struct CudaCurves {
    std::vector<device::Storage> storage;CurvesView edges,vertices;int device;
    explicit CudaCurves(const Curves& cpu,int id):device(id){
        device::Guard guard(id);
        auto upload=[&](const auto& v){using T=typename std::decay_t<decltype(v)>::value_type;auto p=device::allocate(v.size()*sizeof(T),id);storage.push_back(p);
            if(!v.empty())device::check(cudaMemcpyAsync(p->ptr,v.data(),p->bytes,cudaMemcpyHostToDevice,device::stream()));return static_cast<const T*>(p->ptr);};
        try {
            edges.points=upload(cpu.points);edges.arcs=upload(cpu.arcs);edges.offsets=upload(cpu.offsets);edges.owners=upload(cpu.owners);
            edges.curves=int(cpu.offsets.size()-1);vertices=edges;
            edges.nodes=upload(cpu.edges.nodes);edges.order=upload(cpu.edges.order);edges.roots=upload(cpu.edges.roots);edges.root=cpu.edges.root;
            vertices.nodes=upload(cpu.vertices.nodes);vertices.order=upload(cpu.vertices.order);vertices.roots=upload(cpu.vertices.roots);vertices.root=cpu.vertices.root;vertices.vertices=true;
            device::complete();
        }catch(...){cudaStreamSynchronize(device::stream());throw;}
    }
};
struct CudaVoxels { VoxelGrid grid;device::Storage keys;std::size_t count;int device; };
void interpolate_cuda(CurvesView,const std::int64_t*,const float*,float*,bool*,std::size_t);
void gather_cuda(CurvesView,const std::int64_t*,bool,float*,float*,std::int64_t*,bool*,std::size_t);
void intersect_cuda(CurvesView,const std::int64_t*,const float*,const float*,int,float*,float*,std::int64_t*,bool*,std::size_t);
void closest_cuda(CurvesView,const float*,const std::int64_t*,float,float*,float*,float*,std::int64_t*,std::int64_t*,bool*,std::size_t);
void transform_cuda(const float*,const float*,std::size_t,bool,float*,std::size_t);
void boxes_cuda(CurvesView,const float*,const float*,const std::int64_t*,int,std::int64_t*,std::int64_t*,bool*,std::size_t);
CudaVoxels voxelize_cuda(const float*,const std::int64_t*,std::size_t,VoxelGrid,int);
void contains_cuda(const CudaVoxels&,const float*,const std::int64_t*,bool*,std::size_t);
void dense_cuda(const CudaVoxels&,std::uint8_t*);
void keys_cuda(const CudaVoxels&,std::int64_t*,std::size_t);
#endif
}
