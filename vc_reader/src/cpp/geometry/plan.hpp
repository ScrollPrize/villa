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

// Batches over a curve bank or voxel set, on the CPU (device < 0) or the CUDA device
// holding the arrays. Outputs may not overlap inputs. Returns once the results are ready.
template<class F> void on_device(int id,F f){
    if(id<0){f();return;}
#ifdef VC_READER_CUDA
    device::Guard guard(id);
    try{f();}catch(...){cudaStreamSynchronize(device::stream());throw;}
#else
    throw std::runtime_error("geometry CUDA operations require a CUDA build");
#endif
}
#ifdef VC_READER_CUDA
#define VCR_GEOMETRY_CUDA(call) call
#else
#define VCR_GEOMETRY_CUDA(call)
#endif
inline void gather_batch(const CurvesView& view,int id,const std::int64_t* ids,bool segments,float* out,float* arcs,std::int64_t* owners,bool* valid,std::size_t n){
    const std::size_t width=segments?2:1;
    on_device(id,[&]{if(id<0){for(std::size_t i=0;i<n;++i)gather(view,ids[i],segments,out+3*width*i,arcs+width*i,owners+i,valid+i);}else{
        VCR_GEOMETRY_CUDA(gather_cuda(view,ids,segments,out,arcs,owners,valid,n);)
    }});
}
inline void interpolate_batch(const CurvesView& view,int id,const std::int64_t* ids,const float* s,float* out,bool* valid,std::size_t n){
    on_device(id,[&]{if(id<0){for(std::size_t i=0;i<n;++i)interpolate(view,ids[i],s[i],out+3*i,valid+i);}else{
        VCR_GEOMETRY_CUDA(interpolate_cuda(view,ids,s,out,valid,n);)
    }});
}
// planes: (n, 4). spans: optional (n, 2) arclength intervals. direction: -1, 0 or 1.
inline void intersect_batch(const CurvesView& view,int id,const std::int64_t* ids,const float* planes,const float* spans,int direction,float* out,float* arcs,std::int64_t* index,bool* valid,std::size_t n){
    if(direction< -1||direction>1)throw std::invalid_argument("direction must be -1, 0 or 1");
    on_device(id,[&]{if(id<0){for(std::size_t i=0;i<n;++i)intersect(view,ids[i],planes+4*i,spans?spans+2*i:nullptr,direction,out+3*i,arcs+i,index+i,valid+i);}else{
        VCR_GEOMETRY_CUDA(intersect_cuda(view,ids,planes,spans,direction,out,arcs,index,valid,n);)
    }});
}
// ids: optional, one curve per point (otherwise any curve).
inline void nearest_batch(const CurvesView& view,int id,const float* points,const std::int64_t* ids,float limit,float* out,float* distance,float* arcs,std::int64_t* index,std::int64_t* owner,bool* valid,std::size_t n){
    if(!(limit>=0))throw std::invalid_argument("max_distance must be nonnegative");
    on_device(id,[&]{if(id<0){for(std::size_t i=0;i<n;++i)closest(view,vector(points+3*i),ids?ids[i]:-1,limit,out+3*i,distance+i,arcs+i,index+i,owner+i,valid+i);}else{
        VCR_GEOMETRY_CUDA(closest_cuda(view,points,ids,limit,out,distance,arcs,index,owner,valid,n);)
    }});
}
// out: (n, capacity) ids of what each box holds.
inline void boxes_batch(const CurvesView& view,int id,const float* lower,const float* upper,const std::int64_t* ids,int capacity,std::int64_t* out,std::int64_t* counts,bool* valid,std::size_t n){
    if(capacity<1)throw std::invalid_argument("capacity must be positive");
    on_device(id,[&]{if(id<0){for(std::size_t i=0;i<n;++i)query_box(view,vector(lower+3*i),vector(upper+3*i),ids?ids[i]:-1,capacity,out+i*capacity,counts+i,valid+i);}else{
        VCR_GEOMETRY_CUDA(boxes_cuda(view,lower,upper,ids,capacity,out,counts,valid,n);)
    }});
}
// matrices: `count` row-major 3x4 transforms, one or one per point. Vectors skip translation.
inline void transform_batch(int id,const float* points,const float* matrices,std::size_t count,bool vectors,float* out,std::size_t n){
    if(count!=1&&count!=n)throw std::invalid_argument("provide one transform or one per point");
    on_device(id,[&]{if(id<0){for(std::size_t i=0;i<n;++i){auto m=matrices+(count==1?0:12*i);auto v=vector(points+3*i);store(out+3*i,vectors?linear(m,v):transform(m,v));}}else{
        VCR_GEOMETRY_CUDA(transform_cuda(points,matrices,count,vectors,out,n);)
    }});
}
inline VoxelGrid voxel_grid(std::array<std::int64_t,3> shape,std::int64_t batches){
    VoxelGrid g{{shape[0],shape[1],shape[2]},batches,1};
    if(batches<1)throw std::invalid_argument("batches must be positive");
    for(auto n:shape){if(n<1||n>(1<<24)||g.volume>(INT64_MAX-1)/n)throw std::invalid_argument("invalid voxel shape");g.volume*=n;}
    if(g.volume>(INT64_MAX-1)/batches)throw std::invalid_argument("voxel keys overflow int64");
    return g;
}
inline void contains_batch(const Voxels& voxels,const float* points,const std::int64_t* ids,bool* out,std::size_t n){
    for(std::size_t i=0;i<n;++i)out[i]=contains(voxels.keys.data(),voxels.keys.size(),voxel_key(vector(points+3*i),ids?ids[i]:0,voxels.grid));
}
// The present keys in order, then missing_key up to the output's capacity.
inline void copy_keys(const Voxels& voxels,std::int64_t* out,std::size_t capacity){
    if(capacity<voxels.keys.size())throw std::invalid_argument("key output capacity is too small");
    std::copy(voxels.keys.begin(),voxels.keys.end(),out);std::fill(out+voxels.keys.size(),out+capacity,missing_key);
}
// out: uint8 (batches, *shape), 1 at present voxels.
inline void fill_dense(const Voxels& voxels,std::uint8_t* out){
    std::fill_n(out,voxels.grid.batches*voxels.grid.volume,0);for(auto k:voxels.keys)out[k]=1;
}
#undef VCR_GEOMETRY_CUDA
}
