#pragma once
#include "plan.hpp"
namespace vcr::geometry {
using Array=nb::ndarray<nb::c_contig>;
inline void check(Array a,nb::dlpack::dtype dtype,std::vector<std::size_t> shape,int id){
    if(a.dtype()!=dtype||a.ndim()!=shape.size()||a.device_type()!=(id<0?1:2)||(id>=0&&a.device_id()!=id))
        throw nb::value_error("geometry array dtype, rank or device mismatch");
    for(std::size_t i=0;i<shape.size();++i)if(a.shape(i)!=shape[i])throw nb::value_error("geometry array shape mismatch");
}
inline void disjoint(std::initializer_list<Array> inputs,std::initializer_list<Array> outputs){
    auto overlap=[](Array a,Array b){auto p=reinterpret_cast<std::uintptr_t>(a.data()),q=reinterpret_cast<std::uintptr_t>(b.data());return a.nbytes()&&b.nbytes()&&p<q+b.nbytes()&&q<p+a.nbytes();};
    for(auto out:outputs){for(auto in:inputs)if(overlap(out,in))throw nb::value_error("geometry outputs must not overlap inputs");}
    for(auto a=outputs.begin();a!=outputs.end();++a)for(auto b=a+1;b!=outputs.end();++b)if(overlap(*a,*b))throw nb::value_error("geometry outputs must not overlap each other");
}
template<class F> void execute(int id,F f){
    nb::gil_scoped_release release;
    if(id<0){f();return;}
#ifdef VC_READER_CUDA
    device::Guard guard(id);
    try{f();}catch(...){cudaStreamSynchronize(device::stream());throw;}
#else
    throw std::runtime_error("geometry CUDA operations require a CUDA build");
#endif
}
struct Bank {CurvesView view;int device;};
inline Bank bank(nb::handle obj,bool vertices=false){
    if(nb::isinstance<Curves>(obj))return {nb::cast<const Curves&>(obj).view(vertices),-1};
#ifdef VC_READER_CUDA
    auto& c=nb::cast<const CudaCurves&>(obj);return {vertices?c.vertices:c.edges,c.device};
#else
    throw nb::value_error("expected geometry curve storage");
#endif
}
inline VoxelGrid grid(std::array<std::int64_t,3> shape,std::int64_t batches){
    VoxelGrid g{{shape[0],shape[1],shape[2]},batches,1};
    if(batches<1)throw nb::value_error("batches must be positive");
    for(auto n:shape){if(n<1||n>(1<<24)||g.volume>(INT64_MAX-1)/n)throw nb::value_error("invalid voxel shape");g.volume*=n;}
    if(g.volume>(INT64_MAX-1)/batches)throw nb::value_error("voxel keys overflow int64");return g;
}
inline int array_device(Array a){if(a.device_type()==1)return -1;if(a.device_type()==2)return a.device_id();throw nb::value_error("geometry supports CPU and CUDA arrays");}
inline void bind(nb::module_& m){
    nb::class_<Curves>(m,"GeometryCurves")
      .def("__init__",[](Curves* self,Array p,Array offsets,Array s){
        auto n=p.shape(0),k=offsets.shape(0);check(p,nb::dtype<float>(),{n,3},-1);check(offsets,nb::dtype<std::int64_t>(),{k},-1);check(s,nb::dtype<float>(),{s.size()},-1);
        if(s.size()&&s.size()!=n)throw nb::value_error("arclength count mismatch");
        execute(-1,[&]{new(self) Curves(static_cast<const float*>(p.data()),n,static_cast<const std::int64_t*>(offsets.data()),k,s.size()?static_cast<const float*>(s.data()):nullptr);});
      })
      .def_prop_ro("nbytes",&Curves::nbytes);
    nb::class_<Voxels>(m,"GeometryVoxels").def_prop_ro("nbytes",[](const Voxels& v){return v.keys.size()*8;});
#ifdef VC_READER_CUDA
    nb::class_<CudaCurves>(m,"GeometryCudaCurves")
      .def("__init__",[](CudaCurves* self,const Curves& cpu,int id){execute(id,[&]{new(self) CudaCurves(cpu,id);});})
      .def_prop_ro("nbytes",[](const CudaCurves& c){std::size_t n=0;for(auto& s:c.storage)n+=s->bytes;return n;});
    nb::class_<CudaVoxels>(m,"GeometryCudaVoxels").def_prop_ro("nbytes",[](const CudaVoxels& v){return v.keys->bytes;});
    m.def("_geometry_voxels_cuda",[](const Voxels& cpu,int id){
        CudaVoxels out{cpu.grid,{},cpu.keys.size(),id};execute(id,[&]{out.keys=device::allocate(out.count*8,id);
            if(out.count)device::check(cudaMemcpyAsync(out.keys->ptr,cpu.keys.data(),out.count*8,cudaMemcpyHostToDevice,device::stream()));device::complete();});return out;
    });
#endif
    m.def("_geometry_gather",[](nb::object obj,Array ids,bool segments,Array out,Array arc,Array owners,Array valid){
        auto b=bank(obj);auto n=ids.size();std::size_t width=segments?2:1;
        check(ids,nb::dtype<std::int64_t>(),{n},b.device);
        check(out,nb::dtype<float>(),segments?std::vector<std::size_t>{n,2,3}:std::vector<std::size_t>{n,3},b.device);
        check(arc,nb::dtype<float>(),segments?std::vector<std::size_t>{n,2}:std::vector<std::size_t>{n},b.device);
        check(owners,nb::dtype<std::int64_t>(),{n},b.device);check(valid,nb::dtype<bool>(),{n},b.device);disjoint({ids},{out,arc,owners,valid});
        auto ip=static_cast<const std::int64_t*>(ids.data());auto p=static_cast<float*>(out.data());auto a=static_cast<float*>(arc.data());auto own=static_cast<std::int64_t*>(owners.data());auto v=static_cast<bool*>(valid.data());
        execute(b.device,[&]{if(b.device<0){for(std::size_t i=0;i<n;++i)gather(b.view,ip[i],segments,p+3*width*i,a+width*i,own+i,v+i);}else{
#ifdef VC_READER_CUDA
            gather_cuda(b.view,ip,segments,p,a,own,v,n);
#endif
        }});
    });
    m.def("_geometry_interpolate",[](nb::object obj,Array ids,Array s,Array out,Array valid){
        auto b=bank(obj);auto n=ids.size();check(ids,nb::dtype<std::int64_t>(),{n},b.device);check(s,nb::dtype<float>(),{n},b.device);
        check(out,nb::dtype<float>(),{n,3},b.device);check(valid,nb::dtype<bool>(),{n},b.device);disjoint({ids,s},{out,valid});
        auto ip=static_cast<const std::int64_t*>(ids.data());auto sp=static_cast<const float*>(s.data());auto op=static_cast<float*>(out.data());auto vp=static_cast<bool*>(valid.data());
        execute(b.device,[&]{if(b.device<0){for(std::size_t i=0;i<n;++i)interpolate(b.view,ip[i],sp[i],op+3*i,vp+i);}else{
#ifdef VC_READER_CUDA
            interpolate_cuda(b.view,ip,sp,op,vp,n);
#endif
        }});
    });
    m.def("_geometry_intersect",[](nb::object obj,Array ids,Array planes,Array intervals,int direction,Array out,Array arc,Array index,Array valid){
        auto b=bank(obj);auto n=ids.size();check(ids,nb::dtype<std::int64_t>(),{n},b.device);check(planes,nb::dtype<float>(),{n,4},b.device);
        check(intervals,nb::dtype<float>(),{intervals.size()?n:0,2},b.device);if(direction< -1||direction>1)throw nb::value_error("direction must be -1, 0 or 1");
        check(out,nb::dtype<float>(),{n,3},b.device);check(arc,nb::dtype<float>(),{n},b.device);check(index,nb::dtype<std::int64_t>(),{n},b.device);check(valid,nb::dtype<bool>(),{n},b.device);
        disjoint({ids,planes,intervals},{out,arc,index,valid});
        auto ip=static_cast<const std::int64_t*>(ids.data());auto p=static_cast<const float*>(planes.data());auto span=intervals.size()?static_cast<const float*>(intervals.data()):nullptr;
        auto o=static_cast<float*>(out.data());auto a=static_cast<float*>(arc.data());auto ix=static_cast<std::int64_t*>(index.data());auto v=static_cast<bool*>(valid.data());
        execute(b.device,[&]{if(b.device<0){for(std::size_t i=0;i<n;++i)intersect(b.view,ip[i],p+4*i,span?span+2*i:nullptr,direction,o+3*i,a+i,ix+i,v+i);}else{
#ifdef VC_READER_CUDA
            intersect_cuda(b.view,ip,p,span,direction,o,a,ix,v,n);
#endif
        }});
    });
    m.def("_geometry_nearest",[](nb::object obj,Array points,Array ids,bool vertices,float limit,Array out,Array distance,Array arc,Array index,Array owner,Array valid){
        auto b=bank(obj,vertices);auto n=points.shape(0);check(points,nb::dtype<float>(),{n,3},b.device);check(ids,nb::dtype<std::int64_t>(),{ids.size()?n:0},b.device);
        if(!(limit>=0))throw nb::value_error("max_distance must be nonnegative");
        check(out,nb::dtype<float>(),{n,3},b.device);check(distance,nb::dtype<float>(),{n},b.device);check(arc,nb::dtype<float>(),{n},b.device);
        check(index,nb::dtype<std::int64_t>(),{n},b.device);check(owner,nb::dtype<std::int64_t>(),{n},b.device);check(valid,nb::dtype<bool>(),{n},b.device);
        disjoint({points,ids},{out,distance,arc,index,owner,valid});
        auto p=static_cast<const float*>(points.data());auto ip=ids.size()?static_cast<const std::int64_t*>(ids.data()):nullptr;auto o=static_cast<float*>(out.data());auto d=static_cast<float*>(distance.data());
        auto a=static_cast<float*>(arc.data());auto ix=static_cast<std::int64_t*>(index.data());auto own=static_cast<std::int64_t*>(owner.data());auto v=static_cast<bool*>(valid.data());
        execute(b.device,[&]{if(b.device<0){for(std::size_t i=0;i<n;++i)closest(b.view,vector(p+3*i),ip?ip[i]:-1,limit,o+3*i,d+i,a+i,ix+i,own+i,v+i);}else{
#ifdef VC_READER_CUDA
            closest_cuda(b.view,p,ip,limit,o,d,a,ix,own,v,n);
#endif
        }});
    });
    m.def("_geometry_boxes",[](nb::object obj,Array lower,Array upper,Array ids,bool vertices,int capacity,Array out,Array counts,Array valid){
        auto b=bank(obj,vertices);auto n=lower.shape(0);
        if(capacity<1)throw nb::value_error("capacity must be positive");
        check(lower,nb::dtype<float>(),{n,3},b.device);check(upper,nb::dtype<float>(),{n,3},b.device);check(ids,nb::dtype<std::int64_t>(),{ids.size()?n:0},b.device);
        check(out,nb::dtype<std::int64_t>(),{n,std::size_t(capacity)},b.device);check(counts,nb::dtype<std::int64_t>(),{n},b.device);check(valid,nb::dtype<bool>(),{n},b.device);
        disjoint({lower,upper,ids},{out,counts,valid});
        auto lo=static_cast<const float*>(lower.data());auto hi=static_cast<const float*>(upper.data());auto ip=ids.size()?static_cast<const std::int64_t*>(ids.data()):nullptr;
        auto o=static_cast<std::int64_t*>(out.data());auto c=static_cast<std::int64_t*>(counts.data());auto v=static_cast<bool*>(valid.data());
        execute(b.device,[&]{if(b.device<0){for(std::size_t i=0;i<n;++i)query_box(b.view,vector(lo+3*i),vector(hi+3*i),ip?ip[i]:-1,capacity,o+i*capacity,c+i,v+i);}else{
#ifdef VC_READER_CUDA
            boxes_cuda(b.view,lo,hi,ip,capacity,o,c,v,n);
#endif
        }});
    });
    m.def("_geometry_transform",[](Array points,Array matrices,bool vectors,Array out){
        int id=array_device(points);auto n=points.shape(0),count=matrices.shape(0);check(points,nb::dtype<float>(),{n,3},id);check(matrices,nb::dtype<float>(),{count,3,4},id);check(out,nb::dtype<float>(),{n,3},id);
        if(count!=1&&count!=n)throw nb::value_error("provide one transform or one per point");disjoint({points,matrices},{out});
        auto p=static_cast<const float*>(points.data());auto mat=static_cast<const float*>(matrices.data());auto o=static_cast<float*>(out.data());
        execute(id,[&]{if(id<0){for(std::size_t i=0;i<n;++i){auto m=mat+(count==1?0:12*i);auto v=vector(p+3*i);store(o+3*i,vectors?linear(m,v):transform(m,v));}}else{
#ifdef VC_READER_CUDA
            transform_cuda(p,mat,count,vectors,o,n);
#endif
        }});
    });
    m.def("_geometry_voxelize",[](Array points,Array ids,std::array<std::int64_t,3> shape,std::int64_t batches)->nb::object{
        int id=array_device(points);auto n=points.shape(0);check(points,nb::dtype<float>(),{n,3},id);check(ids,nb::dtype<std::int64_t>(),{ids.size()?n:0},id);auto g=grid(shape,batches);
        auto p=static_cast<const float*>(points.data());auto ip=ids.size()?static_cast<const std::int64_t*>(ids.data()):nullptr;
        if(id<0){std::unique_ptr<Voxels> result;execute(id,[&]{result=std::make_unique<Voxels>(p,ip,n,g);});return nb::cast(result.release(),nb::rv_policy::take_ownership);}
#ifdef VC_READER_CUDA
        CudaVoxels result;execute(id,[&]{result=voxelize_cuda(p,ip,n,g,id);});return nb::cast(std::move(result));
#else
        throw nb::value_error("CUDA build required");
#endif
    });
    m.def("_geometry_contains",[](nb::object obj,Array points,Array ids,Array out){
        int id=array_device(points);auto n=points.shape(0);check(points,nb::dtype<float>(),{n,3},id);check(ids,nb::dtype<std::int64_t>(),{ids.size()?n:0},id);check(out,nb::dtype<bool>(),{n},id);disjoint({points,ids},{out});
        auto p=static_cast<const float*>(points.data());auto ip=ids.size()?static_cast<const std::int64_t*>(ids.data()):nullptr;auto o=static_cast<bool*>(out.data());
        if(id<0){auto& v=nb::cast<const Voxels&>(obj);execute(id,[&]{for(std::size_t i=0;i<n;++i)o[i]=contains(v.keys.data(),v.keys.size(),voxel_key(vector(p+3*i),ip?ip[i]:0,v.grid));});}
#ifdef VC_READER_CUDA
        else{auto& v=nb::cast<const CudaVoxels&>(obj);if(v.device!=id)throw nb::value_error("voxel device mismatch");execute(id,[&]{contains_cuda(v,p,ip,o,n);});}
#endif
    });
    m.def("_geometry_dense",[](nb::object obj,Array out){
        int id=array_device(out);VoxelGrid g;
        if(id<0)g=nb::cast<const Voxels&>(obj).grid;
#ifdef VC_READER_CUDA
        else{auto& v=nb::cast<const CudaVoxels&>(obj);if(v.device!=id)throw nb::value_error("voxel device mismatch");g=v.grid;}
#endif
        check(out,nb::dtype<std::uint8_t>(),{std::size_t(g.batches),std::size_t(g.shape[0]),std::size_t(g.shape[1]),std::size_t(g.shape[2])},id);auto o=static_cast<std::uint8_t*>(out.data());
        if(id<0){auto& v=nb::cast<const Voxels&>(obj);execute(id,[&]{std::fill_n(o,g.batches*g.volume,0);for(auto k:v.keys)o[k]=1;});}
#ifdef VC_READER_CUDA
        else{auto& v=nb::cast<const CudaVoxels&>(obj);execute(id,[&]{dense_cuda(v,o);});}
#endif
    });
}
}
