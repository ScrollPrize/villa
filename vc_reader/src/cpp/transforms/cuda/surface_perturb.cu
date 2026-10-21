#include "../cpu/tracing.hpp"
#include "label_values.cuh"
#include "sort_keys.cuh"
#include "select_indices.cuh"
#include "random.cuh"
namespace vcr::transforms::cuda {
template<class T> __device__ void surface_normals(KernelIndex vcr_launch, const T* input,double* normals,std::uint8_t* valid,std::size_t h,std::size_t w) {
    EACH_INDEX(h*w) valid[i]=surface_detail::normal([&](std::size_t j){return double(input[j]);},h,w,i,normals+3*i);
}
struct surface_normals_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { surface_normals(index,args...); }
};
__device__ void choose_surface_center(KernelIndex vcr_launch, const std::size_t* valid,const std::size_t* count,std::uint64_t seed,std::size_t n,std::size_t* center) {Random random{seed};*center=*count?valid[random_index(random,*count)]:n;}
struct choose_surface_center_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { choose_surface_center(index,args...); }
};
__device__ void surface_distances(KernelIndex vcr_launch, double* distance,std::size_t n,std::size_t w,const std::size_t* center) {EACH_INDEX(n)distance[i]=surface_detail::distance(i,*center,w);}
struct surface_distances_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { surface_distances(index,args...); }
};
__device__ void add_surface_blob(KernelIndex vcr_launch, double* displacement,std::size_t n,std::size_t w,const std::size_t* center,const double* sorted,
        std::size_t points,double radius_sigma,double sigma,double amplitude) {
    if (*center>=n) return;
    const double radius2=points?sorted[(points<n?points:n)-1]:(radius_sigma*sigma)*(radius_sigma*sigma);
    if(points)sigma=fmax(.3,sqrt(fmax(radius2,1e-6))/radius_sigma);
    EACH_INDEX(n) displacement[i]+=surface_detail::contribution(surface_detail::distance(i,*center,w),radius2,sigma,amplitude);
}
struct add_surface_blob_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { add_surface_blob(index,args...); }
};
__device__ void surface_field(KernelIndex vcr_launch, const double* normals,const double* displacement,float* field,std::size_t n,double limit) {
    EACH_INDEX(n)for(std::size_t c=0;c<3;++c)field[c*n+i]=normals[3*i+c]*fmin(limit,fmax(-limit,displacement[i]));
}
struct surface_field_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { surface_field(index,args...); }
};
template<class T> __device__ void move_surface(KernelIndex vcr_launch, T* input,const float* field,std::size_t n,int* status) {EACH_INDEX(n)input[i]=checked_label_cast<T>(double(input[i])+field[i],status);}
struct move_surface_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { move_surface(index,args...); }
};
Parameters sample_surface(const SurfacePerturb& t,const Sample&,Random& r,bool record,Session& session) {
    const auto& a=session.arrays.at(t.on);
    require(a.shape.size()==3 && a.shape[0]==3,"SurfacePerturb requires a (3, H, W) coordinate field");
    require(t.radius_sigma>=.5 && std::isfinite(t.radius_sigma) && t.max_displacement>=0 && std::isfinite(t.max_displacement),"invalid surface perturbation limits");
    const auto h=a.shape[1],w=a.shape[2],n=h*w;Parameters p;ChannelPlan plan;plan.key=t.on;auto field=std::make_shared<Buffer<float>>(3*n);
    check(memset_async(field->ptr,0,field->size*sizeof(float),vcr::device::stream()));
    if(h>=2 && w>=2 && t.max_displacement>0) {
        Buffer<double> normals(3*n),displacement(n),distance(n);Buffer<std::uint8_t> valid(n);Buffer<std::size_t> indices(n),count(1),center(1);
        a.typed([&](auto v){launch(blocks(n),256,0,vcr::device::stream(), surface_normals_launch{}, v.ptr,normals.ptr,valid.ptr,h,w);});
        select_flagged_indices(valid,indices,count);
        check(memset_async(displacement.ptr,0,n*sizeof(double),vcr::device::stream()));
        for(int blob=draw_integer(t.blobs,r);blob>0;--blob) {
            launch(1,1,0,vcr::device::stream(), choose_surface_center_launch{}, indices.ptr,count.ptr,r.state,n,center.ptr);
            r.next();
            double sigma=0;
            if(t.points_affected){launch(blocks(n),256,0,vcr::device::stream(), surface_distances_launch{}, distance.ptr,n,w,center.ptr);sort_keys(distance,n);}
            else sigma=std::max(.3,draw(t.sigma_fraction,r)*std::min(h,w));
            const double amplitude=draw(t.amplitude,r)*(r.chance(.5)?1:-1);
            launch(blocks(n),256,0,vcr::device::stream(), add_surface_blob_launch{}, displacement.ptr,n,w,center.ptr,distance.ptr,t.points_affected,t.radius_sigma,sigma,amplitude);
        }
        launch(blocks(n),256,0,vcr::device::stream(), surface_field_launch{}, normals.ptr,displacement.ptr,field->ptr,n,t.max_displacement);check(cudaGetLastError());
    }
    plan.device_field=field;if(record){plan.field.resize(field->size);field->download(plan.field.data());}p.channels.push_back(std::move(plan));return p;
}
void apply_cuda(const SurfacePerturb&,Sample&,const Parameters& p,Session& session) {
    for(const auto& c:p.channels){auto& a=session.arrays.at(c.key);auto field=field_buffer(c.field,c.device_field);require(field->size==a.size(),"surface replay shape mismatch");
        a.typed([&](auto v){launch(blocks(v.size),256,0,vcr::device::stream(), move_surface_launch{}, v.ptr,field->ptr,v.size,session.check_flags());});check(cudaGetLastError());}
}
}
