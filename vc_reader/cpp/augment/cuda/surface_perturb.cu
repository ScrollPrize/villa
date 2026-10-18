#include "../tracing.hpp"
#include "label_values.cuh"
#include "sort_keys.cuh"
#include "select_indices.cuh"
namespace vcr::augment::cuda {
template<class T> __global__ void surface_normals(const T* input,double* normals,std::uint8_t* valid,std::size_t h,std::size_t w) {
    EACH_INDEX(h*w) valid[i]=surface_detail::normal([&](std::size_t j){return double(input[j]);},h,w,i,normals+3*i);
}
__global__ void choose_surface_center(const std::size_t* valid,std::size_t index,std::size_t* center) {*center=valid[index];}
__global__ void surface_distances(double* distance,std::size_t n,std::size_t w,const std::size_t* center) {EACH_INDEX(n)distance[i]=surface_detail::distance(i,*center,w);}
__global__ void add_surface_blob(double* displacement,std::size_t n,std::size_t w,const std::size_t* center,const double* sorted,
        std::size_t points,double radius_sigma,double sigma,double amplitude) {
    const double radius2=points?sorted[(points<n?points:n)-1]:(radius_sigma*sigma)*(radius_sigma*sigma);
    if(points)sigma=fmax(.3,sqrt(fmax(radius2,1e-6))/radius_sigma);
    EACH_INDEX(n) displacement[i]+=surface_detail::contribution(surface_detail::distance(i,*center,w),radius2,sigma,amplitude);
}
__global__ void surface_field(const double* normals,const float* displacement,float* field,std::size_t n,double limit) {
    EACH_INDEX(n)for(std::size_t c=0;c<3;++c)field[c*n+i]=normals[3*i+c]*fmin(limit,fmax(-limit,displacement[i]));
}
template<class T> __global__ void move_surface(T* input,const float* field,std::size_t n,int* status) {EACH_INDEX(n)input[i]=checked_label_cast<T>(double(input[i])+field[i],status);}
Parameters sample_surface(const SurfacePerturb& t,const Sample&,Random& r,bool record,Session& session) {
    require(!session.async,"SurfacePerturb requires sync=True for valid-normal selection");const auto& a=session.arrays.at(t.on);
    require(a.shape.size()==3 && a.shape[0]==3,"SurfacePerturb requires a (3, H, W) coordinate field");
    require(t.radius_sigma>=.5 && std::isfinite(t.radius_sigma) && t.max_displacement>=0 && std::isfinite(t.max_displacement),"invalid surface perturbation limits");
    const auto h=a.shape[1],w=a.shape[2],n=h*w;Parameters p;ChannelPlan plan;plan.key=t.on;auto field=std::make_shared<Buffer<float>>(3*n);
    check(cudaMemsetAsync(field->ptr,0,field->size*sizeof(float),vcr::device::stream()));
    if(h>=2 && w>=2 && t.max_displacement>0) {
        Buffer<double> normals(3*n),displacement(n),distance(n);Buffer<std::uint8_t> valid(n);Buffer<std::size_t> indices(n),count(1),center(1);
        a.typed([&](auto v){surface_normals<<<blocks(n),256,0,vcr::device::stream()>>>(v.ptr,normals.ptr,valid.ptr,h,w);});
        select_flagged_indices(valid,indices,count);std::size_t total;count.download(&total);
        check(cudaMemsetAsync(displacement.ptr,0,n*sizeof(double),vcr::device::stream()));
        if(total)for(int blob=draw_integer(t.blobs,r);blob>0;--blob) {
            choose_surface_center<<<1,1,0,vcr::device::stream()>>>(indices.ptr,r.index(total),center.ptr);
            double sigma=0;
            if(t.points_affected){surface_distances<<<blocks(n),256,0,vcr::device::stream()>>>(distance.ptr,n,w,center.ptr);sort_keys(distance,n);}
            else sigma=std::max(.3,draw(t.sigma_fraction,r)*std::min(h,w));
            const double amplitude=draw(t.amplitude,r)*(r.chance(.5)?1:-1);
            add_surface_blob<<<blocks(n),256,0,vcr::device::stream()>>>(displacement.ptr,n,w,center.ptr,distance.ptr,t.points_affected,t.radius_sigma,sigma,amplitude);
        }
        surface_field<<<blocks(n),256,0,vcr::device::stream()>>>(normals.ptr,displacement.ptr,field->ptr,n,t.max_displacement);check(cudaGetLastError());
    }
    plan.device_field=field;if(record){plan.field.resize(field->size);field->download(plan.field.data());}p.channels.push_back(std::move(plan));return p;
}
void apply_cuda(const SurfacePerturb&,Sample&,const Parameters& p,Session& session) {
    for(const auto& c:p.channels){auto& a=session.arrays.at(c.key);auto field=field_buffer(c.field,c.device_field);require(field->size==a.size(),"surface replay shape mismatch");
        a.typed([&](auto v){move_surface<<<blocks(v.size),256,0,vcr::device::stream()>>>(v.ptr,field->ptr,v.size,session.check_flags());});check(cudaGetLastError());}
}
}
