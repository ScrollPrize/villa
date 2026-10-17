#pragma once
#include "channel.cuh"
#include "../voxel_transform.hpp"
namespace vcr::augment::cuda {
namespace vd = volume_detail;
__global__ inline void summarize_channel(const double* data,std::size_t size,vd::Stats* result) {
    if(!threadIdx.x && !blockIdx.x)*result=vd::stats(data,size);
}
__global__ inline void check_channel_output(const double* data,std::size_t size,double limit,int* error) {
    EACH_INDEX(size) if(!isfinite(data[i]) || fabs(data[i])>limit)atomicExch(error,1);
}
template<class Kernel> __global__ void voxel_transform_kernel(Kernel kernel,vd::Grid3 grid,const double* input,double* output,
                                                               const vd::Stats* summary,const double* table,const vd::Stats* reference,int* error) {
    vd::ChannelContext context{grid,*summary,table,reference?*reference:vd::Stats{}};
    EACH_INDEX(grid.size){double value=kernel(input,i,context);if(!isfinite(value))atomicExch(error,1);output[i]=value;}
}
template<class Kernel> void launch_voxel_transform(const Kernel& kernel,const Grid& grid,View<double> input,View<double> output,
                                                   const vd::Stats* summary,int* error,const double* table=nullptr,const vd::Stats* reference=nullptr) {
    voxel_transform_kernel<<<blocks(input.size),256,0,vcr::device::stream()>>>(kernel,vd::Grid3(grid.shape),input.ptr,output.ptr,summary,table,reference,error);
    check(cudaGetLastError());
}
template<class Transform,class Function> void with_double_channels(const Transform& t,Sample& s,const Parameters& p,Session& session,Function&& fn) {
    apply_channels(t,s,p,session,[&](const auto&,auto image,const Grid& grid,const ChannelPlan& c){
        Buffer<double> input(image.size),output(image.size);convert(View<double>{input.ptr,image.size},image);Buffer<vd::Stats> summary(1);
        summarize_channel<<<1,1,0,vcr::device::stream()>>>(input.ptr,image.size,summary.ptr);check(cudaGetLastError());
        fn(grid,c,View<double>{input.ptr,image.size},View<double>{output.ptr,image.size},summary.ptr,session.finite_status());
        using Value=typename decltype(image)::value_type;
        check_channel_output<<<blocks(image.size),256,0,vcr::device::stream()>>>(output.ptr,image.size,std::numeric_limits<Value>::max(),session.finite_status());
        check(cudaGetLastError());
        convert(image,View<double>{output.ptr,image.size});
    });
}
template<class Transform> void apply_voxel_channels(const Transform& t,Sample& s,const Parameters& p,Session& session) {
    with_double_channels(t,s,p,session,[&](const Grid& g,const ChannelPlan& c,auto input,auto output,const auto* stats,int* error){
        launch_voxel_transform(make_kernel(t,c),g,input,output,stats,error);
    });
}
} // namespace vcr::augment::cuda
