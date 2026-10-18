#pragma once
#include "channel.cuh"
#include "channel_statistics.cuh"
#include "../voxel_transform.hpp"
namespace vcr::augment {
struct Illumination;struct Vignetting;struct RandomToneCurve;struct SaltAndPepper;struct Equalize;struct CLAHE;struct HistogramMatching;
struct PlasmaBrightnessContrast;struct PixelDistributionAdaptation;
}
namespace vcr::augment::cuda {
template<class T> constexpr bool channel_moments = std::is_same_v<T,PlasmaBrightnessContrast> || std::is_same_v<T,PixelDistributionAdaptation>;
template<class T> constexpr bool channel_bounds = channel_moments<T> || std::is_same_v<T,RandomToneCurve> || std::is_same_v<T,SaltAndPepper> || std::is_same_v<T,Equalize> || std::is_same_v<T,CLAHE> || std::is_same_v<T,HistogramMatching>;
namespace vd = volume_detail;
template<class T> __global__ void check_channel_output(const T* data,std::size_t size,double limit,int* error) {
    EACH_INDEX(size) if(!isfinite(data[i]) || fabs(data[i])>limit)atomicExch(error,1);
}
template<class Kernel,class T,int D> __global__ void voxel_transform_kernel(Kernel kernel,vd::GridND<D> grid,const T* input,T* output,
                                                               const vd::Stats* summary,const double* table,const vd::Stats* reference,int* error,const T* auxiliary) {
    vd::ChannelContextT<T,D> context{grid,*summary,table,reference?*reference:vd::Stats{},auxiliary};
    EACH_INDEX(grid.size){T value=kernel(input,i,context);if(!isfinite(value))atomicExch(error,1);output[i]=value;}
}
template<class Kernel,class T> void launch_voxel_transform(const Kernel& kernel,const Grid& grid,View<T> input,View<T> output,
                                                   const vd::Stats* summary,int* error,const double* table=nullptr,const vd::Stats* reference=nullptr,const T* auxiliary=nullptr) {
    vd::with_grid(grid.shape,[&](const auto& g) {
    voxel_transform_kernel<<<blocks(input.size),256,0,vcr::device::stream()>>>(kernel,g,input.ptr,output.ptr,summary,table,reference,error,auxiliary);
    check(cudaGetLastError());
    });
}
template<class Transform,class Function> void with_voxel_channels(const Transform& t,Sample& s,const Parameters& p,Session& session,Function&& fn) {
    apply_channels(t,s,p,session,[&](const auto&,auto image,const Grid& grid,const ChannelPlan& c){
        auto run=[&](auto values){
            using Value=typename decltype(values)::value_type;
            Buffer<Value> output(values.size);Buffer<vd::Stats> summary(1);
            if constexpr(channel_bounds<Transform>) summarize_channel(values,summary.ptr,channel_moments<Transform>);
            else check(cudaMemsetAsync(summary.ptr,0,sizeof(vd::Stats),vcr::device::stream()));
            fn(grid,c,values,View<Value>{output.ptr,values.size},summary.ptr,session.finite_status());
            check_channel_output<<<blocks(values.size),256,0,vcr::device::stream()>>>(output.ptr,values.size,std::numeric_limits<Value>::max(),session.finite_status());
            check(cudaGetLastError());
            copy_device(values,View<Value>{output.ptr,values.size});
        };
        if constexpr(std::is_same_v<typename decltype(image)::value_type,float>){
            const bool wide=channel_wide_parameters(t,c);
            if(wide){
                Buffer<double> values(image.size);View<double> promoted{values.ptr,values.size};convert(promoted,image);run(promoted);
                check_channel_output<<<blocks(image.size),256,0,vcr::device::stream()>>>(values.ptr,image.size,std::numeric_limits<float>::max(),session.finite_status());
                check(cudaGetLastError());convert(image,promoted);
            }else run(image);
        }else run(image);
    });
}
// Upload reference data in its original dtype, without promoting an entire
// float32 reference volume. Small summaries/histogram CDFs retain their precision.
template<class Function> void with_reference_channel(const vd::ReferenceVolume& reference,std::size_t channel,Function&& fn){
    vd::visit_reference_channel(reference,channel,[&](const auto& values) {
        using T=typename std::decay_t<decltype(values)>::value_type;
        Buffer<T> data(values);fn(View<T>{data.ptr,data.size});
    });
}
template<class Transform> void apply_voxel_channels(const Transform& t,Sample& s,const Parameters& p,Session& session) {
    with_voxel_channels(t,s,p,session,[&](const Grid& g,const ChannelPlan& c,auto input,auto output,const auto* stats,int* error){
        launch_voxel_transform(make_kernel(t,c),g,input,output,stats,error);
    });
}
} // namespace vcr::augment::cuda
