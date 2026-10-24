#include "morphology.cuh"
#include "label_values.cuh"
#include "channel_statistics.cuh"
namespace vcr::transforms::cuda {
template<class T> __device__ void positive_mask(KernelIndex vcr_launch, const T* input,std::uint8_t* output,std::size_t n) { EACH_INDEX(n) output[i]=input[i]>0; }
struct positive_mask_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { positive_mask(index,args...); }
};
template<class T> __device__ void restore_maximum(KernelIndex vcr_launch, T* output,const std::uint8_t* mask,std::size_t n,const volume_detail::Stats* maximum) { EACH_INDEX(n) output[i]=T(double(mask[i])*maximum->high); }
struct restore_maximum_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { restore_maximum(index,args...); }
};
void apply_cuda(const Morphology& t,Sample&,const Parameters& p,Session& session) {
    for (const auto& c:p.channels) {
        auto& a=session.arrays.at(c.key); require_exact_labels(a,session); Layout g{Grid(a.spatial())};
        auto footprint=footprint_offsets(t.repeated_cross?repeated_cross(g.dim,c.values[0]):ball(g.dim,c.values[0]));
        Buffer<std::uint8_t> first(g.size),second(g.size); Buffer<volume_detail::Stats> maximum(1);
        a.typed([&](auto data) {
            const auto* input=data.ptr+c.channel*g.size;
            summarize_channel(View<typename decltype(data)::value_type>{data.ptr+c.channel*g.size,g.size},maximum.ptr,false);
            launch(blocks(g.size),256,0,vcr::device::stream(), positive_mask_launch{}, input,first.ptr,g.size);
            const auto stage=[&](bool dilate) {
                launch(blocks(g.size),256,0,vcr::device::stream(), morphology_kernel_launch{}, first.ptr,second.ptr,g,footprint.ptr,footprint.size,dilate,!dilate && t.neutral_border);
                std::swap(first.ptr,second.ptr);
            };
            if (t.operation==Morph::Dilate || t.operation==Morph::Close) stage(true);
            if (t.operation!=Morph::Dilate) stage(false);
            if (t.operation==Morph::Open) stage(true);
            launch(blocks(g.size),256,0,vcr::device::stream(), restore_maximum_launch{}, data.ptr+c.channel*g.size,first.ptr,g.size,maximum.ptr);
        }); check(cudaGetLastError());
    }
}
}
