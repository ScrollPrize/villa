#include "morphology.cuh"
#include "label_values.cuh"
#include "channel_statistics.cuh"
namespace vcr::augment::cuda {
template<class T> __global__ void positive_mask(const T* input,std::uint8_t* output,std::size_t n) { EACH_INDEX(n) output[i]=input[i]>0; }
template<class T> __global__ void restore_maximum(T* output,const std::uint8_t* mask,std::size_t n,const volume_detail::Stats* maximum) { EACH_INDEX(n) output[i]=T(double(mask[i])*maximum->high); }
void apply_cuda(const Morphology& t,Sample&,const Parameters& p,Session& session) {
    for (const auto& c:p.channels) {
        auto& a=session.arrays.at(c.key); require_exact_labels(a,session); Layout g{Grid(a.spatial())};
        auto footprint=footprint_offsets(t.repeated_cross?repeated_cross(g.dim,c.values[0]):ball(g.dim,c.values[0]));
        Buffer<std::uint8_t> first(g.size),second(g.size); Buffer<volume_detail::Stats> maximum(1);
        a.typed([&](auto data) {
            const auto* input=data.ptr+c.channel*g.size;
            summarize_channel(View<typename decltype(data)::value_type>{data.ptr+c.channel*g.size,g.size},maximum.ptr,false);
            positive_mask<<<blocks(g.size),256,0,vcr::device::stream()>>>(input,first.ptr,g.size);
            const auto stage=[&](bool dilate) {
                morphology_kernel<<<blocks(g.size),256,0,vcr::device::stream()>>>(first.ptr,second.ptr,g,footprint.ptr,footprint.size,dilate,!dilate && t.neutral_border);
                std::swap(first.ptr,second.ptr);
            };
            if (t.operation==Morph::Dilate || t.operation==Morph::Close) stage(true);
            if (t.operation!=Morph::Dilate) stage(false);
            if (t.operation==Morph::Open) stage(true);
            restore_maximum<<<blocks(g.size),256,0,vcr::device::stream()>>>(data.ptr+c.channel*g.size,first.ptr,g.size,maximum.ptr);
        }); check(cudaGetLastError());
    }
}
}
