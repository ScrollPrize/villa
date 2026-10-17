#include "morphology.cuh"
#include "label_values.cuh"
namespace vcr::augment::cuda {
template<class T> __global__ void positive_mask(const T* input,std::uint8_t* output,std::size_t n) { EACH_INDEX(n) output[i]=input[i]>0; }
template<class T> __global__ void channel_maximum(const T* input,std::size_t n,double* maximum) {
    double value=double(input[0]); for (std::size_t i=1;i<n;++i) value=fmax(value,double(input[i])); *maximum=value;
}
template<class T> __global__ void restore_maximum(T* output,const std::uint8_t* mask,std::size_t n,const double* maximum) { EACH_INDEX(n) output[i]=T(double(mask[i])*(*maximum)); }
void apply_cuda(const Morphology& t,Sample&,const Parameters& p,Session& session) {
    for (const auto& c:p.channels) {
        auto& a=session.arrays.at(c.key); require_exact_labels(a,session); Layout g{Grid(a.spatial())};
        auto footprint=footprint_offsets(t.repeated_cross?repeated_cross(g.dim,c.values[0]):ball(g.dim,c.values[0]));
        Buffer<std::uint8_t> first(g.size),second(g.size); Buffer<double> maximum(1);
        a.typed([&](auto data) {
            const auto* input=data.ptr+c.channel*g.size;
            channel_maximum<<<1,1,0,vcr::device::stream()>>>(input,g.size,maximum.ptr);
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
