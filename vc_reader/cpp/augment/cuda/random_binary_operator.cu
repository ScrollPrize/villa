#include "../labels.hpp"
#include "morphology.cuh"
#include "label_values.cuh"
namespace vcr::augment::cuda {
template<class T> __global__ void assign_mask(T* data, const std::uint8_t* result, std::size_t n, std::size_t channel, const std::size_t* channels, std::size_t count) {
    EACH_INDEX(n) {
        if (result[i] && !data[channel*n+i])
            for (std::size_t j = 0; j < count; ++j) if (channels[j] != channel) data[channels[j]*n+i] = 0;
        data[channel*n+i] = T(result[i]);
    }
}
void apply_cuda(const RandomBinaryOperator& t, Sample& s, const Parameters& p, Session& session) {
    auto& image = session.arrays.at(t.on); Grid grid(image.spatial()); Layout g(grid);
    auto channels = selected_channels(t.channels,image.shape[0]); Buffer<std::size_t> selected(channels);
    image.typed([&](auto all) {
        if (!p.channels.empty()) require_exact_labels(image,session);
        for (const auto& c : p.channels) {
            auto f = footprint_offsets(ball(g.dim,c.values[0])); const auto count = f.size;
            Buffer<std::uint8_t> first(g.size),second(g.size);
            auto op = static_cast<Morph>(int(c.values[1]));
            morphology_kernel<<<blocks(g.size),256,0,vcr::device::stream()>>>(all.ptr+c.channel*g.size,first.ptr,g,f.ptr,count,op == Morph::Dilate || op == Morph::Close);
            if (op == Morph::Close || op == Morph::Open) {
                morphology_kernel<<<blocks(g.size),256,0,vcr::device::stream()>>>(first.ptr,second.ptr,g,f.ptr,count,op == Morph::Open);
                std::swap(first.ptr,second.ptr);
            }
            assign_mask<<<blocks(g.size),256,0,vcr::device::stream()>>>(all.ptr,first.ptr,g.size,c.channel,selected.ptr,channels.size());
            check(cudaGetLastError());
        }
    });
}
} // namespace vcr::augment::cuda
