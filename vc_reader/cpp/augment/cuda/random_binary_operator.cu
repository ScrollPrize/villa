#include "../labels.hpp"
#include "resident.cuh"
namespace vcr::augment::cuda {
struct Offset { int v[3]; };
template<class T> __global__ void check_label_ids(View<T> data, int* invalid) {
    EACH_INDEX(data.size) if (data.ptr[i] < -9007199254740991. || data.ptr[i] > 9007199254740991.)
        atomicExch(invalid,1);
}
template<class T, class O> __global__ void morphology_kernel(const T* input, O* output, Layout g, const Offset* footprint, std::size_t count, bool dilate) {
    EACH_INDEX(g.size) {
        bool result = !dilate;
        for (std::size_t j = 0; j < count; ++j) {
            bool valid = true; std::size_t at = 0;
            for (int a = 0; a < g.dim; ++a) {
                auto x = static_cast<long long>(g.coord(i,a)) + (dilate ? -footprint[j].v[a] : footprint[j].v[a]);
                if (x < 0 || x >= static_cast<long long>(g.shape[a])) valid = false; else at += x*g.stride[a];
            }
            bool value = valid && input[at] != 0;
            if (dilate ? value : !value) { result = dilate; break; }
        }
        output[i] = O(result);
    }
}
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
        // Match the CPU value-based label contract without copying the volume.
        if ((image.kind == 'i' || image.kind == 'u') && image.bytes == 8 && !p.channels.empty()) {
            Buffer<int> invalid(1); int flag = 0; invalid.upload(&flag);
            check_label_ids<<<blocks(all.size),256>>>(all,invalid.ptr); check(cudaGetLastError());
            invalid.download(&flag);
            require(!flag,"value-based label transforms require integer IDs within +/- (2**53 - 1)");
        }
        for (const auto& c : p.channels) {
            auto footprint = ball(g.dim,c.values[0]); std::vector<Offset> offsets(footprint.size());
            for (std::size_t i = 0; i < offsets.size(); ++i) std::copy(footprint[i].begin(),footprint[i].end(),offsets[i].v);
            Buffer<Offset> f(offsets); Buffer<std::uint8_t> first(g.size),second(g.size);
            auto op = static_cast<Morph>(int(c.values[1]));
            morphology_kernel<<<blocks(g.size),256>>>(all.ptr+c.channel*g.size,first.ptr,g,f.ptr,offsets.size(),op == Morph::Dilate || op == Morph::Close);
            if (op == Morph::Close || op == Morph::Open) {
                morphology_kernel<<<blocks(g.size),256>>>(first.ptr,second.ptr,g,f.ptr,offsets.size(),op == Morph::Open);
                std::swap(first.ptr,second.ptr);
            }
            assign_mask<<<blocks(g.size),256>>>(all.ptr,first.ptr,g.size,c.channel,selected.ptr,channels.size());
            check(cudaGetLastError());
        }
    });
}
} // namespace vcr::augment::cuda
