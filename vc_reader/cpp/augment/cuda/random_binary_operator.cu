#include "../labels.hpp"
#include "common.cuh"
namespace vcr::augment::cuda {
struct Offset { int v[3]; };
__global__ void morphology_kernel(const double* input, double* output, Layout g, const Offset* footprint, std::size_t count, bool dilate) {
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
        output[i] = double(result);
    }
}
std::vector<double> morphology_cuda(const std::vector<double>& original, const Grid& grid, const Footprint& footprint, Morph op) {
    std::vector<Offset> offsets(footprint.size());
    for (std::size_t i = 0; i < offsets.size(); ++i) std::copy(footprint[i].begin(),footprint[i].end(),offsets[i].v);
    Buffer<Offset> f(offsets); Buffer<double> input(original), output(original.size()); Layout g(grid);
    auto pass = [&](bool dilate) {
        morphology_kernel<<<blocks(g.size),256>>>(input.ptr,output.ptr,g,f.ptr,offsets.size(),dilate);
        check(cudaGetLastError()); std::swap(input.ptr,output.ptr);
    };
    if (op == Morph::Dilate || op == Morph::Close) pass(true);
    if (op != Morph::Dilate) pass(false);
    if (op == Morph::Open) pass(true);
    auto result = original; input.download(result.data()); return result;
}

void apply_cuda(const RandomBinaryOperator& t, Sample& s, const Parameters& p) {
    apply_binary(t,s,p,morphology_cuda);
}
} // namespace vcr::augment::cuda
