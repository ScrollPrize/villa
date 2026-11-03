#include "morphology.cuh"
namespace vcr::transforms::cuda {
// Binary input with the ignore value removed; values keep the labels for the product.
template<class T> __device__ void binarize(KernelIndex vcr_launch, const T* labels, std::uint8_t* skeleton, float* values, std::size_t n, bool ignore, double ignored) {
    EACH_INDEX(n) {
        const double v = double(labels[i]);
        const double kept = ignore && v == ignored ? 0. : v;
        values[i] = float(kept); skeleton[i] = kept > 0;
    }
}
struct binarize_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { binarize(index,args...); }
};
// Zhang-Suen thinning, one block per YX slice: each subpass marks every
// deletable pixel of the slice before deleting any, as on the CPU.
__device__ void thin_slices(KernelIndex vcr_launch, std::uint8_t* image, std::uint8_t* marks, long long height, long long width) {
    auto* slice = image + vcr_launch.block*height*width; auto* mark = marks + vcr_launch.block*height*width;
    const auto at = [&](long long y, long long x) { return y >= 0 && x >= 0 && y < height && x < width && slice[y*width+x] != 0; };
    for (bool changed = true; changed;) {
        changed = false;
        for (int phase = 0; phase < 2; ++phase) {
            int any = 0;
            for (long long i = threadIdx.x; i < height*width; i += blockDim.x) {
                const long long y = i/width, x = i%width; bool remove = false;
                if (slice[i]) {
                    const bool n[8] = {at(y-1,x), at(y-1,x+1), at(y,x+1), at(y+1,x+1), at(y+1,x), at(y+1,x-1), at(y,x-1), at(y-1,x-1)};
                    int neighbors = 0, transitions = 0;
                    for (int k = 0; k < 8; ++k) { neighbors += n[k]; transitions += !n[k] && n[(k+1)%8]; }
                    const bool preserve = phase == 0 ? (n[0] && n[2] && n[4]) || (n[2] && n[4] && n[6]) : (n[0] && n[2] && n[6]) || (n[0] && n[4] && n[6]);
                    remove = neighbors >= 2 && neighbors <= 6 && transitions == 1 && !preserve;
                }
                mark[i] = remove; any |= remove;
            }
            any = __syncthreads_or(any);
            for (long long i = threadIdx.x; i < height*width; i += blockDim.x) if (mark[i]) slice[i] = 0;
            __syncthreads();
            changed |= bool(any);
        }
    }
}
struct thin_slices_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { thin_slices(index,args...); }
};
__device__ void multiply(KernelIndex vcr_launch, const std::uint8_t* skeleton, const float* values, float* out, std::size_t n) {
    EACH_INDEX(n) out[i] = float(skeleton[i]) * values[i];
}
struct multiply_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { multiply(index,args...); }
};
void apply_cuda(const MedialSurface& t, Sample& s, const Parameters&, Session& session) {
    std::vector<std::pair<std::string, Field>> outputs;
    for (const auto& [name,f] : s) if (selected_label(name,f,t.on)) {
        const auto& a = session.arrays.at(name); Grid grid(a.spatial()); Layout g(grid);
        const auto d = grid.shape.size(), h = grid.shape[d-2], w = grid.shape[d-1];
        auto footprint = footprint_offsets(ball(d,1));
        DeviceArray out(a.shape,'f',4);
        Buffer<std::uint8_t> skeleton(grid.size), scratch(grid.size); Buffer<float> values(grid.size);
        const bool ignore = t.ignore.count(name); const double ignored = ignore ? t.ignore.at(name) : 0;
        // dilate/erode ping-pong through scratch; erosion uses the neutral border.
        const auto morph = [&](bool dilate) {
            launch(blocks(grid.size),256,0,vcr::device::stream(), morphology_kernel_launch{}, skeleton.ptr,scratch.ptr,g,footprint.ptr,footprint.size,dilate,!dilate);
            std::swap(skeleton.ptr,scratch.ptr);
        };
        for (std::size_t c = 0; c < a.shape[0]; ++c) {
            a.typed([&](auto labels) { launch(blocks(grid.size),256,0,vcr::device::stream(), binarize_launch{}, labels.ptr+c*grid.size,skeleton.ptr,values.ptr,grid.size,ignore,ignored); });
            launch(unsigned(grid.size/(h*w)),1024,0,vcr::device::stream(), thin_slices_launch{}, skeleton.ptr,scratch.ptr,(long long)h,(long long)w);
            if (t.tube) { morph(true); morph(true); }
            if (t.open) { morph(false); morph(true); }
            if (t.close) { morph(true); morph(false); }
            launch(blocks(grid.size),256,0,vcr::device::stream(), multiply_launch{}, skeleton.ptr,values.ptr,out.view<float>().ptr+c*grid.size,grid.size);
            check(cudaGetLastError());
        }
        Array metadata; metadata.shape = a.shape; metadata.kind = 'f'; metadata.bytes = 4;
        outputs.emplace_back(name+t.suffix,Field{std::move(metadata),Target{Role::Scalar},name});
        session.arrays.emplace(name+t.suffix,std::move(out));
    }
    for (auto& [name,f] : outputs) { require(!s.count(name),"derived skeleton field already exists"); s.emplace(name,std::move(f)); }
}
} // namespace vcr::transforms::cuda
