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
__device__ void multiply(KernelIndex vcr_launch, const std::uint8_t* skeleton, const float* values, float* out, std::size_t n) {
    EACH_INDEX(n) out[i] = float(skeleton[i] != 0) * values[i];
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
            skeleton.swap_storage(scratch);
        };
        for (std::size_t c = 0; c < a.shape[0]; ++c) {
            a.typed([&](auto labels) { launch(blocks(grid.size),256,0,vcr::device::stream(), binarize_launch{}, labels.ptr+c*grid.size,skeleton.ptr,values.ptr,grid.size,ignore,ignored); });
            if(BatchQueue::current)BatchQueue::current->flush();
            thinning::cuda::skeletonDevice(skeleton.ptr,skeleton.ptr,h,w,grid.size/(h*w),t.method,t.preserve_border);
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
