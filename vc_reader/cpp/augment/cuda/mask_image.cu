#include "label_values.cuh"
namespace vcr::augment::cuda {
template<class M> __device__ void negative_mask(KernelIndex vcr_launch, const M* mask, std::uint8_t* negative, std::size_t n) { EACH_INDEX(n) { if constexpr (std::is_unsigned_v<M>) negative[i]=0; else negative[i]=mask[i]<0; } }
struct negative_mask_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { negative_mask(index,args...); }
};
template<class T> __device__ void mask_values(KernelIndex vcr_launch, T* image, const std::uint8_t* negative, std::size_t n, double fill, int* status) {
    EACH_INDEX(n) if (negative[i]) image[i] = checked_label_cast<T>(fill,status);
}
struct mask_values_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { mask_values(index,args...); }
};
void apply_cuda(const MaskImage& t, Sample&, const Parameters&, Session& session) {
    auto& a=session.arrays.at(t.on); const auto& mask=session.arrays.at(t.mask); const auto n=product(a.spatial());
    require(a.spatial()==mask.spatial() && t.mask_channel<mask.shape[0],"invalid image mask");
    Buffer<std::uint8_t> negative(n);
    mask.typed([&](auto v) { launch(blocks(n),256,0,vcr::device::stream(), negative_mask_launch{}, v.ptr+t.mask_channel*n,negative.ptr,n); });
    const auto channels=selected_channels(t.channels,a.shape[0]);
    a.typed([&](auto v) { for (auto c:channels) launch(blocks(n),256,0,vcr::device::stream(), mask_values_launch{}, v.ptr+c*n,negative.ptr,n,t.fill,session.check_flags()); });
    check(cudaGetLastError());
}
}
