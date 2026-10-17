#include "label_values.cuh"
namespace vcr::augment::cuda {
template<class M> __global__ void negative_mask(const M* mask, std::uint8_t* negative, std::size_t n) { EACH_INDEX(n) { if constexpr (std::is_unsigned_v<M>) negative[i]=0; else negative[i]=mask[i]<0; } }
template<class T> __global__ void mask_values(T* image, const std::uint8_t* negative, std::size_t n, double fill, int* status) {
    EACH_INDEX(n) if (negative[i]) image[i] = checked_label_cast<T>(fill,status);
}
void apply_cuda(const MaskImage& t, Sample&, const Parameters&, Session& session) {
    auto& a=session.arrays.at(t.on); const auto& mask=session.arrays.at(t.mask); const auto n=product(a.spatial());
    require(a.spatial()==mask.spatial() && t.mask_channel<mask.shape[0],"invalid image mask");
    Buffer<std::uint8_t> negative(n);
    mask.typed([&](auto v) { negative_mask<<<blocks(n),256,0,vcr::device::stream()>>>(v.ptr+t.mask_channel*n,negative.ptr,n); });
    const auto channels=selected_channels(t.channels,a.shape[0]);
    a.typed([&](auto v) { for (auto c:channels) mask_values<<<blocks(n),256,0,vcr::device::stream()>>>(v.ptr+c*n,negative.ptr,n,t.fill,session.check_flags()); });
    check(cudaGetLastError());
}
}
