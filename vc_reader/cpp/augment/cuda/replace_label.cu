#include "label_values.cuh"
namespace vcr::augment::cuda {
template<class T> __device__ void replace_label(KernelIndex vcr_launch, T* data, std::size_t n, double value, double replacement, int* status) {
    EACH_INDEX(n) if (double(data[i]) == value) data[i] = checked_label_cast<T>(replacement,status);
}
struct replace_label_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { replace_label(index,args...); }
};
void apply_cuda(const ReplaceLabel& t, Sample&, const Parameters&, Session& session) {
    auto& a=session.arrays.at(t.on); require_exact_labels(a,session); require_label_value(t.value); require_label_value(t.replacement);
    const auto channels=selected_channels(t.channels,a.shape[0]); const auto n=product(a.spatial());
    a.typed([&](auto v) { for (auto c:channels) launch(blocks(n),256,0,vcr::device::stream(), replace_label_launch{}, v.ptr+c*n,n,t.value,t.replacement,session.check_flags()); });
    check(cudaGetLastError());
}
}
