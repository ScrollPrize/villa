#include "channel.cuh"
namespace vcr::augment::cuda {
template<class T> __global__ void invert_kernel(View<T> data) {
    EACH_INDEX(data.size) data.ptr[i] = -data.ptr[i];
}
template<class T> __global__ void gamma_kernel(View<T> data, const Statistics* stats, double gamma) {
    EACH_INDEX(data.size) {
        const T low = T(stats->low), range = T(stats->high)-low;
        T value = pow((data.ptr[i]-low)/fmax(range,T(1e-7)),T(gamma))*range+low;
        if constexpr (std::is_same_v<T,float>) {
            // Finite float32 endpoints can still have an overflowing difference.
            if (!isfinite(value) || !isfinite(range) || !isfinite(T(gamma))) {
                const double wide_range = stats->high-stats->low;
                value = T(pow((widen(data.ptr[i])-stats->low)/fmax(wide_range,1e-7),gamma)*wide_range+stats->low);
            }
        }
        data.ptr[i] = value;
    }
}
template<class T> __global__ void finish_gamma(View<T> data, const Statistics* before, const Statistics* after, bool retain, bool invert) {
    EACH_INDEX(data.size) {
        T x = data.ptr[i];
        if (retain) x = T((x-after->mean)*(before->deviation/fmax(after->deviation,1e-7))+before->mean);
        if (invert) x = -x;
        data.ptr[i] = x;
    }
}
void apply_cuda(const Gamma& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto&, auto v, const Grid&, const ChannelPlan& c) {
        const bool invert = c.values[1], retain = c.values[2];
        if (invert) invert_kernel<<<blocks(v.size),256,0,vcr::device::stream()>>>(v);
        auto before = statistics(v,retain);
        gamma_kernel<<<blocks(v.size),256,0,vcr::device::stream()>>>(v,before.ptr,c.values[0]);
        if (retain) {
            auto after = statistics(v,true);
            finish_gamma<<<blocks(v.size),256,0,vcr::device::stream()>>>(v,before.ptr,after.ptr,true,invert);
        } else if (invert) invert_kernel<<<blocks(v.size),256,0,vcr::device::stream()>>>(v);
        check(cudaGetLastError());
    });
}
} // namespace vcr::augment::cuda
