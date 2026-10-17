#include "channel.cuh"
#include "statistics.cuh"
#include "sort_keys.cuh"

namespace vcr::augment::cuda {
namespace {
using namespace normalization_detail;
struct ParametersGPU {
    double center, scale, divisor, count, low, high;
    bool constant;
};
// Selection for masked statistics: one byte per voxel, or null for all voxels.
struct Selection { const std::uint8_t* valid; bool nonzero; };
template<class T> __device__ inline bool skip(const Selection& s, std::size_t i, T x) {
    return (s.valid && !s.valid[i]) || (s.nonzero && x == T(0));
}

// Bounds/count, scaled mean, then centered population variance. Separate
// passes retain the reader's protection against overflow and cancellation.
template<class T, int Phase> __global__ void collect_normalization(
        View<T> input, Partial* output, const ParametersGPU* params, Selection select, int* invalid) {
    if constexpr (Phase != 0) if (params->constant) return;
    __shared__ Partial values[256];
    Partial p{0,0,INFINITY,-INFINITY};
    if constexpr (Phase == 0) if (select.nonzero) {
        // Bounds and count do not depend on order and are exact in T. With
        // zeros excluded, min/max never compare signed zeros. No float64 work.
        T low = INFINITY, high = -INFINITY; unsigned long long count = 0;
        EACH_INDEX(input.size) {
            const T x = input.ptr[i];
            if (skip(select,i,x)) continue;
            if (!isfinite(x)) { atomicExch(invalid,1); continue; }
            low = x < low ? x : low; high = x > high ? x : high; ++count;
        }
        p.low = low; p.high = high; p.weight = double(count);
        values[threadIdx.x] = p; reduce(values);
        if (!threadIdx.x) output[blockIdx.x] = values[0];
        return;
    }
    const double scale = Phase == 0 ? 1. : params->scale, center = Phase == 2 ? params->center : 0.;
    // The loop is versioned on the uniform scale test: an in-loop division
    // branch defeats unrolling. Each element's arithmetic and order are unchanged.
    const auto each = [&](auto unscaled) {
        EACH_INDEX(input.size) {
            // Exact: a T value is zero exactly when its float64 conversion is.
            if (skip(select,i,input.ptr[i])) continue;
            const double x = to_double(input.ptr[i]);
            if constexpr (Phase == 0) {
                if (!isfinite(x)) { atomicExch(invalid,1); continue; }
                p.low = fmin(p.low,x); p.high = fmax(p.high,x); p.weight += 1;
            } else {
                const double scaled = decltype(unscaled)::value ? x : x/scale;
                if constexpr (Phase == 1) p.sum += scaled;
                else { const double d = scaled-center; p.sum += d*d; }
            }
        }
    };
    if (scale == 1) each(std::true_type{}); else each(std::false_type{});
    values[threadIdx.x] = p; reduce(values);
    if (!threadIdx.x) output[blockIdx.x] = values[0];
}

template<int Phase> __global__ void finish_normalization(
        const Partial* input, unsigned count, ParametersGPU* params, bool zscore, double eps) {
    if constexpr (Phase != 0) if (params->constant) return;
    __shared__ Partial values[256];
    Partial p{0,0,INFINITY,-INFINITY};
    for (unsigned i = threadIdx.x; i < count; i += 256) {
        auto q = input[i]; p.sum += q.sum; p.weight += q.weight;
        p.low = fmin(p.low,q.low); p.high = fmax(p.high,q.high);
    }
    values[threadIdx.x] = p; reduce(values);
    if (!threadIdx.x) {
        const auto q = values[0];
        if constexpr (Phase == 0) {
            *params = {q.low,1,q.high-q.low,q.weight,q.low,q.high,q.weight == 0 || q.low == q.high};
            if (params->constant) { params->divisor = 0; return; }
            if (zscore) {
                const double magnitude = fmax(fabs(q.low),fabs(q.high));
                if (magnitude > 1e150 || magnitude < 1e-150) params->scale = magnitude;
            } else if (!isfinite(params->divisor)) {
                params->scale = 2; params->center = q.low/2;
                params->divisor = q.high/2-params->center;
            }
        } else if constexpr (Phase == 1) params->center = q.sum/params->count;
        else params->divisor = fmax(sqrt(q.sum/params->count),eps/params->scale);
    }
}

// RN32(s/d) for float32 output without a per-element float64 division. With
// y = RN64(1/d), one correction q1 = RN64(q0 + RN64(s - q0*d)*y), q0 = RN64(s*y),
// is within one float64 ulp of RN64(s/d). Both round to the same float32 unless a
// float32 rounding midpoint lies within one ulp of q1 or q1 is outside the
// normal float32 range; those cases divide exactly.
// Out of line: an inlined division in the per-element loop slows the common path.
__device__ __noinline__ float divide_exactly(double s, double d) { return float(s/d); }
__device__ inline float divide_to_float(double s, double d, double y) {
    const double q0 = s*y, q1 = fma(fma(-q0,d,s),y,q0);
    const auto bits = static_cast<unsigned long long>(__double_as_longlong(q1));
    const auto magnitude = bits & 0x7fffffffffffffffull;
    // Midpoints have low 29 significand bits equal to 2**28; [2**-125, 2**127).
    const bool safe = magnitude >= 0x3820000000000000ull && magnitude < 0x47e0000000000000ull &&
                      (unsigned(bits) & 0x1fffffffu) - ((1u << 28) - 1) > 2u;
    if (safe) return narrow(q1);
    return divide_exactly(s,d);
}

template<class T> __global__ void normalize_values(View<T> data, const ParametersGPU* params, bool mask_zero) {
    const ParametersGPU q = *params;
    const bool zero = q.constant || q.divisor == 0;
    if constexpr (std::is_same_v<T,float>) if (!zero && q.scale == 1) {
        const double y = 1/q.divisor;
        EACH_INDEX(data.size) {
            const T raw = data.ptr[i];
            data.ptr[i] = mask_zero && raw == T(0) ? T{0} : divide_to_float(widen(raw)-q.center,q.divisor,y);
        }
        return;
    }
    EACH_INDEX(data.size) {
        const T raw = data.ptr[i];
        if (zero || (mask_zero && raw == T(0))) data.ptr[i] = T{0};
        else {
            const double x = to_double(raw);
            const double scaled = q.scale == 1 ? x : x/q.scale;
            data.ptr[i] = from_double<T>((scaled-q.center)/q.divisor);
        }
    }
}

// Validity flags for one channel, counting valid voxels; with none valid,
// statistics use every voxel (flags become all ones).
template<class M> __global__ void select_valid(const M* mask, std::uint8_t* valid, std::size_t n, unsigned long long* count) {
    unsigned long long local = 0;
    EACH_INDEX(n) { valid[i] = mask[i] != M(0); local += valid[i]; }
    if (local) atomicAdd(count,local);
}
__global__ void select_all_if_empty(std::uint8_t* valid, std::size_t n, const unsigned long long* count) {
    if (*count) return;
    EACH_INDEX(n) valid[i] = 1;
}

// Order statistics sort the selected values; unselected voxels sort last as +inf.
template<class T> __global__ void order_keys(View<T> data, const std::uint8_t* valid, T* keys, int* invalid) {
    EACH_INDEX(data.size) {
        const T x = data.ptr[i];
        if (!isfinite(x)) atomicExch(invalid,1);
        keys[i] = !valid || valid[i] ? x : T(INFINITY);
    }
}
struct OrderGPU { double low, high, center, scale, deviation, count; bool zero; };
// Kernel-copyable mode options (Normalize itself holds channel-selection vectors).
struct Options {
    Normalization mode; bool clip, median; double lower, upper, low, high, center, scale;
    explicit Options(const Normalize& t) : mode(t.mode),clip(t.clip),median(t.median),lower(t.lower),upper(t.upper),
        low(t.low),high(t.high),center(t.center),scale(t.scale) {}
};
// One thread: percentiles and median of the clipped values. Clipping is
// monotone, so the clipped values stay sorted.
template<class T> __global__ void order_parameters(const T* sorted, const unsigned long long* selected, std::size_t n,
                                                   OrderGPU* out, Options t, const ParametersGPU* moments) {
    const std::size_t count = selected ? std::size_t(*selected) ? std::size_t(*selected) : n : n;
    const auto at = [&](std::size_t k) { return double(sorted[k]); };
    const bool clip = t.mode == Normalization::Percentile || t.clip;
    OrderGPU q{at(0),at(count-1),0,1,0,double(count),false};
    if (clip) { q.low = sorted_percentile(at,count,t.lower); q.high = sorted_percentile(at,count,t.upper); }
    const auto clipped = [&](std::size_t k) { return double(clamp_to(sorted[k],q.low,q.high)); };
    if (t.mode == Normalization::Percentile) {
        q.scale = percentile_scale(q.low,q.high,t.median);
        q.center = t.median ? sorted_percentile(clipped,count,50) : q.low;
        q.zero = q.scale == 0;
    } else {
        q.center = sorted_percentile(clipped,count,50);
        q.deviation = moments->divisor*moments->scale;
    }
    *out = q;
}
template<class T> __global__ void deviation_keys(View<T> data, const std::uint8_t* valid, double* keys, const OrderGPU* params) {
    const OrderGPU q = *params;
    EACH_INDEX(data.size)
        keys[i] = !valid || valid[i] ? fabs(double(clamp_to(data.ptr[i],q.low,q.high))-q.center) : double(INFINITY);
}
__global__ void robust_parameters(const double* sorted, OrderGPU* params) {
    auto& q = *params;
    const auto at = [&](std::size_t k) { return sorted[k]; };
    q.scale = robust_scale(sorted_percentile(at,std::size_t(q.count),50),q.deviation,q.high-q.low);
}
template<class T> __global__ void order_values(View<T> data, const OrderGPU* params, bool clip) {
    const OrderGPU q = *params;
    EACH_INDEX(data.size) {
        if (q.zero) { data.ptr[i] = T(0); continue; }
        const T x = clip ? clamp_to(data.ptr[i],q.low,q.high) : data.ptr[i];
        data.ptr[i] = finite_or_zero<T>((double(x)-q.center)/q.scale);
    }
}
template<class T> __global__ void fixed_values(View<T> data, Options t, int* invalid) {
    EACH_INDEX(data.size) {
        const T x = data.ptr[i];
        if (!isfinite(x)) atomicExch(invalid,1);
        data.ptr[i] = finite_or_zero<T>((double(clamp_to(x,t.low,t.high))-t.center)/t.scale);
    }
}
// Mean/standard deviation (and bounds) of the selected values.
template<class T> void moments(View<T> data, Selection select, ParametersGPU* params, int* invalid, double eps, bool zscore) {
    const unsigned count = std::min(blocks(data.size),1024u);
    Buffer<Partial> partial(count);
    collect_normalization<T,0><<<count,256,0,vcr::device::stream()>>>(data,partial.ptr,nullptr,select,invalid);
    finish_normalization<0><<<1,256,0,vcr::device::stream()>>>(partial.ptr,count,params,zscore,eps);
    if (!zscore) return;
    collect_normalization<T,1><<<count,256,0,vcr::device::stream()>>>(data,partial.ptr,params,select,invalid);
    finish_normalization<1><<<1,256,0,vcr::device::stream()>>>(partial.ptr,count,params,true,eps);
    collect_normalization<T,2><<<count,256,0,vcr::device::stream()>>>(data,partial.ptr,params,select,invalid);
    finish_normalization<2><<<1,256,0,vcr::device::stream()>>>(partial.ptr,count,params,true,eps);
}
} // namespace

void apply_cuda(const Normalize& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[&](const auto& t, auto data, const Grid&, const ChannelPlan& c) {
        using T = typename decltype(data)::value_type;
        auto* invalid = session.finite_status();
        if (t.mode == Normalization::Fixed) {
            fixed_values<<<blocks(data.size),256,0,vcr::device::stream()>>>(data,Options(t),invalid); check(cudaGetLastError()); return;
        }
        std::unique_ptr<Buffer<std::uint8_t>> valid;
        Buffer<unsigned long long> selected(1);
        const auto& f = s.at(c.key);
        if (t.masked && session.arrays.count(f.target.valid)) {
            const auto& mask = session.arrays.at(f.target.valid);
            valid = std::make_unique<Buffer<std::uint8_t>>(data.size);
            check(cudaMemsetAsync(selected.ptr,0,sizeof(unsigned long long),vcr::device::stream()));
            const std::size_t offset = (mask.shape[0] == 1 ? 0 : c.channel)*data.size;
            mask.typed([&](auto m) { select_valid<<<blocks(data.size),256,0,vcr::device::stream()>>>(m.ptr+offset,valid->ptr,data.size,selected.ptr); });
            select_all_if_empty<<<blocks(data.size),256,0,vcr::device::stream()>>>(valid->ptr,data.size,selected.ptr);
        }
        const Selection select{valid ? valid->ptr : nullptr,t.mode == Normalization::ZScore && t.nonzero};
        Buffer<ParametersGPU> params(1);
        if (t.mode == Normalization::MinMax || t.mode == Normalization::ZScore) {
            const bool zscore = t.mode == Normalization::ZScore;
            moments(data,select,params.ptr,invalid,t.eps,zscore);
            normalize_values<<<blocks(data.size),256,0,vcr::device::stream()>>>(data,params.ptr,select.nonzero);
            check(cudaGetLastError()); return;
        }
        if (t.mode == Normalization::Robust) moments(data,select,params.ptr,invalid,0.,true);
        Buffer<T> keys(data.size); Buffer<OrderGPU> order(1);
        order_keys<<<blocks(data.size),256,0,vcr::device::stream()>>>(data,select.valid,keys.ptr,invalid);
        sort_keys(keys,data.size);
        order_parameters<<<1,1,0,vcr::device::stream()>>>(keys.ptr,valid ? selected.ptr : nullptr,data.size,order.ptr,Options(t),params.ptr);
        if (t.mode == Normalization::Robust) {
            Buffer<double> deviations(data.size);
            deviation_keys<<<blocks(data.size),256,0,vcr::device::stream()>>>(data,select.valid,deviations.ptr,order.ptr);
            sort_keys(deviations,data.size);
            robust_parameters<<<1,1,0,vcr::device::stream()>>>(deviations.ptr,order.ptr);
        }
        order_values<<<blocks(data.size),256,0,vcr::device::stream()>>>(data,order.ptr,t.mode == Normalization::Percentile || t.clip);
        check(cudaGetLastError());
    });
}
} // namespace vcr::augment::cuda
