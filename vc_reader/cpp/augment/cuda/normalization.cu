#include "channel.cuh"
#include "statistics.cuh"

namespace vcr::augment::cuda {
namespace {
struct ParametersGPU {
    double center, scale, divisor, count;
    bool constant;
};

// Bounds/count, scaled mean, then centered population variance. Separate
// passes retain the reader's protection against overflow and cancellation.
template<class T, int Phase> __global__ void collect_normalization(
        View<T> input, Partial* output, const ParametersGPU* params, bool mask_zero, int* invalid) {
    if constexpr (Phase != 0) if (params->constant) return;
    __shared__ Partial values[256];
    Partial p{0,0,INFINITY,-INFINITY};
    if constexpr (Phase == 0) if (mask_zero) {
        // Bounds and count do not depend on order and are exact in T. With
        // zeros excluded, min/max never compare signed zeros. No float64 work.
        T low = INFINITY, high = -INFINITY; unsigned long long count = 0;
        EACH_INDEX(input.size) {
            const T x = input.ptr[i];
            if (x == T(0)) continue;
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
            if (mask_zero && input.ptr[i] == T(0)) continue;
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
        const Partial* input, unsigned count, ParametersGPU* params, bool zscore) {
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
            *params = {q.low,1,q.high-q.low,q.weight,q.weight == 0 || q.low == q.high};
            if (params->constant) { params->divisor = 0; return; }
            if (zscore) {
                const double magnitude = fmax(fabs(q.low),fabs(q.high));
                if (magnitude > 1e150 || magnitude < 1e-150) params->scale = magnitude;
            } else if (!isfinite(params->divisor)) {
                params->scale = 2; params->center = q.low/2;
                params->divisor = q.high/2-params->center;
            }
        } else if constexpr (Phase == 1) params->center = q.sum/params->count;
        else params->divisor = sqrt(q.sum/params->count);
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
} // namespace

void apply_cuda(const Normalize& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[&](const auto& t, auto data, const Grid&, const ChannelPlan&) {
        using T = typename decltype(data)::value_type;
        const bool zscore = t.mode == Normalization::ZScore;
        const unsigned count = std::min(blocks(data.size),1024u);
        Buffer<Partial> partial(count); Buffer<ParametersGPU> params(1);
        auto* invalid = session.finite_status();
        collect_normalization<T,0><<<count,256>>>(data,partial.ptr,nullptr,zscore,invalid);
        finish_normalization<0><<<1,256>>>(partial.ptr,count,params.ptr,zscore);
        if (zscore) {
            collect_normalization<T,1><<<count,256>>>(data,partial.ptr,params.ptr,true,invalid);
            finish_normalization<1><<<1,256>>>(partial.ptr,count,params.ptr,true);
            collect_normalization<T,2><<<count,256>>>(data,partial.ptr,params.ptr,true,invalid);
            finish_normalization<2><<<1,256>>>(partial.ptr,count,params.ptr,true);
        }
        normalize_values<<<blocks(data.size),256>>>(data,params.ptr,zscore);
        check(cudaGetLastError());
    });
}
} // namespace vcr::augment::cuda
