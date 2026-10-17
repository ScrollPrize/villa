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
    EACH_INDEX(input.size) {
        const double x = input.ptr[i];
        if (mask_zero && x == 0) continue;
        if constexpr (Phase == 0) {
            if (!isfinite(x)) { atomicExch(invalid,1); continue; }
            p.low = fmin(p.low,x); p.high = fmax(p.high,x); p.weight += 1;
        } else {
            const double scaled = params->scale == 1 ? x : x/params->scale;
            if constexpr (Phase == 1) p.sum += scaled;
            else { const double d = scaled-params->center; p.sum += d*d; }
        }
    }
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

template<class T> __global__ void normalize_values(View<T> data, const ParametersGPU* params, bool mask_zero) {
    EACH_INDEX(data.size) {
        const double x = data.ptr[i];
        if (params->constant || params->divisor == 0 || (mask_zero && x == 0)) data.ptr[i] = T{0};
        else {
            const double scaled = params->scale == 1 ? x : x/params->scale;
            data.ptr[i] = T((scaled-params->center)/params->divisor);
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
