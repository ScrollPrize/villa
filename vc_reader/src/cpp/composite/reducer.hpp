// The reduction of one pixel's layers, shared by the CPU loop and the CUDA kernel.
#pragma once
#include "vc_reader/composite.hpp"

#include <cmath>

#if defined(__CUDACC__)
#define VCR_COMPOSITE_HD __host__ __device__ inline
#else
#define VCR_COMPOSITE_HD inline
#endif

namespace vcr::composite::detail {

// Throws for requests the reduction cannot hold (a median of more than 256 layers).
void checkLayers(std::size_t layers, const Options& options);

// Streams the layers of one pixel front to back; NaN when no layer survives the cutoff.
struct Reducer {
    Options c;
    int count = 0;
    float acc = 0.0f, best = 0.0f, alpha = 0.0f, transmittance = 1.0f;
    bool done = false;
    float stack[kMaxMedianLayers];

    VCR_COMPOSITE_HD explicit Reducer(const Options& config) : c(config) {}

    VCR_COMPOSITE_HD void add(float v)
    {
        if (!(v >= c.isoCutoff) || done)
            return;
        const bool first = count++ == 0;
        switch (c.method) {
        case Method::Max: best = first || v > best ? v : best; break;
        case Method::Min: best = first || v < best ? v : best; break;
        case Method::Median: stack[count - 1] = v; break;
        case Method::MinAbs: best = first || fabsf(v - 127.5f) < fabsf(best - 127.5f) ? v : best; break;
        case Method::Alpha: {
            const float range = (c.alphaMax - c.alphaMin) * 255.0f;
            if (range == 0.0f)
                return;
            float normalized = (v - c.alphaMin * 255.0f) / range;
            if (normalized <= 0.0f)
                return;
            normalized = normalized > 1.0f ? 1.0f : normalized;
            if (alpha >= c.alphaCutoff) {
                done = true;
                return;
            }
            float opacity = normalized * c.alphaOpacity;
            opacity = opacity > 1.0f ? 1.0f : opacity;
            const float weight = (1.0f - alpha) * opacity;
            acc += weight * normalized;
            alpha += weight;
            break;
        }
        case Method::BeerLambert: {
            if (v < 0.255f)
                return;
            const float layer = expf(-(c.extinction / 255.0f) * v);
            acc += v * (c.emission / 255.0f) * transmittance * (1.0f - layer);
            transmittance *= layer;
            done = transmittance < 0.001f;
            break;
        }
        default: acc += v; break;
        }
    }

    VCR_COMPOSITE_HD float result()
    {
        if (count == 0)
            return NAN;
        switch (c.method) {
        case Method::Max:
        case Method::Min:
        case Method::MinAbs: return best;
        case Method::Median: {
            for (int i = 1; i < count; ++i) {  // small stacks: insertion sort
                const float v = stack[i];
                int j = i - 1;
                for (; j >= 0 && stack[j] > v; --j)
                    stack[j + 1] = stack[j];
                stack[j + 1] = v;
            }
            return stack[count / 2];
        }
        case Method::Alpha: return acc * 255.0f;
        case Method::BeerLambert: {
            const float out = (acc + c.ambient * transmittance) * 255.0f;
            return out < 255.0f ? out : 255.0f;
        }
        default: return acc / float(count);
        }
    }
};

}  // namespace vcr::composite::detail
