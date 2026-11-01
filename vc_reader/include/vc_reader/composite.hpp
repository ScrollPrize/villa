#pragma once
// Reduces stacks of sampled layers to one value per pixel: mean, max, min, median,
// the layer closest to mid-grey, and front-to-back alpha and Beer-Lambert
// compositing. CPU and CUDA evaluate the same float32 formulas.

#include <cstddef>

namespace vcr::composite {

enum class Method { Mean, Max, Min, Median, MinAbs, Alpha, BeerLambert };

// Layers below isoCutoff, and NaN layers (no sample), are dropped; a pixel without
// remaining layers is NaN. Alpha and Beer-Lambert expect values on a 0-255 scale
// (alphaMin/alphaMax as fractions of it) and integrate front to back, layer 0 first.
struct Options {
    Method method = Method::Mean;
    float isoCutoff = 0.0f;
    float alphaMin = 0.0f, alphaMax = 1.0f, alphaOpacity = 1.0f, alphaCutoff = 1.0f;
    float extinction = 1.5f, emission = 1.5f, ambient = 0.1f;
    bool operator==(const Options&) const = default;
};

// Median keeps every layer of a pixel; other methods stream.
inline constexpr std::size_t kMaxMedianLayers = 256;

// out[p], p < n: the reduction of values[k * n + p] over the layers k < layers.
void reduce(const float* values, std::size_t layers, std::size_t n, const Options& options, float* out);

#ifdef VC_READER_CUDA
namespace cuda {
// The same on the calling thread's CUDA device and stream. `values` and `out` are
// device memory; the reduction is complete when the call returns.
void reduce(const float* values, std::size_t layers, std::size_t n, const Options& options, float* out);
}  // namespace cuda
#endif

}  // namespace vcr::composite
