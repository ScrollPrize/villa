#pragma once
// Sampling a 3-D array at arbitrary positions: points, planes, bands of layers along
// per-pixel directions, and composites of those bands. Coordinates are XYZ in the
// array's voxel grid (voxel i at coordinate i; the array is stored ZYX). Values are
// float32: points none of whose neighbours lie in the array read as the fill value,
// non-finite points give NaN. Nearest rounds half ties up; cubic is Catmull-Rom.
// CPU and CUDA backends share one implementation of the interpolation and
// composite formulas.

#include "vc_reader/native/array.hpp"
#include "vc_reader/chunks.hpp"
#include "vc_reader/native/element_type.hpp"

#include <array>
#include <cstddef>
#include <span>

#ifdef VC_READER_CUDA
#include "vc_reader/native/cuda_buffer.hpp"
#endif

namespace vcr::slice {

enum class Order : int { Nearest = 0, Linear = 1, Cubic = 3 };

// How a band of layers becomes one value. Layers below isoCutoff are dropped; a
// pixel without remaining layers is NaN. Alpha and Beer-Lambert expect values on a
// 0-255 scale (alphaMin/alphaMax as fractions of it) and integrate front to back.
struct Composite {
    enum class Method { Mean, Max, Min, Median, MinAbs, Alpha, BeerLambert };
    Method method = Method::Mean;
    float isoCutoff = 0.0f;
    float alphaMin = 0.0f, alphaMax = 1.0f, alphaOpacity = 1.0f, alphaCutoff = 1.0f;
    float extinction = 1.5f, emission = 1.5f, ambient = 0.1f;
};

// Median keeps every layer of a pixel; other methods stream.
inline constexpr std::size_t kMaxMedianLayers = 256;

// A non-blocking snapshot source. `array` must be the registered source's level.
// Sampling probes only: no fetches, queue changes or LRU promotion. Callers publish
// view demand separately. Missing/error chunks invalidate any touching footprint;
// known fill and out-of-array neighbours are available. Ready callbacks trigger retries.
struct Resident {
    Array& array;
    chunks::Service& service;
    chunks::SourceId source;
    std::int32_t level = 0;
};

// Blocking sampling through the same decoded budget used by resident viewers.
// Fetches missing chunks with foreground priority, pins them for each wave and
// throws on fetch errors. Arrays must be registered without a separate MemoryCache.
struct Cached {
    Array& array;
    chunks::Service& service;
    chunks::SourceId source;
    std::int32_t level = 0;
};

// `type` says how the array's elements read as numbers.
void points(Array& array, const ElementType& type, const float* xyz, std::size_t n, Order order, float* out);
// out[row * width + col] at origin + col * u + row * v.
void plane(Array& array, const ElementType& type, std::array<float, 3> origin, std::array<float, 3> u,
           std::array<float, 3> v, std::size_t width, std::size_t height, Order order, float* out);
// out[k * n + p] at base[p] + dirs[p] * offsets[k]. `width` > 0 says the n
// pixels form a row-major image that wide, which keeps reads compact.
void band(Array& array, const ElementType& type, const float* base, const float* dirs, std::size_t n,
          std::size_t width, std::span<const float> offsets, Order order, float* out);
// out[p]: the composite of pixel p's band, in offsets order.
void composite(Array& array, const ElementType& type, const float* base, const float* dirs, std::size_t n,
               std::size_t width, std::span<const float> offsets, Order order, const Composite& method, float* out);

void points(Cached& cached, const ElementType& type, const float* xyz, std::size_t n, Order order, float* out);
// out[row * width + col] at origin + col * u + row * v.
void plane(Cached& cached, const ElementType& type, std::array<float, 3> origin, std::array<float, 3> u,
           std::array<float, 3> v, std::size_t width, std::size_t height, Order order, float* out);
// out[k * n + p] at base[p] + dirs[p] * offsets[k]. `width` > 0 says the n
// pixels form a row-major image that wide, which keeps reads compact.
void band(Cached& cached, const ElementType& type, const float* base, const float* dirs, std::size_t n,
          std::size_t width, std::span<const float> offsets, Order order, float* out);
// out[p]: the composite of pixel p's band, in offsets order.
void composite(Cached& cached, const ElementType& type, const float* base, const float* dirs, std::size_t n,
               std::size_t width, std::span<const float> offsets, Order order, const Composite& method, float* out);

// Resident overloads return one validity byte per output (1 = all required
// chunks available). Invalid outputs are NaN, including incomplete composites.
// `type` says how the array's elements read as numbers.
void points(Resident& resident, const ElementType& type, const float* xyz, std::size_t n, Order order, float* out,
           std::uint8_t* valid = nullptr);
// out[row * width + col] at origin + col * u + row * v.
void plane(Resident& resident, const ElementType& type, std::array<float, 3> origin, std::array<float, 3> u,
           std::array<float, 3> v, std::size_t width, std::size_t height, Order order, float* out,
           std::uint8_t* valid = nullptr);
// out[k * n + p] at base[p] + dirs[p] * offsets[k]. `width` > 0 says the n
// pixels form a row-major image that wide, which keeps reads compact.
void band(Resident& resident, const ElementType& type, const float* base, const float* dirs, std::size_t n,
          std::size_t width, std::span<const float> offsets, Order order, float* out, std::uint8_t* valid = nullptr);
// out[p]: the composite of pixel p's band, in offsets order.
void composite(Resident& resident, const ElementType& type, const float* base, const float* dirs, std::size_t n,
               std::size_t width, std::span<const float> offsets, Order order, const Composite& method, float* out,
           std::uint8_t* valid = nullptr);

#ifdef VC_READER_CUDA
// The same requests computed on the calling thread's CUDA device and stream.
// Inputs are host memory; outputs are float32 device buffers laid out as above
// and complete when the call returns.
namespace cuda {
device::Storage points(Array& array, const ElementType& type, const float* xyz, std::size_t n, Order order);
device::Storage plane(Array& array, const ElementType& type, std::array<float, 3> origin, std::array<float, 3> u,
                      std::array<float, 3> v, std::size_t width, std::size_t height, Order order);
device::Storage band(Array& array, const ElementType& type, const float* base, const float* dirs, std::size_t n,
                     std::size_t width, std::span<const float> offsets, Order order);
device::Storage composite(Array& array, const ElementType& type, const float* base, const float* dirs, std::size_t n,
                          std::size_t width, std::span<const float> offsets, Order order, const Composite& method);
device::Storage points(Cached& cached, const ElementType& type, const float* xyz, std::size_t n, Order order);
device::Storage plane(Cached& cached, const ElementType& type, std::array<float, 3> origin, std::array<float, 3> u,
                      std::array<float, 3> v, std::size_t width, std::size_t height, Order order);
device::Storage band(Cached& cached, const ElementType& type, const float* base, const float* dirs, std::size_t n,
                     std::size_t width, std::span<const float> offsets, Order order);
device::Storage composite(Cached& cached, const ElementType& type, const float* base, const float* dirs, std::size_t n,
                          std::size_t width, std::span<const float> offsets, Order order, const Composite& method);
device::Storage points(Resident& resident, const ElementType& type, const float* xyz, std::size_t n, Order order,
           device::Storage* valid = nullptr);
device::Storage plane(Resident& resident, const ElementType& type, std::array<float, 3> origin, std::array<float, 3> u,
                      std::array<float, 3> v, std::size_t width, std::size_t height, Order order,
           device::Storage* valid = nullptr);
device::Storage band(Resident& resident, const ElementType& type, const float* base, const float* dirs, std::size_t n,
                     std::size_t width, std::span<const float> offsets, Order order, device::Storage* valid = nullptr);
device::Storage composite(Resident& resident, const ElementType& type, const float* base, const float* dirs, std::size_t n,
                          std::size_t width, std::span<const float> offsets, Order order, const Composite& method,
           device::Storage* valid = nullptr);
}  // namespace cuda
#endif

}  // namespace vcr::slice
