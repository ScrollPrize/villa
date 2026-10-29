// Interpolation from dense float32 blocks and composite reductions, shared by the
// CPU sampler and the CUDA kernels so both compute the same formulas.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#ifdef VC_READER_CUDA
#include <cuda_runtime.h>
#endif

#if defined(__CUDACC__)
#define VCR_SLICE_HD __host__ __device__ inline
#else
#define VCR_SLICE_HD inline
#endif

namespace vcr::slice::detail {

VCR_SLICE_HD bool finite(float v)
{
#if defined(__CUDA_ARCH__)
    return isfinite(v);
#else
    return std::isfinite(v);
#endif
}

// A dense copy of the array over [origin, origin + shape) (ZYX), with the fill
// value wherever that box leaves the array.
struct Block {
    std::int64_t offset;  // first element in the packed block buffer
    std::int64_t oz, oy, ox;
    std::int32_t nz, ny, nx;
};

// Pixels [x0, x0 + w) x [y0, y0 + h) of a request's unit image, sampled from one block.
struct Tile {
    std::int64_t x0, y0, w, h;
    bool hasBlock = false;  // false: no sample of the tile reaches the array
    Block block{};
};

// Catmull-Rom basis, the cubic used for tricubic sampling.
VCR_SLICE_HD float cubicWeight(float t)
{
    const float at = fabsf(t);
    if (at < 1.0f)
        return (1.5f * at - 2.5f) * at * at + 1.0f;
    if (at < 2.0f)
        return ((-0.5f * at + 2.5f) * at - 4.0f) * at + 2.0f;
    return 0.0f;
}

// Neighbourhood of a point for an interpolation order, as [first, last] voxel indices.
VCR_SLICE_HD void footprint(float c, int order, std::int64_t& first, std::int64_t& last)
{
    if (order == 0) {
        first = last = std::int64_t(rintf(c));
        return;
    }
    const std::int64_t f = std::int64_t(floorf(c));
    first = order == 1 ? f : f - 1;
    last = order == 1 ? f + 1 : f + 2;
}

// A point none of whose neighbours lies in the array (it reads as fill). Coordinates
// far beyond any array are outside before they reach integer conversion.
VCR_SLICE_HD bool outside(float x, float y, float z, std::int64_t sz, std::int64_t sy, std::int64_t sx, int order)
{
    const float c[3] = {z, y, x};
    const std::int64_t n[3] = {sz, sy, sx};
    for (int a = 0; a < 3; ++a) {
        if (!(fabsf(c[a]) < 1.0e15f))
            return true;
        std::int64_t first, last;
        footprint(c[a], order, first, last);
        if (last < 0 || first >= n[a])
            return true;
    }
    return false;
}

VCR_SLICE_HD float at(const float* data, const Block& b, std::int64_t z, std::int64_t y, std::int64_t x)
{
    return data[b.offset + ((z - b.oz) * b.ny + (y - b.oy)) * b.nx + (x - b.ox)];
}

// Samples the block at XYZ (voxel i at coordinate i). The block holds the point's
// whole neighbourhood. Nearest rounds ties to even.
VCR_SLICE_HD float sample(const float* data, const Block& b, float x, float y, float z, int order)
{
    if (order == 0)
        return at(data, b, std::int64_t(rintf(z)), std::int64_t(rintf(y)), std::int64_t(rintf(x)));
    const float fz = floorf(z), fy = floorf(y), fx = floorf(x);
    const std::int64_t iz = std::int64_t(fz), iy = std::int64_t(fy), ix = std::int64_t(fx);
    const float tz = z - fz, ty = y - fy, tx = x - fx;
    if (order == 1) {
        float c[2][2];
        for (int dz = 0; dz < 2; ++dz)
            for (int dy = 0; dy < 2; ++dy) {
                const float v0 = at(data, b, iz + dz, iy + dy, ix), v1 = at(data, b, iz + dz, iy + dy, ix + 1);
                c[dz][dy] = v0 + tx * (v1 - v0);
            }
        const float c0 = c[0][0] + ty * (c[0][1] - c[0][0]);
        const float c1 = c[1][0] + ty * (c[1][1] - c[1][0]);
        return c0 + tz * (c1 - c0);
    }
    float wz[4], wy[4], wx[4];
    for (int k = 0; k < 4; ++k) {
        wz[k] = cubicWeight(tz - float(k - 1));
        wy[k] = cubicWeight(ty - float(k - 1));
        wx[k] = cubicWeight(tx - float(k - 1));
    }
    float result = 0.0f;
    for (int dz = 0; dz < 4; ++dz) {
        float plane = 0.0f;
        for (int dy = 0; dy < 4; ++dy) {
            float row = 0.0f;
            for (int dx = 0; dx < 4; ++dx)
                row += wx[dx] * at(data, b, iz + dz - 1, iy + dy - 1, ix + dx - 1);
            plane += wy[dy] * row;
        }
        result += wz[dz] * plane;
    }
    return result;
}

// Where the samples of a request lie. Unit p (an output pixel, or a point) has K
// samples; the units form an image `width` wide when width > 0.
struct Source {
    enum Kind : int { Points = 0, Plane, Band } kind = Points;
    const float* xyz = nullptr;                   // Points: n XYZ triples
    float origin[3] = {}, u[3] = {}, v[3] = {};   // Plane: origin + col * u + row * v
    const float* base = nullptr;                  // Band: base[p] + dirs[p] * offsets[k]
    const float* dirs = nullptr;
    const float* offsets = nullptr;
    int K = 1;
    std::int64_t n = 0, width = 0;

    // False for a non-finite point (it reads as NaN).
    VCR_SLICE_HD bool point(std::int64_t p, int k, float& x, float& y, float& z) const
    {
        if (kind == Points) {
            x = xyz[3 * p], y = xyz[3 * p + 1], z = xyz[3 * p + 2];
        } else if (kind == Plane) {
            const float col = float(p % width), row = float(p / width);
            x = origin[0] + col * u[0] + row * v[0];
            y = origin[1] + col * u[1] + row * v[1];
            z = origin[2] + col * u[2] + row * v[2];
        } else {
            const float* b = base + 3 * p;
            const float* d = dirs + 3 * p;
            const float o = offsets[k];
            x = b[0] + d[0] * o, y = b[1] + d[1] * o, z = b[2] + d[2] * o;
        }
        return finite(x) && finite(y) && finite(z);
    }
};

enum Method : int { Mean = 0, Max, Min, Median, MinAbs, Alpha, BeerLambert };
constexpr int kMaxMedianLayers = 256;

struct CompositeConfig {
    int method = Mean;
    float isoCutoff = 0.0f;  // layers below this are dropped from the stack
    float alphaMin = 0.0f, alphaMax = 1.0f, alphaOpacity = 1.0f, alphaCutoff = 1.0f;  // [0,1] value scale
    float extinction = 1.5f, emission = 1.5f, ambient = 0.1f;                         // Beer-Lambert
};

// Streams the layers of one pixel front to back; NaN when no layer survives the cutoff.
struct Reducer {
    CompositeConfig c;
    int count = 0;
    float acc = 0.0f, best = 0.0f, alpha = 0.0f, transmittance = 1.0f;
    bool done = false;
    float stack[kMaxMedianLayers];

    VCR_SLICE_HD explicit Reducer(const CompositeConfig& config) : c(config) {}

    VCR_SLICE_HD void add(float v)
    {
        if (!(v >= c.isoCutoff) || done)
            return;
        const bool first = count++ == 0;
        switch (c.method) {
        case Max: best = first || v > best ? v : best; break;
        case Min: best = first || v < best ? v : best; break;
        case Median: stack[count - 1] = v; break;
        case MinAbs: best = first || fabsf(v - 127.5f) < fabsf(best - 127.5f) ? v : best; break;
        case Alpha: {
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
        case BeerLambert: {
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

    VCR_SLICE_HD float result()
    {
        if (count == 0)
            return NAN;
        switch (c.method) {
        case Max:
        case Min:
        case MinAbs: return best;
        case Median: {
            for (int i = 1; i < count; ++i) {  // small stacks: insertion sort
                const float v = stack[i];
                int j = i - 1;
                for (; j >= 0 && stack[j] > v; --j)
                    stack[j + 1] = stack[j];
                stack[j + 1] = v;
            }
            return stack[count / 2];
        }
        case Alpha: return acc * 255.0f;
        case BeerLambert: {
            const float out = (acc + c.ambient * transmittance) * 255.0f;
            return out < 255.0f ? out : 255.0f;
        }
        default: return acc / float(count);
        }
    }
};

#ifdef VC_READER_CUDA
// Samples the tiles of one wave on the device: stores K values per pixel
// (composite false) or reduces them (composite true). Pointers are device memory.
void launch(const Source& source, const Tile* tiles, std::size_t count, const float* blocks, std::int64_t width,
            const std::int64_t shape[3], int order, float fill, bool composite, const CompositeConfig& config, float* out,
            cudaStream_t stream);
#endif

}  // namespace vcr::slice::detail
