// CUDA sampling of planned waves: one thread block per tile, threads over its pixels.
#include "math.hpp"

#include <stdexcept>
#include <string>

namespace vcr::slice::detail {
namespace {
__global__ void sampleTiles(Source source, const Tile* tiles, const float* blocks, std::int64_t width,
                            std::int64_t sz, std::int64_t sy, std::int64_t sx, int order, float fill, bool composite,
                            CompositeConfig config, float* out, const std::uint8_t* coverage, std::uint8_t* valid,
                            float* gradient)
{
    const Tile t = tiles[blockIdx.x];
    const std::int64_t count = t.w * t.h;
    for (std::int64_t i = threadIdx.x; i < count; i += blockDim.x) {
        const std::int64_t p = (t.y0 + i / t.w) * width + t.x0 + i % t.w;
        if (p >= source.n)
            continue;
        auto value = [&](int k) {
            float x, y, z;
            if (!source.point(p, k, x, y, z))
                return NAN;
            if (!t.hasBlock || outside(x, y, z, sz, sy, sx, order))
                return fill;
            return sample(blocks, t.block, x, y, z, order);
        };
        auto available = [&](int k) {
            float x,y,z;
            if (!source.point(p,k,x,y,z)) return false;
            return !t.hasBlock || outside(x,y,z,sz,sy,sx,order) || covered(coverage,t.block,x,y,z,order);
        };
        if (gradient) {  // points only: one sample each
            float x, y, z, g[3] = {0.0f, 0.0f, 0.0f};
            if (!source.point(p, 0, x, y, z)) {
                out[p] = g[0] = g[1] = g[2] = NAN;
            } else if (!t.hasBlock || outside(x, y, z, sz, sy, sx, order)) {
                out[p] = fill;
            } else {
                out[p] = sampleGradient(blocks, t.block, x, y, z, order, g);
            }
            for (int a = 0; a < 3; ++a)
                gradient[3 * p + a] = g[a];
            continue;
        }
        if (!composite) {
            for (int k = 0; k < source.K; ++k) {
                const auto i=std::int64_t(k)*source.n+p;
                const bool ok=available(k);
                out[i]=ok ? value(k) : NAN;
                if (valid) valid[i]=ok;
            }
            continue;
        }
        float x, y, z;
        if (!source.point(p, 0, x, y, z)) {
            if (valid) valid[p]=0;
            out[p] = NAN;
            continue;
        }
        Reducer r(config);
        for (int k = 0; k < source.K && !r.done; ++k) {
            const float v = value(k);
            if (!isnan(v))
                r.add(v);
        }
        bool ok=true;
        for (int k=0; k<source.K; ++k) ok=available(k) && ok;
        if (valid) valid[p]=ok;
        out[p]=ok ? r.result() : NAN;
    }
}
}  // namespace

void launch(const Source& source, const Tile* tiles, std::size_t count, const float* blocks, std::int64_t width,
            const std::int64_t shape[3], int order, float fill, bool composite, const CompositeConfig& config, float* out,
            cudaStream_t stream, const std::uint8_t* coverage, std::uint8_t* valid, float* gradient)
{
    if (!count)
        return;
    sampleTiles<<<unsigned(count), 128, 0, stream>>>(source, tiles, blocks, width, shape[0], shape[1], shape[2], order,
                                                     fill, composite, config, out, coverage, valid, gradient);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUDA sampling launch: ") + cudaGetErrorString(error));
}

}  // namespace vcr::slice::detail
