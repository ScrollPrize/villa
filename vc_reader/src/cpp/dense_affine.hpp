#pragma once
#include "sampling_math.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace vcr::sampling {
// A bounded contiguous workspace is useful for densely sampled affine volumes.
// Thin planes and widely spaced grids retain direct chunk sampling: their
// enclosing boxes contain too much data that will never be sampled.
struct DenseAffineBlock {
    std::array<std::int64_t,3> start, size, local_start;
    std::size_t count = 1;

    template<class Grid> bool plan(const Grid& grid, std::size_t n) {
        if (n < 4096) return false;
        if (grid.shape[0]+grid.shape[1]+grid.shape[2] > (64u<<20)/(3*sizeof(float))) return false;
        for (int a=0; a<3; ++a) {
            float low=INFINITY, high=-INFINITY;
            for (int corner=0; corner<8; ++corner) {
                const float q=affine(grid.origin[a],grid.axes.data()+3*a,
                    float(corner&1 ? grid.shape[0]-1 : 0),
                    float(corner&2 ? grid.shape[1]-1 : 0),
                    float(corner&4 ? grid.shape[2]-1 : 0));
                if (!std::isfinite(q)) return false;
                low=std::min(low,q); high=std::max(high,q);
            }
            // Keep integer conversion and workspace addressing bounded. The
            // general sampler handles larger or overflowing local coordinates.
            if (!std::isfinite(low) || !std::isfinite(high) || low < -4194304 || high > 4194304)
                return false;
            local_start[a]=std::int64_t(std::floor(low));
            start[a]=grid.base(a)+local_start[a];
            size[a]=std::int64_t(std::floor(high))-local_start[a]+2;
            if (std::size_t(size[a]) > (64u<<20)/count) return false;
            count*=std::size_t(size[a]);
        }
        return count/8+(count%8 != 0) <= n;
    }

    template<class Grid, class Parallel>
    void sample(const Grid& grid, const std::uint8_t* raw, float* out, Parallel&& parallel) const {
        const std::int64_t strides[3]={size[1]*size[2],size[2],1};
        auto load=[](const std::uint8_t* p){return float(*p);};
        // Multiplications are rounded before storage, exactly as in affine().
        // Storage grows with the grid axes, never with the output voxel count.
        std::array<std::vector<std::array<float,3>>,3> terms;
        for (int d=0; d<3; ++d) {
            terms[d].resize(grid.shape[d]);
            for (std::size_t i=0; i<grid.shape[d]; ++i)
                for (int a=0; a<3; ++a)
                    terms[d][i][a]=d==0 ? step(grid.origin[a],grid.axes[3*a],float(i))
                                              : grid.axes[3*a+d]*float(i);
        }
        parallel(grid.shape[0],[&](std::size_t z) {
            std::size_t p=z*grid.shape[1]*grid.shape[2];
            for (std::size_t y=0; y<grid.shape[1]; ++y) {
                std::array<float,3> row;
                for (int a=0; a<3; ++a) row[a]=terms[0][z][a]+terms[1][y][a];
                for (std::size_t x=0; x<grid.shape[2]; ++x,++p) {
                    float frac[3]; std::int64_t offset=0;
                    for (int a=0; a<3; ++a) {
                        const float q=row[a]+terms[2][x][a];
                        // Truncation plus one correction avoids scalar floorf
                        // on CPUs without vector floor instructions.
                        int lo=int(q); lo-=q<float(lo);
                        frac[a]=q-float(lo);
                        offset+=(lo-local_start[a])*strides[a];
                    }
                    out[p]=trilinear(frac,raw+offset,strides,load);
                }
            }
        });
    }
};
} // namespace vcr::sampling
