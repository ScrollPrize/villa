#pragma once
#include "core.hpp"

namespace vcr::augment {
struct LocalOptions { Scalar scale = Uniform{10, 30}, loc = Uniform{-1, 2}; };

// A normalized separable field: four scalars followed by Z/Y/X (or Y/X)
// coefficients. Storage grows with axis lengths; consumers evaluate weights.
struct LocalFieldView {
    const float *z, *y, *x;
    std::size_t nz, ny, nx, size;
    float low, range, center, scale;

    LocalFieldView(const std::vector<float>& field, const Grid& g, const float* data = nullptr) {
        const auto dim = g.shape.size();
        require(dim == 2 || dim == 3, "local fields require 2D or 3D images");
        const auto count = std::accumulate(g.shape.begin(),g.shape.end(),std::size_t{4});
        require(field.size() == count, "invalid compact local field; regenerate the replay plan");
        low=field[0]; range=field[1]; center=field[2]; scale=field[3];
        nz=dim == 3 ? g.shape[0] : 1; ny=g.shape[dim-2]; nx=g.shape.back(); size=g.size;
        const float* axes=(data ? data : field.data())+4;
        z=dim == 3 ? axes : nullptr; y=axes+(z ? nz : 0); x=y+ny;
    }
    VCR_HD float value(float prefix, std::size_t ix) const {
        const float raw=prefix*x[ix];
        const float normalized=range ? (raw-low)/range : 0;
        return (normalized-center)/scale;
    }
    template<class Index> VCR_HD float at(Index i) const {
        const Index row=i/Index(nx), ix=i%Index(nx);
        const float prefix=(z ? z[row/Index(ny)] : 1.f)*y[row%Index(ny)];
        return value(prefix,ix);
    }
    VCR_HD float operator[](std::size_t i) const {
        return size <= 0xffffffffull ? at(unsigned(i)) : at(i);
    }
    // CPU consumers hoist row indexing and products out of their voxel loop.
    template<class F> void each(F&& fn) const {
        for(std::size_t iz=0;iz<nz;++iz) for(std::size_t iy=0;iy<ny;++iy) {
            const float prefix=(z ? z[iz] : 1.f)*y[iy];
            const auto base=(iz*ny+iy)*nx;
            for(std::size_t ix=0;ix<nx;++ix) fn(base+ix,value(prefix,ix));
        }
    }
};

inline std::vector<std::vector<float>> local_axes(const LocalOptions& t, const Grid& g, Random& r) {
    std::vector<std::vector<float>> kernels(g.shape.size());
    for (std::size_t a = 0; a < g.shape.size(); ++a) {
        double center = draw(t.loc, r) * g.shape[a], sigma = draw(t.scale, r);
        require(sigma > 0, "local Gaussian scale must be positive");
        kernels[a].resize(g.shape[a]);
        for (std::size_t x = 0; x < g.shape[a]; ++x)
            kernels[a][x] = .5 * (std::erf((x + .5 - center) / (sigma * std::sqrt(2.))) - std::erf((x - .5 - center) / (sigma * std::sqrt(2.))));
        // erf differences need extra precision in the tails; normalize the
        // axis before forming products so bounded float32 fields cannot underflow.
        const float peak = *std::max_element(kernels[a].begin(), kernels[a].end());
        if (peak) for (auto& value : kernels[a]) value /= peak;
    }
    return kernels;
}
inline std::vector<float> local_coefficients(const LocalOptions& t, const Grid& g, Random& r, bool centered = false) {
    const auto axes=local_axes(t,g,r);
    float low=1, high=1;
    // Double only for the short axis sums/product: centering subtracts nearly
    // equal values for broad Gaussians. Stored coefficients/weights are float32.
    double product_mean=1;
    std::vector<float> field(4);
    for(const auto& axis : axes) {
        const auto [lo,hi]=std::minmax_element(axis.begin(),axis.end());
        low*=*lo; high*=*hi;
        if(centered) product_mean*=std::accumulate(axis.begin(),axis.end(),0.)/axis.size();
        field.insert(field.end(),axis.begin(),axis.end());
    }
    field[0]=low; field[1]=high-low; field[2]=0; field[3]=1;
    if(centered && field[1] > 0) {
        float m=float((product_mean-low)/field[1]);
        // Near-constant fields amplify product rounding. Sum the actual weights
        // in that exceptional case, without allocating a dense field.
        if(field[1] < .01f) {
            double sum=0;
            LocalFieldView(field,g).each([&](std::size_t, float w){sum+=w;});
            m=float(sum/g.size);
        }
        field[2]=m;
        field[3]=std::max(m,1-m);
    }
    return field;
}
} // namespace vcr::augment
