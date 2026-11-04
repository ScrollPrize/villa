#pragma once
#include "../ops/coherence_enhancing_diffusion.hpp"
#include "../cpu/structure_tensor.hpp"
#include "volume_grid.hpp"

namespace vcr::transforms::diffusion_detail {
inline StructureTensor tensor_options(const CoherenceEnhancingDiffusion& t) {
    StructureTensor st;
    st.sigma = t.sigma; st.rho = t.rho; st.boundary = Boundary::Nearest;
    return st;
}

// CED eigenvalue map: mu_j = alpha + (1-alpha) exp(-(lambda/(l_max-l_j))^m).
// See Mirebeau et al., Anisotropic Diffusion in ITK, equation (11).
// The input components are replaced with D, in the same upper-triangle order.
template<class T> VCR_HD inline void diffusion_tensor(T* components, std::size_t n, std::size_t i,
                                                      int d, T lambda, T exponent, T alpha) {
    using namespace structure_tensor_detail;
    T matrix[9], values[3], vectors[9];
    for (int c = 0; c < component_count(d); ++c) {
        int a, b; component_axes(c, d, a, b);
        matrix[a*d+b] = matrix[b*d+a] = components[c*n+i];
    }
    vcr::math::symmetric_eigen(matrix, d, values, vectors);
    T rates[3];
    for (int j = 0; j < d; ++j) {
        const T gap = values[d-1] - values[j];
        rates[j] = gap > 0 ? alpha + (T(1)-alpha)*exp(-pow(lambda/gap, exponent)) : alpha;
    }
    for (int c = 0; c < component_count(d); ++c) {
        int a, b; component_axes(c, d, a, b);
        T value = 0;
        for (int j = 0; j < d; ++j) value += rates[j]*vectors[a*d+j]*vectors[b*d+j];
        components[c*n+i] = value;
    }
}

// Forward differences G and their negative adjoint implement div(D G u).
// Zero outward flux conserves mass; eigenvalues(D) <= 1 give ||G*DG|| <= 4d.
// The explicit update is L2 stable for dt <= 1/(2d), including singleton axes.
template<class T, class Layout> VCR_HD inline void flux(const T* input, const T* tensor, T* out,
                                                       const Layout& g, std::size_t i, int d) {
    T gradient[3], result[3]{};
    for (int a = 0; a < d; ++a)
        gradient[a] = g.coord(i,a)+1 < g.n[a] ? input[i+g.stride[a]]-input[i] : T(0);
    for (int c = 0; c < structure_tensor_detail::component_count(d); ++c) {
        int a, b; structure_tensor_detail::component_axes(c,d,a,b);
        const T v = tensor[c*g.size+i];
        result[a] += v*gradient[b];
        if (a != b) result[b] += v*gradient[a];
    }
    for (int a = 0; a < d; ++a)
        out[a*g.size+i] = g.coord(i,a)+1 < g.n[a] ? result[a] : T(0);
}
template<class T, class Layout> VCR_HD inline T update(const T* input, const T* flux,
                                                      const Layout& g, std::size_t i, int d, T dt) {
    T divergence = 0;
    for (int a = 0; a < d; ++a) {
        divergence += flux[a*g.size+i];
        if (g.coord(i,a) > 0) divergence -= flux[a*g.size+i-g.stride[a]];
    }
    return input[i] + dt*divergence;
}
}
