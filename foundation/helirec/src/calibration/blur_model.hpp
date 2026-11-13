#pragma once
#include <cmath>
#include "calibration/model_math.hpp"
#include <vector>
#include <type_traits>
#include <ceres/jet.h>
#include "helirec/common.hpp"

namespace helirec::calibration::fitting {
// The complete residual is differentiated: both low-frequency normalisation
// and the eliminated nuisance spectrum depend on the field parameters.
struct BlurFieldProblem {
    int tiles = 0, bins = 0;
    std::vector<double> Y, X, Yr, FX, FY, wgt;
    std::vector<char> low;
    template <class S> struct Workspace { std::vector<S> model, numerator, denominator; };
    // One residual evaluator per fit, used serially by Ceres. Reuse both scalar
    // and derivative workspaces instead of allocating/zeroing the model per call.
    mutable Workspace<double> scalar_workspace;
    mutable Workspace<ceres::Jet<double,6>> derivative_workspace;
    template <class S> Workspace<S>& workspace() const {
        if constexpr (std::is_same_v<S,double>) return scalar_workspace;
        else return derivative_workspace;
    }

    template <class S> bool operator()(const S* th, S* r) const {
        auto& work = workspace<S>();
        auto& M = work.model;
        auto& num = work.numerator;
        auto& den = work.denominator;
        M.resize(static_cast<std::size_t>(tiles)*bins);
        num.assign(bins,S(0)); den.assign(bins,S(0));
        for (int t = 0; t < tiles; ++t) {
            const S sx = th[0]*S(X[t])+th[1]*S(Yr[t])+th[4], sy = th[2]*S(X[t])+th[3]*S(Yr[t])+th[5];
            S* mt = &M[static_cast<std::size_t>(t)*bins];
            S lo = S(0);
            int nlo = 0;
            for (int b = 0; b < bins; ++b) {
                const S v = sinc_pi(sx*S(FX[b])+sy*S(FY[b]));
                mt[b] = v*v;
                if (low[b]) { lo += mt[b]; ++nlo; }
            }
            lo /= S(nlo);
            if (!(lo > S(0))) return false;
            const S inv_lo = S(1)/lo;
            for (int b = 0; b < bins; ++b) {
                mt[b] *= inv_lo;
                num[b] += S(Y[static_cast<std::size_t>(t)*bins+b])*mt[b];
                den[b] += mt[b]*mt[b];
            }
        }
        // The eliminated spectrum is common to all tiles. Differentiate its
        // quotient once per bin, rather than repeating it for each residual.
        for (int b = 0; b < bins; ++b)
            num[b] /= den[b] > S(1e-30) ? den[b] : S(1e-30);
        for (int t = 0; t < tiles; ++t)
            for (int b = 0; b < bins; ++b) {
                const std::size_t i = static_cast<std::size_t>(t)*bins+b;
                r[i] = (S(Y[i])-M[i]*num[b])*S(wgt[b]);
            }
        return true;
    }
    void residual(const double* th, std::vector<double>& r) const {
        r.resize(static_cast<std::size_t>(tiles)*bins);
        if (!(*this)(th,r.data())) throw Error("detector_blur_field_fit: invalid model normalisation");
    }
};

}
