#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <ceres/cost_function.h>
#include "calibration/model_math.hpp"
#include "helirec/common.hpp"

namespace helirec::calibration::fitting {
// Complete derivative, including low-frequency normalization and the eliminated
// nuisance spectrum. Each tile depends on only its two smear components; carry
// those two derivatives through normalization before mapping to the six field
// parameters. This avoids six-component arithmetic in the trigonometric model.
struct BlurFieldProblem {
    int tiles = 0, bins = 0;
    std::vector<double> Y, X, Yr, FX, FY, wgt;
    std::vector<char> low;
    using Derivative = std::array<double,6>;
    struct Workspace {
        std::vector<double> model, dx, dy, numerator, denominator;
        std::vector<Derivative> d_numerator, d_denominator;
    };
    // Used serially by Ceres; independent fits own independent workspaces.
    mutable Workspace work;

    bool evaluate(const double* th, double* r, double* jacobian) const {
        auto& [M,dx,dy,num,den,dnum,dden] = work;
        M.resize(static_cast<std::size_t>(tiles)*bins);
        num.assign(bins,0.); den.assign(bins,0.);
        if (jacobian) {
            dx.resize(M.size()); dy.resize(M.size());
            dnum.assign(bins,Derivative{}); dden.assign(bins,Derivative{});
        }
        for (int t = 0; t < tiles; ++t) {
            const double sx = th[0]*X[t]+th[1]*Yr[t]+th[4], sy = th[2]*X[t]+th[3]*Yr[t]+th[5];
            const std::size_t first = static_cast<std::size_t>(t)*bins;
            double lo = 0, lx = 0, ly = 0;
            int nlo = 0;
            for (int b = 0; b < bins; ++b) {
                const std::size_t i = first+b;
                const double a = sx*FX[b]+sy*FY[b], z = M_PI*a, v = sinc_pi(a);
                M[i] = v*v;
                if (jacobian) {
                    const double dv = M_PI*(z*z < 1e-12 ? -z/3+z*z*z/30 : (std::cos(z)-v)/z);
                    dx[i] = 2*v*dv*FX[b]; dy[i] = 2*v*dv*FY[b];
                }
                if (low[b]) {
                    lo += M[i]; ++nlo;
                    if (jacobian) { lx += dx[i]; ly += dy[i]; }
                }
            }
            lo /= nlo; lx /= nlo; ly /= nlo;
            if (!(lo > 0)) return false;
            const double inv_lo = 1/lo;
            for (int b = 0; b < bins; ++b) {
                const std::size_t i = first+b;
                M[i] *= inv_lo;
                num[b] += Y[i]*M[i]; den[b] += M[i]*M[i];
                if (jacobian) {
                    dx[i] = (dx[i]-M[i]*lx)*inv_lo;
                    dy[i] = (dy[i]-M[i]*ly)*inv_lo;
                    const Derivative g = {dx[i]*X[t],dx[i]*Yr[t],dy[i]*X[t],dy[i]*Yr[t],dx[i],dy[i]};
                    for (int k = 0; k < 6; ++k) {
                        dnum[b][k] += Y[i]*g[k]; dden[b][k] += 2*M[i]*g[k];
                    }
                }
            }
        }
        // One quotient per bin: the eliminated spectrum is common to all tiles.
        for (int b = 0; b < bins; ++b) {
            const double d = std::max(den[b],1e-30);
            num[b] /= d;
            if (jacobian)
                for (int k = 0; k < 6; ++k)
                    dnum[b][k] = (dnum[b][k]-(den[b] > 1e-30 ? num[b]*dden[b][k] : 0))/d;
        }
        for (int t = 0; t < tiles; ++t)
            for (int b = 0; b < bins; ++b) {
                const std::size_t i = static_cast<std::size_t>(t)*bins+b;
                r[i] = (Y[i]-M[i]*num[b])*wgt[b];
                if (jacobian) {
                    const Derivative g = {dx[i]*X[t],dx[i]*Yr[t],dy[i]*X[t],dy[i]*Yr[t],dx[i],dy[i]};
                    for (int k = 0; k < 6; ++k) jacobian[6*i+k] = -(g[k]*num[b]+M[i]*dnum[b][k])*wgt[b];
                }
            }
        return true;
    }
    void residual(const double* th, std::vector<double>& r) const {
        r.resize(static_cast<std::size_t>(tiles)*bins);
        if (!evaluate(th,r.data(),nullptr)) throw Error("detector_blur_field_fit: invalid model normalization");
    }
};
class BlurFieldCost final : public ceres::CostFunction {
    const BlurFieldProblem& model_;
  public:
    explicit BlurFieldCost(const BlurFieldProblem& model) : model_(model) {
        set_num_residuals(model.tiles*model.bins);
        mutable_parameter_block_sizes()->push_back(6);
    }
    bool Evaluate(double const* const* parameters, double* residual, double** jacobian) const override {
        return model_.evaluate(parameters[0],residual,jacobian ? jacobian[0] : nullptr);
    }
};
}
