#include "helirec/calibration/refill.hpp"
#include "calibration/ceres_support.hpp"
#include <ceres/covariance.h>
#include <algorithm>
#include <array>
#include <numeric>

namespace helirec::calibration {
namespace {
struct Dip {
    double time, row, u, index, raster, exposure, observed, noise;
    template <typename S> bool operator()(const S* q, const S* amplitude, S* residual) const {
        using std::exp;
        const S theta = S(1e-3) * exp(q[1]);
        const S s = S(time) + q[0] * S(1e-6 * row) - (q[2]*S(1e-3) + S(index)*(S(raster)+q[3]*S(1e-5)));
        const auto primitive = [&](const S& z) -> S { return z > S(0) ? S(1) - exp(-z/theta) : S(0); };
        const S alpha = S(1) + S(.1*u)*(q[4] + S(u)*(q[5] + S(u)*(q[6] + S(u)*q[7])));
        const S model = -amplitude[0]*S(1e-3)*alpha*(primitive(s+S(exposure))-primitive(s));
        residual[0] = (S(observed)-model)/S(noise);
        return true;
    }
};
}
RefillSolution fit_refill_shots(const double* x, std::size_t frames, int n_rows, double T, double E, int pre, const SolveOptions& options) {
    fitting::validate(options);
    if (!x || n_rows < 20 || !std::isfinite(T) || !std::isfinite(E) || T <= 0 || E <= 0 || pre < 0 || static_cast<std::size_t>(pre) > frames)
        throw Error("fit_refill_shots: invalid dimensions, timing or pre-refill interval");
    RefillSolution out;
    out.solver_version = CERES_VERSION_STRING;
    const int bands = n_rows/20;
    if (frames > static_cast<std::size_t>(std::numeric_limits<int>::max())) throw Error("fit_refill_shots: too many frames");
    const int nf = static_cast<int>(frames);
    std::vector<double> level(frames*bands), mean(frames);
    for (std::size_t n = 0; n < frames; ++n) {
        for (int r = 0; r < n_rows; ++r) {
            const double v = x[n*n_rows+r];
            if (!std::isfinite(v)) throw Error("fit_refill_shots: nonfinite row level");
            mean[n] += v/n_rows;
            if (r < bands*20) level[n*bands+r/20] += v/20;
        }
    }
    const auto dip = [&](int n, int b) {
        const auto at = [&](int k) { return b < 0 ? mean[k] : level[static_cast<std::size_t>(k)*bands+b]; };
        double before = 0, after = 0;
        for (int k = n-7; k < n-2; ++k) before += at(k)/5;
        for (int k = n+4; k < n+9; ++k) after += at(k)/5;
        return at(n) - (before + (after-before)*(4.5/10.5));
    };
    for (int n = std::max(16,pre-8); n < nf-16; ++n) {
        const double d = dip(n,-1);
        if (d >= -.003) continue;
        if (!out.shots.empty() && n-out.shots.back() <= 2) {
            if (d < dip(out.shots.back(),-1)) out.shots.back() = n;
        } else out.shots.push_back(n);
    }
    if (out.shots.size() < 12 || bands < 5) {
        out.termination = "INSUFFICIENT_SUPPORT"; out.message = "refill fit needs 12 shots and 5 row bands"; return out;
    }
    std::vector<double> quiet;
    for (int n = 16; n < (pre > 80 ? pre-16 : nf-16); ++n) {
        if (pre <= 80 && std::any_of(out.shots.begin(),out.shots.end(),[&](int shot) { return std::abs(shot-n) <= 3; })) continue;
        for (int b = 0; b < bands; ++b) quiet.push_back(dip(n,b));
    }
    if (!quiet.empty()) {
        const double m = std::accumulate(quiet.begin(),quiet.end(),0.)/quiet.size();
        for (double v : quiet) out.noise += (v-m)*(v-m);
        out.noise = std::sqrt(out.noise/quiet.size());
    }
    if (!(out.noise > 0) || !std::isfinite(out.noise)) {
        out.termination = "INVALID_NOISE"; out.message = "refill fit has no positive finite quiet-frame noise estimate"; return out;
    }
    const std::size_t ns = out.shots.size();
    std::vector<double> time(ns), index(ns,0), gaps(ns-1);
    for (std::size_t j = 0; j < ns; ++j) time[j] = (out.shots[j]-out.shots[0])*T;
    for (std::size_t j = 1; j < ns; ++j) gaps[j-1] = time[j]-time[j-1];
    const double minimum = *std::min_element(gaps.begin(),gaps.end());
    double raster = 0; int ng = 0;
    for (double g : gaps) if (g < 1.5*minimum) { raster += g; ++ng; }
    raster /= ng;
    for (std::size_t j = 1; j < ns; ++j) index[j] = index[j-1] + std::nearbyint(gaps[j-1]/raster);
    const auto line = fit_line(index,time,options);
    if (!line.usable) { static_cast<FitSummary&>(out) = line; return out; }
    raster = line.slope;
    std::vector<Dip> data;
    data.reserve(ns*3*bands);
    for (std::size_t j = 0; j < ns; ++j)
        for (int o = -1; o <= 1; ++o)
            for (int b = 0; b < bands; ++b) {
                const double vc = (b+.5)*20-.5;
                data.push_back({time[j]+o*T,vc,(vc-(n_rows-1)*.5)/(n_rows*.5),index[j],raster,E,dip(out.shots[j]+o,b),out.noise});
            }
    double best = std::numeric_limits<double>::infinity();
    for (double sign : {-1.,1.}) {
        std::array<double,8> q{}, start{};
        std::vector<double> amplitude(ns,9.);
        q[0] = sign*.6*T/n_rows/1e-6; q[1] = std::log(3.);
        double start_cost = std::numeric_limits<double>::infinity();
        for (int t = 0; t < 42; ++t) {
            q[2] = (-1.3*T + t*T/16)/1e-3;
            double cost = 0;
            for (std::size_t k = 0; k < data.size(); ++k) { double r; data[k](q.data(),&amplitude[k/(3*bands)],&r); cost += r*r; }
            if (cost < start_cost) { start_cost = cost; start = q; }
        }
        q = start;
        ceres::Problem problem;
        for (std::size_t k = 0; k < data.size(); ++k)
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Dip,1,8,1>(new Dip(data[k])),nullptr,q.data(),&amplitude[k/(3*bands)]);
        // Keep exp in a representable physical range; decay remains strictly positive.
        problem.SetParameterLowerBound(q.data(),1,std::log(T*1e-6/1e-3));
        problem.SetParameterUpperBound(q.data(),1,std::log(T*1e3/1e-3));
        ceres::Solver::Summary summary;
        ceres::Solve(fitting::options(static_cast<int>(8+ns),options),&problem,&summary);
        if (!summary.IsSolutionUsable() || !std::isfinite(summary.final_cost) || summary.final_cost >= best) continue;
        best = summary.final_cost;
        static_cast<FitSummary&>(out) = fitting::summary(summary);
        out.x = {q[0]*1e-6,std::exp(q[1])*1e-3,q[2]*1e-3,raster+q[3]*1e-5,q[4]*.1,q[5]*.1,q[6]*.1,q[7]*.1};
        for (double a : amplitude) out.x.push_back(a*1e-3);
        out.residual.resize(data.size());
        for (std::size_t k = 0; k < data.size(); ++k) data[k](q.data(),&amplitude[k/(3*bands)],&out.residual[k]);
        out.uncertainty_available = false;
        out.row_time_se = std::numeric_limits<double>::quiet_NaN();
        const auto dof = static_cast<std::ptrdiff_t>(data.size())-static_cast<std::ptrdiff_t>(8+ns);
        ceres::Covariance::Options cov_options;
        cov_options.algorithm_type = ceres::DENSE_SVD;
        cov_options.num_threads = 1;
        cov_options.min_reciprocal_condition_number = options.rcond;
        ceres::Covariance covariance(cov_options);
        if (dof > 0 && covariance.Compute(std::vector<std::pair<const double*,const double*>>{{q.data(),q.data()}},&problem)) {
            double cov[64];
            if (covariance.GetCovarianceBlock(q.data(),q.data(),cov) && cov[0] >= 0 && std::isfinite(cov[0])) {
                out.row_time_se = 1e-6*std::sqrt(cov[0]*2*best/dof);
                out.uncertainty_available = std::isfinite(out.row_time_se);
            }
        }
    }
    if (!out.usable) { out.termination = "FAILURE"; out.message = "neither refill sign initialization produced a usable fit"; }
    return out;
}
}
