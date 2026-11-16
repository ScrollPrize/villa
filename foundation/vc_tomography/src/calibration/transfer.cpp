#include "helirec/calibration/transfer.hpp"
#include "calibration/ceres_support.hpp"
#include "calibration/model_math.hpp"
#include <array>
#include <set>

namespace helirec::calibration {
namespace {
template <class T> T gain(const T* p, double frequency) {
    using std::exp;
    const T k2 = T(std::pow(2*M_PI*frequency,2));
    return (T(1)+p[0]*k2)*(p[1]*exp(-T(.5)*p[2]*p[2]*k2)+(T(1)-p[1])*exp(-T(.5)*p[3]*p[3]*k2));
}
struct FilmResidual {
    double frequency, observed;
    template <class T> bool operator()(const T* p,T* residual) const {
        using std::log;
        const T predicted = gain(p,frequency);
        if (!(predicted > T(0))) return false;
        residual[0] = T(std::log(observed))-log(predicted);
        return true;
    }
};
struct SmearResidual {
    double frequency, observed;
    template <class T> bool operator()(const T* p,T* residual) const {
        residual[0] = T(observed)-fitting::sinc_pi(p[0]*T(frequency));
        return true;
    }
};
template <class Residual, int N>
TransferFit fit_curve(const std::vector<double>& frequency,const std::vector<double>& data,
                      std::array<double,N> initial,const std::array<double,N>& lower,const std::array<double,N>& upper,
                      const SolveOptions& options) {
    fitting::validate(options);
    if (frequency.size() != data.size() || frequency.size() < N) throw Error("transfer fit: insufficient observations or mismatched sizes");
    std::set<double> distinct;
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (!std::isfinite(frequency[i]) || frequency[i] < 0 || !std::isfinite(data[i])) throw Error("transfer fit: invalid observation");
        if (frequency[i] > 0) distinct.insert(frequency[i]);
    }
    if (distinct.size() < N) throw Error("transfer fit: insufficient distinct positive frequencies");
    ceres::Problem problem;
    for (std::size_t i = 0; i < data.size(); ++i)
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Residual,1,N>(new Residual{frequency[i],data[i]}),nullptr,initial.data());
    for (int i = 0; i < N; ++i) {
        problem.SetParameterLowerBound(initial.data(),i,lower[i]);
        problem.SetParameterUpperBound(initial.data(),i,upper[i]);
    }
    ceres::Solver::Summary summary;
    ceres::Solve(fitting::options(N,options),&problem,&summary);
    TransferFit out;
    static_cast<FitSummary&>(out) = fitting::summary(summary);
    out.x.assign(initial.begin(),initial.end());
    out.residual.resize(data.size());
    double ss = 0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (!Residual{frequency[i],data[i]}(initial.data(),&out.residual[i])) out.usable = false;
        ss += out.residual[i]*out.residual[i];
    }
    out.rms = std::sqrt(ss/data.size());
    return out;
}
}
TransferFit fit_transfer(const std::vector<double>& frequency,const std::vector<double>& observed,const SolveOptions& options) {
    for (double y : observed) if (!(y > 0)) throw Error("transfer fit: logarithmic gains must be positive");
    return fit_curve<FilmResidual,4>(frequency,observed,{18.,.5,.9,1.9},{1.,0.,.2,.2},{100.,1.,4.,6.},options);
}
TransferFit fit_smear(const std::vector<double>& frequency,const std::vector<double>& ratio,const SolveOptions& options) {
    return fit_curve<SmearResidual,1>(frequency,ratio,{1.},{0.},{2.},options);
}
std::vector<double> transfer_model(const std::vector<double>& p,const std::vector<double>& frequency) {
    if (p.size() != 4) throw Error("transfer_model: four parameters required");
    for (double v : p) if (!std::isfinite(v)) throw Error("transfer_model: nonfinite parameter");
    std::vector<double> values;
    values.reserve(frequency.size());
    for (double f : frequency) {
        if (!std::isfinite(f) || f < 0) throw Error("transfer_model: invalid frequency");
        values.push_back(gain(p.data(),f));
    }
    return values;
}
}
