#pragma once
#include "helirec/calibration/solve.hpp"

namespace helirec::calibration {
struct TransferFit : FitSummary {
    std::vector<double> x, residual;
    double rms = 0;
};
// Film transfer: (1 + A k^2) [a exp(-s1^2 k^2/2) + (1-a) exp(-s2^2 k^2/2)],
// k=2*pi*frequency. Bounds [1,0,.2,.2]..[100,1,4,6]; logarithmic residuals.
TransferFit fit_transfer(const std::vector<double>& frequency, const std::vector<double>& gain, const SolveOptions& options = {});
std::vector<double> transfer_model(const std::vector<double>& parameters, const std::vector<double>& frequency);
// Column smear ratio sinc(smear*frequency), smear in [0,2] pixels.
TransferFit fit_smear(const std::vector<double>& frequency, const std::vector<double>& ratio, const SolveOptions& options = {});
}
