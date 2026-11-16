#pragma once
#include <limits>
#include "helirec/calibration/solve.hpp"

namespace helirec::calibration {
struct RefillSolution : FitSummary {
    // Signed seconds/row, positive decay [s], first shot offset [s], raster [s],
    // four spatial amplitude coefficients, then one amplitude per shot.
    std::vector<double> x, residual;
    std::vector<int> shots;
    double noise = 0;
    double row_time_se = std::numeric_limits<double>::quiet_NaN();
    bool uncertainty_available = false;
};
// x is (frames, n_rows): logarithmic row levels around a refill. pre counts the
// leading frames before the refill. Period T and exposure gate E are seconds.
// Shot selection, both sign initialisations, fitting and covariance run in C++.
RefillSolution fit_refill_shots(const double* x, std::size_t frames, int n_rows, double T, double E, int pre,
                              const SolveOptions& options = {});
}
