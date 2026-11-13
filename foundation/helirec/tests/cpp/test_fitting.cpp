#include "calibration/blur_model.hpp"
#include "calibration/util.hpp"
#include <ceres/ceres.h>
#include <cmath>
#include <iostream>

int main() {
    // Acquisition diagnostics report invalid observations as an unavailable fit.
    const std::vector<double> time = {1e12,1e12+1,1e12+2};
    const std::vector<double> invalid = {1,std::numeric_limits<double>::quiet_NaN(),3};
    if (helirec::calibration::fit_line(time,invalid,0,time.size()).usable) return 1;
    using helirec::calibration::fitting::BlurFieldProblem;
    BlurFieldProblem model;
    model.tiles = 9; model.bins = 8;
    model.FX = {.06,.08,.12,.15,.18,.22,.26,.29};
    model.FY = {.02,-.04,.08,-.1,.12,-.17,.15,-.12};
    model.low = {1,1,0,0,0,0,0,0};
    model.wgt.assign(8,1.);
    for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
        model.X.push_back(x*.7); model.Yr.push_back(y*.5);
        for (int b = 0; b < 8; ++b) model.Y.push_back(1.+.1*std::sin(2*x+3*y+b));
    }
    helirec::calibration::fitting::BlurFieldCost cost(model);
    for (auto q : {std::array<double,6>{2.,.2,-.1,1.,.3,-.2},std::array<double,6>{0,0,0,0,0,0}}) {
        double r[72], jac[72*6], *j[] = {jac};
        const double* parameters[] = {q.data()};
        if (!cost.Evaluate(parameters,r,j)) return 1;
        for (int k = 0; k < 6; ++k) {
            const double saved = q[k], h = 1e-5;
            double plus[72],minus[72];
            q[k] = saved+h; cost.Evaluate(parameters,plus,nullptr);
            q[k] = saved-h; cost.Evaluate(parameters,minus,nullptr);
            q[k] = saved;
            for (int i = 0; i < 72; ++i) {
                const double numeric = (plus[i]-minus[i])/(2*h);
                if (std::abs(jac[6*i+k]-numeric) > 1e-8+1e-5*std::abs(numeric)) {
                    std::cerr << "blur derivative mismatch: " << i << ' ' << k << '\n';
                    return 1;
                }
            }
        }
    }
    return 0;
}
