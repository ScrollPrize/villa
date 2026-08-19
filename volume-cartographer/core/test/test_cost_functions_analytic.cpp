#include "vc/core/util/Geometry.hpp"
#include "vc/tracer/CostFunctions.hpp"

#include <array>
#include <chrono>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

namespace {

double relerr(double x, double y)
{
    double denom = std::max(1.0, std::max(std::fabs(x), std::fabs(y)));
    return std::fabs(x - y) / denom;
}

}

TEST_CASE("DistLossAnalytic matches AutoDiffCostFunction<DistLoss> away from the target-distance kink")
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> coord(-50.0, 50.0);
    std::uniform_real_distribution<double> distd(0.1, 20.0);
    std::uniform_real_distribution<double> wd(0.1, 5.0);
    std::uniform_real_distribution<double> kink_off(0.05, 0.95);

    double max_res_relerr = 0.0;
    double max_jac_relerr = 0.0;

    for (int trial = 0; trial < 50000; ++trial) {
        double a[3] = {coord(rng), coord(rng), coord(rng)};
        double target_d = distd(rng);
        double w = wd(rng);

        double dir[3] = {coord(rng), coord(rng), coord(rng)};
        double n = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]) + 1e-9;
        for (double& c : dir) c /= n;

        double side = (trial % 2 == 0) ? 1.0 : -1.0;
        double actual_d = target_d * (1.0 + side * kink_off(rng));
        double b[3];
        for (int i = 0; i < 3; ++i) b[i] = a[i] + actual_d * dir[i];

        std::unique_ptr<ceres::CostFunction> auto_cf(DistLoss::Create(target_d, w));
        std::unique_ptr<ceres::CostFunction> analytic_cf_owner(CreateDistLossAnalytic(target_d, w));
        auto& analytic_cf = *static_cast<DistLossAnalytic*>(analytic_cf_owner.get());

        double res_auto[1], res_analytic[1];
        double jac_a_auto[3], jac_b_auto[3], jac_a_analytic[3], jac_b_analytic[3];
        double* jacs_auto[2] = {jac_a_auto, jac_b_auto};
        double* jacs_analytic[2] = {jac_a_analytic, jac_b_analytic};
        const double* params[2] = {a, b};

        auto_cf->Evaluate(params, res_auto, jacs_auto);
        analytic_cf.Evaluate(params, res_analytic, jacs_analytic);

        max_res_relerr = std::max(max_res_relerr, relerr(res_auto[0], res_analytic[0]));
        for (int i = 0; i < 3; ++i) {
            max_jac_relerr = std::max(max_jac_relerr, relerr(jac_a_auto[i], jac_a_analytic[i]));
            max_jac_relerr = std::max(max_jac_relerr, relerr(jac_b_auto[i], jac_b_analytic[i]));
        }
    }

    CHECK(max_res_relerr < 1e-9);
    CHECK(max_jac_relerr < 1e-9);
}

TEST_CASE("DistLossAnalytic and DistLoss agree on branch selection exactly at the target-distance kink")
{
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> coord(-50.0, 50.0);
    std::uniform_real_distribution<double> distd(0.1, 20.0);
    std::uniform_real_distribution<double> wd(0.1, 5.0);

    for (int trial = 0; trial < 2000; ++trial) {
        double a[3] = {coord(rng), coord(rng), coord(rng)};
        double target_d = distd(rng);
        double w = wd(rng);
        double dir[3] = {coord(rng), coord(rng), coord(rng)};
        double n = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]) + 1e-9;
        double b[3];
        for (int i = 0; i < 3; ++i) b[i] = a[i] + target_d / n * dir[i];

        std::unique_ptr<ceres::CostFunction> auto_cf(DistLoss::Create(target_d, w));
        std::unique_ptr<ceres::CostFunction> analytic_cf_owner(CreateDistLossAnalytic(target_d, w));
        auto& analytic_cf = *static_cast<DistLossAnalytic*>(analytic_cf_owner.get());
        double res_auto[1], res_analytic[1];
        const double* params[2] = {a, b};
        auto_cf->Evaluate(params, res_auto, nullptr);
        analytic_cf.Evaluate(params, res_analytic, nullptr);

        CHECK(std::fabs(res_auto[0] - res_analytic[0]) < 1e-9);
    }
}

TEST_CASE("DistLossAnalytic reproduces the invalid-corner sentinel short-circuit")
{
    double sentinel[3] = {-1.0, -1.0, -1.0};
    double other[3] = {1.0, 2.0, 3.0};
    DistLossAnalytic cf(5.0, 1.0);

    double res[1];
    double jac_a[3], jac_b[3];
    double* jacs[2] = {jac_a, jac_b};
    const double* params[2] = {sentinel, other};

    cf.Evaluate(params, res, jacs);
    CHECK(res[0] == doctest::Approx(0.0));
    CHECK(jac_a[0] == doctest::Approx(0.0));
    CHECK(jac_a[1] == doctest::Approx(0.0));
    CHECK(jac_a[2] == doctest::Approx(0.0));
    CHECK(jac_b[0] == doctest::Approx(0.0));
    CHECK(jac_b[1] == doctest::Approx(0.0));
    CHECK(jac_b[2] == doctest::Approx(0.0));
}

TEST_CASE("DistLossAnalytic Evaluate() is substantially faster than a raw AutoDiffCostFunction<DistLoss>")
{
    std::unique_ptr<ceres::CostFunction> auto_cf(
        new ceres::AutoDiffCostFunction<DistLoss, 1, 3, 3>(new DistLoss(5.0f, 1.0f)));
    std::unique_ptr<ceres::CostFunction> analytic_cf(new DistLossAnalytic(5.0, 1.0));

    std::mt19937 rng(2024);
    std::uniform_real_distribution<double> coord(-10.0, 10.0);
    const int n_inputs = 4096;
    std::vector<std::array<double, 6>> inputs(n_inputs);
    for (auto& in : inputs) for (double& c : in) c = coord(rng);

    volatile double sink_res = 0.0;
    volatile double sink_jac = 0.0;

    const int n_calls = 500000;
    const int trials  = 7;

    auto time_call = [&](ceres::CostFunction* cf) {
        double best_ms = std::numeric_limits<double>::infinity();
        for (int t = 0; t < trials; ++t) {
            double local_res_sum = 0.0;
            double local_jac_sum = 0.0;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < n_calls; ++i) {
                auto& in = inputs[i & (n_inputs - 1)];
                double a[3] = { in[0], in[1], in[2] };
                double b[3] = { in[3], in[4], in[5] };
                a[0] += local_res_sum * 1e-30;
                const double* params[2] = { a, b };
                double res[1];
                double ja[3], jb[3];
                double* jacs[2] = { ja, jb };
                cf->Evaluate(params, res, jacs);
                local_res_sum += res[0];
                local_jac_sum += ja[0] + jb[0];
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            sink_res = local_res_sum;
            sink_jac = local_jac_sum;
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            best_ms = std::min(best_ms, ms);
        }
        return best_ms;
    };

    time_call(auto_cf.get());
    time_call(analytic_cf.get());

    double ms_auto = time_call(auto_cf.get());
    double ms_analytic = time_call(analytic_cf.get());

    MESSAGE("AutoDiffCostFunction: " << ms_auto << " ms  ("
            << (ms_auto * 1e6 / n_calls) << " ns/call)  |  "
            << "SizedCostFunction: " << ms_analytic << " ms  ("
            << (ms_analytic * 1e6 / n_calls) << " ns/call)  |  "
            << "speedup: " << ms_auto / ms_analytic << "x");

    CHECK(ms_analytic * 1.5 < ms_auto);
}