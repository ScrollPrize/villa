#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "vc/core/util/Geometry.hpp"
#ifndef COST_HEADER
#define COST_HEADER "vc/tracer/CostFunctions.hpp"
#endif
#include COST_HEADER

namespace fs = std::filesystem;
using Quad = std::array<std::array<double, 3>, 4>;
using Values = std::array<double, 13>;
using N = NormalConstraintPlane;

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static Quad quad(double z, double dx = 0) {
    return {{{100 + dx, 100, z}, {112 + dx, 100, z + 1},
             {112 + dx, 100, z - 1}, {112 + dx, 110, z + 1}}};
}

struct Evaluator {
    std::unique_ptr<N> direct;
    std::unique_ptr<ceres::CostFunction> autodiff;
    Evaluator(const vc::core::util::NormalGridVolume& volume, double wn,
              double ws, bool invert, bool ad) {
        if (ad) autodiff.reset(N::Create(volume, 0, wn, ws, nullptr, false, -1, -1, invert));
        else direct = std::make_unique<N>(volume, 0, wn, ws, nullptr, false, -1, -1, invert);
    }
    Values operator()(const Quad& q) {
        const Quad before = q;
        Values result{};
        if (autodiff) {
            const double* parameters[] = {q[0].data(), q[1].data(), q[2].data(), q[3].data()};
            double* jacobians[] = {result.data() + 1, result.data() + 4,
                                   result.data() + 7, result.data() + 10};
            require(autodiff->Evaluate(parameters, result.data(), jacobians), "Evaluate failed");
        } else {
            require(direct->operator()<double>(q[0].data(), q[1].data(), q[2].data(), q[3].data(), result.data()), "direct failed");
        }
        require(q == before, "parameters changed");
        for (double value : result) require(std::isfinite(value), "nonfinite value");
        std::cout << "evaluation," << bool(autodiff);
        for (double value : result) std::cout << ',' << value;
        std::cout << '\n';
        return result;
    }
};

int vc_normal_cache_fixture_main(int argc, char** argv) {
    if (argc != 2) return 2;
    try {
        const fs::path root = argv[1];
        require(!fs::exists(root), "output directory already exists");
        fs::create_directories(root / "xy");
        fs::create_directories(root / "xz");
        fs::create_directories(root / "yz");
        {
            std::ofstream metadata(root / "metadata.json");
            metadata << "{\"sparse-volume\":1,\"spiral-step\":20.0}\n";
            metadata.close();
            require(bool(metadata), "metadata write failed");
            vc::core::util::GridStore a(cv::Rect(0, 0, 256, 256), 16);
            a.add({{70, 105}, {112, 105}, {160, 105}});
            a.save((root / "xy" / "000010.grid").string());
            vc::core::util::GridStore b(cv::Rect(0, 0, 256, 256), 16);
            b.add({{100, 70}, {100, 100}, {100, 160}});
            b.save((root / "xy" / "000030.grid").string());
        }
        vc::core::util::NormalGridVolume volume(root.string(), 0);
        require(volume.coordinateScale() == 1.0 && volume.level() == 0, "wrong volume scale");
        {
            auto a = volume.query_nearest({100, 100, 10}, 0);
            auto b = volume.query_nearest({100, 100, 30}, 0);
            require(a && b && a.get() != b.get(), "distinct selected stores required");
        }
        int controls_failed = 0, histories_failed = 0, comparisons = 0;
        auto compare = [&](const char* label, const Values& actual, const Values& expected,
                           bool ad, bool control) {
            double residual_error = std::abs(actual[0] - expected[0]), jacobian_error = 0;
            if (ad) for (int k = 1; k < 13; ++k)
                jacobian_error = std::max(jacobian_error, std::abs(actual[k] - expected[k]));
            bool pass = residual_error <= 1e-12 && jacobian_error <= 1e-10;
            ++comparisons;
            if (!pass) { if (control) ++controls_failed; else ++histories_failed; }
            std::cout << label << ',' << control << ',' << pass << ',' << actual[0]
                      << ',' << expected[0] << ',' << residual_error << ',' << jacobian_error << '\n';
        };
        std::cout << std::setprecision(17);
        for (bool ad : {false, true}) for (bool invert : {false, true}) {
            for (const auto weights : {std::array<double, 2>{1, 0},
                                      std::array<double, 2>{1, 0.1},
                                      std::array<double, 2>{0, 0.1}}) {
                std::cout << "configuration," << ad << ',' << invert << ',' << weights[0] << ',' << weights[1] << '\n';
                auto make = [&]() { return Evaluator(volume, weights[0], weights[1], invert, ad); };
                auto cold_a = make(), cold_b = make(), cold_bx = make();
                const Values a = cold_a(quad(10)), b = cold_b(quad(30)), bx = cold_bx(quad(30, 20));
                require(std::abs(a[0] - b[0]) > 1e-7, "fixture lacks distinct cold residuals");
                compare("repeat-A", cold_a(quad(10)), a, ad, true);
                compare("repeat-B", cold_b(quad(30)), b, ad, true);
                auto aba = make();
                compare("ABA-A", aba(quad(10)), a, ad, true);
                compare("ABA-B", aba(quad(30)), b, ad, false);
                compare("ABA-return-A", aba(quad(10)), a, ad, false);
                auto bab = make();
                compare("BAB-B", bab(quad(30)), b, ad, true);
                compare("BAB-A", bab(quad(10)), a, ad, false);
                compare("BAB-return-B", bab(quad(30)), b, ad, false);
                auto motion = make();
                (void)motion(quad(10));
                compare("projected-motion", motion(quad(30, 20)), bx, ad, true);
            }
        }
        std::cout << "summary," << comparisons << ',' << controls_failed << ',' << histories_failed << '\n';
        return controls_failed || histories_failed ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "fixture/control error: " << error.what() << '\n';
        return 2;
    }
}
