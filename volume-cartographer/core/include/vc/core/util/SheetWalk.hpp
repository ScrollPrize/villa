#pragma once

#include <opencv2/core.hpp>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <optional>
#include <thread>
#include <vector>

// Shared CT sheet integration used by manual extensions and GrowPatch seeds.
namespace vc::sheet {
enum class SheetWalkMode { Projection = 0, Streamline = 1 };
enum class SheetIntegrator { Midpoint = 0, Heun = 1, RK4 = 2 };
struct SheetWalkOptions
{
    // Maximum internal arc-length step in volume voxels; 0 uses one
    // step per output vertex. Output mesh spacing is unchanged.
    double maxStep{0.0};
    SheetWalkMode mode{SheetWalkMode::Projection};
    SheetIntegrator integrator{SheetIntegrator::Midpoint};
    bool continuousNormals() const { return maxStep > 0 || mode == SheetWalkMode::Streamline; }
};
using SheetNormalFn = std::function<std::optional<cv::Vec3f>(const cv::Vec3f& world)>;
struct SheetWalk
{
    std::uint64_t extensionId{0};
    int cells{0};
    std::vector<cv::Vec3f> origins;
    std::vector<cv::Vec3f> steps;
    SheetWalkOptions options;
};

void walkSheetLine(const SheetWalk& walk, std::size_t line,
                   const SheetNormalFn& normal, std::vector<cv::Vec3f>& guess);
void walkSheetLines(const SheetWalk& walk, const SheetNormalFn& normal,
                    std::vector<cv::Vec3f>& guess, unsigned workers);
// Called after each output vertex, including on worker threads. Observers
// synchronize their own state and must not modify the walk. Keep the original
// overloads for existing Manual Grow callers (including shared-library builds).
using SheetWalkProgress = std::function<void()>;
void walkSheetLine(const SheetWalk& walk, std::size_t line,
                   const SheetNormalFn& normal, std::vector<cv::Vec3f>& guess,
                   const SheetWalkProgress& vertexCompleted);
void walkSheetLines(const SheetWalk& walk, const SheetNormalFn& normal,
                    std::vector<cv::Vec3f>& guess, unsigned workers,
                    const SheetWalkProgress& vertexCompleted);
// Grow one ordered edge as a front. After each output step, redistribute
// vertices at the source edge's spacing and choose tangents from the new edge.
// Origins must be ordered along one side, all growing in the same direction.
void walkSheetStrip(const SheetWalk& walk, const SheetNormalFn& normal,
                    std::vector<cv::Vec3f>& guess, unsigned workers,
                    const SheetWalkProgress& vertexCompleted = {});
// Runs body(i) for every i in [0, n) on up to `workers` threads. Each i must
// write only its own outputs, so the results do not depend on the schedule.
// If any body throws, the exception of the lowest such i is rethrown: the
// one a plain loop would have stopped at.
template <class Body>
void forEachIndex(std::size_t n, unsigned workers, const Body& body)
{
    const auto count = std::min<std::size_t>(std::max(1u, workers), n);
    if (count <= 1) {
        for (std::size_t i = 0; i < n; ++i) body(i);
        return;
    }
    std::vector<std::exception_ptr> errors(n);
    std::atomic<std::size_t> next{0};
    auto work = [&]() {
        for (std::size_t i = next++; i < n; i = next++) {
            try {
                body(i);
            } catch (...) {
                errors[i] = std::current_exception();
            }
        }
    };
    {
        // jthread also joins already-started workers if creating another fails.
        std::vector<std::jthread> threads;
        for (std::size_t t = 1; t < count; ++t) threads.emplace_back(work);
        work();
    }
    for (const auto& error : errors)
        if (error) std::rethrow_exception(error);
}
} // namespace vc::sheet
