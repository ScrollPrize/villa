#pragma once

#include <opencv2/core.hpp>
#include <cstdint>
#include <functional>
#include <optional>
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
} // namespace vc::sheet
