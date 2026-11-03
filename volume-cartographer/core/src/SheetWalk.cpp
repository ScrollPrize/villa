#include "vc/core/util/SheetWalk.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <mutex>
#include <thread>

namespace vc::sheet {
namespace {
namespace Walk = vc::sheet;
using Vec = cv::Vec3d;
struct WalkState { Vec x, v; };
WalkState add(const WalkState& a, const WalkState& b, double h)
{
    return {a.x + h * b.x, a.v + h * b.v};
}
std::optional<Vec> walkNormal(const Walk::SheetNormalFn& normal, const Vec& x)
{
    if (!normal) return std::nullopt;
    const auto n = normal(cv::Vec3f(x));
    if (!n) return std::nullopt;
    const double length = cv::norm(*n);
    if (!std::isfinite(length) || length <= 0.5) return std::nullopt;
    return Vec(*n) / length;
}
Vec tangent(const Vec& v, const Vec& n)
{
    const Vec t = v - n * v.dot(n);
    const double length = cv::norm(t);
    return length > 1e-3 * cv::norm(v) ? t / length : v / cv::norm(v);
}
// Minimum-turn transport: x' = v, v' = -n (v . Dn v).
// Central differences require only unsigned normals, not tensor derivatives.
// Align signs before differencing; reversing an eigenvector must not steer us.
std::optional<WalkState> derivative(const WalkState& s, const Walk::SheetNormalFn& normal,
                                    const Vec* startNormal = nullptr)
{
    const double speed = cv::norm(s.v);
    if (!std::isfinite(speed) || speed < 1e-12) return std::nullopt;
    const Vec direction = s.v / speed;
    constexpr double epsilon = 0.25; // volume voxels; derivative accuracy floor
    auto n = startNormal ? std::optional<Vec>(*startNormal) : walkNormal(normal, s.x);
    auto plus = walkNormal(normal, s.x + epsilon * direction);
    auto minus = walkNormal(normal, s.x - epsilon * direction);
    if (!n || !plus || !minus) return std::nullopt;
    if (plus->dot(*n) < 0) *plus = -*plus;
    if (minus->dot(*n) < 0) *minus = -*minus;
    const Vec dn = (*plus - *minus) * (speed / (2 * epsilon));
    return WalkState{s.v, -*n * s.v.dot(dn)};
}
std::optional<WalkState> integrate(WalkState s, double h, Walk::SheetIntegrator method,
                                   const Walk::SheetNormalFn& normal,
                                   std::optional<Vec>& endpointNormal)
{
    // A successful preceding substep evaluated this exact state.x already.
    const auto n = endpointNormal ? endpointNormal : walkNormal(normal, s.x);
    // Any failed stage falls back to projection, so its endpoint cannot be reused.
    endpointNormal.reset();
    if (!n) return std::nullopt;
    s.v = tangent(s.v, *n);
    const auto k1 = derivative(s, normal, &*n);
    if (!k1) return std::nullopt;
    WalkState next;
    if (method == Walk::SheetIntegrator::Heun) {
        const auto k2 = derivative(add(s, *k1, h), normal);
        if (!k2) return std::nullopt;
        next = add(add(s, *k1, h / 2), *k2, h / 2);
    } else {
        const auto k2 = derivative(add(s, *k1, h / 2), normal);
        if (!k2) return std::nullopt;
        if (method == Walk::SheetIntegrator::RK4) {
            const auto k3 = derivative(add(s, *k2, h / 2), normal);
            if (!k3) return std::nullopt;
            const auto k4 = derivative(add(s, *k3, h), normal);
            if (!k4) return std::nullopt;
            next = add(add(add(add(s, *k1, h / 6), *k2, h / 3), *k3, h / 3), *k4, h / 6);
        } else {
            next = add(s, *k2, h);
        }
    }
    const auto endNormal = walkNormal(normal, next.x);
    if (!endNormal || !std::isfinite(cv::norm(next.v)) || cv::norm(next.v) < 1e-12)
        return std::nullopt;
    next.v = tangent(next.v, *endNormal);
    endpointNormal = endNormal;
    return next;
}
void walkSheetSubsteps(const Walk::SheetWalk& walk, std::size_t line,
                       const Walk::SheetNormalFn& normal, std::vector<cv::Vec3f>& guess,
                       const SheetWalkProgress& vertexCompleted)
{
    const double length = cv::norm(walk.steps[line]);
    WalkState state{Vec(walk.origins[line]), Vec(walk.steps[line])};
    if (!std::isfinite(length) || length <= 0) {
        for (int k = 0; k < walk.cells; ++k) {
            guess[line * walk.cells + k] = walk.origins[line];
            if (vertexCompleted) vertexCompleted();
        }
        return;
    }
    state.v /= length;
    const double maxStep = std::isfinite(walk.options.maxStep) && walk.options.maxStep > 0
        ? std::max(0.01, walk.options.maxStep) : length;
    const double count = std::max(1.0, std::ceil(length / maxStep));
    const double h = length / count;
    std::optional<Vec> endpointNormal;
    for (int k = 0; k < walk.cells; ++k) {
        for (double j = 0; j < count; ++j) {
            if (walk.options.mode == Walk::SheetWalkMode::Streamline) {
                if (auto next = integrate(state, h, walk.options.integrator, normal, endpointNormal)) {
                    state = *next;
                    continue;
                }
            }
            // Projection mode, also the fallback if any integration stage
            // lacks a normal. Missing midpoint normals continue straight.
            if (auto n = walkNormal(normal, state.x + state.v * (h / 2)))
                state.v = tangent(state.v, *n);
            state.x += h * state.v;
        }
        guess[line * walk.cells + k] = cv::Vec3f(state.x);
        if (vertexCompleted) vertexCompleted();
    }
}
} // namespace

void walkSheetLine(const SheetWalk& walk, std::size_t line,
                   const SheetNormalFn& normal, std::vector<cv::Vec3f>& guess)
{
    walkSheetLine(walk, line, normal, guess, {});
}

void walkSheetLine(const SheetWalk& walk,
                                  std::size_t line,
                                  const SheetNormalFn& normal,
                                  std::vector<cv::Vec3f>& guess,
                                  const SheetWalkProgress& vertexCompleted)
{
    if (line >= walk.origins.size() || line >= walk.steps.size() ||
        guess.size() < (line + 1) * static_cast<std::size_t>(walk.cells)) {
        return;
    }
    if (walk.options.maxStep > 0 || walk.options.mode == SheetWalkMode::Streamline) {
        walkSheetSubsteps(walk, line, normal, guess, vertexCompleted);
        return;
    }
    cv::Vec3f point = walk.origins[line];
    cv::Vec3f step = walk.steps[line];
    const float length = static_cast<float>(cv::norm(step));
    for (int k = 0; k < walk.cells; ++k) {
        if (normal && length > 0.0f) {
            if (const auto n = normal(point + step * 0.5f)) {
                const float nLength = static_cast<float>(cv::norm(*n));
                if (nLength > 0.5f) {
                    const cv::Vec3f unit = *n / nLength;
                    const cv::Vec3f along = step - unit * step.dot(unit);
                    const float alongLength = static_cast<float>(cv::norm(along));
                    if (alongLength > 1e-3f * length) {
                        step = along * (length / alongLength);
                    }
                }
            }
        }
        point += step;
        guess[line * static_cast<std::size_t>(walk.cells) + static_cast<std::size_t>(k)] = point;
        if (vertexCompleted) vertexCompleted();
    }
}

void walkSheetLines(const SheetWalk& walk, const SheetNormalFn& normal,
                    std::vector<cv::Vec3f>& guess, unsigned workers)
{
    walkSheetLines(walk, normal, guess, workers, {});
}

void walkSheetLines(const SheetWalk& walk,
                                  const SheetNormalFn& normal,
                                  std::vector<cv::Vec3f>& guess,
                                  unsigned workers,
                                  const SheetWalkProgress& vertexCompleted)
{
    if (walk.origins.empty()) return;
    const auto count = std::min<std::size_t>(std::max(1u, workers), walk.origins.size());
    std::atomic<std::size_t> next{0};
    std::exception_ptr failure;
    std::mutex failureMutex;
    std::atomic<bool> failed{false};
    auto work = [&]() {
        try {
            for (std::size_t line = next++; line < walk.origins.size() && !failed; line = next++) {
                walkSheetLine(walk, line, normal, guess, vertexCompleted);
            }
        } catch (...) {
            std::lock_guard lock(failureMutex);
            if (!failure) failure = std::current_exception();
            failed = true;
        }
    };
    // jthread also joins already-started workers if creating another fails.
    std::vector<std::jthread> threads;
    for (std::size_t t = 1; t < count; ++t) threads.emplace_back(work);
    work();
    for (auto& thread : threads) thread.join();
    if (failure) std::rethrow_exception(failure);
}

} // namespace vc::sheet
