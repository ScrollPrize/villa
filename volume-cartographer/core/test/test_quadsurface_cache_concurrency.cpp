#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"
#include "omp.h"

#include <opencv2/core.hpp>

#include <atomic>
#include <barrier>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <thread>

namespace {

bool sameBytes(const cv::Mat& a, const cv::Mat& b)
{
    if (a.size() != b.size() || a.type() != b.type()) return false;
    const auto bytes = static_cast<std::size_t>(a.cols) * a.elemSize();
    for (int row = 0; row < a.rows; ++row) {
        if (std::memcmp(a.ptr(row), b.ptr(row), bytes) != 0) return false;
    }
    return true;
}

void checkConcurrentEviction(QuadSurface& surface, bool requestNormals, float depth)
{
    omp_set_dynamic(0);
    omp_set_num_threads(2);
    const cv::Size size(192, 192);
    const auto points = surface.rawPoints();
    const cv::Vec3f offset(-0.5f * points.cols, -0.5f * points.rows, depth);
    const float scale = surface.scale()[0];
    const cv::Mat mask = surface.validMask().clone();
    cv::Mat_<cv::Vec3f> coords, normals;
    surface.gen(&coords, requestNormals ? &normals : nullptr, size,
                {0, 0, 0}, scale, offset);
    // gen() returns TLS scratch views: the oracle must own separate storage.
    const cv::Mat expectedCoords = coords.clone();
    const cv::Mat expectedNormals = normals.clone();
    CHECK(cv::countNonZero(mask) > 0);
    bool hasFiniteCoords = false;
    for (int row = 0; row < coords.rows; ++row) {
        for (int col = 0; col < coords.cols; ++col) {
            hasFiniteCoords |= std::isfinite(coords(row, col)[0]);
        }
    }
    REQUIRE(hasFiniteCoords);
    // The view starts at grid (0,0), covering both components and interior holes.
    for (int row = 0; row < mask.rows && row < coords.rows; ++row) {
        for (int col = 0; col < mask.cols && col < coords.cols; ++col) {
            if (mask.at<uint8_t>(row, col) == 0) {
                CHECK_FALSE(std::isfinite(coords(row, col)[0]));
            }
        }
    }

    std::barrier start(3);
    std::atomic<int> finished{0};
    std::atomic<int> evictions{0};
    std::atomic<bool> failed{false};
    auto render = [&] {
        omp_set_num_threads(2);
        start.arrive_and_wait();
        try {
            for (int i = 0; i < 48; ++i) {
                const cv::Mat retainedMask = surface.validMask();
                cv::Mat_<cv::Vec3f> actualCoords, actualNormals;
                surface.gen(&actualCoords, requestNormals ? &actualNormals : nullptr,
                            size, {0, 0, 0}, scale, offset);
                // Compare NaN payloads as well as finite output, with no tolerance.
                if (!sameBytes(actualCoords, expectedCoords) ||
                    (requestNormals && !sameBytes(actualNormals, expectedNormals)) ||
                    !sameBytes(retainedMask, mask)) {
                    failed.store(true);
                }
            }
        } catch (...) {
            failed.store(true);
        }
        ++finished;
    };
    std::jthread first(render);
    std::jthread second(render);
    std::jthread evict([&] {
        start.arrive_and_wait();
        do {
            surface.unloadCaches();
            ++evictions;
            std::this_thread::yield();
        } while (finished.load() != 2);
    });
    first.join();
    second.join();
    evict.join();
    CHECK(evictions.load() > 0);
    CHECK_FALSE(failed.load());

    surface.unloadCaches();
    CHECK(sameBytes(surface.validMask(), mask));
}

cv::Mat_<cv::Vec3f> grid()
{
    cv::Mat_<cv::Vec3f> points(64, 64);
    for (int row = 0; row < points.rows; ++row) {
        for (int col = 0; col < points.cols; ++col) {
            points(row, col) = {float(col), float(row),
                                10.0f + std::sin(float(col) * 0.1f)};
        }
    }
    return points;
}

} // namespace

TEST_CASE("derived-cache eviction preserves concurrent gen output")
{
    const auto points = grid();
    QuadSurface allValid(points, {1, 1});
    allValid.setComponents({{0, 32}, {32, 64}});
    checkConcurrentEviction(allValid, true, 1.5f);
    checkConcurrentEviction(allValid, false, 0.0f);

    auto invalidPoints = points.clone();
    invalidPoints(30, 30) = {-1, -1, -1};
    QuadSurface invalid(invalidPoints, {1, 1});
    invalid.setStrictQuadRenderValidity(true);
    checkConcurrentEviction(invalid, true, 0.0f);
    checkConcurrentEviction(invalid, false, 1.5f);
}

TEST_CASE("PHerc0172 fixture survives derived-cache eviction while rendering")
{
    const auto path = std::filesystem::path(VC_TEST_FIXTURES_DIR) /
                      "segments" / "20241113090990";
    REQUIRE(std::filesystem::exists(path / "meta.json"));
    QuadSurface surface(path);
    surface.ensureLoaded();
    checkConcurrentEviction(surface, true, 1.5f);
}
