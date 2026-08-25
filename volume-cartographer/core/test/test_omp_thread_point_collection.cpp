#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/OMPThreadPointCollection.hpp"

#include <atomic>
#include <cstdlib>
#include <unordered_set>
#include <vector>

namespace {

long long point_key(const cv::Vec2i& point)
{
    return (static_cast<long long>(point[1]) << 32) |
           static_cast<unsigned int>(point[0]);
}

long long squared_distance(const cv::Vec2i& lhs, const cv::Vec2i& rhs)
{
    const long long dx = lhs[0] - rhs[0];
    const long long dy = lhs[1] - rhs[1];
    return dx * dx + dy * dy;
}

}  // namespace

// This is a production-class concurrency contract/stress test. It exercises
// OmpThreadPointCol directly, but is not expected to fail reliably on the
// unfixed implementation; the race is schedule-dependent and is demonstrated
// separately by the paired sanitizer reproducer documented in the PR.
TEST_CASE("OmpThreadPointCol concurrent retrieval preserves the point contract")
{
    constexpr int workerCount = 4;
    constexpr int side = 64;
    constexpr int rounds = 16;
    constexpr int maxWaves = side * side;
    constexpr float minimumDistance = 8.0f;
    constexpr long long minimumSquaredDistance = 64;

    omp_set_dynamic(0);
    omp_set_num_threads(workerCount);
    std::srand(0);

    std::vector<cv::Vec2i> candidates;
    candidates.reserve(side * side);
    for (int y = 0; y < side; ++y) {
        for (int x = 0; x < side; ++x) {
            candidates.emplace_back(x, y);
        }
    }

    for (int round = 0; round < rounds; ++round) {
        OmpThreadPointCol collection(minimumDistance, candidates);
        std::vector<cv::Vec2i> published(workerCount, {-1, -1});
        std::vector<std::vector<cv::Vec2i>> retrieved(workerCount);
        std::atomic<bool> finished{false};

#pragma omp parallel num_threads(workerCount)
        {
            const int threadId = omp_get_thread_num();
            for (int wave = 0; wave < maxWaves; ++wave) {
#pragma omp barrier
                published[threadId] = collection.next();
#pragma omp barrier

#pragma omp single
                {
                    bool anyPublished = false;
                    for (int lhs = 0; lhs < workerCount; ++lhs) {
                        if (published[lhs][0] == -1) {
                            continue;
                        }
                        anyPublished = true;
                        for (int rhs = lhs + 1; rhs < workerCount; ++rhs) {
                            if (published[rhs][0] == -1) {
                                continue;
                            }
                            CHECK(squared_distance(published[lhs], published[rhs]) >=
                                  minimumSquaredDistance);
                        }
                    }
                    finished.store(!anyPublished, std::memory_order_relaxed);
                }
#pragma omp barrier

                if (published[threadId][0] != -1) {
                    retrieved[threadId].push_back(published[threadId]);
                }
                if (finished.load(std::memory_order_relaxed)) {
                    break;
                }
            }
        }

        std::unordered_set<long long> unique;
        for (const auto& workerPoints : retrieved) {
            for (const auto& point : workerPoints) {
                unique.insert(point_key(point));
            }
        }

        CHECK_FALSE(unique.empty());
    }
}
