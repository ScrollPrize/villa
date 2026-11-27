// Every straight triple of a fully valid plane must own one loss_status slot, in any visiting order.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../include/vc/core/util/OMPThreadPointCollection.hpp"
#include "../src/GrowPatchStraightLoss.hpp"

#include <opencv2/core/mat.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <omp.h>
#include <thread>
#include <set>
#include <vector>
#include <tuple>

static int claimed_triples(bool reverse)
{
    const int n = 24;
    cv::Mat_<uint16_t> status(n, n, uint16_t(0));
    std::vector<cv::Vec2i> pts;
    for (int y = 2; y < n - 2; ++y)
        for (int x = 2; x < n - 2; ++x)
            pts.push_back({y, x});
    if (reverse)
        std::reverse(pts.begin(), pts.end());
    int claimed = 0;
    for (const cv::Vec2i &p : pts)
        for (const StraightLossSpec &t : kStraightLosses) {
            const cv::Vec2i key = straight_loss_key(p, t.o2);
            if (!(status(key) & (1 << t.bit))) {
                status(key) |= (1 << t.bit);
                ++claimed;
            }
        }
    return claimed;
}

TEST_CASE("straight loss slots: one slot per triple, independent of order")
{
    std::set<std::tuple<int, int, int, int>> triples;  // centre and direction
    for (int y = 2; y < 22; ++y)
        for (int x = 2; x < 22; ++x)
            for (const StraightLossSpec &t : kStraightLosses) {
                const cv::Vec2i c = cv::Vec2i(y, x) + t.o2, d = t.o3 - t.o1;
                triples.insert({c[0], c[1], d[0], d[1]});
            }
    CHECK(triples.size() == 1836);
    CHECK(claimed_triples(false) == int(triples.size()));
    CHECK(claimed_triples(true) == int(triples.size()));
}

// The vertices a solve moves (within radius of its centre), and the ones it reads: those and every vertex within kLossReach
// cells of them. Two solves conflict when one reads or moves a vertex the other moves.
static bool conflict_by_sets(const cv::Vec2i &a, const cv::Vec2i &b, int radius)
{
    std::set<std::pair<int, int>> moved_a, moved_b, read_a, read_b;
    for (int y = -radius; y <= radius; ++y)
        for (int x = -radius; x <= radius; ++x)
            if (y * y + x * x <= radius * radius) {
                moved_a.insert({a[0] + y, a[1] + x});
                moved_b.insert({b[0] + y, b[1] + x});
                for (int ry = -kLossReach; ry <= kLossReach; ++ry)
                    for (int rx = -kLossReach; rx <= kLossReach; ++rx) {
                        read_a.insert({a[0] + y + ry, a[1] + x + rx});
                        read_b.insert({b[0] + y + ry, b[1] + x + rx});
                    }
            }
    for (const auto &q : moved_b)
        if (read_a.count(q))
            return true;
    for (const auto &q : moved_a)
        if (read_b.count(q))
            return true;
    return false;
}

TEST_CASE("local solves: the conflict table equals the sets of moved and read vertices")
{
    for (int radius : {1, 2, 3}) {
        const LocalSolveConflicts conflicts(radius);
        const cv::Vec2i a(30, 30);
        for (int dy = -14; dy <= 14; ++dy)
            for (int dx = -14; dx <= 14; ++dx) {
                const cv::Vec2i b(30 + dy, 30 + dx);
                CHECK(conflicts(a, b) == conflict_by_sets(a, b, radius));
                CHECK(conflicts(a, b) == conflicts(b, a));
            }
    }
}

TEST_CASE("local solves: pairs the distance rule let through while one reads what the other moves")
{
    const LocalSolveConflicts conflicts(3);
    // centres at least 7 apart (the old rule, 2 * radius + 1) that still conflict
    const std::vector<std::pair<cv::Vec2i, cv::Vec2i>> pairs = {
        {{10, 10}, {10, 17}}, {{10, 10}, {10, 18}}, {{10, 8}, {15, 14}}, {{10, 10}, {16, 16}}};
    for (const auto &[a, b] : pairs) {
        CHECK(cv::norm(a - b) >= 7);
        CHECK(conflicts(a, b));
    }
    CHECK_FALSE(conflicts({10, 10}, {10, 19}));
    CHECK_FALSE(conflicts({10, 10}, {19, 10}));
}

TEST_CASE("local solves: candidates taken together never conflict")
{
    const LocalSolveConflicts conflicts(3);
    std::vector<cv::Vec2i> cands;
    for (int y = 4; y < 60; ++y)
        for (int x = 4; x < 60; ++x)
            cands.push_back({y, x});
    std::vector<cv::Vec2i> in_flight;
    int idx = 7;
    while (true) {
        const cv::Point2i p = extract_point_without_conflict(cands, in_flight, idx, conflicts);
        if (p.x == -1)
            break;
        for (const cv::Vec2i &q : in_flight)
            CHECK_FALSE(conflicts(cv::Vec2i(p.x, p.y), q));
        in_flight.push_back({p.x, p.y});
    }
    // at least one far apart batch exists, and a far candidate remains available after the first is taken
    CHECK(in_flight.size() >= 20);
}

TEST_CASE("local solves: threads taking candidates together are never in conflict")
{
    const LocalSolveConflicts conflicts(3);
    std::vector<cv::Vec2i> cands;
    for (int y = 4; y < 70; ++y)
        for (int x = 4; x < 70; ++x)
            cands.push_back({y, x});
    omp_set_num_threads(8);
    OmpThreadPointCol col([&conflicts](const cv::Vec2i &a, const cv::Vec2i &b) { return conflicts(a, b); }, cands);
    std::mutex mutex;
    std::vector<cv::Vec2i> in_flight;
    std::atomic<int> taken{0}, violations{0};
#pragma omp parallel
    {
        while (true) {
            const cv::Point2i p = col.next();
            if (p.x == -1)
                break;
            {
                std::lock_guard lock(mutex);
                for (const cv::Vec2i &q : in_flight)
                    if (conflicts(cv::Vec2i(p.x, p.y), q))
                        ++violations;
                in_flight.push_back({p.x, p.y});
            }
            ++taken;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            {
                std::lock_guard lock(mutex);
                in_flight.erase(std::find(in_flight.begin(), in_flight.end(), cv::Vec2i(p.x, p.y)));
            }
        }
    }
    CHECK(violations == 0);
    CHECK(taken > 100);
}
