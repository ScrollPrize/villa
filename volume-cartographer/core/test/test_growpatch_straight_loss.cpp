// Every straight triple of a fully valid plane must own one loss_status slot, in any visiting order.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/GrowPatchStraightLoss.hpp"

#include <opencv2/core/mat.hpp>

#include <algorithm>
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
