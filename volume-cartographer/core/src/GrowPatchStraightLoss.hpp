#pragma once

#include <opencv2/core/matx.hpp>

#include <algorithm>
#include <array>
#include <vector>

// The twelve straight-loss triples of add_missing_losses and the status slot
// (loss_status pixel and bit) that records "this triple has its loss" so that
// neighbouring points do not add it twice. A slot must identify one triple.
struct StraightLossSpec {
    int bit;
    cv::Vec2i o1, o2, o3;
};

inline const std::array<StraightLossSpec, 12> kStraightLosses = {{
    {0, {0,-2}, {0,-1}, {0,0}}, {0, {0,-1}, {0,0}, {0,1}}, {0, {0,0}, {0,1}, {0,2}},
    {1, {-2,0}, {-1,0}, {0,0}}, {1, {-1,0}, {0,0}, {1,0}}, {1, {0,0}, {1,0}, {2,0}},
    {12, {-2,-2}, {-1,-1}, {0,0}}, {12, {-1,-1}, {0,0}, {1,1}}, {12, {0,0}, {1,1}, {2,2}},
    {13, {-2,2}, {-1,1}, {0,0}}, {13, {-1,1}, {0,0}, {1,-1}}, {13, {0,0}, {1,-1}, {2,-2}},
}};

// loss_status pixel that holds the bit of the triple (p+o1, p+o2, p+o3): its centre.
inline cv::Vec2i straight_loss_key(const cv::Vec2i &p, const cv::Vec2i &o2)
{
    return p + o2;
}

// How far from a vertex a solve centred nearby reads: the losses add_missing_losses builds at a moved vertex use
// vertices up to two cells away along a row, a column or a diagonal (the offsets of kStraightLosses).
inline constexpr int kLossReach = 2;

// Whether two local solves of the same radius, centred on `a` and `b`, would touch each other's vertices. A solve moves
// the vertices within `radius` of its centre and reads those within kLossReach cells of them, so two solves conflict
// when a vertex one moves is read (or moved) by the other. Solves that do not conflict can run on two threads at once.
class LocalSolveConflicts {
public:
    explicit LocalSolveConflicts(int radius) : radius_(radius), span_(2 * radius + kLossReach), table_(size_t(2 * span_ + 1) * size_t(2 * span_ + 1), 0)
    {
        std::vector<cv::Vec2i> disc;
        for (int y = -radius; y <= radius; ++y)
            for (int x = -radius; x <= radius; ++x)
                if (y * y + x * x <= radius * radius)
                    disc.push_back({y, x});
        for (int dy = -span_; dy <= span_; ++dy)
            for (int dx = -span_; dx <= span_; ++dx) {
                bool conflict = false;
                for (size_t i = 0; i < disc.size() && !conflict; ++i)
                    for (size_t j = 0; j < disc.size() && !conflict; ++j)
                        conflict = std::max(std::abs(disc[i][0] - (dy + disc[j][0])), std::abs(disc[i][1] - (dx + disc[j][1]))) <= kLossReach;
                table_[index(dy, dx)] = conflict;
            }
    }
    bool operator()(const cv::Vec2i &a, const cv::Vec2i &b) const
    {
        const int dy = b[0] - a[0], dx = b[1] - a[1];
        return std::abs(dy) <= span_ && std::abs(dx) <= span_ && table_[index(dy, dx)];
    }
    int radius() const { return radius_; }

private:
    size_t index(int dy, int dx) const { return size_t(dy + span_) * size_t(2 * span_ + 1) + size_t(dx + span_); }
    int radius_, span_;
    std::vector<char> table_;
};
