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
    explicit LocalSolveConflicts(int radius) : radius_(radius), span_(2 * radius + kLossReach), half_width_(2 * span_ + 1, -1)
    {
        // Rows y1 and y2 of the two discs come within reach where |dy - (y1 - y2)| and |dx| - both half-widths are <= kLossReach.
        std::vector<int> row_half_width(2 * radius + 1, 0);
        for (int y = -radius; y <= radius; ++y)
            while ((row_half_width[y + radius] + 1) * (row_half_width[y + radius] + 1) + y * y <= radius * radius)
                ++row_half_width[y + radius];
        for (int y1 = -radius; y1 <= radius; ++y1)
            for (int y2 = -radius; y2 <= radius; ++y2)
                for (int r = -kLossReach; r <= kLossReach; ++r) {
                    int &w = half_width_[y1 - y2 + r + span_];
                    w = std::max(w, row_half_width[y1 + radius] + row_half_width[y2 + radius] + kLossReach);
                }
    }
    bool operator()(const cv::Vec2i &a, const cv::Vec2i &b) const
    {
        const int dy = b[0] - a[0], dx = b[1] - a[1];
        return std::abs(dy) <= span_ && std::abs(dx) <= half_width_[dy + span_];
    }
    int radius() const { return radius_; }

private:
    int radius_, span_;
    std::vector<int> half_width_;
};
