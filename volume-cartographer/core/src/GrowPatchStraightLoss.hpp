#pragma once

#include <opencv2/core/matx.hpp>

#include <array>

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
