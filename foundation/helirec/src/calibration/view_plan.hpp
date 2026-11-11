#pragma once
// Which frames and rows of two stretches record the same lines, according to what the record states, and what a
// registration of the two blocks must then return. Pure bookkeeping, no frames: registration.cpp tests it on
// synthetic data.
//
// A trajectory gives, per frame, the stated angle [deg, cumulative] and the stated position along the axis in
// detector rows, counted so that a larger value puts the sample's content on a higher row: content at
// sample-fixed height w appears on detector row w + rows[frame].
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace helirec::calibration {

struct Trajectory {
    std::vector<double> angle_deg, rows;
};

struct BlockPlan {
    std::int64_t p0 = 0, j0 = 0;  // frames p0 .. p0 + n - 1 of A against frames j0 .. j0 + n - 1 of B
    int n = 0;
    std::vector<int> row_offset;  // per frame: B's rows are cut row_offset[q] rows above A's
    double mean_row_offset = 0;
    double stated_dq = 0;  // a(q, ..) is at the angle of b(q + stated_dq, ..): what the registration along frames must return
    double stated_dv = 0;  // what the registration along rows must return
    double stated_D = 0;   // rows: displacement of B's content against A's, at equal angle
    double shear = 0;      // rows per frame by which B's content moves against A's along the block
    double step_deg = 0;   // B's angle per frame
};

inline double wrap180(double deg) { return deg - 360.0 * std::round(deg / 360.0); }

// slope of v per index around i, over +- half frames (clipped to the series)
inline double local_slope(const std::vector<double>& v, std::int64_t i, std::int64_t half) {
    const std::int64_t n = static_cast<std::int64_t>(v.size());
    const std::int64_t a = std::max<std::int64_t>(0, i - half), b = std::min<std::int64_t>(n - 1, i + half);
    return b > a ? (v[b] - v[a]) / static_cast<double>(b - a) : 0.0;
}

// The block of n frames of A starting at p0, against the frames of B at the same angle plus angle_offset_deg
// (180 for opposite views). B may hold that angle in several turns: frame_hint says which, as the wanted
// difference of frame index (B minus A), to within a few thousand frames. false: B does not hold the block.
inline bool plan_block(const Trajectory& a, const Trajectory& b, std::int64_t p0, int n, std::int64_t frame_hint, double angle_offset_deg,
                       BlockPlan& plan) {
    const std::int64_t na = static_cast<std::int64_t>(a.angle_deg.size()), nb = static_cast<std::int64_t>(b.angle_deg.size());
    if (p0 < 0 || p0 + n > na || n < 1) return false;
    const std::int64_t mid = p0 + n / 2;
    std::int64_t guess = std::clamp<std::int64_t>(mid + frame_hint, 0, nb - 1);
    const double step = local_slope(b.angle_deg, guess, 64);
    if (step == 0.0) return false;
    // the frame of B, in the turn nearest to the hint, whose angle is A's plus the offset
    const double j_star = static_cast<double>(guess) + wrap180(a.angle_deg[mid] + angle_offset_deg - b.angle_deg[guess]) / step;
    const std::int64_t J = static_cast<std::int64_t>(std::llround(j_star)) - mid;
    if (p0 + J < 1 || p0 + J + n > nb - 1) return false;
    plan.p0 = p0;
    plan.j0 = p0 + J;
    plan.n = n;
    plan.step_deg = step;
    plan.row_offset.resize(n);
    const double slope_a = local_slope(a.rows, mid, 16), slope_b = local_slope(b.rows, mid + J, 16);
    double sum_frac = 0, sum_f = 0, sum_D = 0, sum_off = 0;
    for (int q = 0; q < n; ++q) {
        const std::int64_t p = p0 + q, j = p + J;
        const double frac = wrap180(a.angle_deg[p] + angle_offset_deg - b.angle_deg[j]) / step;  // frames
        const double delta = b.rows[j] - a.rows[p];
        plan.row_offset[q] = static_cast<int>(std::lround(delta));
        sum_frac += frac;
        sum_f += delta - plan.row_offset[q];
        sum_D += delta + frac * slope_b;
        sum_off += plan.row_offset[q];
    }
    plan.stated_dq = sum_frac / n;
    plan.stated_dv = sum_f / n + slope_a * plan.stated_dq;
    plan.stated_D = sum_D / n;
    plan.mean_row_offset = sum_off / n;
    plan.shear = slope_b - slope_a;
    return true;
}

// Displacement of B's content against A's at equal angle [rows], from the registration (d_q, d_v) of the blocks
// cut according to `plan`. It does not depend on what the record states beyond the integer rows the blocks were
// cut at and the shear used to cut them.
inline double measured_D(const BlockPlan& plan, double d_q, double d_v) { return plan.mean_row_offset + d_v + plan.shear * d_q; }

}  // namespace helirec::calibration
