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

// Origins of corresponding subwindows. For mirrored B the crop moves in the opposite
// direction, so a + b + width - 1 (twice the nominal mirror axis) is invariant.
inline void crop_column_origins(int& a, int& b, int width, int cropped_width, double fraction, bool mirror, int inset = 0) {
    const int spare = width - cropped_width;
    const int offset = inset + static_cast<int>(std::lround((spare - 2 * inset) * fraction));
    a += offset;
    b += mirror ? spare - offset : offset;
}

struct Trajectory {
    std::vector<double> angle_deg, rows;
};

struct BlockPlan {
    std::int64_t p0 = 0, j0 = 0;  // frames p0 .. p0 + n - 1 of A against frames j0 .. j0 + n - 1 of B
    int n = 0, margin = 0;        // B's block also takes `margin` frames before and after
    int row_offset = 0;           // B's block is cut this many rows above A's, in every frame
    // per frame of B's block, margins included (n + 2 margin values): the rows by which, according to the record,
    // B's content in that frame still lies above A's after the cut. The registration removes it before comparing.
    std::vector<double> row_shift;
    double max_row_shift = 0;
    double stated_dq = 0;  // a(q, ..) is at the angle of b(q + stated_dq, ..): what the registration along frames must return
    double stated_D = 0;   // rows: displacement of B's content against A's at equal angle
    double slope_a = 0, slope_b = 0;  // rows per frame the content of A and of B travels along the axis
    double step_deg = 0;              // B's angle per frame
    // what the registration along rows must return if the stated heights are right
    double stated_dv() const { return slope_a * stated_dq; }
};

inline double wrap180(double deg) { return deg - 360.0 * std::round(deg / 360.0); }

// slope of v per index around i, over +- half frames (clipped to the series)
inline double local_slope(const std::vector<double>& v, std::int64_t i, std::int64_t half) {
    const std::int64_t n = static_cast<std::int64_t>(v.size());
    const std::int64_t a = std::max<std::int64_t>(0, i - half), b = std::min<std::int64_t>(n - 1, i + half);
    return b > a ? (v[b] - v[a]) / static_cast<double>(b - a) : 0.0;
}

// The block of n frames of A starting at p0 against the frames of B that lie J frames further on in B's own
// numbering, where the angle of B is compared with A's plus angle_offset_deg (180 for opposite views).
// false: the stretches do not hold the block and its margins.
inline bool plan_block_at(const Trajectory& a, const Trajectory& b, std::int64_t p0, int n, int margin, std::int64_t J, double angle_offset_deg,
                          BlockPlan& plan) {
    const std::int64_t na = static_cast<std::int64_t>(a.angle_deg.size()), nb = static_cast<std::int64_t>(b.angle_deg.size());
    if (n < 1 || margin < 0 || p0 - margin < 0 || p0 + n + margin > na) return false;
    if (p0 + J - margin < 0 || p0 + J + n + margin > nb) return false;
    const std::int64_t mid = p0 + n / 2;
    const double step = local_slope(b.angle_deg, mid + J, 64);
    if (step == 0.0) return false;
    plan.p0 = p0;
    plan.j0 = p0 + J;
    plan.n = n;
    plan.margin = margin;
    plan.step_deg = step;
    plan.slope_a = local_slope(a.rows, mid, 16);
    plan.slope_b = local_slope(b.rows, mid + J, 16);
    double sum_frac = 0, sum_delta = 0;
    for (int q = 0; q < n; ++q) {
        const std::int64_t p = p0 + q, j = p + J;
        sum_frac += wrap180(a.angle_deg[p] + angle_offset_deg - b.angle_deg[j]) / step;  // frames
        sum_delta += b.rows[j] - a.rows[p];
    }
    plan.stated_dq = sum_frac / n;
    plan.row_offset = static_cast<int>(std::lround(sum_delta / n));
    plan.stated_D = sum_delta / n + plan.stated_dq * plan.slope_b;
    plan.row_shift.resize(static_cast<std::size_t>(n) + 2 * margin);
    plan.max_row_shift = 0;
    for (int q = -margin; q < n + margin; ++q) {
        const double g = b.rows[p0 + J + q] - a.rows[p0 + q] - plan.row_offset;
        plan.row_shift[static_cast<std::size_t>(q + margin)] = g;
        plan.max_row_shift = std::max(plan.max_row_shift, std::fabs(g));
    }
    return true;
}

// The same, with B's frames found from the angles: the frames of B at A's angle plus the offset. B may hold that
// angle in several turns: frame_hint says which, as the wanted difference of frame index (B minus A), to within
// a few thousand frames.
inline bool plan_block(const Trajectory& a, const Trajectory& b, std::int64_t p0, int n, int margin, std::int64_t frame_hint,
                       double angle_offset_deg, BlockPlan& plan) {
    const std::int64_t na = static_cast<std::int64_t>(a.angle_deg.size()), nb = static_cast<std::int64_t>(b.angle_deg.size());
    if (n < 1 || p0 < 0 || p0 + n > na) return false;
    const std::int64_t mid = p0 + n / 2;
    const std::int64_t guess = std::clamp<std::int64_t>(mid + frame_hint, 0, nb - 1);
    const double step = local_slope(b.angle_deg, guess, 64);
    if (step == 0.0) return false;
    const double j_star = static_cast<double>(guess) + wrap180(a.angle_deg[mid] + angle_offset_deg - b.angle_deg[guess]) / step;
    return plan_block_at(a, b, p0, n, margin, static_cast<std::int64_t>(std::llround(j_star)) - mid, angle_offset_deg, plan);
}

// Displacement of B's content against A's at equal angle [rows], from the registration (d_q, d_v) of blocks cut
// and pre-aligned according to `plan`: d_v - slope_a d_q is by how much the content lies above where the record
// puts it, and the last term moves the comparison from the stated to the measured frame of equal angle.
inline double measured_D(const BlockPlan& plan, double d_q, double d_v) {
    return plan.stated_D + (d_v - plan.slope_a * d_q) + plan.slope_b * (d_q - plan.stated_dq);
}

}  // namespace helirec::calibration
