#pragma once
// Host-side tables of the half-acquisition rebalance, shared by the CPU and CUDA implementations: the contributions
// of the opposite rays (theta_i + pi; legacy: the nabuxx table of theta_i + k pi), the mirror map of odd k about the
// rotation axis, and the empty-angle holes.
#include <cmath>
#include <utility>
#include <vector>

#include "helirec/sino.hpp"

namespace helirec::rebalance_core {

struct Contribution {
    int k0, k1;
    float alpha;
    bool has_two, odd;
};

inline bool locate(double pos, int n, Contribution& c) {
    if (pos <= 0.0 + 1e-12) return false;
    if (pos > static_cast<double>(n)) {
        if (pos - n > 1e-8) return false;
        c.k0 = n - 1; c.k1 = -1; c.alpha = 0.f; c.has_two = false;
        return true;
    }
    const int k0 = static_cast<int>(std::floor(pos));
    if (k0 == n) { c.k0 = n - 1; c.k1 = -1; c.alpha = 0.f; c.has_two = false; return true; }
    if (k0 + 1 < n) { c.k0 = k0; c.k1 = k0 + 1; c.alpha = static_cast<float>(pos - k0); c.has_two = true; return true; }
    c.k0 = k0; c.k1 = -1; c.alpha = 0.f; c.has_two = false;
    return true;
}

struct OddMap { int src0, src1; float w0, w1; };

inline std::vector<OddMap> odd_shift_map(int n_cols, float rot_center) {
    const float center = (static_cast<float>(n_cols) - 1.0f) * 0.5f;
    const float delta = 2.0f * (rot_center - center);
    const int s0 = static_cast<int>(std::floor(delta)), s1 = s0 + 1;
    const float alpha = delta - static_cast<float>(s0);
    const bool exact = std::abs(alpha) < 1e-12f;
    std::vector<OddMap> m(n_cols);
    auto src = [&](int c, int shift) {
        const int sc = c - shift;
        return (sc < 0 || sc >= n_cols) ? -1 : n_cols - 1 - sc;
    };
    for (int c = 0; c < n_cols; ++c) {
        m[c].src0 = src(c, s0);
        if (exact) { m[c].src1 = -1; m[c].w0 = 1.f; m[c].w1 = 0.f; }
        else { m[c].src1 = src(c, s1); m[c].w0 = 1.0f - alpha; m[c].w1 = alpha; }
    }
    return m;
}

// table[i]: the contributions to angle i (theta_i = 2 pi i / n_proj, one full turn).
// Default: the opposite view theta_i + pi only, located periodically (every distinct line counted once: itself +
// its mirrored opposite view; theta + 2 pi k is the same grid sample and must not be added again).
// legacy: the reference's table (theta_i + k pi, k = +-1 .. +-3 located on [0, n] without wrapping, endpoint n mapped
// to the last angle), in the reference's order.
inline std::vector<std::vector<Contribution>> contribution_table(int n_proj, bool legacy) {
    std::vector<std::vector<Contribution>> table(n_proj);
    if (!legacy) {
        for (int i = 0; i < n_proj; ++i) {
            const double pos = std::fmod(i + 0.5 * n_proj, static_cast<double>(n_proj));
            const int k0 = static_cast<int>(std::floor(pos));
            const double a = pos - k0;
            Contribution c;
            c.odd = true;
            c.k0 = k0 % n_proj;
            c.has_two = a > 1e-12;
            c.k1 = c.has_two ? (k0 + 1) % n_proj : -1;
            c.alpha = c.has_two ? static_cast<float>(a) : 0.f;
            table[i].push_back(c);
        }
        return table;
    }
    const int n_span = 2;
    const double dtheta = 2.0 * M_PI / n_proj;
    for (int i = 0; i < n_proj; ++i)
        for (int ih = -n_span - 1; ih <= n_span + 1; ++ih) {
            if (ih == 0) continue;
            Contribution c;
            if (!locate((dtheta * i + ih * M_PI) / dtheta, n_proj, c)) continue;
            c.odd = (ih & 1) != 0;
            table[i].push_back(c);
        }
    return table;
}

// runs [s, e) of empty angles (e may exceed n_proj: a run wrapping through angle 0)
inline std::vector<std::pair<int, int>> holes_from(const std::vector<char>& nonempty) {
    const int n_proj = static_cast<int>(nonempty.size());
    std::vector<std::pair<int, int>> holes;
    for (int i = 0; i < n_proj;) {
        if (nonempty[i]) { ++i; continue; }
        const int s = i;
        while (i < n_proj && !nonempty[i]) ++i;
        holes.push_back({s, i});
    }
    if (n_proj && !nonempty[0] && !nonempty[n_proj - 1] && holes.size() >= 2) {
        auto first = holes.front(), last = holes.back();
        holes.erase(holes.begin());
        holes.back() = {last.first, first.second + n_proj};
    }
    return holes;
}

// A sinogram with one row per recorded view over more than one turn (sino.hpp, rebalance_turns): `turn` rows per full
// turn (even), rows i and i + turn at the same angle. table[i]: the other recordings of the lines of row i, i.e. the
// rows i + k turn / 2 (k != 0) inside the sinogram, in ascending order; those of odd k are opposite views (mirrored).
inline std::vector<std::vector<Contribution>> contribution_table_turns(int n_proj, int turn) {
    std::vector<std::vector<Contribution>> table(n_proj);
    const int half = turn / 2;
    for (int i = 0; i < n_proj; ++i)
        for (int k = -(i / half); i + k * half < n_proj; ++k) {
            if (k == 0) continue;
            Contribution c;
            c.k0 = i + k * half;
            c.k1 = -1;
            c.alpha = 0.f;
            c.has_two = false;
            c.odd = (k & 1) != 0;
            table[i].push_back(c);
        }
    return table;
}

// The angles nothing was recorded at, for such a sinogram (sino.hpp, TurnHoles). peak: the largest weight in the
// mid-row of each row (n_proj >= turn of them; 0: an empty row, the emptiness test of rebalance). An angle (row modulo
// `turn`) is empty when every row at it is. Each is filled in its first row, from the rows of the angles either side
// of its run (as fill_holes fills a run of one turn): of an angle's rows, the one whose peak is largest (the first of
// equal ones), i.e. the view that records the slice rather than a row that holds only what pads it.
inline TurnHoles turn_holes_from(const std::vector<float>& peak, int turn) {
    const int n_proj = static_cast<int>(peak.size());
    TurnHoles out;
    std::vector<char> angle(turn, 0);
    std::vector<int> best(turn, -1);
    for (int r = 0; r < n_proj; ++r)
        if (peak[r] > 0.f && (best[r % turn] < 0 || peak[r] > peak[best[r % turn]])) {
            best[r % turn] = r;
            angle[r % turn] = 1;
        }
    bool any = false;
    for (char v : angle) any |= v;
    if (!any) {
        out.all_empty = true;
        return out;
    }
    for (auto [s, e] : holes_from(angle)) {
        const int left = best[(s + turn - 1) % turn], right = best[e % turn];
        for (int idx = s; idx < e; ++idx) {
            out.dst.push_back(idx % turn);
            out.left.push_back(left);
            out.right.push_back(right);
        }
    }
    return out;
}

}  // namespace helirec::rebalance_core
