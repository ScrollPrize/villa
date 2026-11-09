#pragma once
// Per-pixel accumulation / luminosity / mend arithmetic shared by CPU and CUDA.
// The reference runs on the CPU without FMA: every op is a separately rounded float op.
#include <cstddef>
#include <cstdint>

#include "hd_math.hpp"
#include "helirec/accumulate.hpp"

namespace helirec::accum_core {

using namespace helirec::hd;

// flat-fielded intensity of projection k at window row t, source column c
HR_HD float flat_field(const AccumScanView& s, int k, int t, int c) {
    const int r = s.row0[k] + t;  // raw detector row
    const std::size_t pix = static_cast<std::size_t>(r) * s.det_cols + c;
    const std::size_t slot = s.raw_slot ? static_cast<std::size_t>(s.raw_slot[k]) : static_cast<std::size_t>(k);
    const float u = static_cast<float>(s.raw[(slot * s.win_rows + t) * s.det_cols + c]);
    const float sig = div(sub(u, s.dark[pix]), s.current_div[k]);
    const std::size_t plane = static_cast<std::size_t>(s.det_rows) * s.det_cols;
    const float f = s.flat_frac[k];
    const float flat = add(mul(sub(1.0f, f), s.flats[s.flat_i0[k] * plane + pix]), mul(f, s.flats[s.flat_i1[k] * plane + pix]));
    return div(sig, flat);
}

// vertically interpolated value at window row t (raw row r = row0 + t), with the reference's
// edge padding: rows above / below the detector take the value at raw row 0 / det_rows - 2
// and the weight non_zero.
HR_HD void column_sample(const AccumScanView& s, int k, int t, int c, float non_zero, float& v, float& w) {
    const int r = s.row0[k] + t;
    const float a = s.fract_v[k];
    int tv;  // window row of the (interpolated) sample used
    bool pad = false;
    if (r < 0) { tv = -s.row0[k]; pad = true; }
    else if (r > s.det_rows - 2) { tv = s.det_rows - 2 - s.row0[k]; pad = true; }
    else tv = t;
    float val = flat_field(s, k, tv, c);
    if (a != 0.0f) val = add(mul(sub(1.0f, a), val), mul(a, flat_field(s, k, tv + 1, c)));
    float wt;
    if (pad) {
        wt = non_zero;
    } else {
        const int rr = s.row0[k] + tv;
        const float dw = s.weight_factor[k];
        const float w0 = mul(s.weight[static_cast<std::size_t>(rr) * s.det_cols + c], dw);
        wt = w0;
        if (a != 0.0f) {
            const float w1 = mul(s.weight[static_cast<std::size_t>(rr + 1) * s.det_cols + c], dw);
            wt = add(mul(sub(1.0f, a), w0), mul(a, w1));
        }
    }
    v = mul(val, wt);
    w = wt;
}

// all contributions of projection k (role: 0 lower bin, 1 upper bin) to merged column X, row t,
// in project_radio order: [c-1 right share] [left border pad (c == 0)] [c left share] [right border pad]
HR_HD void contribute(const AccumScanView& s, int k, int role, int t, int X, int NX, float non_zero, int border,
                      float& R, float& RW) {
    const float al = s.alpha[k];
    const float A = role == 0 ? sub(1.0f, al) : al;
    const int W = s.det_cols;
    const int rel = X - s.sx;                // column relative to the scan's origin (k in the reference)
    const int c0 = rel - s.ixs;              // source column whose left share lands on X
    const int c1 = c0 - 1;                   // source column whose right share lands on X
    auto add_term = [&](int c, float fx) {
        float v, w;
        column_sample(s, k, t, c, non_zero, v, w);
        R = add(R, mul(mul(v, A), fx));
        RW = add(RW, mul(mul(w, A), fx));
    };
    // i_col = c1 (processed before c0): its right share (fx = fract_x)
    if (c1 >= 0 && c1 < W) add_term(c1, s.fract_x);
    // i_col = 0: left border pad over relative columns [max(0, ixs + 1 - border), ixs + 1)
    if (rel >= imax(0, s.ixs + 1 - border) && rel < s.ixs + 1 && X >= 0 && X < NX) add_term(0, non_zero);
    // i_col = c0: its left share (fx = 1 - fract_x)
    if (c0 >= 0 && c0 < W) add_term(c0, sub(1.0f, s.fract_x));
    // i_col = W - 1: right border pad over relative columns [ixs + W, min(NX, ixs + W + border))
    if (rel >= s.ixs + W && rel < imin(NX, s.ixs + W + border) && X >= 0 && X < NX) add_term(W - 1, non_zero);
}

// ---------------------------------------------------------------- luminosity
HR_HD void window_sum(const float* R, const float* RW, int H, int NX, int j, int t, int cx, const float* gx,
                      const float* gz, int radius, float& sr, float& sw) {
    const int t0 = imax(0, t - radius), t1 = imin(H - 1, t + radius);
    const int x0 = imax(0, cx - radius), x1 = imin(NX - 1, cx + radius);
    sr = 0.f;
    sw = 0.f;
    for (int jj = t0; jj <= t1; ++jj) {
        const float gzw = gz[jj - t + radius];
        const std::size_t row = (static_cast<std::size_t>(j) * H + jj) * NX;
        for (int col = x0; col <= x1; ++col) {
            const float g = mul(gzw, gx[col - cx + radius]);
            sr = add(sr, mul(g, R[row + col]));
            sw = add(sw, mul(g, RW[row + col]));
        }
    }
}

HR_HD float luminosity_diff(float A_ha, float B_ha, float A_tot, float B_tot) {
    // slot 0: HA sums; slot 1: -HA + total (accumulated as in store_luminosity_infos)
    const float c1 = add(-A_ha, A_tot), w1 = add(-B_ha, B_tot);
    const float lum0 = (B_ha != 0.0f) ? div(A_ha, B_ha) : 0.0f;
    const float lum1 = (w1 != 0.0f) ? div(c1, w1) : 0.0f;
    return sub(lum1, lum0);
}

// ---------------------------------------------------------------- mend
// one row (length NX) of R / RW, nabuxx mend_null_weights semantics on the usable columns [x0, x1) of the merged row:
// columns there whose weight is below thr x (mean weight over [x0, x1)) take the value of the nearest column (either
// side, anywhere in the row) whose weight reaches that level (ties: the left one)
HR_HD void mend_row(float* r, const float* w, int NX, float thr, int x0, int x1) {
    x0 = x0 < 0 ? 0 : x0;
    x1 = x1 > NX ? NX : x1;
    const int n_valid = x1 - x0;
    if (n_valid <= 0) return;
    float s = 0.f;
    for (int x = x0; x < x1; ++x) s = add(s, w[x]);
    const float avg = div(s, static_cast<float>(n_valid));
    const float lim = mul(avg, thr);
    for (int x = x0; x < x1; ++x) {
        if (!(w[x] < lim)) continue;
        int l = -1, rr = -1;
        for (int y = x - 1; y >= 0; --y)
            if (w[y] >= lim) { l = y; break; }
        for (int y = x + 1; y < NX; ++y)
            if (w[y] >= lim) { rr = y; break; }
        if (l >= 0 && (rr < 0 || !(rr - x < x - l))) r[x] = r[l];
        else if (rr >= 0) r[x] = r[rr];
    }
}

}  // namespace helirec::accum_core
