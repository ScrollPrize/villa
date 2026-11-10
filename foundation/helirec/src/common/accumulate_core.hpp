#pragma once
// Per-pixel accumulation / luminosity / mend arithmetic shared by CPU and CUDA.
// The reference runs on the CPU without FMA: every op is a separately rounded float op.
#include <cstddef>
#include <cstdint>

#include "hd_math.hpp"
#include "helirec/accumulate.hpp"
#include "sino_core.hpp"
#include "sld_core.hpp"

namespace helirec::accum_core {

using namespace helirec::hd;

// flat-fielded intensity of projection k at window row t, source column c
HR_HD float flat_field(const AccumScanView& s, int k, int t, int c) {
    const int r = s.row0[k] + t;  // raw detector row
    const std::size_t pix = static_cast<std::size_t>(r) * s.det_cols + c;
    const std::size_t slot = s.raw_slot ? static_cast<std::size_t>(s.raw_slot[k]) : static_cast<std::size_t>(k);
    const float u = static_cast<float>(s.raw[(slot * s.win_rows + t) * s.det_cols + c]);
    float sig = div(sub(u, s.dark[pix]), s.current_div[k]);
    if (s.glare) {   // scintillator light diffusion: minus the glare, interpolated between its sampled projections
        const std::size_t cell = static_cast<std::size_t>(s.glare_rows) * s.glare_cols;
        const float gf = s.glare_frac[k];
        const float g0 = sld_core::glare_sample(s.glare + s.glare_i0[k] * cell, s.glare_rows, s.glare_cols, s.glare_inv_bin, r, c);
        const float g1 = sld_core::glare_sample(s.glare + s.glare_i1[k] * cell, s.glare_rows, s.glare_cols, s.glare_inv_bin, r, c);
        sig = sub(sig, add(mul(sub(1.0f, gf), g0), mul(gf, g1)));
    }
    const std::size_t plane = static_cast<std::size_t>(s.det_rows) * s.det_cols;
    const float f = s.flat_frac[k];
    const float flat = add(mul(sub(1.0f, f), s.flats[s.flat_i0[k] * plane + pix]), mul(f, s.flats[s.flat_i1[k] * plane + pix]));
    return div(sig, flat);
}

// vertically interpolated value at window row t (raw row r = row0 + t), with the reference's
// edge padding: rows above / below the detector take the value at raw row 0 / det_rows - 2
// and the weight non_zero.
HR_HD void column_value(const AccumScanView& s, int k, int t, int c, float non_zero, float& val, float& wt) {
    const int r = s.row0[k] + t;
    const float a = s.fract_v[k];
    int tv;  // window row of the (interpolated) sample used
    bool pad = false;
    if (r < 0) { tv = -s.row0[k]; pad = true; }
    else if (r > s.det_rows - 2) { tv = s.det_rows - 2 - s.row0[k]; pad = true; }
    else tv = t;
    val = flat_field(s, k, tv, c);
    if (a != 0.0f) val = add(mul(sub(1.0f, a), val), mul(a, flat_field(s, k, tv + 1, c)));
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
}

// the same sample as the splat uses it: value x weight, weight
HR_HD void column_sample(const AccumScanView& s, int k, int t, int c, float non_zero, float& v, float& w) {
    float val, wt;
    column_value(s, k, t, c, non_zero, val, wt);
    v = mul(val, wt);
    w = wt;
}

// Interpolation corners carry their own measured weights; invalid corners and detector
// padding never contribute. In particular, do not multiply an invalid value by zero.
HR_HD void measured_column(const AccumScanView& s, int k, int t, int c, const AccumParams& p,
                           float& v, float& w) {
    v = w = 0.f;
    const float a = s.fract_v[k];
    const int base_row = s.row0[k] + t;
    if (base_row < 0 || base_row >= s.det_rows || (a > 0.f && base_row + 1 >= s.det_rows)) return;
    for (int dz = 0; dz < 2; ++dz) {
        const float f = dz ? a : sub(1.f, a);
        const int r = s.row0[k] + t + dz;
        if (f <= 0.f || r < 0 || r >= s.det_rows) continue;
        const float wt = mul(mul(s.weight[static_cast<std::size_t>(r) * s.det_cols + c], s.weight_factor[k]), f);
        if (!(wt > 0.f) || !hd::is_finite(wt)) continue;
        float val = flat_field(s, k, t + dz, c);
        if (!hd::is_finite(val)) continue;
        if (p.log_values) val = sino_core::minus_log(val, p.log_clip_min, p.log_clip_max);
        v = add(v, mul(val, wt));
        w = add(w, wt);
    }
}

// Fractional column offset with the scan's kernel (AccumScanView::x_taps > 0) in place of the two splat shares: the
// value at source position c0 - fract_x from the x_taps columns around it, with the splat's weight
// (fract_x w(c0 - 1) + (1 - fract_x) w(c0)), so that RW is the splat's. Only where every tap is a measured column
// with a positive weight (never a value the splat would not have used); false: nothing added, use the splat.
HR_HD bool kernel_term(const AccumScanView& s, int k, int t, int c0, float A, const AccumParams& p, float& R, float& RW) {
    const int T = s.x_taps, first = c0 - T / 2;
    if (first < 0 || first + T > s.det_cols) return false;
    if (p.measured_only && A <= 0.f) return true;
    float val = 0.f, wlin = 0.f;
    for (int i = 0; i < T; ++i) {
        float v, w;
        if (p.measured_only) {
            measured_column(s, k, t, first + i, p, v, w);
            if (!(w > 0.f)) return false;
            v = div(v, w);
        } else {
            column_value(s, k, t, first + i, p.non_zero, v, w);
            if (!(w > 0.f)) return false;
        }
        val = add(val, mul(s.x_w[i], v));
        if (i == T / 2 - 1) wlin = add(wlin, mul(s.fract_x, w));
        if (i == T / 2) wlin = add(wlin, mul(sub(1.0f, s.fract_x), w));
    }
    R = add(R, mul(mul(val, wlin), A));
    RW = add(RW, mul(wlin, A));
    return true;
}

// all contributions of projection k (role: 0 lower bin, 1 upper bin) to merged column X, row t,
// in project_radio order: [c-1 right share] [left border pad (c == 0)] [c left share] [right border pad]
HR_HD void contribute(const AccumScanView& s, int k, int role, int t, int X, const AccumParams& p,
                      float& R, float& RW) {
    const int NX = p.NX, border = p.border;
    const float non_zero = p.non_zero;
    const float al = s.alpha[k];
    const float A = role == 0 ? sub(1.0f, al) : al;
    const int W = s.det_cols;
    const int rel = X - s.sx;                // column relative to the scan's origin (k in the reference)
    const int c0 = rel - s.ixs;              // source column whose left share lands on X
    const int c1 = c0 - 1;                   // source column whose right share lands on X
    // (all taps inside the detector: no border pad lands on X either)
    if (s.x_taps > 0 && kernel_term(s, k, t, c0, A, p, R, RW)) return;
    auto add_term = [&](int c, float fx) {
        if (p.measured_only && (A <= 0.f || fx <= 0.f)) return;
        float v, w;
        if (p.measured_only) measured_column(s, k, t, c, p, v, w);
        else column_sample(s, k, t, c, non_zero, v, w);
        R = add(R, mul(mul(v, A), fx));
        RW = add(RW, mul(mul(w, A), fx));
    };
    // i_col = c1 (processed before c0): its right share (fx = fract_x)
    if (c1 >= 0 && c1 < W) add_term(c1, s.fract_x);
    // i_col = 0: left border pad over relative columns [max(0, ixs + 1 - border), ixs + 1)
    if (!p.measured_only && rel >= imax(0, s.ixs + 1 - border) && rel < s.ixs + 1 && X >= 0 && X < NX) add_term(0, non_zero);
    // i_col = c0: its left share (fx = 1 - fract_x)
    if (c0 >= 0 && c0 < W) add_term(c0, sub(1.0f, s.fract_x));
    // i_col = W - 1: right border pad over relative columns [ixs + W, min(NX, ixs + W + border))
    if (!p.measured_only && rel >= s.ixs + W && rel < imin(NX, s.ixs + W + border) && X >= 0 && X < NX) add_term(W - 1, non_zero);
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
HR_HD int nearest_supported(const float* w, int NX, int x, float lim) {
    for (int d = 1; d < NX; ++d) {
        if (x - d >= 0 && w[x - d] >= lim && w[x - d] > 0.f) return x - d;
        if (x + d < NX && w[x + d] >= lim && w[x + d] > 0.f) return x + d;
    }
    return -1;
}

HR_HD void measured_support(float* r, float* w, int NX, float thr) {
    float sum = 0.f;
    int count = 0;
    for (int x = 0; x < NX; ++x) {
        if (w[x] > 0.f && hd::is_finite(w[x]) && hd::is_finite(r[x])) { sum = add(sum, w[x]); ++count; }
        else w[x] = 0.f;
    }
    const float lim = count ? mul(thr, div(sum, static_cast<float>(count))) : 0.f;
    for (int x = 0; x < NX; ++x) if (w[x] < lim) w[x] = 0.f;
}

// Filter context only. Read exclusively from measured pixels, which are never modified,
// so rows can be filled in parallel without making the result depend on scheduling.
HR_HD void fill_measured_row(float* frame, const float* w, int H, int NX, int t, float empty) {
    int src = -1;
    for (int d = 0; d < H && src < 0; ++d) {
        for (int side = 0; side < 2 && src < 0; ++side) {
            const int y = side ? t + d : t - d;
            if (y < 0 || y >= H) continue;
            for (int x = 0; x < NX; ++x) if (w[static_cast<std::size_t>(y) * NX + x] > 0.f) { src = y; break; }
        }
    }
    float* row = frame + static_cast<std::size_t>(t) * NX;
    const float* own_w = w + static_cast<std::size_t>(t) * NX;
    for (int x = 0; x < NX; ++x) {
        if (own_w[x] > 0.f) continue;
        if (src < 0) { row[x] = empty; continue; }
        const float* sw = w + static_cast<std::size_t>(src) * NX;
        const int q = sw[x] > 0.f ? x : nearest_supported(sw, NX, x, 0.f);
        row[x] = q >= 0 ? frame[static_cast<std::size_t>(src) * NX + q] : empty;
    }
}

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
        const int q = nearest_supported(w, NX, x, lim);
        if (q >= 0) r[x] = r[q];
    }
}

}  // namespace helirec::accum_core
