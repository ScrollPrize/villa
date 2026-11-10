// Port of nabuxx span_info.cpp / etfscan.cpp geometry (float/double choices kept).
#include "helirec/helical.hpp"

#include <algorithm>
#include <cmath>

namespace helirec {

SpanInfo make_span_info(const ScanTrajectory& t, double tol_steps) {
    const int n = static_cast<int>(t.n());
    if (n < 2 || static_cast<int>(t.z_translation_m.size()) != n) throw Error("make_span_info: bad trajectory");
    SpanInfo s;
    s.z_pix_per_proj.resize(n);
    s.x_pix_per_proj.resize(n);
    s.integer_shift_v.resize(n);
    s.fract_complement.resize(n);
    const double zp0 = t.z_translation_m[0] / t.pixel_size_m;
    for (int i = 0; i < n; ++i) {
        s.z_pix_per_proj[i] = static_cast<float>(t.z_translation_m[i] / t.pixel_size_m - zp0);
        const double lat = t.lateral_correction_m.empty() ? 0.0 : t.lateral_correction_m[i] / t.pixel_size_m;
        s.x_pix_per_proj[i] = static_cast<float>(lat) + t.x_readjustment;
        const float trans_v = s.z_pix_per_proj[i] + t.z_readjustment;
        const int c = static_cast<int>(std::ceil(trans_v));
        s.integer_shift_v[i] = c;
        s.fract_complement[i] = static_cast<float>(c) - trans_v;
    }
    const float zmax = *std::max_element(s.z_pix_per_proj.begin(), s.z_pix_per_proj.end());
    const float zmin = *std::min_element(s.z_pix_per_proj.begin(), s.z_pix_per_proj.end());
    const int top = std::max(t.det_rows, static_cast<int>(std::ceil(static_cast<float>(t.det_rows) + zmax + t.z_readjustment)));
    const int bottom = std::min(0, static_cast<int>(std::floor(0.0f + zmin + t.z_readjustment)));
    s.h_first = bottom;
    const int nh = top - bottom + 1;
    s.sunshine_start.assign(nh, -1);
    s.sunshine_end.assign(nh, -1);
    const double avg_step = (t.angles_deg[n - 1] - t.angles_deg[0]) / (n - 1);
    const float bonus = static_cast<float>(std::fabs(avg_step * tol_steps));
#pragma omp parallel for schedule(dynamic, 16)
    for (int ih = 0; ih < nh; ++ih) {
        const int h = bottom + ih;
        int first = -1, last = -1;
        for (int i = 0; i < n; ++i) {
            const int pos = h - s.integer_shift_v[i];
            const int lim = t.det_rows - (std::fabs(s.fract_complement[i]) > 1e-6f ? 1 : 0);
            if (pos >= 0 && pos < lim) {
                if (first < 0) first = i;
                last = i;
            }
        }
        if (first < 0) continue;
        const double a0 = t.angles_deg[first], a1 = t.angles_deg[last];
        const float delta = static_cast<float>(std::max(a0, a1) - std::min(a0, a1));
        if (delta + bonus >= 360.0f) {
            s.sunshine_start[ih] = first;
            s.sunshine_end[ih] = last + 1;
        }
    }
    s.doable_lo = s.doable_hi = 0;
    int lo = 0, hi = 0;
    bool found = false;  // heights can be negative: no sentinel value
    for (int ih = 0; ih < nh; ++ih)
        if (s.sunshine_start[ih] >= 0) {
            if (!found) lo = bottom + ih;
            found = true;
            hi = bottom + ih;
        }
    if (!found) throw Error("make_span_info: no doable height");
    s.doable_lo = lo;
    s.doable_hi = hi;
    return s;
}

std::pair<int, int> doable_span(const std::vector<SpanInfo>& spans) {
    if (spans.empty()) throw Error("doable_span: no scans");
    int lo = spans[0].doable_lo, hi = spans[0].doable_hi;
    for (const auto& s : spans) {
        lo = std::max(lo, s.doable_lo);
        hi = std::min(hi, s.doable_hi);
    }
    if (hi <= lo) throw Error("doable_span: empty intersection");
    return {lo, hi};  // used half-open, as the reference
}

std::pair<int, int> chunk_projection_span(const SpanInfo& s, int h0, int h1) {
    int first = -1, last = -1;
    for (int h = h0; h <= h1; ++h) {
        const int ih = h - s.h_first;
        if (ih < 0 || ih >= static_cast<int>(s.sunshine_start.size()) || s.sunshine_start[ih] < 0)
            throw Error("chunk_projection_span: height " + std::to_string(h) + " is not doable");
        first = first < 0 ? s.sunshine_start[ih] : std::min(first, s.sunshine_start[ih]);
        last = std::max(last, s.sunshine_end[ih] - 1);
    }
    return {first, last + 1};
}

int grid_angle_count(const std::vector<double>& a) {
    double sum = 0;
    for (std::size_t i = 1; i < a.size(); ++i) sum += a[i] - a[i - 1];
    const double step = std::fabs(sum / static_cast<double>(a.size() - 1));
    return static_cast<int>(std::nearbyint(360.0 / step));
}

double angle_grid_phase(const std::vector<double>& a, int n_grid) {
    if (a.empty()) return 0;
    const double f0 = a[0] * n_grid / 360.0;
    return f0 - std::nearbyint(f0);
}

AngleBins nearest_angle_bins(const std::vector<double>& a, int n_grid, double phase) {
    AngleBins out;
    const std::size_t n = a.size();
    out.bin.resize(n);
    out.residual.resize(n);
    const double per_deg = n_grid / 360.0;
    long b = 0;
    for (std::size_t p = 0; p < n; ++p) {
        const double f = a[p] * per_deg;
        b = p == 0 ? std::lround(f - phase) : b + std::lround(f - a[p - 1] * per_deg);
        out.bin[p] = static_cast<int32_t>(b);
        out.residual[p] = static_cast<float>(f - static_cast<double>(b));
    }
    return out;
}

FlatInterpolation interpolate_flats(const FlatSeriesInfo& flats, float ang) {
    const auto& fa=flats.angles_deg;
    const int nf=static_cast<int>(fa.size());
    if(nf==0) throw Error("flat interpolation: no flats");
    const int i1=static_cast<int>(std::lower_bound(fa.begin(),fa.end(),ang)-fa.begin());
    if(i1==0) return {0,0,0};
    if(i1==nf) return {nf-1,nf-1,0};
    return {i1-1,i1,(ang-fa[i1-1])/(fa[i1]-fa[i1-1])};
}

std::pair<int, int> chunk_projections(const ScanTrajectory& t, const SpanInfo& s, int h_lo, int h_hi, int margin,
                                      bool full_context, bool padded_rows) {
    // the reference includes the height h_hi itself (inclusive end of span_v) when selecting projections
    auto [p0, p1] = chunk_projection_span(s, h_lo, h_hi);
    if (full_context) {
        // All observations contributing to the central slices AND their filter/luminosity
        // context. Selecting only the central heights makes the context depend on chunking.
        p0 = static_cast<int>(t.n());
        p1 = 0;
        for (int p = 0; p < static_cast<int>(t.n()); ++p) {
            const int first = s.integer_shift_v[p];
            const int last = first + t.det_rows - 1 - ((padded_rows || s.fract_complement[p] > 0.f) ? 1 : 0);
            if (last >= h_lo - margin && first < h_hi + margin) {
                p0 = std::min(p0, p);
                p1 = p + 1;
            }
        }
        if (p1 <= p0) throw Error("prepare_chunk_scan: no projection sees the chunk");
    }
    return {p0, p1};
}

int max_chunk_projections(const ScanTrajectory& t, const SpanInfo& s, int lo, int hi, int rows, int margin,
                          bool full_context, bool padded_rows) {
    int most = 0;
#pragma omp parallel for schedule(static) reduction(max : most)
    for (int h = lo; h < hi; ++h) {
        try {
            const auto [p0, p1] = chunk_projections(t, s, h, std::min(hi, h + rows), margin, full_context, padded_rows);
            most = std::max(most, p1 - p0);
        } catch (const Error&) {   // (not a chunk that can be reconstructed)
        }
    }
    return most;
}

ChunkScan prepare_chunk_scan(const ScanTrajectory& t, const SpanInfo& s, const std::vector<double>& control_A,
                             const FlatSeriesInfo& flats, int h_lo, int h_hi, int margin, int n_grid,
                             double nominal_current, bool full_context, bool padded_rows) {
    ChunkScan c;
    const auto [p0, p1] = chunk_projections(t, s, h_lo, h_hi, margin, full_context, padded_rows);
    c.p0 = p0;
    c.p1 = p1;
    const int n = p1 - p0;
    c.row0.resize(n);
    c.fract_v.resize(n);
    c.j0.resize(n);
    c.alpha.resize(n);
    c.current_div.resize(n);
    c.weight_factor.resize(n);
    c.flat_frac.resize(n);
    c.flat_i0.resize(n);
    c.flat_i1.resize(n);
    for (int k = 0; k < n; ++k) {
        const int p = p0 + k;
        c.row0[k] = h_lo - margin - s.integer_shift_v[p];
        c.fract_v[k] = s.fract_complement[p];
        // angular splat (InterpInfo::setup_for_iproj)
        double a = t.angles_deg[p] * (2 * M_PI / 360.0);
        a = std::fmod(a, 2 * M_PI);
        if (a < 0) a += 2 * M_PI;
        const float f_pos = static_cast<float>(n_grid * (a / (2 * M_PI)));
        c.j0[k] = static_cast<int>(std::floor(f_pos));
        c.alpha[k] = f_pos - static_cast<float>(c.j0[k]);  // j0 NOT wrapped (reference drops j0 == n_grid)
        // current normalisation (float32(I/0.2), as the reference)
        const float cd = static_cast<float>(control_A[p] / nominal_current);
        c.current_div[k] = cd;
        c.weight_factor[k] = cd;
        // flat interpolation in angle (lower_bound semantics, clamped)
        const float ang = static_cast<float>(t.angles_deg[p]);
        const auto flat = interpolate_flats(flats, ang);
        c.flat_i0[k]=flat.i0; c.flat_i1[k]=flat.i1; c.flat_frac[k]=flat.fraction;
    }
    // horizontal placement: merged column of source column cc = sx + floor(x_shift) + cc (+ fract)
    const float x_shift = -s.x_pix_per_proj[p0];  // constant here (no lateral correction drift assumed)
    for (int k = 1; k < n; ++k)
        if (s.x_pix_per_proj[p0 + k] != s.x_pix_per_proj[p0])
            throw Error("prepare_chunk_scan: per-projection lateral correction is not supported yet");
    c.sx = t.shift_x;
    c.ixs = static_cast<int>(std::floor(x_shift));
    c.fract_x = x_shift - static_cast<float>(c.ixs);
    return c;
}

}  // namespace helirec
