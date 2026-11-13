// The record against views that recorded the same lines: opposite views of a turn at rest, a turn at rest
// against the helix, the helix a turn apart, and one scan against the other.
//
// Every block is cut from raw frames at the whole rows, columns and frames the record implies (view_plan.hpp),
// converted to line integrals with the scan's own dark and mean flat (frames.hpp), and registered
// (registration.hpp). What is reported is the registration against what the record states.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>

#include "calibration/util.hpp"
#include "calibration/compare_blocks.hpp"
#include "calibration/view_plan.hpp"
#include "helirec/calibration/registration.hpp"
#include "helirec/calibration/record_checks.hpp"
#include "helirec/calibration/session.hpp"

namespace helirec::calibration {

namespace {

using Scan = ViewScan;
using Timing = ViewTiming;
Timing& timing() { return view_timing(); }

std::string pm(const Spread& s, const char* unit) {
    return fmt("%+.4f +- %.4f %s (sd %.4f, n %zu)", s.mean, s.n > 1 ? s.sem() : NAN, unit, s.sd, s.n);
}

// y = mean + once-per-turn + twice-per-turn in the angle [deg]: the mean and the two amplitudes
struct Harmonics {
    double mean = 0, once = 0, twice = 0, rms = 0;
    bool ok = false;
};
Harmonics harmonics(const std::vector<double>& angle_deg, const std::vector<double>& y) {
    Harmonics h;
    const std::size_t n = y.size();
    if (n < 8) return h;
    if (angle_deg.size() != n) return h;
    LinearProblem p;
    p.unknowns = 5;
    p.y = y;
    for (std::size_t i = 0; i < n; ++i) {
        const double t = angle_deg[i] * M_PI / 180.0;
        const double b[5] = {1.0, std::cos(t), std::sin(t), std::cos(2 * t), std::sin(2 * t)};
        for (int c = 0; c < 5; ++c) { p.row.push_back(i); p.col.push_back(c); p.value.push_back(b[c]); }
    }
    LinearSolution fit;
    try { fit = solve_linear(p); } catch (const Error&) { return h; }
    if (!fit.usable) return h;
    const auto& x = fit.x;
    h.mean = x[0];
    h.once = std::hypot(x[1], x[2]);
    h.twice = std::hypot(x[3], x[4]);
    double ss = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double t = angle_deg[i] * M_PI / 180.0;
        const double r = y[i] - (x[0] + x[1] * std::cos(t) + x[2] * std::sin(t) + x[3] * std::cos(2 * t) + x[4] * std::sin(2 * t));
        ss += r * r;
    }
    h.rms = std::sqrt(ss / n);
    h.ok = true;
    return h;
}
std::string harmonics_text(const Harmonics& h, const char* unit) {
    if (!h.ok) return "; harmonic fit unavailable (insufficient support or dependent predictors)";
    return fmt("; against the rotation angle: mean %+.4f %s, %.4f once per turn, %.4f twice per turn (amplitudes), %.4f rms left", h.mean, unit, h.once, h.twice, h.rms);
}

std::vector<std::int64_t> spaced(std::int64_t first, std::int64_t last, int count) {
    std::vector<std::int64_t> out;
    if (last < first) return out;
    if (count < 2 || last == first) {
        out.push_back((first + last) / 2);
        return out;
    }
    for (int i = 0; i < count; ++i) out.push_back(first + (last - first) * i / (count - 1));
    return out;
}

// first rows of bands of n rows spread over [lo, hi)
std::vector<int> row_bands(int lo, int hi, int n) {
    std::vector<int> out;
    const int usable = hi - lo;
    const int count = usable / n;
    if (count < 1) return out;
    const int gap = count > 1 ? (usable - count * n) / (count - 1) : 0;
    for (int i = 0; i < count; ++i) out.push_back(lo + i * (n + gap));
    return out;
}

// every registration, for the table: what was compared, where, what the record stated and what was found
std::vector<std::string>& table_lines() {
    static std::vector<std::string> lines;
    return lines;
}

void progress(const char* what, const Measured& m) {
    const std::string comparison(what);
    if (comparison != "row sense +" && comparison != "row sense -" && comparison != "column sense +" && comparison != "column sense -") {
        ++timing().attempted_views;
        timing().rejected_views += !m.ok;
    }
    table_lines().push_back(fmt("%s\t%lld\t%lld\t%d\t%d\t%d\t%d\t%d\t%d\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.4f\t%d\t%d\t%d", what, static_cast<long long>(m.plan.p0),
                                static_cast<long long>(m.plan.j0), m.plan.n, m.cut.row_a, m.cut.rows.n, m.plan.row_offset, m.cut.col_a, m.cut.col_b, m.plan.stated_dq, m.d[0],
                                m.plan.stated_dv(), m.d[1], m.plan.stated_D, measured_D(m.plan, m.d[0], m.d[1]), m.d[2], m.correlation, m.evaluations, m.ok ? 1 : 0, m.cut.cols.n));
    std::fprintf(stderr, "  %-22s A %6lld B %6lld rows %4d+%-3d cols %4d+%-4d d = (%+.4f %+.4f %+.4f) stated (%+.4f %+.4f) corr %.3f, %d ev%s\n", what,
                 static_cast<long long>(m.plan.p0), static_cast<long long>(m.plan.j0), m.cut.row_a, m.cut.rows.n, m.cut.col_a, m.cut.cols.n, m.d[0], m.d[1],
                 m.d[2], m.plan.stated_dq, m.plan.stated_dv(), m.correlation, m.evaluations, m.ok ? "" : "  (rejected)");
}

std::string joined(const std::vector<std::string>& v) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + v[i];
    return s;
}

}  // namespace

void check_views(const Record& first, Frames& first_frames, const FlatField& first_ff, const Record* second, Frames* second_frames,
                 const FlatField* second_ff, const ViewOptions& opt, Report& rep) {
    if ((opt.columns != 0 && opt.columns < 32) || !std::isfinite(opt.column_fraction) ||
        opt.column_fraction < 0 || opt.column_fraction > 1)
        throw Error("view columns must be 0 or at least 32; column_fraction must be in [0, 1]");
    std::vector<Scan> scans = {{&first, &first_frames, &first_ff}};
    if (second) scans.push_back({second, second_frames, second_ff});
    const Tolerances tol = tolerances(first, opt.eps);
    const double px_mm = pixel_um(first) * 1e-3;
    const int rows = first_frames.rows(), cols = first_frames.cols();
    const Size fsize = pick_size(opt.frames + 24, opt.frames, 8);  // frames per block and their margin
    const int nq = fsize.n, mq = fsize.m;
    const RegistrationOptions ropt;
    table_lines().clear();
    registrars().items.clear();
    registrars().device = opt.device;
    timing() = Timing();

    // projection stretches of a scan: at rest (no height latched) or helical
    const auto stretches = [](const Record& rec, bool helical) {
        std::vector<int> out;
        for (int i : rec.stretches_with(Role::Projection))
            if (rec.stretches[i].latched() && (!rec.stretches[i].z_center.empty()) == helical) out.push_back(i);
        return out;
    };
    const std::vector<int> rest = stretches(first, false), helix = stretches(first, true);
    const std::int64_t per_turn = static_cast<std::int64_t>(first.technique.number("scan/tomo_n"));
    const Size full_cols = pick_size(cols, cols, 8);  // the widest block of columns
    const int full_col0 = (cols - full_cols.n) / 2;

    // ---------------------------------------------------------------- the sense of the rows against the axis stage
    // A turn at rest against the helix, cut with the travel taken towards higher and towards lower rows.
    double row_sign = 0;
    if (!rest.empty() && !helix.empty()) {
        const Stretch& sr = first.stretches[rest[0]];
        const Stretch& sh = first.stretches[helix[0]];
        const std::int64_t hint = rest[0] < helix[0] ? 0 : sh.n_frames - sr.n_frames;
        double corr[2] = {0, 0};
        for (int k = 0; k < 2; ++k) {
            const double sign = k == 0 ? 1.0 : -1.0;
            const Trajectory ta = trajectory(first, sr, sign), tb = trajectory(first, sh, sign);
            const Pairing P{scans[0], scans[0], rest[0], helix[0], &ta, &tb, 0.0, false};
            // a place where the two differ by 100 to 400 rows
            for (std::int64_t p0 : spaced(sr.n_frames / 8, sr.n_frames * 7 / 8, 13)) {
                BlockPlan plan;
                if (!plan_block(ta, tb, p0, nq, mq, hint, 0.0, plan)) continue;
                if (std::abs(plan.row_offset) < 100 || std::abs(plan.row_offset) > 400) continue;
                Cut cut;
                cut.rows = pick_size(rows - std::abs(plan.row_offset) - 4, opt.rows, static_cast<int>(std::ceil(plan.max_row_shift)) + 8);
                cut.cols = full_cols;
                cut.col_a = cut.col_b = full_col0;
                if (!cut.rows.n || !place_rows(rows, plan.row_offset, cut.rows, cut.row_a)) continue;
                Measured m;
                ViewOptions probe = opt;
                probe.columns = 0;  // determine coordinate signs from the full overlap
                measure(P, plan, cut, ropt, probe, m);
                progress(k == 0 ? "row sense +" : "row sense -", m);
                corr[k] = std::max(0.0, m.correlation);
                break;
            }
        }
        row_sign = corr[0] >= corr[1] ? 1.0 : -1.0;
        Finding& f = rep.add("5.1.6", first.label, "sense of the detector's rows against the axis stage");
        f.stated = fmt("data_axes and flipping, without their conventions; %s decreases through the helix", first.translation_z[1].c_str());
        f.found = fmt("a turn at rest registers on the helix with correlation %.3f if a larger %s puts the sample on higher rows, and %.3f if on lower rows: a larger %s = %s rows",
                      corr[0], first.translation_z[1].c_str(), corr[1], first.translation_z[1].c_str(), row_sign > 0 ? "higher" : "lower");
        f.verdict = std::max(corr[0], corr[1]) > 0.3 && std::min(corr[0], corr[1]) < 0.5 * std::max(corr[0], corr[1]) ? "verified" : "unverified";
        f.note = "verdict: the frames determine the sense. It is the sense against the stage; the handedness of the result is not determined";
        f.numbers = {{"row_sign", row_sign}, {"correlation_plus", corr[0]}, {"correlation_minus", corr[1]}};
        if (f.verdict != "verified") row_sign = 0;
    }

    // ---------------------------------------------------------------- opposite views of a turn at rest (first scan)
    std::vector<double> axis_all;
    for (int si : rest) {
        const Stretch& s = first.stretches[si];
        const Trajectory t = trajectory(first, s, 1.0);
        const Pairing P{scans[0], scans[0], si, si, &t, &t, 180.0, true};
        // locate the axis without assuming it: the whole width against itself reversed, searched over half the detector
        double axis0 = NAN;
        {
            BlockPlan plan;
            Cut cut;
            cut.rows = pick_size(rows - 8, opt.rows, 8);
            cut.cols = full_cols;
            cut.col_a = cut.col_b = full_col0;
            if (plan_block(t, t, s.n_frames / 4, nq, mq, per_turn / 2, 180.0, plan) && place_rows(rows, 0, cut.rows, cut.row_a)) {
                const int n[3] = {nq, cut.rows.n, cut.cols.n}, mg[3] = {mq, cut.rows.m, cut.cols.m};
                const int L[3] = {n[0] + 2 * mg[0], n[1] + 2 * mg[1], n[2] + 2 * mg[2]};
                std::vector<std::uint16_t> raw(static_cast<std::size_t>(L[0]) * L[1] * cols);
                std::vector<float> a(static_cast<std::size_t>(n[0]) * n[1] * n[2]), b(static_cast<std::size_t>(L[0]) * L[1] * L[2]);
                std::vector<std::int64_t> frame(L[0]);
                std::vector<std::int32_t> row0(L[0], cut.row_a);
                for (int q = 0; q < n[0]; ++q) frame[q] = plan.p0 + q;
                first_frames.read_rows(si, frame.data(), row0.data(), n[0], n[1], raw.data());
                line_integrals(first_ff, raw.data(), row0.data(), n[0], n[1], cut.col_a, n[2], false, a.data());
                for (int q = 0; q < L[0]; ++q) frame[q] = plan.j0 - mg[0] + q, row0[q] = cut.row_a - mg[1];
                first_frames.read_rows(si, frame.data(), row0.data(), L[0], L[1], raw.data());
                line_integrals(first_ff, raw.data(), row0.data(), L[0], L[1], cut.col_b - mg[2], L[2], true, b.data());
                RegistrationOptions wide = ropt;
                wide.search[2] = n[2] / 2 - 1;
                const Registration r = register_blocks(a.data(), b.data(), n, mg, nullptr, wide, opt.device);
                axis0 = mirror_axis(cut.col_a, cut.col_b, n[2], r.shift[2]);
                std::fprintf(stderr, "  axis search, stretch %s: column %.2f (correlation %.3f)\n", s.entry.c_str(), axis0, r.correlation);
                if (r.correlation < 0.15 || r.at_edge) axis0 = NAN;
            }
        }
        if (!std::isfinite(axis0)) {
            Finding& f = rep.add("5.3.1 4.3", first.label, "opposite views of the turn at rest, stretch " + s.entry);
            f.found = "no mirror axis found inside the detector's field";
            f.verdict = "unverified";
            continue;
        }
        // the widest window symmetric about that column
        const int half = static_cast<int>(std::min(axis0, cols - axis0)) - 2;
        Cut cut;
        cut.cols = pick_size(2 * half, 2 * half, 8);
        cut.col_a = cut.col_b = static_cast<int>(std::lround(axis0 - cut.cols.n / 2.0));
        cut.rows = pick_size(rows, opt.rows, 8);
        std::vector<double> axis, dq_res, dq, D, at;
        const std::vector<int> bands = row_bands(cut.rows.m + 4, rows - cut.rows.m - 4, cut.rows.n);
        std::vector<std::vector<double>> axis_by_band(bands.size()), D_by_band(bands.size());
        for (std::int64_t p0 : spaced(16, s.n_frames - per_turn / 2 - nq - 16, opt.blocks)) {
            BlockPlan plan;
            if (!plan_block(t, t, p0, nq, mq, per_turn / 2, 180.0, plan)) continue;
            for (std::size_t bi = 0; bi < bands.size(); ++bi) {
                cut.row_a = bands[bi];
                Measured m;
                measure(P, plan, cut, ropt, opt, m);
                progress("opposite views", m);
                if (!m.ok) continue;
                const double c = mirror_axis(m.cut.col_a, m.cut.col_b, m.cut.cols.n, m.d[2]);
                axis.push_back(c);
                at.push_back(static_cast<double>(m.plan.p0) / 1000.0);
                axis_by_band[bi].push_back(c);
                dq.push_back(static_cast<double>(m.plan.j0 - m.plan.p0) + m.d[0]);
                dq_res.push_back(m.d[0] - m.plan.stated_dq);
                D.push_back(measured_D(m.plan, m.d[0], m.d[1]) - m.plan.stated_D);
                D_by_band[bi].push_back(D.back());
            }
        }
        if (axis.size() < 3) continue;
        const Spread sa = spread(axis), sq = spread(dq_res), sqa = spread(dq), sD = spread(D);
        axis_all.insert(axis_all.end(), axis.begin(), axis.end());
        const double nominal = cols / 2.0 + positioner(s, first.translation_y[1]) / px_mm;
        const Line axis_trend = fit_line(at, axis, 0, at.size()), D_trend = fit_line(at, D, 0, at.size());
        {
            Finding& f = rep.add("5.3.1", first.label, "column of the rotation axis from opposite views, stretch " + s.entry);
            f.stated = fmt("nominal only: %s = %.4f mm, column %.2f if %s = 0 is the middle column", first.translation_y[1].c_str(), positioner(s, first.translation_y[1]),
                           nominal, first.translation_y[1].c_str());
            std::vector<std::string> per_band;
            for (std::size_t bi = 0; bi < bands.size(); ++bi)
                if (!axis_by_band[bi].empty()) {
                    const Spread sb = spread(axis_by_band[bi]);
                    per_band.push_back(fmt("rows %d-%d: %.3f +- %.3f", bands[bi], bands[bi] + cut.rows.n - 1, sb.mean, sb.sem()));
                    f.numbers.push_back({fmt("axis_at_row_%d", bands[bi] + cut.rows.n / 2), sb.mean});
                }
            f.found = fmt("column %.3f +- %.3f (sd %.3f over %zu blocks of %d frames x %d rows x %d columns; %.3f to %.3f); nominal minus measured %+.2f px; target %.6g px; by row band: %s",
                          sa.mean, sa.sem(), sa.sd, sa.n, nq, cut.rows.n, diagnostic_columns(cut.cols, opt).n, sa.min, sa.max, nominal - sa.mean, tol.axis_px(), joined(per_band).c_str());
            f.found += axis_trend.usable ? fmt("; along the stretch it changes by %+.4f px per 1000 frames", axis_trend.slope) : "; axis trend unavailable: " + axis_trend.message;
            f.numbers.push_back({"axis_px_per_1000_frames", axis_trend.slope});
            std::vector<double> residual(axis.size());
            for (std::size_t i = 0; i < axis.size(); ++i) residual[i] = axis[i] - nominal;
            assess_residuals(f, residual, tol.axis_px(), "px");
            f.note = "verdict: the assumed nominal column against the apparent mirror axis. Row dependence can include axis roll, detector mapping, "
                     "projection geometry and registration weighting; this comparison does not identify their individual contributions";
            f.numbers.insert(f.numbers.begin(), {{"axis_column", sa.mean}, {"axis_sem", sa.sem()}, {"axis_sd", sa.sd}, {"nominal", nominal}});
        }
        {
            Finding& f = rep.add("5.4.1", first.label, "angle over half a turn from opposite views, stretch " + s.entry);
            f.stated = fmt("encoder: the opposite view of frame p is %.4f frames after p", sqa.mean - sq.mean);
            f.found = fmt("registered: %.4f frames after p; measured minus stated %s = %+.5f deg; target %.6g frames (%.6g deg)", sqa.mean, pm(sq, "frames").c_str(),
                          sq.mean * tol.step_deg, tol.angle_frames(), tol.angle_deg());
            assess_residuals(f, dq_res, tol.angle_frames(), "frames");
            f.note = "The comparison assumes a 180-degree conjugate angle for every column. In a divergent beam the conjugate angle varies across columns; "
                     "an apparent residual is not by itself an encoder error. Detector mapping, motion and registration uncertainty also require testing";
            f.numbers = {{"residual_frames", sq.mean}, {"residual_sem", sq.sem()}, {"residual_sd", sq.sd}};
        }
        {
            Finding& f = rep.add("4.4", first.label, "height of opposite views in the turn at rest, stretch " + s.entry);
            f.stated = fmt("%s does not move during the stretch", first.translation_z[1].c_str());
            std::vector<std::string> per_band;
            for (std::size_t bi = 0; bi < bands.size(); ++bi)
                if (!D_by_band[bi].empty()) per_band.push_back(fmt("rows %d-%d: %+.3f", bands[bi], bands[bi] + cut.rows.n - 1, spread(D_by_band[bi]).mean));
            f.found = fmt("opposite views are displaced along the axis by %s; target %.6g rows; by row band: %s; along the stretch it changes by %+.4f rows per 1000 frames (%+.3f at the first pair, %+.3f at the last)",
                          pm(sD, "rows").c_str(), tol.rows(), joined(per_band).c_str(), D_trend.slope, D_trend.intercept + D_trend.slope * at.front(), D_trend.intercept + D_trend.slope * at.back());
            if (!D_trend.usable) f.found += "; height trend unavailable: " + D_trend.message;
            f.numbers.push_back({"rows_per_1000_frames", D_trend.slope});
            assess_residuals(f, D, tol.rows(), "rows");
            f.note = "an axis rolled against the columns displaces opposite views by rows in proportion to the distance from the axis, with both signs inside one block: this mean does not show it";
            f.numbers = {{"rows", sD.mean}, {"rows_sem", sD.sem()}};
        }
    }
    const double axis_first = axis_all.empty() ? NAN : spread(axis_all).mean;
    if (!axis_all.empty()) {
        Finding& f = rep.add("4.3 6.3.1", first.label, "which scan holds both sides of the axis");
        f.stated = "two scans that differ in the lateral stage";
        f.found = fmt("opposite views of this scan mirror about column %.2f, %.0f columns inside the field of %d: lines within %.0f px of the axis are recorded from both sides",
                      axis_first, std::min(axis_first, cols - axis_first), cols, std::min(axis_first, cols - axis_first));
        f.verdict = "verified";
        f.note = "verdict: this is the scan with the axis inside the field";
    }

    // ---------------------------------------------------------------- a turn at rest against the helix; the helix a turn apart
    for (std::size_t k = 0; k < scans.size() && row_sign != 0; ++k) {
        const Scan& S = scans[k];
        const Record& rec = *S.rec;
        const std::vector<int> rest_k = stretches(rec, false), helix_k = stretches(rec, true);
        for (int hi : helix_k) {
            const Stretch& sh = rec.stretches[hi];
            const Trajectory th = trajectory(rec, sh, row_sign);
            for (int ri : rest_k) {
                const Stretch& sr = rec.stretches[ri];
                const Trajectory tr = trajectory(rec, sr, row_sign);
                const Pairing P{S, S, ri, hi, &tr, &th, 0.0, false};
                const std::int64_t hint = ri < hi ? 0 : sh.n_frames - sr.n_frames;
                std::vector<double> Ds, Dres, qres, ures;
                for (std::int64_t p0 : spaced(16, sr.n_frames - nq - 16, 3 * opt.blocks)) {
                    BlockPlan plan;
                    if (!plan_block(tr, th, p0, nq, mq, hint, 0.0, plan)) continue;
                    const int N = plan.row_offset;
                    // B's content must not sit where a pattern fixed to the detector would register (near zero rows)
                    if (std::abs(N) < 2 * ropt.search[1]) continue;
                    Cut cut;
                    cut.rows = pick_size(rows - std::abs(N) - 4, opt.rows, static_cast<int>(std::ceil(plan.max_row_shift)) + 8);
                    cut.cols = full_cols;
                    cut.col_a = cut.col_b = full_col0;
                    if (cut.rows.n < 64 || !place_rows(rows, N, cut.rows, cut.row_a)) continue;
                    Measured m;
                    measure(P, plan, cut, ropt, opt, m);
                    progress("rest against helix", m);
                    if (!m.ok) continue;
                    Ds.push_back(m.plan.stated_D);
                    Dres.push_back(measured_D(m.plan, m.d[0], m.d[1]) - m.plan.stated_D);
                    qres.push_back(m.d[0] - m.plan.stated_dq);
                    ures.push_back(m.d[2]);
                }
                if (Ds.size() < 4) continue;
                const Line lD = fit_line(Ds, Dres, 0, Ds.size()), lq = fit_line(Ds, qres, 0, Ds.size()), lu = fit_line(Ds, ures, 0, Ds.size());
                const Spread sD = spread(Dres), sq = spread(qres), su = spread(ures), sDs = spread(Ds);
                const double far = std::fabs(sDs.max) > std::fabs(sDs.min) ? sDs.max : sDs.min;  // the largest displacement used
                const std::string which = "turn at rest " + sr.entry + " against helix " + sh.entry;
                {
                    Finding& f = rep.add("5.4.3 5.4.5 5.4.6", rec.label, "height of the helix against the " + which);
                    f.stated = fmt("%s at rest and the latched heights of the helix, in rows through %.4g um: the helix's content lies %.1f to %.1f rows from the rest turn's in the %zu blocks used",
                                   rec.translation_z[1].c_str(), px_mm * 1e3, sDs.min, sDs.max, Ds.size());
                    f.found = fmt("measured minus stated displacement: %+.4f rows on average, %+.3f to %+.3f; as a line in the displacement: %+.4f rows at zero, %+.3e rows per row "
                                  "(%+.3f rows at %.0f rows; rms about the line %.4f rows); targets %.6g rows and %.6g",
                                  sD.mean, sD.min, sD.max, lD.intercept, lD.slope, lD.intercept + lD.slope * far, far, lD.rms, tol.rows(), tol.travel_scale());
                    assess_residuals(f, Dres, tol.rows(), "rows");
                    if (std::fabs(lD.slope) > tol.travel_scale() || std::fabs(lD.intercept) > tol.rows()) f.verdict = "fails";
                    f.note = "Under a uniform spatial mapping, the slope estimates the error of (stage scale / vertical pixel pitch) and the intercept estimates "
                             "a boundary-height offset. Spatial mapping, exposure timing and registration bias can also enter; this fit does not uniquely identify "
                             "a stage-scale error or a wrong static height readback";
                    if (!lD.usable) { f.found += "; line fit unavailable: " + lD.message; f.verdict = "unverified"; }
                    f.numbers = {{"offset_rows", lD.intercept}, {"scale_error", lD.slope}, {"rms_rows", lD.rms}, {"mean_rows", sD.mean}, {"min_rows", sD.min}, {"max_rows", sD.max}};
                }
                {
                    Finding& f = rep.add("5.4.1 5.4.6 5.4.9", rec.label, "angle of the helix against the " + which);
                    f.stated = "the encoder angles of both stretches";
                    f.found = fmt("measured minus stated: %s = %+.5f deg, %+.3f to %+.3f frames; as a line in the row displacement: %+.4f frames at zero, %+.3e frames per row "
                                  "(%+.3f frames at %.0f rows; rms %.4f); target %.6g frames",
                                  pm(sq, "frames").c_str(), sq.mean * tol.step_deg, sq.min, sq.max, lq.intercept, lq.slope, lq.intercept + lq.slope * far, far, lq.rms, tol.angle_frames());
                    assess_residuals(f, qres, tol.angle_frames(), "frames");
                    f.note = fmt("the views of a block lie on detector rows a displacement apart. If the detector's rows were exposed one after another, the angle would differ in "
                                 "proportion to that displacement: the slope is %+.2f us per row, %+.2f ms over the %d rows read out",
                                 lq.slope * sh.timer_period[0] * 1e6, lq.slope * sh.timer_period[0] * 1e3 * rows, rows);
                    if (!lq.usable) { f.found += "; line fit unavailable: " + lq.message; f.verdict = "unverified"; }
                    f.numbers = {{"offset_frames", lq.intercept}, {"frames_per_row", lq.slope}, {"mean_frames", sq.mean}, {"rms_frames", lq.rms}};
                }
                {
                    Finding& f = rep.add("5.4.7", rec.label, "lateral position of the helix against the " + which);
                    f.stated = fmt("not stated; %s is the same in both stretches", rec.translation_y[1].c_str());
                    f.found = fmt("columns: %s, %+.3f to %+.3f; as a line in the row displacement: %+.4f px at zero, %+.3e px per row (%+.3f px at %.0f rows; rms %.4f)",
                                  pm(su, "px").c_str(), su.min, su.max, lu.intercept, lu.slope, lu.intercept + lu.slope * far, far, lu.rms);
                    f.note = fmt("a slope is the travel's direction against the detector's columns (target %.6g rad) or the detector's distortion between the two bands", tol.eps / 2 / tol.H);
                    f.numbers = {{"offset_px", lu.intercept}, {"px_per_row", lu.slope}};
                    assess_residuals(f, ures, tol.eps, "px");
                    if (!lu.usable) { f.found += "; line fit unavailable: " + lu.message; f.verdict = "unverified"; }
                    f.note += "; residuals are against zero unmodelled lateral motion, not a measured trajectory";
                }
            }
            // the helix a turn apart
            {
                const Pairing P{S, S, hi, hi, &th, &th, 0.0, false};
                std::vector<double> Pm, Ps, Dres, qres, ures, ang;
                for (std::int64_t p0 : spaced(16, sh.n_frames - per_turn - nq - 64, 2 * opt.blocks)) {
                    BlockPlan plan;
                    if (!plan_block(th, th, p0, nq, mq, per_turn, 0.0, plan)) continue;
                    const int N = plan.row_offset;
                    Cut cut;
                    cut.rows = pick_size(rows - std::abs(N) - 4, opt.rows, static_cast<int>(std::ceil(plan.max_row_shift)) + 6);
                    cut.cols = full_cols;
                    cut.col_a = cut.col_b = full_col0;
                    if (cut.rows.n < 64 || !place_rows(rows, N, cut.rows, cut.row_a)) continue;
                    Measured m;
                    measure(P, plan, cut, ropt, opt, m);
                    progress("helix a turn apart", m);
                    if (!m.ok) continue;
                    const double Dm = measured_D(m.plan, m.d[0], m.d[1]);
                    Pm.push_back(std::fabs(Dm));
                    Ps.push_back(std::fabs(m.plan.stated_D));
                    Dres.push_back(std::fabs(Dm) - std::fabs(m.plan.stated_D));
                    qres.push_back(m.d[0] - m.plan.stated_dq);
                    ures.push_back(m.d[2]);
                    ang.push_back(th.angle_deg[m.plan.p0 + nq / 2]);
                }
                if (Pm.size() >= 3) {
                    const Harmonics hD = harmonics(ang, Dres), hq = harmonics(ang, qres), hu = harmonics(ang, ures);
                    const Spread sP = spread(Pm), sPs = spread(Ps), sD = spread(Dres), sq = spread(qres), su = spread(ures);
                    const double tolP = tol.eps * sP.mean / tol.H;
                    {
                        Finding& f = rep.add("5.4.5 5.1.2", rec.label, "travel along the axis per turn in rows, helix " + sh.entry + " a turn apart");
                        f.stated = fmt("encoder against encoder through %.4g um: %.4f rows per turn", px_mm * 1e3, sPs.mean);
                        f.found = fmt("views a turn apart are displaced by %.4f +- %.4f rows (sd %.4f over %zu blocks of %d frames); measured minus stated %s; target %.6g rows per turn",
                                      sP.mean, sP.sem(), sP.sd, sP.n, nq, pm(sD, "rows").c_str(), tolP);
                        f.found += harmonics_text(hD, "rows");
                        assess_residuals(f, Dres, tolP, "rows/turn");
                        f.note = fmt("measured is (stage scale / pixel pitch along the columns): the difference is %+.2e of the travel, i.e. a pitch of %.4f um if the stage's scale is exact. "
                                     "The two views lie at the top and the bottom of the detector, so distortion (5.1.4) enters",
                                     sD.mean / sPs.mean, px_mm * 1e3 * sPs.mean / sP.mean);
                        f.numbers = {{"rows_per_turn", sP.mean}, {"rows_per_turn_sem", sP.sem()}, {"stated_rows_per_turn", sPs.mean}, {"measured_minus_stated", sD.mean}};
                    }
                    {
                        Finding& f = rep.add("5.4.1 5.4.9", rec.label, "angle over a whole turn, helix " + sh.entry + " a turn apart");
                        f.stated = "encoder: the frame one turn after frame p";
                        f.found = fmt("measured minus stated %s = %+.5f deg; target %.6g frames (%.6g deg)", pm(sq, "frames").c_str(), sq.mean * tol.step_deg, tol.angle_frames(), tol.angle_deg());
                        f.found += harmonics_text(hq, "frames");
                        assess_residuals(f, qres, tol.angle_frames(), "frames");
                        f.note = fmt("the two views lie %.0f rows apart on the detector: rows exposed one after another would show here as %+.2f us per row, %+.2f ms over the %d rows read out",
                                     sP.mean, sq.mean / sP.mean * sh.timer_period[0] * 1e6, sq.mean / sP.mean * sh.timer_period[0] * 1e3 * rows, rows);
                        f.numbers = {{"residual_frames", sq.mean}, {"residual_sem", sq.sem()}};
                    }
                    {
                        Finding& f = rep.add("5.4.7", rec.label, "lateral position a turn apart, helix " + sh.entry);
                        f.stated = "not stated";
                        f.found = fmt("columns: %s over %.0f rows of travel = %+.2e px per row", pm(su, "px").c_str(), sP.mean, su.mean / sP.mean);
                        f.found += harmonics_text(hu, "px");
                        f.numbers = {{"px", su.mean}, {"px_sem", su.sem()}};
                        assess_residuals(f, ures, tol.eps, "px");
                        f.note = "Residuals against zero unmodelled lateral motion. Angle-dependent displacement can include sample motion and detector distortion; its cause is not identified.";
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------- the second scan against the first
    if (second) {
        const std::vector<int> pa = first.stretches_with(Role::Projection), pb = second->stretches_with(Role::Projection);
        const std::string y = first.translation_y[1];
        const std::string both = first.label + "+" + second->label;
        const double sign_rows = row_sign != 0 ? row_sign : 1.0;
        double col_sign = 0;
        std::vector<double> move_all;
        for (std::size_t k = 0; k < pa.size() && k < pb.size(); ++k) {
            const Stretch& sa = first.stretches[pa[k]];
            const Stretch& sb = second->stretches[pb[k]];
            if (!sa.latched() || !sb.latched()) continue;
            const double move_px = (positioner(sb, y) - positioner(sa, y)) / px_mm;
            const int M = static_cast<int>(std::lround(std::fabs(move_px)));
            const Trajectory ta = trajectory(first, sa, sign_rows), tb = trajectory(*second, sb, sign_rows);
            const Pairing P{scans[0], scans[1], pa[k], pb[k], &ta, &tb, 0.0, false};
            Cut cut;
            cut.cols = pick_size(cols - M, cols - M, 8);
            cut.rows = pick_size(rows, opt.rows, 8);
            const auto set_columns = [&](double sign) {
                cut.col_a = sign > 0 ? (cols - M - cut.cols.n) / 2 : M + (cols - M - cut.cols.n) / 2;
                cut.col_b = sign > 0 ? cut.col_a + M : cut.col_a - M;
            };
            if (col_sign == 0) {
                // which way the sample moved on the detector: both, once
                double corr[2] = {0, 0};
                for (int c = 0; c < 2; ++c) {
                    BlockPlan plan;
                    if (!plan_block(ta, tb, sa.n_frames / 3, nq, mq, 0, 0.0, plan)) continue;
                    set_columns(c == 0 ? 1.0 : -1.0);
                    if (!place_rows(rows, plan.row_offset, cut.rows, cut.row_a)) continue;
                    Measured m;
                    ViewOptions probe = opt;
                    probe.columns = 0;  // determine coordinate signs from the full overlap
                    measure(P, plan, cut, ropt, probe, m);
                    progress(c == 0 ? "column sense +" : "column sense -", m);
                    corr[c] = std::max(0.0, m.correlation);
                }
                col_sign = corr[0] >= corr[1] ? 1.0 : -1.0;
                Finding& f = rep.add("5.1.6", both, "sense of the detector's columns against the lateral stage");
                f.stated = fmt("%s %.4f -> %.4f mm", y.c_str(), positioner(sa, y), positioner(sb, y));
                f.found = fmt("the second scan registers on the first with correlation %.3f if its content lies %d columns higher, and %.3f if lower: a larger %s = %s columns",
                              corr[0], M, corr[1], y.c_str(), (col_sign > 0) == (move_px > 0) ? "higher" : "lower");
                f.verdict = std::max(corr[0], corr[1]) > 0.3 && std::min(corr[0], corr[1]) < 0.5 * std::max(corr[0], corr[1]) ? "verified" : "unverified";
                f.note = "verdict: the frames determine the sense";
                f.numbers = {{"column_sign", col_sign}, {"correlation_higher", corr[0]}, {"correlation_lower", corr[1]}};
                if (f.verdict != "verified") break;
            }
            set_columns(col_sign);
            std::vector<double> move, Dm, Dres, qres, ang;
            const std::vector<int> bands = row_bands(cut.rows.m + 6, rows - cut.rows.m - 6, cut.rows.n);
            std::vector<std::vector<double>> move_by_band(bands.size()), D_by_band(bands.size());
            for (std::int64_t p0 : spaced(16, sa.n_frames - nq - 16, opt.blocks)) {
                BlockPlan plan;
                if (!plan_block(ta, tb, p0, nq, mq, 0, 0.0, plan)) continue;
                for (std::size_t bi = 0; bi < bands.size(); ++bi) {
                    cut.row_a = bands[bi];
                    Measured m;
                    measure(P, plan, cut, ropt, opt, m);
                    progress("second scan on first", m);
                    if (!m.ok) continue;
                    const double mv = (cut.col_b - cut.col_a) + m.d[2];
                    move.push_back(mv);
                    move_by_band[bi].push_back(mv);
                    const double D = measured_D(m.plan, m.d[0], m.d[1]);
                    Dm.push_back(D);
                    D_by_band[bi].push_back(D - m.plan.stated_D);
                    Dres.push_back(D - m.plan.stated_D);
                    qres.push_back(m.d[0] - m.plan.stated_dq);
                    ang.push_back(ta.angle_deg[m.plan.p0 + nq / 2]);
                }
            }
            if (move.size() < 3) continue;
            std::vector<double> move_res(move.size());
            for (std::size_t i = 0; i < move.size(); ++i) move_res[i] = std::fabs(move[i]) - std::fabs(move_px);
            const Harmonics hm = harmonics(ang, move_res), hD = harmonics(ang, Dres), hq = harmonics(ang, qres);
            const Spread sm = spread(move), sD = spread(Dres), sDm = spread(Dm), sq = spread(qres);
            move_all.insert(move_all.end(), move.begin(), move.end());
            std::vector<double> res(move.size());
            for (std::size_t i = 0; i < move.size(); ++i) res[i] = std::fabs(move[i]) - std::fabs(move_px);
            const Spread sr = spread(res);
            std::vector<std::string> bands_move, bands_D;
            for (std::size_t bi = 0; bi < bands.size(); ++bi)
                if (!move_by_band[bi].empty()) {
                    bands_move.push_back(fmt("rows %d-%d: %.3f", bands[bi], bands[bi] + cut.rows.n - 1, std::fabs(spread(move_by_band[bi]).mean)));
                    bands_D.push_back(fmt("rows %d-%d: %+.3f", bands[bi], bands[bi] + cut.rows.n - 1, spread(D_by_band[bi]).mean));
                }
            {
                Finding& f = rep.add("6.1.1 5.1.1", both, "lateral displacement of the second scan's content, stretch " + sa.entry);
                f.stated = fmt("%s move %.4f mm = %.3f px at %.4g um", y.c_str(), positioner(sb, y) - positioner(sa, y), std::fabs(move_px), px_mm * 1e3);
                f.found = fmt("%.4f +- %.4f columns (sd %.4f over %zu blocks of %d frames x %d rows x %d columns); measured minus stated %+.4f px; target %.6g px; by row band: %s",
                              std::fabs(sm.mean), sm.sem(), sm.sd, sm.n, nq, cut.rows.n, diagnostic_columns(cut.cols, opt).n, sr.mean, tol.eps, joined(bands_move).c_str());
                f.found += harmonics_text(hm, "px");
                assess_residuals(f, res, tol.eps, "px");
                f.note = fmt("measured is (lateral stage scale / pixel pitch along the rows): the difference is %+.2e of the move, i.e. a pitch of %.4f um if the stage's scale is exact. "
                             "The same rays fall on detector columns a move apart, so distortion (5.1.4, 6.3.3) enters; a change with the row band is a roll of one scan against the other (6.1.4) or distortion",
                             sr.mean / std::fabs(move_px), px_mm * 1e3 * std::fabs(move_px) / std::fabs(sm.mean));
                f.numbers = {{"move_columns", sm.mean}, {"move_sem", sm.sem()}, {"move_sd", sm.sd}, {"stated_px", std::fabs(move_px)}, {"measured_minus_stated", sr.mean}};
            }
            {
                Finding& f = rep.add("6.1.2 6.2.3", both, "displacement along the axis of the second scan's content, stretch " + sa.entry);
                f.stated = fmt("the two scans' heights: second minus first %+.3f rows at %.4g um (counted towards higher rows)", sDm.mean - sD.mean, px_mm * 1e3);
                f.found = fmt("%+.4f rows; measured minus stated %s; target %.6g rows; by row band: %s", sDm.mean, pm(sD, "rows").c_str(), tol.rows(), joined(bands_D).c_str());
                f.found += harmonics_text(hD, "rows");
                assess_residuals(f, Dres, tol.rows(), "rows");
                f.numbers = {{"rows", sDm.mean}, {"measured_minus_stated", sD.mean}, {"sem", sD.sem()}};
            }
            {
                Finding& f = rep.add("6.2.1 6.1.6", both, "angle of the second scan against the first at equal projection index, stretch " + sa.entry);
                f.stated = "the encoder angles of both scans";
                f.found = fmt("measured minus stated %s = %+.5f deg (%.3f px at R); target %.6g frames (%.6g deg)", pm(sq, "frames").c_str(), sq.mean * tol.step_deg,
                              std::fabs(sq.mean * tol.step_deg) / 57.29577951308232 * tol.R, tol.angle_frames(), tol.angle_deg());
                f.found += harmonics_text(hq, "frames");
                assess_residuals(f, qres, tol.angle_frames(), "frames");
                {
                    const double ss = first.technique.number("scan/source_sample_distance", NAN);
                    const double fan = std::fabs(positioner(sb, y) - positioner(sa, y)) / ss;  // rad
                    f.note = fmt("An apparent angle difference can include sample rotation, detector geometry, timing and registration bias; it does not identify the cause. One geometric contribution is: moved %.4f mm sideways in a beam "
                                 "from a source %.0f mm away, the sample is crossed by rays turned by %.2e rad = %.3f frames (the record does not give the sign). Angle dependence alone does not distinguish motion from distortion: the same rays fall on detector columns a move apart, and a difference of scale between those columns (5.1.4) acts on a sample that is not round like a rotation that goes twice per turn",
                                 std::fabs(positioner(sb, y) - positioner(sa, y)), ss, fan, fan * 57.29577951308232 / tol.step_deg);
                }
                f.numbers = {{"residual_frames", sq.mean}, {"residual_sem", sq.sem()}};
            }
        }
        if (!move_all.empty() && std::isfinite(axis_first)) {
            const Spread sm = spread(move_all);
            const Stretch& sb = second->stretches[pb.at(0)];
            const double nominal = cols / 2.0 + positioner(sb, y) / px_mm;
            Finding& f = rep.add("6.3.2 5.3.1", second->label, "column of the second scan's rotation axis");
            f.stated = fmt("nominal only: column %.2f if %s = 0 is the middle column", nominal, y.c_str());
            f.found = fmt("first scan's axis %.3f %+.3f columns of measured displacement = column %.3f (the last column is %d); nominal minus measured %+.2f px", axis_first, sm.mean,
                          axis_first + sm.mean, cols - 1, nominal - (axis_first + sm.mean));
            std::vector<double> residual(move_all.size());
            for (std::size_t i = 0; i < move_all.size(); ++i) residual[i] = axis_first + move_all[i] - nominal;
            assess_residuals(f, residual, tol.axis_px(), "px");
            f.note = "the second scan has no opposite views of its own; this value inherits the errors of both measurements and assumes the axis moved exactly as the sample's content did";
            f.numbers = {{"axis_column", axis_first + sm.mean}, {"nominal", nominal}};
        }
    }
    {
        const Timing& t = timing();
        Finding& f = rep.add("4.1 5 6", first.label + (second ? "+" + second->label : ""), "coverage of sampled view comparisons");
        f.verdict = "unverified";
        f.found = fmt("%d attempted view comparisons, %d rejected registrations (direction probes excluded)", t.attempted_views, t.rejected_views);
        f.note = "Accepted blocks sample only parts of the acquisition; below-target blocks cannot certify every projection or pixel. Rejected registrations and untested regions remain unaccounted for. No corrected geometry or held-out reconstruction has been validated";
        f.numbers = {{"attempted", static_cast<double>(t.attempted_views)}, {"rejected", static_cast<double>(t.rejected_views)}};
    }
    registrars().items.clear();  // their work memory
    {
        const Timing& t = timing();
        std::fprintf(stderr, "  views on %s: %d blocks, %d registrations; reading %.1f s, line integrals %.1f s, registration %.1f s (%.3f s each)\n",
                     device_name(opt.device), t.blocks, t.registrations, t.read, t.integrals, t.registration, t.registrations ? t.registration / t.registrations : 0.0);
    }
    if (!opt.table.empty()) {
        std::FILE* out = std::fopen(opt.table.c_str(), "w");
        if (!out) throw Error("cannot write " + opt.table);
        std::fputs("comparison\tframe_a\tframe_b\tframes\trow_a\trows\trow_offset\tcol_a\tcol_b\tstated_dq\tdq\tstated_dv\tdv\tstated_D\tD\tdu\tcorrelation\tevaluations\tok\tcolumns\n", out);
        for (const std::string& line : table_lines()) std::fprintf(out, "%s\n", line.c_str());
        std::fclose(out);
    }
}

}  // namespace helirec::calibration
