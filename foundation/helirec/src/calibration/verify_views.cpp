// The record against views that recorded the same lines: opposite views of a turn at rest, a turn at rest
// against the helix, the helix a turn apart, and one scan against the other.
//
// Every block is cut from raw frames at the integer rows, columns and frames the record implies (view_plan.hpp),
// converted to line integrals with the scan's own dark and mean flat (frames.hpp), and registered
// (registration.hpp). What is reported is the registration against what the record states.
#include <cmath>
#include <cstdio>

#include "calibration/util.hpp"
#include "calibration/view_plan.hpp"
#include "helirec/calibration/registration.hpp"
#include "helirec/calibration/verify.hpp"

namespace helirec::calibration {

namespace {

struct Scan {
    const Record* rec;
    Frames* frames;
    const FlatField* ff;
};

// largest even m <= n whose prime factors are 2, 3, 5, 7
int good_size(int n) {
    for (int m = n - (n % 2); m >= 8; m -= 2) {
        int r = m;
        for (int p : {2, 3, 5, 7})
            while (r % p == 0) r /= p;
        if (r == 1) return m;
    }
    return std::max(2, n - (n % 2));
}

Trajectory trajectory(const Record& rec, const Stretch& s, double row_sign) {
    Trajectory t;
    t.angle_deg = stated_angle(s);
    const auto z = stated_height(rec, s);
    const double px_mm = pixel_um(rec) * 1e-3;
    t.rows.resize(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) t.rows[i] = row_sign * z[i] / px_mm;
    return t;
}

struct Cut {
    int row_a = 0, n_rows = 0, col_a = 0, col_b = 0, n_cols = 0;
    bool mirror = false;
};

struct Measured {
    BlockPlan plan;
    Cut cut;
    double d[3] = {0, 0, 0};  // registration, including the whole samples the blocks were re-cut by
    double correlation = 0;
    bool ok = false;
};

// cut both blocks, register, and re-cut B by whole samples until the remaining displacement is below one sample
bool measure(const Scan& A, int sa, const Scan& B, int sb, const BlockPlan& plan, const Cut& cut, const RegistrationOptions& ropt, Measured& m) {
    const int n = plan.n, rows = A.frames->rows(), cols = A.frames->cols();
    const std::int64_t nb = B.rec->stretches[sb].n_frames;
    m.plan = plan;
    m.cut = cut;
    m.ok = false;
    if (cut.row_a < 0 || cut.row_a + cut.n_rows > rows || cut.col_a < 0 || cut.col_a + cut.n_cols > cols) return false;
    std::vector<std::uint16_t> raw(static_cast<std::size_t>(n) * cut.n_rows * cols);
    std::vector<float> a(static_cast<std::size_t>(n) * cut.n_rows * cut.n_cols), b(a.size());
    std::vector<std::int64_t> frame(n);
    std::vector<std::int32_t> row0(n);
    for (int q = 0; q < n; ++q) frame[q] = plan.p0 + q, row0[q] = cut.row_a;
    A.frames->read_rows(sa, frame.data(), row0.data(), n, cut.n_rows, raw.data());
    line_integrals(*A.ff, raw.data(), row0.data(), n, cut.n_rows, cut.col_a, cut.n_cols, false, a.data());
    int extra[3] = {0, 0, 0};
    for (int pass = 0; pass < 5; ++pass) {
        const int col_b = cut.mirror ? cut.col_b - extra[2] : cut.col_b + extra[2];
        if (col_b < 0 || col_b + cut.n_cols > cols) return false;
        for (int q = 0; q < n; ++q) {
            frame[q] = plan.j0 + extra[0] + q;
            row0[q] = cut.row_a + plan.row_offset[q] + extra[1];
            if (frame[q] < 0 || frame[q] >= nb || row0[q] < 0 || row0[q] + cut.n_rows > rows) return false;
        }
        B.frames->read_rows(sb, frame.data(), row0.data(), n, cut.n_rows, raw.data());
        line_integrals(*B.ff, raw.data(), row0.data(), n, cut.n_rows, col_b, cut.n_cols, cut.mirror, b.data());
        const Registration r = register_blocks(a.data(), b.data(), n, cut.n_rows, cut.n_cols, ropt);
        m.correlation = r.correlation;
        bool recut = false;
        for (int j = 0; j < 3; ++j) {
            m.d[j] = extra[j] + r.shift[j];
            if (std::fabs(r.shift[j]) > 0.75) extra[j] += static_cast<int>(std::lround(r.shift[j])), recut = true;
        }
        if (!recut) {
            m.ok = r.converged && !r.at_edge && r.correlation > 0.15;
            return m.ok;
        }
    }
    return false;
}

std::string pm(const Spread& s, const char* unit, const char* format = "%+.4f") {
    const std::string f = std::string(format) + " +- %.4f %s (sd %.4f, n %zu)";
    return fmt(f.c_str(), s.mean, s.n > 1 ? s.sem() : NAN, unit, s.sd, s.n);
}

// holds: the mean is within the tolerance and known better than it. fails: the mean is outside by more than
// three standard errors. open: neither.
std::string verdict(const Spread& residual, double tolerance) {
    if (residual.n < 3) return "open";
    const double sem = residual.sem(), m = std::fabs(residual.mean);
    if (m <= tolerance && sem <= tolerance) return "holds";
    if (m - 3 * sem > tolerance) return "fails";
    return "open";
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

// row bands of n_rows rows spread over the detector, keeping `margin` rows off both ends
std::vector<int> row_bands(int rows, int n_rows, int margin) {
    std::vector<int> out;
    const int usable = rows - 2 * margin;
    const int count = std::max(1, usable / n_rows);
    const int gap = count > 1 ? (usable - count * n_rows) / (count - 1) : 0;
    for (int i = 0; i < count; ++i) out.push_back(margin + i * (n_rows + gap));
    return out;
}

// first row of A's block so that both it and B's block, D rows higher, lie centred in the rows both can use
int centred_row(int rows, int n_rows, int D) { return D > 0 ? (rows - n_rows - D) / 2 : -D + (rows + D - n_rows) / 2; }

void progress(const char* what, const Measured& m) {
    std::fprintf(stderr, "  %-28s A %6lld  B %6lld  rows %4d+%-4d  d = (%+.4f, %+.4f, %+.4f)  stated (%+.4f, %+.4f)  corr %.3f%s\n", what,
                 static_cast<long long>(m.plan.p0), static_cast<long long>(m.plan.j0), m.cut.row_a, m.cut.n_rows, m.d[0], m.d[1], m.d[2], m.plan.stated_dq,
                 m.plan.stated_dv, m.correlation, m.ok ? "" : "  (rejected)");
}

}  // namespace

void check_views(const Record& first, Frames& first_frames, const FlatField& first_ff, const Record* second, Frames* second_frames,
                 const FlatField* second_ff, const ViewOptions& opt, Report& rep) {
    std::vector<Scan> scans = {{&first, &first_frames, &first_ff}};
    if (second) scans.push_back({second, second_frames, second_ff});
    const Tolerances tol = tolerances(first, opt.eps);
    const double px_mm = pixel_um(first) * 1e-3;
    const int rows = first_frames.rows(), cols = first_frames.cols();
    const int nq = opt.frames;
    const RegistrationOptions ropt;

    // projection stretches of a scan: at rest (no height latched) or helical
    const auto stretches = [](const Record& rec, bool helical) {
        std::vector<int> out;
        for (int i : rec.stretches_with(Role::Projection))
            if (rec.stretches[i].latched() && (!rec.stretches[i].z_center.empty()) == helical) out.push_back(i);
        return out;
    };
    const std::vector<int> rest = stretches(first, false), helix = stretches(first, true);
    const std::int64_t per_turn = static_cast<std::int64_t>(first.technique.number("scan/tomo_n"));

    // ---------------------------------------------------------------- the sense of the rows against the axis stage
    // A turn at rest against the helix, cut with the travel taken towards higher and towards lower rows.
    double row_sign = 0;
    if (!rest.empty() && !helix.empty()) {
        const Stretch& sr = first.stretches[rest[0]];
        const Stretch& sh = first.stretches[helix[0]];
        const std::int64_t hint = rest[0] < helix[0] ? 0 : sh.n_frames - sr.n_frames;
        double corr[2] = {0, 0};
        std::string detail;
        // a place where the two differ by 100 to 400 rows
        for (int k = 0; k < 2; ++k) {
            const double sign = k == 0 ? 1.0 : -1.0;
            const Trajectory ta = trajectory(first, sr, sign), tb = trajectory(first, sh, sign);
            for (std::int64_t p0 : spaced(sr.n_frames / 8, sr.n_frames * 7 / 8, 9)) {
                BlockPlan plan;
                if (!plan_block(ta, tb, p0, nq, hint, 0.0, plan)) continue;
                const double D = plan.stated_D;
                if (std::fabs(D) < 100 || std::fabs(D) > 400) continue;
                Cut cut;
                cut.n_rows = good_size(std::min(opt.rows, rows - static_cast<int>(std::fabs(D)) - 64));
                cut.row_a = centred_row(rows, cut.n_rows, static_cast<int>(std::lround(D)));
                cut.n_cols = good_size(cols - 8);
                cut.col_a = cut.col_b = (cols - cut.n_cols) / 2;
                Measured m;
                measure(scans[0], rest[0], scans[0], helix[0], plan, cut, ropt, m);
                progress(k == 0 ? "row sense +" : "row sense -", m);
                corr[k] = m.ok ? m.correlation : 0.0;
                break;
            }
        }
        row_sign = corr[0] >= corr[1] ? 1.0 : -1.0;
        Finding& f = rep.add("5.1.6", first.label, "sense of the detector's rows against the axis stage");
        f.stated = fmt("data_axes and flipping, without their conventions; %s decreases through the helix", first.translation_z[1].c_str());
        f.found = fmt("a turn at rest against the helix registers with correlation %.3f if a larger %s puts the sample on higher rows, %.3f if on lower rows: larger %s = %s rows",
                      corr[0], first.translation_z[1].c_str(), corr[1], first.translation_z[1].c_str(), row_sign > 0 ? "higher" : "lower");
        f.verdict = std::max(corr[0], corr[1]) > 0.3 && std::min(corr[0], corr[1]) < 0.5 * std::max(corr[0], corr[1]) ? "holds" : "open";
        f.note = "verdict: the sense is determined by the frames. It is the sense against the stage; the handedness of the result is not determined";
        f.numbers = {{"row_sign", row_sign}, {"correlation_plus", corr[0]}, {"correlation_minus", corr[1]}};
        if (f.verdict != "holds") row_sign = 0;
    }

    // ---------------------------------------------------------------- opposite views of a turn at rest (first scan)
    std::vector<double> axis_all;
    for (int si : rest) {
        const Stretch& s = first.stretches[si];
        const Trajectory t = trajectory(first, s, 1.0);
        // locate the axis without assuming it: the whole width against itself reversed, searched over half the detector
        double axis0 = NAN;
        {
            BlockPlan plan;
            if (plan_block(t, t, s.n_frames / 4, nq, per_turn / 2, 180.0, plan)) {
                Cut cut;
                cut.mirror = true;
                cut.n_rows = good_size(std::min(opt.rows, rows - 64));
                cut.row_a = (rows - cut.n_rows) / 2;
                cut.n_cols = good_size(cols - 8);
                cut.col_a = cut.col_b = (cols - cut.n_cols) / 2;
                RegistrationOptions wide = ropt;
                wide.search[2] = cut.n_cols / 2 - 1;
                Measured m;
                // one registration over the wide window, no re-cutting: the peak's position is all that is wanted
                const int n = plan.n;
                std::vector<std::uint16_t> raw(static_cast<std::size_t>(n) * cut.n_rows * cols);
                std::vector<float> a(static_cast<std::size_t>(n) * cut.n_rows * cut.n_cols), b(a.size());
                std::vector<std::int64_t> frame(n);
                std::vector<std::int32_t> row0(n, cut.row_a);
                for (int q = 0; q < n; ++q) frame[q] = plan.p0 + q;
                first_frames.read_rows(si, frame.data(), row0.data(), n, cut.n_rows, raw.data());
                line_integrals(first_ff, raw.data(), row0.data(), n, cut.n_rows, cut.col_a, cut.n_cols, false, a.data());
                for (int q = 0; q < n; ++q) frame[q] = plan.j0 + q;
                first_frames.read_rows(si, frame.data(), row0.data(), n, cut.n_rows, raw.data());
                line_integrals(first_ff, raw.data(), row0.data(), n, cut.n_rows, cut.col_b, cut.n_cols, true, b.data());
                const Registration r = register_blocks(a.data(), b.data(), n, cut.n_rows, cut.n_cols, wide);
                axis0 = mirror_axis(cut.col_a, cut.col_b, cut.n_cols, r.shift[2]);
                std::fprintf(stderr, "  axis search, stretch %s: column %.2f (correlation %.3f)\n", s.entry.c_str(), axis0, r.correlation);
                if (r.correlation < 0.15 || r.at_edge) axis0 = NAN;
            }
        }
        if (!std::isfinite(axis0)) {
            Finding& f = rep.add("5.3.1 4.3", first.label, "opposite views of the turn at rest, stretch " + s.entry);
            f.found = "no mirror axis found inside the detector's field";
            f.verdict = "open";
            continue;
        }
        // the widest window symmetric about that column
        const int half = static_cast<int>(std::min(axis0, cols - axis0)) - 8;
        Cut cut;
        cut.mirror = true;
        cut.n_cols = good_size(2 * half);
        cut.col_a = cut.col_b = static_cast<int>(std::lround(axis0 - cut.n_cols / 2.0));
        cut.n_rows = good_size(opt.rows);
        std::vector<double> axis, dq_res, dq, D, band_row;
        std::vector<std::vector<double>> axis_by_band;
        const std::vector<int> bands = row_bands(rows, cut.n_rows, 16);
        axis_by_band.resize(bands.size());
        for (std::int64_t p0 : spaced(16, s.n_frames - per_turn / 2 - nq - 16, opt.blocks)) {
            BlockPlan plan;
            if (!plan_block(t, t, p0, nq, per_turn / 2, 180.0, plan)) continue;
            for (std::size_t bi = 0; bi < bands.size(); ++bi) {
                cut.row_a = bands[bi];
                Measured m;
                measure(scans[0], si, scans[0], si, plan, cut, ropt, m);
                progress("opposite views", m);
                if (!m.ok) continue;
                const double c = mirror_axis(cut.col_a, cut.col_b, cut.n_cols, m.d[2]);
                axis.push_back(c);
                axis_by_band[bi].push_back(c);
                dq.push_back(static_cast<double>(m.plan.j0 - m.plan.p0) + m.d[0]);
                dq_res.push_back(m.d[0] - m.plan.stated_dq);
                D.push_back(measured_D(m.plan, m.d[0], m.d[1]) - m.plan.stated_D);
            }
        }
        const Spread sa = spread(axis), sq = spread(dq_res), sqa = spread(dq), sD = spread(D);
        axis_all.insert(axis_all.end(), axis.begin(), axis.end());
        const double nominal = cols / 2.0 + positioner(s, first.translation_y[1]) / px_mm;
        {
            Finding& f = rep.add("5.3.1", first.label, "column of the rotation axis from opposite views, stretch " + s.entry);
            f.stated = fmt("nominal only: %s = %.4f mm, column %.2f if %s = 0 is the middle column", first.translation_y[1].c_str(), positioner(s, first.translation_y[1]),
                           nominal, first.translation_y[1].c_str());
            f.found = fmt("column %.3f +- %.3f (sd %.3f over %zu blocks of %d frames x %d rows x %d columns; %.3f to %.3f); nominal minus measured %+.2f px; tolerance %.2f px",
                          sa.mean, sa.sem(), sa.sd, sa.n, nq, cut.n_rows, cut.n_cols, sa.min, sa.max, nominal - sa.mean, tol.axis_px());
            std::vector<std::string> per_band;
            for (std::size_t bi = 0; bi < bands.size(); ++bi)
                if (!axis_by_band[bi].empty()) per_band.push_back(fmt("rows %d-%d: %.3f", bands[bi], bands[bi] + cut.n_rows - 1, spread(axis_by_band[bi]).mean));
            std::string bands_text;
            for (std::size_t i = 0; i < per_band.size(); ++i) bands_text += (i ? ", " : "") + per_band[i];
            f.found += "; by row band: " + bands_text;
            f.verdict = sa.n < 3 ? "open" : (std::fabs(nominal - sa.mean) <= tol.axis_px() ? "holds" : "fails");
            f.note = "verdict: the nominal column against the measured one. The measured column is where the detector's columns, used as they are, mirror best; "
                     "a change with the row band is the axis's roll against the columns (5.3.2) or distortion (5.1.4), which this does not separate";
            f.numbers = {{"axis_column", sa.mean}, {"axis_sem", sa.sem()}, {"axis_sd", sa.sd}, {"nominal", nominal}};
            for (std::size_t bi = 0; bi < bands.size(); ++bi)
                if (!axis_by_band[bi].empty()) f.numbers.push_back({fmt("axis_rows_%d", bands[bi] + cut.n_rows / 2), spread(axis_by_band[bi]).mean});
        }
        {
            Finding& f = rep.add("5.4.1", first.label, "angle over half a turn from opposite views, stretch " + s.entry);
            f.stated = fmt("encoder: the opposite view of frame p is %.4f frames after p", sqa.mean - sq.mean);
            f.found = fmt("registered: %.4f frames after p; measured minus stated %s = %+.5f deg; tolerance %.3f frames (%.4f deg)", sqa.mean, pm(sq, "frames").c_str(),
                          sq.mean * tol.step_deg, tol.angle_frames(), tol.angle_deg());
            f.verdict = verdict(sq, tol.angle_frames());
            f.numbers = {{"residual_frames", sq.mean}, {"residual_sem", sq.sem()}, {"residual_sd", sq.sd}};
        }
        {
            Finding& f = rep.add("4.4", first.label, "height of opposite views in the turn at rest, stretch " + s.entry);
            f.stated = fmt("%s does not move during the stretch", first.translation_z[1].c_str());
            f.found = fmt("opposite views are displaced along the axis by %s; tolerance %.2f rows", pm(sD, "rows").c_str(), tol.rows());
            f.verdict = verdict(sD, tol.rows());
            f.note = "an axis rolled against the columns displaces opposite views by rows in proportion to the distance from the axis, with both signs inside one block; this mean does not show it";
            f.numbers = {{"rows", sD.mean}, {"rows_sem", sD.sem()}};
        }
    }
    const double axis_first = axis_all.empty() ? NAN : spread(axis_all).mean;
    if (rest.size() > 1 && !axis_all.empty()) {
        Finding& f = rep.add("5.3.4 4.3", first.label, "the axis in the first and the last turn at rest; which scan holds both sides of the axis");
        f.found = fmt("mean axis column over the turns at rest: %.3f, which is %.1f columns inside the field of %d", axis_first, std::min(axis_first, cols - axis_first), cols);
        f.note = "opposite views exist only in a scan whose axis is inside the field: this is the first scan of the pair";
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
                const std::int64_t hint = ri < hi ? 0 : sh.n_frames - sr.n_frames;
                std::vector<double> Ds, Dres, qres, ures;
                for (std::int64_t p0 : spaced(16, sr.n_frames - nq - 16, 3 * opt.blocks)) {
                    BlockPlan plan;
                    if (!plan_block(tr, th, p0, nq, hint, 0.0, plan)) continue;
                    const int D = static_cast<int>(std::lround(plan.stated_D));
                    // B's content must not sit where a pattern fixed to the detector would register (near zero rows)
                    if (std::abs(D) < 2 * ropt.search[1] || std::abs(D) > rows - 96) continue;
                    Cut cut;
                    cut.n_rows = good_size(std::min(opt.rows, rows - std::abs(D) - 32));
                    cut.row_a = centred_row(rows, cut.n_rows, D);
                    cut.n_cols = good_size(cols - 8);
                    cut.col_a = cut.col_b = (cols - cut.n_cols) / 2;
                    Measured m;
                    measure(S, ri, S, hi, plan, cut, ropt, m);
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
                const std::string which = "turn at rest " + sr.entry + " against helix " + sh.entry;
                {
                    Finding& f = rep.add("5.4.3 5.4.5 5.4.6", rec.label, "height of the helix against the " + which);
                    f.stated = fmt("%s at rest and the latched heights of the helix, in rows through %.4g um: the helix's content is displaced by %.1f to %.1f rows in the blocks used",
                                   rec.translation_z[1].c_str(), px_mm * 1e3, sDs.min, sDs.max);
                    f.found = fmt("measured minus stated displacement: %s, %.3f to %.3f; as a line in the displacement: %+.4f rows at zero displacement, %+.3e rows per row "
                                  "(rms about the line %.4f rows); tolerances %.2f rows and %.1e",
                                  pm(sD, "rows").c_str(), sD.min, sD.max, lD.intercept, lD.slope, lD.rms, tol.rows(), tol.travel_scale());
                    const bool scale_ok = std::fabs(lD.slope) <= tol.travel_scale(), offset_ok = std::fabs(lD.intercept) <= tol.rows();
                    f.verdict = scale_ok && offset_ok && sD.max <= tol.rows() && sD.min >= -tol.rows() ? "holds" : "fails";
                    f.note = "the slope is the error of (stage scale / pixel pitch along the columns); the value at zero displacement is the error of the height offset between the "
                             "stretches. The two views of a block lie on detector bands a displacement apart, so distortion (5.1.4) enters";
                    f.numbers = {{"offset_rows", lD.intercept}, {"scale_error", lD.slope}, {"rms_rows", lD.rms}, {"mean_rows", sD.mean}};
                }
                {
                    Finding& f = rep.add("5.4.1 5.4.6 5.4.9", rec.label, "angle of the helix against the " + which);
                    f.stated = "the encoder angles of both stretches";
                    f.found = fmt("measured minus stated: %s = %+.5f deg; as a line in the row displacement: %+.4f frames at zero, %+.3e frames per row (rms %.4f); tolerance %.3f frames",
                                  pm(sq, "frames").c_str(), sq.mean * tol.step_deg, lq.intercept, lq.slope, lq.rms, tol.angle_frames());
                    f.verdict = std::fabs(lq.intercept) <= tol.angle_frames() && std::fabs(lq.intercept + lq.slope * (sDs.max > -sDs.min ? sDs.max : sDs.min)) <= tol.angle_frames() ? "holds" : "fails";
                    f.note = fmt("the views of a block lie on detector rows a displacement apart. If the detector's rows were exposed one after another, the angle would differ in "
                                 "proportion to that displacement: the slope is %+.2f us per row, %+.2f ms over the %d rows read out",
                                 lq.slope * rec.stretches[hi].timer_period[0] * 1e6, lq.slope * rec.stretches[hi].timer_period[0] * 1e3 * rows, rows);
                    f.numbers = {{"offset_frames", lq.intercept}, {"frames_per_row", lq.slope}, {"mean_frames", sq.mean}};
                }
                {
                    Finding& f = rep.add("5.4.7", rec.label, "lateral position of the helix against the " + which);
                    f.stated = fmt("not stated; %s is the same in both stretches", rec.translation_y[1].c_str());
                    f.found = fmt("columns: %s; as a line in the row displacement: %+.4f px at zero, %+.3e px per row (rms %.4f)", pm(su, "px").c_str(), lu.intercept, lu.slope, lu.rms);
                    f.note = fmt("a slope is the travel's direction against the detector's columns (tolerance %.1e rad) or the detector's distortion between the two bands", tol.eps / 2 / tol.H);
                    f.numbers = {{"offset_px", lu.intercept}, {"px_per_row", lu.slope}};
                }
            }
            // the helix a turn apart
            {
                std::vector<double> P, Ps, Dres, qres, ures;
                for (std::int64_t p0 : spaced(16, sh.n_frames - per_turn - nq - 64, 2 * opt.blocks)) {
                    BlockPlan plan;
                    if (!plan_block(th, th, p0, nq, per_turn, 0.0, plan)) continue;
                    const int D = static_cast<int>(std::lround(plan.stated_D));
                    if (std::abs(D) > rows - 64) continue;
                    Cut cut;
                    cut.n_rows = good_size(rows - std::abs(D) - 16);
                    cut.row_a = centred_row(rows, cut.n_rows, D);
                    cut.n_cols = good_size(cols - 8);
                    cut.col_a = cut.col_b = (cols - cut.n_cols) / 2;
                    Measured m;
                    measure(S, hi, S, hi, plan, cut, ropt, m);
                    progress("helix a turn apart", m);
                    if (!m.ok) continue;
                    const double Dm = measured_D(m.plan, m.d[0], m.d[1]);
                    P.push_back(std::fabs(Dm));
                    Ps.push_back(std::fabs(m.plan.stated_D));
                    Dres.push_back(std::fabs(Dm) - std::fabs(m.plan.stated_D));
                    qres.push_back(m.d[0] - m.plan.stated_dq);
                    ures.push_back(m.d[2]);
                }
                if (P.size() >= 3) {
                    const Spread sP = spread(P), sPs = spread(Ps), sD = spread(Dres), sq = spread(qres), su = spread(ures);
                    const double tolP = tol.eps * sP.mean / tol.H;
                    {
                        Finding& f = rep.add("5.4.5 5.1.2", rec.label, "travel along the axis per turn in rows, helix " + sh.entry + " a turn apart");
                        f.stated = fmt("encoder against encoder through %.4g um: %.4f rows per turn", px_mm * 1e3, sPs.mean);
                        f.found = fmt("views a turn apart are displaced by %.4f +- %.4f rows (sd %.4f over %zu blocks of %d frames); measured minus stated %s; tolerance %.4f rows per turn",
                                      sP.mean, sP.sem(), sP.sd, sP.n, nq, pm(sD, "rows").c_str(), tolP);
                        f.verdict = verdict(sD, tolP);
                        f.note = fmt("measured is (stage scale / pixel pitch along the columns): the difference is %+.2e of the travel, i.e. a pitch of %.4f um if the stage's scale is exact. "
                                     "The two views lie at the top and the bottom of the detector, so distortion (5.1.4) enters",
                                     sD.mean / sPs.mean, px_mm * 1e3 * sPs.mean / sP.mean);
                        f.numbers = {{"rows_per_turn", sP.mean}, {"rows_per_turn_sem", sP.sem()}, {"stated_rows_per_turn", sPs.mean}, {"measured_minus_stated", sD.mean}};
                    }
                    {
                        Finding& f = rep.add("5.4.1 5.4.9", rec.label, "angle over a whole turn, helix " + sh.entry + " a turn apart");
                        f.stated = "encoder: the frame one turn after frame p";
                        f.found = fmt("measured minus stated %s = %+.5f deg; tolerance %.3f frames (%.4f deg)", pm(sq, "frames").c_str(), sq.mean * tol.step_deg, tol.angle_frames(), tol.angle_deg());
                        f.verdict = verdict(sq, tol.angle_frames());
                        f.note = fmt("the two views lie %.0f rows apart on the detector: rows exposed one after another would show here as %+.2f us per row, %+.2f ms over the %d rows read out",
                                     sP.mean, sq.mean / sP.mean * sh.timer_period[0] * 1e6, sq.mean / sP.mean * sh.timer_period[0] * 1e3 * rows, rows);
                        f.numbers = {{"residual_frames", sq.mean}, {"residual_sem", sq.sem()}};
                    }
                    {
                        Finding& f = rep.add("5.4.7", rec.label, "lateral position a turn apart, helix " + sh.entry);
                        f.stated = "not stated";
                        f.found = fmt("columns: %s over %.0f rows of travel = %+.2e px per row", pm(su, "px").c_str(), sP.mean, su.mean / sP.mean);
                        f.numbers = {{"px", su.mean}, {"px_sem", su.sem()}};
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
            Cut cut;
            cut.n_cols = good_size(cols - M - 8);
            cut.n_rows = good_size(opt.rows);
            if (col_sign == 0) {
                // which way the sample moved on the detector: both, once
                double corr[2] = {0, 0};
                for (int c = 0; c < 2; ++c) {
                    BlockPlan plan;
                    if (!plan_block(ta, tb, sa.n_frames / 3, nq, 0, 0.0, plan)) continue;
                    cut.row_a = (rows - cut.n_rows) / 2;
                    cut.col_a = c == 0 ? (cols - M - cut.n_cols) / 2 : M + (cols - M - cut.n_cols) / 2;
                    cut.col_b = c == 0 ? cut.col_a + M : cut.col_a - M;
                    Measured m;
                    measure(scans[0], pa[k], scans[1], pb[k], plan, cut, ropt, m);
                    progress(c == 0 ? "column sense +" : "column sense -", m);
                    corr[c] = m.ok ? m.correlation : 0.0;
                }
                col_sign = corr[0] >= corr[1] ? 1.0 : -1.0;
                Finding& f = rep.add("5.1.6", both, "sense of the detector's columns against the lateral stage");
                f.stated = fmt("%s %.4f -> %.4f mm", y.c_str(), positioner(sa, y), positioner(sb, y));
                f.found = fmt("the second scan registers on the first with correlation %.3f if its content lies %d columns higher, %.3f if lower: larger %s = %s columns",
                              corr[0], M, corr[1], y.c_str(), (col_sign > 0) == (move_px > 0) ? "higher" : "lower");
                f.verdict = std::max(corr[0], corr[1]) > 0.3 && std::min(corr[0], corr[1]) < 0.5 * std::max(corr[0], corr[1]) ? "holds" : "open";
                f.note = "verdict: the sense is determined by the frames";
                f.numbers = {{"column_sign", col_sign}};
                if (f.verdict != "holds") break;
            }
            cut.col_a = col_sign > 0 ? (cols - M - cut.n_cols) / 2 : M + (cols - M - cut.n_cols) / 2;
            cut.col_b = col_sign > 0 ? cut.col_a + M : cut.col_a - M;
            std::vector<double> move, Dm, Dres, qres;
            const std::vector<int> bands = row_bands(rows, cut.n_rows, 16);
            std::vector<std::vector<double>> move_by_band(bands.size()), D_by_band(bands.size());
            for (std::int64_t p0 : spaced(16, sa.n_frames - nq - 16, opt.blocks)) {
                BlockPlan plan;
                if (!plan_block(ta, tb, p0, nq, 0, 0.0, plan)) continue;
                for (std::size_t bi = 0; bi < bands.size(); ++bi) {
                    cut.row_a = bands[bi];
                    Measured m;
                    measure(scans[0], pa[k], scans[1], pb[k], plan, cut, ropt, m);
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
                }
            }
            if (move.size() < 3) continue;
            const Spread sm = spread(move), sD = spread(Dres), sDm = spread(Dm), sq = spread(qres);
            move_all.insert(move_all.end(), move.begin(), move.end());
            const double stated_move = col_sign * std::fabs(move_px);
            std::vector<double> res(move.size());
            for (std::size_t i = 0; i < move.size(); ++i) res[i] = move[i] - stated_move;
            const Spread sr = spread(res);
            std::string bands_text, bands_D;
            for (std::size_t bi = 0; bi < bands.size(); ++bi)
                if (!move_by_band[bi].empty()) {
                    bands_text += fmt("%srows %d-%d: %.3f", bands_text.empty() ? "" : ", ", bands[bi], bands[bi] + cut.n_rows - 1, spread(move_by_band[bi]).mean);
                    bands_D += fmt("%srows %d-%d: %+.3f", bands_D.empty() ? "" : ", ", bands[bi], bands[bi] + cut.n_rows - 1, spread(D_by_band[bi]).mean);
                }
            {
                Finding& f = rep.add("6.1.1 5.1.1", both, "lateral displacement of the second scan's content, stretch " + sa.entry);
                f.stated = fmt("%s move %.4f mm = %.3f px at %.4g um", y.c_str(), positioner(sb, y) - positioner(sa, y), std::fabs(move_px), px_mm * 1e3);
                f.found = fmt("%.4f +- %.4f columns (sd %.4f over %zu blocks of %d frames x %d rows x %d columns); measured minus stated %+.4f px; tolerance %.2f px; by row band: %s",
                              std::fabs(sm.mean), sm.sem(), sm.sd, sm.n, nq, cut.n_rows, cut.n_cols, std::fabs(sm.mean) - std::fabs(move_px), tol.eps, bands_text.c_str());
                f.verdict = verdict(sr, tol.eps);
                f.note = fmt("measured is (lateral stage scale / pixel pitch along the rows): the difference is %+.2e of the move, i.e. a pitch of %.4f um if the stage's scale is exact. "
                             "The same rays fall on detector columns a move apart, so distortion (5.1.4, 6.3.3) enters; a change with the row band is a roll of one scan against the other (6.1.4) or distortion",
                             (std::fabs(sm.mean) - std::fabs(move_px)) / std::fabs(move_px), px_mm * 1e3 * std::fabs(move_px) / std::fabs(sm.mean));
                f.numbers = {{"move_columns", sm.mean}, {"move_sem", sm.sem()}, {"move_sd", sm.sd}, {"stated_px", stated_move}};
            }
            {
                Finding& f = rep.add("6.1.2 6.2.3", both, "displacement along the axis of the second scan's content, stretch " + sa.entry);
                f.stated = fmt("the two scans' heights: second minus first %+.3f rows at %.4g um", sDm.mean - sD.mean, px_mm * 1e3);
                f.found = fmt("%+.4f rows; measured minus stated %s; tolerance %.2f rows; by row band: %s", sDm.mean, pm(sD, "rows").c_str(), tol.rows(), bands_D.c_str());
                f.verdict = verdict(sD, tol.rows());
                f.numbers = {{"rows", sDm.mean}, {"measured_minus_stated", sD.mean}, {"sem", sD.sem()}};
            }
            {
                Finding& f = rep.add("6.2.1 6.1.6", both, "angle of the second scan against the first at equal projection index, stretch " + sa.entry);
                f.stated = "the encoder angles of both scans";
                f.found = fmt("measured minus stated %s = %+.5f deg (%.3f px at R); tolerance %.3f frames (%.4f deg)", pm(sq, "frames").c_str(), sq.mean * tol.step_deg,
                              std::fabs(sq.mean * tol.step_deg) / 57.29577951308232 * tol.R, tol.angle_frames(), tol.angle_deg());
                f.verdict = verdict(sq, tol.angle_frames());
                f.note = "a difference here is the sample turned about the axis between the scans without the encoder showing it (6.1.6)";
                f.numbers = {{"residual_frames", sq.mean}, {"residual_sem", sq.sem()}};
            }
        }
        if (!move_all.empty() && std::isfinite(axis_first)) {
            const Spread sm = spread(move_all);
            const Stretch& sb = second->stretches[pb.at(0)];
            const double nominal = cols / 2.0 + positioner(sb, y) / px_mm;
            Finding& f = rep.add("6.3.2 5.3.1", second->label, "column of the second scan's rotation axis");
            f.stated = fmt("nominal only: column %.2f if %s = 0 is the middle column", nominal, y.c_str());
            f.found = fmt("first scan's axis %.3f + measured displacement %.3f = column %.3f (the field ends at column %d); nominal minus measured %+.2f px", axis_first, sm.mean,
                          axis_first + sm.mean, cols - 1, nominal - (axis_first + sm.mean));
            f.verdict = std::fabs(nominal - (axis_first + sm.mean)) <= tol.axis_px() ? "holds" : "fails";
            f.note = "the second scan has no opposite views of its own; this value inherits the errors of both measurements and assumes the axis moved exactly as the sample's content did";
            f.numbers = {{"axis_column", axis_first + sm.mean}, {"nominal", nominal}};
        }
    }
}

}  // namespace helirec::calibration
