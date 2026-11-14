// helirec/calibration/geometry_calibration.hpp: the layout of the scans, the plan of their comparisons, their
// measurement, and the step that runs them with the solve (calibrate_geometry; the solve is geometry_fit.cpp).
#include "helirec/calibration/geometry_calibration.hpp"

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <set>

#include "calibration/util.hpp"
#include "helirec/calibration/solve.hpp"
#include "io/json.hpp"

namespace helirec::calibration::geometry_calibration {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

void say(const Log& log, const std::string& line) {
    if (log) log(line);
}

// -ln((raw - dark) / mean flat) of rows [row0, row0 + nrows) of one frame, the whole width: (nrows, cols)
std::vector<float> integrals(Session& session, int scan, int stretch, std::int64_t frame, int row0, int nrows) {
    Frames& f = session.frames(scan);
    const std::size_t cols = static_cast<std::size_t>(f.cols());
    std::vector<std::uint16_t> raw(static_cast<std::size_t>(nrows) * cols);
    const std::int32_t r0 = row0;
    f.read_rows(stretch, &frame, &r0, 1, nrows, raw.data());
    std::vector<float> out(static_cast<std::size_t>(nrows) * cols);
    line_integrals(session.flat_field(scan), raw.data(), &r0, 1, nrows, 0, static_cast<int>(cols), false, out.data());
    return out;
}

// the columns' profile of 256 rows of a frame, less its running mean over 31 columns
std::vector<double> profile(Session& session, int scan, int stretch, std::int64_t frame, int row0, int nrows, int cols) {
    const std::vector<float> p = integrals(session, scan, stretch, frame, row0, nrows);
    std::vector<float> mean(static_cast<std::size_t>(cols), 0.0f);
    for (int r = 0; r < nrows; ++r)
        for (int c = 0; c < cols; ++c) mean[static_cast<std::size_t>(c)] += p[static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(c)];
    for (float& v : mean) v /= static_cast<float>(nrows);
    const int k = 31, half = k / 2;
    std::vector<double> out(static_cast<std::size_t>(cols));
    for (int c = 0; c < cols; ++c) {
        double s = 0;
        for (int j = -half; j <= half; ++j) s += static_cast<double>(mean[static_cast<std::size_t>(std::clamp(c + j, 0, cols - 1))]) * (1.0 / k);
        out[static_cast<std::size_t>(c)] = static_cast<double>(mean[static_cast<std::size_t>(c)]) - s;
    }
    return out;
}

// cc[k + n] += sum_c a[c] b[c + k] for k in (-n, n). least > 0: each lag's sum divided by the two profiles'
// lengths over the columns the lag has in common (their correlation there, -1..1), and lags with fewer than
// `least` columns in common left out: a lag far from zero is then not at a disadvantage for its few columns.
void add_lag(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& cc, long least = 0) {
    const long n = static_cast<long>(a.size());
#pragma omp parallel for schedule(static)
    for (long k = -n + 1; k < n; ++k) {
        double s = 0;
        const long lo = std::max(0L, -k), hi = std::min(n, n - k);
        for (long c = lo; c < hi; ++c) s += a[static_cast<std::size_t>(c)] * b[static_cast<std::size_t>(c + k)];
        if (least > 0) {
            if (hi - lo < least) continue;
            double sa = 0, sb = 0;
            for (long c = lo; c < hi; ++c) sa += a[static_cast<std::size_t>(c)] * a[static_cast<std::size_t>(c)], sb += b[static_cast<std::size_t>(c + k)] * b[static_cast<std::size_t>(c + k)];
            s = sa > 0 && sb > 0 ? s / std::sqrt(sa * sb) : 0.0;
        }
        cc[static_cast<std::size_t>(k + n)] += s;
    }
}

// the columns two profiles must have in common for a lag to count: 256 on a detector as stored, an eighth of the
// width on a narrower one (frames read at a quarter leave the opposite views of a tile 150 columns)
long least_in_common(int cols) { return cols >= 2048 ? 256 : std::max(32, cols / 8); }

std::size_t argmax(const std::vector<double>& v) { return static_cast<std::size_t>(std::max_element(v.begin(), v.end()) - v.begin()); }

std::vector<std::int64_t> whole(const std::vector<double>& v) {
    std::vector<std::int64_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = static_cast<std::int64_t>(v[i]);  // towards zero, as numpy's astype(int)
    return out;
}

Json num(double v) { return Json::of(v); }
Json count(long long v) { return Json::whole_number(v); }
}  // namespace

bool Blocked::holds(double frame, int row_lo, int row_hi) const {
    if (frames.empty()) return false;
    const int rows = static_cast<int>(mask.size() / frames.size());
    std::size_t k = 0;
    for (std::size_t i = 1; i < frames.size(); ++i)
        if (std::fabs(static_cast<double>(frames[i]) - frame) < std::fabs(static_cast<double>(frames[k]) - frame)) k = i;
    const int lo = std::max(0, row_lo), hi = std::min(rows, std::max(0, row_hi));
    for (int r = lo; r < hi; ++r)
        if (mask[k * static_cast<std::size_t>(rows) + static_cast<std::size_t>(r)]) return true;
    return false;
}

namespace {

// Rows that hold something across the whole width in frames [first, end) of a stretch, sampled 16 times a turn:
// the mean of the lowest columns of a row is not air there. Nothing found: no frames. label: the stretch, for the log.
Blocked blocked_rows(Session& session, int scan, int stretch, std::int64_t first, std::int64_t end, int per_turn, const std::string& label, const Log& log) {
    const int rows = session.frames(scan).rows(), cols = session.frames(scan).cols();
    const int few = std::max(8, cols / 50);
    const std::int64_t n = end - first;
    const int samples = static_cast<int>(std::max<std::int64_t>(8, n / std::max(1, per_turn / 16)));
    std::vector<std::int64_t> frames = whole(linspace(static_cast<double>(first), static_cast<double>(end - 1), samples));
    frames.erase(std::unique(frames.begin(), frames.end()), frames.end());
    Blocked b;
    b.frames = frames;
    b.mask.assign(frames.size() * static_cast<std::size_t>(rows), 0);
    std::vector<std::pair<int, int>> spans;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const std::vector<float> p = integrals(session, scan, stretch, frames[i], 0, rows);
        std::vector<float> q(static_cast<std::size_t>(rows));
#pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            std::vector<float> line(p.begin() + static_cast<std::ptrdiff_t>(r) * cols, p.begin() + static_cast<std::ptrdiff_t>(r + 1) * cols);
            std::nth_element(line.begin(), line.begin() + few, line.end());
            float s = 0;
            for (int c = 0; c < few; ++c) s += line[static_cast<std::size_t>(c)];
            q[static_cast<std::size_t>(r)] = s / static_cast<float>(few);
        }
        std::vector<float> sorted = q;
        std::sort(sorted.begin(), sorted.end());
        const std::size_t m = sorted.size() / 2;
        const float med = sorted.size() % 2 ? sorted[m] : (sorted[m - 1] + sorted[m]) / 2.0f;
        const float limit = med + 0.1f;
        int lo = -1, hi = -1;
        for (int r = 0; r < rows; ++r) {
            bool hit = false;  // with 24 rows around a row that holds something
            for (int j = std::max(0, r - 24); j <= std::min(rows - 1, r + 24) && !hit; ++j) hit = q[static_cast<std::size_t>(j)] > limit;
            if (!hit) continue;
            b.mask[i * static_cast<std::size_t>(rows) + static_cast<std::size_t>(r)] = 1;
            if (lo < 0) lo = r;
            hi = r;
        }
        if (lo >= 0) spans.emplace_back(lo, hi);
    }
    if (spans.empty()) return Blocked{};
    std::set<std::pair<int, int>> distinct(spans.begin(), spans.end());
    std::string where;
    if (distinct.size() == 1) where = fmt("rows %d..%d", spans[0].first, spans[0].second);
    else {
        int lo = spans[0].first, hi = spans[0].second;
        for (const auto& s : spans) lo = std::min(lo, s.first), hi = std::max(hi, s.second);
        where = fmt("rows between %d and %d, changing with the frame,", lo, hi);
    }
    say(log, fmt("  %s: %s hold something across the whole width in %zu of %zu sampled frames; kept out", label.c_str(), where.c_str(), spans.size(), frames.size()));
    return b;
}

// The sense of the rows against the axis stage in which the requests register better: the sign, its mean
// correlation, and the other sense's.
struct RowSense {
    double sign = -1, score = -1, other = -1;
};
RowSense row_sense(Session& session, const std::vector<ViewRequest>& reqs, Device device) {
    RowSense out;
    for (const double sign : {-1.0, 1.0}) {
        std::vector<double> got;
        for (const ViewMeasurement& r : session.measure(reqs, sign, device))
            if (r.planned && r.ok) got.push_back(r.correlation);
        const double score = got.empty() ? 0.0 : mean_of(got);
        if (score > out.score) out.other = out.score, out.score = score, out.sign = sign;
        else out.other = score;
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------- the layout, the plan, the measurement

namespace {

// The axis column of a scan, coarsely: the column profiles of frames half a turn apart in one stretch, the second
// reversed and its rows `sense` x the record's travel further on, correlated over all lags that leave 256 columns
// in common. score: their mean correlation at the lag found. (course.rows as the record has them, sense +1.)
struct CoarseAxis {
    double axis = std::nan(""), score = 0;
};
CoarseAxis coarse_axis(Session& session, int scan, const StretchLayout& st, int per_turn, double sense) {
    const int rows = session.frames(scan).rows(), cols = session.frames(scan).cols(), N = per_turn, band = 256;
    if (rows < band) throw Error("geometry calibration: the detector has fewer than 256 rows");
    std::vector<double> cc(2 * static_cast<std::size_t>(cols), 0.0);
    int pairs = 0;
    for (const std::int64_t offset : whole(linspace(N / 20, N / 2 - N / 20, 8))) {
        const std::int64_t p = st.first + offset;
        if (p + N / 2 >= st.end) continue;
        const int D = static_cast<int>(std::lround(sense * (st.course.rows[static_cast<std::size_t>(p + N / 2)] - st.course.rows[static_cast<std::size_t>(p)])));
        const int lo = std::max(0, -D), hi = rows - band - std::max(0, D);
        if (hi < lo) return {};  // half a turn's travel leaves no rows in common
        // the rows nearest the middle that hold nothing across the whole width in either frame
        int r0 = (lo + hi) / 2;
        const auto is_free = [&](int r) { return !st.blocked.holds(static_cast<double>(p), r, r + band) && !st.blocked.holds(static_cast<double>(p + N / 2), r + D, r + D + band); };
        if (!is_free(r0)) {
            int best = -1;
            for (int r = lo; r <= hi; r += 16)
                if (is_free(r) && (best < 0 || std::abs(r - r0) < std::abs(best - r0))) best = r;
            if (best < 0) continue;
            r0 = best;
        }
        const std::vector<double> a = profile(session, scan, st.index, p, r0, band, cols);
        std::vector<double> b = profile(session, scan, st.index, p + N / 2, r0 + D, band, cols);
        std::reverse(b.begin(), b.end());
        add_lag(a, b, cc, least_in_common(cols));
        ++pairs;
    }
    if (!pairs) return {};
    const std::size_t k = argmax(cc);
    return {(cols - 1 - (static_cast<double>(k) - cols)) / 2.0, cc[k] / pairs};
}

// the widest window of a scan with its axis at `axis`: a power of two from 32 to 512 inside the strip its opposite
// views share, with room for the cut's margins (B's block takes up to 32 columns on either side, and the windows
// keep 40 from the strip's ends); 0: none
int window_width(double axis, int cols) {
    const double strip = 2.0 * std::min(axis, cols - 1 - axis);
    if (strip < 32 + 84) return 0;
    return std::min(512, static_cast<int>(std::pow(2.0, std::floor(std::log2(strip - 84)))));
}

// the first columns of A's windows of width W whose mirror image about `axis` lies on the detector with a margin:
// up to three, or none
std::vector<std::int64_t> mirrored_windows(double axis, int cols, int W) {
    const int axis2 = static_cast<int>(std::nearbyint(2 * axis)), edge = 40;
    const int lo = std::max(0, axis2 - (W - 1) - (cols - W)) + edge, hi = std::min(cols - W, axis2 - (W - 1)) - edge;
    if (hi < lo) return {};
    const int count = std::clamp(1 + (hi - lo) / W, 1, 3);
    if (count == 1) return {(lo + hi) / 2};
    return whole(linspace(lo, hi, count));
}

ViewRequest opposite_request(const StretchLayout& st, int scan, std::int64_t frame, int per_turn, double axis, int W, std::int64_t col_a, int rows, int row_a, double fan) {
    ViewRequest q;
    q.label = "opposite";
    q.scan_a = q.scan_b = scan;
    q.stretch_a = q.stretch_b = st.index;
    q.frame_a = frame, q.frame_hint = per_turn / 2;
    q.angle_offset_deg = 180.0, q.mirror = true;
    q.frames = 32, q.rows = rows, q.cols = W;
    q.row_a = row_a, q.col_a = static_cast<int>(col_a);
    q.col_b = static_cast<int>(std::nearbyint(2 * axis)) - (W - 1) - static_cast<int>(col_a);
    q.frame_shear = 2 * fan;  // the opposite view of a ray at the fan angle g lies half a turn and 2 g on
    return q;
}

// How many columns higher scan k holds what scan j holds at equal frames, coarsely: the column profiles of four
// frames of a stretch both have (a rest turn if there is one), correlated over all lags that leave 256 columns in
// common. score: their mean correlation at the lag found (0: no stretch of the same name with frames in common).
struct CoarseMove {
    int move = 0;
    double score = 0;
};
CoarseMove coarse_move(Session& session, const Acquisition& acq, int j, int k) {
    const ScanLayout& A = acq.scans[static_cast<std::size_t>(j)];
    const ScanLayout& B = acq.scans[static_cast<std::size_t>(k)];
    const int rows = acq.rows, cols = acq.cols, band = 256;
    const StretchLayout* sa = nullptr;
    const StretchLayout* sb = nullptr;
    for (const bool helix : {false, true})
        for (const StretchLayout& a : A.stretches)
            for (const StretchLayout& b : B.stretches)
                if (!sa && a.helix == helix && a.name == b.name && std::min(a.end, b.end) - std::max(a.first, b.first) > 64) sa = &a, sb = &b;
    if (!sa) return {};
    const std::int64_t first = std::max(sa->first, sb->first), end = std::min(sa->end, sb->end);
    std::vector<double> cc(2 * static_cast<std::size_t>(cols), 0.0);
    int pairs = 0;
    for (const std::int64_t p : whole(linspace(static_cast<double>(first + (end - first) / 10), static_cast<double>(first + (end - first) / 2), 4))) {
        // the rows nearest the middle that hold nothing across the whole width in either scan
        int r0 = -1;
        for (int r = 0; r + band <= rows; r += 16)
            if (!sa->blocked.holds(static_cast<double>(p), r, r + band) && !sb->blocked.holds(static_cast<double>(p), r, r + band) &&
                (r0 < 0 || std::abs(r - (rows - band) / 2) < std::abs(r0 - (rows - band) / 2)))
                r0 = r;
        if (r0 < 0) continue;
        add_lag(profile(session, j, sa->index, p, r0, band, cols), profile(session, k, sb->index, p, r0, band, cols), cc, least_in_common(cols));
        ++pairs;
    }
    if (!pairs) return {};
    const std::size_t at = argmax(cc);
    return {static_cast<int>(at) - cols, cc[at] / pairs};
}

}  // namespace

bool axis_in_field(const ScanLayout& scan, int cols) { return !std::isnan(scan.axis) && window_width(scan.axis, cols) > 0; }

Acquisition read_acquisition(Session& session, Device device, const Log& log) {
    Acquisition acq;
    const int S = static_cast<int>(session.size());
    acq.rows = session.frames(0).rows(), acq.cols = session.frames(0).cols();
    acq.pixel_um = pixel_um(session.record(0));
    const int rows = acq.rows, cols = acq.cols;
    // the projection stretches on disk, named from the record's own order
    for (int k = 0; k < S; ++k) {
        const Record& rec = session.record(k);
        if (session.frames(k).rows() != rows || session.frames(k).cols() != cols || pixel_um(rec) != acq.pixel_um)
            throw Error("geometry calibration: scan " + std::to_string(k) + " has another detector format or pixel size than the first");
        ScanLayout sc;
        sc.name = rec.label.empty() ? "scan" + std::to_string(k) : rec.label;
        int rest = 0, helices = 0;
        std::string found, absent;
        std::int64_t before = 0;  // projection frames of the stretches before this one
        for (std::size_t i = 0; i < rec.stretches.size(); ++i) {
            const Stretch& st = rec.stretches[i];
            if (st.role != Role::Projection) continue;
            const std::int64_t offset = before;
            before += st.n_frames;
            if (!st.latched()) continue;
            StretchLayout sl;
            sl.offset = offset;
            sl.helix = !st.z_center.empty();
            sl.name = sl.helix ? "helix" : "rest" + std::to_string(++rest);
            sl.index = static_cast<int>(i);
            for (const auto& [first, end] : st.frames_on_disk())  // the longest run of frames on disk
                if (end - first > sl.end - sl.first) sl.first = first, sl.end = end;
            if (sl.end - sl.first < 64) {
                absent += " " + sl.name;
                continue;
            }
            if (sl.helix && ++helices > 1) throw Error("geometry calibration: the record of scan " + sc.name + " holds more than one helix");
            sl.course = trajectory(rec, st, 1.0);
            found += fmt(" %s %lld..%lld of %lld", sl.name.c_str(), static_cast<long long>(sl.first), static_cast<long long>(sl.end - 1), static_cast<long long>(st.n_frames));
            sc.stretches.push_back(std::move(sl));
        }
        if (sc.stretches.empty()) throw Error("geometry calibration: no projection stretch of scan " + sc.name + " is on disk");
        say(log, fmt("  scan %d (%s): frames on disk:%s%s%s", k, sc.name.c_str(), found.c_str(), absent.empty() ? "" : "; not on disk:", absent.c_str()));
        acq.scans.push_back(std::move(sc));
    }
    // frames per turn and the angle per frame, the same in every scan
    double step_deg = 0;
    for (int k = 0; k < S; ++k) {
        const ScanLayout& sc = acq.scans[static_cast<std::size_t>(k)];
        const StretchLayout* st = &sc.stretches.front();
        for (const StretchLayout& t : sc.stretches)
            if (t.helix) st = &t;
        const std::vector<double>& angle = st->course.angle_deg;
        if (angle.size() < 2) throw Error("geometry calibration: a stretch holds fewer than two frames");
        std::vector<double> d(angle.size() - 1);
        for (std::size_t i = 0; i + 1 < angle.size(); ++i) d[i] = angle[i + 1] - angle[i];
        const double step = median_of(d);
        const int per_turn = static_cast<int>(std::nearbyint(360.0 / std::fabs(step)));
        if (k == 0) acq.per_turn = per_turn, step_deg = step;
        else if (per_turn != acq.per_turn) throw Error("geometry calibration: scan " + sc.name + " has another number of frames per turn than the first");
    }
    const int N = acq.per_turn;
    acq.fan = acq.pixel_um * 1e-3 / session.record(0).technique.number("scan/source_sample_distance") / (std::fabs(step_deg) * (kPi / 180.0));
    {
        const StretchLayout* st = &acq.scans[0].stretches.front();
        for (const StretchLayout& t : acq.scans[0].stretches)
            if (t.helix) st = &t;
        const std::vector<double>& period = session.record(0).stretches[static_cast<std::size_t>(st->index)].timer_period;
        if (!period.empty()) acq.period_s = median_of(period);
    }
    // rows that hold something across the whole width
    for (int k = 0; k < S; ++k)
        for (StretchLayout& st : acq.scans[static_cast<std::size_t>(k)].stretches)
            st.blocked = blocked_rows(session, k, st.index, st.first, st.end, N, fmt("scan %d %s", k, st.name.c_str()), log);
    // per scan the coarse axis column and, from its helix, the sense of the rows
    double sense_sum = 0;
    int senses = 0;
    std::string said;
    for (int k = 0; k < S; ++k) {
        ScanLayout& sc = acq.scans[static_cast<std::size_t>(k)];
        const StretchLayout* helix = nullptr;
        const StretchLayout* rest = nullptr;
        for (const StretchLayout& t : sc.stretches) {
            if (t.helix) helix = &t;
            else if (!rest && t.end - t.first > N / 2 + N / 10) rest = &t;
        }
        // a rest turn's opposite views lie on the same rows whatever the rows' sense; a helix' do not: both tried
        CoarseAxis axis[2];  // by the sense: -1, +1
        if (rest) axis[0] = axis[1] = coarse_axis(session, k, *rest, N, 1.0);
        else if (helix) axis[0] = coarse_axis(session, k, *helix, N, -1.0), axis[1] = coarse_axis(session, k, *helix, N, 1.0);
        const CoarseAxis& likely = axis[1].score > axis[0].score ? axis[1] : axis[0];
        const bool in_field = likely.score >= 0.2 && window_width(likely.axis, cols) > 0;
        double sense = 0;
        RowSense found;
        if (helix && helix->end - helix->first > 2 * N + 200) {
            const double travel = std::fabs(helix->course.rows[static_cast<std::size_t>(helix->first + N)] - helix->course.rows[static_cast<std::size_t>(helix->first)]);
            if (rows - travel >= 128) {  // the helix a turn apart shares a patch of rows
                std::vector<ViewRequest> reqs;
                const int W = 512 <= cols - 64 ? 512 : 256;
                for (const std::int64_t p : whole(linspace(static_cast<double>(helix->first + 64), static_cast<double>(helix->end - N - 128), 3))) {
                    ViewRequest q;
                    q.scan_a = q.scan_b = k;
                    q.stretch_a = q.stretch_b = helix->index;
                    q.frame_a = p, q.frame_hint = N;
                    q.frames = 32, q.rows = 128, q.cols = W, q.row_a = -1;
                    q.col_a = q.col_b = (cols - W) / 2;
                    reqs.push_back(q);
                }
                found = row_sense(session, reqs, device);
                sense = found.sign;
            }
        }
        if (helix && sense == 0 && in_field) {  // the helix' opposite views, mirrored about the axis found
            const int W = window_width(likely.axis, cols);
            const std::vector<std::int64_t> windows = mirrored_windows(likely.axis, cols, W);
            std::vector<ViewRequest> reqs;
            if (!windows.empty() && helix->end - helix->first > N / 2 + 256)
                for (const std::int64_t p : whole(linspace(static_cast<double>(helix->first + 64), static_cast<double>(helix->end - N / 2 - 128), 3)))
                    reqs.push_back(opposite_request(*helix, k, p, N, likely.axis, W, windows[windows.size() / 2], std::min(256, rows / 4), -1, acq.fan));
            if (!reqs.empty()) {
                found = row_sense(session, reqs, device);
                sense = found.sign;
            }
        }
        if (helix && sense != 0) {
            if (found.score < 0.3) throw Error("the helix of scan " + sc.name + " registers with itself in neither sense of the rows (a turn on, or half a turn on mirrored)");
            sense_sum += sense, ++senses;
            said += fmt("%s%s %+.0f (%.2f, the other sense %.2f)", said.empty() ? "" : ", ", sc.name.c_str(), sense, found.score, found.other);
        }
        const CoarseAxis& chosen = sense > 0 ? axis[1] : sense < 0 ? axis[0] : likely;
        if (chosen.score >= 0.2 && window_width(chosen.axis, cols) > 0) sc.axis = chosen.axis;
        say(log, fmt("  scan %d (%s): axis column about %.1f (profiles half a turn apart correlate %.2f)%s", k, sc.name.c_str(), chosen.axis, chosen.score,
                     std::isnan(sc.axis) ? ": not taken as inside the field" : ""));
    }
    if (!senses) throw Error("geometry calibration: no scan has a helix whose views register with each other: the sense of the rows is not known");
    if (std::fabs(sense_sum) != senses) throw Error("geometry calibration: the scans disagree about the sense of the rows against the axis stage: " + said);
    acq.row_sign = sense_sum > 0 ? 1.0 : -1.0;
    // what the motors say of each scan's place, and the anchor
    const double px_mm = acq.pixel_um * 1e-3;
    for (int k = 0; k < S; ++k) {
        const Record& rec = session.record(k);
        ScanLayout& sc = acq.scans[static_cast<std::size_t>(k)];
        const auto& at_start = rec.stretches[static_cast<std::size_t>(sc.stretches.front().index)].positioners_start;
        const auto place = [&](const std::string& motor) {
            const auto it = at_start.find(motor);
            return motor.empty() || it == at_start.end() ? std::nan("") : it->second / px_mm;
        };
        sc.on_stage[0] = place(tomo_motor(rec, "sample_u")), sc.on_stage[1] = place(tomo_motor(rec, "sample_v"));
        sc.across_beam = place(rec.translation_y[1]);
    }
    // (a record that does not name the stages on the rotation says nothing against the same place)
    const auto apart = [&](int j, int k) {
        const ScanLayout& a = acq.scans[static_cast<std::size_t>(j)];
        const ScanLayout& b = acq.scans[static_cast<std::size_t>(k)];
        const double d = std::hypot(a.on_stage[0] - b.on_stage[0], a.on_stage[1] - b.on_stage[1]);
        return std::isnan(d) ? 0.0 : d;
    };
    {
        bool mosaic = false;
        double middle[2] = {0, 0};
        int known = 0;
        for (int k = 0; k < S; ++k) {
            mosaic = mosaic || apart(0, k) > 1.0;
            const ScanLayout& sc = acq.scans[static_cast<std::size_t>(k)];
            if (!std::isnan(sc.on_stage[0]) && !std::isnan(sc.on_stage[1])) middle[0] += sc.on_stage[0], middle[1] += sc.on_stage[1], ++known;
        }
        acq.anchor = -1;
        double nearest = 0;
        for (int k = 0; k < S; ++k) {
            const ScanLayout& sc = acq.scans[static_cast<std::size_t>(k)];
            if (!axis_in_field(sc, cols)) continue;
            const double away = mosaic && known ? std::hypot(sc.on_stage[0] - middle[0] / known, sc.on_stage[1] - middle[1] / known) : 0.0;
            if (acq.anchor < 0 || away < nearest) acq.anchor = k, nearest = away;  // of scans at one place: the first
        }
        if (acq.anchor < 0) throw Error("geometry calibration: no scan has its rotation axis inside the detector's field");
        for (int k = 0; k < S; ++k) acq.scans[static_cast<std::size_t>(k)].moved_on_stage = apart(acq.anchor, k) > 1.0;
    }
    // a scan whose axis is not in its field: from where its content lies against a scan at the same place
    for (int k = 0; k < S; ++k) {
        ScanLayout& sc = acq.scans[static_cast<std::size_t>(k)];
        if (!std::isnan(sc.axis)) continue;
        int ref = -1;
        for (int j = 0; j < S && ref < 0; ++j)
            if (j != k && axis_in_field(acq.scans[static_cast<std::size_t>(j)], cols) && apart(j, k) <= 1.0) ref = j;
        if (ref < 0) {
            say(log, fmt("  scan %d (%s): no scan with its axis in the field shares its place on the rotation stage: its axis column is not known", k, sc.name.c_str()));
            continue;
        }
        const ScanLayout& other = acq.scans[static_cast<std::size_t>(ref)];
        const CoarseMove found = coarse_move(session, acq, ref, k);
        const double motors = sc.across_beam - other.across_beam;
        if (found.score < 0.2) {
            say(log, fmt("  scan %d (%s): its content was not found in scan %d (profiles at equal frames correlate %.2f): its axis column is not known", k, sc.name.c_str(), ref, found.score));
            continue;
        }
        sc.axis = other.axis + found.move;
        say(log, fmt("  scan %d (%s): its content lies %+d columns from scan %d's (profiles at equal frames correlate %.2f; the stage across the beam moved %+.1f px by its motor): axis column about %.1f",
                     k, sc.name.c_str(), found.move, ref, found.score, motors, sc.axis));
    }
    acq.width = 512;
    for (ScanLayout& sc : acq.scans) {
        if (axis_in_field(sc, cols)) acq.width = std::min(acq.width, window_width(sc.axis, cols));
        const StretchLayout* ref = &sc.stretches.front();
        for (StretchLayout& st : sc.stretches) {
            for (double& v : st.course.rows) v *= acq.row_sign;
            if (st.helix) ref = &st;
        }
        sc.mid_height = ref->course.rows[static_cast<std::size_t>((ref->first + ref->end) / 2)];
    }
    say(log, fmt("  %d x %d px, %d frames per turn; row sense %+.0f: %s; windows of %d columns; anchor: scan %d (%s)", rows, cols, N, acq.row_sign, said.c_str(), acq.width, acq.anchor,
                 acq.scans[static_cast<std::size_t>(acq.anchor)].name.c_str()));
    return acq;
}

std::string acquisition_json(const Acquisition& acq) {
    Json scans = Json::list();
    for (const ScanLayout& sc : acq.scans) {
        Json stretches = Json::list();
        for (const StretchLayout& st : sc.stretches) {
            Json d = Json::dict();
            d.set("name", Json::of(st.name)).set("index", count(st.index)).set("helix", Json::of_bool(st.helix)).set("frames", count(static_cast<long long>(st.course.rows.size())));
            d.set("first", count(st.first)).set("end", count(st.end)).set("offset", count(st.offset)).set("blocked_frames", count(static_cast<long long>(st.blocked.frames.size())));
            stretches.add(d);
        }
        Json d = Json::dict();
        d.set("name", Json::of(sc.name)).set("mid_height", num(sc.mid_height)).set("coarse_axis", num(sc.axis)).set("axis_in_field", Json::of_bool(axis_in_field(sc, acq.cols)));
        d.set("on_stage_px", Json::list().add(num(sc.on_stage[0])).add(num(sc.on_stage[1]))).set("across_beam_px", num(sc.across_beam)).set("moved_on_stage", Json::of_bool(sc.moved_on_stage));
        scans.add(d.set("stretches", stretches));
    }
    Json out = Json::dict();
    out.set("rows", count(acq.rows)).set("cols", count(acq.cols)).set("frames_per_turn", count(acq.per_turn)).set("pixel_um", num(acq.pixel_um)).set("row_sign", num(acq.row_sign));
    out.set("row_time", num(acq.row_time)).set("frame_period_s", num(acq.period_s)).set("fan", num(acq.fan)).set("per_depth", num(acq.per_depth)).set("window_columns", count(acq.width));
    out.set("anchor", count(acq.anchor)).set("scans", scans);
    return out.text();
}

std::vector<Planned> plan(const Acquisition& acq, const PlanCounts& n) {
    const int R = acq.rows, C = acq.cols, N = acq.per_turn, W = acq.width;
    const int nrows = std::min(256, R / 4);
    if (n.angles_per_turn < 1 || n.max_positions < 2) throw Error("geometry calibration: a plan needs an angle per turn and two positions");
    std::vector<Planned> out;
    const auto add_between = [&](const char* kind, int scan_a, int a, int scan_b, int b, const ViewRequest& q, bool from_block) {
        Planned p;
        p.kind = kind, p.scan_a = scan_a, p.scan_b = scan_b, p.a = a, p.b = b, p.from_block = from_block;
        p.request = q;
        p.request.label = kind;
        out.push_back(p);
    };
    const auto add = [&](const char* kind, int scan, int a, int b, const ViewRequest& q, bool from_block = false) { add_between(kind, scan, a, scan, b, q, from_block); };
    // Frames from p_lo to p_hi at the angles a turn / angles_per_turn apart. Where that is more than `most`, the
    // positions are spread evenly over the span and each moved to the nearest frame at its angle: the angles in
    // turn, so that they stay evenly spread around the turn.
    const auto around_the_turn = [&](std::int64_t p_lo, std::int64_t p_hi, int most) {
        const double base = static_cast<double>(N) / n.angles_per_turn, span = static_cast<double>(p_hi - p_lo);
        std::vector<std::int64_t> positions;
        if (std::floor(span / base) + 1 <= most) {
            for (double at = 0; at <= span; at += base) positions.push_back(p_lo + static_cast<std::int64_t>(std::llround(at)));
            return positions;
        }
        const int count = std::max(n.angles_per_turn, most - most % n.angles_per_turn);
        for (int j = 0; j < count; ++j) {
            const double phase = (j % n.angles_per_turn) * base;
            double at = N * std::round((j * span / (count - 1) - phase) / N) + phase;
            if (at < 0) at += N;
            if (at > span) at -= N;
            if (at >= 0 && at <= span) positions.push_back(p_lo + static_cast<std::int64_t>(std::llround(at)));
        }
        std::sort(positions.begin(), positions.end());
        positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
        return positions;
    };
    // First rows of A's block for which it and B's, Nr rows higher and with its margins, lie on the detector: as
    // many bands as fit side by side, up to five.
    const auto bands = [&](int Nr, int nr) {
        const int m = 40, a0 = std::max(0, m - Nr), a1 = std::min(R - nr, R - nr - m - Nr);
        if (a1 < a0) return std::vector<std::int64_t>();
        const int count = std::clamp(1 + (a1 - a0) / nr, 1, 5);
        if (count == 1) return std::vector<std::int64_t>{(a0 + a1) / 2};
        return whole(linspace(a0, a1, count));
    };
    const std::vector<std::int64_t> across = whole(linspace(13, C - W - 13, std::clamp((C - 26) / W, 1, 5)));
    const auto same_columns = [&](const StretchLayout& a, const StretchLayout& b, int scan, std::int64_t frame, std::int64_t hint, std::int64_t col, int rows_, int row_a) {
        ViewRequest q;
        q.scan_a = q.scan_b = scan;
        q.stretch_a = a.index, q.stretch_b = b.index;
        q.frame_a = frame, q.frame_hint = hint;
        q.frames = 32, q.rows = rows_, q.cols = W;
        q.row_a = row_a, q.col_a = q.col_b = static_cast<int>(col);
        return q;
    };
    for (int k = 0; k < static_cast<int>(acq.scans.size()); ++k) {
        const ScanLayout& sc = acq.scans[static_cast<std::size_t>(k)];
        int helix = -1;
        for (int a = 0; a < static_cast<int>(sc.stretches.size()); ++a)
            if (sc.stretches[static_cast<std::size_t>(a)].helix) helix = a;
        // opposite views
        if (axis_in_field(sc, C)) {
            const std::vector<std::int64_t> windows = mirrored_windows(sc.axis, C, W);
            if (windows.empty()) throw Error("the rotation axis of scan " + sc.name + " is too close to the detector's edge for opposite views");
            for (int a = 0; a < static_cast<int>(sc.stretches.size()); ++a) {
                const StretchLayout& st = sc.stretches[static_cast<std::size_t>(a)];
                const std::int64_t p_lo = st.first + 16, p_hi = st.end - N / 2 - 80;
                if (p_hi < p_lo) continue;
                for (const std::int64_t p : around_the_turn(p_lo, p_hi, n.max_positions)) {
                    const int Nr = static_cast<int>(std::lround(st.course.rows[static_cast<std::size_t>(p + 16 + N / 2)] - st.course.rows[static_cast<std::size_t>(p + 16)]));
                    for (const std::int64_t row_a : bands(Nr, nrows))
                        for (const std::int64_t ca : windows) add("opposite", k, a, a, opposite_request(st, k, p, N, sc.axis, W, ca, nrows, static_cast<int>(row_a), acq.fan), st.helix);
                }
            }
        }
        if (helix < 0) continue;
        const StretchLayout& hx = sc.stretches[static_cast<std::size_t>(helix)];
        // the helix a turn apart, where a turn's travel leaves a patch of rows in common
        if (hx.end - hx.first > N + 200) {
            const double travel = std::fabs(hx.course.rows[static_cast<std::size_t>(hx.first + N)] - hx.course.rows[static_cast<std::size_t>(hx.first)]);
            if (R - travel >= 128)
                for (const std::int64_t p : whole(linspace(static_cast<double>(hx.first + 16), static_cast<double>(hx.end - N - 100), n.n_turn)))
                    for (const std::int64_t ca : across) add("turn", k, helix, helix, same_columns(hx, hx, k, p, N, ca, 128, -1));
        }
        // each rest turn against the helix at equal angle, in the turn of the helix nearest the rest turn's height
        for (int a = 0; a < static_cast<int>(sc.stretches.size()); ++a) {
            const StretchLayout& rt = sc.stretches[static_cast<std::size_t>(a)];
            if (rt.helix || hx.end - hx.first < rt.end - rt.first) continue;
            const double Hr = rt.course.rows[static_cast<std::size_t>(rt.first)];
            std::int64_t nearest = hx.first;
            for (std::int64_t j = hx.first; j < hx.end; ++j)
                if (std::fabs(hx.course.rows[static_cast<std::size_t>(j)] - Hr) < std::fabs(hx.course.rows[static_cast<std::size_t>(nearest)] - Hr)) nearest = j;
            const std::int64_t hint = std::clamp(nearest - rt.first, hx.first - rt.first, hx.end - rt.end);
            for (const std::int64_t p : whole(linspace(static_cast<double>(rt.first + 16), static_cast<double>(rt.end - 50), n.n_rest))) {
                const int Nr = static_cast<int>(std::lround(hx.course.rows[static_cast<std::size_t>(p + 16 + hint)] - rt.course.rows[static_cast<std::size_t>(p + 16)]));
                if (std::abs(Nr) < 48 || std::abs(Nr) > R - 128 - 60) continue;  // too near (a pattern fixed to the detector would register) or off the detector
                const int nr = std::abs(Nr) <= R - nrows - 84 ? nrows : 128;
                for (const std::int64_t row_a : bands(Nr, nr))
                    for (const std::int64_t ca : across) add("resthelix", k, a, helix, same_columns(rt, hx, k, p, hint, ca, nr, static_cast<int>(row_a)));
            }
        }
    }
    // one scan against another with the same place on the rotation stage: the stretches of the same name at equal
    // angle, across the columns both hold
    for (int j = 0; j < static_cast<int>(acq.scans.size()); ++j)
        for (int k = j + 1; k < static_cast<int>(acq.scans.size()); ++k) {
            const ScanLayout& A = acq.scans[static_cast<std::size_t>(j)];
            const ScanLayout& B = acq.scans[static_cast<std::size_t>(k)];
            if (std::isnan(A.axis) || std::isnan(B.axis) || A.moved_on_stage || B.moved_on_stage) continue;
            const int move = static_cast<int>(std::lround(B.axis - A.axis));  // B holds A's content this many columns higher
            const int lo = std::max(0, -move) + 13, hi = std::min(C - W, C - W - move) - 13;
            if (hi < lo) continue;
            const std::vector<std::int64_t> windows = whole(linspace(lo, hi, std::clamp((hi - lo + 2 * W - 1) / W, 1, 6)));
            for (int a = 0; a < static_cast<int>(A.stretches.size()); ++a)
                for (int b = 0; b < static_cast<int>(B.stretches.size()); ++b) {
                    const StretchLayout& sa = A.stretches[static_cast<std::size_t>(a)];
                    const StretchLayout& sb = B.stretches[static_cast<std::size_t>(b)];
                    if (sa.name != sb.name) continue;
                    const std::int64_t p_lo = std::max(sa.first, sb.first) + 16, p_hi = std::min(sa.end, sb.end) - 80;
                    if (p_hi < p_lo) continue;
                    const std::vector<std::int64_t> positions =
                        sa.helix ? around_the_turn(p_lo, p_hi, n.pair_positions) : whole(linspace(static_cast<double>(p_lo), static_cast<double>(p_hi), n.n_pair_rest));
                    for (const std::int64_t p : positions) {
                        const int Nr = static_cast<int>(std::lround(sb.course.rows[static_cast<std::size_t>(p + 16)] - sa.course.rows[static_cast<std::size_t>(p + 16)]));
                        if (std::abs(move) < 48 && std::abs(Nr) < 48) continue;  // a pattern fixed to the detector would register
                        for (const std::int64_t row_a : bands(Nr, nrows))
                            for (const std::int64_t ca : windows) {
                                ViewRequest q;
                                q.scan_a = j, q.scan_b = k;
                                q.stretch_a = sa.index, q.stretch_b = sb.index;
                                q.frame_a = p, q.frame_hint = 0;
                                q.frames = 32, q.rows = nrows, q.cols = W;
                                q.row_a = static_cast<int>(row_a), q.col_a = static_cast<int>(ca), q.col_b = static_cast<int>(ca) + move;
                                add_between("pair", j, a, k, b, q, true);
                            }
                    }
                }
        }
    return out;
}

std::vector<Planned> plan_tiles(const Acquisition& acq, const Parameters& places, const std::vector<std::array<int, 2>>& pairs, const PlanCounts& n) {
    const int R = acq.rows, C = acq.cols, N = acq.per_turn, W = acq.width, S = static_cast<int>(acq.scans.size());
    const int nrows = std::min(256, R / 4);
    if (places.scans.size() != acq.scans.size()) throw Error("geometry calibration: the places and the acquisition differ in their scans");
    if (n.tile_positions < 2 || n.tile_angles < 1 || n.tile_bands < 1 || n.tile_windows < 1) throw Error("geometry calibration: a plan of tiles needs two positions, an angle per turn, a band and a window");
    std::vector<Planned> out;
    for (const auto& pair : pairs) {
        const int j = pair[0], k = pair[1];
        if (j < 0 || k < 0 || j >= S || k >= S || j == k) throw Error("geometry calibration: a pair of tiles names two of the acquisition's scans");
        const ScanLayout& A = acq.scans[static_cast<std::size_t>(j)];
        const ScanLayout& B = acq.scans[static_cast<std::size_t>(k)];
        const ScanParameters& pa = places.scans[static_cast<std::size_t>(j)];
        const ScanParameters& pb = places.scans[static_cast<std::size_t>(k)];
        int ha = -1, hb = -1;
        for (int a = 0; a < static_cast<int>(A.stretches.size()); ++a)
            if (A.stretches[static_cast<std::size_t>(a)].helix) ha = a;
        for (int b = 0; b < static_cast<int>(B.stretches.size()); ++b)
            if (B.stretches[static_cast<std::size_t>(b)].helix) hb = b;
        if (ha < 0 || hb < 0) continue;
        const StretchLayout& sa = A.stretches[static_cast<std::size_t>(ha)];
        const StretchLayout& sb = B.stretches[static_cast<std::size_t>(hb)];
        // where B holds A's content: d_k - d_j as a length and a direction, the heights' difference beyond the record's
        const double d0 = pb.d[0] - pa.d[0], d1 = pb.d[1] - pa.d[1], apart = std::hypot(d0, d1);
        const double phase_deg = std::atan2(d1, d0) * (180.0 / kPi);
        // the rows by which B holds, in its frame q, what A holds in its frame p: the model's, without what a turn does
        const auto higher = [&](std::int64_t q, std::size_t p) {
            return static_cast<int>(std::lround(pb.height + (1.0 + pb.axial_scale) * sb.course.rows[static_cast<std::size_t>(q)] - pa.height - (1.0 + pa.axial_scale) * sa.course.rows[p]));
        };
        // the offset's course over a block's frames (32 and the registration's margins), which the cut must hold
        const double step_deg = sa.course.angle_deg[static_cast<std::size_t>(sa.first + 1)] - sa.course.angle_deg[static_cast<std::size_t>(sa.first)];
        if (step_deg == 0.0) continue;
        const double step = std::fabs(step_deg) * (kPi / 180.0);
        const int edge = 13 + static_cast<int>(std::ceil(apart * step * 30.0));
        // B's frame less A's at equal angle, by the records (Session::measure finds the frames themselves from them)
        const std::int64_t shift = static_cast<std::int64_t>(std::llround((sa.course.angle_deg[static_cast<std::size_t>(sa.first)] - sb.course.angle_deg[static_cast<std::size_t>(sb.first)]) / step_deg)) +
                                   (sb.first - sa.first);
        const std::int64_t p_lo = sa.first + 16, p_hi = sa.end - 80;
        if (p_hi < p_lo) continue;
        // positions evenly around the turn, as the opposite views' (the angles in turn along the helix)
        std::vector<std::int64_t> positions;
        {
            const double base = static_cast<double>(N) / n.tile_angles, span = static_cast<double>(p_hi - p_lo);
            const int count = std::max(n.tile_angles, n.tile_positions - n.tile_positions % n.tile_angles);
            for (int i = 0; i < count; ++i) {
                const double at_angle = (i % n.tile_angles) * base;
                double at = N * std::round((i * span / std::max(1, count - 1) - at_angle) / N) + at_angle;
                if (at < 0) at += N;
                if (at > span) at -= N;
                if (at >= 0 && at <= span) positions.push_back(p_lo + static_cast<std::int64_t>(std::llround(at)));
            }
            std::sort(positions.begin(), positions.end());
            positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
        }
        const auto bands = [&](int Nr) {
            const int m = 40, a0 = std::max(0, m - Nr), a1 = std::min(R - nrows, R - nrows - m - Nr);
            if (a1 < a0) return std::vector<std::int64_t>();
            const int count = std::clamp(1 + (a1 - a0) / nrows, 1, n.tile_bands);
            if (count == 1) return std::vector<std::int64_t>{(a0 + a1) / 2};
            return whole(linspace(a0, a1, count));
        };
        const auto windows = [&](int lo, int hi) {
            if (hi < lo) return std::vector<std::int64_t>();
            const int count = std::clamp((hi - lo + 2 * W - 1) / W, 1, n.tile_windows);
            if (count == 1) return std::vector<std::int64_t>{(lo + hi) / 2};
            return whole(linspace(lo, hi, count));
        };
        for (const std::int64_t p : positions) {
            const std::size_t mid = static_cast<std::size_t>(p + 16);
            const double th = sa.course.angle_deg[mid] * (kPi / 180.0);
            const double sway = d0 * std::cos(th) + d1 * std::sin(th);  // (d_k - d_j) . n(th)
            // at equal angle
            const std::int64_t q_same = p + shift;
            if (q_same - 16 >= sb.first && q_same + 80 <= sb.end) {
                const int move = static_cast<int>(std::lround(pb.axis - pa.axis + sway));
                const int Nr = higher(q_same + 16, mid);
                if (std::abs(move) >= 48 || std::abs(Nr) >= 48)  // (else a pattern fixed to the detector would register)
                    for (const std::int64_t row_a : bands(Nr))
                        for (const std::int64_t ca : windows(std::max(0, -move) + edge, std::min(C - W, C - W - move) - edge)) {
                            Planned t;
                            t.kind = "tiles", t.scan_a = j, t.scan_b = k, t.a = ha, t.b = hb, t.from_block = true;
                            ViewRequest& q = t.request;
                            q.label = t.kind;
                            q.scan_a = j, q.scan_b = k, q.stretch_a = sa.index, q.stretch_b = sb.index;
                            q.frame_a = p, q.frame_hint = shift;
                            q.frames = 32, q.rows = nrows, q.cols = W;
                            q.row_a = static_cast<int>(row_a), q.col_a = static_cast<int>(ca);
                            q.col_b = static_cast<int>(ca) + static_cast<int>(std::lround(pb.axis - pa.axis));
                            q.sway_px = apart, q.sway_phase_deg = phase_deg;
                            q.sway_row_deg = acq.row_time * step_deg;
                            out.push_back(t);
                        }
            }
            // half a turn on, mirrored: B's columns (before the reversal) are sum - A's, sum = axis_j + axis_k - (d_k - d_j) . n(th_a)
            // (half a turn later; where B's frames end before that, half a turn earlier)
            std::int64_t half = N / 2;
            if (p + shift + half + 80 > sb.end) half = -half;
            const std::int64_t q_opp = p + shift + half;
            if (q_opp - 16 >= sb.first && q_opp + 80 <= sb.end) {
                const int sum = static_cast<int>(std::lround(pa.axis + pb.axis - sway));
                const int Nr = higher(q_opp + 16, mid);
                for (const std::int64_t row_a : bands(Nr))
                    for (const std::int64_t ca : windows(std::max(0, sum - (W - 1) - (C - W)) + edge, std::min(C - W, sum - (W - 1)) - edge)) {
                        Planned t;
                        t.kind = "tiles_opposite", t.scan_a = j, t.scan_b = k, t.a = ha, t.b = hb, t.from_block = true;
                        ViewRequest& q = t.request;
                        q.label = t.kind;
                        q.scan_a = j, q.scan_b = k, q.stretch_a = sa.index, q.stretch_b = sb.index;
                        q.frame_a = p, q.frame_hint = shift + half;
                        q.angle_offset_deg = 180.0, q.mirror = true;
                        q.frames = 32, q.rows = nrows, q.cols = W;
                        q.row_a = static_cast<int>(row_a), q.col_a = static_cast<int>(ca);
                        // (the sway is of B's own angle, half a turn on: there it is +(d_k - d_j) . n, the same statement)
                        q.col_b = static_cast<int>(std::lround(pa.axis + pb.axis)) - (W - 1) - static_cast<int>(ca);
                        q.sway_px = apart, q.sway_phase_deg = phase_deg;
                        q.sway_row_deg = acq.row_time * step_deg;
                        q.frame_shear = 2 * acq.fan;
                        out.push_back(t);
                    }
            }
        }
    }
    return out;
}

std::vector<Comparison> measure(Session& session, const Acquisition& acq, const std::vector<Planned>& planned, Device device, int patch, std::vector<int>* kept) {
    session.set_detector_map(DetectorMap{});  // the frames as stored: the map is the model's
    // the blocks in the order of their frames (neighbours read the same part of the files), those whose patches
    // start from the block apart
    std::vector<std::size_t> order(planned.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) {
        const Planned& a = planned[x];
        const Planned& b = planned[y];
        if (a.from_block != b.from_block) return a.from_block < b.from_block;
        if (a.scan_a != b.scan_a) return a.scan_a < b.scan_a;
        if (a.a != b.a) return a.a < b.a;
        return a.request.frame_a < b.request.frame_a;
    });
    std::vector<Comparison> out;
    if (kept) kept->clear();
    const auto stretch = [&](int scan, int a) -> const StretchLayout& { return acq.scans.at(static_cast<std::size_t>(scan)).stretches.at(static_cast<std::size_t>(a)); };
    for (std::size_t i0 = 0; i0 < order.size();) {
        std::size_t i1 = i0;
        while (i1 < order.size() && i1 - i0 < 400 && planned[order[i1]].from_block == planned[order[i0]].from_block) ++i1;
        std::vector<ViewRequest> reqs;
        for (std::size_t i = i0; i < i1; ++i) reqs.push_back(planned[order[i]].request);
        std::vector<std::pair<std::size_t, PatchMeasurement>> results;
        if (patch) {
            for (const PatchMeasurement& r : session.measure_patches(reqs, patch, patch, acq.row_sign, device, planned[order[i0]].from_block))
                if (r.i >= 0) results.emplace_back(order[i0 + static_cast<std::size_t>(r.request)], r);
        } else {
            const std::vector<ViewMeasurement> got = session.measure(reqs, acq.row_sign, device);
            for (std::size_t i = 0; i < got.size(); ++i) {
                PatchMeasurement r;
                r.m = got[i];
                results.emplace_back(order[i0 + i], r);
            }
        }
        for (const auto& [index, r] : results) {
            const Planned& p = planned[index];
            const ViewMeasurement& m = r.m;
            if (!(m.planned && m.ok && m.correlation > 0.3)) continue;
            if (stretch(p.scan_a, p.a).blocked.holds(static_cast<double>(m.frame_a), m.row_a - 8, m.row_a + m.rows + 8)) continue;
            if (stretch(p.scan_b, p.b).blocked.holds(static_cast<double>(m.frame_b), m.row_a + m.row_offset - 8, m.row_a + m.row_offset + m.rows + 8)) continue;
            Comparison c = comparison_of(acq, p.kind, p.scan_a, p.a, p.scan_b, p.b, p.request, m);
            std::copy(r.curvature, r.curvature + 6, c.curvature);
            out.push_back(c);
            if (kept) kept->push_back(static_cast<int>(index));
        }
        i0 = i1;
    }
    return out;
}

// ---------------------------------------------------------------- the step

void to_stored_pixels(Acquisition& acq, GeometrySolution& solution, GeometryReport& told, int rows, int cols, int f) {
    if (f < 1) throw Error("geometry: a calibration is at 1 / f of the stored frames, f at least 1");
    if (solution.parameters.scans.size() != acq.scans.size() || told.scans.size() != acq.scans.size()) throw Error("geometry: the solution, the report and the acquisition differ in their scans");
    const double F = f;
    acq.rows = rows, acq.cols = cols;
    acq.pixel_um /= F, acq.row_time /= F, acq.fan /= F, acq.per_depth /= F, acq.width *= f;
    for (ScanLayout& sc : acq.scans) {
        sc.mid_height *= F, sc.axis *= F, sc.on_stage[0] *= F, sc.on_stage[1] *= F, sc.across_beam *= F;
        for (StretchLayout& st : sc.stretches) {
            for (double& v : st.course.rows) v *= F;
            st.blocked = Blocked{};  // (its rows were the reduced frames')
        }
    }
    Parameters& p = solution.parameters;
    for (double& v : p.stage.height_once) v *= F;
    for (double& v : p.stage.axis_twice) v *= F;
    p.map.c0 *= F, p.map.cs *= F, p.map.v0 *= F, p.map.vs *= F;
    p.about_height *= F;
    for (double& v : p.map.eu) v *= F;
    for (double& v : p.map.ev) v *= F;
    for (ScanParameters& q : p.scans) {
        q.axis = q.axis * F + p.roll * (F - 1.0) / 2.0;
        q.d[0] *= F, q.d[1] *= F, q.height *= F, q.height_once[0] *= F, q.height_once[1] *= F;
        for (double& v : q.stretch_height) v *= F;
    }
    for (std::array<double, 3>& r : solution.residual) r[1] *= F, r[2] *= F;
    for (std::array<double, 3>& r : solution.lens_residual) r[0] *= F, r[1] *= F, r[2] *= F;
    for (GeometrySolution::Left& l : solution.left)
        for (int j = l.kind == "lens" ? 0 : 1; j < 3; ++j) l.rms[j] *= F, l.median[j] *= F;  // (a comparison's first is its frames)
    for (ScanReport& r : told.scans) {
        for (int j = 1; j < 3; ++j) r.mean[j] *= F, r.left[j] *= F, r.once[j][0] *= F, r.once[j][1] *= F, r.twice[j][0] *= F, r.twice[j][1] *= F;
        r.row_time /= F, r.rows_apart *= F, r.departure *= F, r.departure_between_views *= F, r.from_anchor *= F;
    }
    for (PairReport& r : told.pairs)
        for (int j = 1; j < 3; ++j) r.mean[j] *= F, r.left[j] *= F, r.rms[j] *= F, r.median[j] *= F, r.once[j][0] *= F, r.once[j][1] *= F;
    told.axis.range *= F, told.axis.sd *= F, told.axis.mean = told.axis.mean * F + p.roll * (F - 1.0) / 2.0;
}

void save_text(const std::string& text, const std::string& path) {
    const std::string tmp = path + ".tmp" + std::to_string(static_cast<long>(::getpid()));
    {
        std::ofstream o(tmp);
        if (!o) throw Error("cannot write " + tmp);
        o << text << "\n";
        if (!o) throw Error("cannot write " + tmp);
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) throw Error("cannot write " + path);
}

std::string calibrate_geometry(const std::vector<std::string>& scans, const Options& options) {
    if (scans.empty()) throw Error("the geometry calibration takes the master files of the scans of one sample");
    const Log& log = options.log;
    const auto t_start = std::chrono::steady_clock::now();
    const auto seconds_since = [](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); };
    if (options.reduce < 0 || options.reduce > 3) throw Error("geometry calibration: reduce is 0, 1, 2 or 3");
    Session session(scans, options.threads, options.reduce, options.projections);
    const int reduced_by = 1 << options.reduce;
    if (options.reduce || !options.projections.empty())
        say(log, fmt("frames at 1/%d of the stored ones%s; the lines below up to the solve are in those frames' pixels, what is found and written in stored pixels", reduced_by,
                     options.projections.empty() ? "" : (", from " + options.projections).c_str()));
    say(log, "layout");
    Acquisition acq = read_acquisition(session, options.device, log);
    if (!(acq.period_s > 0)) throw Error("geometry calibration: the record does not hold the frames' period");
    double row_time = options.row_time;
    Json timing;
    if (std::isnan(row_time)) {
        say(log, "rows' timing");
        timing = Json::parse(row_time_from_refill(session, acq, log));
        if (!timing.is(Json::Object))
            throw Error("no stretch on disk holds a refill to measure the rows' timing from: give the row time (seconds per stored row; 0 if it is not known) for this camera configuration");
        row_time = timing.find("seconds_per_row")->number;
    }
    acq.row_time = row_time / acq.period_s;
    say(log, "comparisons");
    auto t = std::chrono::steady_clock::now();
    PlanCounts how_many;
    how_many.angles_per_turn = std::min(48, how_many.angles_per_turn * reduced_by), how_many.max_positions *= reduced_by;
    const std::vector<Planned> planned = plan(acq, how_many);
    if (options.patch < 32) throw Error("geometry calibration: patches of at least 32 x 32");
    const int patch = std::max(32, options.patch / reduced_by);
    measure_timing(true);
    const std::vector<Comparison> comparisons = measure(session, acq, planned, options.device, patch);
    const MeasureTiming spent = measure_timing();
    std::map<std::string, long long> accepted;
    for (const Comparison& c : comparisons) ++accepted[c.kind];
    std::string said;
    for (const auto& [kind, n] : accepted) said += fmt("%s%s %lld", said.empty() ? "" : ", ", kind.c_str(), n);
    say(log, fmt("  %zu blocks planned, %zu patches of %d accepted (%s) in %.0f s (reading %.0f, converting %.0f, registering %.0f)", planned.size(), comparisons.size(), patch, said.c_str(),
                 seconds_since(t), spent.read, spent.integrals, spent.registration));
    say(log, "solve");
    Parameters start = empty_parameters(acq);
    for (std::size_t k = 0; k < acq.scans.size(); ++k)
        if (!std::isnan(acq.scans[k].axis)) start.scans[k].axis = acq.scans[k].axis;
    GeometryOptions how;
    if (options.order > 0) {
        start.map = geometry::empty_map(acq.rows, acq.cols, options.order);
        how.freed.push_back("map");
    }
    GeometrySolution solution = solve_geometry(acq, comparisons, start, how);
    GeometryReport told = report(acq, solution, comparisons);
    if (options.reduce) {
        const Stretch& first = session.record(0).stretches.at(static_cast<std::size_t>(acq.scans[0].stretches.at(0).index));
        to_stored_pixels(acq, solution, told, first.rows, first.cols, reduced_by);
        row_time = acq.row_time * acq.period_s;  // (the same seconds per stored row)
    }
    for (const GeometrySolution::Left& l : solution.left)
        if (l.scan_a >= 0)
            say(log, fmt("  %-9s scan %d%s: %5zu left %.3f frames, %.3f rows, %.3f px rms", l.kind.c_str(), l.scan_a, l.scan_b == l.scan_a ? "" : fmt(" against %d", l.scan_b).c_str(), l.n, l.rms[0], l.rms[1],
                         l.rms[2]));
    for (std::size_t k = 0; k < acq.scans.size(); ++k) {
        const ScanParameters& p = solution.parameters.scans[k];
        say(log, fmt("  scan %zu (%s): axis column %.3f, %+.2e per row of travel; axial scale %+.5f; height %+.3f rows", k, acq.scans[k].name.c_str(), p.axis, p.axis_slope, p.axial_scale, p.height));
    }
    if (!solution.unused.empty()) {
        std::string names;
        for (const std::string& n : solution.unused) names += (names.empty() ? "" : ", ") + n;
        say(log, "  no comparison gives: " + names);
    }
    say(log, "report");
    const auto polar = [](const double* v) { return std::hypot(v[0], v[1]); };
    for (std::size_t k = 0; k < acq.scans.size(); ++k) {
        const ScanReport& r = told.scans[k];
        const ScanParameters& p = solution.parameters.scans[k];
        if (r.stretch >= 0) {
            say(log, fmt("  scan %zu (%s): %zu opposite views of %s leave, once | twice per turn: rows %.3f | %.3f, columns %.3f | %.3f px, frames %.3f | %.3f about %+.3f", k, acq.scans[k].name.c_str(), r.n,
                         acq.scans[k].stretches[static_cast<std::size_t>(r.stretch)].name.c_str(), polar(r.once[1]), polar(r.twice[1]), polar(r.once[2]), polar(r.twice[2]), polar(r.once[0]), polar(r.twice[0]),
                         r.mean[0]));
            if (std::isfinite(r.row_time))
                say(log, fmt("    the views lie %.0f rows apart: by their frames a row is exposed %+.3f us after the one before (the model has %+.3f)", r.rows_apart, r.row_time * acq.period_s * 1e6, row_time * 1e6));
            if (std::isfinite(r.height_slope))
                say(log, fmt("    the recorded height leaves a straight line by %.3f rows rms (%.3f between the views); as the sample's motion it has the slope %.2f +- %.2f (1: it is; 0: the line is)", r.departure,
                             r.departure_between_views, r.height_slope, r.height_slope_error));
        }
        if (std::isfinite(r.from_anchor))
            say(log, fmt("  scan %zu (%s): %.0f px from the anchor; its own views' height once per turn %.3f rows: the axis leans at least %.0f urad towards the beam (its direction less the scan's: %+.0f deg)", k,
                         acq.scans[k].name.c_str(), r.from_anchor, polar(p.height_once), r.pitch * 1e6, r.pitch_phase_deg));
    }
    if (told.axis.n > 1)
        say(log, fmt("  over the %d scans with opposite views: axis column %.3f px (%.3f rms), its slope %.2e per row (%.2e rms), axial scale %.0f ppm (%.0f rms, mean %.0f)", told.axis.n, told.axis.range, told.axis.sd,
                     told.axis_slope.range, told.axis_slope.sd, told.axial_scale.range * 1e6, told.axial_scale.sd * 1e6, told.axial_scale.mean * 1e6));
    if (acq.scans.size() > 1) say(log, fmt("  %zu pairs of scans reconstruct a part of the sample in common; %zu triangles of them", told.neighbours.edges.size(), told.neighbours.triangles.size()));
    Json names = Json::list();
    for (const std::string& s : scans) names.add(Json::of(s));
    Json counts = Json::dict();
    for (const auto& [kind, n] : accepted) counts.set(kind, count(n));
    Json found = Json::parse(solution_json(acq, solution));
    Json parameters = Json::dict(), solve = Json::dict(), map;
    for (const auto& [key, value] : found.object) {
        if (key == "map") map = value;
        else if (key == "roll" || key == "pitch" || key == "stage" || key == "about_height" || key == "scans") parameters.set(key, value);
        else solve.set(key, value);
    }
    Json cal = Json::dict();
    cal.set("step", Json::of(std::string("geometry"))).set("model", Json::of(std::string("joint"))).set("scans", names).set("seconds", num(std::nearbyint(seconds_since(t_start) * 10) / 10));
    // what the frames were that it was measured on (the values below are in stored pixels whatever they were)
    cal.set("frames", Json::dict().set("reduce", count(options.reduce)).set("projections", Json::of(options.projections)));
    cal.set("layout", Json::parse(acquisition_json(acq)));
    cal.set("row_time", Json::dict().set("seconds_per_row", num(row_time)).set("frame_period_s", num(acq.period_s)).set("frames_per_row", num(acq.row_time)).set("measured", timing));
    cal.set("map", map).set("parameters", parameters).set("solution", solve).set("report", Json::parse(report_json(acq, told)));
    cal.set("comparisons", Json::dict().set("blocks", count(static_cast<long long>(planned.size()))).set("patch", count(patch * reduced_by)).set("accepted", counts));
    return cal.text();
}

}  // namespace helirec::calibration::geometry_calibration
