// helirec/calibration/geometry_calibration.hpp: the layout of the two scans, the plan of comparisons, a pass of
// them, the fit of the detector's map and the poses, and the step that runs them (calibrate_geometry).
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

DetectorMap session_map(const geometry::Map& m) {
    DetectorMap d;
    d.i = m.i, d.j = m.j, d.eu = m.eu, d.ev = m.ev;
    d.c0 = m.c0, d.cs = m.cs, d.v0 = m.v0, d.vs = m.vs;
    return d;
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

std::size_t argmax(const std::vector<double>& v) { return static_cast<std::size_t>(std::max_element(v.begin(), v.end()) - v.begin()); }

std::vector<std::int64_t> whole(const std::vector<double>& v) {
    std::vector<std::int64_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = static_cast<std::int64_t>(v[i]);  // towards zero, as numpy's astype(int)
    return out;
}

Json num(double v) { return Json::of(v); }
Json count(long long v) { return Json::whole_number(v); }
// how a solve ended, as the passes record it
Json solver_json(const Fit& f) {
    Json d = Json::dict();
    d.set("cost", num(f.solver.cost)).set("iterations", count(f.solver.iterations)).set("evaluations", count(f.solver.evaluations)).set("usable", Json::of_bool(f.solver.usable));
    d.set("termination", Json::of(f.solver.termination)).set("message", Json::of(f.solver.message)).set("solver_version", Json::of(f.solver.solver_version));
    d.set("rank", count(f.rank)).set("active", count(f.active)).set("unused", count(f.unused));
    return d;
}
// a straight line through points, or an error if there is none
Line line_through(const std::vector<double>& x, const std::vector<double>& y) {
    const Line l = fit_line(x, y, 0, x.size());
    if (!l.usable) throw Error(l.termination + ": " + l.message);
    return l;
}
Json numbers(const std::vector<double>& v) {
    Json j = Json::list();
    for (double x : v) j.add(num(x));
    return j;
}

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

bool Layout::is_blocked(int scan, int name, double frame, int row_lo, int row_hi) const {
    const auto it = blocked.find({scan, name});
    return it != blocked.end() && it->second.holds(frame, row_lo, row_hi);
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

// ---------------------------------------------------------------- the layout

Layout read_layout(Session& session, Device device, const Log& log) {
    Layout lay;
    lay.rows = session.frames(0).rows(), lay.cols = session.frames(0).cols();
    lay.pixel_um = pixel_um(session.record(0));
    const int rows = lay.rows, cols = lay.cols;
    // {rest1, helix, rest2} from the record's own order of its projection stretches
    for (int k = 0; k < 2; ++k) {
        const Record& rec = session.record(k);
        int rest = 0;
        bool helix = false;
        std::string found;
        for (std::size_t i = 0; i < rec.stretches.size(); ++i) {
            const Stretch& st = rec.stretches[i];
            if (st.role != Role::Projection || !st.latched()) continue;
            int a = -1;
            if (!st.z_center.empty()) {
                a = Helix, helix = true;
                found += " helix";
            } else {
                ++rest;
                if (rest <= 2) a = rest == 1 ? Rest1 : Rest2;
                found += " rest" + std::to_string(rest);
            }
            if (a >= 0) lay.stretch[k][a] = static_cast<int>(i), lay.length[k][a] = st.n_frames;
        }
        if (rest != 2 || !helix)
            throw Error("the geometry step needs a rest turn, a helix and a rest turn; the record of scan " + std::to_string(k) + " holds" + (found.empty() ? " none" : found));
    }
    {
        const std::vector<double> angle = stated_angle(session.record(0).stretches[static_cast<std::size_t>(lay.stretch[0][Helix])]);
        if (angle.size() < 2) throw Error("geometry calibration: the helix holds fewer than two frames");
        std::vector<double> d(angle.size() - 1);
        for (std::size_t i = 0; i + 1 < angle.size(); ++i) d[i] = angle[i + 1] - angle[i];
        lay.per_turn = static_cast<int>(std::nearbyint(360.0 / std::fabs(median_of(d))));
    }
    lay.width = static_cast<int>(std::pow(2.0, std::floor(std::log2(0.45 * cols))));
    // rows that hold something across the whole width
    for (int k = 0; k < 2; ++k)
        for (int a = 0; a < 3; ++a) {
            Blocked b = blocked_rows(session, k, lay.stretch[k][a], 0, lay.length[k][a], lay.per_turn, fmt("scan %d %s", k, kNames[a]), log);
            if (!b.frames.empty()) lay.blocked[{k, a}] = std::move(b);
        }
    // the axis column and the lateral move, coarsely: profiles of single frames in rows free of that
    std::vector<std::uint8_t> free_row(static_cast<std::size_t>(rows), 1);
    for (int k = 0; k < 2; ++k) {
        const auto it = lay.blocked.find({k, Rest1});
        if (it == lay.blocked.end()) continue;
        for (std::size_t i = 0; i < it->second.frames.size(); ++i)
            for (int r = 0; r < rows; ++r)
                if (it->second.mask[i * static_cast<std::size_t>(rows) + static_cast<std::size_t>(r)]) free_row[static_cast<std::size_t>(r)] = 0;
    }
    if (rows < 256) throw Error("geometry calibration: the detector has fewer than 256 rows");
    int r0 = std::clamp(rows / 2 - 128, 0, rows - 256);
    const auto all_free = [&](int from) {
        for (int r = from; r < from + 256; ++r)
            if (!free_row[static_cast<std::size_t>(r)]) return false;
        return true;
    };
    if (!all_free(r0)) {
        std::vector<int> runs;
        for (int r = 0; r + 256 <= rows; ++r)
            if (all_free(r)) runs.push_back(r);
        if (runs.empty()) throw Error("geometry calibration: no 256 rows of the first rest turns are free of what lies across the whole width");
        r0 = runs[runs.size() / 2];
    }
    const int N = lay.per_turn;
    std::vector<double> cc_axis(2 * static_cast<std::size_t>(cols), 0.0), cc_move(2 * static_cast<std::size_t>(cols), 0.0);
    for (const std::int64_t p : whole(linspace(N / 20, N / 2 - N / 20, 4))) {
        const std::vector<double> a = profile(session, 0, lay.stretch[0][Rest1], p, r0, 256, cols);
        std::vector<double> b = profile(session, 0, lay.stretch[0][Rest1], p + N / 2, r0, 256, cols);
        const std::vector<double> t = profile(session, 1, lay.stretch[1][Rest1], p, r0, 256, cols);
        std::reverse(b.begin(), b.end());
        add_lag(a, b, cc_axis);
        add_lag(a, t, cc_move);
    }
    lay.axis = (cols - 1 - (static_cast<double>(argmax(cc_axis)) - cols)) / 2.0;
    lay.move = static_cast<int>(argmax(cc_move)) - cols;
    // which way the rows run against the axis stage: the sense in which the helix a turn apart registers
    std::vector<ViewRequest> reqs;
    for (const std::int64_t p : whole(linspace(N, static_cast<double>(lay.length[0][Helix] - 2 * N - 64), 3))) {
        ViewRequest q;
        q.scan_a = q.scan_b = 0;
        q.stretch_a = q.stretch_b = lay.stretch[0][Helix];
        q.frame_a = p, q.frame_hint = N;
        q.frames = 32, q.rows = 128, q.cols = lay.width, q.row_a = -1;
        q.col_a = q.col_b = (cols - lay.width) / 2;
        reqs.push_back(q);
    }
    const RowSense sense = row_sense(session, reqs, device);
    if (sense.score < 0.3)
        throw Error("the helix a turn apart does not register in either sense of the rows: the travel per turn must be less than the detector's height");
    lay.row_sign = sense.sign;
    say(log, fmt("  %d x %d px, %d frames per turn; mirror column about %.1f; second scan %+d columns; row sense %+.0f; windows of %d columns", rows, cols, lay.per_turn,
                 lay.axis, lay.move, lay.row_sign, lay.width));
    return lay;
}

// ---------------------------------------------------------------- the comparisons

std::vector<Planned> plan(const Layout& lay, int n_opp, int n_rest, int n_turn, int n_pair) {
    const int R = lay.rows, C = lay.cols, N = lay.per_turn, W = lay.width;
    const int nrows = std::min(256, R / 4);
    const std::vector<std::int64_t> bands = whole(linspace(24, R - nrows - 24, 5));
    const int axis2 = static_cast<int>(std::nearbyint(2 * lay.axis));
    std::vector<Planned> out;
    const auto add = [&](const char* kind, int sa, int a, int sb, int b, std::int64_t frame, std::int64_t hint, std::int64_t col_a, std::int64_t col_b, int rows_,
                         std::int64_t row_a, bool mirror = false) {
        Planned p;
        p.kind = kind, p.scan_a = sa, p.scan_b = sb, p.a = a, p.b = b;
        ViewRequest& q = p.request;
        q.label = kind;
        q.scan_a = sa, q.scan_b = sb, q.stretch_a = lay.stretch[sa][a], q.stretch_b = lay.stretch[sb][b];
        q.frame_a = frame, q.frame_hint = hint;
        q.angle_offset_deg = mirror ? 180.0 : 0.0, q.mirror = mirror;
        q.frames = 32, q.rows = rows_, q.cols = W;
        q.row_a = static_cast<int>(row_a), q.col_a = static_cast<int>(col_a), q.col_b = static_cast<int>(col_b);
        out.push_back(p);
    };
    int lo = std::max(0, axis2 - (W - 1) - (C - W)), hi = std::min(C - W, axis2 - (W - 1));  // windows whose mirror image lies on the detector
    if (hi - lo < 32) throw Error("the rotation axis is too close to the detector's edge for opposite views in the first scan");
    for (const int a : {Rest1, Rest2})
        for (const std::int64_t p : whole(linspace(16, N / 2 - 80, n_opp)))
            for (const std::int64_t row_a : bands)
                for (const std::int64_t ca : whole(linspace(lo + 16, hi - 16, 3))) add("opposite", 0, a, 0, a, p, N / 2, ca, axis2 - (W - 1) - ca, nrows, row_a, true);
    const std::vector<std::int64_t> across = whole(linspace(13, C - W - 13, 3));
    for (int scan = 0; scan < 2; ++scan) {
        const std::vector<double>& hrows = lay.heights[scan][Helix];
        for (const int a : {Rest1, Rest2}) {
            const std::int64_t hint = a == Rest1 ? 0 : lay.length[scan][Helix] - lay.length[scan][a];
            const std::vector<double>& rrows = lay.heights[scan][a];
            for (const std::int64_t p : whole(linspace(16, static_cast<double>(lay.length[scan][a] - 50), n_rest))) {
                const int Nr = static_cast<int>(std::nearbyint(hrows.at(static_cast<std::size_t>(p + 16 + hint)) - rrows.at(static_cast<std::size_t>(p + 16))));
                if (std::abs(Nr) < 48 || std::abs(Nr) > R - 128 - 60) continue;  // too near (a pattern fixed to the detector would register) or off the detector
                const int nr = std::abs(Nr) <= R - nrows - 84 ? nrows : 128;
                const int m = 40;
                const int a0 = std::max(0, m - Nr), a1 = std::min(R - nr, R - nr - m - Nr);
                if (a1 < a0) continue;
                std::vector<int> row_as;
                if (a1 - a0 > 300) row_as = {a0, (a0 + a1) / 2, a1};
                else if (a1 - a0 > 120) row_as = {a0, a1};
                else row_as = {(a0 + a1) / 2};
                for (const int row_a : row_as)
                    for (const std::int64_t ca : across) add("resthelix", scan, a, scan, Helix, p, hint, ca, ca, nr, row_a);
            }
        }
        for (const std::int64_t p : whole(linspace(16, static_cast<double>(lay.length[scan][Helix] - N - 100), n_turn)))
            for (const std::int64_t ca : across) add("turn", scan, Helix, scan, Helix, p, N, ca, ca, 128, -1);
    }
    lo = std::max(0, -lay.move), hi = std::min(C - W, C - W - lay.move);  // windows both scans hold
    if (hi - lo < 32) throw Error("the two scans do not overlap by a window's width");
    for (int a = 0; a < 3; ++a)
        for (const std::int64_t p : whole(linspace(16, static_cast<double>(std::min(lay.length[0][a], lay.length[1][a]) - 64), n_pair)))
            for (const std::int64_t row_a : bands)
                for (const std::int64_t ca : whole(linspace(lo + 13, hi - 13, 4))) add("pair", 0, a, 1, a, p, 0, ca, ca + lay.move, nrows, row_a);
    return out;
}

std::vector<Row> measure(Session& session, const Layout& lay, const geometry::Map& m, const double lines[3][2], Device device, int patch) {
    const std::vector<Planned> all = plan(lay);
    // the comparisons within a scan on the map alone; one scan against the other per stretch, the second scan
    // with its line of that stretch added
    std::vector<std::pair<int, std::vector<Planned>>> groups;  // (-1 or the stretch, its comparisons), in the plan's order
    for (const Planned& p : all) {
        const int key = p.kind == "pair" ? p.a : -1;
        auto it = std::find_if(groups.begin(), groups.end(), [&](const auto& g) { return g.first == key; });
        if (it == groups.end()) groups.emplace_back(key, std::vector<Planned>()), it = groups.end() - 1;
        it->second.push_back(p);
    }
    std::vector<Row> out;
    for (auto& [key, items] : groups) {
        session.set_detector_map(session_map(m));
        if (key >= 0) session.set_detector_map(session_map(geometry::with_line(m, lines[key][0], lines[key][1], overlap_centre(lay) + lay.move)), 1);
        if (patch)  // the blocks in the order of their frames: neighbours read the same part of the files
            std::stable_sort(items.begin(), items.end(), [](const Planned& x, const Planned& y) {
                if (x.scan_a != y.scan_a) return x.scan_a < y.scan_a;
                const std::string nx = kNames[x.a], ny = kNames[y.a];
                if (nx != ny) return nx < ny;
                return x.request.frame_a < y.request.frame_a;
            });
        for (std::size_t i0 = 0; i0 < items.size(); i0 += 400) {
            const std::size_t i1 = std::min(items.size(), i0 + 400);
            std::vector<ViewRequest> reqs;
            for (std::size_t i = i0; i < i1; ++i) reqs.push_back(items[i].request);
            std::vector<std::pair<const Planned*, ViewMeasurement>> results;
            if (patch) {
                for (const PatchMeasurement& r : session.measure_patches(reqs, patch, patch, lay.row_sign, device))
                    if (r.i >= 0) results.emplace_back(&items[i0 + static_cast<std::size_t>(r.request)], r.m);
            } else {
                const std::vector<ViewMeasurement> got = session.measure(reqs, lay.row_sign, device);
                for (std::size_t i = 0; i < got.size(); ++i) results.emplace_back(&items[i0 + i], got[i]);
            }
            for (const auto& [p, r] : results) {
                if (!(r.planned && r.ok && r.correlation > 0.3)) continue;
                if (lay.is_blocked(p->scan_a, p->a, static_cast<double>(r.frame_a), r.row_a - 8, r.row_a + r.rows + 8)) continue;
                if (lay.is_blocked(p->scan_b, p->b, static_cast<double>(r.frame_b), r.row_a + r.row_offset - 8, r.row_a + r.row_offset + r.rows + 8)) continue;
                Row row;
                row.kind = p->kind, row.scan_a = p->scan_a, row.scan_b = p->scan_b, row.a = p->a, row.b = p->b;
                row.m = r;
                out.push_back(row);
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------- any number of scans: the layout, the plan, a pass

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
        add_lag(a, b, cc, 256);
        ++pairs;
    }
    if (!pairs) return {};
    const std::size_t k = argmax(cc);
    return {(cols - 1 - (static_cast<double>(k) - cols)) / 2.0, cc[k] / pairs};
}

// the widest window of a scan with its axis at `axis`: a power of two up to 512 inside the strip its opposite
// views share, with room for the cut's margins; 0: none
int window_width(double axis, int cols) {
    const double strip = 2.0 * std::min(axis, cols - 1 - axis);
    if (strip < 128 + 96) return 0;
    return std::min(512, static_cast<int>(std::pow(2.0, std::floor(std::log2(strip - 96)))));
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

}  // namespace

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
        for (std::size_t i = 0; i < rec.stretches.size(); ++i) {
            const Stretch& st = rec.stretches[i];
            if (st.role != Role::Projection || !st.latched()) continue;
            StretchLayout sl;
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
                    reqs.push_back(opposite_request(*helix, k, p, N, likely.axis, W, windows[windows.size() / 2], 256, -1, acq.fan));
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
    acq.width = 512;
    for (ScanLayout& sc : acq.scans) {
        if (!std::isnan(sc.axis)) acq.width = std::min(acq.width, window_width(sc.axis, cols));
        const StretchLayout* ref = &sc.stretches.front();
        for (StretchLayout& st : sc.stretches) {
            for (double& v : st.course.rows) v *= acq.row_sign;
            if (st.helix) ref = &st;
        }
        sc.mid_height = ref->course.rows[static_cast<std::size_t>((ref->first + ref->end) / 2)];
    }
    say(log, fmt("  %d x %d px, %d frames per turn; row sense %+.0f: %s; windows of %d columns", rows, cols, N, acq.row_sign, said.c_str(), acq.width));
    return acq;
}

std::string acquisition_json(const Acquisition& acq) {
    Json scans = Json::list();
    for (const ScanLayout& sc : acq.scans) {
        Json stretches = Json::list();
        for (const StretchLayout& st : sc.stretches) {
            Json d = Json::dict();
            d.set("name", Json::of(st.name)).set("index", count(st.index)).set("helix", Json::of_bool(st.helix)).set("frames", count(static_cast<long long>(st.course.rows.size())));
            d.set("first", count(st.first)).set("end", count(st.end)).set("blocked_frames", count(static_cast<long long>(st.blocked.frames.size())));
            stretches.add(d);
        }
        scans.add(Json::dict().set("name", Json::of(sc.name)).set("mid_height", num(sc.mid_height)).set("coarse_axis", num(sc.axis)).set("stretches", stretches));
    }
    Json out = Json::dict();
    out.set("rows", count(acq.rows)).set("cols", count(acq.cols)).set("frames_per_turn", count(acq.per_turn)).set("pixel_um", num(acq.pixel_um)).set("row_sign", num(acq.row_sign));
    out.set("row_time", num(acq.row_time)).set("fan", num(acq.fan)).set("window_columns", count(acq.width)).set("scans", scans);
    return out.text();
}

std::vector<Planned> plan(const Acquisition& acq, const PlanCounts& n) {
    const int R = acq.rows, C = acq.cols, N = acq.per_turn, W = acq.width;
    const int nrows = std::min(256, R / 4);
    if (n.angles_per_turn < 1 || n.max_positions < 2) throw Error("geometry calibration: a plan needs an angle per turn and two positions");
    std::vector<Planned> out;
    const auto add = [&](const char* kind, int scan, int a, int b, const ViewRequest& q, bool from_block = false) {
        Planned p;
        p.kind = kind, p.scan_a = p.scan_b = scan, p.a = a, p.b = b, p.from_block = from_block;
        p.request = q;
        p.request.label = kind;
        out.push_back(p);
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
        if (!std::isnan(sc.axis)) {
            const std::vector<std::int64_t> windows = mirrored_windows(sc.axis, C, W);
            if (windows.empty()) throw Error("the rotation axis of scan " + sc.name + " is too close to the detector's edge for opposite views");
            for (int a = 0; a < static_cast<int>(sc.stretches.size()); ++a) {
                const StretchLayout& st = sc.stretches[static_cast<std::size_t>(a)];
                const std::int64_t p_lo = st.first + 16, p_hi = st.end - N / 2 - 80;
                if (p_hi < p_lo) continue;
                // The angles a turn / angles_per_turn apart. Along a helix too long for every such angle of every
                // turn, the positions are spread evenly along it and each moved to the nearest frame at its angle:
                // the angles in turn, so that they stay evenly spread around the turn.
                const double base = static_cast<double>(N) / n.angles_per_turn, span = static_cast<double>(p_hi - p_lo);
                std::vector<std::int64_t> positions;
                if (std::floor(span / base) + 1 <= n.max_positions) {
                    for (double at = 0; at <= span; at += base) positions.push_back(p_lo + static_cast<std::int64_t>(std::llround(at)));
                } else {
                    const int count = std::max(n.angles_per_turn, n.max_positions - n.max_positions % n.angles_per_turn);
                    for (int j = 0; j < count; ++j) {
                        const double phase = (j % n.angles_per_turn) * base;
                        double at = N * std::round((j * span / (count - 1) - phase) / N) + phase;
                        if (at < 0) at += N;
                        if (at > span) at -= N;
                        if (at >= 0 && at <= span) positions.push_back(p_lo + static_cast<std::int64_t>(std::llround(at)));
                    }
                    std::sort(positions.begin(), positions.end());
                    positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
                }
                for (const std::int64_t p : positions) {
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

// ---------------------------------------------------------------- the fit

std::vector<double> least_squares(const std::vector<double>& X, std::size_t n, std::size_t k, const std::vector<double>& y, const std::vector<int>& columns,
                                  const std::vector<std::uint8_t>& rows) {
    std::vector<double> b(k, 0.0);
    std::vector<int> cols = columns;
    if (cols.empty() && columns.empty())
        for (std::size_t c = 0; c < k; ++c) cols.push_back(static_cast<int>(c));
    if (cols.empty() || !n) return b;
    LinearProblem p;
    p.unknowns = cols.size();
    std::int64_t eq = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!rows.empty() && !rows[i]) continue;
        for (std::size_t c = 0; c < cols.size(); ++c) {
            const double v = X[i * k + static_cast<std::size_t>(cols[c])];
            if (v == 0) continue;
            p.row.push_back(eq), p.col.push_back(static_cast<std::int64_t>(c)), p.value.push_back(v);
        }
        p.y.push_back(y[i]);
        ++eq;
    }
    if (p.y.empty()) throw Error("INSUFFICIENT_SUPPORT: no observations for regression");
    const LinearSolution s = solve_linear(p);
    if (!s.usable) throw Error(s.termination + ": " + s.message);
    for (std::size_t c = 0; c < cols.size(); ++c) b[static_cast<std::size_t>(cols[c])] = s.x[c];
    return b;
}

Fit fit(const std::vector<Row>& rows, const Layout& lay, const geometry::Map& m, double tau, double period, double huber, int max_iterations) {
    const std::size_t nt = m.i.size(), n = rows.size();
    // the stretches with pairs, in the order of their names; the frame positions
    std::vector<int> pair_st;
    for (const int a : {Helix, Rest1, Rest2})  // "helix" < "rest1" < "rest2"
        if (std::any_of(rows.begin(), rows.end(), [&](const Row& r) { return r.kind == "pair" && r.a == a; })) pair_st.push_back(a);
    const std::size_t K = pair_st.size();
    const double cref = overlap_centre(lay), vmid = m.v0;
    std::map<GroupKey, std::size_t> gi;
    for (const Row& r : rows) gi[GroupKey{r.kind, r.scan_a, r.a, r.m.frame_a}] = 0;
    {
        std::size_t k = 0;
        for (auto& g : gi) g.second = k++;
    }
    const std::size_t G = gi.size();
    // the unknowns: eu, ev (the map's terms), psi, rho, the lines a and b of all but the last stretch, u0, v0
    const std::size_t o_eu = 0, o_ev = nt, o_psi = 2 * nt, o_rho = o_psi + 1, o_a = o_rho + K, o_b = o_a + (K ? K - 1 : 0), o_u0 = o_b + (K ? K - 1 : 0), o_v0 = o_u0 + G;
    LinearProblem p;
    p.unknowns = o_v0 + G;
    p.y.assign(2 * n, 0.0);
    const auto put = [&](std::size_t eq, std::size_t unknown, double v) {
        if (v == 0) return;
        p.row.push_back(static_cast<std::int64_t>(eq)), p.col.push_back(static_cast<std::int64_t>(unknown)), p.value.push_back(v);
    };
    const auto phi = [&](double c, double v, std::size_t k) { return std::pow((c - m.c0) / m.cs, m.i[k]) * std::pow((v - m.v0) / m.vs, m.j[k]); };
    for (std::size_t i = 0; i < n; ++i) {
        const Row& r = rows[i];
        const ViewMeasurement& q = r.m;
        const bool mirror = r.kind == "opposite", pair = r.kind == "pair";
        const double cA = q.col_a + (q.cols - 1) / 2.0, cB = q.col_b + (q.cols - 1) / 2.0;
        const double vA = q.row_a + (q.rows - 1) / 2.0, vB = vA + q.row_offset;
        const double dq = q.dq - q.stated_dq;
        p.y[i] = mirror ? static_cast<double>(q.col_a + q.col_b + q.cols - 1) - q.du : q.du;
        p.y[n + i] = (q.D - q.stated_D) - q.slope_b * dq - tau / period * (q.slope_b * vB - q.slope_a * vA);
        for (std::size_t k = 0; k < nt; ++k) {
            const double fa = phi(cA, vA, k), fb = phi(cB, vB, k);
            if (!(m.i[k] == 1 && m.j[k] == 0)) put(i, o_eu + k, mirror ? -(fb + fa) : -(fb - fa));  // Eu's line in the column is a scale: not observable
            put(n + i, o_ev + k, -(fb - fa));
        }
        const std::size_t g = gi.at(GroupKey{r.kind, r.scan_a, r.a, q.frame_a});
        put(i, o_u0 + g, 1.0);
        put(n + i, o_v0 + g, 1.0);
        if (mirror) {
            put(i, o_psi, 2.0 * (vA - vmid));
            put(n + i, o_psi, 2.0 * (cA - lay.axis));
        }
        if (pair) {
            const std::size_t st = static_cast<std::size_t>(std::find(pair_st.begin(), pair_st.end(), r.a) - pair_st.begin());
            put(i, o_rho + st, -(vA - vmid));
            put(n + i, o_rho + st, cA - cref);
            for (std::size_t s = 0; s + 1 < K; ++s) {  // the last stretch's line is minus the sum of the others
                const double v = st == s ? cA - cref : (st == K - 1 ? -(cA - cref) : 0.0);
                put(i, o_a + s, v);
                put(n + i, o_b + s, v);
            }
        }
    }
    for (std::size_t k = 0; k < nt; ++k)  // the gauge, stated: Eu's line in the column stays zero
        if (m.i[k] == 1 && m.j[k] == 0) p.fixed.push_back(static_cast<std::int64_t>(o_eu + k)), p.fixed_value.push_back(0.0);
    SolveOptions opt;
    opt.huber = huber, opt.max_iterations = max_iterations;
    const LinearSolution sol = solve_linear(p, opt);
    if (!sol.usable) throw Error(sol.termination + ": " + sol.message);
    Fit f;
    f.n = n;
    f.solver = sol, f.rank = sol.rank, f.active = sol.active, f.unused = sol.unused;
    f.eu.assign(sol.x.begin() + static_cast<std::ptrdiff_t>(o_eu), sol.x.begin() + static_cast<std::ptrdiff_t>(o_eu + nt));
    f.ev.assign(sol.x.begin() + static_cast<std::ptrdiff_t>(o_ev), sol.x.begin() + static_cast<std::ptrdiff_t>(o_ev + nt));
    f.psi = sol.x[o_psi];
    for (std::size_t s = 0; s < K; ++s) f.rho[pair_st[s]] = sol.x[o_rho + s];
    for (int key = 0; key < 2; ++key) {
        const std::size_t o = key == 0 ? o_a : o_b;
        double sum = 0;
        for (std::size_t s = 0; s < K; ++s) {
            const double v = s + 1 < K ? sol.x[o + s] : -sum;
            sum += v;
            f.lines[pair_st[s]][static_cast<std::size_t>(key)] = v;
        }
    }
    for (const char* kind : kKinds) {
        double su = 0, sv = 0;
        std::size_t c = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (rows[i].kind == kind) su += sol.residual[i] * sol.residual[i], sv += sol.residual[n + i] * sol.residual[n + i], ++c;
        if (c) f.left[kind] = {c, std::sqrt(su / static_cast<double>(c)), std::sqrt(sv / static_cast<double>(c))};
    }
    for (const auto& [key, g] : gi) f.groups[key] = {sol.x[o_u0 + g], sol.x[o_v0 + g]};
    return f;
}

std::array<double, 2> update_range(const geometry::Map& m, const std::vector<double>& eu, const std::vector<double>& ev) {
    geometry::Map d = m;
    d.eu = eu, d.ev = ev;
    const std::vector<double> cs = linspace(0, 2 * m.c0 - 1, 40), vs = linspace(0, 2 * m.v0, 24);
    std::vector<double> c, v;
    for (const double y : vs)
        for (const double x : cs) c.push_back(x), v.push_back(y);
    std::vector<double> u(c.size()), w(c.size());
    geometry::map_displacement(d, c.data(), v.data(), c.size(), u.data(), w.data());
    const double mu = mean_of(u), mw = mean_of(w);
    double du = 0, dv = 0;
    for (std::size_t i = 0; i < u.size(); ++i) du = std::max(du, std::fabs(u[i] - mu)), dv = std::max(dv, std::fabs(w[i] - mw));
    return {du, dv};
}

// ---------------------------------------------------------------- the step

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
    if (scans.size() != 2) throw Error("the geometry calibration takes two scans: the one with the rotation axis inside the detector's field first");
    const Log& log = options.log;
    const auto t_start = std::chrono::steady_clock::now();
    Session session(scans, options.threads);
    say(log, "layout");
    Layout lay = read_layout(session, options.device, log);
    for (int k = 0; k < 2; ++k)
        for (int a = 0; a < 3; ++a) {
            const Record& rec = session.record(k);
            std::vector<double> h = stated_height(rec, rec.stretches[static_cast<std::size_t>(lay.stretch[k][a])]);
            for (double& v : h) v = lay.row_sign * v / (lay.pixel_um * 1e-3);
            lay.heights[k][a] = std::move(h);
        }
    const double period = median_of(session.record(0).stretches[static_cast<std::size_t>(lay.stretch[0][Helix])].timer_period);
    double row_time = options.row_time;
    Json timing;
    if (std::isnan(row_time)) {
        say(log, "rows' timing");
        timing = Json::parse(row_time_from_refill(session, lay, log));
        if (!timing.is(Json::Object))
            throw Error("no stretch holds a refill to measure the rows' timing from: give the row time (seconds per stored row) for this camera configuration");
        row_time = timing.find("seconds_per_row")->number;
    }
    geometry::Map m = geometry::empty_map(lay.rows, lay.cols, options.order);
    double lines[3][2] = {{0, 0}, {0, 0}, {0, 0}};
    Json history = Json::list();
    say(log, "map");
    std::vector<Row> rows;
    Fit f;
    double du = 0, dv = 0, dl = 0;
    bool ended = false;
    for (int it = 0; it < options.passes; ++it) {
        rows = measure(session, lay, m, lines, options.device, options.patch);
        f = fit(rows, lay, m, row_time, period);
        const std::array<double, 2> range = update_range(m, f.eu, f.ev);
        du = range[0], dv = range[1];
        dl = 0;
        for (const auto& [a, l] : f.lines) dl = std::max({dl, std::fabs(l[0]), std::fabs(l[1])});
        dl *= lay.cols / 2.0;
        Json left = Json::dict();
        std::string report;
        for (const char* kind : kKinds) {
            const auto found = f.left.find(kind);
            if (found == f.left.end()) continue;
            left.set(kind, Json::dict().set("n", count(static_cast<long long>(found->second.n))).set("px", num(found->second.px)).set("rows", num(found->second.rows)));
            report += fmt("%s%s %.3f px / %.3f rows", report.empty() ? "" : ", ", kind, found->second.px, found->second.rows);
        }
        history.add(Json::dict().set("accepted", count(static_cast<long long>(f.n))).set("correction_px", num(du)).set("correction_rows", num(dv)).set("line_change_px", num(dl)).set("left", left).set("solver", solver_json(f)));
        say(log, fmt("  pass %d: %zu comparisons; the map is corrected by up to %.3f px and %.3f rows, the second scan's lines by %.3f; left %s", it + 1, f.n, du, dv, dl,
                     report.c_str()));
        if (std::max({du, dv, dl}) < options.tolerance && it) {
            ended = true;
            break;
        }
        for (std::size_t k = 0; k < m.eu.size(); ++k) m.eu[k] += f.eu[k], m.ev[k] += f.ev[k];
        for (int a = 0; a < 3; ++a) {
            const auto found = f.lines.find(a);
            if (found == f.lines.end()) throw Error("geometry calibration: no comparison of the two scans was accepted in " + std::string(kNames[a]));
            lines[a][0] += found->second[0], lines[a][1] += found->second[1];
        }
    }
    if (!ended) say(log, fmt("  not converged to %g in %d passes", options.tolerance, options.passes));
    // the poses of the last pass
    const auto mean_or_null = [](const std::vector<double>& v) { return v.empty() ? Json() : num(mean_of(v)); };
    Json rho = Json::dict();
    for (const auto& [a, v] : f.rho) rho.set(kNames[a], num(v));  // (a map by stretch index: helix, rest1, rest2 by name below)
    {
        Json ordered = Json::dict();
        for (const int a : {Helix, Rest1, Rest2})
            if (const Json* v = rho.find(kNames[a])) ordered.set(kNames[a], *v);
        rho = ordered;
    }
    Json mirror_column = Json::dict();
    for (const int a : {Rest1, Rest2}) {
        std::vector<double> u;
        for (const auto& [key, uv] : f.groups)
            if (key.kind == "opposite" && key.a == a) u.push_back(uv[0] / 2);
        mirror_column.set(kNames[a], mean_or_null(u));
    }
    Json second = Json::dict();
    for (int a = 0; a < 3; ++a) {
        std::vector<double> u, v;
        for (const auto& [key, uv] : f.groups)
            if (key.kind == "pair" && key.a == a) u.push_back(uv[0]), v.push_back(uv[1]);
        if (u.empty()) throw Error("geometry calibration: no comparison of the two scans was accepted in " + std::string(kNames[a]));
        second.set(kNames[a], Json::dict().set("columns", num(lay.move + mean_of(u))).set("rows_beyond_record", num(mean_of(v))).set("line", numbers({lines[a][0], lines[a][1]})));
    }
    Json heights = Json::dict();
    const auto stated_D_of = [&](const char* kind, int scan, int a, std::int64_t frame) {
        for (const Row& r : rows)
            if (r.kind == kind && r.scan_a == scan && (a < 0 || r.a == a) && r.m.frame_a == frame) return r.m.stated_D;
        throw Error("geometry calibration: a frame position without its comparison");
    };
    for (int scan = 0; scan < 2; ++scan) {
        for (const int a : {Rest1, Rest2}) {
            std::vector<double> x, y;
            for (const auto& [key, uv] : f.groups)
                if (key.kind == "resthelix" && key.scan == scan && key.a == a) x.push_back(stated_D_of("resthelix", scan, a, key.frame)), y.push_back(uv[1]);
            if (x.size() >= 4) {
                const Line l = line_through(x, y);
                heights.set(fmt("scan %d %s", scan, kNames[a]), Json::dict().set("offset_rows", num(l.intercept)).set("scale_error", num(l.slope)));
            }
        }
        std::vector<double> ratio, sum;
        for (const auto& [key, uv] : f.groups)
            if (key.kind == "turn" && key.scan == scan) {
                const double d = stated_D_of("turn", scan, -1, key.frame);
                ratio.push_back(uv[1] / d), sum.push_back(uv[1] + d);
            }
        if (!ratio.empty()) heights.set(fmt("scan %d helix", scan), Json::dict().set("scale_error", num(mean_of(ratio))).set("rows_per_turn", num(mean_of(sum))));
    }
    Json poses = Json::dict();
    poses.set("axis_roll_rad", num(f.psi)).set("rotation_about_beam_rad", rho).set("mirror_column", mirror_column).set("second_scan", second).set("heights", heights);
    if (options.motion) {
        say(log, "motion within a turn");
        poses.set("motion", Json::parse(measure_motion(session, lay, m, lines, row_time, period, options.device, log)));
    }
    Json frames = Json::list(), blocked = Json::dict();
    for (int k = 0; k < 2; ++k) {
        Json l = Json::dict();
        for (int a = 0; a < 3; ++a) l.set(kNames[a], count(lay.length[k][a]));
        frames.add(l);
    }
    for (const auto& [key, b] : lay.blocked) {
        Json fr = Json::list(), first = Json::list(), last = Json::list();
        for (std::size_t i = 0; i < b.frames.size(); ++i) {
            int lo = -1, hi = -1;
            for (int r = 0; r < lay.rows; ++r)
                if (b.mask[i * static_cast<std::size_t>(lay.rows) + static_cast<std::size_t>(r)]) {
                    if (lo < 0) lo = r;
                    hi = r;
                }
            if (lo < 0) continue;
            fr.add(count(b.frames[i])), first.add(count(lo)), last.add(count(hi));
        }
        blocked.set(fmt("scan %d %s", key.first, kNames[key.second]), Json::dict().set("frames", fr).set("first_row", first).set("last_row", last));
    }
    Json layout = Json::dict();
    layout.set("rows", count(lay.rows)).set("cols", count(lay.cols)).set("frames", frames).set("frames_per_turn", count(lay.per_turn)).set("pixel_um", num(lay.pixel_um));
    layout.set("row_sign", num(lay.row_sign)).set("coarse_mirror_column", num(lay.axis)).set("coarse_move", count(lay.move)).set("window_columns", count(lay.width));
    layout.set("blocked_rows", blocked);
    Json map = Json::dict();
    Json mi = Json::list(), mj = Json::list();
    for (const int v : m.i) mi.add(count(v));
    for (const int v : m.j) mj.add(count(v));
    map.set("c0", num(m.c0)).set("cs", num(m.cs)).set("v0", num(m.v0)).set("vs", num(m.vs)).set("i", mi).set("j", mj).set("eu", numbers(m.eu)).set("ev", numbers(m.ev));
    Json names = Json::list();
    for (const std::string& s : scans) names.add(Json::of(s));
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    Json cal = Json::dict();
    cal.set("step", Json::of(std::string("geometry"))).set("scans", names).set("seconds", num(std::nearbyint(seconds * 10) / 10)).set("layout", layout);
    cal.set("row_time", Json::dict().set("seconds_per_row", num(row_time)).set("frame_period_s", num(period)).set("measured", timing));
    cal.set("map", map).set("poses", poses).set("passes", history).set("converged", Json::of_bool(std::max({du, dv, dl}) < options.tolerance));
    cal.set("patch", count(options.patch)).set("solver", solver_json(f));
    return cal.text();
}

}  // namespace helirec::calibration::geometry_calibration
