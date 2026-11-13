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

// cc[k + n] += sum_c a[c] b[c + k] for k in (-n, n)
void add_lag(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& cc) {
    const long n = static_cast<long>(a.size());
#pragma omp parallel for schedule(static)
    for (long k = -n + 1; k < n; ++k) {
        double s = 0;
        const long lo = std::max(0L, -k), hi = std::min(n, n - k);
        for (long c = lo; c < hi; ++c) s += a[static_cast<std::size_t>(c)] * b[static_cast<std::size_t>(c + k)];
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

bool Layout::is_blocked(int scan, int name, double frame, int row_lo, int row_hi) const {
    const auto it = blocked.find({scan, name});
    if (it == blocked.end()) return false;
    const Blocked& b = it->second;
    std::size_t k = 0;
    for (std::size_t i = 1; i < b.frames.size(); ++i)
        if (std::fabs(static_cast<double>(b.frames[i]) - frame) < std::fabs(static_cast<double>(b.frames[k]) - frame)) k = i;
    const int lo = std::max(0, row_lo), hi = std::min(rows, std::max(0, row_hi));
    for (int r = lo; r < hi; ++r)
        if (b.mask[k * static_cast<std::size_t>(rows) + static_cast<std::size_t>(r)]) return true;
    return false;
}

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
    // rows that hold something across the whole width: the mean of the lowest columns of a row is not air there
    const int few = std::max(8, cols / 50);
    for (int k = 0; k < 2; ++k)
        for (int a = 0; a < 3; ++a) {
            const std::int64_t n = lay.length[k][a];
            const int samples = static_cast<int>(std::max<std::int64_t>(8, n / std::max(1, lay.per_turn / 16)));
            std::vector<std::int64_t> frames = whole(linspace(0.0, static_cast<double>(n - 1), samples));
            frames.erase(std::unique(frames.begin(), frames.end()), frames.end());
            Blocked b;
            b.frames = frames;
            b.mask.assign(frames.size() * static_cast<std::size_t>(rows), 0);
            bool any = false;
            std::vector<std::pair<int, int>> spans;
            for (std::size_t i = 0; i < frames.size(); ++i) {
                const std::vector<float> p = integrals(session, k, lay.stretch[k][a], frames[i], 0, rows);
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
                if (lo >= 0) any = true, spans.emplace_back(lo, hi);
            }
            if (!any) continue;
            std::set<std::pair<int, int>> distinct(spans.begin(), spans.end());
            std::string where;
            if (distinct.size() == 1) where = fmt("rows %d..%d", spans[0].first, spans[0].second);
            else {
                int lo = spans[0].first, hi = spans[0].second;
                for (const auto& s : spans) lo = std::min(lo, s.first), hi = std::max(hi, s.second);
                where = fmt("rows between %d and %d, changing with the frame,", lo, hi);
            }
            say(log, fmt("  scan %d %s: %s hold something across the whole width in %zu of %zu sampled frames; kept out", k, kNames[a], where.c_str(), spans.size(), frames.size()));
            lay.blocked[{k, a}] = std::move(b);
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
    double best_score = -1, best_sign = -1;
    for (const double sign : {-1.0, 1.0}) {
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
        std::vector<double> got;
        for (const ViewMeasurement& r : session.measure(reqs, sign, device))
            if (r.planned && r.ok) got.push_back(r.correlation);
        const double score = got.empty() ? 0.0 : mean_of(got);
        if (score > best_score) best_score = score, best_sign = sign;
    }
    if (best_score < 0.3)
        throw Error("the helix a turn apart does not register in either sense of the rows: the travel per turn must be less than the detector's height");
    lay.row_sign = best_sign;
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
