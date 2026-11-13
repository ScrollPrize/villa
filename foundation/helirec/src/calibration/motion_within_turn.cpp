// helirec/calibration/geometry_calibration.hpp: what the sample and the rotating body do within a turn, on the
// grid of the detector's map, and the second scan against the first at equal frames.
//
//   sideways    the body's place against the rotation axis in its own frame, d(t): two views at equal angle differ
//               by [d(t2) - d(t1)] . (cos, sin)(angle); opposite views see [d(t) - d(t + half a turn)] . n / 2 in
//               their mirror column. The first scan has both and gets a point every half turn, the second every
//               turn. Beside it the travel's direction against the columns and a constant per rest turn.
//   along axis  z(angle): the sample's height once per turn, from opposite views inside the helix (they show
//               -2 z), and what the rest turns' opposite views hold beyond it: a drift at rest.
//   angle       frames per turn of travel by which the helix is further on than the rows' timing gives.
//   heights     the rest turns against the helix (offset, scale) and the helix' own scale, on full-width blocks.
//   second scan the second scan against the first at equal frames, on the whole overlap, densely along every
//               stretch: per stretch a place (columns, rows, angle); once per turn, common to the stretches, the
//               rotating body's place against the axis in the second scan minus the first's and its counterpart
//               along the axis; in the helix a line in the travel; twice per turn, recorded and not explained.
#include <algorithm>
#include <cmath>
#include <set>

#include "calibration/util.hpp"
#include "helirec/calibration/geometry_calibration.hpp"
#include "io/json.hpp"

namespace helirec::calibration::geometry_calibration {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

Json num(double v) { return Json::of(v); }
Json count(long long v) { return Json::whole_number(v); }
Json pair_of(double a, double b) { return Json::list().add(num(a)).add(num(b)); }
// the root mean square, or null of nothing
Json rms_json(const std::vector<double>& v) { return v.empty() ? Json() : num(rms_of(v)); }

bool good(int n) {
    if (n < 2 || n % 2) return false;
    for (const int q : {2, 3, 5, 7})
        while (n % q == 0) n /= q;
    return n == 1;
}

DetectorMap session_map(const geometry::Map& m) {
    DetectorMap d;
    d.i = m.i, d.j = m.j, d.eu = m.eu, d.ev = m.ev;
    d.c0 = m.c0, d.cs = m.cs, d.v0 = m.v0, d.vs = m.vs;
    return d;
}

std::vector<double> matvec(const std::vector<double>& X, std::size_t n, std::size_t k, const std::vector<double>& b) {
    std::vector<double> out(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0;
        for (std::size_t c = 0; c < k; ++c) s += X[i * k + c] * b[c];
        out[i] = s;
    }
    return out;
}

}  // namespace

int block_size(int limit) {
    for (int n = limit; n > 15; --n) {
        if (!good(n)) continue;
        for (int m = 8; m <= 32; ++m)
            if (good(n + 2 * m)) return n;
    }
    throw Error("no block size");
}

std::vector<double> nodes_design(const std::vector<double>& t1, const std::vector<double>& t2, const std::vector<double>& th, int n_extra, double h, double total,
                                 int& K) {
    K = static_cast<int>(std::ceil(total / h - 1e-9));
    const std::size_t n = th.size(), k = static_cast<std::size_t>(2 * K + n_extra);
    std::vector<double> X(n * k, 0.0);
    for (int side = 0; side < 2; ++side) {
        const double sign = side == 0 ? 1.0 : -1.0;
        const std::vector<double>& t = side == 0 ? t2 : t1;
        for (std::size_t i = 0; i < n; ++i) {
            const double x = std::clamp(t[i] / h, 0.0, K - 1e-9);
            const int node = static_cast<int>(std::floor(x));
            const double f = x - node;
            for (int part = 0; part < 2; ++part) {
                const int at = node + part;
                const double w = part == 0 ? 1 - f : f;
                if (at <= 0) continue;  // d(0) = 0
                X[i * k + static_cast<std::size_t>(2 * (at - 1))] += sign * w * std::cos(th[i]);
                X[i * k + static_cast<std::size_t>(2 * (at - 1) + 1)] += sign * w * std::sin(th[i]);
            }
        }
    }
    return X;
}

std::string second_scan_fit(const std::vector<PairRow>& table, int rows, double total) {
    const std::size_t n = table.size();
    std::vector<int> names;
    for (int a = 0; a < 3; ++a)
        if (std::any_of(table.begin(), table.end(), [&](const PairRow& r) { return r.a == a; })) names.push_back(a);
    // turns of travel from the helix' middle
    double helix_mean = 0;
    std::size_t in_helix = 0;
    for (const PairRow& r : table)
        if (r.a == Helix) helix_mean += r.t, ++in_helix;
    if (in_helix) helix_mean /= static_cast<double>(in_helix);
    // the columns of the designs: per stretch an offset and a line in the row, the travel; then what turns
    const std::size_t n1 = 2 * names.size() + 1;
    const int K = static_cast<int>(std::ceil(total - 1e-9)) + 1;
    const auto design = [&](bool course) {  // base + (a course once per turn with a point every turn | once per turn) + twice per turn
        const std::size_t nh = course ? static_cast<std::size_t>(2 * K) : 2, k = n1 + nh + 2;
        std::vector<double> X(n * k, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            const PairRow& r = table[i];
            double* x = X.data() + i * k;
            for (std::size_t s = 0; s < names.size(); ++s)
                if (r.a == names[s]) x[s] = 1.0, x[names.size() + s] = r.v - (rows - 1) / 2.0;
            x[n1 - 1] = r.a == Helix ? r.t - helix_mean : 0.0;
            if (course) {
                for (int node = 0; node < K; ++node) {
                    const double w = std::clamp(1 - std::fabs(r.t - node), 0.0, 1.0);
                    x[n1 + static_cast<std::size_t>(2 * node)] = w * std::cos(r.th);
                    x[n1 + static_cast<std::size_t>(2 * node) + 1] = w * std::sin(r.th);
                }
            } else {
                x[n1] = std::cos(r.th), x[n1 + 1] = std::sin(r.th);
            }
            x[n1 + nh] = std::cos(2 * r.th), x[n1 + nh + 1] = std::sin(2 * r.th);
        }
        return X;
    };
    Json P = Json::dict();
    Json blocks = Json::dict();
    for (const int a : names) blocks.set(kNames[a], count(std::count_if(table.begin(), table.end(), [&](const PairRow& r) { return r.a == a; })));
    P.set("blocks", blocks);
    for (int what = 0; what < 3; ++what) {
        const char* key = what == 0 ? "columns" : what == 1 ? "rows" : "angle_frames";
        const bool course = what == 0;
        const std::size_t nh = course ? static_cast<std::size_t>(2 * K) : 2, k = n1 + nh + 2;
        const std::vector<double> X = design(course);
        std::vector<double> val(n);
        for (std::size_t i = 0; i < n; ++i) val[i] = what == 0 ? table[i].du : what == 1 ? table[i].y : table[i].q;
        const auto first = [](std::size_t count_) {
            std::vector<int> c(count_);
            for (std::size_t i = 0; i < count_; ++i) c[i] = static_cast<int>(i);
            return c;
        };
        const std::vector<double> b = least_squares(X, n, k, val), b1 = least_squares(X, n, k, val, first(n1 + nh)), b0 = least_squares(X, n, k, val, first(n1));
        const auto left = [&](const std::vector<double>& coeff) {
            std::vector<double> res = matvec(X, n, k, coeff);
            for (std::size_t i = 0; i < n; ++i) res[i] = val[i] - res[i];
            return res;
        };
        const std::vector<double> r0 = left(b0), r1 = left(b1), r2 = left(b);
        Json offset = Json::dict(), per_row = Json::dict(), by_stretch = Json::dict();
        for (std::size_t s = 0; s < names.size(); ++s) {
            offset.set(kNames[names[s]], num(b1[s]));
            per_row.set(kNames[names[s]], num(b1[names.size() + s]));
            std::vector<double> part;
            for (std::size_t i = 0; i < n; ++i)
                if (table[i].a == names[s]) part.push_back(r1[i]);
            by_stretch.set(kNames[names[s]], num(rms_of(part)));
        }
        Json d = Json::dict();
        d.set("offset", offset).set("per_row", per_row).set("per_turn_of_travel", num(b1[n1 - 1])).set("twice_per_turn", pair_of(b[n1 + nh], b[n1 + nh + 1]));
        d.set("left", Json::dict().set("offsets_only", num(rms_of(r0))).set("with_once_per_turn", num(rms_of(r1))).set("with_twice_per_turn", num(rms_of(r2))));
        d.set("left_by_stretch", by_stretch);
        if (course) {
            Json nodes = Json::list();  // at 0, 1, 2, .. turns from the start
            for (int node = 0; node < K; ++node) nodes.add(pair_of(b1[n1 + static_cast<std::size_t>(2 * node)], b1[n1 + static_cast<std::size_t>(2 * node) + 1]));
            d.set("once_per_turn_nodes", nodes);
        } else {
            d.set("once_per_turn", pair_of(b1[n1], b1[n1 + 1]));
        }
        P.set(key, d);
    }
    return P.text();
}

std::string measure_motion(Session& session, const Layout& lay, const geometry::Map& m, const double lines[3][2], double tau, double period, Device device,
                           const Log& log) {
    const int R = lay.rows, C = lay.cols, N = lay.per_turn;
    const int wide = block_size(C - 28);
    double step = 0;
    {
        const std::vector<double> angle = stated_angle(session.record(0).stretches[static_cast<std::size_t>(lay.stretch[0][Helix])]);
        std::vector<double> d(angle.size() - 1);
        for (std::size_t i = 0; i + 1 < angle.size(); ++i) d[i] = angle[i + 1] - angle[i];
        step = median_of(d);
    }
    const double fan = lay.pixel_um * 1e-3 / session.record(0).technique.number("scan/source_sample_distance") / (std::fabs(step) * (kPi / 180.0));
    double start[2][3], total = 0;
    for (int k = 0; k < 2; ++k) {
        start[k][Rest1] = 0.0;
        start[k][Helix] = static_cast<double>(lay.length[k][Rest1]) / N;
        start[k][Rest2] = static_cast<double>(lay.length[k][Rest1] + lay.length[k][Helix]) / N;
        total = std::max(total, start[k][Rest2] + static_cast<double>(lay.length[k][Rest2]) / N);
    }
    std::vector<Planned> plan_;
    const auto add = [&](const char* kind, int scan, int a, int b, std::int64_t frame, std::int64_t hint, int cols, int col_a, int col_b, int rows_, int row_a,
                         bool mirror = false) {
        Planned p;
        p.kind = kind, p.scan_a = p.scan_b = scan, p.a = a, p.b = b;
        ViewRequest& q = p.request;
        q.label = kind;
        q.scan_a = q.scan_b = scan, q.stretch_a = lay.stretch[scan][a], q.stretch_b = lay.stretch[scan][b];
        q.frame_a = frame, q.frame_hint = hint;
        q.angle_offset_deg = mirror ? 180.0 : 0.0, q.mirror = mirror;
        q.frames = 32, q.rows = rows_, q.cols = cols, q.row_a = row_a, q.col_a = col_a, q.col_b = col_b;
        q.frame_shear = mirror ? 2 * fan : 0.0;
        plan_.push_back(p);
    };
    for (int scan = 0; scan < 2; ++scan) {
        const std::vector<double>& hrows = lay.heights[scan][Helix];
        for (const int a : {Rest1, Rest2}) {
            const std::int64_t hint = a == Rest1 ? 0 : lay.length[scan][Helix] - lay.length[scan][a];
            for (std::int64_t p = 16; p < lay.length[scan][a] - 48; p += std::max(1, N / 125)) {
                const double Nr = std::fabs(hrows.at(static_cast<std::size_t>(p + 16 + hint)) - lay.heights[scan][a].at(static_cast<std::size_t>(p + 16)));
                add("resthelix", scan, a, Helix, p, hint, wide, 14, 14, Nr <= R - 340 ? 256 : 128, -1);
            }
        }
        for (std::int64_t p = 16; p < lay.length[scan][Helix] - N - 48; p += std::max(1, N / 62)) add("turn", scan, Helix, Helix, p, N, wide, 14, 14, 128, -1);
    }
    const int axis2 = static_cast<int>(std::nearbyint(2 * lay.axis));
    const int w_opp = block_size(std::min(C - 1, axis2) - std::max(0, axis2 - (C - 1)) - 64);  // the columns whose mirror image lies on the detector
    const int ca = (axis2 - (w_opp - 1)) / 2;
    const int nrows = std::min(256, R / 4);
    for (const int a : {Rest1, Rest2})
        for (std::int64_t p = 16; p < N / 2 - 80; p += std::max(1, N / 62))
            for (const double row_a : linspace(24, R - nrows - 24, 5)) add("opposite", 0, a, a, p, N / 2, w_opp, ca, axis2 - (w_opp - 1) - ca, nrows, static_cast<int>(row_a), true);
    {
        const std::vector<double>& hrows = lay.heights[0][Helix];
        const int Dh = static_cast<int>(std::nearbyint(hrows.at(static_cast<std::size_t>(N / 2 + 2000)) - hrows.at(2000)));
        const int lo = std::max(0, 40 - Dh), hi = std::min(R - nrows, R - nrows - 40 - Dh);
        if (hi >= lo)
            for (std::int64_t p = 16; p < lay.length[0][Helix] - N / 2 - 64; p += std::max(1, N / 31))
                for (const int row_a : std::set<int>{lo, hi}) add("helix opposite", 0, Helix, Helix, p, N / 2, w_opp, ca, axis2 - (w_opp - 1) - ca, nrows, row_a, true);
    }
    // the second scan against the first at equal frames: the whole overlap, three bands of rows
    const int w_pair = block_size(C - std::abs(lay.move) - 28);
    const int pc = std::max(0, -lay.move) + 14;
    std::vector<Planned> pairs[3];
    for (int a = 0; a < 3; ++a)
        for (std::int64_t p = 16; p < std::min(lay.length[0][a], lay.length[1][a]) - 64; p += std::max(1, a != Helix ? N / 62 : N / 31))
            for (const double row_a : linspace(24, R - nrows - 24, 3)) {
                Planned q;
                q.kind = "pair", q.scan_a = 0, q.scan_b = 1, q.a = q.b = a;
                ViewRequest& v = q.request;
                v.label = "pair";
                v.scan_a = 0, v.scan_b = 1, v.stretch_a = lay.stretch[0][a], v.stretch_b = lay.stretch[1][a];
                v.frame_a = p, v.frame_hint = 0;
                v.frames = 32, v.rows = nrows, v.cols = w_pair, v.row_a = static_cast<int>(row_a), v.col_a = pc, v.col_b = pc + lay.move;
                pairs[a].push_back(q);
            }
    std::vector<Row> rows;
    const auto run = [&](const std::vector<Planned>& todo) {
        for (std::size_t i0 = 0; i0 < todo.size(); i0 += 250) {
            const std::size_t i1 = std::min(todo.size(), i0 + 250);
            std::vector<ViewRequest> reqs;
            for (std::size_t i = i0; i < i1; ++i) reqs.push_back(todo[i].request);
            // each block read once: where the registration moves a block by whole frames, rows or columns it is
            // cut again from that reading (the block as its one patch)
            std::vector<ViewMeasurement> got(reqs.size());
            for (const PatchMeasurement& r : session.measure_patches(reqs, 1 << 20, 1 << 20, lay.row_sign, device)) got[static_cast<std::size_t>(r.request)] = r.m;
            for (std::size_t i = 0; i < got.size(); ++i) {
                const Planned& p = todo[i0 + i];
                const ViewMeasurement& r = got[i];
                if (!(r.planned && r.ok && r.correlation > 0.3)) continue;
                if (lay.is_blocked(p.scan_a, p.a, static_cast<double>(r.frame_a), r.row_a - 8, r.row_a + r.rows + 8) ||
                    lay.is_blocked(p.scan_b, p.b, static_cast<double>(r.frame_b), r.row_a + r.row_offset - 8, r.row_a + r.row_offset + r.rows + 8))
                    continue;
                Row d;
                d.kind = p.kind, d.scan_a = p.scan_a, d.scan_b = p.scan_b, d.a = p.a, d.b = p.b;
                d.m = r;
                const double vA = r.row_a + (r.rows - 1) / 2.0;
                d.q = r.dq - r.stated_dq;
                d.y = (r.D - r.stated_D) - r.slope_b * d.q - tau / period * (r.slope_b * (vA + r.row_offset) - r.slope_a * vA);
                d.t1 = start[p.scan_a][p.a] + static_cast<double>(r.frame_a + 16) / N;
                d.t2 = start[p.scan_b][p.b] + static_cast<double>(r.frame_b + 16) / N;
                d.th = r.angle_a_deg * (kPi / 180.0);
                d.vA = vA;
                d.X = 0.5 * (r.col_a + r.col_b + r.cols - 1 - r.du);
                // the encoder's rows from the helix' first frame
                d.Pa = lay.heights[p.scan_a][p.a].at(static_cast<std::size_t>(r.frame_a + 16)) - lay.heights[p.scan_a][Helix][0];
                d.Pb = lay.heights[p.scan_b][p.b].at(static_cast<std::size_t>(r.frame_b + 16)) - lay.heights[p.scan_b][Helix][0];
                rows.push_back(d);
            }
        }
    };
    session.set_detector_map(session_map(m));
    run(plan_);
    std::size_t n_planned = plan_.size();
    for (int a = 0; a < 3; ++a) {
        if (pairs[a].empty()) continue;
        session.set_detector_map(session_map(geometry::with_line(m, lines[a][0], lines[a][1], overlap_centre(lay) + lay.move)), 1);
        run(pairs[a]);
        n_planned += pairs[a].size();
    }
    session.set_detector_map(session_map(m));
    if (log) log(fmt("  %zu of %zu comparisons", rows.size(), n_planned));

    Json out = Json::dict();
    {
        Json starts = Json::list();
        for (int k = 0; k < 2; ++k) {
            Json s = Json::dict();
            for (int a = 0; a < 3; ++a) s.set(kNames[a], num(start[k][a]));
            starts.add(s);
        }
        out.set("turns_at_start", starts).set("comparisons", count(static_cast<long long>(rows.size())));
    }
    for (int scan = 0; scan < 2; ++scan) {
        Json S = Json::dict();
        for (const int a : {Rest1, Rest2}) {
            std::vector<double> D, y, qa, fb;
            for (const Row& r : rows)
                if (r.kind == "resthelix" && r.scan_a == scan && r.a == a)
                    D.push_back(r.m.stated_D), y.push_back(r.y), qa.push_back(r.q + tau / period * r.m.row_offset), fb.push_back(static_cast<double>(r.m.frame_b));
            if (D.size() < 8) continue;
            const Line ly = fit_line(D, y, 0, D.size()), lq = fit_line(D, qa, 0, D.size());  // the helix' angle beyond the rows' timing [frames]
            std::vector<double> ry(D.size()), rq(D.size());
            Json table = Json::list();
            for (std::size_t i = 0; i < D.size(); ++i) {
                ry[i] = y[i] - (ly.slope * D[i] + ly.intercept), rq[i] = qa[i] - (lq.slope * D[i] + lq.intercept);
                table.add(Json::list().add(num(fb[i] + 16.0)).add(num(D[i])).add(num(y[i])).add(num(qa[i])));  // helix frame, encoder's rows from the rest turn, image minus encoder, angle
            }
            S.set(kNames[a], Json::dict().set("offset_rows", num(ly.intercept)).set("scale_error", num(ly.slope)).set("left_rows", num(sd_of(ry))).set("blocks", count(static_cast<long long>(D.size())))
                                 .set("angle_frames", num(lq.intercept)).set("angle_frames_per_row", num(lq.slope)).set("angle_left_frames", num(sd_of(rq))).set("table", table));
        }
        {
            std::vector<double> ratio, sum, angle;
            Json table = Json::list();
            for (const Row& r : rows)
                if (r.kind == "turn" && r.scan_a == scan) {
                    const double a_ = r.q + tau / period * r.m.row_offset;
                    ratio.push_back(r.y / r.m.stated_D), sum.push_back(r.m.stated_D + r.y), angle.push_back(a_);
                    table.add(Json::list().add(num(static_cast<double>(r.m.frame_a) + 16.0)).add(num(r.m.stated_D)).add(num(r.y)).add(num(a_)));  // first frame, encoder's rows to a turn on, image minus encoder, angle [frames]
                }
            const double nan = std::nan("");
            S.set("helix", Json::dict().set("scale_error", num(ratio.empty() ? nan : mean_of(ratio))).set("rows_per_turn", num(sum.empty() ? nan : mean_of(sum)))
                               .set("angle_frames_per_turn", num(angle.empty() ? nan : mean_of(angle))).set("blocks", count(static_cast<long long>(ratio.size()))).set("table", table));
        }
        // sideways: the body's course, the travel's direction and the rest turns' constants; the first scan with its
        // opposite views (their mirror column) beside the views at equal angle
        std::vector<const Row*> same, opp;
        for (const Row& r : rows) {
            if (r.scan_a == scan && (r.kind == "resthelix" || r.kind == "turn")) same.push_back(&r);
            if (scan == 0 && (r.kind == "opposite" || r.kind == "helix opposite")) opp.push_back(&r);
        }
        const double h = scan == 0 ? 0.5 : 1.0;
        const auto column = [](const std::vector<const Row*>& v, double Row::*field) {
            std::vector<double> out_(v.size());
            for (std::size_t i = 0; i < v.size(); ++i) out_[i] = v[i]->*field;
            return out_;
        };
        int K = 0;
        std::vector<double> X = nodes_design(column(same, &Row::t1), column(same, &Row::t2), column(same, &Row::th), 7, h, total, K);
        const std::size_t k = static_cast<std::size_t>(2 * K + 7), K2 = static_cast<std::size_t>(2 * K);
        std::vector<double> y(same.size());
        std::vector<int> kind(same.size(), 0);  // 0: same angle, 1: opposite
        for (std::size_t i = 0; i < same.size(); ++i) {
            const Row& r = *same[i];
            X[i * k + K2] = r.m.stated_D;
            X[i * k + K2 + 1] = r.kind == "resthelix" && r.a == Rest1 ? 1.0 : 0.0;
            X[i * k + K2 + 2] = r.kind == "resthelix" && r.a == Rest2 ? 1.0 : 0.0;
            y[i] = r.m.du;
        }
        if (scan == 0 && !opp.empty()) {
            int K_ = 0;
            std::vector<double> Xo = nodes_design(column(opp, &Row::t1), column(opp, &Row::t2), column(opp, &Row::th), 7, h, total, K_);
            for (std::size_t i = 0; i < opp.size(); ++i) {
                const Row& r = *opp[i];
                for (std::size_t c = 0; c < K2; ++c) Xo[i * k + c] *= -0.5;
                Xo[i * k + K2 + 3 + static_cast<std::size_t>(r.a)] = 1.0;
                // The axis' column at row v of a frame at travel P: c0 + roll (v - middle) + (travel's direction - roll) P.
                // Opposite views inside the helix lie at two travels: their mirror column holds the direction at their
                // mean and the roll at the first one's (the row is the first one's).
                Xo[i * k + K2] = r.a == Helix ? 0.5 * (r.Pa + r.Pb) : 0.0;
                Xo[i * k + K2 + 6] = r.vA - (R - 1) / 2.0 - (r.a == Helix ? r.Pa : 0.0);
                y.push_back(r.X - lay.axis);
                kind.push_back(1);
            }
            X.insert(X.end(), Xo.begin(), Xo.end());
        }
        const std::size_t n = y.size();
        std::vector<int> use, use_tail;
        for (std::size_t c = 0; c < k; ++c) {
            bool any = false;
            for (std::size_t i = 0; i < n && !any; ++i) any = X[i * k + c] != 0;
            if (!any) continue;
            use.push_back(static_cast<int>(c));
            if (c >= K2) use_tail.push_back(static_cast<int>(c));
        }
        std::vector<std::uint8_t> half(n);
        for (std::size_t i = 0; i < n; ++i) half[i] = i % 2 == 0;
        const std::vector<double> zero(k, 0.0);
        const auto solve = [&](const std::vector<int>& cols, const std::vector<std::uint8_t>& sel) { return cols.empty() ? zero : least_squares(X, n, k, y, cols, sel); };
        const std::vector<double> b = solve(use, {}), b0 = solve(use_tail, {}), bh = solve(use, half), bh0 = solve(use_tail, half);
        const auto left = [&](const std::vector<double>& coeff, int of_kind, bool held) {
            const std::vector<double> fitted = matvec(X, n, k, coeff);
            std::vector<double> res;
            for (std::size_t i = 0; i < n; ++i)
                if (kind[i] == of_kind && (!held || !half[i])) res.push_back(y[i] - fitted[i]);
            return res;
        };
        Json left_px = Json::dict(), left_out = Json::dict();
        for (const int of_kind : {1, 0}) {  // "opposite" before "same angle"
            if (std::find(kind.begin(), kind.end(), of_kind) == kind.end()) continue;
            const char* name = of_kind ? "opposite" : "same angle";
            left_px.set(name, Json::list().add(rms_json(left(b0, of_kind, false))).add(rms_json(left(b, of_kind, false))));
            left_out.set(name, Json::list().add(rms_json(left(bh0, of_kind, true))).add(rms_json(left(bh, of_kind, true))));
        }
        Json nodes = Json::list();
        for (int node = 0; node < K; ++node) nodes.add(pair_of(b[static_cast<std::size_t>(2 * node)], b[static_cast<std::size_t>(2 * node) + 1]));
        Json side = Json::dict();
        side.set("node_spacing_turns", num(h)).set("nodes_px", nodes).set("travel_px_per_row", num(b[K2])).set("rest1_px", num(b[K2 + 1])).set("rest2_px", num(b[K2 + 2]));
        side.set("left_px", left_px).set("left_px_at_positions_left_out", left_out);
        if (scan == 0) {
            // rest1, rest2: the axis' column at the middle row in those turns; helix: the same at the helix' first frame
            const double P_end = lay.heights[0][Rest2][0] - lay.heights[0][Helix][0];
            const double mc_rest1 = lay.axis + b[K2 + 3], mc_helix = lay.axis + b[K2 + 4], mc_rest2 = lay.axis + b[K2 + 5], per_row = b[K2 + 6];
            Json mc = Json::dict();
            mc.set("rest1", num(mc_rest1)).set("helix", num(mc_helix)).set("rest2", num(mc_rest2)).set("per_row", num(per_row));
            // same-angle views against opposite views
            mc.set("closure_px", Json::dict().set("rest1", num(b[K2 + 1] - (mc_helix - mc_rest1))).set("rest2", num(b[K2 + 2] - (mc_helix - mc_rest2 + (b[K2] - per_row) * P_end))));
            side.set("mirror_column", mc);
        }
        S.set("sideways", side);
        out.set(fmt("scan %d", scan), S);
    }
    {
        std::vector<const Row*> ho;
        for (const Row& r : rows)
            if (r.kind == "helix opposite") ho.push_back(&r);
        if (ho.size() >= 16) {
            const std::size_t n = ho.size();
            double t_mean = 0;
            for (const Row* r : ho) t_mean += r->t1;
            t_mean /= static_cast<double>(n);
            std::vector<double> H(n * 5), y(n);
            for (std::size_t i = 0; i < n; ++i) {
                const Row& r = *ho[i];
                H[i * 5] = 1.0, H[i * 5 + 1] = r.t1 - t_mean, H[i * 5 + 2] = r.vA - (R - 1) / 2.0, H[i * 5 + 3] = std::cos(r.th), H[i * 5 + 4] = std::sin(r.th);
                y[i] = r.y;
            }
            const std::vector<double> bz = least_squares(H, n, 5, y);
            Json A = Json::dict(), drift = Json::dict();
            A.set("z_cos", num(-bz[3] / 2)).set("z_sin", num(-bz[4] / 2)).set("amplitude_rows", num(std::hypot(bz[3], bz[4]) / 2)).set("blocks", count(static_cast<long long>(n)));
            std::vector<std::pair<std::string, Json>> sds;
            for (const int a : {Rest1, Rest2}) {
                std::vector<double> dv, left;
                for (const Row& r : rows)
                    if (r.kind == "opposite" && r.a == a) dv.push_back(r.m.dv), left.push_back(r.m.dv - (bz[3] * std::cos(r.th) + bz[4] * std::sin(r.th)));
                if (dv.size() < 8) continue;
                drift.set(kNames[a], num(2 * mean_of(left)));
                sds.emplace_back(std::string(kNames[a]) + "_opposite_height_sd", pair_of(sd_of(dv), sd_of(left)));
            }
            A.set("drift_rows_per_turn", drift);
            for (auto& [key, v] : sds) A.set(key, v);
            out.set("along_axis", A);
        }
    }
    {
        std::vector<PairRow> table;
        Json rows_json = Json::list();
        for (const Row& r : rows)
            if (r.kind == "pair") {
                table.push_back(PairRow{r.a, r.t1, r.th, r.vA, r.m.du, r.y, r.q});
                // stretch, turns from the start, angle [rad], middle row, columns, rows, angle [frames]
                rows_json.add(Json::list().add(Json::of(std::string(kNames[r.a]))).add(num(r.t1)).add(num(r.th)).add(num(r.vA)).add(num(r.m.du)).add(num(r.y)).add(num(r.q)));
            }
        if (table.size() >= 60) {
            Json second = Json::parse(second_scan_fit(table, R, total));
            second.set("window_columns", Json::list().add(count(pc)).add(count(pc + w_pair - 1))).set("table", rows_json);
            out.set("second_scan", second);
        }
    }
    return out.text();
}

}  // namespace helirec::calibration::geometry_calibration
