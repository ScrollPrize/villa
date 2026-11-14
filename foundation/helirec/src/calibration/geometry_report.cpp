// helirec/calibration/geometry_calibration.hpp: what a solution of the joint geometry model says beyond its
// parameters (report), and which scans' reconstructions overlap (neighbours).
#include <limits>
#include <map>
#include <tuple>

#include "calibration/geometry_model.hpp"
#include "calibration/util.hpp"
#include "helirec/calibration/geometry_calibration.hpp"
#include "io/json.hpp"

namespace helirec::calibration::geometry_calibration {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// y = sum_k x[k] columns[k]: least squares, then once more under a Huber loss at twice the spread the first
// leaves, so that a patch registered on something else does not carry the fit. left (optional): y less the fit.
// Nothing (an empty list) where the columns do not determine it.
std::vector<double> regress(const std::vector<std::vector<double>>& columns, const std::vector<double>& y, std::vector<double>* left = nullptr) {
    const std::size_t n = y.size(), m = columns.size();
    if (n < 4 * m) return {};
    LinearProblem p;
    p.unknowns = m;
    p.y = y;
    for (std::size_t k = 0; k < m; ++k)
        for (std::size_t i = 0; i < n; ++i) p.row.push_back(static_cast<std::int64_t>(i)), p.col.push_back(static_cast<std::int64_t>(k)), p.value.push_back(columns[k][i]);
    try {
        LinearSolution s = solve_linear(p);
        if (s.usable && s.spread > 0) {
            SolveOptions robust;
            robust.huber = 2 * s.spread;
            const LinearSolution again = solve_linear(p, robust);
            if (again.usable) s = again;
        }
        if (!s.usable || s.unused) return {};
        if (left) *left = s.residual;
        return s.x;
    } catch (const Error&) {
        return {};  // dependent columns: the angles do not go around the turn
    }
}

double wrapped_deg(double a) { return a - 360.0 * std::round(a / 360.0); }

Json num(double v) { return std::isfinite(v) ? Json::of(v) : Json(); }
Json pair_of(const double* v) { return Json::list().add(num(v[0])).add(num(v[1])); }

}  // namespace

ScanReport::ScanReport() {
    for (int j = 0; j < 3; ++j) mean[j] = left[j] = once[j][0] = once[j][1] = twice[j][0] = twice[j][1] = kNaN;
    row_time = rows_apart = departure = departure_between_views = height_slope = height_slope_error = from_anchor = pitch = pitch_phase_deg = kNaN;
}

PairReport::PairReport() {
    for (int j = 0; j < 3; ++j) mean[j] = left[j] = once[j][0] = once[j][1] = rms[j] = median[j] = kNaN;
}

Neighbours neighbours(const Acquisition& acq) {
    const int S = static_cast<int>(acq.scans.size());
    // the disc a scan alone reconstructs: out to the detector's edge farther from its axis
    const auto radius = [&](const ScanLayout& sc) { return std::isnan(sc.axis) ? kNaN : std::max(sc.axis, acq.cols - 1 - sc.axis); };
    const auto apart = [&](const ScanLayout& a, const ScanLayout& b) {
        const double d = std::hypot(a.on_stage[0] - b.on_stage[0], a.on_stage[1] - b.on_stage[1]);
        return std::isnan(d) ? 0.0 : d;
    };
    Neighbours out;
    std::vector<std::uint8_t> edge(static_cast<std::size_t>(S) * static_cast<std::size_t>(S), 0);
    for (int j = 0; j < S; ++j)
        for (int k = j + 1; k < S; ++k) {
            const ScanLayout& a = acq.scans[static_cast<std::size_t>(j)];
            const ScanLayout& b = acq.scans[static_cast<std::size_t>(k)];
            if (!(apart(a, b) < radius(a) + radius(b))) continue;  // (not, too, where a radius is not known)
            out.edges.push_back({j, k});
            edge[static_cast<std::size_t>(j) * static_cast<std::size_t>(S) + static_cast<std::size_t>(k)] = 1;
        }
    const auto is_edge = [&](int j, int k) { return edge[static_cast<std::size_t>(j) * static_cast<std::size_t>(S) + static_cast<std::size_t>(k)] != 0; };
    for (int i = 0; i < S; ++i)
        for (int j = i + 1; j < S; ++j)
            for (int k = j + 1; k < S && is_edge(i, j); ++k)
                if (is_edge(i, k) && is_edge(j, k)) out.triangles.push_back({i, j, k});
    return out;
}

GeometryReport report(const Acquisition& acq, const GeometrySolution& solution, const std::vector<Comparison>& comparisons) {
    if (solution.residual.size() != comparisons.size()) throw Error("geometry: a report needs the solution's residual of every comparison");
    if (solution.parameters.scans.size() != acq.scans.size()) throw Error("geometry: the solution and the acquisition differ in their scans");
    GeometryReport out;
    out.neighbours = neighbours(acq);
    const int S = static_cast<int>(acq.scans.size());
    out.scans.resize(static_cast<std::size_t>(S));
    std::vector<double> axes, slopes, scales;
    for (int k = 0; k < S; ++k) {
        const ScanLayout& sc = acq.scans[static_cast<std::size_t>(k)];
        const ScanParameters& sp = solution.parameters.scans[static_cast<std::size_t>(k)];
        ScanReport& r = out.scans[static_cast<std::size_t>(k)];
        // the scan's opposite views: of its helix, or of the stretch that has the most
        std::vector<std::size_t> per_stretch(sc.stretches.size(), 0);
        for (const Comparison& c : comparisons)
            if (c.opposite && c.a.scan == k && c.b.scan == k && c.a.stretch == c.b.stretch) ++per_stretch.at(static_cast<std::size_t>(c.a.stretch));
        for (std::size_t a = 0; a < per_stretch.size(); ++a) {
            if (!per_stretch[a]) continue;
            if (r.stretch < 0 || (sc.stretches[a].helix && !sc.stretches[static_cast<std::size_t>(r.stretch)].helix) ||
                (sc.stretches[a].helix == sc.stretches[static_cast<std::size_t>(r.stretch)].helix && per_stretch[a] > per_stretch[static_cast<std::size_t>(r.stretch)]))
                r.stretch = static_cast<int>(a);
        }
        if (r.stretch >= 0) {
            const StretchLayout& st = sc.stretches[static_cast<std::size_t>(r.stretch)];
            // the record's height against a straight line over the frames on disk
            std::vector<double> frame, height;
            for (std::int64_t f = st.first; f < st.end; ++f) frame.push_back(static_cast<double>(f)), height.push_back(st.course.rows[static_cast<std::size_t>(f)]);
            const Line line = fit_line(frame, height, 0, frame.size());
            std::vector<double> departure(st.course.rows.size());
            for (std::size_t f = 0; f < departure.size(); ++f) departure[f] = st.course.rows[f] - (line.intercept + line.slope * static_cast<double>(f));
            if (std::isfinite(line.slope)) r.departure = rms_of(std::vector<double>(departure.begin() + st.first, departure.begin() + st.end));

            std::vector<double> th, res[3], between, timing, apart;
            for (std::size_t i = 0; i < comparisons.size(); ++i) {
                const Comparison& c = comparisons[i];
                if (!(c.opposite && c.a.scan == k && c.b.scan == k && c.a.stretch == r.stretch && c.b.stretch == r.stretch)) continue;
                th.push_back(ray_of(acq, solution.parameters, c.a).angle_deg * (kPi / 180.0));
                for (int j = 0; j < 3; ++j) res[j].push_back(solution.residual[i][static_cast<std::size_t>(j)]);
                between.push_back(model::at_frame(departure, exposure_frame(acq, c.b)) - model::at_frame(departure, exposure_frame(acq, c.a)));
                const double rows = c.b.row - c.a.row;
                apart.push_back(rows);
                if (std::fabs(rows) >= 16) timing.push_back(solution.residual[i][0] / rows);
            }
            r.n = th.size();
            std::vector<double> one(r.n, 1.0), c1(r.n), s1(r.n), c2(r.n), s2(r.n);
            for (std::size_t i = 0; i < r.n; ++i) c1[i] = std::cos(th[i]), s1[i] = std::sin(th[i]), c2[i] = std::cos(2 * th[i]), s2[i] = std::sin(2 * th[i]);
            for (int j = 0; j < 3; ++j) {
                std::vector<double> left;
                const std::vector<double> x = regress({one, c1, s1, c2, s2}, res[j], &left);
                if (x.empty()) continue;
                r.mean[j] = x[0], r.once[j][0] = x[1], r.once[j][1] = x[2], r.twice[j][0] = x[3], r.twice[j][1] = x[4];
                r.left[j] = rms_of(left);
            }
            r.rows_apart = median_of(apart);
            if (timing.size() >= 8) r.row_time = acq.row_time - median_of(timing);
            // whose the record's departure from the line is: the rows left against its difference between the views
            if (std::isfinite(r.departure)) {
                r.departure_between_views = sd_of(between);
                std::vector<double> left, own;
                const std::vector<double> x = regress({one, between, c1, s1}, res[1], &left);
                const std::vector<double> y = regress({one, c1, s1}, between, &own);  // what of the difference the sinusoid cannot stand for
                double ss = 0;
                for (double v : own) ss += v * v;
                if (!x.empty() && !y.empty() && ss > 0) r.height_slope = 1.0 + x[1], r.height_slope_error = rms_of(left) / std::sqrt(ss);
            }
            axes.push_back(sp.axis), slopes.push_back(sp.axis_slope), scales.push_back(sp.axial_scale);
        }
        const ScanLayout& anchor = acq.scans.at(static_cast<std::size_t>(acq.anchor));
        const double dx = sc.on_stage[0] - anchor.on_stage[0], dy = sc.on_stage[1] - anchor.on_stage[1];
        if (sc.moved_on_stage && std::isfinite(dx) && std::isfinite(dy) && std::hypot(dx, dy) > 0) {
            r.from_anchor = std::hypot(dx, dy);
            const double amplitude = std::hypot(sp.height_once[0], sp.height_once[1]);
            r.pitch = amplitude / r.from_anchor;
            if (amplitude > 0) r.pitch_phase_deg = wrapped_deg((std::atan2(sp.height_once[1], sp.height_once[0]) - std::atan2(dy, dx)) * (180.0 / kPi));
        }
    }
    const auto over = [](const std::vector<double>& v) {
        GeometryReport::Over o;
        o.n = static_cast<int>(v.size());
        if (v.empty()) return o;
        o.range = *std::max_element(v.begin(), v.end()) - *std::min_element(v.begin(), v.end());
        o.sd = sd_of(v), o.mean = mean_of(v);
        return o;
    };
    out.axis = over(axes), out.axis_slope = over(slopes), out.axial_scale = over(scales);
    // one scan against another, per kind: what is left as a course of the angle
    {
        std::map<std::tuple<int, int, std::string>, std::vector<std::size_t>> groups;
        for (std::size_t i = 0; i < comparisons.size(); ++i)
            if (comparisons[i].a.scan != comparisons[i].b.scan) groups[{comparisons[i].a.scan, comparisons[i].b.scan, comparisons[i].kind}].push_back(i);
        for (const auto& [key, members] : groups) {
            PairReport r;
            r.scan_a = std::get<0>(key), r.scan_b = std::get<1>(key), r.kind = std::get<2>(key);
            r.n = members.size();
            std::vector<double> one(r.n, 1.0), c1(r.n), s1(r.n), res[3];
            for (std::size_t m = 0; m < r.n; ++m) {
                const double th = ray_of(acq, solution.parameters, comparisons[members[m]].a).angle_deg * (kPi / 180.0);
                c1[m] = std::cos(th), s1[m] = std::sin(th);
                for (int j = 0; j < 3; ++j) res[j].push_back(solution.residual[members[m]][static_cast<std::size_t>(j)]);
            }
            for (int j = 0; j < 3; ++j) {
                r.rms[j] = rms_of(res[j]), r.median[j] = median_of(res[j]);
                std::vector<double> left;
                const std::vector<double> x = regress({one, c1, s1}, res[j], &left);
                if (x.empty()) continue;
                r.mean[j] = x[0], r.once[j][0] = x[1], r.once[j][1] = x[2], r.left[j] = rms_of(left);
            }
            out.pairs.push_back(r);
        }
    }
    return out;
}

std::string report_json(const Acquisition& acq, const GeometryReport& r) {
    if (r.scans.size() != acq.scans.size()) throw Error("geometry: the report and the acquisition differ in their scans");
    Json scans = Json::list();
    for (std::size_t k = 0; k < r.scans.size(); ++k) {
        const ScanReport& s = r.scans[k];
        Json d = Json::dict();
        d.set("name", Json::of(acq.scans[k].name));
        if (s.stretch >= 0) {
            const char* component[3] = {"frames", "rows", "columns"};
            Json left = Json::dict();
            for (int j = 0; j < 3; ++j)
                left.set(component[j], Json::dict().set("mean", num(s.mean[j])).set("once", pair_of(s.once[j])).set("twice", pair_of(s.twice[j])).set("rms_after", num(s.left[j])));
            Json views = Json::dict();
            views.set("stretch", Json::of(acq.scans[k].stretches[static_cast<std::size_t>(s.stretch)].name)).set("n", Json::whole_number(static_cast<long long>(s.n))).set("left", left);
            views.set("rows_apart", num(s.rows_apart)).set("row_time_frames_per_row", num(s.row_time)).set("row_time_seconds_per_row", num(acq.period_s > 0 ? s.row_time * acq.period_s : kNaN));
            views.set("height_departure_rows", num(s.departure)).set("height_departure_between_views_rows", num(s.departure_between_views));
            views.set("height_slope", num(s.height_slope)).set("height_slope_error", num(s.height_slope_error));
            d.set("opposite_views", views);
        }
        if (std::isfinite(s.from_anchor)) d.set("from_anchor_px", num(s.from_anchor)).set("pitch_at_least_rad", num(s.pitch)).set("pitch_phase_less_direction_deg", num(s.pitch_phase_deg));
        scans.add(d);
    }
    const auto over = [](const GeometryReport::Over& o) { return Json::dict().set("n", Json::whole_number(o.n)).set("range", num(o.range)).set("sd", num(o.sd)).set("mean", num(o.mean)); };
    Json edges = Json::list(), triangles = Json::list();
    for (const auto& e : r.neighbours.edges) edges.add(Json::list().add(Json::whole_number(e[0])).add(Json::whole_number(e[1])));
    for (const auto& t : r.neighbours.triangles) triangles.add(Json::list().add(Json::whole_number(t[0])).add(Json::whole_number(t[1])).add(Json::whole_number(t[2])));
    Json pairs = Json::list();
    for (const PairReport& p : r.pairs) {
        const char* component[3] = {"frames", "rows", "columns"};
        Json left = Json::dict();
        for (int j = 0; j < 3; ++j)
            left.set(component[j], Json::dict().set("rms", num(p.rms[j])).set("median", num(p.median[j])).set("mean", num(p.mean[j])).set("once", pair_of(p.once[j])).set("rms_after", num(p.left[j])));
        pairs.add(Json::dict().set("kind", Json::of(p.kind)).set("scan_a", Json::whole_number(p.scan_a)).set("scan_b", Json::whole_number(p.scan_b)).set("n", Json::whole_number(static_cast<long long>(p.n))).set("left", left));
    }
    Json out = Json::dict();
    out.set("scans", scans).set("pairs", pairs);
    out.set("over_scans", Json::dict().set("axis", over(r.axis)).set("axis_slope", over(r.axis_slope)).set("axial_scale", over(r.axial_scale)));
    out.set("edges", edges).set("triangles", triangles);
    return out.text();
}

}  // namespace helirec::calibration::geometry_calibration
