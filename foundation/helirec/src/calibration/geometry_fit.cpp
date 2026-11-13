// The joint geometry model's solver (helirec/calibration/geometry_calibration.hpp): comparisons as pairs of
// sightings, one Ceres problem with a block per owner, the refusal of dependent unknowns, and what is left.
#include <ceres/dynamic_autodiff_cost_function.h>

#include <map>
#include <set>

#include "calibration/ceres_support.hpp"
#include "calibration/geometry_model.hpp"
#include "calibration/util.hpp"
#include "helirec/calibration/geometry_calibration.hpp"
#include "io/json.hpp"

namespace helirec::calibration::geometry_calibration {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

// a per-frame series at a fraction of a frame: a line between frames, continued beyond the ends
double at_frame(const std::vector<double>& v, double f) {
    const long n = static_cast<long>(v.size());
    if (n < 1) throw Error("geometry: a stretch without frames");
    if (n == 1) return v[0];
    const long i = std::clamp(static_cast<long>(std::floor(f)), 0L, n - 2);
    const double w = f - static_cast<double>(i);
    return v[static_cast<std::size_t>(i)] * (1.0 - w) + v[static_cast<std::size_t>(i) + 1] * w;
}

const StretchLayout& stretch_of(const Acquisition& acq, const Sighting& g) {
    if (g.scan < 0 || g.scan >= static_cast<int>(acq.scans.size())) throw Error("geometry: a sighting names scan " + std::to_string(g.scan) + ", which the acquisition does not hold");
    const ScanLayout& sc = acq.scans[static_cast<std::size_t>(g.scan)];
    if (g.stretch < 0 || g.stretch >= static_cast<int>(sc.stretches.size())) throw Error("geometry: a sighting names stretch " + std::to_string(g.stretch) + " of scan " + sc.name + ", which it does not hold");
    return sc.stretches[static_cast<std::size_t>(g.stretch)];
}

double mid_row(const Acquisition& acq) { return (acq.rows - 1) / 2.0; }

// the frame at whose time the sighting's row was exposed
double time_of(const Acquisition& acq, const Sighting& g) { return g.frame + acq.row_time * (g.row - mid_row(acq)); }

model::At at_of(const Acquisition& acq, const Sighting& g) {
    const StretchLayout& st = stretch_of(acq, g);
    const double t = time_of(acq, g);
    return {at_frame(st.course.angle_deg, t) * (kPi / 180.0), at_frame(st.course.rows, t), acq.scans[static_cast<std::size_t>(g.scan)].mid_height, mid_row(acq)};
}

// the angle [deg] by which the ray recorded at the sighting's column is further on than the stage: the beam
// diverges, and the ray of a column further on is that of a parallel beam further on in angle
double fan_angle(const Acquisition& acq, const Sighting& g) {
    if (acq.fan == 0.0) return 0.0;
    const StretchLayout& st = stretch_of(acq, g);
    const double t = time_of(acq, g);
    const double step = at_frame(st.course.angle_deg, t + 0.5) - at_frame(st.course.angle_deg, t - 0.5);
    return acq.fan * std::fabs(step) * (g.col - (acq.cols - 1) / 2.0);
}

// The unknowns as the problem holds them: a block per owner.
struct Blocks {
    double detector[1] = {0};
    double stage[4] = {0, 0, 0, 0};
    std::vector<std::array<double, 3>> own, place;       // per scan
    std::vector<std::vector<std::array<double, 1>>> stretch;  // per scan, per stretch
};
Blocks blocks_of(const Acquisition& acq, const Parameters& p) {
    if (p.scans.size() != acq.scans.size()) throw Error("geometry: the parameters hold " + std::to_string(p.scans.size()) + " scans, the acquisition " + std::to_string(acq.scans.size()));
    Blocks b;
    b.detector[0] = p.roll;
    b.stage[0] = p.stage.height_once[0], b.stage[1] = p.stage.height_once[1], b.stage[2] = p.stage.axis_twice[0], b.stage[3] = p.stage.axis_twice[1];
    for (std::size_t k = 0; k < p.scans.size(); ++k) {
        const ScanParameters& s = p.scans[k];
        if (s.stretch_height.size() != acq.scans[k].stretches.size()) throw Error("geometry: the parameters of scan " + acq.scans[k].name + " do not hold one height per stretch");
        b.own.push_back({s.axis, s.axis_slope, s.axial_scale});
        b.place.push_back({s.d[0], s.d[1], s.height});
        b.stretch.emplace_back();
        for (double h : s.stretch_height) b.stretch.back().push_back({h});
    }
    return b;
}
Parameters parameters_of(const Blocks& b) {
    Parameters p;
    p.roll = b.detector[0];
    p.stage.height_once[0] = b.stage[0], p.stage.height_once[1] = b.stage[1], p.stage.axis_twice[0] = b.stage[2], p.stage.axis_twice[1] = b.stage[3];
    for (std::size_t k = 0; k < b.own.size(); ++k) {
        ScanParameters s;
        s.axis = b.own[k][0], s.axis_slope = b.own[k][1], s.axial_scale = b.own[k][2];
        s.d[0] = b.place[k][0], s.d[1] = b.place[k][1], s.height = b.place[k][2];
        for (const auto& h : b.stretch[k]) s.stretch_height.push_back(h[0]);
        p.scans.push_back(s);
    }
    return p;
}

// one unknown: its name, its block and its place in it, and whether it is free unless held
struct Slot {
    std::string name;
    double* block = nullptr;
    int size = 0, index = 0;
    bool free = false;
};
std::vector<Slot> slots_of(const Acquisition& acq, Blocks& b) {
    std::vector<Slot> out;
    out.push_back({"roll", b.detector, 1, 0, true});
    const char* stage[4] = {"height_once_cos", "height_once_sin", "axis_twice_cos", "axis_twice_sin"};
    for (int i = 0; i < 4; ++i) out.push_back({std::string("stage.") + stage[i], b.stage, 4, i, true});
    for (std::size_t k = 0; k < acq.scans.size(); ++k) {
        const std::string scan = "scan" + std::to_string(k) + ".";
        const char* own[3] = {"axis", "axis_slope", "axial_scale"};
        for (int i = 0; i < 3; ++i) out.push_back({scan + own[i], b.own[k].data(), 3, i, true});
        const char* place[3] = {"d0", "d1", "height"};
        for (int i = 0; i < 3; ++i) out.push_back({scan + place[i], b.place[k].data(), 3, i, false});
        for (std::size_t a = 0; a < acq.scans[k].stretches.size(); ++a) {
            const StretchLayout& st = acq.scans[k].stretches[a];
            out.push_back({scan + st.name + ".height", b.stretch[k][a].data(), 1, 0, !st.helix});
        }
    }
    return out;
}

// One comparison's residual: the two rays' difference in the sample (columns, rows).
struct SameLines {
    model::At a, b;
    double xa = 0, ya = 0, xb = 0, yb = 0;  // ideal positions
    bool opposite = false;
    int at_a[3] = {0, 0, 0}, at_b[3] = {0, 0, 0};  // where the residual's parameters hold own, place, stretch (0 is the detector, 1 the stage)

    template <class T>
    bool operator()(T const* const* p, T* r) const {
        T sa, za, sb, zb;
        model::ray_of(p[0], p[1], p[at_a[0]], p[at_a[1]], p[at_a[2]], a, T(xa), T(ya), sa, za);
        model::ray_of(p[0], p[1], p[at_b[0]], p[at_b[1]], p[at_b[2]], b, T(xb), T(yb), sb, zb);
        r[0] = opposite ? sb + sa : sb - sa;
        r[1] = zb - za;
        return true;
    }
};

// the problem's blocks of one comparison, each once, and where the residual finds them
struct Wiring {
    SameLines f;
    std::vector<double*> blocks;
    std::vector<int> sizes;
};
Wiring wire(const Acquisition& acq, Blocks& b, const Comparison& c) {
    Wiring w;
    w.f.a = at_of(acq, c.a), w.f.b = at_of(acq, c.b);
    w.f.xa = c.a.col, w.f.ya = c.a.row, w.f.xb = c.b.col, w.f.yb = c.b.row;
    w.f.opposite = c.opposite;
    w.blocks.push_back(b.detector), w.sizes.push_back(1);
    w.blocks.push_back(b.stage), w.sizes.push_back(4);
    const auto place = [&](double* block, int size) {
        for (std::size_t i = 0; i < w.blocks.size(); ++i)
            if (w.blocks[i] == block) return static_cast<int>(i);
        w.blocks.push_back(block), w.sizes.push_back(size);
        return static_cast<int>(w.blocks.size()) - 1;
    };
    const auto side = [&](const Sighting& g, int* at) {
        const std::size_t k = static_cast<std::size_t>(g.scan), a = static_cast<std::size_t>(g.stretch);
        at[0] = place(b.own[k].data(), 3), at[1] = place(b.place[k].data(), 3), at[2] = place(b.stretch[k][a].data(), 1);
    };
    side(c.a, w.f.at_a), side(c.b, w.f.at_b);
    return w;
}

std::array<double, 3> residual_of(const Acquisition& acq, const Parameters& p, const Comparison& c) {
    const Ray a = ray_of(acq, p, c.a), b = ray_of(acq, p, c.b);
    const StretchLayout& sb = stretch_of(acq, c.b);
    // B's angle per frame around the sighting
    const double t = time_of(acq, c.b);
    const double step = at_frame(sb.course.angle_deg, t + 0.5) - at_frame(sb.course.angle_deg, t - 0.5);
    double dth = b.angle_deg - a.angle_deg - (c.opposite ? 180.0 : 0.0);
    dth -= 360.0 * std::round(dth / 360.0);
    return {step != 0 ? dth / step : 0.0, b.z - a.z, c.opposite ? b.s + a.s : b.s - a.s};
}

double median(std::vector<double> v) { return median_of(std::move(v)); }

}  // namespace

Parameters empty_parameters(const Acquisition& acq) {
    Parameters p;
    for (const ScanLayout& s : acq.scans) {
        ScanParameters q;
        q.stretch_height.assign(s.stretches.size(), 0.0);
        p.scans.push_back(q);
    }
    return p;
}

Ray ray_of(const Acquisition& acq, const Parameters& p, const Sighting& g) {
    stretch_of(acq, g);
    if (p.scans.size() != acq.scans.size()) throw Error("geometry: the parameters and the acquisition differ in their scans");
    const ScanParameters& sp = p.scans[static_cast<std::size_t>(g.scan)];
    const model::At at = at_of(acq, g);
    const double detector[1] = {p.roll}, own[3] = {sp.axis, sp.axis_slope, sp.axial_scale}, place[3] = {sp.d[0], sp.d[1], sp.height};
    const double stage[4] = {p.stage.height_once[0], p.stage.height_once[1], p.stage.axis_twice[0], p.stage.axis_twice[1]};
    const double stretch[1] = {sp.stretch_height.at(static_cast<std::size_t>(g.stretch))};
    Ray r;
    r.angle_deg = at.th * (180.0 / kPi) + fan_angle(acq, g);
    model::ray_of(detector, stage, own, place, stretch, at, g.col, g.row, r.s, r.z);
    return r;
}

std::vector<std::array<double, 3>> residuals(const Acquisition& acq, const Parameters& p, const std::vector<Comparison>& comparisons) {
    std::vector<std::array<double, 3>> out;
    out.reserve(comparisons.size());
    for (const Comparison& c : comparisons) out.push_back(residual_of(acq, p, c));
    return out;
}

Comparison comparison_of(const Acquisition& acq, const std::string& kind, int scan_a, int stretch_a, int scan_b, int stretch_b, const ViewRequest& q, const ViewMeasurement& m) {
    Comparison c;
    c.kind = kind;
    c.opposite = q.mirror;
    c.correlation = m.correlation;
    c.a.scan = scan_a, c.a.stretch = stretch_a, c.b.scan = scan_b, c.b.stretch = stretch_b;
    const Trajectory& ta = stretch_of(acq, c.a).course;
    const Trajectory& tb = stretch_of(acq, c.b).course;
    // A's block: frames frame_a .., rows row_a .., columns col_a ..; its middle
    const double half = (m.frames - 1) / 2.0, across = (m.cols - 1) / 2.0;
    c.a.frame = static_cast<double>(m.frame_a) + half;
    c.a.row = m.row_a + (m.rows - 1) / 2.0;
    c.a.col = m.col_a + across;
    // B's block was cut where the record puts the same lines, frame by frame, and the registration found A's
    // content m.dq frames, m.dv rows and m.du columns further on in it (registration.hpp; view_plan.hpp states the
    // cut: frame q of B's block lies rows_b - rows_a of that frame above A's rows)
    c.b.frame = static_cast<double>(m.frame_b) + half + m.dq;
    c.b.row = c.a.row + m.dv + at_frame(tb.rows, c.b.frame) - at_frame(ta.rows, c.a.frame + m.dq);
    c.b.col = q.mirror ? m.col_b + across - m.du : m.col_b + across + m.du;
    if (q.sway_px != 0.0) {  // B was read that much further on (session.hpp)
        const double angle = at_frame(tb.angle_deg, c.b.frame) + q.sway_row_deg * (c.b.row - mid_row(acq)) - q.sway_phase_deg;
        c.b.col += q.sway_px * std::cos(angle * (kPi / 180.0));
    }
    return c;
}

GeometrySolution solve_geometry(const Acquisition& acq, const std::vector<Comparison>& comparisons, const Parameters& start, const GeometryOptions& options) {
    SolveOptions so;
    so.huber = options.huber, so.max_iterations = options.max_iterations;
    fitting::validate(so);
    Blocks blocks = blocks_of(acq, start);
    std::vector<Slot> slots = slots_of(acq, blocks);
    {
        std::set<std::string> known;
        for (const Slot& s : slots) known.insert(s.name);
        for (const auto* list : {&options.held, &options.freed})
            for (const std::string& name : *list)
                if (!known.count(name)) throw Error("geometry: no unknown is named " + name);
        for (Slot& s : slots) {
            if (std::count(options.freed.begin(), options.freed.end(), s.name)) s.free = true;
            if (std::count(options.held.begin(), options.held.end(), s.name)) s.free = false;
        }
    }
    std::vector<Wiring> wiring;
    wiring.reserve(comparisons.size());
    for (const Comparison& c : comparisons) wiring.push_back(wire(acq, blocks, c));

    // The problem with the free unknowns as they stand in `slots`, and its Jacobian's columns: one per free unknown
    // that some comparison's block holds, in the order of the slots.
    struct Built {
        std::unique_ptr<ceres::Problem> problem;
        std::vector<double*> free_blocks;
        std::vector<std::size_t> column_slot;  // per column of the Jacobian over free_blocks: the slot
    };
    const auto build = [&]() {
        Built out;
        out.problem = std::make_unique<ceres::Problem>();
        ceres::Problem& problem = *out.problem;
        ceres::LossFunction* loss = options.huber > 0 ? new ceres::HuberLoss(options.huber) : nullptr;
        for (const Wiring& w : wiring) {
            auto* cost = new ceres::DynamicAutoDiffCostFunction<SameLines, 4>(new SameLines(w.f));
            for (int size : w.sizes) cost->AddParameterBlock(size);
            cost->SetNumResiduals(2);
            problem.AddResidualBlock(cost, loss, w.blocks);
        }
        if (wiring.empty() && loss) delete loss;
        // each block that the problem holds: constant, or with a part of it held
        std::map<double*, std::vector<std::size_t>> by_block;
        std::vector<double*> order;
        for (std::size_t i = 0; i < slots.size(); ++i) {
            if (!problem.HasParameterBlock(slots[i].block)) continue;
            if (!by_block.count(slots[i].block)) order.push_back(slots[i].block);
            by_block[slots[i].block].push_back(i);
        }
        for (double* block : order) {
            const std::vector<std::size_t>& in = by_block[block];
            std::vector<int> held;
            for (std::size_t i : in)
                if (!slots[i].free) held.push_back(slots[i].index);
            if (held.size() == in.size()) {
                problem.SetParameterBlockConstant(block);
                continue;
            }
            if (!held.empty()) problem.SetManifold(block, new ceres::SubsetManifold(slots[in[0]].size, held));
            out.free_blocks.push_back(block);
            for (std::size_t i : in)
                if (slots[i].free) out.column_slot.push_back(i);
        }
        return out;
    };
    // the Jacobian over the free unknowns with its columns normalised, and the columns' norms
    const auto design = [&](Built& b, std::vector<double>& norm) {
        ceres::Problem::EvaluateOptions eo;
        eo.parameter_blocks = b.free_blocks;
        eo.apply_loss_function = false;
        ceres::CRSMatrix J;
        if (!b.problem->Evaluate(eo, nullptr, nullptr, nullptr, &J)) throw Error("geometry: the residuals could not be evaluated");
        norm.assign(b.column_slot.size(), 0.0);
        for (std::size_t e = 0; e < J.values.size(); ++e) norm[static_cast<std::size_t>(J.cols[e])] += J.values[e] * J.values[e];
        for (double& v : norm) v = std::sqrt(v);
        std::vector<Eigen::Triplet<double>> entries;
        entries.reserve(J.values.size());
        for (int r = 0; r < J.num_rows; ++r)
            for (int e = J.rows[static_cast<std::size_t>(r)]; e < J.rows[static_cast<std::size_t>(r) + 1]; ++e) {
                const double n = norm[static_cast<std::size_t>(J.cols[static_cast<std::size_t>(e)])];
                if (n > 0) entries.emplace_back(r, J.cols[static_cast<std::size_t>(e)], J.values[static_cast<std::size_t>(e)] / n);
            }
        Eigen::SparseMatrix<double> d(J.num_rows, static_cast<int>(b.column_slot.size()));
        d.setFromTriplets(entries.begin(), entries.end());
        return d;
    };

    GeometrySolution out;
    out.solver.solver_version = CERES_VERSION_STRING;
    // unknowns asked free that no comparison depends on are held where they started, and said
    {
        Built first = build();
        std::set<std::size_t> used(first.column_slot.begin(), first.column_slot.end());
        std::vector<double> norm;
        if (!first.column_slot.empty()) design(first, norm);
        for (std::size_t c = 0; c < first.column_slot.size(); ++c)
            if (norm[c] == 0.0) used.erase(first.column_slot[c]);
        for (std::size_t i = 0; i < slots.size(); ++i)
            if (slots[i].free && !used.count(i)) slots[i].free = false, out.unused.push_back(slots[i].name);
    }
    Built built = build();
    for (std::size_t i : built.column_slot) out.free.push_back(slots[i].name);
    if (comparisons.empty() || out.free.empty()) throw Error("INSUFFICIENT_SUPPORT: the geometry's solve has " + std::to_string(comparisons.size()) + " comparisons for " + std::to_string(out.free.size()) + " unknowns");
    ceres::Solver::Summary summary;
    ceres::Solve(fitting::options(static_cast<int>(out.free.size()), so), built.problem.get(), &summary);
    out.solver = fitting::summary(summary);
    {
        std::vector<double> norm;
        const Eigen::SparseMatrix<double> d = design(built, norm);
        std::vector<int> dependent;
        out.rank = fitting::design_rank(d, so.rcond, &dependent);
        if (out.rank != static_cast<int>(out.free.size())) {
            std::string names;
            for (int c : dependent) names += (names.empty() ? "" : ", ") + out.free[static_cast<std::size_t>(c)];
            throw Error("RANK_DEFICIENT: the geometry's free unknowns depend on each other (rank " + std::to_string(out.rank) + " of " + std::to_string(out.free.size()) +
                        "); hold one of each dependent set, among them: " + names);
        }
    }
    if (!out.solver.usable) throw Error(out.solver.termination + ": " + out.solver.message);
    out.parameters = parameters_of(blocks);

    out.residual = residuals(acq, out.parameters, comparisons);
    std::map<std::tuple<std::string, int, int>, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < comparisons.size(); ++i) {
        const Comparison& c = comparisons[i];
        groups[{c.kind, -1, -1}].push_back(i);
        groups[{c.kind, c.a.scan, c.b.scan}].push_back(i);
    }
    for (const auto& [key, members] : groups) {
        GeometrySolution::Left l;
        l.kind = std::get<0>(key), l.scan_a = std::get<1>(key), l.scan_b = std::get<2>(key);
        l.n = members.size();
        for (int j = 0; j < 3; ++j) {
            std::vector<double> v;
            for (std::size_t i : members) v.push_back(out.residual[i][static_cast<std::size_t>(j)]);
            l.rms[j] = rms_of(v), l.median[j] = median(v);
        }
        out.left.push_back(l);
    }
    return out;
}

std::string solution_json(const Acquisition& acq, const GeometrySolution& s) {
    const auto num = [](double v) { return Json::of(v); };
    const auto names = [](const std::vector<std::string>& v) {
        Json j = Json::list();
        for (const std::string& x : v) j.add(Json::of(x));
        return j;
    };
    Json scans = Json::list();
    for (std::size_t k = 0; k < s.parameters.scans.size(); ++k) {
        const ScanParameters& p = s.parameters.scans[k];
        Json heights = Json::dict();
        for (std::size_t a = 0; a < p.stretch_height.size(); ++a) heights.set(acq.scans[k].stretches[a].name, num(p.stretch_height[a]));
        Json d = Json::dict();
        d.set("name", Json::of(acq.scans[k].name)).set("axis", num(p.axis)).set("axis_slope", num(p.axis_slope)).set("axial_scale", num(p.axial_scale));
        d.set("d", Json::list().add(num(p.d[0])).add(num(p.d[1]))).set("height", num(p.height)).set("stretch_height", heights).set("mid_height", num(acq.scans[k].mid_height));
        scans.add(d);
    }
    Json left = Json::list();
    for (const GeometrySolution::Left& l : s.left) {
        Json d = Json::dict();
        d.set("kind", Json::of(l.kind)).set("scan_a", Json::whole_number(l.scan_a)).set("scan_b", Json::whole_number(l.scan_b)).set("n", Json::whole_number(static_cast<long long>(l.n)));
        d.set("rms", Json::list().add(num(l.rms[0])).add(num(l.rms[1])).add(num(l.rms[2]))).set("median", Json::list().add(num(l.median[0])).add(num(l.median[1])).add(num(l.median[2])));
        left.add(d);
    }
    Json solver = Json::dict();
    solver.set("cost", num(s.solver.cost)).set("iterations", Json::whole_number(s.solver.iterations)).set("evaluations", Json::whole_number(s.solver.evaluations));
    solver.set("usable", Json::of_bool(s.solver.usable)).set("termination", Json::of(s.solver.termination)).set("message", Json::of(s.solver.message));
    solver.set("solver_version", Json::of(s.solver.solver_version)).set("rank", Json::whole_number(s.rank));
    Json out = Json::dict();
    const StageParameters& st = s.parameters.stage;
    Json stage = Json::dict();
    stage.set("height_once", Json::list().add(num(st.height_once[0])).add(num(st.height_once[1]))).set("axis_twice", Json::list().add(num(st.axis_twice[0])).add(num(st.axis_twice[1])));
    out.set("roll", num(s.parameters.roll)).set("stage", stage).set("scans", scans).set("free", names(s.free)).set("unused", names(s.unused)).set("solver", solver).set("left", left);
    return out.text();
}

}  // namespace helirec::calibration::geometry_calibration
