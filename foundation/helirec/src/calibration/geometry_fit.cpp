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

using model::at_frame;

const StretchLayout& stretch_of(const Acquisition& acq, const Sighting& g) {
    if (g.scan < 0 || g.scan >= static_cast<int>(acq.scans.size())) throw Error("geometry: a sighting names scan " + std::to_string(g.scan) + ", which the acquisition does not hold");
    const ScanLayout& sc = acq.scans[static_cast<std::size_t>(g.scan)];
    if (g.stretch < 0 || g.stretch >= static_cast<int>(sc.stretches.size())) throw Error("geometry: a sighting names stretch " + std::to_string(g.stretch) + " of scan " + sc.name + ", which it does not hold");
    return sc.stretches[static_cast<std::size_t>(g.stretch)];
}

double mid_row(const Acquisition& acq) { return (acq.rows - 1) / 2.0; }

double time_of(const Acquisition& acq, const Sighting& g) { return exposure_frame(acq, g); }

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

// the terms of a map, or an error if its lists differ in length
int terms_of(const geometry::Map& m) {
    if (m.j.size() != m.i.size() || m.eu.size() != m.i.size() || m.ev.size() != m.i.size()) throw Error("geometry: the map's lists differ in length");
    return static_cast<int>(m.i.size());
}

// The unknowns as the problem holds them: a block per owner.
struct Blocks {
    double detector[1] = {0};
    double stage[4] = {0, 0, 0, 0};
    geometry::Map shape;      // the map's terms and normalisation
    std::vector<double> map;  // Eu's coefficients, then Ev's (empty: no map)
    std::vector<std::array<double, 3>> own, place;       // per scan
    std::vector<std::array<double, 2>> once;             // per scan: the height once per turn of its own views
    std::vector<std::vector<std::array<double, 1>>> stretch;  // per scan, per stretch
    int terms() const { return static_cast<int>(shape.i.size()); }
};
Blocks blocks_of(const Acquisition& acq, const Parameters& p) {
    if (p.scans.size() != acq.scans.size()) throw Error("geometry: the parameters hold " + std::to_string(p.scans.size()) + " scans, the acquisition " + std::to_string(acq.scans.size()));
    Blocks b;
    terms_of(p.map);
    b.shape = p.map;
    b.map = p.map.eu;
    b.map.insert(b.map.end(), p.map.ev.begin(), p.map.ev.end());
    b.detector[0] = p.roll;
    b.stage[0] = p.stage.height_once[0], b.stage[1] = p.stage.height_once[1], b.stage[2] = p.stage.axis_twice[0], b.stage[3] = p.stage.axis_twice[1];
    for (std::size_t k = 0; k < p.scans.size(); ++k) {
        const ScanParameters& s = p.scans[k];
        if (s.stretch_height.size() != acq.scans[k].stretches.size()) throw Error("geometry: the parameters of scan " + acq.scans[k].name + " do not hold one height per stretch");
        b.own.push_back({s.axis, s.axis_slope, s.axial_scale});
        b.place.push_back({s.d[0], s.d[1], s.height});
        b.once.push_back({s.height_once[0], s.height_once[1]});
        b.stretch.emplace_back();
        for (double h : s.stretch_height) b.stretch.back().push_back({h});
    }
    return b;
}
Parameters parameters_of(const Blocks& b) {
    Parameters p;
    p.map = b.shape;
    p.map.eu.assign(b.map.begin(), b.map.begin() + b.terms());
    p.map.ev.assign(b.map.begin() + b.terms(), b.map.end());
    p.roll = b.detector[0];
    p.stage.height_once[0] = b.stage[0], p.stage.height_once[1] = b.stage[1], p.stage.axis_twice[0] = b.stage[2], p.stage.axis_twice[1] = b.stage[3];
    for (std::size_t k = 0; k < b.own.size(); ++k) {
        ScanParameters s;
        s.axis = b.own[k][0], s.axis_slope = b.own[k][1], s.axial_scale = b.own[k][2];
        s.d[0] = b.place[k][0], s.d[1] = b.place[k][1], s.height = b.place[k][2];
        s.height_once[0] = b.once[k][0], s.height_once[1] = b.once[k][1];
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
    int place_of = -1;   // the scan whose place (d, height) it is part of; -1: none
    bool gauge = false;  // never free: nothing can tell it from another unknown
};
std::vector<Slot> slots_of(const Acquisition& acq, Blocks& b) {
    if (acq.anchor < 0 || acq.anchor >= static_cast<int>(acq.scans.size())) throw Error("geometry: the acquisition's anchor is not one of its scans");
    std::vector<Slot> out;
    out.push_back({"roll", b.detector, 1, 0, true});
    const char* stage[4] = {"height_once_cos", "height_once_sin", "axis_twice_cos", "axis_twice_sin"};
    for (int i = 0; i < 4; ++i) out.push_back({std::string("stage.") + stage[i], b.stage, 4, i, true});
    const int nt = b.terms();
    for (int half = 0; half < 2; ++half)
        for (int k = 0; k < nt; ++k) {
            const std::size_t t = static_cast<std::size_t>(k);
            Slot s{fmt("map.%s.i%dj%d", half ? "ev" : "eu", b.shape.i[t], b.shape.j[t]), b.map.data(), 2 * nt, half * nt + k, false};
            s.gauge = !half && b.shape.i[t] == 1 && b.shape.j[t] == 0;  // Eu's line in the column: a scale of the columns
            out.push_back(s);
        }
    for (std::size_t k = 0; k < acq.scans.size(); ++k) {
        const std::string scan = "scan" + std::to_string(k) + ".";
        const bool anchor = static_cast<int>(k) == acq.anchor;
        const char* own[3] = {"axis", "axis_slope", "axial_scale"};
        for (int i = 0; i < 3; ++i) out.push_back({scan + own[i], b.own[k].data(), 3, i, true});
        const char* place[3] = {"d0", "d1", "height"};
        for (int i = 0; i < 3; ++i) {
            Slot s{scan + place[i], b.place[k].data(), 3, i, !anchor};
            s.place_of = static_cast<int>(k);
            out.push_back(s);
        }
        // a tile's own views beside the anchor's: the anchor's are the stage's
        const char* once[2] = {"height_once_cos", "height_once_sin"};
        for (int i = 0; i < 2; ++i) out.push_back({scan + once[i], b.once[k].data(), 2, i, !anchor && acq.scans[k].moved_on_stage});
        for (std::size_t a = 0; a < acq.scans[k].stretches.size(); ++a) {
            const StretchLayout& st = acq.scans[k].stretches[a];
            out.push_back({scan + st.name + ".height", b.stretch[k][a].data(), 1, 0, !st.helix});
        }
    }
    return out;
}
// whether a name of GeometryOptions names the unknown: the unknown's own name, or its beginning up to a dot
bool names(const std::string& given, const std::string& unknown) {
    return unknown == given || (unknown.size() > given.size() && unknown.compare(0, given.size(), given) == 0 && unknown[given.size()] == '.');
}

// One comparison's residual: the two rays' difference in the sample (columns, rows).
struct SameLines {
    model::At a, b;
    double ca = 0, va = 0, cb = 0, vb = 0;  // the stored pixels
    std::vector<double> phi_a, phi_b;       // the map's terms there
    bool opposite = false;
    int at_map = -1;                               // where the residual's parameters hold the map; -1: none
    int at_once = -1;                              // and the scan's height once per turn in its own views; -1: two scans
    int at_a[3] = {0, 0, 0}, at_b[3] = {0, 0, 0};  // and own, place, stretch of each side (0 is the detector, 1 the stage)

    template <class T>
    bool operator()(T const* const* p, T* r) const {
        const int terms = static_cast<int>(phi_a.size());
        const T* map = at_map < 0 ? nullptr : p[at_map];
        T xa, ya, xb, yb, sa, za, sb, zb;
        model::ideal_position(map, phi_a.data(), terms, ca, va, xa, ya);
        model::ideal_position(map, phi_b.data(), terms, cb, vb, xb, yb);
        model::ray_of(p[0], p[1], p[at_a[0]], p[at_a[1]], p[at_a[2]], a, xa, ya, sa, za);
        model::ray_of(p[0], p[1], p[at_b[0]], p[at_b[1]], p[at_b[2]], b, xb, yb, sb, zb);
        r[0] = opposite ? sb + sa : sb - sa;
        r[1] = zb - za;
        if (at_once >= 0) r[1] -= model::own_lift(p[at_once], a, b);
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
    w.f.ca = c.a.col, w.f.va = c.a.row, w.f.cb = c.b.col, w.f.vb = c.b.row;
    w.f.opposite = c.opposite;
    w.blocks.push_back(b.detector), w.sizes.push_back(1);
    w.blocks.push_back(b.stage), w.sizes.push_back(4);
    if (const int nt = b.terms()) {
        w.f.phi_a.resize(static_cast<std::size_t>(nt)), w.f.phi_b.resize(static_cast<std::size_t>(nt));
        geometry::map_basis(b.shape, c.a.col, c.a.row, w.f.phi_a.data());
        geometry::map_basis(b.shape, c.b.col, c.b.row, w.f.phi_b.data());
        w.f.at_map = static_cast<int>(w.blocks.size());
        w.blocks.push_back(b.map.data()), w.sizes.push_back(2 * nt);
    }
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
    if (c.a.scan == c.b.scan) w.f.at_once = place(b.once[static_cast<std::size_t>(c.a.scan)].data(), 2);
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
    double rows = b.z - a.z;
    if (c.a.scan == c.b.scan) rows -= model::own_lift(p.scans[static_cast<std::size_t>(c.a.scan)].height_once, at_of(acq, c.a), at_of(acq, c.b));
    return {step != 0 ? dth / step : 0.0, rows, c.opposite ? b.s + a.s : b.s - a.s};
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
    const int nt = terms_of(p.map);
    std::vector<double> map = p.map.eu, phi(static_cast<std::size_t>(nt));
    map.insert(map.end(), p.map.ev.begin(), p.map.ev.end());
    geometry::map_basis(p.map, g.col, g.row, phi.data());
    double X, Y;
    model::ideal_position(map.data(), phi.data(), nt, g.col, g.row, X, Y);
    Ray r;
    r.angle_deg = at.th * (180.0 / kPi) + fan_angle(acq, g);
    model::ray_of(detector, stage, own, place, stretch, at, X, Y, r.s, r.z);
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
        for (const auto* list : {&options.held, &options.freed})
            for (const std::string& name : *list)
                if (std::none_of(slots.begin(), slots.end(), [&](const Slot& s) { return names(name, s.name); })) throw Error("geometry: no unknown is named " + name);
        const auto among = [](const std::vector<std::string>& list, const Slot& s) { return std::any_of(list.begin(), list.end(), [&](const std::string& n) { return names(n, s.name); }); };
        for (Slot& s : slots) {
            if (among(options.freed, s)) s.free = true;
            if (among(options.held, s)) s.free = false;
            if (s.gauge && s.free) {
                if (std::count(options.freed.begin(), options.freed.end(), s.name)) throw Error("geometry: " + s.name + " cannot be told from the pixel size and is not solved for");
                s.free = false;
            }
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
    // unknowns asked free that no comparison depends on are held where they started, and said: first a scan's place,
    // which only comparisons with another scan give
    {
        std::vector<std::uint8_t> joined(acq.scans.size(), 0);
        for (const Comparison& c : comparisons)
            if (c.a.scan != c.b.scan) joined[static_cast<std::size_t>(c.a.scan)] = joined[static_cast<std::size_t>(c.b.scan)] = 1;
        for (Slot& s : slots)
            if (s.free && s.place_of >= 0 && !joined[static_cast<std::size_t>(s.place_of)]) s.free = false, out.unused.push_back(s.name);
    }
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

namespace {
Json num(double v) { return Json::of(v); }
Json numbers(const std::vector<double>& v) {
    Json j = Json::list();
    for (double x : v) j.add(num(x));
    return j;
}
// the parameters by name, the map apart (a calibration file holds it at its top, where the reconstruction reads it)
void parameters_json(const Acquisition& acq, const Parameters& p, Json& out, Json& map) {
    if (p.scans.size() != acq.scans.size()) throw Error("geometry: the parameters and the acquisition differ in their scans");
    Json scans = Json::list();
    for (std::size_t k = 0; k < p.scans.size(); ++k) {
        const ScanParameters& q = p.scans[k];
        Json heights = Json::dict();
        for (std::size_t a = 0; a < q.stretch_height.size(); ++a) heights.set(acq.scans[k].stretches[a].name, num(q.stretch_height[a]));
        Json d = Json::dict();
        d.set("name", Json::of(acq.scans[k].name)).set("axis", num(q.axis)).set("axis_slope", num(q.axis_slope)).set("axial_scale", num(q.axial_scale));
        d.set("d", Json::list().add(num(q.d[0])).add(num(q.d[1]))).set("height", num(q.height)).set("stretch_height", heights).set("mid_height", num(acq.scans[k].mid_height));
        d.set("height_once", Json::list().add(num(q.height_once[0])).add(num(q.height_once[1])));
        scans.add(d);
    }
    Json stage = Json::dict();
    stage.set("height_once", Json::list().add(num(p.stage.height_once[0])).add(num(p.stage.height_once[1]))).set("axis_twice", Json::list().add(num(p.stage.axis_twice[0])).add(num(p.stage.axis_twice[1])));
    out.set("roll", num(p.roll)).set("stage", stage).set("scans", scans);
    Json mi = Json::list(), mj = Json::list();
    for (const int v : p.map.i) mi.add(Json::whole_number(v));
    for (const int v : p.map.j) mj.add(Json::whole_number(v));
    map = Json::dict();
    map.set("c0", num(p.map.c0)).set("cs", num(p.map.cs)).set("v0", num(p.map.v0)).set("vs", num(p.map.vs)).set("i", mi).set("j", mj).set("eu", numbers(p.map.eu)).set("ev", numbers(p.map.ev));
}
}  // namespace

std::string parameters_json(const Acquisition& acq, const Parameters& p) {
    Json out = Json::dict(), map;
    parameters_json(acq, p, out, map);
    out.set("map", map);
    return out.text();
}

std::string solution_json(const Acquisition& acq, const GeometrySolution& s) {
    const auto names = [](const std::vector<std::string>& v) {
        Json j = Json::list();
        for (const std::string& x : v) j.add(Json::of(x));
        return j;
    };
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
    Json out = Json::dict(), map;
    parameters_json(acq, s.parameters, out, map);
    out.set("map", map).set("free", names(s.free)).set("unused", names(s.unused)).set("solver", solver).set("left", left);
    return out.text();
}

}  // namespace helirec::calibration::geometry_calibration
