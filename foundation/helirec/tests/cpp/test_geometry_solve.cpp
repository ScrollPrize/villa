// The joint geometry model from C++ (helirec/calibration/geometry_calibration.hpp): sightings of points in 3D,
// made here by explicit rotation matrices and a parallel projection (nothing of the model's formulas), are the
// lines through those points under the true parameters; a solve from elsewhere finds the parameters again; and
// unknowns that depend on each other are refused. tests/test_geometry_solve.py does the same through the bindings,
// with more cases.
#include <cmath>
#include <iostream>
#include <random>

#include "helirec/calibration/geometry_calibration.hpp"

namespace gc = helirec::calibration::geometry_calibration;

namespace {

constexpr int kRows = 400, kCols = 600, kPerTurn = 360;
constexpr double kPi = 3.141592653589793238462643383279502884;

struct Truth {
    double axis = 310.4, axis_slope = 4e-4, axial_scale = 3.3e-3, d[2] = {7.0, -4.0}, height = 2.5, roll = 3e-4, row_time = 2e-3, rest_height = -0.4;
    double height_once[2] = {0.21, -0.13}, axis_twice[2] = {0.11, 0.17};  // what the stage does within a turn
};

void times(const double m[3][3], const double v[3], double out[3]) {
    for (int i = 0; i < 3; ++i) out[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
}

// where the point p is recorded at the time of frame t of a stretch: (column, row)
void seen(const gc::Acquisition& acq, const Truth& tr, int stretch, const double p[3], double t, double& col, double& row) {
    const gc::StretchLayout& st = acq.scans[0].stretches[static_cast<std::size_t>(stretch)];
    const auto at = [&](const std::vector<double>& v) {
        const std::size_t i = static_cast<std::size_t>(std::floor(t));
        return v[i] + (v[i + 1] - v[i]) * (t - static_cast<double>(i));
    };
    const double th = at(st.course.angle_deg) * (kPi / 180.0), H = at(st.course.rows), mid = (kRows - 1) / 2.0;
    const double stage[3][3] = {{std::cos(th), std::sin(th), 0}, {-std::sin(th), std::cos(th), 0}, {0, 0, 1}};  // the stage turned by th
    const double on_stage[3] = {p[0] + tr.d[0], p[1] + tr.d[1], p[2]};
    double lab[3];
    times(stage, on_stage, lab);  // lab[1] is the depth along the beam: parallel rays drop it
    const double x = lab[0] + tr.axis_slope * (H - acq.scans[0].mid_height) + tr.axis_twice[0] * std::cos(2 * th) + tr.axis_twice[1] * std::sin(2 * th);
    const double z = lab[2] + tr.height + (st.helix ? 0.0 : tr.rest_height) + (1.0 + tr.axial_scale) * H + tr.height_once[0] * std::cos(th) + tr.height_once[1] * std::sin(th);
    // the detector, with the axis rolled against its columns
    col = tr.axis + std::cos(tr.roll) * x + std::sin(tr.roll) * (z - mid);
    row = mid - std::sin(tr.roll) * x + std::cos(tr.roll) * (z - mid);
}

gc::Sighting sighting(const gc::Acquisition& acq, const Truth& tr, int stretch, const double p[3], double t) {
    gc::Sighting g;
    g.scan = 0, g.stretch = stretch;
    seen(acq, tr, stretch, p, t, g.col, g.row);
    g.frame = t - tr.row_time * (g.row - (kRows - 1) / 2.0);
    return g;
}

bool inside(const gc::Sighting& g) { return g.row > 20 && g.row < kRows - 20 && g.col > 20 && g.col < kCols - 20; }

int fail(const char* what) {
    std::cerr << "geometry solve: " << what << '\n';
    return 1;
}

}  // namespace

int main() {
    const Truth tr;
    gc::Acquisition acq;
    acq.rows = kRows, acq.cols = kCols, acq.row_time = tr.row_time;
    gc::ScanLayout scan;
    scan.name = "scan";
    gc::StretchLayout rest, helix;
    rest.name = "rest1", helix.name = "helix", helix.helix = true;
    for (int f = 0; f < kPerTurn; ++f) rest.course.angle_deg.push_back(12.0 + f), rest.course.rows.push_back(0.0);
    for (int f = 0; f < 4 * kPerTurn; ++f) helix.course.angle_deg.push_back(12.0 + kPerTurn + f), helix.course.rows.push_back(f * (300.0 / kPerTurn));
    scan.mid_height = helix.course.rows[helix.course.rows.size() / 2];
    scan.stretches = {rest, helix};
    acq.scans = {scan};

    gc::Parameters truth = gc::empty_parameters(acq);
    truth.roll = tr.roll;
    for (int i = 0; i < 2; ++i) truth.stage.height_once[i] = tr.height_once[i], truth.stage.axis_twice[i] = tr.axis_twice[i];
    gc::ScanParameters& sp = truth.scans[0];
    sp.axis = tr.axis, sp.axis_slope = tr.axis_slope, sp.axial_scale = tr.axial_scale, sp.d[0] = tr.d[0], sp.d[1] = tr.d[1], sp.height = tr.height;
    sp.stretch_height = {tr.rest_height, 0.0};

    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> noise(0.0, 0.004);
    std::vector<gc::Comparison> comparisons;
    double worst = 0;
    for (int i = 0; i < 6000; ++i) {
        const int stretch = i % 4 == 0 ? 0 : 1;
        const std::vector<double>& rows = acq.scans[0].stretches[static_cast<std::size_t>(stretch)].course.rows;
        const double last = static_cast<double>(rows.size()) - 3.0;
        const double t = 2.0 + unit(rng) * (last - 2.0);
        const double r = 250.0 * std::sqrt(unit(rng)), a = 2 * kPi * unit(rng);
        const double p[3] = {r * std::cos(a), r * std::sin(a), 30.0 + unit(rng) * (kRows - 60.0) - rows[static_cast<std::size_t>(t)]};
        const gc::Sighting ga = sighting(acq, tr, stretch, p, t);
        if (!inside(ga)) continue;
        // the sighting is the line through p at the angle it was seen at
        const gc::Ray ray = gc::ray_of(acq, truth, ga);
        const double th = ray.angle_deg * (kPi / 180.0);
        worst = std::max({worst, std::fabs(ray.s - (p[0] * std::cos(th) + p[1] * std::sin(th))), std::fabs(ray.z - p[2])});
        // the same point half a turn on, and a rest turn's point in the helix at the same angle
        gc::Comparison c;
        c.a = ga;
        if (stretch == 0 && i % 8 == 0) {
            c.kind = "resthelix", c.b = sighting(acq, tr, 1, p, t);
        } else {
            if (t + kPerTurn / 2.0 > last) continue;
            c.kind = "opposite", c.opposite = true, c.b = sighting(acq, tr, stretch, p, t + kPerTurn / 2.0);
        }
        if (!inside(c.b)) continue;
        c.b.row += noise(rng), c.b.col += noise(rng);
        comparisons.push_back(c);
    }
    if (worst > 2e-4) return fail("a sighting is not the line through its point");
    if (comparisons.size() < 1000) return fail("too few comparisons were made");

    gc::Parameters start = gc::empty_parameters(acq);
    start.scans[0].axis = tr.axis - 2.2;
    const gc::GeometrySolution got = gc::solve_geometry(acq, comparisons, start);
    const gc::ScanParameters& q = got.parameters.scans[0];
    if (got.rank != 9 || got.free.size() != 9 || !got.unused.empty()) return fail("the free unknowns are not the roll, the stage's four, the scan's own three and the rest turn's height");
    for (int i = 0; i < 2; ++i)
        if (std::fabs(got.parameters.stage.height_once[i] - tr.height_once[i]) > 2e-3 || std::fabs(got.parameters.stage.axis_twice[i] - tr.axis_twice[i]) > 2e-3)
            return fail("the stage's motion within a turn was not found again");
    if (std::fabs(q.axis - tr.axis) > 2e-3 || std::fabs(q.axis_slope - tr.axis_slope) > 5e-6 || std::fabs(q.axial_scale - tr.axial_scale) > 2e-5 ||
        std::fabs(got.parameters.roll - tr.roll) > 1e-5 || std::fabs(q.stretch_height[0] - tr.rest_height) > 3e-3)
        return fail("the parameters were not found again");
    for (const auto& l : got.left)
        if (l.rms[1] > 0.006 || l.rms[2] > 0.006) return fail("more is left than the noise");

    // A calibration made on frames read at a quarter, put into stored pixels (to_stored_pixels): the same detector
    // seen through every fourth pixel has every position a quarter of the stored one, its middle row 1.5 stored
    // rows above the stored middle; converted, it is the stored calibration, and its rays are the stored rays.
    {
        const int f = 4;
        gc::Acquisition still = acq;      // (rows exposed at once: the rows' timing keeps the reduced frames' middle row)
        still.row_time = 0;
        gc::Acquisition quarter = still;
        quarter.rows = kRows / f, quarter.cols = kCols / f;
        quarter.scans[0].mid_height /= f;
        for (gc::StretchLayout& st : quarter.scans[0].stretches)
            for (double& v : st.course.rows) v /= f;
        gc::GeometrySolution reduced;
        reduced.parameters = truth;
        gc::ScanParameters& r = reduced.parameters.scans[0];
        r.axis = (sp.axis - truth.roll * (f - 1) / 2.0) / f, r.d[0] /= f, r.d[1] /= f, r.height /= f;
        for (double& v : r.stretch_height) v /= f;
        for (int i = 0; i < 2; ++i) reduced.parameters.stage.height_once[i] /= f, reduced.parameters.stage.axis_twice[i] /= f;
        const gc::Sighting stored{0, 1, 500.25, 133.0, 402.0}, there{0, 1, 500.25, 133.0 / f, 402.0 / f};
        const gc::Ray want = gc::ray_of(still, truth, stored), at_quarter = gc::ray_of(quarter, reduced.parameters, there);
        if (std::fabs(at_quarter.s * f - want.s) > 1e-9 || std::fabs(at_quarter.z * f - want.z) > 1e-9 || std::fabs(at_quarter.angle_deg - want.angle_deg) > 1e-12)
            return fail("a quarter's ray is not a quarter of the stored ray");
        reduced.residual = {{0.5, 0.25, -0.5}};
        gc::GeometryReport told;
        told.scans.resize(1);
        told.scans[0].rows_apart = 37.5, told.scans[0].row_time = 8e-3;
        gc::to_stored_pixels(quarter, reduced, told, kRows, kCols, f);
        const gc::ScanParameters& back = reduced.parameters.scans[0];
        if (quarter.rows != kRows || quarter.cols != kCols || std::fabs(quarter.scans[0].mid_height - still.scans[0].mid_height) > 1e-9 ||
            std::fabs(quarter.scans[0].stretches[1].course.rows[700] - still.scans[0].stretches[1].course.rows[700]) > 1e-9)
            return fail("the layout was not put into stored pixels");
        if (std::fabs(back.axis - sp.axis) > 1e-9 || std::fabs(back.d[0] - sp.d[0]) > 1e-12 || std::fabs(back.height - sp.height) > 1e-12 || back.axis_slope != sp.axis_slope ||
            back.axial_scale != sp.axial_scale || reduced.parameters.roll != truth.roll || std::fabs(back.stretch_height[0] - sp.stretch_height[0]) > 1e-12 ||
            std::fabs(reduced.parameters.stage.height_once[0] - tr.height_once[0]) > 1e-12 || std::fabs(reduced.parameters.stage.axis_twice[1] - tr.axis_twice[1]) > 1e-12)
            return fail("the parameters were not put into stored pixels");
        const gc::Ray again = gc::ray_of(quarter, reduced.parameters, stored);
        if (std::fabs(again.s - want.s) > 1e-9 || std::fabs(again.z - want.z) > 1e-9) return fail("the converted calibration's ray is not the stored ray");
        if (reduced.residual[0][0] != 0.5 || reduced.residual[0][1] != 1.0 || reduced.residual[0][2] != -2.0 || told.scans[0].rows_apart != 150.0 || told.scans[0].row_time != 2e-3)
            return fail("the residuals and the report were not put into stored pixels");
    }

    gc::GeometryOptions both;
    both.freed = {"scan0.helix.height"};
    try {
        gc::solve_geometry(acq, comparisons, start, both);
        return fail("every stretch's height free was not refused");
    } catch (const helirec::Error& e) {
        if (std::string(e.what()).find("RANK_DEFICIENT") == std::string::npos) return fail(e.what());
    }
    return 0;
}
