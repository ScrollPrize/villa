// vc_tomography/geometry.hpp: the detector's map, the poses of every frame from a geometry calibration, and the
// reconstruction's per-projection terms.
#include "vc_tomography/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include "calibration/util.hpp"
#include "io/json.hpp"
#include "io/text_file.hpp"

namespace vc_tomography::geometry {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
inline double radians(double d) { return d * (kPi / 180.0); }
inline double degrees(double r) { return r * (180.0 / kPi); }

// (the names by value: taken by reference, GCC's -Wdangling-reference supposes the result may refer to them)
const Json& member(const Json& j, std::string_view key, std::string_view where) {
    const Json* v = j.find(std::string(key));
    if (!v) throw Error("geometry: " + std::string(where) + " holds no \"" + std::string(key) + "\"");
    return *v;
}
double number_of(const Json& v, const std::string& what) {
    if (!v.is(Json::Number)) throw Error("geometry: " + what + " is not a number");
    return v.number;
}
double number(const Json& j, const std::string& key, const std::string& where) {
    return number_of(member(j, key, where), "\"" + key + "\" of " + where);
}
std::vector<double> numbers(const Json& v, const std::string& what) {
    if (!v.is(Json::Array)) throw Error("geometry: " + what + " is not a list");
    std::vector<double> out(v.array.size());
    for (std::size_t k = 0; k < out.size(); ++k) out[k] = number_of(v.array[k], "an entry of " + what);
    return out;
}
std::vector<double> numbers(const Json& j, const std::string& key, const std::string& where) {
    return numbers(member(j, key, where), "\"" + key + "\" of " + where);
}
// The travel per frame [rows] as a line through the 33 frames around each (the registration's blocks are 32
// frames; the encoder's own frame-to-frame differences are noise against a stage that moves steadily). The ends
// are continued with the first and the last step.
void travel_per_frame(const double* h, std::size_t n, double* out) {
    if (n < 2) throw Error("geometry: a stretch of fewer than two frames");
    const int half = 16;
    const double norm = 2992.0;  // sum of k^2 over -16 .. 16
    std::vector<double> pad(n + 2 * half);
    for (int k = 0; k < half; ++k) pad[static_cast<std::size_t>(k)] = h[0] + (h[1] - h[0]) * (k - half);
    std::copy(h, h + n, pad.begin() + half);
    for (int k = 1; k <= half; ++k) pad[n + half + static_cast<std::size_t>(k) - 1] = h[n - 1] + (h[n - 1] - h[n - 2]) * k;
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0;
        for (int j = 0; j <= 2 * half; ++j) s += pad[i + static_cast<std::size_t>(j)] * ((j - half) / norm);
        out[i] = s;
    }
}

// central differences, one-sided at the ends
void gradient(const double* a, std::size_t n, double* out) {
    if (n < 2) throw Error("geometry: a stretch of fewer than two frames");
    out[0] = a[1] - a[0];
    out[n - 1] = a[n - 1] - a[n - 2];
    for (std::size_t i = 1; i + 1 < n; ++i) out[i] = (a[i + 1] - a[i - 1]) / 2.0;
}

template <class T>
std::vector<float> to_float(const std::vector<T>& v) {
    return std::vector<float>(v.begin(), v.end());
}

double period_of(const Json& cal) {
    const Json& rt = member(cal, "row_time", "the geometry");
    const Json* p = rt.find("frame_period_s");
    if (p && p->is(Json::Number) && p->number != 0) return p->number;
    const Json* m = rt.find("measured");
    if (m && m->is(Json::Object)) {
        p = m->find("period_s");
        if (p && p->is(Json::Number) && p->number != 0) return p->number;
    }
    throw Error("the geometry does not say the frame period");
}

Map map_of(const Json& cal) {
    const Json& m = member(cal, "map", "the geometry");
    Map out;
    for (double v : numbers(m, "i", "the map")) out.i.push_back(static_cast<int>(v));
    for (double v : numbers(m, "j", "the map")) out.j.push_back(static_cast<int>(v));
    out.eu = numbers(m, "eu", "the map");
    out.ev = numbers(m, "ev", "the map");
    out.c0 = number(m, "c0", "the map"), out.cs = number(m, "cs", "the map");
    out.v0 = number(m, "v0", "the map"), out.vs = number(m, "vs", "the map");
    if (out.j.size() != out.i.size() || out.eu.size() != out.i.size() || out.ev.size() != out.i.size())
        throw Error("geometry: the map's lists differ in length");
    return out;
}

// whether the calibration is the joint model's (vc_tomography/calibration/geometry_calibration.hpp)
bool is_joint(const Json& cal) {
    const Json* m = cal.find("model");
    return m && m->is(Json::String) && m->string == "joint";
}

// a projection stretch of one scan of a joint calibration, as its layout lists it
struct JointStretch {
    std::string name;
    bool helix = false;
    std::size_t frames = 0, offset = 0;  // its length, and its first frame among the scan's projection frames
};
std::vector<JointStretch> joint_stretches(const Json& cal, std::size_t scan) {
    const Json& scans = member(member(cal, "layout", "the geometry"), "scans", "the layout");
    if (!scans.is(Json::Array) || scan >= scans.array.size()) throw Error("geometry: the layout does not hold scan " + std::to_string(scan));
    const Json& list = member(scans.array[scan], "stretches", "a scan of the layout");
    if (!list.is(Json::Array)) throw Error("geometry: a scan's stretches are not a list");
    std::vector<JointStretch> out;
    for (const Json& s : list.array) {
        JointStretch st;
        const Json& name = member(s, "name", "a stretch");
        st.name = name.is(Json::String) ? name.string : std::string();
        const Json* helix = s.find("helix");
        st.helix = helix && helix->is(Json::Bool) && helix->boolean;
        st.frames = static_cast<std::size_t>(number(s, "frames", "a stretch"));
        st.offset = static_cast<std::size_t>(number(s, "offset", "a stretch"));
        out.push_back(st);
    }
    return out;
}
// The stretch a scan is placed by: its helix; of a scan without one (a circular scan: turns at rest alone) its
// first stretch. main_stretch: which of the scan's stretches that is.
std::size_t main_stretch(const std::vector<JointStretch>& stretches) {
    for (std::size_t k = 0; k < stretches.size(); ++k)
        if (stretches[k].helix) return k;
    if (stretches.empty()) throw Error("geometry: a scan of the calibration has no stretch");
    return 0;
}
std::array<std::size_t, 2> joint_helix(const Json& cal, std::size_t scan) {
    const std::vector<JointStretch> stretches = joint_stretches(cal, scan);
    const JointStretch& st = stretches[main_stretch(stretches)];
    return {st.offset, st.offset + st.frames};
}
// the frames of a stretch (as a joint calibration's layout lists it) that were on disk, among the scan's projection
// frames: [first, end); a stretch that does not say is whole
std::array<std::size_t, 2> stretch_on_disk(const Json& s) {
    const std::size_t offset = static_cast<std::size_t>(number(s, "offset", "a stretch"));
    const Json* first = s.find("first");
    const Json* end = s.find("end");
    return {offset + (first && first->is(Json::Number) ? static_cast<std::size_t>(first->number) : 0),
            offset + static_cast<std::size_t>(end && end->is(Json::Number) ? end->number : number(s, "frames", "a stretch"))};
}
double two_numbers(const Json& j, const std::string& key, const std::string& where, std::size_t i) {
    const std::vector<double> v = numbers(j, key, where);
    if (v.size() < 2) throw Error("geometry: \"" + key + "\" of " + where + " is not two numbers");
    return v[i];
}

// a course given at knots `knots` (equally spaced), a line between them and beyond the ends
struct Knots {
    std::vector<double> at;
    double value(const std::vector<double>& v, double frame) const {
        const double x = (frame - at[0]) / (at[1] - at[0]);
        const long last = static_cast<long>(at.size()) - 2;
        const long i = std::clamp(static_cast<long>(std::floor(x)), 0L, last);
        const double w = x - static_cast<double>(i);
        return v[static_cast<std::size_t>(i)] * (1.0 - w) + v[static_cast<std::size_t>(i) + 1] * w;
    }
};

}  // namespace

// ---------------------------------------------------------------- the map

Map parse_map(const std::string& json_object_text) {
    const Json doc = Json::dict().set("map", Json::parse(json_object_text));
    return map_of(doc);
}

double overlap_centre(int cols, double move) {
    const double C = cols;
    return 0.5 * (std::max(0.0, -move) + std::min(C - 1, C - 1 - move));
}

std::vector<std::array<int, 2>> map_terms(int order) {
    std::vector<std::array<int, 2>> t;
    for (int i = 0; i <= order; ++i)
        for (int j = 0; j <= order; ++j)
            if (i + j >= 2 && i + j <= order) t.push_back({i, j});
    t.push_back({1, 0});
    return t;
}

Map empty_map(int rows, int cols, int order) {
    Map m;
    for (const auto& t : map_terms(order)) m.i.push_back(t[0]), m.j.push_back(t[1]);
    m.eu.assign(m.i.size(), 0.0);
    m.ev.assign(m.i.size(), 0.0);
    m.c0 = cols / 2.0, m.cs = cols / 2.0, m.v0 = (rows - 1) / 2.0, m.vs = rows / 2.0;
    return m;
}

void map_displacement(const Map& m, const double* c, const double* v, std::size_t n, double* eu, double* ev) {
    for (std::size_t q = 0; q < n; ++q) m.displacement(c[q], v[q], eu[q], ev[q]);
}

// ---------------------------------------------------------------- a two-scan acquisition

Calibration Calibration::parse(const std::string& json_text) {
    Calibration c;
    c.doc_ = std::make_shared<const Json>(Json::parse(json_text));
    if (!c.doc_->is(Json::Object)) throw Error("geometry: the calibration is not an object");
    if (!is_joint(*c.doc_))
        throw Error("geometry: the calibration is not a joint one (\"model\": \"joint\"); files of the earlier two-scan step are no longer read: "
                    "make the calibration again with vc_tomography_calibration geometry");
    return c;
}
Calibration Calibration::load(const std::string& path) { return parse(io::read_text(path)); }

int Calibration::rows() const { return static_cast<int>(number(member(*doc_, "layout", "the geometry"), "rows", "the layout")); }
int Calibration::cols() const { return static_cast<int>(number(member(*doc_, "layout", "the geometry"), "cols", "the layout")); }
double Calibration::row_sign() const { return number(member(*doc_, "layout", "the geometry"), "row_sign", "the layout"); }
Map Calibration::map() const { return map_of(*doc_); }
std::vector<std::string> Calibration::scans() const {
    const Json& v = member(*doc_, "scans", "the geometry");
    if (!v.is(Json::Array)) throw Error("geometry: \"scans\" is not a list");
    std::vector<std::string> out;
    for (const Json& s : v.array) out.push_back(s.is(Json::String) ? s.string : std::string());
    return out;
}
std::vector<double> Calibration::heights_rows(const std::vector<double>& z_translation_m, double pixel_size_m) const {
    const double sign = row_sign();
    std::vector<double> h(z_translation_m.size());
    for (std::size_t i = 0; i < h.size(); ++i) h[i] = sign * z_translation_m[i] / pixel_size_m;
    return h;
}
double Calibration::rolling_shutter() const {
    return number(member(*doc_, "row_time", "the geometry"), "seconds_per_row", "the rows' timing") * (rows() - 1) / period_of(*doc_);
}
double Calibration::fan() const {
    const Json& lay = member(*doc_, "layout", "the geometry");
    const Json* v = lay.find("fan");
    if (!v || !v->is(Json::Number) || v->number == 0.0) return 0.0;
    return v->number * (2 * kPi / number(lay, "frames_per_turn", "the layout"));
}
bool Calibration::rows_from_middle() const {
    const Json* v = member(*doc_, "row_time", "the geometry").find("from_middle");
    return v && v->is(Json::Bool) && v->boolean;
}
double Calibration::overlap_centre() const {
    const Json& lay = member(*doc_, "layout", "the geometry");
    const Json& scans = member(lay, "scans", "the layout");
    if (!scans.is(Json::Array) || scans.array.size() < 2) throw Error("geometry: the calibration is not of two scans");
    const double move = std::nearbyint(number(scans.array[1], "coarse_axis", "the second scan") - number(scans.array[0], "coarse_axis", "the first scan"));
    return geometry::overlap_centre(cols(), move);
}
std::array<std::size_t, 2> Calibration::helix_frames() const {
    return joint_helix(*doc_, 0);
}

namespace {

// One scan's poses under a joint calibration (vc_tomography/geometry.hpp states what the poses are), frame by frame of
// the stretches the calibration holds of it; `placed`: per frame whether it is of one (the others stay at zero).
// own: in the scan's own frame, its place d and height left out. The rows are the model's (not counted from any
// origin).
Poses joint_scan_poses(const Json& cal, int rows, std::size_t s, const std::vector<double>& ang, const std::vector<double>& H, bool own, std::vector<std::uint8_t>& placed) {
    const Json& par = member(cal, "parameters", "the geometry");
    const Json& scans = member(par, "scans", "the parameters");
    if (!scans.is(Json::Array) || s >= scans.array.size()) throw Error("geometry: the calibration does not hold scan " + std::to_string(s));
    const double mid = (rows - 1) / 2.0;
    const double per_row = number(member(cal, "row_time", "the geometry"), "frames_per_row", "the rows' timing");
    // the rows counted from the middle row to both ends (a camera whose two halves are read at once)
    const Json* two_sided = member(cal, "row_time", "the geometry").find("from_middle");
    const bool from_middle = two_sided && two_sided->is(Json::Bool) && two_sided->boolean;
    const double roll = number(par, "roll", "the parameters");
    // the pitch of the rotation axis towards the beam, where the calibration has it (a mosaic's tiles give it)
    const Json* leans = par.find("pitch");
    const double pitch = leans && leans->is(Json::Number) ? leans->number : 0.0;
    const Json& stage = member(par, "stage", "the parameters");
    const double lift[2] = {two_numbers(stage, "height_once", "the stage", 0), two_numbers(stage, "height_once", "the stage", 1)};
    const double sway[2] = {two_numbers(stage, "axis_twice", "the stage", 0), two_numbers(stage, "axis_twice", "the stage", 1)};
    const Json& p = scans.array[s];
    const double axis = number(p, "axis", "a scan"), axis_slope = number(p, "axis_slope", "a scan"), axial = 1.0 + number(p, "axial_scale", "a scan");
    // the scale of the scan's image across the beam, where the calibration has one: what the scan records u columns
    // from its rotation axis lies u (1 + column_scale) from it
    const Json* widens = p.find("column_scale");
    const double column_scale = widens && widens->is(Json::Number) ? widens->number : 0.0;
    const double at[2] = {two_numbers(p, "d", "a scan", 0), two_numbers(p, "d", "a scan", 1)};  // where the sample sits on the rotation stage
    const double d[2] = {own ? 0.0 : at[0], own ? 0.0 : at[1]};
    const double height = own ? 0.0 : number(p, "height", "a scan"), H_mid = number(p, "mid_height", "a scan");
    // The sample as this scan holds it is tilted against the anchor's (the scan's rotation: x, y), about the
    // sample's origin at the height about_height. With along = tilt . (-sin th, cos th), the tilt's part about the
    // beam's direction at this frame, the scan records the sample's ray (s, z) at (s + along (z - about_height),
    // z - along s) (model::turned_back, to the first order): the frame is turned by `along` about the beam, which
    // the two slopes and the axis hold exactly. The tilt's part about the direction across the beam moves the rows by
    // the depth along the ray, which poses cannot hold: it is left out, and so is the spin. Like the place, it is a
    // rigid motion of the scan's volume, so the scan's own frame has none of it.
    double tilt[2] = {0, 0}, about = 0;
    if (!own) {
        if (const Json* turn = p.find("rotation"))
            if (turn->is(Json::Array) && turn->array.size() >= 2) tilt[0] = number_of(turn->array[0], "a scan's rotation"), tilt[1] = number_of(turn->array[1], "a scan's rotation");
        const Json* from = par.find("about_height");
        about = from && from->is(Json::Number) ? from->number : 0.0;
    }
    const Json& heights = member(p, "stretch_height", "a scan");
    const std::size_t n = ang.size();
    if (H.size() != n) throw Error("geometry: the angles and the heights of a scan differ in length");
    std::vector<double> extra(n, 0.0), step(n, 0.0), travel(n, 0.0);  // per frame: its stretch's height; the angle's step; the travel
    placed.assign(n, 0);
    for (const JointStretch& st : joint_stretches(cal, s)) {
        if (st.offset + st.frames > n) throw Error("the record holds " + std::to_string(n) + " frames of a scan, the geometry more: it was made from another scan");
        const Json* given = heights.find(st.name);  // (a stretch without one stands at the scan's height)
        const double h = given ? number_of(*given, "a stretch's height") : 0.0;
        for (std::size_t k = st.offset; k < st.offset + st.frames; ++k) extra[k] = h, placed[k] = 1;
        if (st.frames < 2) continue;
        gradient(ang.data() + st.offset, st.frames, step.data() + st.offset);
        travel_per_frame(H.data() + st.offset, st.frames, travel.data() + st.offset);
    }
    Poses q;
    for (auto* v : {&q.row, &q.row_per_col, &q.axis, &q.col_per_row, &q.col_scale, &q.row_scale, &q.angle_deg, &q.angle_deg_per_row}) v->assign(n, 0.0);
    for (std::size_t k = 0; k < n; ++k) {
        if (!placed[k]) continue;
        const double th = radians(ang[k]), c = std::cos(th), sn = std::sin(th);
        const double dn = d[0] * c + d[1] * sn;
        double while_read = axial * travel[k] * per_row;  // rows the sample travels per row read
        if (from_middle) {
            // (a travel of under a millionth of a row over the frame is the rounding of a height that stands)
            if (std::fabs(while_read) * rows > 1e-6)
                throw Error("geometry: the rows' timing counts the rows from the middle row to both ends and the scan travels along the axis while a frame is read: the poses hold that for scans at rest only");
            while_read = 0.0;
        }
        // About a pitched axis a ray climbs by the pitch per pixel of depth, and the sample's origin lies
        // at . (-sin th, cos th) deep from the axis: the row at which this frame holds the rays through the origin's
        // depth. It is not part of a rigid motion of the scan's volume, so the scan's own frame has it too: with it
        // every scan of a mosaic is reconstructed about the sample's origin, and what the pitch leaves in a volume
        // (the reconstruction takes no ray's inclination) is the same in all of them where they overlap.
        const double leaning = pitch * (at[1] * c - at[0] * sn);
        q.row[k] = height + extra[k] + axial * H[k] + lift[0] * c + lift[1] * sn - roll * dn - while_read * mid + leaning;
        q.row_scale[k] = -while_read;
        q.axis[k] = axis + axis_slope * (H[k] - H_mid) + sway[0] * (c * c - sn * sn) + sway[1] * (2.0 * sn * c) + dn;
        const double along = tilt[1] * c - tilt[0] * sn;
        q.row_per_col[k] = -roll - along, q.col_per_row[k] = roll + along;
        q.axis[k] += along * (mid - q.row[k] - about);  // (the ray at the middle row is of the height mid - row)
        q.angle_deg[k] = ang[k];
        q.angle_deg_per_row[k] = step[k] * per_row;
        q.col_scale[k] = column_scale;
    }
    // about the rotation axis' column (its course over the frames and the map's part there are under a pixel, which
    // times the scale is nothing)
    q.scale_col = axis;
    q.rows_from_middle = from_middle;
    return q;
}

// Calibration::scan_poses: every scan of a joint calibration. whole: it is an error if the calibration does not
// hold every frame given (a stretch that was not on disk).
std::vector<Poses> joint_poses(const Json& cal, int rows, const std::vector<std::vector<double>>& ang, const std::vector<std::vector<double>>& H, bool whole, std::string* not_applied) {
    const Json& scans = member(member(cal, "parameters", "the geometry"), "scans", "the parameters");
    const Json& placed = member(member(cal, "layout", "the geometry"), "scans", "the layout");
    if (!scans.is(Json::Array) || !placed.is(Json::Array) || scans.array.size() != placed.array.size() || scans.array.empty())
        throw Error("geometry: the calibration's parameters and layout do not hold the same scans");
    const std::size_t S = scans.array.size();
    if (ang.size() != S || H.size() != S)
        throw Error("geometry: the calibration is of " + std::to_string(S) + " scans, and the angles and heights of " + std::to_string(ang.size()) + " are given");
    const std::array<std::size_t, 2> first_helix = joint_helix(cal, 0);
    if (H[0].size() <= first_helix[0]) throw Error("geometry: the record holds fewer frames than the calibration's first scan");
    std::vector<Poses> out(S);
    // the slices are numbered from the first scan's row at its helix' first frame (without a helix: its first
    // stretch's): its height, that stretch's and the record's
    double origin = 0;
    {
        const Json& p = scans.array[0];
        double extra = 0;
        const std::vector<JointStretch> stretches = joint_stretches(cal, 0);
        if (const Json* given = member(p, "stretch_height", "a scan").find(stretches[main_stretch(stretches)].name)) extra = number_of(*given, "a stretch's height");
        origin = number(p, "height", "a scan") + extra + (1.0 + number(p, "axial_scale", "a scan")) * H[0][first_helix[0]];
    }
    for (std::size_t s = 0; s < S; ++s) {
        std::vector<std::uint8_t> seen;
        out[s] = joint_scan_poses(cal, rows, s, ang[s], H[s], false, seen);
        if (whole && std::find(seen.begin(), seen.end(), 0) != seen.end())
            throw Error("the geometry was made without some of the scan's projection stretches (they were not on disk): it places no frame of them");
        for (double& v : out[s].row) v -= origin;
    }
    if (not_applied) {
        Json na = Json::dict();
        if (const Json* solution = cal.find("solution"))
            if (const Json* left = solution->find("left")) na.set("what the model leaves of the comparisons, per kind and scans [rms and median of frames, rows, columns]; the frames are not fitted", *left);
        // the scans' rotations: the poses hold the tilt's part about the beam's direction and nothing else of them
        Json turns = Json::list();
        bool turned = false;
        for (const Json& p : scans.array) {
            const Json* turn = p.find("rotation");
            if (!turn || !turn->is(Json::Array)) {
                turns.add(Json());
                continue;
            }
            for (const Json& v : turn->array) turned = turned || (v.is(Json::Number) && v.number != 0.0);
            turns.add(*turn);
        }
        if (turned)
            na.set("the scans' rotations against the first (x, y, z) [rad]: the tilt's part about the direction across the beam (rows by the depth along the ray) and the spin", turns);
        *not_applied = na.text();
    }
    return out;
}

}  // namespace

std::size_t Calibration::scan_named(const std::string& name) const {
    const Json& placed = member(member(*doc_, "layout", "the geometry"), "scans", "the layout");
    if (!placed.is(Json::Array)) throw Error("geometry: the layout's scans are not a list");
    std::size_t found = placed.array.size();
    for (std::size_t k = 0; k < placed.array.size(); ++k) {
        const Json* n = placed.array[k].find("name");
        if (!n || !n->is(Json::String) || n->string != name) continue;
        if (found != placed.array.size()) throw Error("geometry: the calibration holds more than one scan named '" + name + "'");
        found = k;
    }
    if (found == placed.array.size()) throw Error("geometry: the calibration holds no scan named '" + name + "'");
    return found;
}

std::array<std::size_t, 2> Calibration::frames_on_disk(std::size_t scan) const {
    const Json& scans = member(member(*doc_, "layout", "the geometry"), "scans", "the layout");
    if (!scans.is(Json::Array) || scan >= scans.array.size()) throw Error("geometry: the layout does not hold scan " + std::to_string(scan));
    const Json& list = member(scans.array[scan], "stretches", "a scan of the layout");
    if (!list.is(Json::Array) || list.array.empty()) throw Error("geometry: scan " + std::to_string(scan) + " of the calibration has no stretch");
    return stretch_on_disk(list.array[main_stretch(joint_stretches(*doc_, scan))]);
}

double Calibration::axis_column(std::size_t scan) const {
    const Json& scans = member(member(*doc_, "parameters", "the geometry"), "scans", "the parameters");
    if (!scans.is(Json::Array) || scan >= scans.array.size()) throw Error("geometry: the calibration does not hold scan " + std::to_string(scan));
    return number(scans.array[scan], "axis", "a scan");
}

Poses Calibration::own_poses(std::size_t scan, const std::vector<double>& angles_deg, const std::vector<double>& heights_rows) const {
    std::vector<std::uint8_t> placed;
    return joint_scan_poses(*doc_, rows(), scan, angles_deg, heights_rows, true, placed);
}

std::size_t Calibration::scan_count() const {
    const Json& placed = member(member(*doc_, "layout", "the geometry"), "scans", "the layout");
    if (!placed.is(Json::Array)) throw Error("geometry: the layout's scans are not a list");
    return placed.array.size();
}

std::vector<Poses> Calibration::scan_poses(const std::vector<std::vector<double>>& ang, const std::vector<std::vector<double>>& H, std::string* not_applied) const {
    return joint_poses(*doc_, rows(), ang, H, false, not_applied);
}

std::vector<std::array<std::size_t, 2>> Calibration::frames_placed(std::size_t scan) const {
    const Json& scans = member(member(*doc_, "layout", "the geometry"), "scans", "the layout");
    if (!scans.is(Json::Array) || scan >= scans.array.size()) throw Error("geometry: the layout does not hold scan " + std::to_string(scan));
    std::vector<std::array<std::size_t, 2>> runs;
    for (const Json& s : member(scans.array[scan], "stretches", "a scan of the layout").array) {
        const std::array<std::size_t, 2> run = stretch_on_disk(s);
        if (run[1] <= run[0]) continue;
        if (!runs.empty() && runs.back()[1] == run[0]) runs.back()[1] = run[1];
        else runs.push_back(run);
    }
    return runs;
}

MergedLayout Calibration::merged_layout(const std::vector<Poses>& poses, const std::vector<std::vector<double>>& H, const std::vector<std::size_t>& first_frame, double quantum) const {
    const Json& lay = member(*doc_, "layout", "the geometry");
    // the frames of the first scan's helix that were on disk
    const std::array<std::size_t, 2> helix = frames_on_disk(0);
    const std::size_t a = helix[0], b = helix[1], S = poses.size();
    const double sign = number(lay, "row_sign", "the layout");
    if (!S || H.size() != S || (!first_frame.empty() && first_frame.size() != S)) throw Error("geometry: the poses, the heights and the first frames are not of the same scans");
    // Scans without a helix (circular scans) do not travel, and their frames are not one another's (a scan may be
    // a frame short): each is placed by the medians over its own stretch's frames on disk.
    const std::vector<JointStretch> stretches = joint_stretches(*doc_, 0);
    const bool travels = stretches[main_stretch(stretches)].helix;
    const auto frames_of = [&](std::size_t s) { return travels || s == 0 ? helix : frames_on_disk(s); };
    for (std::size_t s = 0; s < S; ++s)
        if (poses[s].axis.size() < frames_of(s)[1] || H[s].size() < frames_of(s)[1]) throw Error("geometry: the poses do not reach the helix' end");
    const auto from = [&](std::size_t s) { return first_frame.empty() ? std::size_t{0} : first_frame[s]; };
    const auto q = [&](double v) { return std::nearbyint(v / quantum) * quantum; };  // halves to the even one, as the step always rounded
    const auto axis_of = [&](std::size_t s) {
        const std::array<std::size_t, 2> f = frames_of(s);
        return calibration::median_of(std::vector<double>(poses[s].axis.begin() + static_cast<std::ptrdiff_t>(f[0]), poses[s].axis.begin() + static_cast<std::ptrdiff_t>(f[1])));
    };
    // the row a slice is read at beyond what the record alone gives (z_pix: the record's height from the scan's first frame)
    const auto row_of = [&](std::size_t s, std::size_t k) { return poses[s].row[k] + sign * (H[s][k] - H[s][from(s)]); };
    MergedLayout out;
    const double axis0 = axis_of(0);
    out.cor_main_detector = q(axis0);
    out.axis_first_scan = axis0, out.quantum = quantum;
    for (std::size_t s = 1; s < S; ++s) {
        const double axis = axis_of(s);
        double rows_ = 0;
        if (travels) {  // at equal frames: the two scans travel alike
            std::vector<double> diff(b - a);
            for (std::size_t k = a; k < b; ++k) diff[k - a] = row_of(s, k) - row_of(0, k);
            rows_ = calibration::median_of(diff);
        } else {
            const auto rows_over = [&](std::size_t scan) {
                const std::array<std::size_t, 2> f = frames_of(scan);
                std::vector<double> v(f[1] - f[0]);
                for (std::size_t k = f[0]; k < f[1]; ++k) v[k - f[0]] = row_of(scan, k);
                return calibration::median_of(v);
            };
            rows_ = rows_over(s) - rows_over(0);
        }
        out.vectors.push_back({q(-rows_), q(axis0 - axis)});
        out.measured.push_back({axis, rows_});
    }
    if (S > 1) out.vector = out.vectors[0], out.axis_second_scan = out.measured[0][0], out.second_scan_rows = out.measured[0][1];
    return out;
}

Terms reconstruction_terms(const Poses& pose, const std::vector<double>& H, double row_sign, int rows, double rot_center, double column_offset,
                           double z_readjustment, double row_origin) {
    const std::size_t n = pose.row.size();
    if (H.size() != n) throw Error("geometry: the heights and the poses differ in length");
    Terms t;
    t.angles_deg = pose.angle_deg;
    t.row_offset.resize(n), t.col_offset.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
        const double z_pix = row_sign * (H[k] - H[0]);
        t.row_offset[k] = static_cast<float>(-(pose.row[k] + row_origin) - z_readjustment - z_pix);
        t.col_offset[k] = static_cast<float>(pose.axis[k] + column_offset - rot_center);
    }
    t.row_slope = to_float(pose.row_per_col), t.col_slope = to_float(pose.col_per_row);
    t.col_scale = to_float(pose.col_scale), t.row_scale = to_float(pose.row_scale);
    t.slope_col = rot_center - column_offset;
    t.slope_row = (rows - 1) / 2.0;
    t.scale_col = pose.scale_col;
    return t;
}

// ---------------------------------------------------------------- the tiles of a mosaic

TileGeometry TileGeometry::parse(const std::string& json_text) {
    TileGeometry g;
    g.doc_ = std::make_shared<const Json>(Json::parse(json_text));
    if (!g.doc_->is(Json::Object)) throw Error("geometry: the tiles' description is not an object");
    return g;
}
TileGeometry TileGeometry::load(const std::string& path) { return parse(io::read_text(path)); }

TileGeometry::Pose TileGeometry::frame_pose(const std::string& tile, const std::vector<double>& angle_deg, const std::vector<double>& frame, double row) const {
    const Json& cal = *doc_;
    const Json& t = member(member(cal, "tiles", "the description"), tile, "the tiles");
    Knots K{numbers(member(cal, "layout", "the description"), "knot_frames", "the layout")};
    if (K.at.size() < 2) throw Error("geometry: fewer than two knots");
    if (angle_deg.size() != frame.size()) throw Error("geometry: the angles and the frames differ in length");
    const double v0 = number(member(cal, "map", "the description"), "v0", "the map");
    const double rho = number(member(cal, "row_time", "the description"), "rad_per_row", "the rows' timing");
    const auto T = numbers(cal, "angle_course_rad", "the description");
    const auto C = numbers(t, "column_course", "the tile"), Rc = numbers(t, "height_course", "the tile");
    if (T.size() != K.at.size() || C.size() != K.at.size() || Rc.size() != K.at.size()) throw Error("geometry: a course is not given at the knots");
    const Json* yj = t.find("yaw_rad");
    const double yaw = yj && yj->is(Json::Number) ? yj->number : 0.0;
    const auto place = numbers(t, "place_px", "the tile");
    if (place.size() < 2) throw Error("geometry: a tile's place is not two numbers");
    const double a = number(t, "axis_column", "the tile"), h = number(t, "height_rows", "the tile");
    const double at_row = (row < 0 ? v0 : row) - v0;
    Pose out;
    const std::size_t n = frame.size();
    out.angle_deg.resize(n), out.column.resize(n), out.height_rows.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
        const double th = radians(angle_deg[k]) + K.value(T, frame[k]) + rho * at_row + yaw;
        out.angle_deg[k] = degrees(th);
        out.column[k] = a + place[0] * std::cos(th) + place[1] * std::sin(th) + K.value(C, frame[k]);
        out.height_rows[k] = h + K.value(Rc, frame[k]);
    }
    return out;
}

Terms TileGeometry::reconstruction_terms(const std::string& tile, const std::vector<double>& angle_deg, const std::vector<double>& Hrec,
                                         const std::vector<double>& frame, double rot_center, double row_origin) const {
    const Json& cal = *doc_;
    const Json& lay = member(cal, "layout", "the description");
    const Json& t = member(member(cal, "tiles", "the description"), tile, "the tiles");
    const std::size_t n = frame.size();
    if (Hrec.size() != n) throw Error("geometry: the heights and the frames differ in length");
    const Pose p = frame_pose(tile, angle_deg, frame);  // at the middle row
    const auto place = numbers(t, "place_px", "the tile");
    const double scale = number(cal, "axial_scale", "the description"), sign = number(lay, "row_sign", "the layout");
    const double per_row = number(member(cal, "row_time", "the description"), "rad_per_row", "the rows' timing") / radians(std::fabs(number(lay, "step_deg", "the layout")));  // frames per row
    const double chi = number(cal, "rows_slope_opposite", "the description"), psi = number(cal, "axis_roll_rad", "the description");
    std::vector<double> row(n), travel(n);
    for (std::size_t k = 0; k < n; ++k) row[k] = (1.0 + scale) * Hrec[k] + p.height_rows[k];  // content at height w is on the row w + row
    travel_per_frame(row.data(), n, travel.data());
    Terms out;
    out.angles_deg = p.angle_deg;
    out.row_offset.resize(n), out.col_offset.resize(n), out.row_scale.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
        const double th = radians(p.angle_deg[k]);
        const double axis = p.column[k] - (place[0] * std::cos(th) + place[1] * std::sin(th));
        const double z_pix = sign * (Hrec[k] - Hrec[0]);
        out.row_offset[k] = static_cast<float>(-(row[k] + row_origin) - z_pix);
        out.col_offset[k] = static_cast<float>(axis - rot_center);
        out.row_scale[k] = static_cast<float>(-travel[k] * per_row);
    }
    out.row_slope.assign(n, static_cast<float>(-chi));
    out.col_slope.assign(n, static_cast<float>(psi));
    out.col_scale.assign(n, 0.0f);
    out.slope_col = rot_center;
    out.slope_row = (number(lay, "rows", "the layout") - 1) / 2.0;
    return out;
}

}  // namespace vc_tomography::geometry
