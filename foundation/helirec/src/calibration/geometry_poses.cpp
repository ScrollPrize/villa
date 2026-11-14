// helirec/geometry.hpp: the detector's map, the poses of every frame from a geometry calibration, and the
// reconstruction's per-projection terms.
#include "helirec/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include "calibration/util.hpp"
#include "io/json.hpp"

namespace helirec::geometry {

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
// a list of lists of numbers, each at least `width` long
std::vector<std::vector<double>> table(const Json& v, std::size_t width, const std::string& what) {
    if (!v.is(Json::Array)) throw Error("geometry: " + what + " is not a list");
    std::vector<std::vector<double>> out;
    out.reserve(v.array.size());
    for (const Json& r : v.array) {
        out.push_back(numbers(r, "a row of " + what));
        if (out.back().size() < width) throw Error("geometry: a row of " + what + " is too short");
    }
    return out;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw Error("geometry: cannot read " + path);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
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

double median(std::vector<double> v) {
    if (v.empty()) throw Error("geometry: the median of nothing");
    const std::size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m), v.end());
    if (v.size() % 2) return v[m];
    const double hi = v[m];
    const double lo = *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m));
    return (lo + hi) / 2.0;
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

double overlap_centre_of(const Json& lay) {
    return overlap_centre(static_cast<int>(number(lay, "cols", "the layout")), number(lay, "coarse_move", "the layout"));
}

// whether the calibration is the joint model's (helirec/calibration/geometry_calibration.hpp)
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
std::array<std::size_t, 2> joint_helix(const Json& cal, std::size_t scan) {
    for (const JointStretch& st : joint_stretches(cal, scan))
        if (st.helix) return {st.offset, st.offset + st.frames};
    throw Error("geometry: scan " + std::to_string(scan) + " of the calibration has no helix");
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

void map_basis(const Map& m, double c, double v, double* phi) {
    const double x = (c - m.c0) / m.cs, y = (v - m.v0) / m.vs;
    for (std::size_t k = 0; k < m.i.size(); ++k) phi[k] = std::pow(x, m.i[k]) * std::pow(y, m.j[k]);
}

void map_displacement(const Map& m, const double* c, const double* v, std::size_t n, double* eu, double* ev) {
    std::vector<double> phi(m.i.size());
    for (std::size_t q = 0; q < n; ++q) {
        map_basis(m, c[q], v[q], phi.data());
        double su = 0, sv = 0;
        for (std::size_t k = 0; k < m.i.size(); ++k) {
            su += m.eu[k] * phi[k];
            sv += m.ev[k] * phi[k];
        }
        eu[q] = su, ev[q] = sv;
    }
}

// ---------------------------------------------------------------- a two-scan acquisition

Calibration Calibration::parse(const std::string& json_text) {
    Calibration c;
    c.doc_ = std::make_shared<const Json>(Json::parse(json_text));
    if (!c.doc_->is(Json::Object)) throw Error("geometry: the calibration is not an object");
    return c;
}
Calibration Calibration::load(const std::string& path) { return parse(read_file(path)); }

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
double Calibration::overlap_centre() const {
    const Json& lay = member(*doc_, "layout", "the geometry");
    if (!is_joint(*doc_)) return overlap_centre_of(lay);
    const Json& scans = member(lay, "scans", "the layout");
    if (!scans.is(Json::Array) || scans.array.size() < 2) throw Error("geometry: the calibration is not of two scans");
    const double move = std::nearbyint(number(scans.array[1], "coarse_axis", "the second scan") - number(scans.array[0], "coarse_axis", "the first scan"));
    return geometry::overlap_centre(cols(), move);
}
std::array<std::size_t, 2> Calibration::helix_frames() const {
    if (is_joint(*doc_)) return joint_helix(*doc_, 0);
    const Json& frames = member(member(*doc_, "layout", "the geometry"), "frames", "the layout");
    if (!frames.is(Json::Array) || frames.array.empty()) throw Error("geometry: the layout's frames are not a list");
    const std::size_t a = static_cast<std::size_t>(number(frames.array[0], "rest1", "the layout's frames"));
    return {a, a + static_cast<std::size_t>(number(frames.array[0], "helix", "the layout's frames"))};
}

namespace {

// One scan's poses under a joint calibration (helirec/geometry.hpp states what the poses are), frame by frame of
// the stretches the calibration holds of it; `placed`: per frame whether it is of one (the others stay at zero).
// own: in the scan's own frame, its place d and height left out. The rows are the model's (not counted from any
// origin).
Poses joint_scan_poses(const Json& cal, int rows, std::size_t s, const std::vector<double>& ang, const std::vector<double>& H, bool own, std::vector<std::uint8_t>& placed) {
    const Json& par = member(cal, "parameters", "the geometry");
    const Json& scans = member(par, "scans", "the parameters");
    if (!scans.is(Json::Array) || s >= scans.array.size()) throw Error("geometry: the calibration does not hold scan " + std::to_string(s));
    const double mid = (rows - 1) / 2.0;
    const double per_row = number(member(cal, "row_time", "the geometry"), "frames_per_row", "the rows' timing");
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
        const double while_read = axial * travel[k] * per_row;  // rows the sample travels per row read
        // About a pitched axis a ray climbs by the pitch per pixel of depth, and the sample's origin lies
        // at . (-sin th, cos th) deep from the axis: the row at which this frame holds the rays through the origin's
        // depth. It is not part of a rigid motion of the scan's volume, so the scan's own frame has it too: with it
        // every scan of a mosaic is reconstructed about the sample's origin, and what the pitch leaves in a volume
        // (the reconstruction takes no ray's inclination) is the same in all of them where they overlap.
        const double leaning = pitch * (at[1] * c - at[0] * sn);
        q.row[k] = height + extra[k] + axial * H[k] + lift[0] * c + lift[1] * sn - roll * dn - while_read * mid + leaning;
        q.row_scale[k] = -while_read;
        q.axis[k] = axis + axis_slope * (H[k] - H_mid) + sway[0] * (c * c - sn * sn) + sway[1] * (2.0 * sn * c) + dn;
        q.row_per_col[k] = -roll, q.col_per_row[k] = roll;
        q.angle_deg[k] = ang[k];
        q.angle_deg_per_row[k] = step[k] * per_row;
        q.col_scale[k] = column_scale;
    }
    // about the rotation axis' column (its course over the frames and the map's part there are under a pixel, which
    // times the scale is nothing)
    q.scale_col = axis;
    return q;
}

// Calibration::frame_poses of a joint calibration.
std::array<Poses, 2> joint_poses(const Json& cal, int rows, const std::array<std::vector<double>, 2>& ang, const std::array<std::vector<double>, 2>& H, std::string* not_applied) {
    const Json& scans = member(member(cal, "parameters", "the geometry"), "scans", "the parameters");
    const Json& placed = member(member(cal, "layout", "the geometry"), "scans", "the layout");
    if (!scans.is(Json::Array) || scans.array.size() != 2 || !placed.is(Json::Array) || placed.array.size() != 2)
        throw Error("geometry: the calibration is not of two scans (the poses of a pair are read here; a mosaic's tiles are read one at a time, own_poses)");
    const std::array<std::size_t, 2> first_helix = joint_helix(cal, 0);
    if (H[0].size() <= first_helix[0]) throw Error("geometry: the record holds fewer frames than the calibration's first scan");
    std::array<Poses, 2> out;
    // the slices are numbered from the first scan's row at its helix' first frame: its height, its helix' and the record's
    double origin = 0;
    {
        const Json& p = scans.array[0];
        double extra = 0;
        for (const JointStretch& st : joint_stretches(cal, 0))
            if (st.helix)
                if (const Json* given = member(p, "stretch_height", "a scan").find(st.name)) extra = number_of(*given, "a stretch's height");
        origin = number(p, "height", "a scan") + extra + (1.0 + number(p, "axial_scale", "a scan")) * H[0][first_helix[0]];
    }
    for (std::size_t s = 0; s < 2; ++s) {
        std::vector<std::uint8_t> seen;
        out[s] = joint_scan_poses(cal, rows, s, ang[s], H[s], false, seen);
        if (std::find(seen.begin(), seen.end(), 0) != seen.end())
            throw Error("the geometry was made without some of the scan's projection stretches (they were not on disk): it places no frame of them");
        for (double& v : out[s].row) v -= origin;
    }
    if (not_applied) {
        Json na = Json::dict();
        if (const Json* solution = cal.find("solution"))
            if (const Json* left = solution->find("left")) na.set("what the model leaves of the comparisons, per kind and scans [rms and median of frames, rows, columns]; the frames are not fitted", *left);
        *not_applied = na.text();
    }
    return out;
}

}  // namespace

std::size_t Calibration::scan_named(const std::string& name) const {
    const Json& placed = member(member(*doc_, "layout", "the geometry"), "scans", "the layout");
    if (!is_joint(*doc_) || !placed.is(Json::Array)) throw Error("geometry: the calibration does not name its scans (it is not a joint calibration)");
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
    if (!is_joint(*doc_)) throw Error("geometry: the frames a calibration was made on are a joint calibration's");
    const Json& scans = member(member(*doc_, "layout", "the geometry"), "scans", "the layout");
    if (!scans.is(Json::Array) || scan >= scans.array.size()) throw Error("geometry: the layout does not hold scan " + std::to_string(scan));
    for (const Json& s : member(scans.array[scan], "stretches", "a scan of the layout").array) {
        const Json* helix = s.find("helix");
        if (!helix || !helix->is(Json::Bool) || !helix->boolean) continue;
        const std::size_t offset = static_cast<std::size_t>(number(s, "offset", "a stretch"));
        return {offset + static_cast<std::size_t>(number(s, "first", "a stretch")), offset + static_cast<std::size_t>(number(s, "end", "a stretch"))};
    }
    throw Error("geometry: scan " + std::to_string(scan) + " of the calibration has no helix");
}

double Calibration::axis_column(std::size_t scan) const {
    const Json& scans = member(member(*doc_, "parameters", "the geometry"), "scans", "the parameters");
    if (!is_joint(*doc_) || !scans.is(Json::Array) || scan >= scans.array.size()) throw Error("geometry: the calibration does not hold scan " + std::to_string(scan));
    return number(scans.array[scan], "axis", "a scan");
}

Poses Calibration::own_poses(std::size_t scan, const std::vector<double>& angles_deg, const std::vector<double>& heights_rows) const {
    if (!is_joint(*doc_)) throw Error("geometry: a scan's own poses are a joint calibration's");
    std::vector<std::uint8_t> placed;
    return joint_scan_poses(*doc_, rows(), scan, angles_deg, heights_rows, true, placed);
}

std::array<Poses, 2> Calibration::frame_poses(const std::array<std::vector<double>, 2>& ang, const std::array<std::vector<double>, 2>& H,
                                              std::string* not_applied) const {
    const Json& cal = *doc_;
    if (is_joint(cal)) return joint_poses(cal, rows(), ang, H, not_applied);
    const Json& lay = member(cal, "layout", "the geometry");
    const Json& po = member(cal, "poses", "the geometry");
    const Json* mo_ = po.find("motion");
    if (!mo_ || !mo_->is(Json::Object) || !mo_->find("second_scan"))
        throw Error("the geometry holds no motion stage with the second scan's place: run the step with motion");
    const Json& mo = *mo_;
    const int R = rows();
    const double N = number(lay, "frames_per_turn", "the layout");
    const double mid = (R - 1) / 2.0;
    const double tau = number(member(cal, "row_time", "the geometry"), "seconds_per_row", "the rows' timing");
    const double period = period_of(cal);

    static const char* const names[3] = {"rest1", "helix", "rest2"};
    const Json& frames = member(lay, "frames", "the layout");
    if (!frames.is(Json::Array) || frames.array.size() != 2) throw Error("geometry: the layout's frames are not of two scans");
    std::size_t len[3], first[3];
    for (int a = 0; a < 3; ++a) {
        const double l0 = number(frames.array[0], names[a], "the layout's frames"), l1 = number(frames.array[1], names[a], "the layout's frames");
        if (l0 != l1) throw Error("the second scan is placed against the first at equal frames: the stretches must be equally long");
        len[a] = static_cast<std::size_t>(l0);
    }
    first[0] = 0, first[1] = len[0], first[2] = len[0] + len[1];
    const std::size_t n = len[0] + len[1] + len[2];
    for (int s = 0; s < 2; ++s)
        if (ang[s].size() != n || H[s].size() != n)
            throw Error("the record holds " + std::to_string(ang[0].size()) + " and " + std::to_string(ang[1].size()) + " frames, the geometry " + std::to_string(n));
    if (len[1] < len[0] || len[1] < len[2]) throw Error("geometry: a rest turn longer than the helix");

    const Json& S0 = member(mo, "scan 0", "the motion stage");
    const Json& S1 = member(mo, "scan 1", "the motion stage");
    const Json& P2 = member(mo, "second_scan", "the motion stage");
    const Json& starts = member(mo, "turns_at_start", "the motion stage");
    if (!starts.is(Json::Array) || starts.array.empty()) throw Error("geometry: the motion stage's turns_at_start is not a list");
    double start[3];
    for (int a = 0; a < 3; ++a) start[a] = number(starts.array[0], names[a], "turns_at_start");
    const double psi = number(po, "axis_roll_rad", "the poses");
    const Json& side = member(S0, "sideways", "the first scan's motion");
    const Json& mcj = member(side, "mirror_column", "the first scan's sideways motion");
    double mc[3];
    for (int a = 0; a < 3; ++a) mc[a] = number(mcj, names[a], "the mirror column");
    const double mc_per_row = number(mcj, "per_row", "the mirror column");
    const double tx = number(side, "travel_px_per_row", "the first scan's sideways motion");
    const Json& A = member(mo, "along_axis", "the motion stage");
    const double z_cos = number(A, "z_cos", "the motion along the axis"), z_sin = number(A, "z_sin", "the motion along the axis");
    const auto body_nodes = table(member(side, "nodes_px", "the first scan's sideways motion"), 2, "nodes_px");
    const double body_h = number(side, "node_spacing_turns", "the first scan's sideways motion");

    // the helix of the first scan against its encoder: views a turn apart give e(t + 1) - e(t) as a line in t
    const auto tb = table(member(member(S0, "helix", "the first scan's motion"), "table", "the first scan's helix"), 4, "the helix' table");
    std::vector<double> tt(tb.size()), ty(tb.size()), tq(tb.size());
    for (std::size_t k = 0; k < tb.size(); ++k) tt[k] = tb[k][0] / N, ty[k] = tb[k][2], tq[k] = tb[k][3];
    if (tb.size() < 2) throw Error("geometry: the helix' table holds fewer than two views");
    const calibration::Line ly = calibration::fit_line(tt, ty, 0, tt.size()), lq = calibration::fit_line(tt, tq, 0, tt.size());
    const double c1 = ly.slope, c0 = ly.intercept, g1 = lq.slope, g0 = lq.intercept;
    const auto e = [&](double t) { return (c0 - c1 / 2) * t + (c1 / 2) * (t * t); };  // rows beyond the encoder, 0 at the helix' first frame
    const auto G = [&](double t) { return (g0 - g1 / 2) * t + (g1 / 2) * (t * t); };  // frames beyond the encoder's angle and the rows' timing
    const double* H0 = H[0].data();
    const double Hh0 = H0[first[1]];
    const long hint[3] = {0, 0, static_cast<long>(len[1]) - static_cast<long>(len[2])};
    // the rest turns against the helix: their own terms
    double scale_error[3] = {0, 0, 0}, offset_rows[3] = {0, 0, 0}, angle_frames[3] = {0, 0, 0}, angle_frames_per_row[3] = {0, 0, 0};
    for (int a : {0, 2}) {
        const Json& r = member(S0, names[a], "the first scan's motion");
        scale_error[a] = number(r, "scale_error", names[a]), offset_rows[a] = number(r, "offset_rows", names[a]);
        angle_frames[a] = number(r, "angle_frames", names[a]), angle_frames_per_row[a] = number(r, "angle_frames_per_row", names[a]);
    }
    const auto body = [&](double t_abs, double th) {  // the rotating body's place against the axis, on the columns
        const double last = static_cast<double>(body_nodes.size());  // nodes at h, 2h, ..; (0, 0) at 0
        const double x = std::clamp(t_abs / body_h, 0.0, last - 1e-9);
        const std::size_t k = static_cast<std::size_t>(std::floor(x));
        const double f = x - static_cast<double>(k);
        const double a0 = k ? body_nodes[k - 1][0] : 0.0, a1 = k ? body_nodes[k - 1][1] : 0.0;
        const double v0 = a0 * (1 - f) + body_nodes[k][0] * f, v1 = a1 * (1 - f) + body_nodes[k][1] * f;
        return v0 * std::cos(th) + v1 * std::sin(th);
    };
    // the first scan's model at frame kk of stretch a seen at the angle enc_deg: row, axis, G
    const auto first_scan = [&](std::size_t kk, int a, double enc_deg, double& row, double& axis, double& Gk) {
        const double th = radians(enc_deg);
        const double z = z_cos * std::cos(th) + z_sin * std::sin(th);
        const double t_abs = start[a] + (static_cast<double>(kk) + 0.5) / N;
        const double b = body(t_abs, th);
        const double* Hs = H0 + first[a];
        if (a == 1) {
            const double P = Hs[kk] - Hh0;
            row = P + e((static_cast<double>(kk) + 0.5) / N) + z;
            axis = mc[1] + (tx - mc_per_row) * P + b;
            Gk = G((static_cast<double>(kk) + 0.5) / N);
            return;
        }
        const std::size_t f = static_cast<std::size_t>(static_cast<long>(kk) + hint[a]);  // the helix frame at the same angle, in the next turn
        const double Hf = H0[first[1] + f];
        const double D = Hf - Hs[kk];
        row = (Hf - Hh0) + e((static_cast<double>(f) + 0.5) / N) - D * (1 + scale_error[a]) - offset_rows[a] + z;
        axis = mc[a] + b;
        Gk = G((static_cast<double>(f) + 0.5) / N) - (angle_frames[a] + angle_frames_per_row[a] * D);
    };

    // what the comparison of the two scans at equal frames measured
    const Json& Pc = member(P2, "columns", "the second scan's place");
    const Json& Pr = member(P2, "rows", "the second scan's place");
    const auto col_nodes = table(member(Pc, "once_per_turn_nodes", "the second scan's columns"), 2, "once_per_turn_nodes");
    if (col_nodes.size() < 2) throw Error("geometry: the second scan's course once per turn has fewer than two nodes");
    const auto row_once = numbers(Pr, "once_per_turn", "the second scan's rows");
    if (row_once.size() < 2) throw Error("geometry: the second scan's rows once per turn are not two numbers");
    const double col_travel = number(Pc, "per_turn_of_travel", "the second scan's columns"), row_travel = number(Pr, "per_turn_of_travel", "the second scan's rows");
    const double move = number(lay, "coarse_move", "the layout");
    const double centre = overlap_centre_of(lay);
    const double helix_mid = start[1] + 0.5 * static_cast<double>(len[1]) / N;

    std::array<Poses, 2> out;
    std::vector<double> step(n), travel(n);
    for (int s = 0; s < 2; ++s) {
        Poses& d = out[static_cast<std::size_t>(s)];
        for (auto* v : {&d.row, &d.row_per_col, &d.axis, &d.col_per_row, &d.col_scale, &d.row_scale, &d.angle_deg, &d.angle_deg_per_row}) v->assign(n, 0.0);
        const double* an = ang[static_cast<std::size_t>(s)].data();
        const double* Hsc = H[static_cast<std::size_t>(s)].data();
        // per frame, within each stretch (the record steps between the stretches): the angular step and the travel
        for (int a = 0; a < 3; ++a) {
            if (!len[a]) continue;
            gradient(an + first[a], len[a], step.data() + first[a]);
            travel_per_frame(Hsc + first[a], len[a], travel.data() + first[a]);
        }
        for (std::size_t k = 0; k < n; ++k) {
            d.row_scale[k] = -travel[k] * tau / period;  // the sample travels while the rows are read
            d.angle_deg_per_row[k] = step[k] * tau / period;
        }
        for (int a = 0; a < 3; ++a) {
            double rho = 0, la = 0, lb = 0, col_offset = 0, col_per_row = 0, row_offset = 0, bv = 0, u_ref = 0;
            if (s == 1) {
                rho = number(member(po, "rotation_about_beam_rad", "the poses"), names[a], "the rotation about the beam");
                const auto line = numbers(member(member(po, "second_scan", "the poses"), names[a], "the second scan's lines"), "line", "the second scan's line");
                if (line.size() < 2) throw Error("geometry: the second scan's line is not two numbers");
                la = line[0], lb = line[1];
                col_offset = number(member(Pc, "offset", "the second scan's columns"), names[a], "the columns' offset");
                col_per_row = number(member(Pc, "per_row", "the second scan's columns"), names[a], "the columns per row");
                row_offset = number(member(Pr, "offset", "the second scan's rows"), names[a], "the rows' offset");
                bv = number(member(Pr, "per_row", "the second scan's rows"), names[a], "the rows per row");
                u_ref = centre - mc[a];
            }
            for (std::size_t kk = 0; kk < len[a]; ++kk) {
                const std::size_t k = first[a] + kk;
                double row, axis, Gk;
                first_scan(kk, a, an[k], row, axis, Gk);
                double rpc = -psi, cpr = mc_per_row;
                if (s == 1) {
                    const double th = radians(an[k]);
                    const double t_abs = start[a] + (static_cast<double>(kk) + 0.5) / N;
                    const double trav = a == 1 ? t_abs - helix_mid : 0.0;
                    // columns: the second scan's axis = the first's (with the first body's course) + the move + what
                    // the comparison at equal frames holds: the stretch's offset and the course once per turn
                    const double x = std::clamp(t_abs, 0.0, static_cast<double>(col_nodes.size()) - 1 - 1e-9);
                    const std::size_t k0 = static_cast<std::size_t>(std::floor(x));
                    const double f = x - static_cast<double>(k0);
                    const double E0 = col_nodes[k0][0] * (1 - f) + col_nodes[k0 + 1][0] * f, E1 = col_nodes[k0][1] * (1 - f) + col_nodes[k0 + 1][1] * f;
                    axis = axis + move + col_offset + col_travel * trav + E0 * std::cos(th) + E1 * std::sin(th);
                    cpr = cpr + col_per_row;
                    rpc = rpc + rho + lb;
                    const double once = row_once[0] * std::cos(th) + row_once[1] * std::sin(th);
                    row = row + (Hsc[k] - H0[k]) + row_offset + row_travel * trav + once - (rho + lb) * u_ref;
                    // the comparison's line in the row: the second scan's rows lie b (Y - middle) further on, which
                    // is a scale of its rows about the middle row (as a scale about row 0 and an offset)
                    d.row_scale[k] -= bv;
                    row = row - bv * mid;
                    d.col_scale[k] = la;
                }
                d.row[k] = row, d.axis[k] = axis, d.row_per_col[k] = rpc, d.col_per_row[k] = cpr;
                d.angle_deg[k] = an[k] - step[k] * Gk;
            }
        }
        d.scale_col = s == 1 ? centre + move : 0.0;
    }
    if (not_applied) {
        const auto copy = [](const Json& j, const char* key) {
            const Json* v = j.find(key);
            return v ? *v : Json();
        };
        Json na = Json::dict();
        na.set("second scan, twice per turn [px, rows]", Json::list().add(copy(Pc, "twice_per_turn")).add(copy(Pr, "twice_per_turn")));
        const Json* af = P2.find("angle_frames");
        Json angle = Json::dict();
        if (af) angle.set("offset", copy(*af, "offset")).set("per_turn_of_travel", copy(*af, "per_turn_of_travel")).set("once_per_turn", copy(*af, "once_per_turn"));
        na.set("second scan, angle against the first at equal frames [frames] (the stage's angle repeats; the comparison's angle holds the beam's divergence, "
               "which a detector-plane term cannot express)", angle);
        na.set("first scan, drift at rest from opposite views [rows per turn] (the rest turns are placed by the helix instead)", copy(A, "drift_rows_per_turn"));
        Json own = Json::dict();
        for (const char* k : {"rest1", "rest2", "helix"}) {
            const Json* v = S1.find(k);
            if (!v || !v->is(Json::Object)) continue;
            Json part = Json::dict();
            for (const auto& [q, val] : v->object)
                if (q != "table") part.set(q, val);
            own.set(k, part);
        }
        na.set("second scan's own helix and rest turns (it is placed by the first's)", own);
        *not_applied = na.text();
    }
    return out;
}

MergedLayout Calibration::merged_layout(const std::array<Poses, 2>& poses, const std::array<std::vector<double>, 2>& H, double quantum) const {
    const Json& lay = member(*doc_, "layout", "the geometry");
    const std::array<std::size_t, 2> helix = helix_frames();
    const std::size_t a = helix[0], b = helix[1];
    const double sign = number(lay, "row_sign", "the layout");
    for (int s = 0; s < 2; ++s)
        if (poses[static_cast<std::size_t>(s)].axis.size() < b || H[static_cast<std::size_t>(s)].size() < b) throw Error("geometry: the poses do not reach the helix' end");
    double axis[2];
    for (std::size_t s = 0; s < 2; ++s) axis[s] = median(std::vector<double>(poses[s].axis.begin() + static_cast<std::ptrdiff_t>(a), poses[s].axis.begin() + static_cast<std::ptrdiff_t>(b)));
    // the row a slice is read at beyond what the record alone gives (z_pix: the record's height from the scan's first frame)
    std::vector<double> diff(b - a);
    for (std::size_t k = a; k < b; ++k)
        diff[k - a] = (poses[1].row[k] + sign * (H[1][k] - H[1][0])) - (poses[0].row[k] + sign * (H[0][k] - H[0][0]));
    const double rows_ = median(diff);
    const auto q = [&](double v) { return std::nearbyint(v / quantum) * quantum; };  // halves to the even one, as the step always rounded
    MergedLayout out;
    out.cor_main_detector = q(axis[0]);
    out.vector = {q(-rows_), q(axis[0] - axis[1])};
    out.axis_first_scan = axis[0], out.axis_second_scan = axis[1], out.second_scan_rows = rows_, out.quantum = quantum;
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
TileGeometry TileGeometry::load(const std::string& path) { return parse(read_file(path)); }

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

}  // namespace helirec::geometry
