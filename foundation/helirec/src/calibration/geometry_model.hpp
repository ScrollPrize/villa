#pragma once
// The joint geometry model for one sighting (helirec/calibration/geometry_calibration.hpp states the model): the
// ray of the sample that an ideal detector position records. Templated on the scalar, so that the solver's
// residuals (automatic derivatives) and the report go through the same lines.
#include <algorithm>
#include <cmath>
#include <vector>

#include "helirec/common.hpp"

namespace helirec::calibration::geometry_calibration::model {

// a per-frame series of the record at a fraction of a frame: a line between frames, continued beyond the ends
inline double at_frame(const std::vector<double>& v, double f) {
    const long n = static_cast<long>(v.size());
    if (n < 1) throw Error("geometry: a stretch without frames");
    if (n == 1) return v[0];
    const long i = std::clamp(static_cast<long>(std::floor(f)), 0L, n - 2);
    const double w = f - static_cast<double>(i);
    return v[static_cast<std::size_t>(i)] * (1.0 - w) + v[static_cast<std::size_t>(i) + 1] * w;
}

// What a sighting takes from the record: the stage's angle [rad] and height [rows] at the time its row was
// exposed, the height the scan's axis column is counted from, and the detector's middle row.
// per_depth: what the beam's divergence does between scans (Acquisition::per_depth; 0: left out), about the
// detector's middle (mid_col, mid_row).
// th_per_timing, H_per_timing: by how much th and H move per frame per row of the rows' timing beyond the one they
// were read at (the sighting's rows from the middle row x the record's angle and height per frame there).
struct At {
    double th = 0, H = 0, H_mid = 0, mid_row = 0, mid_col = 0, per_depth = 0;
    double th_per_timing = 0, H_per_timing = 0;
};
// The stage's angle and height at the time a sighting's row was exposed, as the solver's scalar: At's, moved by
// what the rows' timing is beyond the one At was read at [frames per row]. The record's course is a line between
// frames, so this is that course itself as long as the row's time stays between the same two frames.
template <class T>
struct Now {
    T th, H;
};
template <class T>
Now<T> now(const At& at, const T& beyond) {
    return {T(at.th) + beyond * at.th_per_timing, T(at.H) + beyond * at.H_per_timing};
}

// The ideal position (X, Y) the stored pixel (c, v) looks along: the detector's map added, X = c + Eu(c, v),
// Y = v + Ev(c, v). map: the coefficients of Eu, then those of Ev, `terms` each; phi: the terms' values at (c, v)
// (helirec/geometry.hpp states the polynomial). terms = 0: no map, the pixel itself.
template <class T>
void ideal_position(const T* map, const double* phi, int terms, double c, double v, T& X, T& Y) {
    X = T(c), Y = T(v);
    for (int k = 0; k < terms; ++k) X += map[k] * phi[k], Y += map[terms + k] * phi[k];
}

// The ray recorded at the ideal position (X, Y): its distance s from the sample's origin across the beam and its
// height z in the sample. The blocks, as the solver holds them:
//     detector [roll]        own [axis, axis_slope, axial_scale]        place [d0, d1, height]        stretch [height]
//     stage [height once per turn: cos, sin; axis column twice per turn: cos, sin]        map [Eu's terms, Ev's terms]
//     timing [the rows' timing]: it is in `when`, the angle and height at the time the sighting's row was exposed
// (and per scan once [cos, sin], which is not of a ray: own_lift below)
// The beam diverges: a scan whose sample sits at d on the rotation stage holds it d . (-sin th, cos th) px nearer
// the source than the rotation axis, and what it records is magnified by that x per_depth about the detector's
// middle against a sample on the axis. The position is taken back to that of the sample on the axis first.
template <class T>
void ray_of(const T* detector, const T* stage, const T* own, const T* place, const T* stretch, const At& at, const Now<T>& when, const T& X_seen, const T& Y_seen, T& s, T& z) {
    using std::cos;
    using std::sin;
    const T c = cos(when.th), sn = sin(when.th);
    T X = X_seen, Y = Y_seen;
    if (at.per_depth != 0.0) {
        const T nearer = at.per_depth * (place[1] * c - place[0] * sn);
        X -= nearer * (X_seen - at.mid_col), Y -= nearer * (Y_seen - at.mid_row);
    }
    const T axis = own[0] + own[1] * (when.H - at.H_mid) + stage[2] * (c * c - sn * sn) + stage[3] * (2.0 * sn * c);
    const T u = X - axis - detector[0] * (Y - at.mid_row);  // columns from the axis, the roll taken out
    s = u - (place[0] * c + place[1] * sn);
    z = Y + detector[0] * u - place[2] - stretch[0] - (T(1.0) + own[2]) * when.H - (stage[0] * c + stage[1] * sn);
}

// What a scan's views of itself show beyond the rays: with the rotation axis pitched towards the beam, views half
// a turn apart are not the same rays, and their rows differ by twice the pitch x the depth of what they register,
// which for a scan whose axis lies off the middle of the sample's detail is a sinusoid of the angle. once: that
// scan's (cos, sin) [rows], as the lift once per turn that its own views would take it for (the stage's sign).
// The rows by which b's lie above a's for it; nothing for two views at one angle.
template <class T>
T own_lift(const T* once, const Now<T>& a, const Now<T>& b) {
    using std::cos;
    using std::sin;
    return once[0] * (cos(b.th) - cos(a.th)) + once[1] * (sin(b.th) - sin(a.th));
}

// What the pitch of the rotation axis towards the beam does to a ray: it climbs by the pitch per pixel of depth, so
// a scan whose sample sits at d on the rotation stage records the ray through the sample's origin pitch x the
// origin's depth higher, the depth being d . (-sin th, cos th). The rows by which the scan's sighting lies higher
// for it. Between two scans at one angle the difference of the two is exact, at every depth of the sample; between
// views half a turn apart it is not (own_lift above, and the note at the residual).
template <class T>
T pitch_lift(const T* pitch, const T* place, const Now<T>& when) {
    using std::cos;
    using std::sin;
    return pitch[0] * (place[1] * cos(when.th) - place[0] * sin(when.th));
}

// A scan holds the sample turned by `turn` (x, y, z [rad], small) about the sample's origin at the height `about`:
// the point p lies at p + turn x (p - origin). Of that a projection at the angle th holds exactly what is the same
// at every depth along the ray: with along = turn . (-sin th, cos th), the tilt's part about the beam's direction
// (the sample leaning across the beam), the ray recorded at (s, z) is the sample's ray at
//     s - along (z - about),        z + along s.
// The tilt's part about the direction across the beam moves the rows by the depth, which no ray holds, and the
// spin is the same view further on in angle (the frames', which are not fitted): neither is here.
template <class T>
void turned_back(const T* turn, const Now<T>& when, double about, T& s, T& z) {
    using std::cos;
    using std::sin;
    const T along = turn[1] * cos(when.th) - turn[0] * sin(when.th);
    const T s_seen = s;
    s -= along * (z - about);
    z += along * s_seen;
}

// Where scan b alone reconstructs the point q of the sample against where scan a does: the difference of the two
// scans' rigid motions there, D_b - D_a with D = (d, height) + rotation x q. q: x, y from the sample's origin, z
// from the height the rotations are about. place [d0, d1, height], turn [x, y, z] (rad), as the solver holds them.
template <class T>
void lens_displacement(const T* place_a, const T* turn_a, const T* place_b, const T* turn_b, const double* q, T* moved) {
    const T w[3] = {turn_b[0] - turn_a[0], turn_b[1] - turn_a[1], turn_b[2] - turn_a[2]};
    moved[0] = place_b[0] - place_a[0] + w[1] * q[2] - w[2] * q[1];
    moved[1] = place_b[1] - place_a[1] + w[2] * q[0] - w[0] * q[2];
    moved[2] = place_b[2] - place_a[2] + w[0] * q[1] - w[1] * q[0];
}

}  // namespace helirec::calibration::geometry_calibration::model
