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
struct At {
    double th = 0, H = 0, H_mid = 0, mid_row = 0;
};

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
// (and per scan once [cos, sin], which is not of a ray: own_lift below)
template <class T>
void ray_of(const T* detector, const T* stage, const T* own, const T* place, const T* stretch, const At& at, const T& X, const T& Y, T& s, T& z) {
    const double c = std::cos(at.th), sn = std::sin(at.th);
    const T axis = own[0] + own[1] * (at.H - at.H_mid) + stage[2] * (c * c - sn * sn) + stage[3] * (2.0 * sn * c);
    const T u = X - axis - detector[0] * (Y - at.mid_row);  // columns from the axis, the roll taken out
    s = u - (place[0] * c + place[1] * sn);
    z = Y + detector[0] * u - place[2] - stretch[0] - (T(1.0) + own[2]) * at.H - (stage[0] * c + stage[1] * sn);
}

// What a scan's views of itself show beyond the rays: with the rotation axis pitched towards the beam, views half
// a turn apart are not the same rays, and their rows differ by twice the pitch x the depth of what they register,
// which for a scan whose axis lies off the middle of the sample's detail is a sinusoid of the angle. once: that
// scan's (cos, sin) [rows], as the lift once per turn that its own views would take it for (the stage's sign).
// The rows by which b's lie above a's for it; nothing for two views at one angle.
template <class T>
T own_lift(const T* once, const At& a, const At& b) {
    return once[0] * (std::cos(b.th) - std::cos(a.th)) + once[1] * (std::sin(b.th) - std::sin(a.th));
}

}  // namespace helirec::calibration::geometry_calibration::model
