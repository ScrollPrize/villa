#pragma once
// The joint geometry model for one sighting (helirec/calibration/geometry_calibration.hpp states the model): the
// ray of the sample that an ideal detector position records. Templated on the scalar, so that the solver's
// residuals (automatic derivatives) and the report go through the same lines.
#include <cmath>

namespace helirec::calibration::geometry_calibration::model {

// What a sighting takes from the record: the stage's angle [rad] and height [rows] at the time its row was
// exposed, the height the scan's axis column is counted from, and the detector's middle row.
struct At {
    double th = 0, H = 0, H_mid = 0, mid_row = 0;
};

// The ray recorded at the ideal position (X, Y): its distance s from the sample's origin across the beam and its
// height z in the sample. The blocks, as the solver holds them:
//     detector [roll]        own [axis, axis_slope, axial_scale]        place [d0, d1, height]        stretch [height]
template <class T>
void ray_of(const T* detector, const T* own, const T* place, const T* stretch, const At& at, const T& X, const T& Y, T& s, T& z) {
    const T axis = own[0] + own[1] * (at.H - at.H_mid);
    const T u = X - axis - detector[0] * (Y - at.mid_row);  // columns from the axis, the roll taken out
    s = u - (place[0] * std::cos(at.th) + place[1] * std::sin(at.th));
    z = Y + detector[0] * u - place[2] - stretch[0] - (T(1.0) + own[2]) * at.H;
}

}  // namespace helirec::calibration::geometry_calibration::model
