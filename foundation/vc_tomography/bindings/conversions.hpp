#pragma once
// Library structs as the Python dicts and tuples more than one binding file returns.
#include "ndarray_utils.hpp"
#include "vc_tomography/calibration/intensity.hpp"
#include "vc_tomography/geometry.hpp"

namespace vc_tomography::py {

// a reconstruction's terms (geometry::Terms) as a dict of its fields
inline nb::dict terms_dict(const geometry::Terms& t) {
    nb::dict d;
    d["angles_deg"] = to_numpy(t.angles_deg);
    d["z_translation_scale"] = t.z_translation_scale;
    d["row_offset"] = to_numpy(t.row_offset), d["row_slope"] = to_numpy(t.row_slope), d["slope_col"] = t.slope_col;
    d["col_offset"] = to_numpy(t.col_offset), d["col_slope"] = to_numpy(t.col_slope), d["slope_row"] = t.slope_row;
    d["col_scale"] = to_numpy(t.col_scale), d["scale_col"] = t.scale_col, d["row_scale"] = to_numpy(t.row_scale), d["scale_row"] = t.scale_row;
    return d;
}

// (dark (rows, cols), flats (series, rows, cols))
inline nb::object darks_flats_tuple(const calibration::DarksFlats& d) {
    const std::size_t r = static_cast<std::size_t>(d.rows), c = static_cast<std::size_t>(d.cols);
    return nb::make_tuple(to_numpy_object(d.dark, {r, c}), to_numpy_object(d.flats, {d.series(), r, c}));
}

}  // namespace vc_tomography::py
