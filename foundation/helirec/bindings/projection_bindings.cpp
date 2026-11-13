#include "ndarray_utils.hpp"
#include "helirec/projection_subset.hpp"
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;
using namespace nb::literals;
using namespace helirec;
using namespace helirec::py;
void bind_projection(nb::module_& m) {
    m.def("plan_projection_subset", [](const std::string& master, double center, int slices, double turns) {
        ProjectionSubset p;
        { nb::gil_scoped_release nogil; p = plan_projection_subset(master, center, slices, turns); }
        nb::dict d;
        d["master"] = p.master; d["label"] = p.label; d["detector"] = p.detector;
        d["rows"] = p.rows; d["cols"] = p.cols; d["slices"] = p.slices;
        d["pixel_um"] = p.pixel_um; d["center_mm"] = p.center_mm; d["thickness_mm"] = p.thickness_mm;
        d["context_mm"] = p.context_mm; d["pitch_mm"] = p.pitch_mm;
        nb::list series;
        for (const auto& s : p.series) {
            nb::dict item;
            item["entry"] = s.entry; item["role"] = s.role; item["first"] = s.first; item["count"] = s.count;
            item["original_count"] = s.original_count;
            item["angle_deg"] = nb::cast(s.angle_deg); item["height_mm"] = nb::cast(s.height_mm); item["epoch"] = nb::cast(s.epoch);
            nb::list pieces;
            for (const auto& f : s.pieces) {
                nb::dict q;
                q["file"] = f.file; q["dataset"] = f.dataset; q["first"] = f.first;
                q["count"] = f.count; q["source_first"] = f.source_first;
                pieces.append(q);
            }
            item["pieces"] = pieces; series.append(item);
        }
        d["series"] = series;
        return d;
    }, "master"_a, "center_mm"_a, "slices"_a, "context_turns"_a = 1.5);
    nb::class_<UncompressedFrameWriter>(m, "UncompressedFrameWriter")
        .def(nb::init<const std::string&, const std::string&, std::int64_t, int, int, const std::string&>(),
             "path"_a, "dataset"_a, "frames"_a, "rows"_a, "cols"_a, "dtype"_a)
        .def("write", [](UncompressedFrameWriter& w, std::int64_t first,
                        nb::ndarray<const std::uint16_t, nb::ndim<3>, nb::c_contig, nb::device::cpu> data) {
            require(data.shape(1) == static_cast<std::size_t>(w.rows()) && data.shape(2) == static_cast<std::size_t>(w.cols()), "writer shape mismatch");
            nb::gil_scoped_release nogil; w.write(first, data.shape(0), data.data());
        }, "first"_a, "data"_a.noconvert())
        .def("write", [](UncompressedFrameWriter& w, std::int64_t first,
                        nb::ndarray<const float, nb::ndim<3>, nb::c_contig, nb::device::cpu> data) {
            require(data.shape(1) == static_cast<std::size_t>(w.rows()) && data.shape(2) == static_cast<std::size_t>(w.cols()), "writer shape mismatch");
            nb::gil_scoped_release nogil; w.write(first, data.shape(0), data.data());
        }, "first"_a, "data"_a.noconvert())
        .def("close", &UncompressedFrameWriter::close, nb::call_guard<nb::gil_scoped_release>());
}
