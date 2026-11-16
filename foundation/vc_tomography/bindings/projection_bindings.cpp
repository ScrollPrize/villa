#include "ndarray_utils.hpp"
#include "helirec/projection_subset.hpp"
#include "helirec/projection_store.hpp"
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;
using namespace nb::literals;
using namespace helirec;
using namespace helirec::py;
void bind_projection(nb::module_& m) {
    m.def("is_projection_store_scan", &is_projection_store_scan, "path"_a,
          "True when path is a scan group of a projection store (helirec/projection_store.hpp)");
    m.def("frames_in_projection_store", &frames_in_projection_store, "frames"_a, "tile_group"_a,
          "The frames of a FrameList (detector files .../scanNNNN/<file>) as frames of the scan groups of STORE.zarr/<tile>: "
          "read them with RawReader(list(f.files), f.dataset, rows, cols, reduce). (helirec::frames_in_projection_store)");
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

    m.def("declare_projection_store_scan", &declare_projection_store_scan, "root"_a, "spec"_a, "reduce"_a, "source"_a, "dtype"_a, "clevel"_a = 1,
          "Create what is missing of ROOT/<tile>/<scan> of a projection store and record level `reduce` as stored; spec is JSON text "
          "(helirec/projection_store.hpp). One caller at a time. (helirec::declare_projection_store_scan)");
    nb::class_<ProjectionStoreWriter>(m, "ProjectionStoreWriter", "Frames into one level of a declared scan group of a projection store; frames are "
                                                                     "numbered as in the detector's scan (helirec::ProjectionStoreWriter)")
        .def(nb::init<const std::string&, int>(), "scan_group"_a, "reduce"_a)
        .def_prop_ro("rows", &ProjectionStoreWriter::rows)
        .def_prop_ro("cols", &ProjectionStoreWriter::cols)
        .def_prop_ro("dtype", &ProjectionStoreWriter::dtype)
        .def("holds", &ProjectionStoreWriter::holds, "first"_a, "count"_a, "whether frames [first, first + count) are stored")
        .def("write", [](ProjectionStoreWriter& w, std::int64_t first, nb::ndarray<const std::uint16_t, nb::ndim<3>, nb::c_contig, nb::device::cpu> data, bool verify) {
            require(w.dtype() == "uint16" && data.shape(1) == static_cast<std::size_t>(w.rows()) && data.shape(2) == static_cast<std::size_t>(w.cols()), "writer shape or dtype mismatch");
            nb::gil_scoped_release nogil; w.write(first, static_cast<std::int64_t>(data.shape(0)), data.data(), verify);
        }, "first"_a, "data"_a.noconvert(), "verify"_a = true)
        .def("write", [](ProjectionStoreWriter& w, std::int64_t first, nb::ndarray<const float, nb::ndim<3>, nb::c_contig, nb::device::cpu> data, bool verify) {
            require(w.dtype() == "float32" && data.shape(1) == static_cast<std::size_t>(w.rows()) && data.shape(2) == static_cast<std::size_t>(w.cols()), "writer shape or dtype mismatch");
            nb::gil_scoped_release nogil; w.write(first, static_cast<std::int64_t>(data.shape(0)), data.data(), verify);
        }, "first"_a, "data"_a.noconvert(), "verify"_a = true);
}
