#pragma once
#include "vc_object_store/geometry/PolylineArc.hpp"
#include <nanobind/ndarray.h>
#include <memory>

namespace {
using ArcOffsets=nb::ndarray<const int64_t,nb::ndim<1>,nb::c_contig,nb::device::cpu>;
using ArcCoordinates=nb::ndarray<const double,nb::ndim<1>,nb::device::cpu>;
struct PyPolylineArcIndex {
    ArcOffsets offsets;
    ArcCoordinates arcs;
    vc::geometry::PolylineArcIndex index;
    PyPolylineArcIndex(ArcOffsets o,ArcCoordinates a):offsets(o),arcs(a),
        index(std::span(o.data(),o.shape(0)),a.data(),a.shape(0),a.stride(0)){}
};

void bindPolylineArcIndex(nb::module_& module){
    using namespace nb::literals;
    nb::class_<PyPolylineArcIndex>(module,"PolylineArcIndex",
        "Bulk interval lookup on packed polylines; retains offsets and strided arc buffers.")
        .def(nb::init<ArcOffsets,ArcCoordinates>(),"offsets"_a,"arc"_a)
        .def("locate",[](const PyPolylineArcIndex& self,ArcOffsets owners,
                          nb::ndarray<const double,nb::ndim<1>,nb::c_contig,nb::device::cpu> values){
            auto out=std::make_unique<vc::geometry::PolylineArcLocations>();
            {nb::gil_scoped_release release;
                *out=self.index.locate(std::span(owners.data(),owners.shape(0)),std::span(values.data(),values.shape(0)));}
            auto* raw=out.release();
            nb::capsule owner(raw,[](void* p)noexcept{delete static_cast<vc::geometry::PolylineArcLocations*>(p);});
            nb::dict result;
            result["segment"]=nb::ndarray<const int64_t,nb::numpy,nb::ndim<1>>(
                raw->segment.data(),{raw->segment.size()},owner);
            result["fraction"]=nb::ndarray<const double,nb::numpy,nb::ndim<1>>(
                raw->fraction.data(),{raw->fraction.size()},owner);
            return result;
        },"owner"_a,"arc"_a);
}
}
