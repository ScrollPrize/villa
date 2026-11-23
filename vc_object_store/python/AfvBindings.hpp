#pragma once
#include "vc_object_store/objects/AfvSource.hpp"

// Imported after StorageBindings.hpp, which owns the common Python conversions.
inline void bindAfvSource(nb::module_& m)
{
    nb::class_<vo::AfvSource>(m,"AfvSource")
        .def(nb::init<const std::filesystem::path&,int64_t,int64_t>(), nb::arg("path"),
             nb::arg("first_id")=0,nb::arg("past_last_id")=INT64_MAX)
        .def_prop_ro("metadata",[](const vo::AfvSource& source){return jsonLoads(source.metadataJson());})
        .def_prop_ro("uuid", &vo::AfvSource::uuid)
        .def_prop_ro("fiber_count", &vo::AfvSource::fiberCount)
        .def_prop_ro("point_count", &vo::AfvSource::pointCount)
        .def("__iter__",[](vo::AfvSource& source)->vo::AfvSource&{return source;},nb::rv_policy::reference_internal)
        .def("__next__",[](vo::AfvSource& source){
            vo::AfvFiber fiber; bool found;
            { nb::gil_scoped_release release; found=source.next(fiber); }
            if (!found) throw nb::stop_iteration();
            return nb::make_tuple(fiber.id,fiber.name,fiber.detail.family,fiber.detail.linePointCount,
                fiber.detail.length,*fiber.detail.bbox,jsonLoads(fiber.annotation),
                nb::bytes(reinterpret_cast<const char*>(fiber.xyz.data()),fiber.xyz.size()));
        });
}
