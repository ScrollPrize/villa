#pragma once
#include "vc_object_store/geometry/FiberWinding.hpp"
#include <nanobind/ndarray.h>
#include <memory>
#include <optional>

namespace {
using WindingI64=nb::ndarray<const int64_t,nb::ndim<1>,nb::c_contig,nb::device::cpu>;
using WindingF64=nb::ndarray<const double,nb::ndim<1>,nb::c_contig,nb::device::cpu>;
using WindingArc=nb::ndarray<const double,nb::ndim<1>,nb::device::cpu>;
using WindingXYZ=nb::ndarray<const double,nb::ndim<2>,nb::c_contig,nb::device::cpu>;

template<class T>auto windingSpan(const T& value){return std::span(value.data(),value.shape(0));}
template<class T>auto windingColumn(std::vector<T>& value,const nb::capsule& owner){
    return nb::ndarray<const T,nb::numpy,nb::ndim<1>>(value.data(),{value.size()},owner);
}
template<class T>nb::capsule windingOwner(std::unique_ptr<T>& value){
    auto* raw=value.release();return nb::capsule(raw,[](void* pointer)noexcept{delete static_cast<T*>(pointer);});
}
inline vc::geometry::Point windingPoint(WindingXYZ points,size_t i){
    return {points.data()[3*i],points.data()[3*i+1],points.data()[3*i+2]};
}
inline std::vector<vc::geometry::Point> windingAxis(WindingXYZ points){
    if(points.shape(1)!=3)throw std::invalid_argument("expected XYZ columns");
    std::vector<vc::geometry::Point> axis(points.shape(0));
    for(size_t i=0;i<axis.size();++i)axis[i]=windingPoint(points,i);
    return axis;
}

struct PyPackedWindingGeometry {
    WindingI64 offsets;
    WindingArc arcs;
    WindingXYZ xyz;
    WindingI64 turns;
    vc::geometry::PackedFiberGeometry geometry;
    PyPackedWindingGeometry(WindingI64 offsets_,WindingArc arcs_,WindingXYZ xyz_,WindingI64 turns_,WindingXYZ axis,double angle)
        :offsets(offsets_),arcs(arcs_),xyz(xyz_),turns(turns_),
         geometry(windingSpan(offsets),arcs.data(),arcs.shape(0),arcs.stride(0),
             std::span(xyz.data(),xyz.shape(0)*xyz.shape(1)),windingSpan(turns),
             vc::geometry::WindingSeam(windingAxis(axis),angle))
    {if(xyz.shape(1)!=3)throw std::invalid_argument("expected packed XYZ triples");}
};

template<class Label>nb::dict windingTargets(const PyPackedWindingGeometry& self,WindingI64 owners,
    nb::ndarray<const Label,nb::ndim<1>,nb::c_contig,nb::device::cpu> labels,
    WindingI64 a,WindingI64 b,WindingF64 arcA,WindingF64 arcB,nb::object directionObject,size_t workers)
{
    std::optional<WindingF64> directions;
    if(!directionObject.is_none())directions=nb::cast<WindingF64>(directionObject);
    auto values=std::make_unique<vc::geometry::FixedFiberTargetColumns<Label>>();
    {nb::gil_scoped_release release;
        *values=vc::geometry::fixedFiberTargetColumns(self.geometry,windingSpan(owners),windingSpan(labels),
            windingSpan(a),windingSpan(b),windingSpan(arcA),windingSpan(arcB),
            directions?std::span<const double>(directions->data(),directions->shape(0)):std::span<const double>{},workers);}
    auto* raw=values.get();auto owner=windingOwner(values);nb::dict result;
    result["target"]=windingColumn(raw->target,owner);
    if(directions){result["ds_a"]=windingColumn(raw->arcA,owner);result["ds_b"]=windingColumn(raw->arcB,owner);}
    return result;
}

void bindFiberWinding(nb::module_& module){
    using namespace nb::literals;
    using Seam=vc::geometry::WindingSeam;
    module.def("pack_winding_geometry",[](WindingI64 offsets,WindingXYZ xyz,WindingXYZ axis,
                                         double angle,int64_t firstOwner,size_t workers){
        if(xyz.shape(1)!=3)throw std::invalid_argument("expected packed XYZ triples");
        auto values=std::make_unique<vc::geometry::PackedWindingColumns>();
        const Seam seam(windingAxis(axis),angle);
        {nb::gil_scoped_release release;
            *values=vc::geometry::packWindingGeometry(windingSpan(offsets),
                std::span(xyz.data(),xyz.shape(0)*3),seam,firstOwner,workers);}
        auto* raw=values.get();auto owner=windingOwner(values);nb::dict result;
        result["source"]=windingColumn(raw->source,owner);result["arc"]=windingColumn(raw->arc,owner);
        result["turns"]=windingColumn(raw->turns,owner);return result;
    },"offsets"_a,"xyz"_a,"axis"_a,"angle"_a=-std::numbers::pi/2,"first_owner"_a=0,"workers"_a=1);
    nb::class_<Seam>(module,"NativeWindingSeam")
        .def("__init__",[](Seam* self,WindingXYZ axis,double angle){new(self) Seam(windingAxis(axis),angle);},
             "axis"_a,"angle"_a=-std::numbers::pi/2)
        .def("relative",[](const Seam& self,WindingXYZ input,size_t workers){
            if(input.shape(1)!=3)throw std::invalid_argument("expected XYZ columns");
            const auto count=input.shape(0);auto values=std::make_unique<std::vector<double>>(count*2);
            {nb::gil_scoped_release release;
                vc::objects::parallelFor((count+4095)/4096,workers,[&](size_t block){
                    for(size_t i=block*4096;i<std::min(count,(block+1)*4096);++i){
                        const auto relative=self.relative(windingPoint(input,i));(*values)[2*i]=relative[0];(*values)[2*i+1]=relative[1];}
                });}
            auto* raw=values.get();auto owner=windingOwner(values);
            return nb::ndarray<const double,nb::numpy,nb::ndim<2>>(raw->data(),{count,size_t(2)},owner);
        },"points"_a,"workers"_a=32)
        .def("segments",[](const Seam& self,WindingXYZ first,WindingXYZ last,size_t workers){
            const auto count=first.shape(0);
            if(first.shape(1)!=3||last.shape(1)!=3||count!=last.shape(0))throw std::invalid_argument("expected matching XYZ segment arrays");
            auto values=std::make_unique<std::vector<int64_t>>(count);
            {nb::gil_scoped_release release;
                vc::objects::parallelFor((count+4095)/4096,workers,[&](size_t block){
                    for(size_t i=block*4096;i<std::min(count,(block+1)*4096);++i)(*values)[i]=self.transport(windingPoint(first,i),windingPoint(last,i));});}
            auto* raw=values.get();auto owner=windingOwner(values);return windingColumn(*raw,owner);
        },"start"_a,"end"_a,"workers"_a=32);
    nb::class_<PyPackedWindingGeometry>(module,"PackedWindingGeometry",
        "Shared packed measured geometry; input buffers are retained and native operations release the GIL.")
        .def(nb::init<WindingI64,WindingArc,WindingXYZ,WindingI64,WindingXYZ,double>(),
             "offsets"_a,"arc"_a,"xyz"_a,"turns"_a,"axis"_a,"angle"_a=-std::numbers::pi/2)
        .def("geometry_priors",[](const PyPackedWindingGeometry& self,WindingI64 owners,WindingI64 family,double spacing,size_t workers){
            auto values=std::make_unique<vc::geometry::FiberGeometryPriors>();
            {nb::gil_scoped_release release;
                *values=vc::geometry::fiberGeometryPriors(self.geometry,windingSpan(owners),windingSpan(family),spacing,workers);}
            auto* raw=values.get();auto owner=windingOwner(values);nb::dict result;
            result["directions"]=windingColumn(raw->directions,owner);result["phase"]=windingColumn(raw->phase,owner);
            result["z"]=windingColumn(raw->z,owner);result["radii"]=windingColumn(raw->radii,owner);
            result["minimum_turn"]=windingColumn(raw->minimumTurn,owner);result["maximum_turn"]=windingColumn(raw->maximumTurn,owner);
            result["sample_owner"]=windingColumn(raw->sampleOwner,owner);result["sample_arc"]=windingColumn(raw->sampleArc,owner);
            result["sample_phase"]=windingColumn(raw->samplePhase,owner);result["sample_z"]=windingColumn(raw->sampleZ,owner);
            return result;
        },"owners"_a,"family"_a,"spacing"_a=1000.,"workers"_a=32)
        .def("fixed_target_columns",&windingTargets<int64_t>,"owners"_a,"labels"_a.noconvert(),
             "a"_a,"b"_a,"arc_a"_a,"arc_b"_a,"directions"_a=nb::none(),"workers"_a=32)
        .def("fixed_target_columns",&windingTargets<double>,"owners"_a,"labels"_a.noconvert(),
             "a"_a,"b"_a,"arc_a"_a,"arc_b"_a,"directions"_a=nb::none(),"workers"_a=32);
    module.def("winding_grouped_medians",[](WindingI64 groups,WindingF64 input,size_t workers){
        auto values=std::make_unique<vc::geometry::GroupedMedians>();
        {nb::gil_scoped_release release;*values=vc::geometry::groupedMedians(windingSpan(groups),windingSpan(input),workers);}
        auto* raw=values.get();auto owner=windingOwner(values);nb::dict result;
        result["groups"]=windingColumn(raw->groups,owner);result["medians"]=windingColumn(raw->medians,owner);return result;
    },"groups"_a,"values"_a,"workers"_a=32);
}
}
