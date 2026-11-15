#pragma once
#include "vc_object_store/geometry/FiberUvEvents.hpp"
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

namespace {
using UvIndexArray=nb::ndarray<const int64_t,nb::ndim<1>,nb::c_contig,nb::device::cpu>;
using UvDoubleArray=nb::ndarray<const double,nb::ndim<1>,nb::device::cpu>;
using UvBoolArray=nb::ndarray<const bool,nb::ndim<1>,nb::c_contig,nb::device::cpu>;

template<class T>auto uvOwnedColumn(std::unique_ptr<T[]> values,size_t count)
{
    auto* raw=values.release();
    nb::capsule owner(raw,[](void* p)noexcept{delete[] static_cast<T*>(p);});
    return nb::ndarray<const T,nb::numpy,nb::ndim<1>>(raw,{count},owner);
}

void bindFiberUvEvents(nb::module_& module)
{
    using namespace nb::literals;
    module.def("prepare_uv_event_columns",[](UvIndexArray a,UvIndexArray b,
        std::optional<UvIndexArray> graphToCatalog,UvIndexArray family,
        UvDoubleArray arcA,UvDoubleArray arcB,UvBoolArray parallel,
        std::optional<UvBoolArray> validA,std::optional<UvBoolArray> validB,
        std::optional<UvBoolArray> validConnector,std::optional<UvBoolArray> nodeMask,size_t workers){
        if(arcA.shape(0)!=a.shape(0)||arcB.shape(0)!=a.shape(0))
            throw std::invalid_argument("UV arc column differs");
        auto indices=[](const UvIndexArray& v){return std::span(v.data(),v.shape(0));};
        auto flags=[](const std::optional<UvBoolArray>& v){
            return v ? std::span<const bool>(v->data(),v->shape(0)) : std::span<const bool>{};};
        vc::geometry::FiberUvEventInputs input{indices(a),indices(b),
            graphToCatalog ? indices(*graphToCatalog) : std::span<const int64_t>{},indices(family),
            {arcA.data(),arcA.stride(0)},{arcB.data(),arcB.stride(0)},
            std::span(parallel.data(),parallel.shape(0)),flags(validA),flags(validB),flags(validConnector),flags(nodeMask)};
        auto output=std::make_unique<vc::geometry::FiberUvEvents>();
        {nb::gil_scoped_release release;*output=vc::geometry::prepareFiberUvEvents(input,workers);}
        nb::dict result;
        // Construction-only unsigned arcs must not stay pinned through the
        // final index columns after the native signing/target pass.
        result["selected_indices"]=uvOwnedColumn(std::move(output->selected),output->nodes);
        result["i"]=uvOwnedColumn(std::move(output->a),output->events);
        result["j"]=uvOwnedColumn(std::move(output->b),output->events);
        result["ds_a"]=uvOwnedColumn(std::move(output->arcA),output->events);
        result["ds_b"]=uvOwnedColumn(std::move(output->arcB),output->events);
        result["parallel"]=uvOwnedColumn(std::move(output->parallel),output->events);
        return result;
    },"i"_a,"j"_a,"graph_to_catalog"_a,"family"_a,"arc_a"_a,"arc_b"_a,"parallel"_a,
      "valid_a"_a=nb::none(),"valid_b"_a=nb::none(),"valid_connector"_a=nb::none(),
      "node_mask"_a=nb::none(),"workers"_a=32,
      "Parallel ordered UV filtering, perpendicular support and remapping into owned solver arrays.");
}
}
