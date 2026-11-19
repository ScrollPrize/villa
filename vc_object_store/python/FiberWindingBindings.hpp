#pragma once
#include "vc_object_store/geometry/FiberWinding.hpp"
#include "vc_object_store/geometry/FiberWindingGraph.hpp"
#include "vc_object_store/geometry/FiberWindingFlow.hpp"
#include "FiberUvEventBindings.hpp"
#include <nanobind/ndarray.h>
#include <memory>
#include <optional>

namespace {
using WindingI64=nb::ndarray<const int64_t,nb::ndim<1>,nb::c_contig,nb::device::cpu>;
using WindingF64=nb::ndarray<const double,nb::ndim<1>,nb::c_contig,nb::device::cpu>;
using WindingArc=nb::ndarray<const double,nb::ndim<1>,nb::device::cpu>;
using WindingXYZ=nb::ndarray<const double,nb::ndim<2>,nb::c_contig,nb::device::cpu>;
using WindingIndexView=nb::ndarray<const int64_t,nb::ndim<1>,nb::device::cpu>;
using WindingFlag=nb::ndarray<const bool,nb::ndim<1>,nb::c_contig,nb::device::cpu>;

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
    PyPackedWindingGeometry(WindingI64 offsets_,WindingArc arcs_,WindingXYZ xyz_,WindingI64 turns_,WindingXYZ axis,double angle,int handedness)
        :offsets(offsets_),arcs(arcs_),xyz(xyz_),turns(turns_),
         geometry(windingSpan(offsets),arcs.data(),arcs.shape(0),arcs.stride(0),
             std::span(xyz.data(),xyz.shape(0)*xyz.shape(1)),windingSpan(turns),
             vc::geometry::WindingSeam(windingAxis(axis),angle,handedness))
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
                                         double angle,int handedness,int64_t firstOwner,size_t workers){
        if(xyz.shape(1)!=3)throw std::invalid_argument("expected packed XYZ triples");
        auto values=std::make_unique<vc::geometry::PackedWindingColumns>();
        const Seam seam(windingAxis(axis),angle,handedness);
        {nb::gil_scoped_release release;
            *values=vc::geometry::packWindingGeometry(windingSpan(offsets),
                std::span(xyz.data(),xyz.shape(0)*3),seam,firstOwner,workers);}
        auto* raw=values.get();auto owner=windingOwner(values);nb::dict result;
        result["source"]=windingColumn(raw->source,owner);result["arc"]=windingColumn(raw->arc,owner);
        result["turns"]=windingColumn(raw->turns,owner);return result;
    },"offsets"_a,"xyz"_a,"axis"_a,"angle"_a=-std::numbers::pi/2,"handedness"_a=1,"first_owner"_a=0,"workers"_a=1);
    nb::class_<Seam>(module,"NativeWindingSeam")
        .def("__init__",[](Seam* self,WindingXYZ axis,double angle,int handedness){new(self) Seam(windingAxis(axis),angle,handedness);},
             "axis"_a,"angle"_a=-std::numbers::pi/2,"handedness"_a=1)
        .def_prop_ro("handedness",&Seam::handedness)
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
        .def(nb::init<WindingI64,WindingArc,WindingXYZ,WindingI64,WindingXYZ,double,int>(),
             "offsets"_a,"arc"_a,"xyz"_a,"turns"_a,"axis"_a,"angle"_a=-std::numbers::pi/2,"handedness"_a=1)
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
             "a"_a,"b"_a,"arc_a"_a,"arc_b"_a,"directions"_a=nb::none(),"workers"_a=32)
        .def("prepare_winding_graph",[](const PyPackedWindingGeometry& self,WindingIndexView a,WindingIndexView b,
            WindingI64 graphToCatalog,WindingI64 family,WindingI64 signs,WindingArc arcA,WindingArc arcB,
            WindingArc distance,WindingFlag parallel,WindingXYZ normalA,WindingXYZ normalB,WindingXYZ connector,
            WindingFlag validA,WindingFlag validB,WindingFlag validConnector,
            double scale,double offset,double parallelWeight,double perpendicularWeight,
            double dotWeightPower,bool dampByWindingDistance,bool enclosures,
            size_t constraintBatch,size_t readBatch,size_t workers){
            if(normalA.shape(1)!=3||normalB.shape(1)!=3||connector.shape(1)!=3)
                throw std::invalid_argument("winding graph normals require XYZ triples");
            vc::geometry::FiberWindingGraphInput input{
                {a.data(),a.shape(0),a.stride(0)},{b.data(),b.shape(0),b.stride(0)},
                windingSpan(graphToCatalog),windingSpan(family),windingSpan(signs),
                {arcA.data(),arcA.shape(0),arcA.stride(0)},{arcB.data(),arcB.shape(0),arcB.stride(0)},
                {distance.data(),distance.shape(0),distance.stride(0)},
                windingSpan(parallel),windingSpan(validA),windingSpan(validB),windingSpan(validConnector),
                std::span(normalA.data(),normalA.shape(0)*3),std::span(normalB.data(),normalB.shape(0)*3),
                std::span(connector.data(),connector.shape(0)*3)};
            vc::geometry::FiberWindingGraphOptions options{scale,offset,perpendicularWeight,parallelWeight,
                dotWeightPower,dampByWindingDistance,enclosures,constraintBatch,readBatch,workers};
            vc::geometry::FiberWindingL1Columns output;
            {nb::gil_scoped_release release;output=vc::geometry::prepareFiberWindingGraph(self.geometry,input,options);}
            nb::dict result;
            result["a"]=uvOwnedColumn(std::move(output.a),output.arcs);
            result["b"]=uvOwnedColumn(std::move(output.b),output.arcs);
            result["target"]=uvOwnedColumn(std::move(output.target),output.arcs);
            result["weight"]=uvOwnedColumn(std::move(output.weight),output.arcs);
            result["read"]=output.read;result["used"]=output.used;result["expanded"]=output.expanded;
            result["without_distance"]=output.withoutDistance;
            if(enclosures){
                result["enclosure_a"]=uvOwnedColumn(std::move(output.enclosureA),output.enclosureArcs);
                result["enclosure_b"]=uvOwnedColumn(std::move(output.enclosureB),output.enclosureArcs);
                result["enclosure_target"]=uvOwnedColumn(std::move(output.enclosureTarget),output.enclosureArcs);
                nb::dict counts;counts["pairs"]=output.enclosurePairs;counts["accepted"]=output.enclosureArcs;
                counts["one_sided"]=output.enclosureOneSided;counts["contradictory"]=output.enclosureContradictory;
                counts["nonunique"]=output.enclosureNonunique;result["enclosure_counts"]=counts;
            }
            return result;
        },"i"_a,"j"_a,"graph_to_catalog"_a,"family"_a,"normal_sign"_a,"arc_a"_a,"arc_b"_a,
          "distance"_a,"parallel"_a,"normal_a"_a,"normal_b"_a,"connector"_a,
          "valid_a"_a,"valid_b"_a,"valid_connector"_a,
          "lasagna_winding_scale"_a=1.,"perpendicular_offset"_a=.5,"parallel_weight"_a=1.,"perpendicular_weight"_a=1.,
          "dot_weight_power"_a=0.,"damp_by_winding_distance"_a=false,
          "enclosures"_a=false,"constraint_batch"_a=500000,"read_batch"_a=32768,"workers"_a=32);
    module.def("fiber_signed_order",[](WindingXYZ connector,WindingXYZ a,WindingXYZ b,double minimum,size_t workers){
        const size_t n=connector.shape(0);
        if(connector.shape(1)!=3||a.shape(1)!=3||b.shape(1)!=3||a.shape(0)!=n||b.shape(0)!=n||!std::isfinite(minimum))
            throw std::invalid_argument("invalid fiber signed-order arrays");
        auto output=std::make_unique_for_overwrite<float[]>(n);
        {nb::gil_scoped_release release;
            vc::objects::parallelFor((n+4095)/4096,workers,[&](size_t block){
                for(size_t i=block*4096;i<std::min(n,(block+1)*4096);++i)
                    output[i]=vc::geometry::fiberSignedOrder(windingPoint(connector,i),windingPoint(a,i),windingPoint(b,i),minimum);
            });}
        return uvOwnedColumn(std::move(output),n);
    },"connector"_a,"inward_a"_a,"inward_b"_a,"cosine_min"_a=.1,"workers"_a=1);
    module.def("winding_min_cost_flow_available",&vc::geometry::windingMinCostFlowAvailable,
        "True when the native library was built with LEMON (VC_OBJECT_STORE_WITH_LEMON).");
    module.def("solve_winding_min_cost_flow",[](int64_t nodes,WindingI64 a,WindingI64 b,WindingI64 target,WindingF64 weight,
                                                 const std::string& backend,int64_t capacityScale){
        vc::geometry::WindingFlowOptions options;
        options.backend=vc::geometry::parseWindingFlowBackend(backend);options.capacityScale=capacityScale;
        vc::geometry::WindingFlowResult output;
        {nb::gil_scoped_release release;
            output=vc::geometry::solveWindingMinCostFlow(nodes,windingSpan(a),windingSpan(b),windingSpan(target),windingSpan(weight),options);}
        nb::dict result;
        result["winding"]=uvOwnedColumn(std::move(output.winding),output.nodes);
        result["flow"]=uvOwnedColumn(std::move(output.flow),output.arcs);
        result["capacity"]=uvOwnedColumn(std::move(output.capacity),output.arcs);
        nb::dict quantization;const auto& q=output.quantization;
        quantization["scale"]=q.scale;quantization["original_arcs"]=q.originalArcs;quantization["retained_arcs"]=q.retainedArcs;
        quantization["dropped_arcs"]=q.droppedArcs;quantization["original_weight"]=q.originalWeight;
        quantization["dropped_original_weight"]=q.droppedWeight;quantization["capacity_absolute_error"]=q.capacityAbsoluteError;
        result["quantization"]=quantization;
        result["backend"]=vc::geometry::windingFlowBackendName(options.backend);
        result["quantized_objective"]=output.quantizedObjective;result["objective"]=output.objective;
        nb::dict timing;timing["graph"]=output.graphSeconds;timing["solve"]=output.solveSeconds;timing["total"]=output.totalSeconds;
        result["timing"]=timing;
        return result;
    },"nodes"_a,"a"_a,"b"_a,"target"_a,"weight"_a,"backend"_a="network_simplex","capacity_scale"_a=1000,
      "Exact integer winding assignment of an aggregated L1 arc list via a LEMON min-cost circulation (CPU, small graphs).");
    module.def("winding_grouped_medians",[](WindingI64 groups,WindingF64 input,size_t workers){
        auto values=std::make_unique<vc::geometry::GroupedMedians>();
        {nb::gil_scoped_release release;*values=vc::geometry::groupedMedians(windingSpan(groups),windingSpan(input),workers);}
        auto* raw=values.get();auto owner=windingOwner(values);nb::dict result;
        result["groups"]=windingColumn(raw->groups,owner);result["medians"]=windingColumn(raw->medians,owner);return result;
    },"groups"_a,"values"_a,"workers"_a=32);
}
}
