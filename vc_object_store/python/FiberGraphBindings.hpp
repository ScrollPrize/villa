#pragma once
#include "vc_object_store/objects/FiberConstraintGraph.hpp"
#include "vc_object_store/ParallelFor.hpp"
#include <nanobind/ndarray.h>

namespace {
nb::dict fiberGraphColumnsToPy(vo::FiberGraphColumns columns,size_t workers)
{
    auto values=std::make_unique<vo::FiberGraphColumns>(std::move(columns));
    std::unique_ptr<uint8_t[]> text;
    {nb::gil_scoped_release release;
        text=vo::fiberGraphUuidAscii(std::span(values->fiberUuids.get(),values->nodes),workers);}
    auto* raw=values.release();nb::capsule owner(raw,[](void* p)noexcept{delete static_cast<vo::FiberGraphColumns*>(p);});
    auto* chars=text.release();nb::capsule textOwner(chars,[](void* p)noexcept{delete[] static_cast<uint8_t*>(p);});
    nb::dict result;result["volume"]=raw->volume;
    result["fiber_uuid_ascii"]=nb::ndarray<const uint8_t,nb::numpy,nb::ndim<2>>(chars,{raw->nodes,size_t(36)},textOwner);
    result["fiber_uuid"]=nb::ndarray<const uint8_t,nb::numpy,nb::ndim<2>>(
        reinterpret_cast<const uint8_t*>(raw->fiberUuids.get()),{raw->nodes,size_t(16)},owner);
    result["lengths"]=nb::ndarray<const double,nb::numpy,nb::ndim<1>>(raw->lengths.get(),{raw->nodes},owner);
    result["edge_uuid"]=nb::ndarray<const uint8_t,nb::numpy,nb::ndim<2>>(
        reinterpret_cast<const uint8_t*>(raw->edgeUuids.get()),{raw->events,size_t(16)},owner);
    result["event_ids"]=nb::ndarray<const int64_t,nb::numpy,nb::ndim<1>>(raw->eventIds.get(),{raw->events},owner);
    result["endpoints"]=nb::ndarray<const int64_t,nb::numpy,nb::ndim<2>>(
        reinterpret_cast<const int64_t*>(raw->endpoints.get()),{raw->events,size_t(2)},owner);
    result["arcs"]=nb::ndarray<const double,nb::numpy,nb::ndim<2>>(
        reinterpret_cast<const double*>(raw->arcs.get()),{raw->events,size_t(2)},owner);
    result["normal_a"]=nb::ndarray<const double,nb::numpy,nb::ndim<2>>(
        reinterpret_cast<const double*>(raw->normalA.get()),{raw->events,size_t(3)},owner);
    result["normal_b"]=nb::ndarray<const double,nb::numpy,nb::ndim<2>>(
        reinterpret_cast<const double*>(raw->normalB.get()),{raw->events,size_t(3)},owner);
    result["connector"]=nb::ndarray<const double,nb::numpy,nb::ndim<2>>(
        reinterpret_cast<const double*>(raw->connector.get()),{raw->events,size_t(3)},owner);
    result["distance"]=nb::ndarray<const double,nb::numpy,nb::ndim<1>>(raw->distance.get(),{raw->events},owner);
    result["parallel"]=nb::ndarray<const uint8_t,nb::numpy,nb::ndim<1>>(raw->parallel.get(),{raw->events},owner);
    result["valid_a"]=nb::ndarray<const uint8_t,nb::numpy,nb::ndim<1>>(raw->validA.get(),{raw->events},owner);
    result["valid_b"]=nb::ndarray<const uint8_t,nb::numpy,nb::ndim<1>>(raw->validB.get(),{raw->events},owner);
    result["valid_connector"]=nb::ndarray<const uint8_t,nb::numpy,nb::ndim<1>>(raw->validConnector.get(),{raw->events},owner);
    return result;
}
}
