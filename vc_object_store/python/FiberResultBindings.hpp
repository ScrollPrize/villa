#pragma once
#include "vc_object_store/objects/FiberGraphSolveExtension.hpp"
#include "vc_object_store/objects/FiberConstraintGraph.hpp"
#include <nanobind/ndarray.h>

namespace {
nb::dict fiberResultColumnsToPy(vo::FiberGraphResultColumns columns)
{
    auto* raw=new vo::FiberGraphResultColumns(std::move(columns));
    nb::capsule owner(raw,[](void* p) noexcept {delete static_cast<vo::FiberGraphResultColumns*>(p);});
    nb::dict result;
    result["fiber_uuid"]=nb::ndarray<const uint8_t,nb::numpy,nb::ndim<2>>(
        reinterpret_cast<const uint8_t*>(raw->fiberUuids.data()),{raw->fiberUuids.size(),size_t(16)},owner);
    result["values"]=nb::ndarray<const int64_t,nb::numpy,nb::ndim<2>>(
        reinterpret_cast<const int64_t*>(raw->integers.data()),{raw->integers.size(),size_t(5)},owner);
    result["valid"]=nb::ndarray<const uint8_t,nb::numpy,nb::ndim<2>>(
        reinterpret_cast<const uint8_t*>(raw->valid.data()),{raw->valid.size(),size_t(5)},owner);
    std::string text;text.reserve(raw->fiberUuids.size()*36);
    for(const auto& key:raw->fiberUuids)text+=vo::fiberGraphUuid(key);
    result["uuid_ascii"]=nb::bytes(text.data(),text.size());
    result["columns"]=vo::FiberGraphResultColumns::names;
    return result;
}
}
