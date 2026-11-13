#pragma once
#include "vc_object_store/objects/FiberAxialPriors.hpp"
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>

namespace {
void bindFiberAxial(nb::module_& module)
{
    using namespace nb::literals;
    module.def("fiber_axial_prior_columns",[](
        nb::ndarray<const int64_t,nb::ndim<1>,nb::c_contig,nb::device::cpu> offsets,
        nb::ndarray<const double,nb::ndim<2>,nb::c_contig,nb::device::cpu> xyz,
        const std::filesystem::path& path,const std::string& volume,
        const std::optional<std::array<int64_t,3>>& shape,double spacing,size_t workers){
        if(xyz.shape(1)!=3)throw std::invalid_argument("axial geometry must be float64 [N,3]");
        std::unique_ptr<float[]> output;
        {nb::gil_scoped_release release;output=vo::fiberAxialPriorColumns(
            std::span(offsets.data(),offsets.shape(0)),
            std::span(xyz.data(),xyz.shape(0)*3),path,volume,shape,spacing,workers);}
        auto* raw=output.release();nb::capsule owner(raw,[](void* p)noexcept{delete[] static_cast<float*>(p);});
        return nb::ndarray<const float,nb::numpy,nb::ndim<1>>(raw,{offsets.shape(0)-1},owner);
    },"offsets"_a,"xyz"_a,"umbilicus"_a,"volume"_a,"base_shape_zyx"_a=nb::none(),"spacing"_a=16.,"workers"_a=32,
      "Owned parallel axial priors using the shared exact midpoint and umbilicus-center kernels.");
}
}
