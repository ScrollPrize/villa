#include <vc_reader/thinning.hpp>
#include <vc_reader/distance_transform.hpp>
#include <vc_reader/components.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <memory>
#include <cstring>
#ifdef VC_READER_CUDA
#include "../cuda_buffer.hpp"
#endif
namespace nb=nanobind;
using namespace nb::literals;
namespace {
using Input=nb::ndarray<const std::uint8_t,nb::c_contig>;
using HostInput=nb::ndarray<const std::uint8_t,nb::c_contig,nb::device::cpu>;
using Output=nb::ndarray<std::uint8_t,nb::c_contig>;
using namespace vcr::thinning;
Backend backend(const std::string& name) {
    if(name=="cpu") return Backend::Cpu;
    if(name=="cuda") return Backend::Cuda;
    throw nb::value_error("backend must be cpu or cuda");
}
template<class A> std::vector<std::size_t> arrayShape(const A& a) {return {a.shape_ptr(),a.shape_ptr()+a.ndim()};}
template<class T> auto array(std::unique_ptr<std::vector<T>> values,const std::vector<std::size_t>& dims) {
    auto* ptr=values.release();
    nb::capsule owner(ptr,[](void* p) noexcept {delete static_cast<std::vector<T>*>(p);});
    return nb::ndarray<nb::numpy,T>(ptr->data(),dims.size(),dims.data(),owner);
}
void skeletonInto(const Input& input,const Output& output,const std::string& algorithm,const std::string& execution,bool border) {
    if(input.ndim()<2 || input.ndim()!=output.ndim()) throw nb::value_error("expected matching arrays with at least two dimensions");
    for(std::size_t a=0;a<input.ndim();++a) if(input.shape(a)!=output.shape(a)) throw nb::value_error("output shape mismatch");
    if(input.device_type()!=output.device_type() || input.device_id()!=output.device_id()) throw nb::value_error("input/output must use the same device");
    const auto m=algorithmFromName(algorithm);const auto b=backend(execution);
    const auto rows=input.shape(input.ndim()-2),cols=input.shape(input.ndim()-1);
    std::size_t slices=1;for(std::size_t a=0;a+2<input.ndim();++a) slices*=input.shape(a);
    nb::gil_scoped_release release;
    if(input.device_type()==nb::device::cpu::value) {
        if(b==Backend::Cpu) {vcr::thinning::skeleton(input.data(),output.data(),rows,cols,slices,m,border);return;}
#ifdef VC_READER_CUDA
        vcr::thinning::cuda::skeleton(input.data(),output.data(),rows,cols,slices,m,border);return;
#endif
    }
#ifdef VC_READER_CUDA
    else if(input.device_type()==nb::device::cuda::value && b==Backend::Cuda) {
        vcr::device::Guard guard(input.device_id());
        vcr::thinning::cuda::skeletonDevice(input.data(),output.data(),rows,cols,slices,m,border);return;
    }
#endif
    throw std::invalid_argument("unsupported device/backend (CUDA support may not be built)");
}
}
NB_MODULE(_thinning,m) {
    nb::class_<Scratch>(m,"TraceWorkspace").def(nb::init<>());
    m.def("skeletonize_into",&skeletonInto,"input"_a.noconvert(),"output"_a.noconvert(),
          "method"_a="zhang_suen","backend"_a="cpu","preserve_border"_a=true,
          "Thin independent YX masks into a preallocated host or CUDA uint8 array.");
    m.def("skeletonize",[](HostInput input,const std::string& algorithm,const std::string& execution,bool border) {
        auto dims=arrayShape(input);auto values=std::make_unique<std::vector<std::uint8_t>>(input.size());
        Output out(values->data(),dims.size(),dims.data());
        Input source(input.data(),dims.size(),dims.data());
        skeletonInto(source,out,algorithm,execution,border);
        return array(std::move(values),dims);
    },"input"_a.noconvert(),"method"_a="zhang_suen","backend"_a="cpu","preserve_border"_a=true);
    m.def("trace_ridges",[](HostInput input,const std::string& execution,double minLength,bool returnTraces,Scratch* workspace) -> nb::object {
        if(input.ndim()!=2) throw nb::value_error("trace_ridges requires a 2D uint8 mask");
        const auto b=backend(execution);
        auto values=std::make_unique<std::vector<std::uint8_t>>(input.size());Traces traces;Scratch local;
        {
            nb::gil_scoped_release release;
            trace(input.data(),input.shape(0),input.shape(1),values->data(),returnTraces||minLength>0?&traces:nullptr,
                  workspace?*workspace:local,nullptr,{minLength},b);
        }
        nb::object image=nb::cast(array(std::move(values),arrayShape(input)));
        if(!returnTraces) return image;
        nb::list result;
        for(const auto& path:traces) {
            auto coordinates=std::make_unique<std::vector<int>>();coordinates->reserve(2*path.size());
            for(auto p:path) {coordinates->push_back(p.x);coordinates->push_back(p.y);}
            result.append(array(std::move(coordinates),{path.size(),2}));
        }
        return nb::make_tuple(image,result);
    },"input"_a.noconvert(),"backend"_a="cpu","min_length"_a=0.,"return_traces"_a=false,"workspace"_a=nb::none());
    m.def("connected_components",[](HostInput input,const std::string& execution,bool full) {
        const auto b=backend(execution);auto dims=arrayShape(input);
        auto values=std::make_unique<std::vector<std::uint32_t>>(input.size());
        { nb::gil_scoped_release release;
          if(b==Backend::Cpu) vcr::components::label(input.data(),values->data(),dims,full);
          else {
#ifdef VC_READER_CUDA
              vcr::components::cuda::label(input.data(),values->data(),dims,full);
#else
              throw std::runtime_error("CUDA support is not built");
#endif
          }
        }
        return array(std::move(values),dims);
    },"input"_a.noconvert(),"backend"_a="cpu","full_connectivity"_a=true);
    m.def("trace_ridges_into",[](Input input,Output output) {
        if(input.ndim()!=2 || output.ndim()!=2 || input.shape(0)!=output.shape(0) || input.shape(1)!=output.shape(1))
            throw nb::value_error("expected matching 2D arrays");
#ifdef VC_READER_CUDA
        if(input.device_type()!=nb::device::cuda::value || output.device_type()!=input.device_type() || output.device_id()!=input.device_id())
            throw nb::value_error("trace_ridges_into requires matching CUDA arrays");
        nb::gil_scoped_release release;vcr::device::Guard guard(input.device_id());
        vcr::thinning::cuda::traceDevice(input.data(),output.data(),input.shape(0),input.shape(1));
#else
        throw std::runtime_error("CUDA support is not built");
#endif
    },"input"_a.noconvert(),"output"_a.noconvert());
    m.def("manhattan_distance",[](HostInput input,const std::string& execution) {
        const auto b=backend(execution);auto dims=arrayShape(input);
        auto values=std::make_unique<std::vector<std::uint8_t>>(input.size());
        {
            nb::gil_scoped_release release;
            if(b==Backend::Cpu) vcr::distance::manhattan(input.data(),values->data(),dims);
            else {
#ifdef VC_READER_CUDA
                vcr::distance::cuda::manhattan(input.data(),values->data(),dims);
#else
                throw std::runtime_error("CUDA support is not built");
#endif
            }
        }
        return array(std::move(values),dims);
    },"input"_a.noconvert(),"backend"_a="cpu");
}
