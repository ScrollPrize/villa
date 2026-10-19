#pragma once
#include "reader.hpp"
#include "../cuda_event.hpp"
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
namespace vcr::reader_cuda {
// Device storage handed to Python. It is marked exported (its destruction then
// synchronizes the device) only when shared through DLPack: arrays that stay
// inside vc_reader free stream-ordered without blocking.
struct Owned {
    device::Storage storage; std::size_t count; nb::dlpack::dtype dtype; int device;
    nb::object array(nb::handle self) const {
        return nb::cast(nb::ndarray<nb::array_api, nb::device::cuda>(storage->ptr,1,&count,self,nullptr,dtype,nb::device::cuda::value,device));
    }
};
inline nb::object owned(device::Storage storage, std::size_t count, nb::dlpack::dtype dtype, int device) {
    return nb::cast(Owned{std::move(storage),count,dtype,device});
}
inline nb::object tensor(Result result, int device) {
    nb::dlpack::dtype dtype{std::uint8_t(result.floating ? nb::dlpack::dtype_code::Float : nb::dlpack::dtype_code::UInt),std::uint8_t(result.floating ? 32 : 8),1};
    return owned(std::move(result.storage),result.count,dtype,device);
}
// Samples, or (samples, validity) when the request asked for a mask.
inline nb::object outputs(Result result, int device) {
    if (!result.valid) return tensor(std::move(result),device);
    Result mask{std::move(result.valid),result.count,false,nullptr};
    return nb::make_tuple(tensor(std::move(result),device),tensor(std::move(mask),device));
}
inline void bind(nb::module_& m) {
    using Host=nb::ndarray<nb::ro,nb::c_contig,nb::device::cpu>;
    using Device=nb::ndarray<nb::array_api,nb::c_contig,nb::device::cuda>;
    m.def("reader_cuda_available",[] { int n=0; return cudaGetDeviceCount(&n)==cudaSuccess && n>0; });
    nb::class_<Owned>(m,"CudaStorage")
        .def("array",[](nb::handle self) { return nb::cast<const Owned&>(self).array(self); })
        .def("export",[](Owned& o) { o.storage->exported=true; });
    m.def("_set_stream",[](std::uintptr_t handle) { device::current_stream()=handle ? device::stream_handle(handle) : nullptr; });
    m.def("_get_stream",[] { return reinterpret_cast<std::uintptr_t>(device::current_stream()); });
    m.def("_cuda_empty",[](std::size_t count,const std::string& kind,std::size_t item,int id) {
        if (count>SIZE_MAX/8 || (kind!="f" && kind!="i" && kind!="u" && kind!="b") ||
            (item!=1 && item!=2 && item!=4 && item!=8) || (kind=="b" && item!=1) || (kind=="f" && item<2))
            throw nb::value_error("invalid CUDA allocation dtype/size");
        auto code=kind=="f"?nb::dlpack::dtype_code::Float:kind=="i"?nb::dlpack::dtype_code::Int:
                  kind=="u"?nb::dlpack::dtype_code::UInt:nb::dlpack::dtype_code::Bool;
        device::Storage storage;
        { nb::gil_scoped_release release; device::Guard guard(id);
          storage=device::allocate(count*item,id); device::complete(); }
        return owned(std::move(storage),count,{std::uint8_t(code),std::uint8_t(item*8),1},id);
    });
    nb::class_<device::Event>(m,"CudaEvent")
        .def("synchronize",&device::Event::synchronize,nb::call_guard<nb::gil_scoped_release>())
        .def("wait",&device::Event::wait)
        .def("done",&device::Event::done);
    m.def("_record_event",[] { return std::make_shared<device::Event>(); });
    nb::class_<PointPlan>(m,"CudaPointPlan");
    m.def("_cuda_view",[](Device input,const std::vector<std::size_t>& shape,std::size_t offset) {
        std::size_t n=1; for (auto s:shape) { if (s && n>SIZE_MAX/s) throw nb::value_error("shape overflow"); n*=s; }
        if (offset>input.size() || n>input.size()-offset) throw nb::value_error("view exceeds storage");
        return nb::ndarray<nb::array_api, nb::device::cuda>(static_cast<char*>(input.data())+offset*input.dtype().bits/8,
            shape.size(),shape.data(),nb::cast(input,nb::rv_policy::reference),nullptr,input.dtype(),nb::device::cuda::value,input.device_id());
    });
    m.def("_cuda_describe",[](nb::ndarray<nb::ro,nb::device::cuda> input) {
        if (input.size()) {
            std::int64_t stride=1;
            for (std::size_t a=input.ndim();a-->0;) {
                if (input.shape(a)>1 && input.stride(a)!=stride) throw nb::type_error("CUDA array must be contiguous");
                stride*=input.shape(a);
            }
        }
        std::vector<std::size_t> shape; for (std::size_t a=0;a<input.ndim();++a) shape.push_back(input.shape(a));
        const auto dt=input.dtype();
        const char kind=dt.code==std::uint8_t(nb::dlpack::dtype_code::Float) ? 'f' : dt.code==std::uint8_t(nb::dlpack::dtype_code::Int) ? 'i' :
                        dt.code==std::uint8_t(nb::dlpack::dtype_code::UInt) ? 'u' : dt.code==std::uint8_t(nb::dlpack::dtype_code::Bool) ? 'b' : '?';
        if (kind=='?' || dt.lanes!=1) throw nb::value_error("unsupported device dtype");
        return nb::make_tuple(shape,std::string(1,kind)+std::to_string(dt.bits/8),input.device_id());
    },"input"_a.noconvert());
    m.def("_cuda_download",[](Device input,Bytes out) {
        if (input.nbytes()!=out.nbytes()) throw nb::value_error("download size mismatch");
        nb::gil_scoped_release release; device::Guard guard(input.device_id());
        if (out.nbytes()) { device::check(cudaMemcpyAsync(out.data(),input.data(),out.nbytes(),cudaMemcpyDeviceToHost,device::stream())); device::complete(); }
    });
    m.def("_cuda_upload",[](Host input,int id) {
        device::Storage storage;
        { nb::gil_scoped_release release; device::Guard guard(id); storage=device::allocate(input.nbytes(),id);
          if (input.nbytes()) device::check(cudaMemcpyAsync(storage->ptr,input.data(),input.nbytes(),cudaMemcpyHostToDevice,device::stream())); device::complete(); }
        std::size_t count=1; for (std::size_t a=0;a<input.ndim();++a) count*=input.shape(a);
        return owned(std::move(storage),count,input.dtype(),id);
    });
    m.def("_cuda_cast",[](Device input) {
        if (input.dtype().code!=std::uint8_t(nb::dlpack::dtype_code::UInt) || input.dtype().bits!=8) throw nb::value_error("astype supports uint8 to float32");
        Result result;
        { nb::gil_scoped_release release; device::Guard guard(input.device_id()); result={device::allocate(input.size()*4,input.device_id()),input.size(),true};
          cast(input.data(),result.storage->ptr,input.size(),1,4); device::complete(); }
        return tensor(std::move(result),input.device_id());
    });
    // Point arrays shared by the host/device point entry points.
    const auto point_dtype=[](auto points,int kind) {
        const auto dtype=points.dtype();
        if (points.ndim()!=2 || points.shape(1)!=3 || dtype.lanes!=1 ||
            (kind==2 ? (dtype.code!=std::uint8_t(nb::dlpack::dtype_code::Float) || dtype.bits!=32) :
                       (dtype.code!=std::uint8_t(nb::dlpack::dtype_code::Int) || (dtype.bits!=32 && dtype.bits!=64)))) throw nb::value_error("invalid point array");
        return int(dtype.bits);
    };
    const auto make_grids=[](const std::vector<std::array<std::int64_t,3>>& shapes,const std::vector<std::array<std::int64_t,3>>& bases,
                             const std::vector<std::array<float,3>>& origins,const std::vector<std::array<float,9>>& axes) {
        if (shapes.size()!=bases.size() || shapes.size()!=origins.size() || shapes.size()!=axes.size()) throw nb::value_error("grid counts differ");
        std::vector<Grid> grids(shapes.size());
        for (std::size_t i=0;i<grids.size();++i) { auto& g=grids[i]; std::copy_n(shapes[i].begin(),3,g.shape); std::copy_n(bases[i].begin(),3,g.base); std::copy_n(origins[i].begin(),3,g.origin); std::copy_n(axes[i].begin(),9,g.axes); }
        return grids;
    };
    nb::class_<Reader>(m,"CudaReader")
        .def(nb::init<Array&,std::size_t,int>(),nb::keep_alive<1,2>())
        .def("clear",&Reader::clear,nb::call_guard<nb::gil_scoped_release>())
        .def("stats",&Reader::stats,nb::call_guard<nb::gil_scoped_release>())
        .def("read_plan",[](Reader& r,const PointPlan& plan,const std::string& source,bool valid) {
            Result result; { nb::gil_scoped_release release; result=r.read_plan(plan,source,valid); }
            return outputs(std::move(result),r.device_id());
        },"plan"_a,"source"_a,"valid"_a=false)
        .def("prepare_points",[](Reader& r,Host points,const Index& base,int kind,int order) {
            const auto dtype=points.dtype();
            if (points.ndim()!=2 || points.shape(1)!=3 || dtype.lanes!=1 ||
                (kind==2 ? (dtype.code!=std::uint8_t(nb::dlpack::dtype_code::Float) || dtype.bits!=32) :
                           (dtype.code!=std::uint8_t(nb::dlpack::dtype_code::Int) || (dtype.bits!=32 && dtype.bits!=64)))) throw nb::value_error("invalid point array");
            nb::gil_scoped_release release;
            return r.prepare_points(points.data(),points.shape(0),dtype.bits,base,kind,order);
        },nb::keep_alive<0,1>())
        .def("grids",[](Reader& r,const std::vector<std::array<std::int64_t,3>>& shapes,
            const std::vector<std::array<std::int64_t,3>>& bases,const std::vector<std::array<float,3>>& origins,
            const std::vector<std::array<float,9>>& axes,int kind,int order,bool floating,const std::string& route,bool valid,nb::object out) -> nb::object {
            if (shapes.size()!=bases.size() || shapes.size()!=origins.size() || shapes.size()!=axes.size()) throw nb::value_error("grid counts differ");
            std::vector<Grid> grids(shapes.size());
            for (std::size_t i=0;i<grids.size();++i) { auto& g=grids[i]; std::copy_n(shapes[i].begin(),3,g.shape); std::copy_n(bases[i].begin(),3,g.base); std::copy_n(origins[i].begin(),3,g.origin); std::copy_n(axes[i].begin(),9,g.axes); }
            device::Storage destination;
            if (!out.is_none()) {
                const auto a=nb::cast<Device>(out);
                const auto expected=floating ? nb::dtype<float>() : nb::dtype<std::uint8_t>();
                if (a.device_id()!=r.device_id() || a.dtype()!=expected) throw nb::value_error("output dtype/device mismatch");
                destination=device::borrow(a.data(),a.nbytes(),a.device_id());
            }
            Result result;
            { nb::gil_scoped_release release;
              try { result=r.grids(std::move(grids),kind,order,floating,route,valid,std::move(destination)); }
              catch (...) {
                  // Keep borrowed output storage alive until outstanding writes finish.
                  device::Guard guard(r.device_id());
                  cudaStreamSynchronize(device::stream());
                  throw;
              } }
            if (!out.is_none()) {
                if (!valid) return nb::none();
                return owned(std::move(result.valid),result.count,nb::dtype<std::uint8_t>(),r.device_id());
            }
            return outputs(std::move(result),r.device_id());
        },"shapes"_a,"bases"_a,"origins"_a,"axes"_a,"kind"_a,"order"_a,"floating"_a,"route"_a,"valid"_a=false,"out"_a=nb::none())
        .def("warm",[make_grids](Reader& r,const std::vector<std::array<std::int64_t,3>>& shapes,const std::vector<std::array<std::int64_t,3>>& bases) {
            auto grids=make_grids(shapes,bases,std::vector<std::array<float,3>>(shapes.size()),std::vector<std::array<float,9>>(shapes.size()));
            nb::gil_scoped_release release; r.warm(std::move(grids));
        })
        .def("points_resident",[point_dtype](Reader& r,Device points,const Index& base,int kind,int order,bool valid,bool sync) {
            const int bits=point_dtype(points,kind);
            if (points.device_id()!=r.device_id()) throw nb::value_error("points are on another CUDA device");
            Result result; { nb::gil_scoped_release release; result=r.points_resident(points.data(),points.shape(0),bits,base,kind,order,valid,sync); }
            return outputs(std::move(result),r.device_id());
        })
        .def("affine_resident",[](Reader& r,Device origins,Device axes,const std::array<std::int64_t,3>& shape,
            std::optional<std::array<std::int64_t,3>> coordinate_origin,int order,bool floating,bool valid,bool sync) {
            const auto od=origins.dtype(), ad=axes.dtype();
            if (origins.ndim()!=2 || origins.shape(1)!=3 || od.code!=std::uint8_t(nb::dlpack::dtype_code::Float) || (od.bits!=32 && od.bits!=64))
                throw nb::value_error("origins must be a (B, 3) float32/float64 CUDA array");
            if (axes.ndim()!=3 || axes.shape(0)!=origins.shape(0) || axes.shape(1)!=3 || axes.shape(2)!=3 || ad.code!=std::uint8_t(nb::dlpack::dtype_code::Float) || ad.bits!=32)
                throw nb::value_error("axes must be a (B, 3, 3) float32 CUDA array");
            if (origins.device_id()!=r.device_id() || axes.device_id()!=r.device_id()) throw nb::value_error("frames are on another CUDA device");
            Result result;
            { nb::gil_scoped_release release;
              result=r.affine_resident(origins.data(),int(od.bits),static_cast<const float*>(axes.data()),origins.shape(0),shape.data(),
                                       coordinate_origin ? coordinate_origin->data() : nullptr,order,floating,valid,sync); }
            return outputs(std::move(result),r.device_id());
        })
        .def("points",[](Reader& r,Host points,const Index& base,int kind,int order,const std::string& route,bool valid) {
            const auto dtype=points.dtype();
            if (points.ndim()!=2 || points.shape(1)!=3 || dtype.lanes!=1 ||
                (kind==2 ? (dtype.code!=std::uint8_t(nb::dlpack::dtype_code::Float) || dtype.bits!=32) :
                           (dtype.code!=std::uint8_t(nb::dlpack::dtype_code::Int) || (dtype.bits!=32 && dtype.bits!=64)))) throw nb::value_error("invalid point array");
            Result result; { nb::gil_scoped_release release; result=r.points(points.data(),points.shape(0),dtype.bits,base,kind,order,route,valid); }
            return outputs(std::move(result),r.device_id());
        },"points"_a,"base"_a,"kind"_a,"order"_a,"route"_a,"valid"_a=false);
}
}
