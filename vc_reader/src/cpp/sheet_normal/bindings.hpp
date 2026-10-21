#pragma once
#include "normal.hpp"
namespace vcr::sheet_normal {
template<class A> inline bool validate(A image,std::size_t batch) {
    auto dt=image.dtype();
    bool floating=dt.code==std::uint8_t(nb::dlpack::dtype_code::Float) && dt.bits==32;
    bool byte=dt.code==std::uint8_t(nb::dlpack::dtype_code::UInt) && dt.bits==8;
    if(dt.lanes!=1 || (!floating && !byte) || image.ndim()!=4 || image.shape(0)!=batch ||
       image.shape(1)!=width || image.shape(2)!=width || image.shape(3)!=width)
        throw nb::value_error("sheet normal patches must be contiguous (B,109,109,109) uint8 or float32 CT in [0,255]");
    return floating;
}
inline void bind(nb::module_& m) {
    using Host=nb::ndarray<nb::ro,nb::c_contig,nb::device::cpu>;
    using Centers=nb::ndarray<const float,nb::ndim<2>,nb::c_contig,nb::device::cpu>;
    auto check=[](Centers c){
        if(c.shape(1)!=4) throw nb::value_error("centers must be (B,4)");
        for(std::size_t b=0;b<c.shape(0);++b) for(int j=0;j<3;++j)
            if(!std::isfinite(c.data()[4*b+j]) || c.data()[4*b+j]<48 || c.data()[4*b+j]>=51)
                throw nb::value_error("center must be in [48,51)");
    };
    const auto check_output=[](auto image,auto out,std::size_t n) {
        if (out.ndim()!=2 || out.shape(0)!=n || out.shape(1)!=4 || out.dtype()!=nb::dtype<float>() || out.device_id()!=image.device_id())
            throw nb::value_error("out must be contiguous (B,4) float32 on the input device");
        auto a=reinterpret_cast<std::uintptr_t>(image.data()),b=reinterpret_cast<std::uintptr_t>(out.data());
        if (image.nbytes() && out.nbytes() && a<b+out.nbytes() && b<a+image.nbytes())
            throw nb::value_error("out must not overlap input patches");
    };
    m.def("_sheet_normals_cpu",[check,check_output](Host image,Centers centers,nb::object destination) -> nb::object {
        check(centers); const auto n=centers.shape(0); bool floating=validate(image,n);
        std::unique_ptr<float[]> out;
        float* data;
        if (destination.is_none()) { out=std::make_unique<float[]>(n*4); data=out.get(); }
        else {
            const auto target=nb::cast<nb::ndarray<float,nb::c_contig,nb::device::cpu>>(destination);
            check_output(image,target,n); data=target.data();
        }
        { nb::gil_scoped_release release;
          if(floating) cpu(static_cast<const float*>(image.data()),centers.data(),data,n);
          else cpu(static_cast<const std::uint8_t*>(image.data()),centers.data(),data,n); }
        if (!destination.is_none()) return destination;
        auto* ptr=out.release(); nb::capsule owner(ptr,[](void* p) noexcept {delete[] static_cast<float*>(p);});
        return nb::cast(nb::ndarray<nb::numpy,float>(ptr,{n,4},owner));
    },"image"_a,"centers"_a,"out"_a=nb::none());
#ifdef VC_READER_CUDA
    using Device=nb::ndarray<nb::array_api,nb::c_contig,nb::device::cuda>;
    m.def("_sheet_normals_cuda",[check,check_output](Device image,Centers centers,nb::object destination) -> nb::object {
        check(centers); const auto n=centers.shape(0); bool floating=validate(image,n);
        device::Storage out;
        if (!destination.is_none()) {
            const auto target=nb::cast<Device>(destination);
            check_output(image,target,n); out=device::borrow(target.data(),target.nbytes(),target.device_id());
        }
        { nb::gil_scoped_release release; device::Guard guard(image.device_id());
          try { out=cuda(image.data(),floating,centers.data(),n,image.device_id(),std::move(out)); }
          catch (...) {
              // External inputs/outputs may be released as soon as we unwind.
              cudaStreamSynchronize(device::stream());
              throw;
          } }
        if (!destination.is_none()) return nb::none();
        return reader_cuda::owned(std::move(out),n*4,nb::dtype<float>(),image.device_id());
    },"image"_a,"centers"_a,"out"_a=nb::none());
#endif
}
}
