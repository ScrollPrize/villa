#pragma once
#include "normal.hpp"
#ifdef VC_READER_CUDA
#include "../cuda_buffer.hpp"
namespace vcr::sheet_normal {
device::Storage cuda(const void* image,bool floating,const float* centers,std::size_t batch,int device);
}
#endif
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
    m.def("_sheet_normals_cpu",[check](Host image,Centers centers) {
        check(centers); const auto n=centers.shape(0); bool floating=validate(image,n);
        auto out=std::make_unique<float[]>(n*4);
        { nb::gil_scoped_release release;
          if(floating) cpu(static_cast<const float*>(image.data()),centers.data(),out.get(),n);
          else cpu(static_cast<const std::uint8_t*>(image.data()),centers.data(),out.get(),n); }
        auto* ptr=out.release(); nb::capsule owner(ptr,[](void* p) noexcept {delete[] static_cast<float*>(p);});
        return nb::ndarray<nb::numpy,float>(ptr,{n,4},owner);
    });
#ifdef VC_READER_CUDA
    using Device=nb::ndarray<nb::array_api,nb::c_contig,nb::device::cuda>;
    m.def("_sheet_normals_cuda",[check](Device image,Centers centers) {
        check(centers); const auto n=centers.shape(0); bool floating=validate(image,n);
        device::Storage out;
        { nb::gil_scoped_release release; device::Guard guard(image.device_id());
          out=cuda(image.data(),floating,centers.data(),n,image.device_id()); }
        return reader_cuda::owned(std::move(out),n*4,nb::dtype<float>(),image.device_id());
    });
#endif
}
}
