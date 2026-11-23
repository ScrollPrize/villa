#pragma once
#include "normal.hpp"
#include "sample.hpp"
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
    using Points=nb::ndarray<const double,nb::ndim<2>,nb::c_contig,nb::device::cpu>;
    using Shape=std::array<std::int64_t,3>;
    const auto check_points=[](Points points,const Shape& shape) {
        if(points.shape(1)!=3) throw nb::value_error("points_xyz must be finite (B,3)");
        for(auto n:shape) if(n<0) throw nb::value_error("shape_zyx must have three nonnegative dimensions");
    };
    m.def("_sheet_normal_regions",[check_points](Points points,Shape shape) {
        check_points(points,shape);
        const auto n=points.shape(0);
        auto starts=std::make_unique<std::int64_t[]>(n*3);
        auto offsets=std::make_unique<float[]>(n*3);
        auto valid=std::make_unique<bool[]>(n);
        for(std::size_t i=0;i<n;++i) region(points.data()+3*i,shape.data(),starts.get()+3*i,offsets.get()+3*i,valid.get()+i);
        auto* s=starts.release();auto* o=offsets.release();auto* v=valid.release();
        nb::capsule so(s,[](void* p) noexcept {delete[] static_cast<std::int64_t*>(p);});
        nb::capsule oo(o,[](void* p) noexcept {delete[] static_cast<float*>(p);});
        nb::capsule vo(v,[](void* p) noexcept {delete[] static_cast<bool*>(p);});
        return nb::make_tuple(nb::ndarray<nb::numpy,std::int64_t>(s,{n,std::size_t(3)},so),
                              nb::ndarray<nb::numpy,float>(o,{n,std::size_t(3)},oo),
                              nb::ndarray<nb::numpy,bool>(v,{n},vo));
    },"points"_a,"shape"_a);
    m.def("_sheet_normal_centers",[](nb::ndarray<const float,nb::ndim<2>,nb::c_contig,nb::device::cpu> offsets,nb::ndarray<const bool,nb::ndim<1>,nb::c_contig,nb::device::cpu> valid) {
        const auto n=offsets.shape(0);
        if(offsets.shape(1)!=3 || valid.shape(0)!=n) throw nb::value_error("offsets_xyz must be finite (B,3) values in [0,3)");
        auto out=std::make_unique<float[]>(n*4);
        for(std::size_t i=0;i<n;++i) center(offsets.data()+3*i,valid.data()[i],out.get()+4*i);
        auto* ptr=out.release();nb::capsule owner(ptr,[](void* p) noexcept {delete[] static_cast<float*>(p);});
        return nb::ndarray<nb::numpy,float>(ptr,{n,std::size_t(4)},owner);
    },"offsets"_a,"valid"_a);
    m.def("_sample_sheet_normals",[](zarr::OpenedArray& array,Points points) {
        if(points.shape(1)!=3) throw nb::value_error("points_xyz must be finite (B,3)");
        const auto n=points.shape(0);
        auto out=std::make_unique<float[]>(n*4);
        { nb::gil_scoped_release release; sheetNormals(array,points.data(),n,out.get()); }
        auto* ptr=out.release();nb::capsule owner(ptr,[](void* p) noexcept {delete[] static_cast<float*>(p);});
        return nb::ndarray<nb::numpy,float>(ptr,{n,std::size_t(4)},owner);
    },"array"_a,"points"_a);
#ifdef VC_READER_CUDA
    m.def("_sample_sheet_normals_cuda",[check_points](reader_cuda::Reader& reader,Shape shape,Points points) {
        check_points(points,shape);
        const auto n=points.shape(0);
        device::Storage out;
        { nb::gil_scoped_release release;
          out=sample_cuda(reader,shape.data(),points.data(),n,reader.device_id()); }
        return reader_cuda::owned(std::move(out),n*4,nb::dtype<float>(),reader.device_id());
    },"reader"_a,"shape"_a,"points"_a);
#endif
    m.attr("SHEET_NORMAL_RECIPE")=recipe;
    auto check=[](Centers c){ if(c.shape(1)!=4) throw nb::value_error("centers must be (B,4)"); };
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
