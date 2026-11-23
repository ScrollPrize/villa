#pragma once
#include "vc_reader/slab.hpp"
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
namespace vcr::display::python {
inline void bind(nb::module_& m) {
    using Projection=SlabProjection;
    nb::class_<Projection>(m,"SlabProjection")
        .def(nb::init<>())
        .def_rw("perspective",&Projection::perspective)
        .def_rw("m00",&Projection::m00).def_rw("m01",&Projection::m01)
        .def_rw("m10",&Projection::m10).def_rw("m11",&Projection::m11)
        .def_rw("center_u",&Projection::centerU).def_rw("center_v",&Projection::centerV)
        .def_prop_rw("position",[](const Projection& p){return std::array<float,3>{p.pos[0],p.pos[1],p.pos[2]};},
            [](Projection& p,const std::array<float,3>& v){std::copy(v.begin(),v.end(),p.pos);})
        .def_prop_rw("ray_base",[](const Projection& p){return std::array<float,3>{p.rayBase[0],p.rayBase[1],p.rayBase[2]};},
            [](Projection& p,const std::array<float,3>& v){std::copy(v.begin(),v.end(),p.rayBase);})
        .def_prop_rw("e1_over_f",[](const Projection& p){return std::array<float,3>{p.e1OverF[0],p.e1OverF[1],p.e1OverF[2]};},
            [](Projection& p,const std::array<float,3>& v){std::copy(v.begin(),v.end(),p.e1OverF);})
        .def_prop_rw("e2_over_f",[](const Projection& p){return std::array<float,3>{p.e2OverF[0],p.e2OverF[1],p.e2OverF[2]};},
            [](Projection& p,const std::array<float,3>& v){std::copy(v.begin(),v.end(),p.e2OverF);});
    m.def("opacity_lut",[](float low,float high,float opacity,float gamma,uint8_t cutoff,float length,bool cuda) {
        nb::gil_scoped_release release;
        if(cuda) {
#ifdef VC_READER_CUDA
            return display::cuda::opacityLut(low,high,opacity,gamma,cutoff,length);
#else
            throw std::runtime_error("CUDA support is not built");
#endif
        }
        return display::opacityLut(low,high,opacity,gamma,cutoff,length);
    },"low"_a,"high"_a,"opacity"_a,"gamma"_a=1.f,"cutoff"_a=0,"segment_length"_a=1.f,"cuda"_a=false);
    // The Python host-array convenience entry point transfers the whole slab for CUDA.
    // C++ callers with resident device buffers use compositeSlab directly.
    m.def("composite_slab",[](nb::ndarray<const uint8_t,nb::ndim<3>,nb::c_contig,nb::device::cpu> values,
                              nb::ndarray<const uint8_t,nb::ndim<3>,nb::c_contig,nb::device::cpu> mask,
                              const std::array<uint32_t,256>& colors,const std::array<float,256>& opacity,
                              Projection projection,
                              nb::ndarray<const float,nb::ndim<2>,nb::c_contig,nb::device::cpu> offsets,
                              nb::ndarray<const float,nb::ndim<1>,nb::c_contig,nb::device::cpu> layerNum,
                              std::array<int,4> margins,float lighting,bool cuda) {
        for(int i=0;i<3;++i)if(values.shape(i)!=mask.shape(i))throw nb::value_error("slab mask shape differs");
        if(!values.shape(0) || offsets.shape(0)!=values.shape(0) || offsets.shape(1)!=2 || layerNum.size()!=values.shape(0))
            throw nb::value_error("slab requires per-layer offsets and depths");
        const int count=int(values.shape(0)),rows=int(values.shape(1)),cols=int(values.shape(2));
        const int width=cols-margins[0]-margins[2],height=rows-margins[1]-margins[3];
        if(width<=0 || height<=0 || *std::min_element(margins.begin(),margins.end())<0)throw nb::value_error("invalid slab margins");
        const std::size_t n=std::size_t(width)*height;
        auto rgbOwner=std::make_unique<uint8_t[]>(n*3),maskOwner=std::make_unique<uint8_t[]>(n);
        auto* rgb=rgbOwner.get();auto* coverage=maskOwner.get();
        {
            nb::gil_scoped_release release;
            std::vector<SlabLayer> layers;
            for(int i=0;i<count;++i)layers.push_back({values.data()+std::size_t(i)*rows*cols,mask.data()+std::size_t(i)*rows*cols,std::size_t(cols),std::size_t(cols)});
            projection.layerOffsets=offsets.data();projection.layerNum=layerNum.data();
            Slab slab{layers.data(),count,rows,cols,margins[0],margins[1],width,height,projection,colors.data(),opacity.data(),lighting};
            if(cuda) {
#ifdef VC_READER_CUDA
                int gpu;device::check(cudaGetDevice(&gpu));
                device::Buffer v(values.size(),gpu),c(mask.size(),gpu),ls(layers.size()*sizeof(SlabLayer),gpu),
                    cs(1024,gpu),op(1024,gpu),os(offsets.size()*sizeof(float),gpu),ns(layerNum.size()*sizeof(float),gpu),
                    out(n*3,gpu),cov(n,gpu);
                auto upload=[](void* dst,const void* src,std::size_t bytes){device::check(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyHostToDevice,device::stream()));};
                upload(v.ptr,values.data(),values.size());upload(c.ptr,mask.data(),mask.size());
                for(int i=0;i<count;++i){layers[i].values=static_cast<uint8_t*>(v.ptr)+std::size_t(i)*rows*cols;layers[i].coverage=static_cast<uint8_t*>(c.ptr)+std::size_t(i)*rows*cols;}
                upload(ls.ptr,layers.data(),layers.size()*sizeof(SlabLayer));upload(cs.ptr,colors.data(),1024);upload(op.ptr,opacity.data(),1024);
                upload(os.ptr,offsets.data(),offsets.size()*sizeof(float));upload(ns.ptr,layerNum.data(),layerNum.size()*sizeof(float));
                slab.layers=static_cast<SlabLayer*>(ls.ptr);slab.colors=static_cast<uint32_t*>(cs.ptr);slab.opacity=static_cast<float*>(op.ptr);
                slab.projection.layerOffsets=static_cast<float*>(os.ptr);slab.projection.layerNum=static_cast<float*>(ns.ptr);
                display::cuda::compositeSlab(slab,static_cast<uint8_t*>(out.ptr),static_cast<uint8_t*>(cov.ptr));
                device::check(cudaMemcpyAsync(rgb,out.ptr,n*3,cudaMemcpyDeviceToHost,device::stream()));
                device::check(cudaMemcpyAsync(coverage,cov.ptr,n,cudaMemcpyDeviceToHost,device::stream()));device::complete();
#else
                throw std::runtime_error("CUDA support is not built");
#endif
            } else display::compositeSlab(slab,rgb,coverage);
        }
        nb::capsule rgbCap(rgbOwner.release(),[](void* p)noexcept{delete[] static_cast<uint8_t*>(p);});
        nb::capsule maskCap(maskOwner.release(),[](void* p)noexcept{delete[] static_cast<uint8_t*>(p);});
        const std::size_t shape[]{std::size_t(height),std::size_t(width),3};
        return nb::make_tuple(nb::ndarray<nb::numpy,uint8_t>(rgb,3,shape,rgbCap),nb::ndarray<nb::numpy,uint8_t>(coverage,2,shape,maskCap));
    },"values"_a.noconvert(),"coverage"_a.noconvert(),"colors"_a,"opacity"_a,"projection"_a,"offsets"_a.noconvert(),
        "layer_depths"_a.noconvert(),"margins"_a=std::array<int,4>{},"lighting"_a=0.f,"cuda"_a=false);
}
}
