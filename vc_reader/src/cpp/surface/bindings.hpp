#pragma once
#include "plan.hpp"
namespace vcr::surface {
using Floats=nb::ndarray<float,nb::c_contig,nb::device::cpu>;
using Integers=nb::ndarray<std::int64_t,nb::c_contig,nb::device::cpu>;
inline Raster configure(const Plan& p,std::array<float,6> mapping,std::array<std::int64_t,2> origin,
    std::array<std::int64_t,2> shape,int channels){
    if(shape[0]<0||shape[1]<0||shape[0]>(1<<24)||shape[1]>(1<<24)||channels<1)throw nb::value_error("invalid label shape");
    for(float x:mapping)if(!std::isfinite(x))throw nb::value_error("nonfinite UV mapping");
    auto g=p.grid;g.channels=channels;g.label_h=int(shape[0]);g.label_w=int(shape[1]);
    std::copy(mapping.begin(),mapping.end(),g.uv_to_label);std::copy(origin.begin(),origin.end(),g.label_origin);
    return g;
}
inline void check(const Raster& g,std::size_t item,std::size_t label,std::size_t out,std::size_t coverage,
                  std::size_t conflict,std::size_t distance,std::size_t winner,std::size_t uv,std::size_t geometry){
    if(label!=std::size_t(g.channels)*g.label_h*g.label_w*item||out!=g.count*g.channels*item||coverage!=g.count||
       conflict!=g.count||distance!=g.count*4||winner!=g.count*8||uv!=g.count*8||geometry!=g.count)throw nb::value_error("raster buffer size mismatch");
}
inline void bind(nb::module_& m){
    nb::class_<Raster>(m,"SurfaceRaster");
    nb::class_<Plan>(m,"SurfacePlan")
      .def("__init__",[](Plan* self,Floats triangles,Integers ids,aligned::Triple shape,std::array<float,12> metric,
                         float radius,std::array<float,3> expansion,std::array<float,3> origin,std::size_t budget){
        if(triangles.ndim()!=3||triangles.shape(1)!=3||triangles.shape(2)!=5||ids.size()!=triangles.shape(0))throw nb::value_error("triangles require (N,3,5) and N IDs");
        new(self) Plan(triangles.data(),ids.data(),ids.size(),shape,metric,radius,expansion,origin,budget);
      })
      .def_prop_ro("nbytes",[](const Plan& p){return sizeof(p)+p.triangles.size()*4+p.ids.size()*8+p.offsets.size()*8+p.candidates.size()*4;})
      .def_prop_ro("upload_bytes",[](const Plan& p){return p.triangles.size()*4+p.ids.size()*8+p.offsets.size()*8+p.candidates.size()*4;})
      .def("configure",&configure)
      .def("cpu",[](const Plan& p,Raster g,Bytes label,const std::string& kind,std::size_t item,
                    Bytes out,Bytes coverage,Bytes conflict,Bytes distance,Bytes winner,Bytes uv,Bytes geometry){
        auto type=sample_type(kind,item,false);if(kind=="f")throw nb::value_error("projection requires categorical integers");
        check(g,item,label.size(),out.size(),coverage.size(),conflict.size(),distance.size(),winner.size(),uv.size(),geometry.size());
        nb::gil_scoped_release release;
        type.visit_load([&](auto load){using T=decltype(load(static_cast<const std::uint8_t*>(nullptr)));
            for(std::size_t i=0;i<g.count;++i)raster(g,p.triangles.data(),p.ids.data(),p.offsets.data(),p.candidates.data(),
                reinterpret_cast<const T*>(label.data()),reinterpret_cast<T*>(out.data()),reinterpret_cast<bool*>(coverage.data()),
                reinterpret_cast<bool*>(conflict.data()),reinterpret_cast<float*>(distance.data()),reinterpret_cast<std::int64_t*>(winner.data()),reinterpret_cast<float*>(uv.data()),reinterpret_cast<bool*>(geometry.data()),i);
        });
      });
    m.def("_surface_interpolate",[](Floats xyz,Bytes mask,Floats queries,Floats normals,Floats output,Bytes valid,
         std::array<float,12> physical,std::array<float,12> destination,float orientation,float offset,int device){
        if(xyz.ndim()!=3||xyz.shape(2)!=3||queries.ndim()!=2||queries.shape(1)!=2||
           mask.size()!=xyz.shape(0)*xyz.shape(1)||output.size()!=queries.shape(0)*3||valid.size()!=queries.shape(0)||
           (normals.size()&&normals.size()!=xyz.size()))throw nb::value_error("surface interpolation shape mismatch");
        Interpolation g;g.height=int(xyz.shape(0));g.width=int(xyz.shape(1));g.orientation=orientation;g.offset=offset;
        std::copy(physical.begin(),physical.end(),g.physical);std::copy(destination.begin(),destination.end(),g.destination);
        nb::gil_scoped_release release;
        if(device<0){for(std::size_t i=0;i<valid.size();++i)interpolate(g,xyz.data(),reinterpret_cast<bool*>(mask.data()),queries.data(),
            normals.size()?normals.data():nullptr,output.data(),reinterpret_cast<bool*>(valid.data()),i);}
        else{
#ifdef VC_READER_CUDA
            interpolate_cuda(g,xyz.data(),reinterpret_cast<bool*>(mask.data()),queries.data(),normals.size()?normals.data():nullptr,
                             output.data(),reinterpret_cast<bool*>(valid.data()),valid.size(),device);
#else
            throw std::runtime_error("CUDA surface sampling requires a CUDA build");
#endif
        }
    });
#ifdef VC_READER_CUDA
    using Device=nb::ndarray<nb::array_api,nb::c_contig,nb::device::cuda>;
    m.def("_surface_region_copy_cuda",[](Device src,Device dst,aligned::Triple shape,aligned::Triple output,
                                        aligned::Triple offset,int channels,std::size_t item){
        std::size_t n=channels,m=channels;
        if(channels<1||!item||src.device_id()!=dst.device_id())throw nb::value_error("invalid region copy");
        for(int a=0;a<3;++a){
            if(shape[a]<0||output[a]<0||shape[a]>output[a]||offset[a]<0||offset[a]>output[a]-shape[a])throw nb::value_error("invalid copy region");
            n*=shape[a];m*=output[a];
        }
        if(n*item!=src.nbytes()||m*item!=dst.nbytes())throw nb::value_error("copy buffer mismatch");
        nb::gil_scoped_release release;device::Guard guard(dst.device_id());
        copy_region_cuda(src.data(),dst.data(),shape,output,offset,channels,item);
    });
    m.def("_surface_raster_cuda",[](const Plan& p,Raster g,Device label,const std::string& kind,std::size_t item,
         Device out,Device coverage,Device conflict,Device distance,Device winner,Device uv,Device geometry){
        auto type=sample_type(kind,item,false);if(kind=="f")throw nb::value_error("projection requires categorical integers");
        check(g,item,label.nbytes(),out.nbytes(),coverage.nbytes(),conflict.nbytes(),distance.nbytes(),winner.nbytes(),uv.nbytes(),geometry.nbytes());
        for(const Device* a:std::initializer_list<const Device*>{&label,&out,&coverage,&conflict,&distance,&winner,&uv,&geometry})if(a->device_id()!=out.device_id())throw nb::value_error("device mismatch");
        nb::gil_scoped_release release;device::Guard guard(out.device_id());
        launch(p,g,label.data(),type,out.data(),static_cast<bool*>(coverage.data()),static_cast<bool*>(conflict.data()),
               static_cast<float*>(distance.data()),static_cast<std::int64_t*>(winner.data()),static_cast<float*>(uv.data()),static_cast<bool*>(geometry.data()),out.device_id());
    });
    m.def("_surface_copy_cuda",[](Device src,Device dst,std::size_t rows,std::size_t source_width,std::size_t dest_width,
                                std::size_t offset,std::size_t item){
        if(src.device_id()!=dst.device_id()||!item||source_width>dest_width||offset>dest_width-source_width||
           rows*source_width*item!=src.nbytes()||rows*dest_width*item!=dst.nbytes())throw nb::value_error("copy shape mismatch");
        nb::gil_scoped_release release;device::Guard guard(dst.device_id());copy_cuda(src.data(),dst.data(),rows,source_width,dest_width,offset,item,dst.device_id());
    });
#endif
}
}
