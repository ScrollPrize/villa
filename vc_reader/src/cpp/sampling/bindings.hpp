#pragma once
#include "vc_reader/resident_sampling.hpp"
namespace vcr::sampling::python {
using geometry::Array;using geometry::check;using geometry::array_device;
inline Dense dense(Array values,Array points,int order,const std::string& boundary,float fill){
    if(values.ndim()<4||values.ndim()>5||points.ndim()!=3)throw nb::value_error("expected B,C,[Z,]Y,X values and B,Q,D points");
    int id=array_device(values);Dense s;s.dimensions=int(values.ndim())-2;s.batches=values.shape(0);s.channels=values.shape(1);s.queries=points.shape(1);
    std::vector<std::size_t> shape{values.shape(0),values.shape(1)};for(int a=0;a<s.dimensions;++a){s.shape[a]=values.shape(a+2);shape.push_back(values.shape(a+2));}
    check(values,nb::dtype<float>(),shape,id);check(points,nb::dtype<float>(),{s.batches,s.queries,std::size_t(s.dimensions)},id);
    s.values=static_cast<const float*>(values.data());s.points=static_cast<const float*>(points.data());s.order=order;s.fill=fill;
    if(boundary=="zeros")s.boundary=Boundary::Zeros;else if(boundary=="border")s.boundary=Boundary::Border;else if(boundary=="constant")s.boundary=Boundary::Constant;else throw nb::value_error("boundary must be zeros, border or constant");
    return s;
}
inline void bind(nb::module_& m){
    m.def("_field_tile",[](Array xyz,Array mask,Array derivatives,std::array<int,2> shape,std::array<int,4> bounds,int mode){
        if(xyz.ndim()!=4)throw nb::value_error("expected B,H,W,3 grid");
        auto h=xyz.shape(1),w=xyz.shape(2);auto th=bounds[1]-bounds[0],tw=bounds[3]-bounds[2];
        if(shape[0]<1||shape[1]<1||th<1||tw<1)throw nb::value_error("invalid tile shape");
        check(xyz,nb::dtype<float>(),{1,h,w,3},-1);check(mask,nb::dtype<bool>(),{1,h,w},-1);check(derivatives,nb::dtype<float>(),{1,3,h,w,3},-1);
        std::size_t n=std::size_t(th)*tw;auto* valid=new bool[n];auto* out=new float[n];
        nb::capsule ov(out,[](void* p)noexcept{delete[] static_cast<float*>(p);}),mv(valid,[](void* p)noexcept{delete[] static_cast<bool*>(p);});
        Tile t{int(h),int(w),shape[0],shape[1],bounds[0],bounds[1],bounds[2],bounds[3]};
        {nb::gil_scoped_release release;sampleSurfaceTile(static_cast<const float*>(xyz.data()),static_cast<const bool*>(mask.data()),static_cast<const float*>(derivatives.data()),t,mode,out,valid);}
        std::size_t dims[]{std::size_t(th),std::size_t(tw)};
        return nb::make_tuple(nb::ndarray<nb::numpy,float>(out,2,dims,ov),nb::ndarray<nb::numpy,bool>(valid,2,dims,mv));
    });

    m.def("_linear_field_tile",[](Array field,Array mask,std::array<int,2> shape,std::array<int,4> bounds,Array out,Array valid){
        if(field.ndim()!=2||bounds[1]<=bounds[0]||bounds[3]<=bounds[2])throw nb::value_error("expected 2D field and nonempty tile");
        auto h=field.shape(0),w=field.shape(1);int id=array_device(field);
        check(field,nb::dtype<float>(),{h,w},id);check(mask,nb::dtype<bool>(),{h,w},id);
        auto th=std::size_t(bounds[1]-bounds[0]),tw=std::size_t(bounds[3]-bounds[2]);
        check(out,nb::dtype<float>(),{th,tw},id);check(valid,nb::dtype<bool>(),{th,tw},id);geometry::disjoint({field,mask},{out,valid});
        Tile t{int(h),int(w),shape[0],shape[1],bounds[0],bounds[1],bounds[2],bounds[3]};
        nb::gil_scoped_release release;sampleFieldTile(static_cast<const float*>(field.data()),static_cast<const bool*>(mask.data()),t,static_cast<float*>(out.data()),static_cast<bool*>(valid.data()),id);
    });
    m.def("_sample_dense",[](Array values,Array points,Array out,int order,const std::string& boundary,float fill){
        auto s=dense(values,points,order,boundary,fill);int id=array_device(values);check(out,nb::dtype<float>(),{s.batches,s.channels,s.queries},id);
        geometry::disjoint({values,points},{out});nb::gil_scoped_release release;sampleDense(s,static_cast<float*>(out.data()),id);
    });
    m.def("_sample_dense_backward",[](Array values,Array points,Array cot,Array vg,Array pg,int order,const std::string& boundary,float fill){
        auto s=dense(values,points,order,boundary,fill);int id=array_device(values);check(cot,nb::dtype<float>(),{s.batches,s.channels,s.queries},id);
        std::vector<std::size_t> shape;for(std::size_t a=0;a<values.ndim();++a)shape.push_back(values.shape(a));
        if(vg.size())check(vg,nb::dtype<float>(),shape,id);if(pg.size())check(pg,nb::dtype<float>(),{s.batches,s.queries,std::size_t(s.dimensions)},id);
        geometry::disjoint({values,points,cot},{vg,pg});nb::gil_scoped_release release;
        backwardDense(s,static_cast<const float*>(cot.data()),vg.size()?static_cast<float*>(vg.data()):nullptr,pg.size()?static_cast<float*>(pg.data()):nullptr,id);
    });
    m.def("_prepare_surfaces",[](Array xyz,Array mask,Array out){
        if(xyz.ndim()!=4)throw nb::value_error("surface vertices must be B,H,W,3");
        auto b=xyz.shape(0),h=xyz.shape(1),w=xyz.shape(2);int id=array_device(xyz);
        check(xyz,nb::dtype<float>(),{b,h,w,3},id);check(mask,nb::dtype<bool>(),{b,h,w},id);check(out,nb::dtype<float>(),{b,3,h,w,3},id);geometry::disjoint({xyz,mask},{out});
        nb::gil_scoped_release release;prepareSurfaces(static_cast<const float*>(xyz.data()),static_cast<const bool*>(mask.data()),int(b),int(h),int(w),static_cast<float*>(out.data()),id);
    });
    m.def("_sample_surfaces",[](Array xyz,Array mask,Array derivatives,Array rc,Array positions,Array row,Array col,Array ok,int mode,bool border){
        if(xyz.ndim()!=4||rc.ndim()!=3)throw nb::value_error("expected B,H,W,3 vertices and B,Q,2 queries");
        auto b=xyz.shape(0),h=xyz.shape(1),w=xyz.shape(2),q=rc.shape(1);int id=array_device(xyz);
        check(xyz,nb::dtype<float>(),{b,h,w,3},id);check(mask,nb::dtype<bool>(),{b,h,w},id);check(derivatives,nb::dtype<float>(),{b,3,h,w,3},id);check(rc,nb::dtype<float>(),{b,q,2},id);
        for(auto a:{positions,row,col})check(a,nb::dtype<float>(),{b,q,3},id);check(ok,nb::dtype<bool>(),{b,q},id);geometry::disjoint({xyz,mask,derivatives,rc},{positions,row,col,ok});
        nb::gil_scoped_release release;sampleSurfaces(static_cast<const float*>(xyz.data()),static_cast<const bool*>(mask.data()),static_cast<const float*>(derivatives.data()),int(b),int(h),int(w),static_cast<const float*>(rc.data()),q,static_cast<float*>(positions.data()),static_cast<float*>(row.data()),static_cast<float*>(col.data()),static_cast<bool*>(ok.data()),mode,border,id);
    });
}
}
