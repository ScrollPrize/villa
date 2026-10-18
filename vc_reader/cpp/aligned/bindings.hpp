#pragma once
#include "plan.hpp"
#include <nanobind/stl/array.h>
namespace vcr::aligned {
inline Field make_field(std::array<float,9> matrix,std::array<float,3> translation,std::array<int,3> components,bool normalize,bool normal) {
    auto sorted=components; std::sort(sorted.begin(),sorted.end());
    if(sorted!=std::array<int,3>{0,1,2}) throw nb::value_error("component order must permute XYZ");
    Field f;
    std::copy(matrix.begin(),matrix.end(),f.matrix); std::copy(translation.begin(),translation.end(),f.translation);
    std::copy(components.begin(),components.end(),f.component); f.normalize=normalize; f.normal=normal; f.enabled=true;
    return f;
}
inline void check_bytes(const Plan& p,std::size_t output,std::size_t valid,std::size_t item) {
    if(output!=p.grid.output_count*p.grid.channels*(p.grid.order ? 4 : item) || valid!=p.grid.output_count)
        throw nb::value_error("aligned output buffer size mismatch");
}
inline void bind(nb::module_& m) {
    nb::class_<Field>(m,"AlignedField").def(nb::init<>());
    m.def("_aligned_make_field", &make_field);
    nb::class_<Plan>(m,"AlignedPlan")
        .def(nb::init<Triple,Triple,Triple,Triple,Triple,std::array<float,3>,std::array<float,9>,Triple,int,int,bool,bool>())
        .def("starts",&Plan::starts)
        .def_prop_ro("nbytes",[](const Plan& p) { return sizeof(p)+p.keys.size()*sizeof(std::int64_t); })
        .def("cpu",[](const Plan& p,const std::vector<Bytes>& inputs,Bytes fill,const std::string& kind,std::size_t item,Bytes out,Bytes valid,Field field) {
            if(field.enabled && (p.grid.channels!=3 || p.grid.order!=1)) throw nb::value_error("geometric fields require three continuous channels");
            const auto type=sample_type(kind,item,false); check_bytes(p,out.size(),valid.size(),item);
            if(inputs.size()!=p.keys.size() || fill.size()!=item) throw nb::value_error("aligned tile count/fill mismatch");
            std::vector<const void*> ptrs;
            for(auto a:inputs) { if(a.size()!=p.grid.tile_count*p.grid.channels*item) throw nb::value_error("tile size mismatch"); ptrs.push_back(a.data()); }
            nb::gil_scoped_release release; p.cpu(ptrs,fill.data(),type,out.data(),reinterpret_cast<bool*>(valid.data()),field);
        })
        .def("valid_cpu",[](const Plan& p,Bytes valid) {
            if(valid.size()!=p.grid.output_count) throw nb::value_error("validity size mismatch");
            nb::gil_scoped_release release;
            for(std::size_t i=0;i<p.grid.count;++i) { valid.data()[output_index(p.grid,i)]=supported(p.grid,i); }
        });
    nb::class_<PointPlan,Plan>(m,"PointPlan")
        .def("__init__",[](PointPlan* self,Triple source,Triple base,
             nb::ndarray<const float,nb::ndim<2>,nb::c_contig,nb::device::cpu> points,
             Triple tile,int channels,int order,bool plane) {
            if(points.shape(1)!=3 || points.shape(0)>32768) throw nb::value_error("point blocks require Nx3, N <= 32768");
            new(self) PointPlan(source,base,points.data(),points.shape(0),tile,channels,order,plane);
        });
    m.def("_aligned_field_cpu",[](Bytes data,Bytes valid,std::array<float,9> matrix,std::array<float,3> translation,
                                 std::array<int,3> components,bool normalize,bool normal) {
        if(data.size()!=valid.size()*3*4) throw nb::value_error("field shape mismatch");
        const auto f=make_field(matrix,translation,components,normalize,normal);
        nb::gil_scoped_release release;
        for(std::size_t i=0;i<valid.size();++i) if(valid.data()[i]) field(reinterpret_cast<float*>(data.data()),reinterpret_cast<bool*>(valid.data()),valid.size(),i,f);
    });
#ifdef VC_READER_CUDA
    using reader_cuda::Owned;
    m.def("_aligned_empty",[](std::size_t count,const std::string& kind,std::size_t item,int id) {
        sample_type(kind,item,false);
        if(count>SIZE_MAX/item) throw nb::value_error("allocation overflow");
        const auto code=kind=="f" ? nb::dlpack::dtype_code::Float : kind=="i" ? nb::dlpack::dtype_code::Int : kind=="b" ? nb::dlpack::dtype_code::Bool : nb::dlpack::dtype_code::UInt;
        return reader_cuda::owned(device::allocate(count*item,id),count,{std::uint8_t(code),std::uint8_t(8*item),1},id);
    });
    m.def("_aligned_cuda",[](const Plan& p,const std::vector<Owned>& inputs,Bytes fill,const std::string& kind,std::size_t item,Owned& out,Owned& valid,Field field) {
        if(field.enabled && (p.grid.channels!=3 || p.grid.order!=1)) throw nb::value_error("geometric fields require three continuous channels");
        const auto type=sample_type(kind,item,false); check_bytes(p,out.storage->bytes,valid.storage->bytes,item);
        if(inputs.size()!=p.keys.size() || fill.size()!=item || out.device!=valid.device) throw nb::value_error("aligned inputs mismatch");
        std::vector<const void*> ptrs;
        for(const auto& a:inputs) { if(a.device!=out.device || a.storage->bytes!=p.grid.tile_count*p.grid.channels*item) throw nb::value_error("tile size/device mismatch"); ptrs.push_back(a.storage->ptr); }
        nb::gil_scoped_release release; device::Guard guard(out.device);
        auto grid=p.grid; grid.field=field;
        launch(grid,p.keys,ptrs,fill.data(),type,out.storage->ptr,static_cast<bool*>(valid.storage->ptr),out.device);
        device::complete();
    });
    m.def("_aligned_cuda_many",[](const std::vector<const Plan*>& plans,const std::vector<std::vector<Owned>>& inputs,
                                  Bytes fill,const std::string& kind,std::size_t item,Owned& out,Owned& valid,Field field) {
        const auto type=sample_type(kind,item,false);
        if(plans.size()!=inputs.size() || plans.size()>64 || fill.size()!=item || out.device!=valid.device)
            throw nb::value_error("invalid aligned batch");
        std::vector<Grid> grids;
        std::vector<std::vector<std::int64_t>> keys;
        std::vector<std::vector<const void*>> pointers;
        for(std::size_t i=0;i<plans.size();++i) {
            const auto& p=*plans[i]; check_bytes(p,out.storage->bytes,valid.storage->bytes,item);
            if(inputs[i].size()!=p.keys.size() || (field.enabled && (p.grid.channels!=3 || p.grid.order!=1)))
                throw nb::value_error("invalid aligned batch channels/tiles");
            auto grid=p.grid; grid.field=field; grids.push_back(grid); keys.push_back(p.keys);
            pointers.emplace_back();
            for(const auto& a:inputs[i]) {
                if(a.device!=out.device || a.storage->bytes!=p.grid.tile_count*p.grid.channels*item)
                    throw nb::value_error("tile size/device mismatch");
                pointers.back().push_back(a.storage->ptr);
            }
        }
        nb::gil_scoped_release release; device::Guard guard(out.device);
        launch_many(grids,keys,pointers,fill.data(),type,out.storage->ptr,static_cast<bool*>(valid.storage->ptr),out.device);
    });
    m.def("_aligned_valid_cuda",[](const Plan& p,Owned& valid) {
        if(valid.storage->bytes!=p.grid.output_count) throw nb::value_error("validity size mismatch");
        nb::gil_scoped_release release; device::Guard guard(valid.device);
        launch_valid(p.grid,static_cast<bool*>(valid.storage->ptr)); device::complete();
    });
    m.def("_aligned_box_valid_cuda",[](Triple source,Triple start,Triple shape,Owned& valid) {
        std::size_t n=1;
        for(int a=0;a<3;++a) {
            if(source[a]<0 || source[a]>(1LL<<40) || start[a]<-(1LL<<60) || start[a]>(1LL<<60) ||
               shape[a]<0 || shape[a]>(1<<24) || (shape[a] && n>SIZE_MAX/std::size_t(shape[a])))
                throw nb::value_error("invalid box shape/origin");
            n*=shape[a];
        }
        if(n!=valid.storage->bytes) throw nb::value_error("validity size mismatch");
        nb::gil_scoped_release release; device::Guard guard(valid.device);
        launch_box(source,start,shape,static_cast<bool*>(valid.storage->ptr)); device::complete();
    });
    m.def("_aligned_field_cuda",[](Owned& data,Owned& valid,std::array<float,9> matrix,std::array<float,3> translation,
                                  std::array<int,3> components,bool normalize,bool normal) {
        if(data.storage->bytes!=valid.storage->bytes*3*4 || data.device!=valid.device) throw nb::value_error("field shape/device mismatch");
        const auto f=make_field(matrix,translation,components,normalize,normal);
        nb::gil_scoped_release release; device::Guard guard(data.device);
        launch_field(static_cast<float*>(data.storage->ptr),static_cast<bool*>(valid.storage->ptr),valid.storage->bytes,f); device::complete();
    });
#endif
}
}
