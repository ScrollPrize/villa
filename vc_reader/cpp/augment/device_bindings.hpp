#pragma once
#include "cuda/device_io.hpp"
#include "../cuda_buffer.hpp"
using DeviceInputArray = nb::ndarray<nb::array_api, nb::c_contig, nb::device::cuda>;
static nb::tuple run_device(const Pipeline& pipeline, const std::map<std::string,DeviceInputArray>& inputs,
                           std::uint64_t seed, bool record, bool profile, bool inplace, const Plan* plan=nullptr) {
    require(pipeline.backend()=="cuda","device inputs require a CUDA pipeline");
    require(!inputs.empty(),"sample needs device arrays");
    const int device=inputs.begin()->second.device_id();
    Sample sample;
    std::map<std::string,cuda::DeviceInput> imported;
    for (const auto& [key,in]:inputs) {
        require(pipeline.targets().count(key),"undeclared array field: "+key);
        require(in.device_id()==device,"all arrays must be on the same CUDA device");
        const auto dt=in.dtype();
        const char kind=dt.code==std::uint8_t(nb::dlpack::dtype_code::Float) ? 'f' : dt.code==std::uint8_t(nb::dlpack::dtype_code::Int) ? 'i' : dt.code==std::uint8_t(nb::dlpack::dtype_code::Bool) ? 'b' : dt.code==std::uint8_t(nb::dlpack::dtype_code::UInt) ? 'u' : '?';
        const auto bytes=dt.bits/8;
        require(dt.lanes==1 && ((kind=='f' && (bytes==4 || bytes==8)) || ((kind=='i' || kind=='u') && (bytes==1 || bytes==2 || bytes==4 || bytes==8)) || (kind=='b' && bytes==1)),"unsupported device dtype");
        require(reinterpret_cast<std::uintptr_t>(in.data())%bytes==0,"unaligned device input");
        Shape shape; for (std::size_t a=0;a<in.ndim();++a) shape.push_back(in.shape(a));
        Array metadata; metadata.shape=shape; metadata.kind=kind; metadata.bytes=bytes;
        require(metadata.size()<=SIZE_MAX/bytes,"array size overflow");
        sample.emplace(key,Field{std::move(metadata),pipeline.targets().at(key),key});
        imported.emplace(key,cuda::DeviceInput{in.data(),std::move(shape),kind,std::size_t(bytes),device});
    }
    if (inplace) for (auto i=inputs.begin();i!=inputs.end();++i) for (auto j=inputs.begin();j!=i;++j) {
        auto a=reinterpret_cast<std::uintptr_t>(i->second.data()), b=reinterpret_cast<std::uintptr_t>(j->second.data());
        require(a+i->second.nbytes()<=b || b+j->second.nbytes()<=a,"inplace augmentation requires nonoverlapping arrays");
    }
    std::shared_ptr<void> context;
    RunResult result;
    { nb::gil_scoped_release release; vcr::device::Guard guard(device);
      validate(sample,false); context=cuda::begin_device(imported,sample,inplace);
      result=plan ? pipeline.replay(std::move(sample),*plan,context) : pipeline.run(std::move(sample),seed,record,profile,context); }
    // Retain input owners for inplace views and the session for newly allocated
    // outputs. The capsule lifetime also works through downstream DLPack users.
    auto* shared=new std::shared_ptr<void>(std::move(context));
    nb::capsule capsule(shared,[](void* p) noexcept { delete static_cast<std::shared_ptr<void>*>(p); });
    nb::list input_owners;
    for (const auto& [key,in]:inputs) input_owners.append(nb::cast(in,nb::rv_policy::reference));
    nb::object owner=nb::make_tuple(capsule,input_owners);
    nb::dict arrays,sources;
    for (const auto& [key,a]:cuda::device_outputs(shared->get())) {
        nb::dlpack::dtype dtype{std::uint8_t(a.kind=='f' ? nb::dlpack::dtype_code::Float : a.kind=='i' ? nb::dlpack::dtype_code::Int : a.kind=='b' ? nb::dlpack::dtype_code::Bool : nb::dlpack::dtype_code::UInt),std::uint8_t(a.bytes*8),1};
        auto array=nb::ndarray<nb::array_api, nb::device::cuda>(a.data,a.shape.size(),a.shape.data(),owner,nullptr,dtype,nb::device::cuda::value,a.device);
        arrays[nb::str(key.c_str())]=nb::make_tuple(array,a.shape,std::string(1,a.kind)+std::to_string(a.bytes),a.device);
        sources[nb::str(key.c_str())]=nb::str(result.sample.at(key).source.c_str());
    }
    return nb::make_tuple(arrays,sources,nb::cast(result.timings),record ? nb::cast(std::move(result.plan)) : nb::none());
}
