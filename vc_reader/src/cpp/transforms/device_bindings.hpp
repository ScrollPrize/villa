#pragma once
#include "cuda/device_run.hpp"
using DeviceInputArray = nb::ndarray<nb::array_api, nb::c_contig, nb::device::cuda>;
// A run's outputs, with the Python objects whose storage its arrays use.
struct BoundOutputs {
    std::shared_ptr<cuda::DeviceOutputs> outputs;
    std::vector<nb::object> working_owners;
    void mark_exported() { outputs->mark_exported(); }
    void check() { outputs->check(); }
};
static char dlpack_kind(nb::dlpack::dtype dt) {
    return dt.code==std::uint8_t(nb::dlpack::dtype_code::Float) ? 'f' : dt.code==std::uint8_t(nb::dlpack::dtype_code::Int) ? 'i' :
           dt.code==std::uint8_t(nb::dlpack::dtype_code::Bool) ? 'b' : dt.code==std::uint8_t(nb::dlpack::dtype_code::UInt) ? 'u' : '?';
}
static nb::dlpack::dtype dlpack_dtype(char kind, std::size_t bytes) {
    return {std::uint8_t(kind=='f' ? nb::dlpack::dtype_code::Float : kind=='i' ? nb::dlpack::dtype_code::Int : kind=='b' ? nb::dlpack::dtype_code::Bool : nb::dlpack::dtype_code::UInt),std::uint8_t(bytes*8),1};
}
static cuda::DeviceInput device_input(const DeviceInputArray& in) {
    const auto dt=in.dtype();
    require(dt.lanes==1,"unsupported device dtype");
    Shape shape; for (std::size_t a=0;a<in.ndim();++a) shape.push_back(in.shape(a));
    return {in.data(),std::move(shape),dlpack_kind(dt),std::size_t(dt.bits/8),in.device_id()};
}
static std::map<std::string,cuda::DeviceInput> device_inputs(const std::map<std::string,DeviceInputArray>& inputs) {
    std::map<std::string,cuda::DeviceInput> out;
    for (const auto& [key,in]:inputs) out.emplace(key,device_input(in));
    return out;
}
static nb::object device_array(void* data, const Shape& shape, char kind, std::size_t bytes, int device, nb::handle owner) {
    return nb::cast(nb::ndarray<nb::array_api, nb::device::cuda>(data,shape.size(),shape.data(),owner,nullptr,dlpack_dtype(kind,bytes),nb::device::cuda::value,device));
}
static nb::tuple run_device(const Pipeline& pipeline, const std::map<std::string,DeviceInputArray>& inputs,
                           std::uint64_t seed, bool record, bool inplace, const Plan* plan=nullptr,
                           const NamedValues& params={}) {
    const auto native=device_inputs(inputs);
    cuda::DeviceRun run;
    { nb::gil_scoped_release release; run=cuda::run_device(pipeline,native,seed,record,inplace,plan,params); }
    // The owner keeps the session (outputs) and, for inplace views, the inputs alive.
    auto shared=std::make_shared<cuda::DeviceOutputs>(); shared->sessions={run.session}; shared->device=run.device;
    nb::object outputs=nb::cast(BoundOutputs{shared,{}});
    nb::list input_owners;
    for (const auto& [key,in]:inputs) input_owners.append(nb::cast(in,nb::rv_policy::reference));
    nb::object owner=nb::make_tuple(outputs,input_owners);
    nb::dict arrays,sources;
    for (const auto& [key,a]:cuda::device_outputs(run.session.get())) {
        arrays[nb::str(key.c_str())]=nb::make_tuple(device_array(a.data,a.shape,a.kind,a.bytes,a.device,owner),a.shape,std::string(1,a.kind)+std::to_string(a.bytes),a.device);
        sources[nb::str(key.c_str())]=nb::str(run.result.sample.at(key).source.c_str());
    }
    for (const auto& [key,levels]:cuda::device_pyramids(run.session.get())) {
        nb::list items;
        for (const auto& a:levels) items.append(nb::make_tuple(device_array(a.data,a.shape,a.kind,a.bytes,a.device,owner),a.shape,std::string(1,a.kind)+std::to_string(a.bytes),a.device));
        arrays[nb::str(key.c_str())]=items;
    }
    return nb::make_tuple(arrays,sources,record ? nb::cast(std::move(run.result.plan)) : nb::none(),outputs);
}
// `allocator(shape, dtype, device)` is Python (a stream-aware tensor allocator): native storage
// comes from it, so each array handed back is the object it returned.
static nb::tuple run_batch(const Pipeline& pipeline, const std::map<std::string,DeviceInputArray>& inputs,
                          const std::vector<std::uint64_t>& seeds, const std::map<std::string,std::vector<double>>& params,
                          nb::object allocator, std::uintptr_t stream) {
    vcr::device::StreamScope scope(vcr::device::stream_handle(stream));
    std::map<void*,nb::object> allocated;
    const auto allocate=[&](const Shape& shape,char kind,std::size_t bytes,int device) {
        nb::gil_scoped_acquire acquire;
        auto result=allocator(nb::cast(shape),std::string(1,kind)+std::to_string(bytes),device);
        auto dest=device_input(nb::cast<DeviceInputArray>(result));
        allocated.insert_or_assign(dest.data,std::move(result));
        return dest;
    };
    const auto native=device_inputs(inputs);
    cuda::BatchRun run;
    { nb::gil_scoped_release release; run=cuda::run_batch(pipeline,native,seeds,params,allocate); }
    const auto packed=[&](const cuda::BatchArray& value) -> nb::object {
        if (value.stacked) return allocated.at(value.arrays.front().data);
        nb::list items;
        for (const auto& a:value.arrays) items.append(allocated.at(a.data));
        return items;
    };
    nb::dict arrays,names;
    for (const auto& [key,value]:run.arrays) arrays[nb::str(key.c_str())]=packed(value);
    for (const auto& [key,levels]:run.pyramids) {
        nb::list items;
        for (const auto& level:levels) items.append(packed(level));
        arrays[nb::str(key.c_str())]=items;
    }
    for (const auto& [key,value]:run.sources) names[nb::str(key.c_str())]=nb::str(value.c_str());
    BoundOutputs holder{run.outputs,{}};
    for (auto& [data,object]:allocated) holder.working_owners.push_back(std::move(object));
    return nb::make_tuple(arrays,names,nb::cast(std::move(holder)));
}
