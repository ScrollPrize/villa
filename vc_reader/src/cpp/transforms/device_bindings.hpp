#pragma once
#include "cuda/device_io.hpp"
#include "cuda/batch_queue.hpp"
#include "../cuda_buffer.hpp"
#include "../cuda_event.hpp"
using DeviceInputArray = nb::ndarray<nb::array_api, nb::c_contig, nb::device::cuda>;
// Owner of one or more device invocations' outputs. Arrays are marked exported
// (their destruction then synchronizes the device) only when shared through
// DLPack; check() raises deferred device checks of asynchronous calls.
struct DeviceOutputs {
    std::vector<std::shared_ptr<void>> sessions;
    std::vector<std::shared_ptr<vcr::device::Buffer>> stacked;
    int device = 0;
    cudaStream_t stream = vcr::device::stream();
    std::vector<nb::object> working_owners;
    void mark_exported() { for (auto& s:sessions) cuda::export_outputs(s.get()); for (auto& b:stacked) b->exported=true; }
    void check() {
        vcr::device::Guard guard(device); vcr::device::StreamScope scope(stream);
        cuda::check_devices(sessions);
    }
};
static char dlpack_kind(nb::dlpack::dtype dt) {
    return dt.code==std::uint8_t(nb::dlpack::dtype_code::Float) ? 'f' : dt.code==std::uint8_t(nb::dlpack::dtype_code::Int) ? 'i' :
           dt.code==std::uint8_t(nb::dlpack::dtype_code::Bool) ? 'b' : dt.code==std::uint8_t(nb::dlpack::dtype_code::UInt) ? 'u' : '?';
}
static nb::dlpack::dtype dlpack_dtype(char kind, std::size_t bytes) {
    return {std::uint8_t(kind=='f' ? nb::dlpack::dtype_code::Float : kind=='i' ? nb::dlpack::dtype_code::Int : kind=='b' ? nb::dlpack::dtype_code::Bool : nb::dlpack::dtype_code::UInt),std::uint8_t(bytes*8),1};
}
// Validated device inputs; `skip` drops leading (batch) axes from each shape.
struct Imported { Sample sample; std::map<std::string,cuda::DeviceInput> arrays; int device; };
static Imported import_device(const Pipeline& pipeline, const std::map<std::string,DeviceInputArray>& inputs, std::size_t skip=0) {
    require(pipeline.backend()=="cuda","device inputs require a CUDA pipeline");
    require(!inputs.empty(),"sample needs device arrays");
    Imported result; result.device=inputs.begin()->second.device_id();
    for (const auto& [key,in]:inputs) {
        require(pipeline.targets().count(key),"undeclared array field: "+key);
        require(in.device_id()==result.device,"all arrays must be on the same CUDA device");
        const auto dt=in.dtype(); const char kind=dlpack_kind(dt); const auto bytes=dt.bits/8;
        require(dt.lanes==1 && ((kind=='f' && (bytes==4 || bytes==8)) || ((kind=='i' || kind=='u') && (bytes==1 || bytes==2 || bytes==4 || bytes==8)) || (kind=='b' && bytes==1)),"unsupported device dtype");
        require(reinterpret_cast<std::uintptr_t>(in.data())%bytes==0,"unaligned device input");
        require(in.ndim()>skip,key+": missing batch axis");
        Shape shape; for (std::size_t a=skip;a<in.ndim();++a) shape.push_back(in.shape(a));
        Array metadata; metadata.shape=shape; metadata.kind=kind; metadata.bytes=bytes;
        require(metadata.size()<=SIZE_MAX/bytes,"array size overflow");
        result.sample.emplace(key,Field{std::move(metadata),pipeline.targets().at(key),key});
        result.arrays.emplace(key,cuda::DeviceInput{in.data(),std::move(shape),kind,std::size_t(bytes),result.device});
    }
    return result;
}
static void require_disjoint(const std::map<std::string,DeviceInputArray>& inputs) {
    for (auto i=inputs.begin();i!=inputs.end();++i) for (auto j=inputs.begin();j!=i;++j) {
        auto a=reinterpret_cast<std::uintptr_t>(i->second.data()), b=reinterpret_cast<std::uintptr_t>(j->second.data());
        require(a+i->second.nbytes()<=b || b+j->second.nbytes()<=a,"inplace transform requires nonoverlapping arrays");
    }
}
static nb::object device_array(void* data, const Shape& shape, char kind, std::size_t bytes, int device, nb::handle owner) {
    return nb::cast(nb::ndarray<nb::array_api, nb::device::cuda>(data,shape.size(),shape.data(),owner,nullptr,dlpack_dtype(kind,bytes),nb::device::cuda::value,device));
}
static nb::tuple run_device(const Pipeline& pipeline, const std::map<std::string,DeviceInputArray>& inputs,
                           std::uint64_t seed, bool record, bool profile, bool inplace, const Plan* plan=nullptr,
                           bool async=false, const NamedValues& params={}) {
    auto imported=import_device(pipeline,inputs);
    if (inplace) require_disjoint(inputs);
    std::shared_ptr<void> context;
    RunResult result;
    { nb::gil_scoped_release release; vcr::device::Guard guard(imported.device); NamedScope named(&params);
      validate(imported.sample,false); context=cuda::begin_device(imported.arrays,imported.sample,inplace,async && !record && !profile);
      result=plan ? pipeline.replay(std::move(imported.sample),*plan,context) : pipeline.run(std::move(imported.sample),seed,record,profile,context); }
    // The owner keeps the session (outputs) and, for inplace views, the inputs alive.
    nb::object outputs=nb::cast(DeviceOutputs{{context},{},imported.device});
    nb::list input_owners;
    for (const auto& [key,in]:inputs) input_owners.append(nb::cast(in,nb::rv_policy::reference));
    nb::object owner=nb::make_tuple(outputs,input_owners);
    nb::dict arrays,sources;
    for (const auto& [key,a]:cuda::device_outputs(context.get())) {
        arrays[nb::str(key.c_str())]=nb::make_tuple(device_array(a.data,a.shape,a.kind,a.bytes,a.device,owner),a.shape,std::string(1,a.kind)+std::to_string(a.bytes),a.device);
        sources[nb::str(key.c_str())]=nb::str(result.sample.at(key).source.c_str());
    }
    for (const auto& [key,levels]:cuda::device_pyramids(context.get())) {
        nb::list items;
        for (const auto& a:levels) items.append(nb::make_tuple(device_array(a.data,a.shape,a.kind,a.bytes,a.device,owner),a.shape,std::string(1,a.kind)+std::to_string(a.bytes),a.device));
        arrays[nb::str(key.c_str())]=items;
    }
    return nb::make_tuple(arrays,sources,nb::cast(result.timings),record ? nb::cast(std::move(result.plan)) : nb::none(),outputs);
}
// One pipeline invocation per leading-axis sample, on the current stream with
// no host synchronization when async. Outputs whose per-sample shapes agree are
// stacked into (B, ...) arrays (into `out` when given); others are lists.
static nb::tuple run_batch(const Pipeline& pipeline, const std::map<std::string,DeviceInputArray>& inputs,
                          const std::vector<std::uint64_t>& seeds, const std::map<std::string,std::vector<double>>& params,
                          bool inplace, bool async, const std::map<std::string,DeviceInputArray>& out,
                          nb::object allocator, nb::object stream) {
    vcr::device::StreamScope scope(stream.is_none() ? vcr::device::stream() : vcr::device::stream_handle(nb::cast<std::uintptr_t>(stream)));
    require(allocator.is_none() || out.empty(), "allocator and preallocated out are mutually exclusive");
    const std::size_t count=seeds.size();
    require(count>0,"batch needs at least one seed");
    for (const auto& [key,in]:inputs) require(in.ndim()>0 && std::size_t(in.shape(0))==count,key+": leading axis must equal len(seeds)");
    for (const auto& [name,values]:params) require(values.size()==count,"parameter "+name+" needs one value per sample");
    if (inplace) require_disjoint(inputs);
    auto imported=import_device(pipeline,inputs,1);
    auto owner=std::make_shared<DeviceOutputs>();
    owner->device=imported.device;
    // Start in consumer-owned contiguous storage. Transforms that retain their
    // input allocation can return it directly, without a final batch-size copy.
    std::map<std::string,nb::object> working;
    std::map<std::string,cuda::DeviceInput> destinations;
    const auto allocate_array = [&](const Shape& shape,const cuda::DeviceInput& first) {
        auto result=allocator(nb::cast(shape),std::string(1,first.kind)+std::to_string(first.bytes),imported.device);
        const auto dest=nb::cast<DeviceInputArray>(result);
        Shape got; for(std::size_t a=0;a<dest.ndim();++a)got.push_back(dest.shape(a));
        require(got==shape && dlpack_kind(dest.dtype())==first.kind && std::size_t(dest.dtype().bits/8)==first.bytes && dest.device_id()==imported.device,
                "allocator must return a contiguous CUDA array of the requested shape, dtype and device");
        return result;
    };
    if(!inplace && !allocator.is_none())for(const auto& [key,in]:imported.arrays) {
        Shape shape{count};shape.insert(shape.end(),in.shape.begin(),in.shape.end());
        auto result=allocate_array(shape,in);
        const auto dest=nb::cast<DeviceInputArray>(result);
        destinations.emplace(key,cuda::DeviceInput{dest.data(),in.shape,in.kind,in.bytes,in.device});
        working.emplace(key,result);owner->working_owners.push_back(std::move(result));
    }
    std::vector<std::map<std::string,cuda::DeviceInput>> results(count);
    std::vector<std::map<std::string,std::vector<cuda::DeviceInput>>> pyramids(count);
    std::map<std::string,std::string> sources;
    {
        nb::gil_scoped_release release; vcr::device::Guard guard(imported.device);
        std::vector<Sample> samples(count,imported.sample);
        cuda::BatchQueue initialization(count);
        for (std::size_t b=0;b<count;++b) {
            initialization.lane(b);
            auto& sample=samples[b]; auto arrays=imported.arrays; auto outputs=destinations;
            for (auto& [key,a]:arrays) a.data=static_cast<char*>(a.data)+b*product(a.shape)*a.bytes;
            for (auto& [key,a]:outputs) a.data=static_cast<char*>(a.data)+b*product(a.shape)*a.bytes;
            NamedValues values; for (const auto& [name,v]:params) values[name]=v[b];
            NamedScope named(&values);
            validate(sample,false);
            auto context=cuda::begin_device(arrays,sample,inplace,async,outputs);
            owner->sessions.push_back(std::move(context));
        }
        initialization.flush();
        cuda::run_batch_steps(*pipeline.program(),samples,seeds,params,owner->sessions);
        for (std::size_t b=0;b<count;++b) {
            results[b]=cuda::device_outputs(owner->sessions[b].get());
            pyramids[b]=cuda::device_pyramids(owner->sessions[b].get());
        }
        for (const auto& [key,f]:samples[0]) sources[key]=f.source;
    }
    nb::object holder=nb::cast(owner);
    nb::list input_owners;
    for (const auto& [key,in]:inputs) input_owners.append(nb::cast(in,nb::rv_policy::reference));
    nb::dict arrays;
    // Optional consumer allocator owns the final tensors. This also handles
    // shape-changing, ragged and pyramid outputs without predicting shapes in
    // Python or exporting C++ allocations to an unrelated stream allocator.
    const auto allocated = [&](const Shape& shape, const std::vector<cuda::DeviceInput>& values) -> nb::object {
        const auto& first=values.front();
        auto result=allocate_array(shape,first);
        const auto dest=nb::cast<DeviceInputArray>(result);
        { nb::gil_scoped_release release; vcr::device::Guard guard(imported.device);
          cuda::copy_batch_output(dest.data(),values); }
        return result;
    };
    const auto pack = [&](const std::string& key,const std::vector<cuda::DeviceInput>& values,bool allow_out) -> nb::object {
        const auto& first=values.front();bool uniform=true;
        for(const auto& a:values)uniform &= a.shape==first.shape && a.kind==first.kind && a.bytes==first.bytes;
        if(!uniform) {
            nb::list items;nb::object list_owner=nb::make_tuple(holder,input_owners);
            for(const auto& a:values)items.append(allocator.is_none() ? device_array(a.data,a.shape,a.kind,a.bytes,a.device,list_owner) : allocated(a.shape,{a}));
            return items;
        }
        Shape shape{count};shape.insert(shape.end(),first.shape.begin(),first.shape.end());
        if(allow_out && working.count(key)) {
            const auto& dest=destinations.at(key);
            bool retained=dest.shape==first.shape && dest.kind==first.kind && dest.bytes==first.bytes;
            const auto bytes=product(first.shape)*first.bytes;
            for(std::size_t b=0;b<values.size();++b)
                retained &= values[b].data==static_cast<char*>(dest.data)+b*bytes;
            if(retained)return working.at(key);
        }
        if (!allocator.is_none()) return allocated(shape,values);
        const std::size_t bytes=product(first.shape)*first.bytes;void* target;nb::object array_owner;
        if(allow_out && out.count(key)) {
            const auto& dest=out.at(key);Shape got;for(std::size_t a=0;a<dest.ndim();++a)got.push_back(dest.shape(a));
            require(got==shape && dlpack_kind(dest.dtype())==first.kind && std::size_t(dest.dtype().bits/8)==first.bytes && dest.device_id()==imported.device,
                    key+": out array must match the stacked output shape, dtype and device");
            target=dest.data();array_owner=nb::cast(dest,nb::rv_policy::reference);
        } else {
            auto buffer=vcr::device::allocate(count*bytes,imported.device);owner->stacked.push_back(buffer);target=buffer->ptr;array_owner=holder;
        }
        {nb::gil_scoped_release release;vcr::device::Guard guard(imported.device);
         cuda::copy_batch_output(target,values);}
        return nb::make_tuple(device_array(target,shape,first.kind,first.bytes,imported.device,array_owner),shape,std::string(1,first.kind)+std::to_string(first.bytes),imported.device);
    };
    for(const auto& [key,first]:results[0]) {
        if(pyramids[0].count(key))continue;
        std::vector<cuda::DeviceInput> values;for(const auto& r:results){require(r.count(key),"batch output keys must agree");values.push_back(r.at(key));}
        arrays[nb::str(key.c_str())]=pack(key,values,true);
    }
    for(const auto& [key,levels]:pyramids[0]) {
        require(!out.count(key),"preallocated out does not accept pyramid lists");nb::list items;
        for(std::size_t level=0;level<levels.size();++level) {
            std::vector<cuda::DeviceInput> values;
            for(const auto& r:pyramids){require(r.count(key) && r.at(key).size()==levels.size(),"batch pyramid levels must agree");values.push_back(r.at(key)[level]);}
            items.append(pack(key,values,false));
        }
        arrays[nb::str(key.c_str())]=items;
    }
    // Stacked per-sample arrays are no longer needed: free them stream-ordered,
    // keeping each session's small status buffers for check().
    std::vector<std::string> stacked;
    for (auto item:arrays) if (!allocator.is_none() || !nb::isinstance<nb::list>(item.second)) stacked.push_back(nb::cast<std::string>(item.first));
    { nb::gil_scoped_release release; vcr::device::Guard guard(imported.device);
      for (auto& s:owner->sessions) cuda::release_outputs(s.get(),stacked);
      if (!async) { cuda::check_devices(owner->sessions); vcr::device::complete(); } }
    nb::dict names; for (const auto& [k,v]:sources) names[nb::str(k.c_str())]=nb::str(v.c_str());
    return nb::make_tuple(arrays,names,holder);
}
