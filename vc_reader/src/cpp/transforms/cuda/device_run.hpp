#pragma once
// Running a pipeline over arrays already on a CUDA device, one sample or a batch.
#include "batch_queue.hpp"
#include "device_io.hpp"
#include "../pipeline.hpp"
#include "../../cuda_buffer.hpp"
#include "../../cuda_stream.hpp"
#include <functional>
#include <map>

namespace vcr::transforms::cuda {
// Validated device inputs; `skip` drops leading (batch) axes from each shape.
struct DeviceSample { Sample sample; std::map<std::string,DeviceInput> arrays; int device = 0; };
inline DeviceSample import_device(const Pipeline& pipeline, const std::map<std::string,DeviceInput>& inputs, std::size_t skip=0) {
    require(pipeline.backend()=="cuda","device inputs require a CUDA pipeline");
    require(!inputs.empty(),"sample needs device arrays");
    DeviceSample result; result.device=inputs.begin()->second.device;
    for (const auto& [key,in]:inputs) {
        require(pipeline.targets().count(key),"undeclared array field: "+key);
        require(in.device==result.device,"all arrays must be on the same CUDA device");
        const char kind=in.kind; const auto bytes=in.bytes;
        require((kind=='f' && (bytes==4 || bytes==8)) || ((kind=='i' || kind=='u') && (bytes==1 || bytes==2 || bytes==4 || bytes==8)) || (kind=='b' && bytes==1),"unsupported device dtype");
        require(reinterpret_cast<std::uintptr_t>(in.data)%bytes==0,"unaligned device input");
        require(in.shape.size()>skip,key+": missing batch axis");
        Shape shape(in.shape.begin()+std::ptrdiff_t(skip),in.shape.end());
        Array metadata; metadata.shape=shape; metadata.kind=kind; metadata.bytes=bytes;
        require(metadata.size()<=SIZE_MAX/bytes,"array size overflow");
        result.sample.emplace(key,Field{std::move(metadata),pipeline.targets().at(key),key});
        result.arrays.emplace(key,DeviceInput{in.data,std::move(shape),kind,bytes,result.device});
    }
    return result;
}
inline void require_disjoint(const std::map<std::string,DeviceInput>& inputs) {
    const auto nbytes=[](const DeviceInput& a) { return product(a.shape)*a.bytes; };
    for (auto i=inputs.begin();i!=inputs.end();++i) for (auto j=inputs.begin();j!=i;++j) {
        auto a=reinterpret_cast<std::uintptr_t>(i->second.data), b=reinterpret_cast<std::uintptr_t>(j->second.data);
        require(a+nbytes(i->second)<=b || b+nbytes(j->second)<=a,"inplace transform requires nonoverlapping arrays");
    }
}
// One sample. `session` owns the outputs (device_outputs, device_pyramids).
struct DeviceRun { RunResult result; std::shared_ptr<void> session; int device = 0; };
inline DeviceRun run_device(const Pipeline& pipeline, const std::map<std::string,DeviceInput>& inputs, std::uint64_t seed, bool record, bool inplace,
                            const Plan* plan=nullptr, const NamedValues& params={}) {
    auto imported=import_device(pipeline,inputs);
    if (inplace) require_disjoint(inputs);
    vcr::device::Guard guard(imported.device); NamedScope named(&params);
    validate(imported.sample,false);
    DeviceRun run; run.device=imported.device;
    run.session=begin_device(imported.arrays,imported.sample,inplace,false);
    run.result=plan ? pipeline.replay(std::move(imported.sample),*plan,run.session) : pipeline.run(std::move(imported.sample),seed,record,run.session);
    return run;
}
// Owner of a run's outputs. check() raises the deferred device checks of asynchronous calls.
struct DeviceOutputs {
    std::vector<std::shared_ptr<void>> sessions;
    int device = 0;
    cudaStream_t stream = vcr::device::stream();
    // Arrays are marked exported (their destruction then synchronizes the device) only when shared.
    void mark_exported() { for (auto& s:sessions) export_outputs(s.get()); }
    void check() {
        vcr::device::Guard guard(device); vcr::device::StreamScope scope(stream);
        check_devices(sessions);
    }
};
// A target of a batch: one stacked (B, ...) array when samples agree, else one array per sample.
struct BatchArray { std::vector<DeviceInput> arrays; bool stacked = true; };
struct BatchRun {
    std::map<std::string,BatchArray> arrays;
    std::map<std::string,std::vector<BatchArray>> pyramids;
    std::map<std::string,std::string> sources;
    std::shared_ptr<DeviceOutputs> outputs;
};
// Supplies contiguous device storage of a shape, element kind and size, on a device. A caller
// owning the final tensors passes its allocator; storage must outlive the returned run.
using BatchAllocator = std::function<DeviceInput(const Shape&, char kind, std::size_t bytes, int device)>;
// One pipeline invocation per leading-axis sample, on the current stream with no host
// synchronization; outputs->check() raises deferred device errors. The allocator supplies every
// output: per-sample shapes that agree are stacked, others are returned per sample.
inline BatchRun run_batch(const Pipeline& pipeline, const std::map<std::string,DeviceInput>& inputs, const std::vector<std::uint64_t>& seeds,
                          const std::map<std::string,std::vector<double>>& params, const BatchAllocator& allocator) {
    const std::size_t count=seeds.size();
    require(count>0,"batch needs at least one seed");
    for (const auto& [key,in]:inputs) require(!in.shape.empty() && in.shape[0]==count,key+": leading axis must equal len(seeds)");
    for (const auto& [name,values]:params) require(values.size()==count,"parameter "+name+" needs one value per sample");
    auto imported=import_device(pipeline,inputs,1);
    BatchRun run;
    run.outputs=std::make_shared<DeviceOutputs>();
    run.outputs->device=imported.device;
    const auto allocate_array = [&](const Shape& shape,const DeviceInput& first) {
        auto dest=allocator(shape,first.kind,first.bytes,imported.device);
        require(dest.shape==shape && dest.kind==first.kind && dest.bytes==first.bytes && dest.device==imported.device,
                "allocator must return a contiguous CUDA array of the requested shape, dtype and device");
        return dest;
    };
    const auto stacked_shape = [&](const Shape& sample) { Shape shape{count}; shape.insert(shape.end(),sample.begin(),sample.end()); return shape; };
    // Start in consumer-owned contiguous storage. Transforms that retain their
    // input allocation can return it directly, without a final batch-size copy.
    std::map<std::string,DeviceInput> working, destinations;
    for(const auto& [key,in]:imported.arrays) {
        auto dest=allocate_array(stacked_shape(in.shape),in);
        destinations.emplace(key,DeviceInput{dest.data,in.shape,in.kind,in.bytes,in.device});
        working.emplace(key,std::move(dest));
    }
    std::vector<std::map<std::string,DeviceInput>> results(count);
    std::vector<std::map<std::string,std::vector<DeviceInput>>> pyramids(count);
    {
        vcr::device::Guard guard(imported.device);
        std::vector<Sample> samples(count,imported.sample);
        BatchQueue initialization(count);
        for (std::size_t b=0;b<count;++b) {
            initialization.lane(b);
            auto& sample=samples[b]; auto arrays=imported.arrays; auto outputs=destinations;
            for (auto& [key,a]:arrays) a.data=static_cast<char*>(a.data)+b*product(a.shape)*a.bytes;
            for (auto& [key,a]:outputs) a.data=static_cast<char*>(a.data)+b*product(a.shape)*a.bytes;
            NamedValues values; for (const auto& [name,v]:params) values[name]=v[b];
            NamedScope named(&values);
            validate(sample,false);
            run.outputs->sessions.push_back(begin_device(arrays,sample,false,true,outputs));
        }
        initialization.flush();
        run_batch_steps(*pipeline.program(),samples,seeds,params,run.outputs->sessions);
        for (std::size_t b=0;b<count;++b) {
            results[b]=device_outputs(run.outputs->sessions[b].get());
            pyramids[b]=device_pyramids(run.outputs->sessions[b].get());
        }
        for (const auto& [key,f]:samples[0]) run.sources[key]=f.source;
    }
    // The consumer's allocator owns the final tensors. This also handles
    // shape-changing, ragged and pyramid outputs without predicting shapes.
    const auto allocated = [&](const Shape& shape, const std::vector<DeviceInput>& values) {
        auto dest=allocate_array(shape,values.front());
        vcr::device::Guard guard(imported.device);
        copy_batch_output(dest.data,values);
        return dest;
    };
    const auto pack = [&](const std::string& key,const std::vector<DeviceInput>& values,bool retainable) {
        const auto& first=values.front();bool uniform=true;
        for(const auto& a:values)uniform &= a.shape==first.shape && a.kind==first.kind && a.bytes==first.bytes;
        BatchArray out;
        if(!uniform) {
            out.stacked=false;
            for(const auto& a:values)out.arrays.push_back(allocated(a.shape,{a}));
            return out;
        }
        if(retainable && working.count(key)) {
            const auto& dest=destinations.at(key);
            bool retained=dest.shape==first.shape && dest.kind==first.kind && dest.bytes==first.bytes;
            const auto bytes=product(first.shape)*first.bytes;
            for(std::size_t b=0;b<values.size();++b)
                retained &= values[b].data==static_cast<char*>(dest.data)+b*bytes;
            if(retained) { out.arrays.push_back(working.at(key)); return out; }
        }
        out.arrays.push_back(allocated(stacked_shape(first.shape),values));
        return out;
    };
    for(const auto& [key,first]:results[0]) {
        if(pyramids[0].count(key))continue;
        std::vector<DeviceInput> values;for(const auto& r:results){require(r.count(key),"batch output keys must agree");values.push_back(r.at(key));}
        run.arrays.emplace(key,pack(key,values,true));
    }
    for(const auto& [key,levels]:pyramids[0]) {
        auto& items=run.pyramids[key];
        for(std::size_t level=0;level<levels.size();++level) {
            std::vector<DeviceInput> values;
            for(const auto& r:pyramids){require(r.count(key) && r.at(key).size()==levels.size(),"batch pyramid levels must agree");values.push_back(r.at(key)[level]);}
            items.push_back(pack(key,values,false));
        }
    }
    // The per-sample arrays are copied out: free them stream-ordered, keeping
    // each session's small status buffers for check().
    std::vector<std::string> copied;
    for (const auto& [key,value]:run.arrays) copied.push_back(key);
    for (const auto& [key,value]:run.pyramids) copied.push_back(key);
    { vcr::device::Guard guard(imported.device);
      for (auto& s:run.outputs->sessions) release_outputs(s.get(),copied); }
    return run;
}
}
