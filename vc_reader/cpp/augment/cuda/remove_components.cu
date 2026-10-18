#include "label_values.cuh"
#include "components.cuh"
namespace vcr::augment::cuda {
__global__ void count_components(const ComponentIndex* parents,std::size_t n,ComponentIndex* sizes){
    EACH_INDEX(n)if(parents[i]!=n)atomicAdd(sizes+parents[i],1ULL);
}
__global__ void eligible_components(const ComponentIndex* sizes,std::size_t n,double max_fraction,std::uint8_t* flags){
    EACH_INDEX(n)flags[i]=sizes[i]>0 && sizes[i]<n*max_fraction;
}
__global__ void component_extent(const ComponentIndex* sizes,const ComponentIndex* eligible,std::size_t index,ComponentIndex* extent){
    const auto root=eligible[index];extent[0]=root;extent[1]=sizes[root];
}
__global__ void component_members(const ComponentIndex* parents,std::size_t n,ComponentIndex root,std::uint8_t* flags){EACH_INDEX(n)flags[i]=parents[i]==root;}
__global__ void component_indices(const ComponentIndex* indices,std::size_t* field,std::size_t count){EACH_INDEX(count)field[i]=std::size_t(indices[i]);}
template<class T> __global__ void erase_component(T* data,std::size_t n,std::size_t channel,long long other,const std::size_t* indices,std::size_t count) {
    EACH_INDEX(count) {const auto j=std::size_t(indices[i]);data[channel*n+j]=0;if(other>=0)data[std::size_t(other)*n+j]=1;}
}
void remove_component_cuda(DeviceArray& a,const ChannelPlan& c) {
    auto field=c.device_indices ? std::static_pointer_cast<Buffer<std::size_t>>(c.device_indices) : std::make_shared<Buffer<std::size_t>>(c.indices); const auto n=product(a.spatial());
    a.typed([&](auto v){erase_component<<<blocks(field->size),256,0,vcr::device::stream()>>>(v.ptr,n,c.channel,static_cast<long long>(c.values[0]),field->ptr,field->size);});check(cudaGetLastError());
}
Parameters sample_components(const RemoveComponents& t,const Sample&,Random& r,bool record,Session& session) {
    require(!session.async,"RemoveComponents requires sync=True for component selection");probability(t.max_fraction);
    const auto& source=session.arrays.at(t.on);require_exact_labels(source,session);session.check_status();
    DeviceArray a(source.shape,source.kind,source.bytes);check(cudaMemcpyAsync(a.buffer->ptr,source.buffer->ptr,source.size()*source.bytes,cudaMemcpyDeviceToDevice,vcr::device::stream()));
    Grid g(a.spatial());const auto channels=selected_channels(t.channels,a.shape[0],&r);Parameters p;
    Buffer<std::uint8_t> flags(g.size);Buffer<ComponentIndex> parents(g.size),sizes(g.size),eligible(g.size),count(1),extent(2);
    for(auto c:channels)if(r.chance(t.p_channel)) {
        a.typed([&](auto v){label_components<false>(v.ptr+c*g.size,g,0,t.full_connectivity,parents);});
        check(cudaMemsetAsync(sizes.ptr,0,sizes.size*sizeof(ComponentIndex),vcr::device::stream()));
        count_components<<<blocks(g.size),256,0,vcr::device::stream()>>>(parents.ptr,g.size,sizes.ptr);
        eligible_components<<<blocks(g.size),256,0,vcr::device::stream()>>>(sizes.ptr,g.size,t.max_fraction,flags.ptr);check(cudaGetLastError());
        select_flagged_indices(flags,eligible,count);
        ComponentIndex total;count.download(&total);if(!total)continue;
        component_extent<<<1,1,0,vcr::device::stream()>>>(sizes.ptr,eligible.ptr,r.index(total),extent.ptr);check(cudaGetLastError());
        ComponentIndex range[2];extent.download(range);auto field=std::make_shared<Buffer<std::size_t>>(range[1]);
        component_members<<<blocks(g.size),256,0,vcr::device::stream()>>>(parents.ptr,g.size,range[0],flags.ptr);check(cudaGetLastError());
        select_flagged_indices(flags,eligible,count);
        component_indices<<<blocks(field->size),256,0,vcr::device::stream()>>>(eligible.ptr,field->ptr,field->size);check(cudaGetLastError());
        ChannelPlan plan{t.on,c,{-1}};plan.device_indices=field;
        if(record){plan.indices.resize(field->size);field->download(plan.indices.data());}
        if(r.chance(t.p_fill) && channels.size()>1) {auto others=channels;others.erase(std::remove(others.begin(),others.end(),c),others.end());plan.values[0]=double(others[r.index(others.size())]);}
        remove_component_cuda(a,plan);p.channels.push_back(std::move(plan));
    }
    return p;
}
void apply_cuda(const RemoveComponents& t,Sample&,const Parameters& p,Session& session){for(const auto& c:p.channels)remove_component_cuda(session.arrays.at(t.on),c);}
}
