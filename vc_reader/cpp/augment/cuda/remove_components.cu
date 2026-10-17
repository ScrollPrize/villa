#include "label_values.cuh"
namespace vcr::augment::cuda {
template<class T> __global__ void find_components(const T* input,component_detail::GridView grid,bool full,double max_fraction,
        std::uint8_t* visited,std::size_t* queue,std::size_t* offsets,std::size_t* eligible,std::size_t* count) {
    const auto total=component_detail::components(input,grid,full,visited,queue,offsets); *count=0;
    for(std::size_t c=0;c<total;++c)if(offsets[c+1]-offsets[c]<grid.size*max_fraction)eligible[(*count)++]=c;
}
__global__ void component_extent(const std::size_t* offsets,const std::size_t* eligible,std::size_t index,std::size_t* extent) {
    const auto c=eligible[index];extent[0]=offsets[c];extent[1]=offsets[c+1]-offsets[c];
}
__global__ void component_indices(const std::size_t* queue,double* indices,std::size_t count) {EACH_INDEX(count)indices[i]=double(queue[i]);}
template<class T> __global__ void erase_component(T* data,std::size_t n,std::size_t channel,long long other,const double* indices,std::size_t count) {
    EACH_INDEX(count) {const auto j=std::size_t(indices[i]);data[channel*n+j]=0;if(other>=0)data[std::size_t(other)*n+j]=1;}
}
void remove_component_cuda(DeviceArray& a,const ChannelPlan& c) {
    auto field=field_buffer(c.field,c.device_field); const auto n=product(a.spatial());
    a.typed([&](auto v){erase_component<<<blocks(field->size),256,0,vcr::device::stream()>>>(v.ptr,n,c.channel,static_cast<long long>(c.values[0]),field->ptr,field->size);});check(cudaGetLastError());
}
Parameters sample_components(const RemoveComponents& t,const Sample&,Random& r,bool record,Session& session) {
    require(!session.async,"RemoveComponents requires sync=True for component selection");probability(t.max_fraction);
    const auto& source=session.arrays.at(t.on);require_exact_labels(source,session);session.check_status();
    DeviceArray a(source.shape,source.kind,source.bytes);check(cudaMemcpyAsync(a.buffer->ptr,source.buffer->ptr,source.size()*source.bytes,cudaMemcpyDeviceToDevice,vcr::device::stream()));
    Grid g(a.spatial());const auto channels=selected_channels(t.channels,a.shape[0],&r);Parameters p;
    Buffer<std::uint8_t> visited(g.size);Buffer<std::size_t> queue(g.size),offsets(g.size+1),eligible(g.size),count(1),extent(2);
    for(auto c:channels)if(r.chance(t.p_channel)) {
        a.typed([&](auto v){find_components<<<1,1,0,vcr::device::stream()>>>(v.ptr+c*g.size,component_detail::GridView(g),t.full_connectivity,t.max_fraction,visited.ptr,queue.ptr,offsets.ptr,eligible.ptr,count.ptr);});check(cudaGetLastError());
        std::size_t total;count.download(&total);if(!total)continue;
        component_extent<<<1,1,0,vcr::device::stream()>>>(offsets.ptr,eligible.ptr,r.index(total),extent.ptr);check(cudaGetLastError());
        std::size_t range[2];extent.download(range);auto field=std::make_shared<Buffer<double>>(range[1]);
        component_indices<<<blocks(field->size),256,0,vcr::device::stream()>>>(queue.ptr+range[0],field->ptr,field->size);check(cudaGetLastError());
        ChannelPlan plan{t.on,c,{-1}};plan.device_field=field;
        if(record){plan.field.resize(field->size);field->download(plan.field.data());}
        if(r.chance(t.p_fill) && channels.size()>1) {auto others=channels;others.erase(std::remove(others.begin(),others.end(),c),others.end());plan.values[0]=double(others[r.index(others.size())]);}
        remove_component_cuda(a,plan);p.channels.push_back(std::move(plan));
    }
    return p;
}
void apply_cuda(const RemoveComponents& t,Sample&,const Parameters& p,Session& session){for(const auto& c:p.channels)remove_component_cuda(session.arrays.at(t.on),c);}
}
