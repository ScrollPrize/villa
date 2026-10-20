#include "label_values.cuh"
#include "components.cuh"
#include "random.cuh"
namespace vcr::transforms::cuda {
__device__ void count_components(KernelIndex vcr_launch, const ComponentIndex* parents,std::size_t n,ComponentIndex* sizes){
    EACH_INDEX(n)if(parents[i]!=n)atomicAdd(sizes+parents[i],1ULL);
}
struct count_components_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { count_components(index,args...); }
};
__device__ void eligible_components(KernelIndex vcr_launch, const ComponentIndex* sizes,std::size_t n,double max_fraction,std::uint8_t* flags){
    EACH_INDEX(n)flags[i]=sizes[i]>0 && sizes[i]<n*max_fraction;
}
struct eligible_components_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { eligible_components(index,args...); }
};
template<class T> __device__ void erase_component(KernelIndex vcr_launch, T* data,std::size_t n,std::size_t channel,long long other,const std::size_t* indices,std::size_t count) {
    EACH_INDEX(count) {const auto j=std::size_t(indices[i]);data[channel*n+j]=0;if(other>=0)data[std::size_t(other)*n+j]=1;}
}
struct erase_component_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { erase_component(index,args...); }
};
struct ComponentChoice { ComponentIndex root; long long other; };
struct ComponentSelection {
    Buffer<ComponentIndex> parents;
    Buffer<ComponentChoice> choice{1};
    explicit ComponentSelection(std::size_t n):parents(n) {}
};
struct ChooseComponent {
    __device__ void operator()(KernelIndex, const ComponentIndex* eligible,const ComponentIndex* count,
            ComponentChoice* choice,std::uint64_t* state,std::size_t n,std::size_t channel,
            const std::size_t* channels,std::size_t num_channels,double p_channel,double p_fill) const {
        Random random{*state}; *choice={ComponentIndex(n),-1};
        if (random.uniform()<p_channel && *count) {
            choice->root=eligible[random_index(random,*count)];
            if (random.uniform()<p_fill && num_channels>1) {
                auto index=random_index(random,num_channels-1);
                for (std::size_t j=0;j<num_channels;++j) if(channels[j]!=channel) {
                    if (!index) { choice->other=static_cast<long long>(channels[j]); break; }
                    --index;
                }
            }
        }
        *state=random.state;
    }
};
struct EraseSelectedComponent {
    template<class T> __device__ void operator()(KernelIndex vcr_launch,T* data,std::size_t n,std::size_t channel,
            const ComponentIndex* parents,const ComponentChoice* choice) const {
        if (choice->root==n) return;
        EACH_INDEX(n) if(parents[i]==choice->root) {
            data[channel*n+i]=0;
            if(choice->other>=0)data[std::size_t(choice->other)*n+i]=1;
        }
    }
};
void remove_component_cuda(DeviceArray& a,const ChannelPlan& c) {
    const auto n=product(a.spatial());
    if (c.device_selection) {
        auto selection=std::static_pointer_cast<ComponentSelection>(c.device_selection);
        a.typed([&](auto v){launch(blocks(n),256,0,vcr::device::stream(),EraseSelectedComponent{},v.ptr,n,c.channel,selection->parents.ptr,selection->choice.ptr);});
    } else {
        auto field=c.device_indices ? std::static_pointer_cast<Buffer<std::size_t>>(c.device_indices) : std::make_shared<Buffer<std::size_t>>(c.indices);
        a.typed([&](auto v){launch(blocks(field->size),256,0,vcr::device::stream(),erase_component_launch{},v.ptr,n,c.channel,static_cast<long long>(c.values[0]),field->ptr,field->size);});
    }
    check(cudaGetLastError());
}
Parameters sample_components(const RemoveComponents& t,const Sample&,Random& r,bool record,Session& session) {
    probability(t.max_fraction); probability(t.p_channel); probability(t.p_fill);
    const auto& source=session.arrays.at(t.on);require_exact_labels(source,session);
    const auto channels=selected_channels(t.channels,source.shape[0],&r);
    if(!t.p_channel)return {};
    DeviceArray a(source.shape,source.kind,source.bytes);
    check(copy_async(a.buffer->ptr,source.buffer->ptr,source.size()*source.bytes,cudaMemcpyDeviceToDevice,vcr::device::stream()));
    Grid g(a.spatial());Parameters p;
    Buffer<std::size_t> selected_channels_device(channels);
    Buffer<std::uint64_t> state(std::vector<std::uint64_t>{r.state});
    Buffer<std::uint8_t> flags(g.size);Buffer<ComponentIndex> sizes(g.size),eligible(g.size),count(1);
    for(auto c:channels) {
        auto selection=std::make_shared<ComponentSelection>(g.size);
        a.typed([&](auto v){label_components<false>(v.ptr+c*g.size,g,0,t.full_connectivity,selection->parents);});
        check(memset_async(sizes.ptr,0,sizes.size*sizeof(ComponentIndex),vcr::device::stream()));
        launch(blocks(g.size),256,0,vcr::device::stream(),count_components_launch{},selection->parents.ptr,g.size,sizes.ptr);
        launch(blocks(g.size),256,0,vcr::device::stream(),eligible_components_launch{},sizes.ptr,g.size,t.max_fraction,flags.ptr);
        select_flagged_indices(flags,eligible,count);
        launch(1,1,0,vcr::device::stream(),ChooseComponent{},eligible.ptr,count.ptr,selection->choice.ptr,state.ptr,g.size,c,selected_channels_device.ptr,channels.size(),t.p_channel,t.p_fill);
        ChannelPlan plan{t.on,c,{-1}};plan.device_selection=selection;
        remove_component_cuda(a,plan);
        if(record) {
            ComponentChoice choice;selection->choice.download(&choice);
            if(choice.root==g.size)continue;
            std::vector<ComponentIndex> parents(g.size);selection->parents.download(parents.data());
            for(std::size_t i=0;i<g.size;++i)if(parents[i]==choice.root)plan.indices.push_back(i);
            plan.values[0]=double(choice.other);plan.device_selection.reset();
        }
        p.channels.push_back(std::move(plan));
    }
    return p;
}
void apply_cuda(const RemoveComponents& t,Sample&,const Parameters& p,Session& session){for(const auto& c:p.channels)remove_component_cuda(session.arrays.at(t.on),c);}
}
