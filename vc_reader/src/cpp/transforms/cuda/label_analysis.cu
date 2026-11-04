#include "resident.cuh"
#include "../cpu/labels.hpp"
namespace vcr::transforms::cuda {
namespace {
struct BinaryMask {
    template<class T> __device__ void operator()(KernelIndex index,const T* input,std::uint8_t* output,std::size_t n) const {
        for(std::size_t i=std::size_t(index.block)*blockDim.x+threadIdx.x;i<n;i+=std::size_t(index.blocks)*blockDim.x)output[i]=input[i]!=0;
    }
};
template<class T> void analyze(const T& t,Sample& s,Session& session) {
    std::vector<std::pair<std::string,Field>> outputs;
    for(const auto& [name,f]:s)if(selected_label(name,f,t.on)) {
        const auto& a=session.arrays.at(name);const auto shape=a.spatial();const auto n=product(shape);
        constexpr bool components=std::is_same_v<T,ConnectedComponents>;
        DeviceArray out(a.shape,components?'i':'u',components?4:1);Buffer<std::uint8_t> mask(n);
        for(std::size_t c=0;c<a.shape[0];++c) {
            a.typed([&](auto values){launch(blocks(n),256,0,device::stream(),BinaryMask{},values.ptr+c*n,mask.ptr,n);});
            // Native analysis consumes these resident buffers immediately.
            if(BatchQueue::current)BatchQueue::current->flush();
            if constexpr(components) vcr::components::cuda::label(mask.ptr,shape,component_connectivity(t,shape.size()),out.view<std::int32_t>().ptr+c*n);
            else {
                const auto h=shape[shape.size()-2],w=shape.back();
                for(std::size_t z=0;z<n;z+=h*w)thinning::cuda::traceDevice(mask.ptr+z,out.view<std::uint8_t>().ptr+c*n+z,h,w);
            }
        }
        Array metadata;metadata.shape=a.shape;metadata.kind=components?'i':'u';metadata.bytes=components?4:1;
        outputs.emplace_back(name+t.suffix,Field{std::move(metadata),Target{Role::Label},name});
        require(!session.arrays.count(name+t.suffix),"derived label field already exists");session.arrays.emplace(name+t.suffix,std::move(out));
    }
    for(auto& [name,f]:outputs){require(!s.count(name),"derived label field already exists");s.emplace(name,std::move(f));}
}
}
void apply_cuda(const ConnectedComponents& t,Sample& s,const Parameters&,Session& session){analyze(t,s,session);}
void apply_cuda(const RidgeTrace& t,Sample& s,const Parameters&,Session& session){analyze(t,s,session);}
}
