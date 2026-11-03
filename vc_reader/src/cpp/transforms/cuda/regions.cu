#include "regions.cuh"
namespace vcr::transforms::cuda {
template<class T> __device__ void encode_region(KernelIndex vcr_launch, const T* input,std::uint8_t* output,std::size_t n,const double* labels,std::size_t count) {
    EACH_INDEX(n) { bool found=false; for (std::size_t j=0;j<count;++j) found |= double(input[i])==labels[j]; output[i]=found; }
}
struct encode_region_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { encode_region(index,args...); }
};
DeviceArray regions_cuda(const DeviceArray& input,std::size_t channel,const std::vector<std::vector<double>>& labels,Session& session) {
    require_exact_labels(input,session);
    for (const auto& group:labels) for (double v:group) require_label_value(v);
    require(channel<input.shape[0] && !labels.empty(),"invalid region channel or empty label groups");
    auto shape=input.shape; shape[0]=labels.size(); DeviceArray out(shape,'b',1); const auto n=product(input.spatial());
    input.typed([&](auto data) { for (std::size_t c=0;c<labels.size();++c) {
        Buffer<double> group(labels[c]); launch(blocks(n),256,0,vcr::device::stream(), encode_region_launch{}, data.ptr+channel*n,out.view<std::uint8_t>().ptr+c*n,n,group.ptr,group.size);
    }}); check(cudaGetLastError()); return out;
}
void apply_cuda(const Regions& t,Sample& s,const Parameters&,Session& session) {
    auto result=regions_cuda(session.arrays.at(t.on),t.channel,t.labels,session);
    auto field=s.at(t.on); field.array=array_metadata(result); field.source=t.on; field.target.role=Role::Label;
    const auto key=t.output.empty()?t.on:t.output; s[key]=std::move(field); session.arrays.insert_or_assign(key,std::move(result));
}
}
