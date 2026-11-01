#include "regions.cuh"
namespace vcr::transforms::cuda {
void apply_cuda(const OneHot& t,Sample& s,const Parameters&,Session& session) {
    std::vector<std::vector<double>> labels; for (double v:t.labels) labels.push_back({v});
    auto encoded=regions_cuda(session.arrays.at(t.on),t.channel,labels,session); const auto n=product(encoded.spatial());
    if (!t.append_to.empty()) {
        auto& image=session.arrays.at(t.append_to); require(image.spatial()==encoded.spatial(),"one-hot image grid mismatch");
        auto shape=image.shape; shape[0]+=encoded.shape[0]; DeviceArray combined(shape,image.kind,image.bytes); const auto count=image.size();
        check(copy_async(combined.buffer->ptr,image.buffer->ptr,count*image.bytes,cudaMemcpyDeviceToDevice,vcr::device::stream()));
        combined.typed([&](auto data) { using T=typename decltype(data)::value_type; convert(View<T>{data.ptr+count,encoded.size()},encoded.view<std::uint8_t>()); });
        s.at(t.append_to).array=array_metadata(combined); image=std::move(combined);
    } else {
        auto field=s.at(t.on); field.array=array_metadata(encoded); field.source=t.on; field.target.role=Role::Label;
        s[t.output]=std::move(field); session.arrays.insert_or_assign(t.output,std::move(encoded));
    }
    if (t.remove_source) {
        auto& a=session.arrays.at(t.on);
        if (a.shape[0]==1) { s.erase(t.on); session.arrays.erase(t.on); }
        else {
            auto shape=a.shape; --shape[0]; DeviceArray remaining(shape,a.kind,a.bytes); std::size_t out=0;
            for (std::size_t c=0;c<a.shape[0];++c) if (c!=t.channel) {
                check(copy_async(remaining.buffer->ptr+out*n*a.bytes,a.buffer->ptr+c*n*a.bytes,n*a.bytes,cudaMemcpyDeviceToDevice,vcr::device::stream())); ++out;
            }
            s.at(t.on).array=array_metadata(remaining); a=std::move(remaining);
        }
    }
}
}
