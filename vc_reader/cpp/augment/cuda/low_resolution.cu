#include "geometry.cuh"
namespace vcr::augment::cuda {
void apply_cuda(const LowResolution&, Sample& s, const Parameters& p, Session& session) {
    for (const auto& c : p.channels) {
        auto& a = session.arrays.at(c.key); Shape shape = a.spatial(); auto n = product(shape);
        Shape one_shape{1}; one_shape.insert(one_shape.end(),shape.begin(),shape.end());
        DeviceArray one(one_shape,a.kind,a.bytes);
        check(cudaMemcpyAsync(one.buffer->ptr,a.buffer->ptr+c.channel*n*a.bytes,n*a.bytes,cudaMemcpyDeviceToDevice));
        for (std::size_t j = 0; j < shape.size(); ++j) shape[j] = std::max<std::size_t>(1,std::size_t(round_even(shape[j]*c.values[j])));
        auto restored = resize_cuda(resize_cuda(one,shape,true,false),a.spatial(),false,false);
        check(cudaMemcpyAsync(a.buffer->ptr+c.channel*n*a.bytes,restored.buffer->ptr,n*a.bytes,cudaMemcpyDeviceToDevice));
    }
    session.restore_valid(s);
}
} // namespace vcr::augment::cuda
