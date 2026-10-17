#include "resident.cuh"
namespace vcr::augment::cuda {
Session::Session(const Sample& s) {
    for (const auto& [key,f] : s) {
        if (sparse(f.target.role)) continue;
        auto& device = arrays.emplace(key,DeviceArray(f.array.shape,f.array.kind,f.array.bytes)).first->second;
        device.buffer->upload(static_cast<const std::byte*>(f.array.data()));
    }
}
void Session::finish(Sample& s) {
    for (auto& [key,device] : arrays) {
        auto host = Array::uninitialized(device.shape,device.kind,device.bytes);
        device.buffer->download(static_cast<std::byte*>(host.data()));
        s.at(key).array = std::move(host);
    }
    check(cudaStreamSynchronize(0));
}
template<class T, class M> __global__ void restore_kernel(T* image, const M* valid, std::size_t n, std::size_t mask_size, double fill) {
    EACH_INDEX(n) if (!valid[i%mask_size]) image[i] = T(fill);
}
void Session::restore_valid(const Sample& s) {
    for (const auto& [key,f] : s) {
        if (f.target.role != Role::Image || !s.count(f.target.valid)) continue;
        auto& image = arrays.at(key); const auto& mask = arrays.at(f.target.valid);
        require(image.spatial() == mask.spatial() && (mask.shape[0] == 1 || mask.shape[0] == image.shape[0]),"validity mask no longer matches image channels");
        mask.typed([&](auto valid) {
            if (image.bytes == 4) restore_kernel<<<blocks(image.size()),256>>>(image.view<float>().ptr,valid.ptr,image.size(),mask.size(),f.target.fill);
            else restore_kernel<<<blocks(image.size()),256>>>(image.view<double>().ptr,valid.ptr,image.size(),mask.size(),f.target.fill);
        });
        check(cudaGetLastError());
    }
}
std::shared_ptr<void> begin(const Sample& s) { return std::make_shared<Session>(s); }
void finish(Sample& s, void* context) { static_cast<Session*>(context)->finish(s); }
void synchronize() { check(cudaStreamSynchronize(0)); }
} // namespace vcr::augment::cuda
