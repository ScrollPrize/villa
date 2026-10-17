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
    if (nonfinite) {
        int invalid = 0; nonfinite->download(&invalid);
        require(!invalid,"normalization requires finite crop values");
    }
    for (auto& [key,device] : arrays) {
        auto host = Array::uninitialized(device.shape,device.kind,device.bytes);
        device.buffer->download(static_cast<std::byte*>(host.data()));
        s.at(key).array = std::move(host);
    }
    check(cudaStreamSynchronize(0));
}
int* Session::finite_status() {
    if (!nonfinite) {
        nonfinite = std::make_unique<Buffer<int>>(1);
        check(cudaMemsetAsync(nonfinite->ptr,0,sizeof(int)));
    }
    return nonfinite->ptr;
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

#include "device_io.hpp"
namespace vcr::augment::cuda {
template<class T> __global__ void validate_device_values(View<T> data, bool mask, int* invalid) {
    EACH_INDEX(data.size) {
        if constexpr (std::is_floating_point_v<T>) if (!isfinite(data.ptr[i])) atomicOr(invalid,1);
        if (mask && data.ptr[i]!=0 && data.ptr[i]!=1) atomicOr(invalid,2);
    }
}
std::shared_ptr<void> begin_device(const std::map<std::string,DeviceInput>& inputs, Sample& sample, bool inplace) {
    auto session=std::make_shared<Session>();
    Buffer<int> invalid(1); int flags=0; check(cudaMemsetAsync(invalid.ptr,0,sizeof(int)));
    for (const auto& [key,in]:inputs) {
        auto& f=sample.at(key);
        require(!sparse(f.target.role) && !directional(f.target.role) && f.target.role!=Role::CoordinateField,
                "device augmentation supports image, scalar, label and validity-mask targets");
        DeviceArray array(in.shape,in.kind,in.bytes,in.data);
        array.typed([&](auto view) { if (view.size) validate_device_values<<<blocks(view.size),256>>>(view,f.target.role==Role::Valid,invalid.ptr); });
        check(cudaGetLastError());
        if (inplace) session->arrays.emplace(key,std::move(array));
        else {
            DeviceArray copy(in.shape,in.kind,in.bytes);
            check(cudaMemcpyAsync(copy.buffer->ptr,in.data,array.size()*array.bytes,cudaMemcpyDeviceToDevice));
            session->arrays.emplace(key,std::move(copy));
        }
    }
    invalid.download(&flags);
    require(!(flags&1),"nonfinite input; use a validity mask for missing data");
    require(!(flags&2),"validity masks must contain only zero or one");
    return session;
}
std::map<std::string,DeviceInput> device_outputs(void* context) {
    std::map<std::string,DeviceInput> out;
    for (const auto& [key,a]:static_cast<Session*>(context)->arrays) {
        a.buffer->exported=true;
        out.emplace(key,DeviceInput{a.buffer->ptr,a.shape,a.kind,a.bytes,a.buffer->device});
    }
    return out;
}
void finish_device(void* context) {
    auto& session=*static_cast<Session*>(context);
    if (session.nonfinite) {
        int invalid=0; session.nonfinite->download(&invalid);
        require(!invalid,"normalization requires finite crop values");
    }
    check(cudaStreamSynchronize(0));
}
}
