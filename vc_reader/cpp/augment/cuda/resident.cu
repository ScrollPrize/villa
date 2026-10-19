#include "resident.cuh"
#include "../../cuda_buffer.hpp"
namespace vcr::augment::cuda {
Session::Session(const Sample& s) {
    for (const auto& [key,f] : s) {
        if (sparse(f.target.role)) continue;
        auto& device = arrays.emplace(key,DeviceArray(f.array.shape,f.array.kind,f.array.bytes)).first->second;
        device.buffer->upload(static_cast<const std::byte*>(f.array.data()));
    }
}
int* Session::check_flags() {
    if (!checks) {
        checks = std::make_unique<Buffer<int>>(1);
        check(memset_async(checks->ptr,0,sizeof(int),vcr::device::stream()));
    }
    return checks->ptr;
}
static void validate_status(int checks, int nonfinite, int geometry) {
    require(!(checks&1),"nonfinite input; use a validity mask for missing data");
    require(!(checks&2),"validity masks must contain only zero or one");
    require(!(checks&8),"value-based label transforms require integer IDs within +/- (2**53 - 1)");
    require(!(checks&16),"value is not representable in target dtype");
    require(!(checks&4),"values too large for float32 filtering without the float64 fallback; use sync=True");
    require(!nonfinite,"normalization requires finite crop values");
    require(!(geometry&2),"point outside output");
    require(!(geometry&1),"deformation inverse failed to converge for a point; reduce deformation strength");
    require(!(geometry&4),"singular spatial transform");
}
void Session::check_status() {
    int a=0,b=0,c=0;
    if (checks) checks->download(&a);
    if (nonfinite) nonfinite->download(&b);
    if (geometry) geometry->download(&c);
    validate_status(a,b,c);
}
struct gather_status_launch {
    __device__ void operator()(KernelIndex, int* output, const int* checks,
                               const int* nonfinite, const int* geometry) const {
        output[0]=checks ? *checks : 0;
        output[1]=nonfinite ? *nonfinite : 0;
        output[2]=geometry ? *geometry : 0;
    }
};
void check_devices(const std::vector<std::shared_ptr<void>>& contexts) {
    // Pack batch status into one transfer while retaining sample order for
    // validation. Individual scalar transfers cost more than the data itself.
    thread_local vcr::device::PinnedBuffer staging;
    staging.reserve(contexts.size()*3*sizeof(int));
    auto* flags=reinterpret_cast<int*>(staging.data());
    std::fill_n(flags,contexts.size()*3,0);
    std::size_t transfers=0;
    for (const auto& context:contexts) {
        const auto& s=*static_cast<Session*>(context.get());
        transfers+=bool(s.checks)+bool(s.nonfinite)+bool(s.geometry);
    }
    bool pending=false;
    try {
        // A handful of scalar reads are cheaper than an allocation and launch.
        if (transfers>4) {
            Buffer<int> packed(contexts.size()*3);
            BatchQueue queue(contexts.size());
            for (std::size_t i=0;i<contexts.size();++i) {
                queue.lane(i);
                const auto& s=*static_cast<Session*>(contexts[i].get());
                launch(1,1,0,vcr::device::stream(),gather_status_launch{},packed.ptr+3*i,
                       s.checks ? s.checks->ptr : nullptr,
                       s.nonfinite ? s.nonfinite->ptr : nullptr,
                       s.geometry ? s.geometry->ptr : nullptr);
            }
            queue.flush();
            check(copy_async(flags,packed.ptr,packed.size*sizeof(int),cudaMemcpyDeviceToHost,vcr::device::stream()));
            check(cudaStreamSynchronize(vcr::device::stream()));
        } else {
            for (std::size_t i=0;i<contexts.size();++i) {
                auto& s=*static_cast<Session*>(contexts[i].get());
                const Buffer<int>* buffers[]={s.checks.get(),s.nonfinite.get(),s.geometry.get()};
                for (int j=0;j<3;++j) if (buffers[j]) {
                    check(copy_async(flags+3*i+j,buffers[j]->ptr,sizeof(int),cudaMemcpyDeviceToHost,vcr::device::stream()));
                    pending=true;
                }
            }
            if (pending) check(cudaStreamSynchronize(vcr::device::stream()));
        }
    } catch (...) {
        cudaStreamSynchronize(vcr::device::stream());
        throw;
    }
    for (std::size_t i=0;i<contexts.size();++i) validate_status(flags[3*i],flags[3*i+1],flags[3*i+2]);
}
int* Session::geometry_status() {
    if (!geometry) {
        geometry = std::make_unique<Buffer<int>>(1);
        check(memset_async(geometry->ptr,0,sizeof(int),vcr::device::stream()));
    }
    return geometry->ptr;
}
void Session::finish(Sample& s) {
    check_status();
    for (auto& [key,device] : arrays) {
        auto host = Array::uninitialized(device.shape,device.kind,device.bytes);
        device.buffer->download(static_cast<std::byte*>(host.data()));
        s.at(key).array = std::move(host);
    }
    for (const auto& [key,levels] : pyramids) {
        auto& output=s.at(key).pyramid; output.clear();
        for (const auto& level:levels) {
            auto host=Array::uninitialized(level.shape,level.kind,level.bytes);
            level.buffer->download(static_cast<std::byte*>(host.data()));output.push_back(std::move(host));
        }
    }
    check(cudaStreamSynchronize(vcr::device::stream()));
}
int* Session::finite_status() {
    if (!nonfinite) {
        nonfinite = std::make_unique<Buffer<int>>(1);
        check(memset_async(nonfinite->ptr,0,sizeof(int),vcr::device::stream()));
    }
    return nonfinite->ptr;
}
template<class T, class M> __device__ void restore_kernel(KernelIndex vcr_launch, T* image, const M* valid, std::size_t n, std::size_t mask_size, double fill) {
    EACH_INDEX(n) if (!valid[i%mask_size]) image[i] = T(fill);
}
struct restore_kernel_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { restore_kernel(index,args...); }
};
void Session::restore_valid(const Sample& s) {
    for (const auto& [key,f] : s) {
        if (f.target.role != Role::Image || !s.count(f.target.valid)) continue;
        auto& image = arrays.at(key); const auto& mask = arrays.at(f.target.valid);
        require(image.spatial() == mask.spatial() && (mask.shape[0] == 1 || mask.shape[0] == image.shape[0]),"validity mask no longer matches image channels");
        mask.typed([&](auto valid) {
            if (image.bytes == 4) launch(blocks(image.size()),256,0,vcr::device::stream(), restore_kernel_launch{}, image.view<float>().ptr,valid.ptr,image.size(),mask.size(),f.target.fill);
            else launch(blocks(image.size()),256,0,vcr::device::stream(), restore_kernel_launch{}, image.view<double>().ptr,valid.ptr,image.size(),mask.size(),f.target.fill);
        });
        check(cudaGetLastError());
    }
}
std::shared_ptr<void> begin(const Sample& s) { return std::make_shared<Session>(s); }
void finish(Sample& s, void* context) { static_cast<Session*>(context)->finish(s); }
void synchronize() { check(cudaStreamSynchronize(vcr::device::stream())); }
} // namespace vcr::augment::cuda

#include "device_io.hpp"
namespace vcr::augment::cuda {
template<class T> __device__ void validate_device_values(KernelIndex vcr_launch, View<T> data, T* output, bool mask, int* invalid) {
    EACH_INDEX(data.size) {
        if constexpr (std::is_floating_point_v<T>) if (!isfinite(data.ptr[i])) atomicOr(invalid,1);
        if (mask && data.ptr[i]!=0 && data.ptr[i]!=1) atomicOr(invalid,2);
        if (output != data.ptr) output[i]=data.ptr[i];
    }
}
struct validate_device_values_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { validate_device_values(index,args...); }
};
std::shared_ptr<void> begin_device(const std::map<std::string,DeviceInput>& inputs, Sample& sample, bool inplace, bool async,
                                   const std::map<std::string,DeviceInput>& destinations) {
    auto session=std::make_shared<Session>(); session->async=async;
    Buffer<int> invalid(async ? 0 : 1); int flags=0;
    int* status=async ? session->check_flags() : invalid.ptr;
    if (!async) check(memset_async(invalid.ptr,0,sizeof(int),vcr::device::stream()));
    for (const auto& [key,in]:inputs) {
        auto& f=sample.at(key);
        DeviceArray array(in.shape,in.kind,in.bytes,in.data);
        // Validate while copying into the working output, avoiding a second
        // read of the entire batch. In-place callers only validate.
        DeviceArray output = inplace ? array : destinations.count(key)
            ? DeviceArray(in.shape,in.kind,in.bytes,destinations.at(key).data)
            : DeviceArray(in.shape,in.kind,in.bytes);
        array.typed([&](auto view) {
            using T=std::remove_pointer_t<decltype(view.ptr)>;
            if (view.size) launch(blocks(view.size),256,0,vcr::device::stream(),validate_device_values_launch{},
                                  view,reinterpret_cast<T*>(output.buffer->ptr),f.target.role==Role::Valid,status);
        });
        check(cudaGetLastError());
        session->arrays.emplace(key,std::move(output));
    }
    if (async) return session;
    invalid.download(&flags);
    require(!(flags&1),"nonfinite input; use a validity mask for missing data");
    require(!(flags&2),"validity masks must contain only zero or one");
    return session;
}
std::map<std::string,DeviceInput> device_outputs(void* context) {
    std::map<std::string,DeviceInput> out;
    for (const auto& [key,a]:static_cast<Session*>(context)->arrays)
        out.emplace(key,DeviceInput{a.buffer->ptr,a.shape,a.kind,a.bytes,a.buffer->device});
    return out;
}
void copy_batch_output(void* destination,const std::vector<DeviceInput>& inputs) {
    BatchQueue copies(inputs.size());
    std::size_t offset=0;
    for(std::size_t b=0;b<inputs.size();++b) {
        copies.lane(b);
        const auto& input=inputs[b];const auto bytes=product(input.shape)*input.bytes;
        check(copy_async(static_cast<std::byte*>(destination)+offset,input.data,bytes,cudaMemcpyDeviceToDevice,vcr::device::stream()));
        offset+=bytes;
    }
    copies.flush();
}
std::map<std::string,std::vector<DeviceInput>> device_pyramids(void* context) {
    std::map<std::string,std::vector<DeviceInput>> out;
    for (const auto& [key,levels]:static_cast<Session*>(context)->pyramids)
        for (const auto& a:levels) out[key].push_back(DeviceInput{a.buffer->ptr,a.shape,a.kind,a.bytes,a.buffer->device});
    return out;
}
void export_outputs(void* context) {
    for (auto& [key,a]:static_cast<Session*>(context)->arrays) a.buffer->exported=true;
    for (auto& [key,levels]:static_cast<Session*>(context)->pyramids) for (auto& a:levels) a.buffer->exported=true;
}
void check_device(void* context) { static_cast<Session*>(context)->check_status(); }
void release_outputs(void* context, const std::vector<std::string>& keys) {
    for (const auto& key:keys) static_cast<Session*>(context)->arrays.erase(key);
}
void finish_device(void* context) {
    auto& session=*static_cast<Session*>(context);
    if (session.async) return;
    session.check_status();
    check(cudaStreamSynchronize(vcr::device::stream()));
}
}
