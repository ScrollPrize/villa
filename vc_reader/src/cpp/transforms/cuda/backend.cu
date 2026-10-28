#include "../pipeline.hpp"
#include "dispatch.hpp"
#include "local_field.cuh"
#include <cuda_runtime.h>
namespace vcr::transforms::cuda {
// Exhaustive dispatch is a build-time coverage check: adding any CPU Op
// requires a CUDA overload (and its definition at link time).
template<class T> concept Supported = requires(const T& t, Sample& s, const Parameters& p, Session& session) { apply_cuda(t,s,p,session); };

bool available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
void validate_program(const std::vector<Op>& ops) {
    for (const auto& op : ops) std::visit([](const auto& t) {
        using T = std::decay_t<decltype(t)>;
        if constexpr (std::is_same_v<T,std::shared_ptr<OneOf>>) validate_program(t->transforms);
        else static_assert(Supported<T>, "every CPU transform must have a CUDA implementation");
    },op);
}
Parameters sample_parameters(const Op& op, const Sample& s, Random& r, bool record, void* context) {
    return std::visit([&](const auto& t) -> Parameters {
        using T = std::decay_t<decltype(t)>;
        if constexpr (std::is_same_v<T,Elastic>) return sample_elastic(t,s,r,record);
        else if constexpr (std::is_same_v<T,StripGeometry>) return sample_strip(t,s,r,record);
        else if constexpr (std::is_same_v<T,SheetCompression>) return sample_sheet(t,s,r,record,*static_cast<Session*>(context));
        else if constexpr (std::is_same_v<T,SurfacePerturb>) return sample_surface(t,s,r,record,*static_cast<Session*>(context));
        else if constexpr (std::is_same_v<T,RemoveComponents>) return sample_components(t,s,r,record,*static_cast<Session*>(context));
        else if constexpr (std::is_same_v<T,ForegroundCrop>) return sample_foreground(t,s,r,record,*static_cast<Session*>(context));
        else if constexpr (std::is_same_v<T,GridDropout> || std::is_same_v<T,StripeMasking> || std::is_same_v<T,ForegroundDropout> || std::is_same_v<T,ObjectDropout>) return sample_dropout(t,s,r,record,*static_cast<Session*>(context));
        else if constexpr (std::is_same_v<T,std::shared_ptr<OneOf>>) throw std::invalid_argument("unresolved OneOf");
        else return sample(t,s,r);
    },op);
}
void execute(const Op& op, Sample& s, const Parameters& p, void* context) {
    std::visit([&](const auto& t) {
        if constexpr (std::is_same_v<std::decay_t<decltype(t)>,std::shared_ptr<OneOf>>) throw std::invalid_argument("unresolved OneOf");
        else apply_cuda(t,s,p,*static_cast<Session*>(context));
    },op);
}

void run_batch_steps(const Program& program, std::vector<Sample>& samples,
                     const std::vector<std::uint64_t>& seeds,
                     const std::map<std::string,std::vector<double>>& params,
                     const std::vector<std::shared_ptr<void>>& contexts) {
    BatchQueue queue(samples.size());
    int device=0, cache_bytes=0;
    check(cudaGetDevice(&device));
    check(cudaDeviceGetAttribute(&cache_bytes,cudaDevAttrL2CacheSize,device));
    // Leave cache space for resident inputs/outputs and library workspaces.
    // Flush only between samples, after all of their dependencies are queued.
    const auto scratch_budget=std::max<std::size_t>(1,std::size_t(cache_bytes)/4);
    for (std::size_t first=0;first<samples.size();) {
        // Keep a group's resident arrays hot across pipeline steps. Single
        // stages need only the scratch bound, so pointwise batches stay fused.
        std::size_t end=samples.size();
        if (program.transforms.size()>1) {
            std::size_t resident_bytes=0;
            end=first;
            do {
                const auto& session=*static_cast<Session*>(contexts[end].get());
                std::size_t bytes=0;
                for (const auto& [name,array]:session.arrays) bytes+=array.size()*array.bytes;
                if (end>first && resident_bytes+bytes>std::size_t(cache_bytes)/2) break;
                resident_bytes+=bytes;
                ++end;
            } while (end<samples.size());
        }
        for (std::size_t step=0;step<program.transforms.size();++step) {
            for (std::size_t b=first;b<end;++b) {
                queue.lane(b);
                auto& sample=samples[b];
                for (const auto& [name,f]:sample) require(f.pyramid.empty(),"DeepSupervision must be the last transform");
                NamedValues values;
                for (const auto& [name,v]:params) values[name]=v[b];
                NamedScope named(&values);
                Random random{split_seed(seeds[b],std::to_string(step))};
                CompactState compact(false);
                prepare(program.transforms[step],sample,random,{step},nullptr,compact,execute,contexts[b].get());
                if (queue.allocated_bytes()>=scratch_budget) queue.flush();
            }
            queue.flush();
        }
        first=end;
    }
    for (const auto& context:contexts) finish_device(context.get());
}

} // namespace vcr::transforms::cuda
