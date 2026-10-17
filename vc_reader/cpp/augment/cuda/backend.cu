#include "../pipeline.hpp"
#include "dispatch.hpp"
#include "local_field.cuh"
#include <cuda_runtime.h>
namespace vcr::augment::cuda {
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
        if constexpr (std::is_same_v<T,BrightnessGradient> || std::is_same_v<T,LocalContrast> ||
                      std::is_same_v<T,LocalGamma> || std::is_same_v<T,LocalSmoothing>) return sample_local(t,s,r,record);
        else if constexpr (std::is_same_v<T,Elastic>) return sample_elastic(t,s,r,record);
        else if constexpr (std::is_same_v<T,StripGeometry>) return sample_strip(t,s,r,record);
        else if constexpr (std::is_same_v<T,SheetCompression>) return sample_sheet(t,s,r,record,*static_cast<Session*>(context));
        else if constexpr (std::is_same_v<T,SurfacePerturb>) return sample_surface(t,s,r,record,*static_cast<Session*>(context));
        else if constexpr (std::is_same_v<T,RemoveComponents>) return sample_components(t,s,r,record,*static_cast<Session*>(context));
        else if constexpr (std::is_same_v<T,ForegroundCrop>) return sample_foreground(t,s,r,*static_cast<Session*>(context));
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

} // namespace vcr::augment::cuda
