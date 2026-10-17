#include "../pipeline.hpp"
#include "dispatch.hpp"
#include "local_field.cuh"
#include <cuda_runtime.h>
namespace vcr::augment::cuda {
template<class T> constexpr bool intensity_supported =
    std::is_same_v<T,BrightnessMultiply> || std::is_same_v<T,BrightnessAdd> || std::is_same_v<T,Contrast> ||
    std::is_same_v<T,Gamma> || std::is_same_v<T,Invert> || std::is_same_v<T,GaussianNoise> ||
    std::is_same_v<T,RicianNoise> || std::is_same_v<T,GaussianBlur> || std::is_same_v<T,MedianFilter> ||
    std::is_same_v<T,Sharpen> || std::is_same_v<T,BrightnessGradient> || std::is_same_v<T,LocalContrast> ||
    std::is_same_v<T,LocalGamma> || std::is_same_v<T,LocalSmoothing> || std::is_same_v<T,BlankRectangle> ||
    std::is_same_v<T,SliceIllumination> || std::is_same_v<T,Smear> || std::is_same_v<T,Normalize>;
template<class T> constexpr bool geometry_supported = std::is_same_v<T,Flip> || std::is_same_v<T,Rotate90> || std::is_same_v<T,Transpose> || std::is_same_v<T,Elastic>;
template<class T> constexpr bool supported = intensity_supported<T> || geometry_supported<T> || std::is_same_v<T,LowResolution> || std::is_same_v<T,ThickSlice> || std::is_same_v<T,RandomBinaryOperator>;

bool available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
void validate_program(const std::vector<Op>& ops) {
    for (const auto& op : ops) std::visit([](const auto& t) {
        using T = std::decay_t<decltype(t)>;
        if constexpr (std::is_same_v<T,std::shared_ptr<OneOf>>) validate_program(t->transforms);
        else require(supported<T>, "transform is not supported by the CUDA backend; use backend='cpu'");
    },op);
}
Parameters sample_parameters(const Op& op, const Sample& s, Random& r, bool record, void*) {
    return std::visit([&](const auto& t) -> Parameters {
        using T = std::decay_t<decltype(t)>;
        if constexpr (std::is_same_v<T,BrightnessGradient> || std::is_same_v<T,LocalContrast> ||
                      std::is_same_v<T,LocalGamma> || std::is_same_v<T,LocalSmoothing>) return sample_local(t,s,r,record);
        else if constexpr (std::is_same_v<T,Elastic>) return sample_elastic(t,s,r,record);
        else if constexpr (std::is_same_v<T,std::shared_ptr<OneOf>>) throw std::invalid_argument("unresolved OneOf");
        else return sample(t,s,r);
    },op);
}
void execute(const Op& op, Sample& s, const Parameters& p, void* context) {
    std::visit([&](const auto& t) {
        if constexpr (supported<std::decay_t<decltype(t)>>) apply_cuda(t,s,p,*static_cast<Session*>(context));
        else throw std::invalid_argument("transform is not supported by the CUDA backend");
    },op);
}

} // namespace vcr::augment::cuda
