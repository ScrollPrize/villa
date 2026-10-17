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
    std::is_same_v<T,SliceIllumination> || std::is_same_v<T,Smear> || std::is_same_v<T,Normalize> || std::is_same_v<T,OrientedBlur>;
template<class T> constexpr bool geometry_supported = std::is_same_v<T,Flip> || std::is_same_v<T,Rotate90> || std::is_same_v<T,Transpose> || std::is_same_v<T,Elastic>;
template<class T> constexpr bool supported = intensity_supported<T> || geometry_supported<T> || std::is_same_v<T,LowResolution> || std::is_same_v<T,ThickSlice> || std::is_same_v<T,RandomBinaryOperator> || std::is_same_v<T,MedialSurface> ||
    std::is_same_v<T,DistanceTransform> || std::is_same_v<T,StructureTensor> || std::is_same_v<T,BiasField> || std::is_same_v<T,PlasmaBrightnessContrast> || std::is_same_v<T,PlasmaShadow> || std::is_same_v<T,Illumination> || std::is_same_v<T,Vignetting> || std::is_same_v<T,RandomToneCurve> || std::is_same_v<T,MultiplicativeNoise> || std::is_same_v<T,ShotNoise> || std::is_same_v<T,AdditiveNoise> || std::is_same_v<T,SaltAndPepper> || std::is_same_v<T,MotionBlur> || std::is_same_v<T,Defocus> || std::is_same_v<T,ZoomBlur> || std::is_same_v<T,Equalize> || std::is_same_v<T,CLAHE> || std::is_same_v<T,HistogramMatching> || std::is_same_v<T,PixelDistributionAdaptation> || std::is_same_v<T,FDA> || std::is_same_v<T,UnsharpMask> || std::is_same_v<T,GibbsRinging> || std::is_same_v<T,RingingOvershoot> || std::is_same_v<T,ForegroundCrop> || std::is_same_v<T,ChannelMisalignment> || std::is_same_v<T,GridDropout> || std::is_same_v<T,StripeMasking> || std::is_same_v<T,ForegroundDropout> || std::is_same_v<T,ObjectDropout>;

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
Parameters sample_parameters(const Op& op, const Sample& s, Random& r, bool record, void* context) {
    return std::visit([&](const auto& t) -> Parameters {
        using T = std::decay_t<decltype(t)>;
        if constexpr (std::is_same_v<T,BrightnessGradient> || std::is_same_v<T,LocalContrast> ||
                      std::is_same_v<T,LocalGamma> || std::is_same_v<T,LocalSmoothing>) return sample_local(t,s,r,record);
        else if constexpr (std::is_same_v<T,Elastic>) return sample_elastic(t,s,r,record);
        else if constexpr (std::is_same_v<T,ForegroundCrop>) return sample_foreground(t,s,r,*static_cast<Session*>(context));
        else if constexpr (std::is_same_v<T,GridDropout> || std::is_same_v<T,StripeMasking> || std::is_same_v<T,ForegroundDropout> || std::is_same_v<T,ObjectDropout>) return sample_dropout(t,s,r,record,*static_cast<Session*>(context));
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
