#pragma once
#include <type_traits>

namespace vcr::augment {
struct RandomToneCurve; struct SaltAndPepper; struct Equalize; struct CLAHE;
struct HistogramMatching; struct PlasmaBrightnessContrast; struct PixelDistributionAdaptation;

// Shared CPU/CUDA requirements: pointwise transforms that do not consume a
// channel summary must not pay for a whole-image statistics pass.
template<class T> constexpr bool channel_moments = std::is_same_v<T,PlasmaBrightnessContrast> || std::is_same_v<T,PixelDistributionAdaptation>;
template<class T> constexpr bool channel_bounds = channel_moments<T> || std::is_same_v<T,RandomToneCurve> || std::is_same_v<T,SaltAndPepper> || std::is_same_v<T,Equalize> || std::is_same_v<T,CLAHE> || std::is_same_v<T,HistogramMatching>;
}
