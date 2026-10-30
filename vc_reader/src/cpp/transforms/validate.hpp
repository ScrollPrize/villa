#pragma once
// Checks of a transform's configuration, independent of any sample. Pipeline runs
// them over its whole program, so every caller gets them however it built a transform.
#include "ops/channel_misalignment.hpp"
#include "ops/distance_transform.hpp"
#include "ops/foreground_crop.hpp"
#include "ops/foreground_dropout.hpp"
#include "ops/fourier_domain_adaptation.hpp"
#include "ops/grid_dropout.hpp"
#include "ops/histogram_matching.hpp"
#include "ops/intensity.hpp"
#include "ops/normalization.hpp"
#include "ops/object_dropout.hpp"
#include "ops/oriented_blur.hpp"
#include "ops/pixel_distribution_adaptation.hpp"
#include "ops/spatial.hpp"
#include "ops/stripe_masking.hpp"
#include "ops/structure_tensor.hpp"

namespace vcr::transforms {
namespace volume_detail {
inline void check(const EraseValues& fill) { require(std::isfinite(fill.image) && std::isfinite(fill.label), "fill values must be finite"); }
inline void check(const ReferenceVolume& reference) {
    require(!reference.shape.empty(), "reference volume is required");
    require(reference.shape.size() == 3 || reference.shape.size() == 4, "reference must have shape (C,Y,X) or (C,Z,Y,X)");
    require(product(reference.shape) > 0, "reference must be nonempty");
    const auto& values = reference.values;
    require(values.shape == reference.shape && values.kind == 'f' && reference.wide == (values.bytes == 8), "reference must be float32 or float64");
    values.typed([&](const auto* data) { for (std::size_t i = 0; i < values.size(); ++i) require(std::isfinite(data[i]), "reference must be finite"); });
}
// A reference holding its own copy of the values.
inline ReferenceVolume reference_volume(const Array& values) {
    require(values.kind == 'f', "reference must be float32 or float64");
    return {values.shape, values, values.bytes == 8};
}
}  // namespace volume_detail

inline void check(const Gamma& t) { probability(t.p_invert); probability(t.p_retain_stats); }
inline void check(const ClipOutliers& t) { probability(t.p_retain_std); }
inline void check(const Sharpen& t) { probability(t.p_clamp); }
inline void check(const Flip& t) { probability(t.p_axis); }
inline void check(const Normalize& t) {
    require(std::isfinite(t.eps) && t.eps >= 0, "eps must be finite and nonnegative");
    require(std::isfinite(t.lower) && std::isfinite(t.upper) && 0 <= t.lower && t.lower <= t.upper && t.upper <= 100,
            "percentiles must satisfy 0 <= lower <= upper <= 100");
    require(std::isfinite(t.center) && std::isfinite(t.scale) && t.scale != 0, "fixed normalization needs finite center and nonzero finite scale");
    require(!std::isnan(t.low) && !std::isnan(t.high) && t.low <= t.high, "bounds must satisfy low <= high");
}
inline void check(const OrientedBlur& t) {
    if (t.axis.empty()) return;
    require((t.axis.size() == 2 || t.axis.size() == 3) && std::all_of(t.axis.begin(), t.axis.end(), [](double v) { return std::isfinite(v); }) &&
            std::any_of(t.axis.begin(), t.axis.end(), [](double v) { return v != 0; }), "axis must be a finite nonzero (y, x) or (z, y, x) vector");
}
inline void check(const DistanceTransform& t) { require(t.suffix != t.normals && !t.suffix.empty(), "distance and normal suffixes must be distinct and nonempty"); }
inline void check(const StructureTensor& t) {
    require(std::isfinite(t.sigma) && t.sigma >= 0 && t.sigma < 1000 && std::isfinite(t.rho) && t.rho >= 0 && t.rho < 1000, "sigma and rho must be in [0, 1000)");
    require(t.derivative == Derivative::Pavel || t.sigma > 0, "gaussian derivatives need sigma > 0");
    require(std::isfinite(t.truncate) && t.truncate > 0 && std::isfinite(t.scale), "truncate must be positive and scale finite");
    require(t.at.empty() || (t.rho > 0 && t.field == StructureField::Components && t.confidence.empty()),
            "point evaluation (at) returns the tensor and needs rho > 0");
    require(std::all_of(t.at.begin(), t.at.end(), [](double v) { return std::isfinite(v); }), "at must be finite");
}
inline void check(const ForegroundCrop& t) {
    require((t.shape.size() == 2 || t.shape.size() == 3) && product(t.shape) > 0, "shape needs two or three positive sizes");
    require(std::isfinite(t.threshold), "threshold must be finite");
}
inline void check(const ForegroundDropout& t) { require(std::isfinite(t.threshold), "threshold must be finite"); volume_detail::check(t.fill); }
inline void check(const ObjectDropout& t) { require(std::isfinite(t.threshold), "threshold must be finite"); volume_detail::check(t.fill); }
inline void check(const GridDropout& t) { volume_detail::check(t.fill); }
inline void check(const StripeMasking& t) { volume_detail::check(t.fill); }
inline void check(const HistogramMatching& t) { volume_detail::check(t.reference); }
inline void check(const PixelDistributionAdaptation& t) { volume_detail::check(t.reference); }
inline void check(const FDA& t) { volume_detail::check(t.reference); }

template<class T> void validate_transform(const T& t) {
    if constexpr (requires { t.options; }) { probability(t.options.p); probability(t.options.p_channel); }
    else if constexpr (requires { t.p; }) probability(t.p);
    if constexpr (requires { check(t); }) check(t);
}
}  // namespace vcr::transforms
