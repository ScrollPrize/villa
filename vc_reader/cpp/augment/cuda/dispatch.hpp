#pragma once
#include "../labels.hpp"
#include "../bias_field.hpp"
#include "../plasma_brightness_contrast.hpp"
#include "../plasma_shadow.hpp"
#include "../illumination.hpp"
#include "../vignetting.hpp"
#include "../random_tone_curve.hpp"
#include "../multiplicative_noise.hpp"
#include "../shot_noise.hpp"
#include "../additive_noise.hpp"
#include "../salt_and_pepper.hpp"
#include "../motion_blur.hpp"
#include "../defocus.hpp"
#include "../zoom_blur.hpp"
#include "../equalize.hpp"
#include "../clahe.hpp"
#include "../histogram_matching.hpp"
#include "../pixel_distribution_adaptation.hpp"
#include "../fourier_domain_adaptation.hpp"
#include "../unsharp_mask.hpp"
#include "../gibbs_ringing.hpp"
#include "../ringing_overshoot.hpp"
#include "../foreground_crop.hpp"
#include "../channel_misalignment.hpp"
#include "../grid_dropout.hpp"
#include "../stripe_masking.hpp"
#include "../foreground_dropout.hpp"
#include "../object_dropout.hpp"
#include "resident.cuh"
namespace vcr::augment::cuda {
Parameters sample_foreground(const ForegroundCrop&, const Sample&, Random&, Session&);
Parameters sample_dropout(const GridDropout&, const Sample&, Random&, bool, Session&);
Parameters sample_dropout(const StripeMasking&, const Sample&, Random&, bool, Session&);
Parameters sample_dropout(const ForegroundDropout&, const Sample&, Random&, bool, Session&);
Parameters sample_dropout(const ObjectDropout&, const Sample&, Random&, bool, Session&);
void apply_cuda(const BiasField&, Sample&, const Parameters&, Session&);
void apply_cuda(const PlasmaBrightnessContrast&, Sample&, const Parameters&, Session&);
void apply_cuda(const PlasmaShadow&, Sample&, const Parameters&, Session&);
void apply_cuda(const Illumination&, Sample&, const Parameters&, Session&);
void apply_cuda(const Vignetting&, Sample&, const Parameters&, Session&);
void apply_cuda(const RandomToneCurve&, Sample&, const Parameters&, Session&);
void apply_cuda(const MultiplicativeNoise&, Sample&, const Parameters&, Session&);
void apply_cuda(const ShotNoise&, Sample&, const Parameters&, Session&);
void apply_cuda(const AdditiveNoise&, Sample&, const Parameters&, Session&);
void apply_cuda(const SaltAndPepper&, Sample&, const Parameters&, Session&);
void apply_cuda(const MotionBlur&, Sample&, const Parameters&, Session&);
void apply_cuda(const Defocus&, Sample&, const Parameters&, Session&);
void apply_cuda(const ZoomBlur&, Sample&, const Parameters&, Session&);
void apply_cuda(const Equalize&, Sample&, const Parameters&, Session&);
void apply_cuda(const CLAHE&, Sample&, const Parameters&, Session&);
void apply_cuda(const HistogramMatching&, Sample&, const Parameters&, Session&);
void apply_cuda(const PixelDistributionAdaptation&, Sample&, const Parameters&, Session&);
void apply_cuda(const FDA&, Sample&, const Parameters&, Session&);
void apply_cuda(const UnsharpMask&, Sample&, const Parameters&, Session&);
void apply_cuda(const GibbsRinging&, Sample&, const Parameters&, Session&);
void apply_cuda(const RingingOvershoot&, Sample&, const Parameters&, Session&);
void apply_cuda(const GridDropout&, Sample&, const Parameters&, Session&);
void apply_cuda(const StripeMasking&, Sample&, const Parameters&, Session&);
void apply_cuda(const ForegroundDropout&, Sample&, const Parameters&, Session&);
void apply_cuda(const ObjectDropout&, Sample&, const Parameters&, Session&);
void apply_cuda(const ForegroundCrop&, Sample&, const Parameters&, Session&);
void apply_cuda(const ChannelMisalignment&, Sample&, const Parameters&, Session&);
void apply_cuda(const Normalize&, Sample&, const Parameters&, Session&);
void apply_cuda(const BrightnessMultiply&, Sample&, const Parameters&, Session&);
void apply_cuda(const BrightnessAdd&, Sample&, const Parameters&, Session&);
void apply_cuda(const Contrast&, Sample&, const Parameters&, Session&);
void apply_cuda(const Invert&, Sample&, const Parameters&, Session&);
void apply_cuda(const BrightnessGradient&, Sample&, const Parameters&, Session&);
void apply_cuda(const LocalContrast&, Sample&, const Parameters&, Session&);
void apply_cuda(const LocalGamma&, Sample&, const Parameters&, Session&);
void apply_cuda(const SliceIllumination&, Sample&, const Parameters&, Session&);
void apply_cuda(const GaussianBlur&, Sample&, const Parameters&, Session&);
void apply_cuda(const LocalSmoothing&, Sample&, const Parameters&, Session&);
void apply_cuda(const Sharpen&, Sample&, const Parameters&, Session&);
void apply_cuda(const MedianFilter&, Sample&, const Parameters&, Session&);
void apply_cuda(const GaussianNoise&, Sample&, const Parameters&, Session&);
void apply_cuda(const RicianNoise&, Sample&, const Parameters&, Session&);
void apply_cuda(const BlankRectangle&, Sample&, const Parameters&, Session&);
void apply_cuda(const Smear&, Sample&, const Parameters&, Session&);
void apply_cuda(const Gamma&, Sample&, const Parameters&, Session&);
void apply_cuda(const Flip&, Sample&, const Parameters&, Session&);
void apply_cuda(const Rotate90&, Sample&, const Parameters&, Session&);
void apply_cuda(const Transpose&, Sample&, const Parameters&, Session&);
void apply_cuda(const Elastic&, Sample&, const Parameters&, Session&);
void apply_cuda(const LowResolution&, Sample&, const Parameters&, Session&);
void apply_cuda(const ThickSlice&, Sample&, const Parameters&, Session&);
void apply_cuda(const RandomBinaryOperator&, Sample&, const Parameters&, Session&);
void apply_cuda(const MedialSurface&, Sample&, const Parameters&, Session&);
void apply_cuda(const OrientedBlur&, Sample&, const Parameters&, Session&);
void apply_cuda(const DistanceTransform&, Sample&, const Parameters&, Session&);
void apply_cuda(const StructureTensor&, Sample&, const Parameters&, Session&);
} // namespace vcr::augment::cuda
