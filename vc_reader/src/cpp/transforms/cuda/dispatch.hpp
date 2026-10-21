#pragma once
#include "../cpu/tracing.hpp"
#include "../cpu/bias_field.hpp"
#include "../cpu/plasma_brightness_contrast.hpp"
#include "../cpu/plasma_shadow.hpp"
#include "../cpu/illumination.hpp"
#include "../cpu/vignetting.hpp"
#include "../cpu/random_tone_curve.hpp"
#include "../cpu/multiplicative_noise.hpp"
#include "../cpu/shot_noise.hpp"
#include "../cpu/additive_noise.hpp"
#include "../cpu/salt_and_pepper.hpp"
#include "../cpu/motion_blur.hpp"
#include "../cpu/defocus.hpp"
#include "../cpu/zoom_blur.hpp"
#include "../cpu/equalize.hpp"
#include "../cpu/clahe.hpp"
#include "../cpu/histogram_matching.hpp"
#include "../cpu/pixel_distribution_adaptation.hpp"
#include "../cpu/fourier_domain_adaptation.hpp"
#include "../cpu/unsharp_mask.hpp"
#include "../cpu/gibbs_ringing.hpp"
#include "../cpu/ringing_overshoot.hpp"
#include "../cpu/foreground_crop.hpp"
#include "../cpu/channel_misalignment.hpp"
#include "../cpu/grid_dropout.hpp"
#include "../cpu/stripe_masking.hpp"
#include "../cpu/foreground_dropout.hpp"
#include "../cpu/object_dropout.hpp"
#include "resident.cuh"
namespace vcr::transforms::cuda {
void apply_cuda(const ClipOutliers&, Sample&, const Parameters&, Session&);
void apply_cuda(const CropPad&, Sample&, const Parameters&, Session&);
void apply_cuda(const Affine&, Sample&, const Parameters&, Session&);
void apply_cuda(const SheetCompression&, Sample&, const Parameters&, Session&);
void apply_cuda(const StripGeometry&, Sample&, const Parameters&, Session&);
void apply_cuda(const SurfacePerturb&, Sample&, const Parameters&, Session&);
void apply_cuda(const Morphology&, Sample&, const Parameters&, Session&);
void apply_cuda(const RemoveComponents&, Sample&, const Parameters&, Session&);
void apply_cuda(const ReplaceLabel&, Sample&, const Parameters&, Session&);
void apply_cuda(const Regions&, Sample&, const Parameters&, Session&);
void apply_cuda(const OneHot&, Sample&, const Parameters&, Session&);
void apply_cuda(const MaskImage&, Sample&, const Parameters&, Session&);
void apply_cuda(const DeepSupervision&, Sample&, const Parameters&, Session&);
void apply_cuda(const ToPseudo2D&, Sample&, const Parameters&, Session&);
void apply_cuda(const FromPseudo2D&, Sample&, const Parameters&, Session&);
Parameters sample_strip(const StripGeometry&, const Sample&, Random&, bool);
Parameters sample_sheet(const SheetCompression&, const Sample&, Random&, bool, Session&);
Parameters sample_surface(const SurfacePerturb&, const Sample&, Random&, bool, Session&);
Parameters sample_components(const RemoveComponents&, const Sample&, Random&, bool, Session&);
Parameters sample_foreground(const ForegroundCrop&, const Sample&, Random&, bool, Session&);
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
} // namespace vcr::transforms::cuda
