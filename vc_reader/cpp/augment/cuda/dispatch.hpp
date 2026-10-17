#pragma once
#include "../labels.hpp"
#include "resident.cuh"
namespace vcr::augment::cuda {
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
} // namespace vcr::augment::cuda
