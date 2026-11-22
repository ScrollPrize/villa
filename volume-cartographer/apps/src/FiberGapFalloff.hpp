#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

// Falloff of a dimensionless (scaled) gap g, shared by the crossing and the
// neighbour-normal weights.
enum class GapFalloff { None, Inverse, InverseSquare, Gaussian, Cauchy, InverseSmooth };

inline GapFalloff parseGapFalloff(const std::string& name)
{
    if (name == "none") return GapFalloff::None;
    if (name == "inverse") return GapFalloff::Inverse;
    if (name == "inverse_square") return GapFalloff::InverseSquare;
    if (name == "gaussian") return GapFalloff::Gaussian;
    if (name == "cauchy") return GapFalloff::Cauchy;
    if (name == "inverse_smooth") return GapFalloff::InverseSmooth;
    throw std::invalid_argument("gap falloff must be none, inverse, inverse_square, gaussian, cauchy or inverse_smooth: " + name);
}

inline double gapFalloffFactor(GapFalloff falloff, double g)
{
    switch (falloff) {
    case GapFalloff::None: return 1.;
    case GapFalloff::Inverse: return std::min(1., 1. / std::max(g, 1e-12));
    case GapFalloff::InverseSquare: { const double r = std::min(1., 1. / std::max(g, 1e-12)); return r * r; }
    case GapFalloff::Gaussian: return std::exp(-0.5 * g * g);
    case GapFalloff::Cauchy: return 1. / (1. + g * g);
    case GapFalloff::InverseSmooth: return 1. / std::sqrt(1. + g * g);
    }
    return 1.;
}

// Gap of offset d measured per axis of the frame (tangent t, side n x t, normal n)
// in units of scales (tangent, side, normal); t and n unit and perpendicular.
inline double scaledFrameGap(const cv::Vec3d& d, const cv::Vec3d& t, const cv::Vec3d& n, const std::array<double, 3>& scales)
{
    const cv::Vec3d side = n.cross(t);
    const double ct = d.dot(t) / scales[0], cs = d.dot(side) / scales[1], cn = d.dot(n) / scales[2];
    return std::sqrt(ct * ct + cs * cs + cn * cn);
}
