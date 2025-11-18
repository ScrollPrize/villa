#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"

namespace vc::surface
{
struct MaskAreaOptions
{
    bool insideIsNonZero = true;
};

struct MaskAreaResult
{
    double area_vx2 = 0.0;
    double area_cm2 = 0.0;
    double median_step_u = 0.0;
    double median_step_v = 0.0;
    std::size_t contributing_quads = 0;
    std::size_t inside_pixels = 0;
};

namespace detail
{
template <typename Vec>
inline bool isFiniteVec(const Vec& v)
{
    return std::isfinite(static_cast<double>(v[0])) &&
           std::isfinite(static_cast<double>(v[1])) &&
           std::isfinite(static_cast<double>(v[2]));
}

template <typename Vec>
inline bool isSentinelInvalid(const Vec& v)
{
    return static_cast<double>(v[0]) == -1.0 &&
           static_cast<double>(v[1]) == -1.0 &&
           static_cast<double>(v[2]) == -1.0;
}

template <typename Vec>
inline double triangleArea(const Vec& a, const Vec& b, const Vec& c)
{
    if (!isFiniteVec(a) || !isFiniteVec(b) || !isFiniteVec(c)) {
        return 0.0;
    }
    if (isSentinelInvalid(a) || isSentinelInvalid(b) || isSentinelInvalid(c)) {
        return 0.0;
    }

    const double ux = static_cast<double>(b[0]) - static_cast<double>(a[0]);
    const double uy = static_cast<double>(b[1]) - static_cast<double>(a[1]);
    const double uz = static_cast<double>(b[2]) - static_cast<double>(a[2]);

    const double vx = static_cast<double>(c[0]) - static_cast<double>(a[0]);
    const double vy = static_cast<double>(c[1]) - static_cast<double>(a[1]);
    const double vz = static_cast<double>(c[2]) - static_cast<double>(a[2]);

    const double cx = uy * vz - uz * vy;
    const double cy = uz * vx - ux * vz;
    const double cz = ux * vy - uy * vx;

    return 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
}

template <typename Vec>
inline double quadArea(const Vec& p00,
                       const Vec& p10,
                       const Vec& p01,
                       const Vec& p11)
{
    return triangleArea(p00, p10, p11) + triangleArea(p00, p11, p01);
}

template <typename Vec>
inline double surfaceArea(const cv::Mat_<Vec>& points)
{
    if (points.empty() || points.rows < 2 || points.cols < 2) {
        return 0.0;
    }

    double total = 0.0;
    for (int y = 0; y < points.rows - 1; ++y) {
        for (int x = 0; x < points.cols - 1; ++x) {
            total += quadArea(points(y, x),
                              points(y, x + 1),
                              points(y + 1, x),
                              points(y + 1, x + 1));
        }
    }
    return total;
}

inline double median(std::vector<double>& values)
{
    if (values.empty()) {
        return 0.0;
    }
    auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    double m = *mid;
    if ((values.size() % 2) == 0) {
        auto left = std::max_element(values.begin(), mid);
        m = 0.5 * (m + *left);
    }
    return m;
}

struct MaskConversion
{
    cv::Mat mask;
    std::size_t insidePixelCount = 0;
};

inline MaskConversion toBinaryMask(const cv::Mat& input, bool insideIsNonZero)
{
    MaskConversion result;
    if (input.empty()) {
        return result;
    }

    cv::Mat single;
    if (input.channels() == 1) {
        single = input;
    } else {
        cv::cvtColor(input, single, cv::COLOR_BGR2GRAY);
    }

    cv::Mat binary;
    cv::compare(single, 0, binary, insideIsNonZero ? cv::CmpTypes::CMP_NE : cv::CmpTypes::CMP_EQ);
    binary.convertTo(result.mask, CV_8U, 255.0);

    result.insidePixelCount = static_cast<std::size_t>(cv::countNonZero(binary));
    return result;
}
} // namespace detail

inline double triangleAreaVox2(const cv::Vec3d& a, const cv::Vec3d& b, const cv::Vec3d& c)
{
    return detail::triangleArea(a, b, c);
}

inline double triangleAreaVox2(const cv::Vec3f& a, const cv::Vec3f& b, const cv::Vec3f& c)
{
    return detail::triangleArea(a, b, c);
}

inline double quadAreaVox2(const cv::Vec3d& p00,
                           const cv::Vec3d& p10,
                           const cv::Vec3d& p01,
                           const cv::Vec3d& p11)
{
    return detail::quadArea(p00, p10, p01, p11);
}

inline double quadAreaVox2(const cv::Vec3f& p00,
                           const cv::Vec3f& p10,
                           const cv::Vec3f& p01,
                           const cv::Vec3f& p11)
{
    return detail::quadArea(p00, p10, p01, p11);
}

inline double computeSurfaceAreaVox2(const cv::Mat_<cv::Vec3f>& points)
{
    return detail::surfaceArea(points);
}

inline double computeSurfaceAreaVox2(const cv::Mat_<cv::Vec3d>& points)
{
    return detail::surfaceArea(points);
}

inline double computeSurfaceAreaVox2(const QuadSurface& surface)
{
    const auto* pointsPtr = surface.rawPointsPtr();
    if (!pointsPtr || pointsPtr->empty()) {
        return 0.0;
    }
    return detail::surfaceArea(*pointsPtr);
}

inline bool computeMaskAreaFromGrid(const cv::Mat_<cv::Vec3f>& coords,
                                    const cv::Mat& maskInput,
                                    double voxelSize,
                                    const MaskAreaOptions& options,
                                    MaskAreaResult& result,
                                    std::string* error = nullptr)
{
    result = {};

    if (coords.empty()) {
        if (error) {
            *error = "Coordinate grid is empty";
        }
        return false;
    }

    auto converted = detail::toBinaryMask(maskInput, options.insideIsNonZero);
    if (converted.mask.empty()) {
        if (error) {
            *error = "Mask is empty";
        }
        return false;
    }

    if (converted.mask.size() != coords.size()) {
        if (error) {
            *error = "Mask and coordinate grid dimensions differ";
        }
        return false;
    }

    const int rows = converted.mask.rows;
    const int cols = converted.mask.cols;

    std::vector<double> stepsU;
    std::vector<double> stepsV;
    stepsU.reserve(static_cast<std::size_t>(rows) * std::max(0, cols - 1));
    stepsV.reserve(static_cast<std::size_t>(cols) * std::max(0, rows - 1));

    double areaVox2 = 0.0;
    std::size_t contributing = 0;

    for (int y = 0; y < rows - 1; ++y) {
        const auto* row0 = converted.mask.ptr<std::uint8_t>(y);
        const auto* row1 = converted.mask.ptr<std::uint8_t>(y + 1);
        for (int x = 0; x < cols - 1; ++x) {
            const bool containsInside = row0[x] || row0[x + 1] || row1[x] || row1[x + 1];
            if (!containsInside) {
                continue;
            }

            const cv::Vec3f& p00 = coords(y, x);
            const cv::Vec3f& p10 = coords(y, x + 1);
            const cv::Vec3f& p01 = coords(y + 1, x);
            const cv::Vec3f& p11 = coords(y + 1, x + 1);

            double quadArea = detail::quadArea(p00, p10, p01, p11);
            if (quadArea <= 0.0 || !std::isfinite(quadArea)) {
                continue;
            }

            areaVox2 += quadArea;
            ++contributing;

            double stepU = cv::norm(p10 - p00);
            double stepV = cv::norm(p01 - p00);
            if (std::isfinite(stepU) && stepU > 0.0) {
                stepsU.push_back(stepU);
            }
            if (std::isfinite(stepV) && stepV > 0.0) {
                stepsV.push_back(stepV);
            }
        }
    }

    double du = detail::median(stepsU);
    double dv = detail::median(stepsV);

    if (areaVox2 <= 0.0 || !std::isfinite(areaVox2)) {
        if (stepsU.empty() || stepsV.empty()) {
            std::vector<double> allU;
            std::vector<double> allV;
            allU.reserve(static_cast<std::size_t>(rows) * std::max(0, cols - 1));
            allV.reserve(static_cast<std::size_t>(cols) * std::max(0, rows - 1));
            for (int y = 0; y < rows; ++y) {
                for (int x = 0; x < cols - 1; ++x) {
                    double val = cv::norm(coords(y, x + 1) - coords(y, x));
                    if (std::isfinite(val)) {
                        allU.push_back(val);
                    }
                }
            }
            for (int y = 0; y < rows - 1; ++y) {
                for (int x = 0; x < cols; ++x) {
                    double val = cv::norm(coords(y + 1, x) - coords(y, x));
                    if (std::isfinite(val)) {
                        allV.push_back(val);
                    }
                }
            }
            if (stepsU.empty()) {
                du = detail::median(allU);
            }
            if (stepsV.empty()) {
                dv = detail::median(allV);
            }
        }

        if (converted.insidePixelCount > 0 && du > 0.0 && dv > 0.0) {
            areaVox2 = static_cast<double>(converted.insidePixelCount) * (du * dv);
            contributing = 0;
        }
    }

    if (!std::isfinite(areaVox2) || areaVox2 <= 0.0) {
        if (error) {
            *error = "Mask did not yield a positive finite area";
        }
        return false;
    }

    result.area_vx2 = areaVox2;
    result.median_step_u = du;
    result.median_step_v = dv;
    result.contributing_quads = contributing;
    result.inside_pixels = converted.insidePixelCount;

    if (std::isfinite(voxelSize) && voxelSize > 0.0) {
        result.area_cm2 = areaVox2 * voxelSize * voxelSize / 1e8;
    }

    return true;
}

inline bool computeMaskArea(QuadSurface& surface,
                            const cv::Mat& mask,
                            double voxelSize,
                            const MaskAreaOptions& options,
                            MaskAreaResult& result,
                            std::string* error = nullptr)
{
    if (mask.empty()) {
        if (error) {
            *error = "Mask is empty";
        }
        return false;
    }

    cv::Vec3f ptr = surface.pointer();
    cv::Vec3f offset(-mask.cols / 2.0f, -mask.rows / 2.0f, 0.0f);

    cv::Mat_<cv::Vec3f> coords;
    try {
        surface.gen(&coords, nullptr, mask.size(), ptr, 1.0f, offset);
    } catch (...) {
        if (error) {
            *error = "Failed generating surface coordinates for mask";
        }
        return false;
    }

    return computeMaskAreaFromGrid(coords, mask, voxelSize, options, result, error);
}

} // namespace vc::surface
