// Summary for vc_objrefine's opt-in offset report: how far the alpha-comp
// scan found papyrus from each vertex, along the vertex normal. Header-only
// so the command-line tool and the regression tests run the identical code;
// see ObjAlphaCompRefinement.cpp for the tool and
// core/test/test_objrefine_offset_report.cpp for the fixtures.
#pragma once

#include "utils/Json.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace vc_objrefine {

// Linear interpolation between order statistics (numpy's default method).
// `sorted` must be ascending and non-empty.
inline double percentile(const std::vector<float>& sorted, double q)
{
    const double pos = q * static_cast<double>(sorted.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] +
           (pos - static_cast<double>(lo)) *
               (static_cast<double>(sorted[hi]) - static_cast<double>(sorted[lo]));
}

// `offset` and `opacity` are same-size maps, NaN where no ray was cast;
// `valid` is the surface's valid-point mask (nonzero = valid point) on the
// same grid, so the summary shows how many valid vertices got a reading. A
// vertex whose accumulated opacity stays below `minOpacity` found no papyrus
// inside the scan range: its offset is a weighted mean of almost nothing (0
// when the opacity is 0), not a distance, so it is counted as no_sheet and
// left out of the statistics.
inline utils::Json summarizeOffsets(const cv::Mat_<float>& offset,
                                    const cv::Mat_<float>& opacity,
                                    const cv::Mat_<std::uint8_t>& valid,
                                    float threshold,
                                    float minOpacity)
{
    if (offset.size() != opacity.size() || offset.size() != valid.size()) {
        throw std::invalid_argument("offset, opacity and valid maps differ in size");
    }

    std::vector<float> values;
    std::uint64_t measured = 0;
    std::uint64_t noSheet = 0;
    std::uint64_t beyond = 0;
    double sum = 0.0;
    for (int y = 0; y < offset.rows; ++y) {
        for (int x = 0; x < offset.cols; ++x) {
            const float off = offset(y, x);
            const float alpha = opacity(y, x);
            if (!std::isfinite(off) || !std::isfinite(alpha)) {
                continue;
            }
            ++measured;
            if (alpha < minOpacity) {
                ++noSheet;
                continue;
            }
            values.push_back(off);
            sum += off;
            if (std::fabs(off) > threshold) {
                ++beyond;
            }
        }
    }

    const auto summarized = static_cast<std::uint64_t>(values.size());
    utils::Json s = utils::Json::object();
    s["rows"] = offset.rows;
    s["cols"] = offset.cols;
    s["threshold"] = static_cast<double>(threshold);
    s["min_opacity"] = static_cast<double>(minOpacity);
    s["valid_vertices"] = static_cast<std::uint64_t>(cv::countNonZero(valid));
    s["measured_vertices"] = measured;
    s["no_sheet_vertices"] = noSheet;
    s["summarized_vertices"] = summarized;
    s["beyond_threshold_vertices"] = beyond;
    if (measured > 0) {
        s["no_sheet_fraction"] =
            static_cast<double>(noSheet) / static_cast<double>(measured);
    } else {
        s["no_sheet_fraction"] = nullptr;
    }

    utils::Json signedStats = utils::Json::object();
    utils::Json absStats = utils::Json::object();
    if (summarized > 0) {
        s["beyond_threshold_fraction"] =
            static_cast<double>(beyond) / static_cast<double>(summarized);
        // One buffer: signed statistics first, then the same values as
        // magnitudes.
        std::sort(values.begin(), values.end());
        signedStats["mean"] = sum / static_cast<double>(summarized);
        signedStats["p05"] = percentile(values, 0.05);
        signedStats["p50"] = percentile(values, 0.50);
        signedStats["p95"] = percentile(values, 0.95);
        for (float& v : values) {
            v = std::fabs(v);
        }
        std::sort(values.begin(), values.end());
        absStats["p50"] = percentile(values, 0.50);
        absStats["p90"] = percentile(values, 0.90);
        absStats["p95"] = percentile(values, 0.95);
        absStats["p99"] = percentile(values, 0.99);
        absStats["max"] = static_cast<double>(values.back());
    } else {
        s["beyond_threshold_fraction"] = nullptr;
    }
    s["offset"] = std::move(signedStats);
    s["abs_offset"] = std::move(absStats);
    return s;
}

}  // namespace vc_objrefine
