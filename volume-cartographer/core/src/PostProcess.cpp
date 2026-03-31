#include "vc/core/util/PostProcess.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace vc {

void applyPostProcess(cv::Mat_<uint8_t>& img, const PostProcessParams& params)
{
    // 1. ISO cutoff
    if (params.isoCutoff > 0) {
        cv::threshold(img, img, params.isoCutoff - 1, 0, cv::THRESH_TOZERO);
    }

    // 2. Composite post-stretch
    if (params.postStretchValues) {
        double minVal, maxVal;
        cv::minMaxLoc(img, &minVal, &maxVal);
        if (maxVal > minVal) {
            img.convertTo(img, CV_8U,
                          255.0 / (maxVal - minVal),
                          -minVal * 255.0 / (maxVal - minVal));
        }
    }

    // 3. Composite small component removal
    if (params.removeSmallComponents && params.minComponentSize > 1) {
        cv::Mat_<uint8_t> binary;
        cv::threshold(img, binary, 0, 255, cv::THRESH_BINARY);

        cv::Mat labels, stats, centroids;
        int numComponents = cv::connectedComponentsWithStats(
            binary, labels, stats, centroids, 8, CV_32S);

        cv::Mat_<uint8_t> keepMask = cv::Mat_<uint8_t>::zeros(img.size());
        for (int i = 1; i < numComponents; i++) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area >= params.minComponentSize) {
                keepMask.setTo(255, labels == i);
            }
        }

        cv::Mat_<uint8_t> filtered;
        img.copyTo(filtered, keepMask);
        img = filtered;
    }

    // 4. Window/level or stretch — use LUT to avoid per-pixel float math
    if (params.stretchValues) {
        double minVal, maxVal;
        cv::minMaxLoc(img, &minVal, &maxVal);
        const double range = std::max(1.0, maxVal - minVal);

        uint8_t lut_data[256];
        for (int i = 0; i < 256; i++) {
            double v = (i - minVal) / range;
            lut_data[i] = static_cast<uint8_t>(std::clamp(v * 255.0, 0.0, 255.0));
        }
        cv::Mat lut(1, 256, CV_8U, lut_data);
        cv::LUT(img, lut, img);
    } else {
        const int windowLowInt = static_cast<int>(
            std::clamp(params.windowLow, 0.0f, 255.0f));
        const int windowHighInt = static_cast<int>(
            std::clamp(params.windowHigh, static_cast<float>(windowLowInt + 1), 255.0f));

        // Skip if window covers full range (identity transform)
        if (windowLowInt > 0 || windowHighInt < 255) {
            const float windowSpan = std::max(
                1.0f, static_cast<float>(windowHighInt - windowLowInt));

            uint8_t lut_data[256];
            for (int i = 0; i < 256; i++) {
                float v = (static_cast<float>(i) - static_cast<float>(windowLowInt)) / windowSpan;
                lut_data[i] = static_cast<uint8_t>(std::clamp(v * 255.0f, 0.0f, 255.0f));
            }
            cv::Mat lut(1, 256, CV_8U, lut_data);
            cv::LUT(img, lut, img);
        }
    }
}

}  // namespace vc
