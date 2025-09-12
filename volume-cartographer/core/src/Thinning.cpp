#include "vc/core/util/Thinning.hpp"
#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm> // For std::min
#include <cmath>
#include <deque>
#include <limits>

// Helper for non-maximum suppression
void nonMaximumSuppression(const cv::Mat& src, cv::Mat& dst, int size) {
    cv::Mat dilated;
    cv::dilate(src, dilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(size, size)));
    cv::compare(src, dilated, dst, cv::CMP_EQ);
}

void customThinning(const cv::Mat& inputImage, cv::Mat& outputImage, int iterations) {
    if (inputImage.empty() || inputImage.type() != CV_8UC1) {
        return;
    }

    // 1. Distance Transform
    cv::Mat distTransform;
    cv::distanceTransform(inputImage, distTransform, cv::DIST_L2, cv::DIST_MASK_PRECISE);

    // 2. Find seeds via non-maximum suppression
    cv::Mat localMaxima;
    nonMaximumSuppression(distTransform, localMaxima, 3); // 3x3 window

    // Only consider maxima that are actually on the foreground
    cv::Mat seeds;
    cv::bitwise_and(localMaxima, inputImage, seeds);

    std::vector<cv::Point> seedPoints;
    cv::findNonZero(seeds, seedPoints);

    // Initialize output image
    outputImage = cv::Mat::zeros(inputImage.size(), CV_8UC1);

    // 3. Trace from each seed
    const int N = 16;
    for (const auto& seed : seedPoints) {
        if (outputImage.at<uchar>(seed) != 0) {
            continue; // Already part of a traced path
        }

        cv::Point currentPoint = seed;
        std::deque<cv::Point> path_history;

        for (int i = 0; i < iterations; ++i) { // Use iterations to limit path length
            outputImage.at<uchar>(currentPoint) = std::min(i + 1, 255); // Use i+1 to avoid 0

            path_history.push_back(currentPoint);
            if (path_history.size() > N) {
                path_history.pop_front();
            }

            float maxDist = -std::numeric_limits<float>::max();
            cv::Point nextPoint = cv::Point(-1, -1);

            // Search 8-connected neighbors
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;

                    cv::Point neighbor(currentPoint.x + dx, currentPoint.y + dy);

                    // Check bounds
                    if (neighbor.x < 0 || neighbor.x >= distTransform.cols ||
                        neighbor.y < 0 || neighbor.y >= distTransform.rows) {
                        continue;
                    }

                    // Don't go back to the immediate previous point
                    if (path_history.size() > 1 && neighbor == path_history[path_history.size() - 2]) {
                        continue;
                    }

                    float dist = distTransform.at<float>(neighbor);

                    // Calculate penalty based on distance to recent path
                    float penalty = 0.0f;
                    if (!path_history.empty()) {
                        float min_dist_sq = std::numeric_limits<float>::max();
                        float weight_for_min_dist = 0.0f;

                        for (size_t k = 0; k < path_history.size(); ++k) {
                            const cv::Point& p_hist = path_history[k];
                            float dx_hist = static_cast<float>(neighbor.x - p_hist.x);
                            float dy_hist = static_cast<float>(neighbor.y - p_hist.y);
                            float d_sq = dx_hist * dx_hist + dy_hist * dy_hist;

                            if (d_sq < min_dist_sq) {
                                min_dist_sq = d_sq;
                                // Weight decreases linearly from 1.0 for oldest (k=0) to 1/N for newest
                                float history_size = static_cast<float>(path_history.size());
                                if (history_size > 1) {
                                    weight_for_min_dist = 1.0f - (static_cast<float>(k) / (history_size - 1.0f)) * (1.0f - 1.0f/N);
                                } else {
                                    weight_for_min_dist = 1.0f;
                                }
                            }
                        }
                        penalty = weight_for_min_dist * std::sqrt(min_dist_sq);
                    }

                    float effective_dist = dist - penalty;

                    if (effective_dist > maxDist) {
                        maxDist = effective_dist;
                        nextPoint = neighbor;
                    }
                }
            }

            // Termination conditions
            if (nextPoint.x == -1 || // No valid neighbor found
                distTransform.at<float>(nextPoint) < 0.1 || // Reached the edge of the shape
                outputImage.at<uchar>(nextPoint) != 0) { // Met an existing path
                break;
            }

            currentPoint = nextPoint;
        }
    }
}
