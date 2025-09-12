#include "vc/core/util/Thinning.hpp"
#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm> // For std::min
#include <cmath>
#include <deque>
#include <limits>
#include <iostream>

// Helper for non-maximum suppression
void nonMaximumSuppression(const cv::Mat& src, cv::Mat& dst, int size) {
    cv::Mat dilated;
    cv::dilate(src, dilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(size, size)));
    cv::compare(src, dilated, dst, cv::CMP_EQ);
}

// Helper function to trace a path from a starting point
static std::vector<cv::Point> tracePath(
    cv::Point startPoint,
    const cv::Mat& distTransform,
    cv::Mat& outputImage,
    int iterations,
    const std::deque<cv::Point>& initial_history,
    const cv::Point& seedPoint)
{
    const int N = 16;
    cv::Point currentPoint = startPoint;
    std::deque<cv::Point> path_history = initial_history;
    std::vector<cv::Point> traced_points;

    for (int i = 0; i < iterations; ++i) {
        if (currentPoint != seedPoint) {
            outputImage.at<uchar>(currentPoint) = std::min(i + 1, 255);
        }
        traced_points.push_back(currentPoint);

        path_history.push_back(currentPoint);
        if (path_history.size() > N) {
            path_history.pop_front();
        }

        float maxDist = -std::numeric_limits<float>::max();
        cv::Point nextPoint(-1, -1);

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                cv::Point neighbor(currentPoint.x + dx, currentPoint.y + dy);
                if (neighbor.x < 0 || neighbor.x >= distTransform.cols || neighbor.y < 0 || neighbor.y >= distTransform.rows) continue;
                if (path_history.size() > 1 && neighbor == path_history[path_history.size() - 2]) continue;

                float dist = distTransform.at<float>(neighbor);
                float penalty = 0.0f;
                if (!path_history.empty()) {
                    float total_weighted_distance = 0.0f;
                    float total_weight = 0.0f;
                    float history_size = static_cast<float>(path_history.size());

                    for (size_t k = 0; k < path_history.size(); ++k) {
                        const auto& p_hist = path_history[k];
                        float dx_hist = static_cast<float>(neighbor.x - p_hist.x);
                        float dy_hist = static_cast<float>(neighbor.y - p_hist.y);
                        float d = std::sqrt(dx_hist * dx_hist + dy_hist * dy_hist);

                        float weight = 1.0f;
                        if (history_size > 1) {
                            weight = 1.0f - (static_cast<float>(k) / (history_size - 1.0f)) * (1.0f - 1.0f/N);
                            // std::cout << k << " " << weight << std::endl;
                        }
                        
                        total_weighted_distance += weight * d;
                        total_weight += weight;
                    }
                    penalty = total_weighted_distance/total_weight;
                }
                float effective_dist = dist + penalty;
                if (effective_dist > maxDist) {
                    maxDist = effective_dist;
                    nextPoint = neighbor;
                }
            }
        }

        if (nextPoint.x == -1 || distTransform.at<float>(nextPoint) < 0.1 || (outputImage.at<uchar>(nextPoint) != 0 && nextPoint != seedPoint)) {
            break;
        }
        currentPoint = nextPoint;
    }
    return traced_points;
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
    for (const auto& seed : seedPoints) {
        if (outputImage.at<uchar>(seed) != 0) {
            continue; // Already part of a traced path
        }
        outputImage.at<uchar>(seed) = 1; // Mark seed as visited

        // --- Trace in the first direction ---
        std::vector<cv::Point> first_path_points = tracePath(seed, distTransform, outputImage, iterations, {}, seed);

        // --- Trace in the second (opposite) direction ---
        std::deque<cv::Point> initial_history;
        if (first_path_points.size() > 1) {
            // Use the second point of the first path to block reversal
            initial_history.push_back(first_path_points[1]);
        }
        tracePath(seed, distTransform, outputImage, iterations, initial_history, seed);
    }
}
