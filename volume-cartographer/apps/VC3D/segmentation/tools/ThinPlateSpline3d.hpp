#pragma once

#include <opencv2/core.hpp>

#include <optional>
#include <vector>

class ThinPlateSpline3d
{
public:
    struct Sample
    {
        cv::Point2d grid;
        cv::Vec3d value;
    };

    [[nodiscard]] bool fit(const std::vector<Sample>& samples, double regularization);
    [[nodiscard]] std::optional<cv::Vec3f> evaluate(const cv::Point2d& grid) const;

private:
    static double kernel(double r);

    std::vector<cv::Point2d> _samples;
    cv::Point2d _centroid{0.0, 0.0};
    double _scale{1.0};
    cv::Mat_<double> _coeffX;
    cv::Mat_<double> _coeffY;
    cv::Mat_<double> _coeffZ;
};
