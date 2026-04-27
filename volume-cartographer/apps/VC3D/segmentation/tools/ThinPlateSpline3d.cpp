#include "ThinPlateSpline3d.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>

namespace
{
double triangleArea2(const cv::Point2d& a, const cv::Point2d& b, const cv::Point2d& c)
{
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}
}

double ThinPlateSpline3d::kernel(double r)
{
    if (!std::isfinite(r) || r <= 0.0) {
        return 0.0;
    }
    return r * r * std::log(r);
}

bool ThinPlateSpline3d::fit(const std::vector<Sample>& samples, double regularization)
{
    _samples.clear();
    _coeffX.release();
    _coeffY.release();
    _coeffZ.release();

    if (samples.size() < 3) {
        return false;
    }

    bool nonCollinear = false;
    for (std::size_t i = 2; i < samples.size(); ++i) {
        if (std::abs(triangleArea2(samples[0].grid, samples[1].grid, samples[i].grid)) > 1e-9) {
            nonCollinear = true;
            break;
        }
    }
    if (!nonCollinear) {
        return false;
    }

    _centroid = {0.0, 0.0};
    for (const auto& sample : samples) {
        if (!std::isfinite(sample.grid.x) || !std::isfinite(sample.grid.y) ||
            !std::isfinite(sample.value[0]) || !std::isfinite(sample.value[1]) ||
            !std::isfinite(sample.value[2])) {
            return false;
        }
        _centroid += sample.grid;
    }
    _centroid *= 1.0 / static_cast<double>(samples.size());

    _scale = 1.0;
    for (const auto& sample : samples) {
        _scale = std::max(_scale, std::abs(sample.grid.x - _centroid.x));
        _scale = std::max(_scale, std::abs(sample.grid.y - _centroid.y));
    }

    _samples.reserve(samples.size());
    for (const auto& sample : samples) {
        _samples.push_back((sample.grid - _centroid) * (1.0 / _scale));
    }

    const int n = static_cast<int>(samples.size());
    const int systemSize = n + 3;
    cv::Mat_<double> lhs(systemSize, systemSize, 0.0);
    cv::Mat_<double> rhsX(systemSize, 1, 0.0);
    cv::Mat_<double> rhsY(systemSize, 1, 0.0);
    cv::Mat_<double> rhsZ(systemSize, 1, 0.0);

    const double lambda = std::max(0.0, regularization);
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            const cv::Point2d delta = _samples[r] - _samples[c];
            lhs(r, c) = kernel(std::sqrt(delta.x * delta.x + delta.y * delta.y));
        }
        lhs(r, r) += lambda;
        lhs(r, n + 0) = 1.0;
        lhs(r, n + 1) = _samples[r].x;
        lhs(r, n + 2) = _samples[r].y;
        lhs(n + 0, r) = 1.0;
        lhs(n + 1, r) = _samples[r].x;
        lhs(n + 2, r) = _samples[r].y;

        rhsX(r, 0) = samples[r].value[0];
        rhsY(r, 0) = samples[r].value[1];
        rhsZ(r, 0) = samples[r].value[2];
    }

    return cv::solve(lhs, rhsX, _coeffX, cv::DECOMP_SVD) &&
           cv::solve(lhs, rhsY, _coeffY, cv::DECOMP_SVD) &&
           cv::solve(lhs, rhsZ, _coeffZ, cv::DECOMP_SVD);
}

std::optional<cv::Vec3f> ThinPlateSpline3d::evaluate(const cv::Point2d& grid) const
{
    if (_samples.empty() || _coeffX.empty() || _coeffY.empty() || _coeffZ.empty()) {
        return std::nullopt;
    }

    const cv::Point2d p = (grid - _centroid) * (1.0 / _scale);
    const int n = static_cast<int>(_samples.size());
    auto evalChannel = [&](const cv::Mat_<double>& coeffs) {
        double value = coeffs(n + 0, 0) + coeffs(n + 1, 0) * p.x + coeffs(n + 2, 0) * p.y;
        for (int i = 0; i < n; ++i) {
            const cv::Point2d delta = p - _samples[i];
            value += coeffs(i, 0) * kernel(std::sqrt(delta.x * delta.x + delta.y * delta.y));
        }
        return value;
    };

    const cv::Vec3d value(evalChannel(_coeffX), evalChannel(_coeffY), evalChannel(_coeffZ));
    if (!std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2])) {
        return std::nullopt;
    }
    return cv::Vec3f(static_cast<float>(value[0]),
                     static_cast<float>(value[1]),
                     static_cast<float>(value[2]));
}
