#include "vc/lasagna/LineSpline.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace vc::lasagna {
namespace {

bool finite(const cv::Vec3d& point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) &&
           std::isfinite(point[2]);
}

cv::Vec3d unit(const cv::Vec3d& value, const char* name)
{
    const double length = cv::norm(value);
    if (!finite(value) || !std::isfinite(length) || length <= 1.0e-12) {
        throw std::invalid_argument(std::string(name) + " must be finite and non-zero");
    }
    return value * (1.0 / length);
}

cv::Vec3d hermite(const cv::Vec3d& p0,
                  const cv::Vec3d& p1,
                  const cv::Vec3d& m0,
                  const cv::Vec3d& m1,
                  double t)
{
    const double t2 = t * t;
    const double t3 = t2 * t;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * p0 +
           (t3 - 2.0 * t2 + t) * m0 +
           (-2.0 * t3 + 3.0 * t2) * p1 +
           (t3 - t2) * m1;
}

} // namespace

LineSplineResult interpolateLineControlPoints(const LineSplineRequest& request)
{
    if (request.controlPoints.size() < 2) {
        throw std::invalid_argument("line spline requires at least two control points");
    }
    if (!std::isfinite(request.sampleSpacing) || request.sampleSpacing <= 0.0) {
        throw std::invalid_argument("line spline sample spacing must be finite and positive");
    }

    const size_t count = request.controlPoints.size();
    std::vector<double> chords(count - 1);
    for (size_t i = 0; i < count; ++i) {
        if (!finite(request.controlPoints[i])) {
            throw std::invalid_argument("line spline control points must be finite");
        }
        if (i > 0) {
            chords[i - 1] = cv::norm(request.controlPoints[i] - request.controlPoints[i - 1]);
            if (!std::isfinite(chords[i - 1]) || chords[i - 1] <= 1.0e-9) {
                throw std::invalid_argument("line spline control points must be distinct");
            }
        }
    }

    std::vector<cv::Vec3d> derivatives(count);
    derivatives.front() = request.leftDirection
        ? unit(*request.leftDirection, "left spline direction") * chords.front()
        : request.controlPoints[1] - request.controlPoints[0];
    derivatives.back() = request.rightDirection
        ? unit(*request.rightDirection, "right spline direction") * chords.back()
        : request.controlPoints.back() - request.controlPoints[count - 2];

    for (size_t i = 1; i + 1 < count; ++i) {
        const cv::Vec3d incoming =
            (request.controlPoints[i] - request.controlPoints[i - 1]) / chords[i - 1];
        const cv::Vec3d outgoing =
            (request.controlPoints[i + 1] - request.controlPoints[i]) / chords[i];
        cv::Vec3d tangent = incoming + outgoing;
        const double tangentLength = cv::norm(tangent);
        tangent = tangentLength <= 1.0e-9 || !std::isfinite(tangentLength)
            ? outgoing
            : tangent / tangentLength;
        const double handle =
            2.0 * chords[i - 1] * chords[i] / (chords[i - 1] + chords[i]);
        derivatives[i] = tangent * handle;
    }

    LineSplineResult result;
    result.controlPointIndices.reserve(count);
    result.points.push_back(request.controlPoints.front());
    result.controlPointIndices.push_back(0);
    for (size_t span = 0; span + 1 < count; ++span) {
        const int samples = std::max(
            1, static_cast<int>(std::ceil(chords[span] / request.sampleSpacing)));
        cv::Vec3d left = derivatives[span];
        cv::Vec3d right = derivatives[span + 1];
        const double maxHandle = 1.5 * chords[span];
        const double leftLength = cv::norm(left);
        const double rightLength = cv::norm(right);
        if (leftLength > maxHandle)
            left *= maxHandle / leftLength;
        if (rightLength > maxHandle)
            right *= maxHandle / rightLength;

        std::vector<cv::Vec3d> candidate;
        bool usable = false;
        for (int reduction = 0; reduction < 12 && !usable; ++reduction) {
            candidate.clear();
            candidate.reserve(static_cast<size_t>(samples));
            const cv::Vec3d chord = request.controlPoints[span + 1] -
                request.controlPoints[span];
            const cv::Vec3d chordDirection = chord / chords[span];
            double previousProgress = 0.0;
            usable = true;
            for (int sample = 1; sample <= samples; ++sample) {
                const double t = static_cast<double>(sample) / samples;
                cv::Vec3d point = hermite(request.controlPoints[span],
                                          request.controlPoints[span + 1],
                                          left,
                                          right,
                                          t);
                if (sample == samples)
                    point = request.controlPoints[span + 1];
                const double progress =
                    (point - request.controlPoints[span]).dot(chordDirection);
                const cv::Vec3d chordPoint = request.controlPoints[span] + chord * t;
                const double deviation = cv::norm(point - chordPoint);
                if (!finite(point) || progress + 1.0e-8 < previousProgress ||
                    progress < -1.0e-8 || progress > chords[span] + 1.0e-8 ||
                    deviation > std::max(request.sampleSpacing, chords[span] * 0.5)) {
                    usable = false;
                    break;
                }
                previousProgress = progress;
                candidate.push_back(point);
            }
            if (!usable) {
                left *= 0.5;
                right *= 0.5;
            }
        }
        if (!usable)
            throw std::runtime_error("line spline could not produce a forward bounded span");
        result.points.insert(result.points.end(), candidate.begin(), candidate.end());
        result.controlPointIndices.push_back(static_cast<int>(result.points.size() - 1));
    }
    return result;
}

} // namespace vc::lasagna
