#pragma once

// The reference ray occlusion costs of GrowPatch.cpp, in a header so that the
// tests in core/test can build them on a synthetic volume.

#include "vc/core/types/ChunkedTensor.hpp"

#include <ceres/ceres.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>

class ReferenceRayOcclusionCost {
public:
    ReferenceRayOcclusionCost(Chunked3d<uint8_t, passTroughComputor>* volume,
                              const cv::Vec3d& target,
                              double threshold,
                              double weight,
                              double step,
                              double max_distance) :
        volume_(volume),
        target_(target),
        threshold_(threshold),
        weight_(weight),
        step_(step > 0.0 ? step : 1.0),
        max_distance_(max_distance)
    {}

    bool operator()(const double* candidate, double* residual) const {
        if (!volume_ || weight_ <= 0.0) {
            residual[0] = 0.0;
            return true;
        }

        const cv::Vec3d start{candidate[0], candidate[1], candidate[2]};
        const double distance = cv::norm(target_ - start);
        if (distance <= 1e-6) {
            residual[0] = 0.0;
            return true;
        }

        if (max_distance_ > 0.0 && distance > max_distance_) {
            residual[0] = 0.0;
            return true;
        }

        const int steps = std::max(1, static_cast<int>(std::ceil(distance / step_)));
        const cv::Vec3d delta = (target_ - start) / static_cast<double>(steps + 1);

        double max_value = std::numeric_limits<double>::lowest();
        bool hit_threshold = false;
        cv::Vec3d current = start;

        const double start_value = sample(start);
        int begin_step = 1;
        if (std::isfinite(start_value) && start_value >= threshold_) {
            bool exited_material = false;
            for (; begin_step <= steps; ++begin_step) {
                current += delta;
                const double value = sample(current);
                if (!std::isfinite(value)) {
                    continue;
                }
                if (value < threshold_) {
                    exited_material = true;
                    ++begin_step;  // start checking one step beyond the exit.
                    break;
                }
            }
            if (!exited_material) {
                residual[0] = 0.0;
                return true;
            }
        } else {
            current = start;
        }

        for (int i = begin_step; i <= steps; ++i) {
            current += delta;
            const double value = sample(current);
            if (!std::isfinite(value)) {
                continue;
            }

            max_value = std::max(max_value, value);
            if (value >= threshold_) {
                hit_threshold = true;
                break;
            }
        }

        if (!hit_threshold) {
            residual[0] = 0.0;
            return true;
        }

        if (!std::isfinite(max_value) || max_value < 0.0) {
            max_value = 0.0;
        }

        // max_value was raised to the hit sample before the loop broke, so it is at
        // least threshold_ here and threshold_ - max_value is never positive: the
        // residual was zero on every path. The occlusion is max_value - threshold_.
        const double diff = std::max(0.0, max_value - threshold_);
        residual[0] = weight_ * diff;
        return true;
    }

private:
    double sample(const cv::Vec3d& xyz) const {
        if (!interp_) {
            interp_ = std::make_unique<CachedChunked3dInterpolator<uint8_t, passTroughComputor>>(*volume_);
        }
        double value = 0.0;
        interp_->Evaluate(xyz[2], xyz[1], xyz[0], &value);
        return value;
    }

    Chunked3d<uint8_t, passTroughComputor>* volume_;
    cv::Vec3d target_;
    double threshold_;
    double weight_;
    double step_;
    double max_distance_;
    mutable std::unique_ptr<CachedChunked3dInterpolator<uint8_t, passTroughComputor>> interp_;
};

// ReferenceRayOcclusionCost with an analytic Jacobian. Under NumericDiffCostFunction CENTRAL the functor
// above marches the ray seven times per evaluation (x, and x plus and minus a step on each coordinate).
// This cost marches it once with the same arithmetic in the same order, so the residual is the same double.
// When the ray hits, the residual is weight * (v - threshold), v the trilinear sample at the k-th of the
// n + 1 steps from start to target, so the hit point is start + k * (target - start) / (n + 1) and
// d residual / d start = weight * (1 - k / (n + 1)) * grad v, with k and n piecewise constant in start.
// grad v comes from the interpolator's own template evaluated once on ceres::Jet at the hit point.
class ReferenceRayOcclusionAnalyticCost : public ceres::SizedCostFunction<1, 3> {
public:
    ReferenceRayOcclusionAnalyticCost(Chunked3d<uint8_t, passTroughComputor>* volume,
                                      const cv::Vec3d& target,
                                      double threshold,
                                      double weight,
                                      double step,
                                      double max_distance) :
        volume_(volume),
        target_(target),
        threshold_(threshold),
        weight_(weight),
        step_(step > 0.0 ? step : 1.0),
        max_distance_(max_distance)
    {}

    struct March {
        bool hit = false;
        int k = 0;
        int n = 0;
        cv::Vec3d point{0.0, 0.0, 0.0};
    };

    // The arithmetic of ReferenceRayOcclusionCost::operator(), statement for statement, with the hit recorded.
    double march(const double* candidate, March* m) const {
        *m = March{};
        if (!volume_ || weight_ <= 0.0) {
            return 0.0;
        }

        const cv::Vec3d start{candidate[0], candidate[1], candidate[2]};
        const double distance = cv::norm(target_ - start);
        if (distance <= 1e-6) {
            return 0.0;
        }

        if (max_distance_ > 0.0 && distance > max_distance_) {
            return 0.0;
        }

        const int steps = std::max(1, static_cast<int>(std::ceil(distance / step_)));
        const cv::Vec3d delta = (target_ - start) / static_cast<double>(steps + 1);
        m->n = steps;

        double max_value = std::numeric_limits<double>::lowest();
        bool hit_threshold = false;
        cv::Vec3d current = start;
        int k = 0;

        const double start_value = sample(start);
        int begin_step = 1;
        if (std::isfinite(start_value) && start_value >= threshold_) {
            bool exited_material = false;
            for (; begin_step <= steps; ++begin_step) {
                current += delta;
                ++k;
                const double value = sample(current);
                if (!std::isfinite(value)) {
                    continue;
                }
                if (value < threshold_) {
                    exited_material = true;
                    ++begin_step;  // start checking one step beyond the exit.
                    break;
                }
            }
            if (!exited_material) {
                return 0.0;
            }
        } else {
            current = start;
            k = 0;
        }

        for (int i = begin_step; i <= steps; ++i) {
            current += delta;
            ++k;
            const double value = sample(current);
            if (!std::isfinite(value)) {
                continue;
            }

            max_value = std::max(max_value, value);
            if (value >= threshold_) {
                hit_threshold = true;
                break;
            }
        }

        if (!hit_threshold) {
            return 0.0;
        }

        if (!std::isfinite(max_value) || max_value < 0.0) {
            max_value = 0.0;
        }

        const double diff = std::max(0.0, max_value - threshold_);
        m->hit = true;
        m->k = k;
        m->point = current;
        return weight_ * diff;
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        March m;
        residuals[0] = march(parameters[0], &m);
        if (jacobians && jacobians[0]) {
            double* J = jacobians[0];
            J[0] = 0.0;
            J[1] = 0.0;
            J[2] = 0.0;
            if (m.hit && residuals[0] > 0.0) {
                using JetT = ceres::Jet<double, 3>;
                JetT v;
                interp().Evaluate(JetT(m.point[2], 2), JetT(m.point[1], 1), JetT(m.point[0], 0), &v);
                const double along = 1.0 - static_cast<double>(m.k) / static_cast<double>(m.n + 1);
                for (int i = 0; i < 3; ++i) {
                    J[i] = weight_ * along * v.v[i];
                }
            }
        }
        return true;
    }

private:
    CachedChunked3dInterpolator<uint8_t, passTroughComputor>& interp() const {
        if (!interp_) {
            interp_ = std::make_unique<CachedChunked3dInterpolator<uint8_t, passTroughComputor>>(*volume_);
        }
        return *interp_;
    }

    double sample(const cv::Vec3d& xyz) const {
        double value = 0.0;
        interp().Evaluate(xyz[2], xyz[1], xyz[0], &value);
        return value;
    }

    Chunked3d<uint8_t, passTroughComputor>* volume_;
    cv::Vec3d target_;
    double threshold_;
    double weight_;
    double step_;
    double max_distance_;
    mutable std::unique_ptr<CachedChunked3dInterpolator<uint8_t, passTroughComputor>> interp_;
};
