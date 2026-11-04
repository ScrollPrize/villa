#pragma once
#include <ceres/ceres.h>
#include <array>
#include <cmath>
#include <stdexcept>

namespace vc::tracer {
// A single point is held, not the tiny initial quad: its other vertices must
// spread to the requested grid spacing. No reference winding labels are used.
class SeedAnchor {
public:
    void configure(const std::array<double, 3>& xyz) {
        if (enabled_) throw std::runtime_error("seed anchor configured twice");
        for (int i = 0; i < 3; ++i) {
            // tifxyz stores float32; use the same coordinate on fresh/resume.
            const float value = static_cast<float>(xyz[i]);
            if (!std::isfinite(value) || value < 0)
                throw std::runtime_error("seed anchor must be finite nonnegative XYZ");
            xyz_[i] = value;
        }
        enabled_ = true;
    }
    bool enabled() const { return enabled_; }
    const std::array<double, 3>& xyz() const { return xyz_; }
    bool matches(const double* p) const {
        return enabled_ && p && p[0] == xyz_[0] && p[1] == xyz_[1] && p[2] == xyz_[2];
    }
    void bind(double* p) {
        if (!matches(p) || point_)
            throw std::runtime_error("seed anchor absent, moved or duplicated");
        point_ = p;
    }
    void verify() const {
        if (enabled_ && !matches(point_))
            throw std::runtime_error("seed anchor was lost or moved");
    }
    void apply(ceres::Problem& problem) const {
        if (!enabled_) return;
        // A window not containing the anchor needs no extra parameter block.
        // No cross-thread read of its coordinates here; verify after joins.
        if (!point_) throw std::runtime_error("seed anchor was not bound");
        if (problem.HasParameterBlock(point_)) problem.SetParameterBlockConstant(point_);
    }
private:
    bool enabled_ = false;
    std::array<double, 3> xyz_{};
    double* point_ = nullptr; // work grid storage is stable for the trace lifetime
};
}
