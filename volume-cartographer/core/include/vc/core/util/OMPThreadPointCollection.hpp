#pragma once
#include <vector>
#include <cmath>
#include <opencv2/core/matx.hpp>

#include "omp.h"

// Collection of points which can be retrieved with minimum distance requirement.
// Uses phase-based partitioning to ensure points returned concurrently are at least
// 'dist' apart, without expensive runtime distance checking.
class OmpThreadPointCol
{
public:
    OmpThreadPointCol(float dist, const std::vector<cv::Vec2i> &src) :
        _dist(dist),
        _current_phase(0)
    {
        partition_into_phases(src);
    }

    template <typename T>
    OmpThreadPointCol(float dist, T src) :
        _dist(dist),
        _current_phase(0)
    {
        std::vector<cv::Vec2i> points(src.begin(), src.end());
        partition_into_phases(points);
    }

    cv::Point2i next()
    {
        cv::Vec2i result = {-1, -1};

        #pragma omp critical
        {
            // Try to get a point from current phase
            while (_current_phase < _phases.size()) {
                if (_phase_indices[_current_phase] < _phases[_current_phase].size()) {
                    result = _phases[_current_phase][_phase_indices[_current_phase]++];
                    break;
                }
                _current_phase++;
            }
        }

        return result;
    }

private:
    void partition_into_phases(const std::vector<cv::Vec2i> &src)
    {
        // Pre-partition points into phases using a checkerboard pattern.
        // Points in the same phase are guaranteed to be at least 'dist' apart.
        int cell_size = std::max(1, static_cast<int>(std::ceil(_dist)));
        const int num_phases = 4;  // 2x2 checkerboard pattern

        _phases.resize(num_phases);
        for (const auto& p : src) {
            int cell_row = p[0] / cell_size;
            int cell_col = p[1] / cell_size;
            int phase = (cell_row % 2) * 2 + (cell_col % 2);
            _phases[phase].push_back(p);
        }

        _phase_indices.resize(num_phases, 0);
    }

    float _dist;
    std::vector<std::vector<cv::Vec2i>> _phases;
    std::vector<size_t> _phase_indices;
    size_t _current_phase;
};
