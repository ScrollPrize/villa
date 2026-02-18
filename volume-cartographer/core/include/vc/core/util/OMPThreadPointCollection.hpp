#pragma once
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <opencv2/core/matx.hpp>

#include "omp.h"


static bool has_min_dist_sq(const cv::Vec2i &p, const std::vector<cv::Vec2i> &list, double min_dist_sq)
{
    for (const auto &o : list) {
        if (o[0] == -1 || o == p)
            continue;

        const int64_t dy = static_cast<int64_t>(o[0]) - static_cast<int64_t>(p[0]);
        const int64_t dx = static_cast<int64_t>(o[1]) - static_cast<int64_t>(p[1]);
        const int64_t d2 = dy * dy + dx * dx;
        if (static_cast<double>(d2) < min_dist_sq)
            return false;
    }

    return true;
}

static cv::Point2i extract_point_min_dist(std::vector<cv::Vec2i> &cands,
                                          const std::vector<cv::Vec2i> &blocked,
                                          int &idx,
                                          float dist)
{
    if (cands.empty()) {
        return {-1, -1};
    }

    const double min_dist_sq = static_cast<double>(dist) * static_cast<double>(dist);
    const int n = static_cast<int>(cands.size());
    int pos = idx % n;

    for (int i = 0; i < n; ++i) {
        cv::Vec2i p = cands[pos];

        if (p[0] != -1 && has_min_dist_sq(p, blocked, min_dist_sq)) {
            cands[pos] = {-1, -1};
            idx = (pos + 1) % n;
            return p;
        }

        pos = (pos + 1 == n) ? 0 : (pos + 1);
    }

    return {-1, -1};
}

//collection of points which can be retrieved with minimum distance requirement
class OmpThreadPointCol
{
public:
    OmpThreadPointCol(float dist, const std::vector<cv::Vec2i> &src) :
        _thread_count(omp_get_max_threads()),
        _dist(dist),
        _points(src),
        _thread_points(_thread_count,{-1,-1}),
        _thread_idx(_thread_count, -1) {};

    template <typename T>
    OmpThreadPointCol(float dist, T src) :
        _thread_count(omp_get_max_threads()),
        _dist(dist),
        _points(src.begin(), src.end()),
        _thread_points(_thread_count,{-1,-1}),
        _thread_idx(_thread_count, -1) {};

    cv::Point2i next()
    {
        int t_id = omp_get_thread_num();
        if (_thread_idx[t_id] == -1) {
            if (_points.empty()) {
                _thread_idx[t_id] = 0;
            } else {
                _thread_idx[t_id] = std::rand() % static_cast<int>(_points.size());
            }
        }
        _thread_points[t_id] = {-1,-1};
#pragma omp critical
        _thread_points[t_id] = extract_point_min_dist(_points, _thread_points, _thread_idx[t_id], _dist);
        return _thread_points[t_id];
    }

protected:
    int _thread_count;
    float _dist;
    std::vector<cv::Vec2i> _points;
    std::vector<cv::Vec2i> _thread_points;
    std::vector<int> _thread_idx;
};
