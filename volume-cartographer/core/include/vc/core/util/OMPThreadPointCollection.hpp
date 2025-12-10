#pragma once
#include <vector>
#include <opencv2/core/matx.hpp>

#include "omp.h"


static float min_dist(const cv::Vec2i &p, const std::vector<cv::Vec2i> &list)
{
    double dist = 10000000000;
    for(auto &o : list) {
        if (o[0] == -1 || o == p)
            continue;
        dist = std::min(cv::norm(o-p), dist);
    }

    return dist;
}

static cv::Point2i extract_point_min_dist(std::vector<cv::Vec2i> &cands, const std::vector<cv::Vec2i> &blocked, int &idx, float dist)
{
    for(int i=0;i<cands.size();i++) {
        cv::Vec2i p = cands[(i + idx) % cands.size()];

        if (p[0] == -1)
            continue;

        if (min_dist(p, blocked) >= dist) {
            cands[(i + idx) % cands.size()] = {-1,-1};
            idx = (i + idx + 1) % cands.size();

            return p;
        }
    }

    return {-1,-1};
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
        if (_thread_idx[t_id] == -1)
            _thread_idx[t_id] = rand() % _thread_count;
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