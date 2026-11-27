#pragma once
#include <functional>
#include <vector>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>

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

// The one selection loop of both modes: the first candidate, from `idx` on, for which `eligible(candidate)` holds is taken out of `cands`
// (its slot is tombstoned and the cursor moves past it); {-1,-1} when no candidate is eligible.
template <class Eligible>
static cv::Point2i extract_point_if(std::vector<cv::Vec2i> &cands, int &idx, const Eligible &eligible)
{
    for(int i=0;i<cands.size();i++) {
        cv::Vec2i p = cands[(i + idx) % cands.size()];

        if (p[0] == -1)
            continue;

        if (eligible(p)) {
            cands[(i + idx) % cands.size()] = {-1,-1};
            idx = (i + idx + 1) % cands.size();

            return p;
        }
    }

    return {-1,-1};
}

// Eligibility by distance: no blocked point closer than `dist`.
static cv::Point2i extract_point_min_dist(std::vector<cv::Vec2i> &cands, const std::vector<cv::Vec2i> &blocked, int &idx, float dist)
{
    return extract_point_if(cands, idx, [&](const cv::Vec2i &p) { return min_dist(p, blocked) >= dist; });
}

// Eligibility by predicate: `conflict(candidate, blocked point)` is false for every blocked point other than the candidate itself.
template <class Conflict>
static cv::Point2i extract_point_without_conflict(std::vector<cv::Vec2i> &cands, const std::vector<cv::Vec2i> &blocked, int &idx, const Conflict &conflict)
{
    return extract_point_if(cands, idx, [&](const cv::Vec2i &p) {
        for (const auto &o : blocked)
            if (o[0] != -1 && o != p && conflict(p, o))
                return false;
        return true;
    });
}

//collection of points which can be retrieved with minimum distance requirement, or, with a predicate, without conflicts
class OmpThreadPointCol
{
public:
    OmpThreadPointCol(float dist, const std::vector<cv::Vec2i> &src) :
        _thread_count(omp_get_max_threads()),
        _dist(dist),
        _points(src),
        _thread_points(_thread_count,{-1,-1}),
        _thread_idx(_thread_count, -1) {};

    // Points are retrieved only while `conflicts(candidate, point being worked on by another thread)` is false for all of them.
    OmpThreadPointCol(std::function<bool(const cv::Vec2i&, const cv::Vec2i&)> conflicts, const std::vector<cv::Vec2i> &src) :
        _thread_count(omp_get_max_threads()),
        _dist(0),
        _conflicts(std::move(conflicts)),
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
#pragma omp critical
        {
            _thread_points[t_id] = {-1,-1};
            _thread_points[t_id] = _conflicts
                ? extract_point_without_conflict(_points, _thread_points, _thread_idx[t_id], _conflicts)
                : extract_point_min_dist(_points, _thread_points, _thread_idx[t_id], _dist);
        }
        return _thread_points[t_id];
    }

protected:
    int _thread_count;
    float _dist;
    std::function<bool(const cv::Vec2i&, const cv::Vec2i&)> _conflicts;
    std::vector<cv::Vec2i> _points;
    std::vector<cv::Vec2i> _thread_points;
    std::vector<int> _thread_idx;
};
