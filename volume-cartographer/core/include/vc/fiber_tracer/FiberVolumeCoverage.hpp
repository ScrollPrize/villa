#pragma once
#include <opencv2/core/types.hpp>
#include <cstdint>
#include <memory>
#include <vector>

namespace vc::fiber_tracer {
// Native implementation of the neural volume policy's owned-point and live-
// segment indexes. Width is an optional extra rejection, never a new hit rule.
// Queries are concurrent readers; publication is synchronized, with index
// construction outside the reader lock. No Python callbacks in this API.
class FiberVolumeCoverage {
public:
    explicit FiberVolumeCoverage(double binSize,size_t mergePoints=50000);
    ~FiberVolumeCoverage();
    void addOwned(const std::vector<cv::Vec3d>&,int64_t id,double width);
    void flushOwned();
    int64_t owner(const cv::Vec3d&,const cv::Vec3d&,double radius,
                  double width,double tolerance,int64_t exclude=-1) const;
    void addLive(const cv::Vec3d&,const cv::Vec3d&,int64_t pair,uint64_t epoch,double width);
    bool liveHit(const cv::Vec3d&,const cv::Vec3d&,double radius,int64_t pair,
                 uint64_t epoch,double width,double tolerance) const;
    void dropLive(int64_t pair);
    static bool compatibleWidth(double candidate,double reference,double tolerance);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace vc::fiber_tracer
