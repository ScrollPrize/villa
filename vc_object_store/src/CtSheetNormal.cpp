#include "vc_object_store/CtSheetNormal.hpp"
#include <vc_reader/reader.hpp>

#include <cmath>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>

namespace vc {
struct CtSheetNormalSampler::Impl {
    vcr::CtReader reader;
    double scale;
    std::mutex mutex;
    std::map<std::array<double, 3>, CtSheetNormal> cache;
    std::deque<std::array<double, 3>> order;

    Impl(const std::filesystem::path& path, int level, double scale)
        : reader(path.string(), level), scale(scale) {}
};

CtSheetNormalSampler::CtSheetNormalSampler(const std::filesystem::path& path, int level, double scale)
{
    if (level < 0 || !(scale > 0) || !std::isfinite(scale))
        throw std::invalid_argument("invalid CT level or scale");
    impl_ = std::make_unique<Impl>(path, level, scale);
}

CtSheetNormalSampler::~CtSheetNormalSampler() = default;

CtSheetNormal CtSheetNormalSampler::sampleNormal(const cv::Vec3d& xyz,
                                                 const std::atomic_bool* cancel) const
{
    auto cancelled = [&] {
        if (cancel && cancel->load()) throw std::runtime_error("CT normals cancelled");
    };
    cancelled();
    std::lock_guard lock(impl_->mutex);
    cancelled();
    const std::array<double, 3> point{xyz[0] * impl_->scale, xyz[1] * impl_->scale, xyz[2] * impl_->scale};
    for (auto coordinate : point) {
        if (!std::isfinite(coordinate)) throw std::invalid_argument("CT query must be finite");
    }
    if (auto found = impl_->cache.find(point); found != impl_->cache.end()) return found->second;
    const auto sample = impl_->reader.sheetNormals(std::span(&point, 1)).front();
    cancelled();
    CtSheetNormal result{{sample[0], sample[1], sample[2]}, sample[3] > 0, sample[3],
                         sample[3] > 0 ? "" : "invalid tensor or insufficient CT support"};
    if (impl_->cache.size() == 256) {
        impl_->cache.erase(impl_->order.front());
        impl_->order.pop_front();
    }
    impl_->order.push_back(point);
    impl_->cache.emplace(point, result);
    return result;
}
} // namespace vc
