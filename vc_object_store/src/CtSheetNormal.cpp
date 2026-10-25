#include "vc_object_store/CtSheetNormal.hpp"
#include <vc_reader/reader.hpp>

#include <algorithm>
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

struct LasagnaSheetNormalSampler::Impl {
    vcr::CtReader nx, ny;
    double scale;
    std::mutex mutex;

    Impl(const std::filesystem::path& x, const std::filesystem::path& y, int level, double scale)
        : nx(x.string(), level), ny(y.string(), level), scale(scale) {}
};

LasagnaSheetNormalSampler::LasagnaSheetNormalSampler(const std::filesystem::path& normalX,
    const std::filesystem::path& normalY, int level, double scale)
{
    if (level < 0 || !(scale > 0) || !std::isfinite(scale))
        throw std::invalid_argument("invalid lasagna normal level or scale");
    impl_ = std::make_unique<Impl>(normalX, normalY, level, scale);
    if (impl_->nx.shape() != impl_->ny.shape())
        throw std::invalid_argument("lasagna nx and ny volumes differ in shape");
}

LasagnaSheetNormalSampler::~LasagnaSheetNormalSampler() = default;

CtSheetNormal LasagnaSheetNormalSampler::sampleNormal(const cv::Vec3d& xyz,
                                                      const std::atomic_bool* cancel) const
{
    if (cancel && cancel->load()) throw std::runtime_error("lasagna normals cancelled");
    std::array<int64_t, 3> zyx;
    for (int axis = 0; axis < 3; ++axis) {
        const double coordinate = xyz[2 - axis] * impl_->scale;
        if (!std::isfinite(coordinate)) throw std::invalid_argument("lasagna normal query must be finite");
        zyx[axis] = int64_t(std::llround(coordinate));
    }
    const auto shape = impl_->nx.shape();
    for (int axis = 0; axis < 3; ++axis)
        if (zyx[axis] < 0 || zyx[axis] >= shape[axis]) return {{0, 0, 0}, false, 0, "outside lasagna normal volume"};
    uint8_t x = 0, y = 0;
    {
        std::lock_guard lock(impl_->mutex);
        impl_->nx.readRegion(zyx, {1, 1, 1}, std::span(&x, 1));
        impl_->ny.readRegion(zyx, {1, 1, 1}, std::span(&y, 1));
    }
    if (x == 0 && y == 0) return {{0, 0, 0}, false, 0, "no lasagna normal"};
    const double dx = (x - 128.0) / 127.0, dy = (y - 128.0) / 127.0;
    const cv::Vec3d normal(dx, dy, std::sqrt(std::max(0.0, 1.0 - dx * dx - dy * dy)));
    return {normal / cv::norm(normal), true, 0, ""};
}
} // namespace vc
