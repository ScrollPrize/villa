#include "vc/core/util/PointIndex.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace {

bool isFinitePoint(const cv::Vec3f& p) noexcept
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

}

struct PointIndex::Impl {
    using Point3 = bg::model::point<float, 3, bg::cs::cartesian>;
    using Entry = std::pair<Point3, uint64_t>;
    using Tree = bgi::rtree<Entry, bgi::quadratic<16>>;

    struct PointData {
        cv::Vec3f position;
        uint64_t collectionId;
    };

    Tree tree;
    std::unordered_map<uint64_t, PointData> pointData;

    static Point3 toBoost(const cv::Vec3f& p)
    {
        return Point3(p[0], p[1], p[2]);
    }

    static cv::Vec3f fromBoost(const Point3& p)
    {
        return cv::Vec3f(bg::get<0>(p), bg::get<1>(p), bg::get<2>(p));
    }
};

PointIndex::PointIndex() : impl_(std::make_unique<Impl>()) {}

PointIndex::~PointIndex() = default;

PointIndex::PointIndex(PointIndex&&) noexcept = default;
PointIndex& PointIndex::operator=(PointIndex&&) noexcept = default;

void PointIndex::clear()
{
    impl_->tree.clear();
    impl_->pointData.clear();
}

bool PointIndex::empty() const
{
    return impl_->pointData.empty();
}

void PointIndex::buildFromMat(const cv::Mat_<cv::Vec3f>& points, uint64_t collectionId)
{
    clear();

    if (points.empty()) {
        return;
    }

    std::vector<Impl::Entry> entries;
    entries.reserve(static_cast<size_t>(points.rows) * points.cols);

    for (auto [j, i, p] : ValidPointRange<const cv::Vec3f>(&points)) {
        if (!isFinitePoint(p)) {
            continue;
        }
        uint64_t id = static_cast<uint64_t>(j) * points.cols + i;
        impl_->pointData[id] = {p, collectionId};
        entries.emplace_back(Impl::toBoost(p), id);
    }

    if (!entries.empty()) {
        impl_->tree = Impl::Tree(entries.begin(), entries.end());
    }
}

std::optional<PointIndex::QueryResult> PointIndex::nearest(
    const cv::Vec3f& position,
    float maxDistance) const
{
    if (impl_->tree.empty() || !isFinitePoint(position) || !std::isfinite(maxDistance)) {
        return std::nullopt;
    }

    std::vector<Impl::Entry> result;
    impl_->tree.query(bgi::nearest(Impl::toBoost(position), 1),
                      std::back_inserter(result));

    if (result.empty()) {
        return std::nullopt;
    }

    const auto& entry = result[0];
    cv::Vec3f pos = Impl::fromBoost(entry.first);
    cv::Vec3f diff = pos - position;
    float distSq = diff.dot(diff);

    if (maxDistance < std::numeric_limits<float>::max()) {
        float maxDistSq = maxDistance * maxDistance;
        if (distSq > maxDistSq) {
            return std::nullopt;
        }
    }

    auto dataIt = impl_->pointData.find(entry.second);
    uint64_t collectionId = (dataIt != impl_->pointData.end()) ? dataIt->second.collectionId : 0;
    return QueryResult{entry.second, collectionId, pos, distSq};
}
