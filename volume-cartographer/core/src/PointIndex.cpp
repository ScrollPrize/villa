#include "vc/core/util/PointIndex.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <algorithm>
#include <unordered_map>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

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

size_t PointIndex::size() const
{
    return impl_->pointData.size();
}

void PointIndex::insert(uint64_t id, uint64_t collectionId, const cv::Vec3f& position)
{
    auto it = impl_->pointData.find(id);
    if (it != impl_->pointData.end()) {
        // Already exists - update instead
        update(id, position);
        return;
    }

    impl_->pointData[id] = {position, collectionId};
    impl_->tree.insert(std::make_pair(Impl::toBoost(position), id));
}

void PointIndex::bulkInsert(const std::vector<std::tuple<uint64_t, uint64_t, cv::Vec3f>>& points)
{
    clear();

    if (points.empty()) {
        return;
    }

    // Build entries vector for packing algorithm
    std::vector<Impl::Entry> entries;
    entries.reserve(points.size());

    for (const auto& [id, collectionId, position] : points) {
        impl_->pointData[id] = {position, collectionId};
        entries.emplace_back(Impl::toBoost(position), id);
    }

    // Construct tree using packing algorithm (much faster than individual inserts)
    impl_->tree = Impl::Tree(entries.begin(), entries.end());
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
        uint64_t id = static_cast<uint64_t>(j) * points.cols + i;
        impl_->pointData[id] = {p, collectionId};
        entries.emplace_back(Impl::toBoost(p), id);
    }

    if (!entries.empty()) {
        impl_->tree = Impl::Tree(entries.begin(), entries.end());
    }
}

void PointIndex::remove(uint64_t id)
{
    auto it = impl_->pointData.find(id);
    if (it == impl_->pointData.end()) {
        return;
    }

    impl_->tree.remove(std::make_pair(Impl::toBoost(it->second.position), id));
    impl_->pointData.erase(it);
}

bool PointIndex::update(uint64_t id, const cv::Vec3f& newPosition)
{
    auto it = impl_->pointData.find(id);
    if (it == impl_->pointData.end()) {
        return false;
    }

    impl_->tree.remove(std::make_pair(Impl::toBoost(it->second.position), id));
    it->second.position = newPosition;
    impl_->tree.insert(std::make_pair(Impl::toBoost(newPosition), id));
    return true;
}

std::vector<PointIndex::QueryResult> PointIndex::queryRadius(
    const cv::Vec3f& center,
    float radius) const
{
    std::vector<QueryResult> results;

    if (radius <= 0.0f || impl_->tree.empty()) {
        return results;
    }

    using Box3 = bg::model::box<Impl::Point3>;
    Impl::Point3 minPt(center[0] - radius, center[1] - radius, center[2] - radius);
    Impl::Point3 maxPt(center[0] + radius, center[1] + radius, center[2] + radius);
    Box3 query(minPt, maxPt);

    const float radiusSq = radius * radius;
    std::vector<Impl::Entry> candidates;
    impl_->tree.query(bgi::intersects(query), std::back_inserter(candidates));

    for (const auto& entry : candidates) {
        cv::Vec3f pos = Impl::fromBoost(entry.first);
        cv::Vec3f diff = pos - center;
        float distSq = diff.dot(diff);
        if (distSq <= radiusSq) {
            auto dataIt = impl_->pointData.find(entry.second);
            uint64_t collectionId = (dataIt != impl_->pointData.end()) ? dataIt->second.collectionId : 0;
            results.push_back({entry.second, collectionId, pos, distSq});
        }
    }

    std::sort(results.begin(), results.end(),
              [](const QueryResult& a, const QueryResult& b) {
                  return a.distanceSq < b.distanceSq;
              });

    return results;
}

std::optional<PointIndex::QueryResult> PointIndex::nearest(
    const cv::Vec3f& position,
    float maxDistance) const
{
    if (impl_->tree.empty()) {
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

std::vector<PointIndex::QueryResult> PointIndex::kNearest(
    const cv::Vec3f& position,
    size_t k,
    float maxDistance) const
{
    std::vector<QueryResult> results;

    if (k == 0 || impl_->tree.empty()) {
        return results;
    }

    std::vector<Impl::Entry> candidates;
    impl_->tree.query(bgi::nearest(Impl::toBoost(position), static_cast<unsigned>(k)),
                      std::back_inserter(candidates));

    float maxDistSq = maxDistance * maxDistance;

    for (const auto& entry : candidates) {
        cv::Vec3f pos = Impl::fromBoost(entry.first);
        cv::Vec3f diff = pos - position;
        float distSq = diff.dot(diff);

        if (maxDistance >= std::numeric_limits<float>::max() || distSq <= maxDistSq) {
            auto dataIt = impl_->pointData.find(entry.second);
            uint64_t collectionId = (dataIt != impl_->pointData.end()) ? dataIt->second.collectionId : 0;
            results.push_back({entry.second, collectionId, pos, distSq});
        }
    }

    std::sort(results.begin(), results.end(),
              [](const QueryResult& a, const QueryResult& b) {
                  return a.distanceSq < b.distanceSq;
              });

    return results;
}

std::optional<PointIndex::QueryResult> PointIndex::nearestInCollection(
    const cv::Vec3f& position,
    uint64_t collectionId,
    float maxDistance) const
{
    if (impl_->tree.empty()) {
        return std::nullopt;
    }

    // Query more candidates than needed to account for filtering by collection
    // Start with 16, increase if needed
    const size_t initialK = 16;
    size_t k = initialK;
    float maxDistSq = maxDistance * maxDistance;

    while (k <= impl_->pointData.size()) {
        std::vector<Impl::Entry> candidates;
        impl_->tree.query(bgi::nearest(Impl::toBoost(position), static_cast<unsigned>(k)),
                          std::back_inserter(candidates));

        for (const auto& entry : candidates) {
            auto dataIt = impl_->pointData.find(entry.second);
            if (dataIt == impl_->pointData.end() || dataIt->second.collectionId != collectionId) {
                continue;
            }

            cv::Vec3f pos = Impl::fromBoost(entry.first);
            cv::Vec3f diff = pos - position;
            float distSq = diff.dot(diff);

            if (maxDistance < std::numeric_limits<float>::max() && distSq > maxDistSq) {
                return std::nullopt;  // Nearest in collection is too far
            }

            return QueryResult{entry.second, collectionId, pos, distSq};
        }

        // Didn't find any in this collection, expand search
        if (k >= impl_->pointData.size()) {
            break;
        }
        k = std::min(k * 2, impl_->pointData.size());
    }

    return std::nullopt;
}
