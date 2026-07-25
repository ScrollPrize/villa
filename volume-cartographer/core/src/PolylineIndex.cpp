#include "vc/core/util/PolylineIndex.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace {
bool finite(const cv::Vec3f& point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}
}

struct PolylineIndex::Impl {
    using Point = bg::model::point<float, 3, bg::cs::cartesian>;
    using Box = bg::model::box<Point>;
    using Entry = std::pair<Box, std::size_t>;
    using Tree = bgi::rtree<Entry, bgi::quadratic<16>>;

    struct Segment {
        std::size_t polyline = 0;
        uint64_t segmentIndex = 0;
    };

    Tree tree;
    std::vector<Polyline> polylines;
    std::vector<Segment> segments;

    static Point point(const cv::Vec3f& value)
    {
        return Point(value[0], value[1], value[2]);
    }

    static Box box(const cv::Vec3f& first, const cv::Vec3f& second, float padding)
    {
        cv::Vec3f lo;
        cv::Vec3f hi;
        for (int axis = 0; axis < 3; ++axis) {
            lo[axis] = std::min(first[axis], second[axis]) - padding;
            hi[axis] = std::max(first[axis], second[axis]) + padding;
        }
        return Box(point(lo), point(hi));
    }
};

PolylineIndex::PolylineIndex() : impl_(std::make_unique<Impl>()) {}
PolylineIndex::~PolylineIndex() = default;
PolylineIndex::PolylineIndex(PolylineIndex&&) noexcept = default;
PolylineIndex& PolylineIndex::operator=(PolylineIndex&&) noexcept = default;

void PolylineIndex::build(std::vector<Polyline> polylines, float padding)
{
    if (!std::isfinite(padding) || padding < 0.0f) {
        throw std::invalid_argument("PolylineIndex padding must be finite and non-negative");
    }

    std::vector<Impl::Entry> entries;
    std::vector<Impl::Segment> segments;
    std::size_t count = 0;
    for (const auto& polyline : polylines) {
        if (polyline.points.empty()) {
            throw std::invalid_argument("PolylineIndex does not accept empty polylines");
        }
        for (const auto& point : polyline.points) {
            if (!finite(point)) {
                throw std::invalid_argument("PolylineIndex does not accept non-finite points");
            }
        }
        count += polyline.points.size() - 1;
    }
    entries.reserve(count);
    segments.reserve(count);

    for (std::size_t polylineIndex = 0; polylineIndex < polylines.size(); ++polylineIndex) {
        const auto& points = polylines[polylineIndex].points;
        for (std::size_t segmentIndex = 0; segmentIndex + 1 < points.size(); ++segmentIndex) {
            const std::size_t storageIndex = segments.size();
            segments.push_back({polylineIndex, static_cast<uint64_t>(segmentIndex)});
            entries.emplace_back(Impl::box(points[segmentIndex], points[segmentIndex + 1], padding),
                                 storageIndex);
        }
    }

    Impl::Tree tree(entries.begin(), entries.end());
    impl_->tree = std::move(tree);
    impl_->segments = std::move(segments);
    impl_->polylines = std::move(polylines);
}

void PolylineIndex::clear()
{
    impl_->tree.clear();
    impl_->segments.clear();
    impl_->polylines.clear();
}

bool PolylineIndex::empty() const { return impl_->segments.empty(); }
std::size_t PolylineIndex::polylineCount() const { return impl_->polylines.size(); }
std::size_t PolylineIndex::segmentCount() const { return impl_->segments.size(); }
const std::vector<PolylineIndex::Polyline>& PolylineIndex::polylines() const { return impl_->polylines; }

std::vector<PolylineIndex::SegmentResult> PolylineIndex::query(
    const cv::Vec3f& minimum,
    const cv::Vec3f& maximum,
    const std::optional<std::string>& category,
    std::size_t limit) const
{
    std::vector<SegmentResult> result;
    if (!finite(minimum) || !finite(maximum) || impl_->tree.empty()) {
        return result;
    }
    cv::Vec3f lo;
    cv::Vec3f hi;
    for (int axis = 0; axis < 3; ++axis) {
        lo[axis] = std::min(minimum[axis], maximum[axis]);
        hi[axis] = std::max(minimum[axis], maximum[axis]);
    }
    const Impl::Box bounds(Impl::point(lo), Impl::point(hi));
    std::vector<Impl::Entry> matches;
    impl_->tree.query(bgi::intersects(bounds), std::back_inserter(matches));
    result.reserve(limit == 0 ? matches.size() : std::min(limit, matches.size()));
    for (const auto& entry : matches) {
        const auto& segment = impl_->segments.at(entry.second);
        const auto& polyline = impl_->polylines.at(segment.polyline);
        if (category && polyline.category != *category) {
            continue;
        }
        const auto index = static_cast<std::size_t>(segment.segmentIndex);
        result.push_back({polyline.objectId, polyline.category, segment.segmentIndex,
                          polyline.points[index], polyline.points[index + 1]});
        if (limit != 0 && result.size() >= limit) {
            break;
        }
    }
    std::sort(result.begin(), result.end(), [](const SegmentResult& left, const SegmentResult& right) {
        return std::tie(left.category, left.objectId, left.segmentIndex)
             < std::tie(right.category, right.objectId, right.segmentIndex);
    });
    return result;
}
