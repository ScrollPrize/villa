#pragma once
#include "internal/Database.hpp"
#include <numeric>

namespace vc::objects::detail {
inline constexpr const char* regionColumns =
    "o.uuid,o.kind,h.revision_uuid,b.array_name,b.row_start,b.col_start,b.row_count,b.col_count,"
    "b.min_x,b.min_y,b.min_z,b.max_x,b.max_y,b.max_z";

inline SpatialRegion readSpatialRegion(const Statement& rows)
{
    SpatialRegion r;
    r.objectUuid = formatUuid(rows.blob16(0)); r.kind = rows.text(1);
    if (!rows.isNull(2)) r.revisionUuid = formatUuid(rows.blob16(2));
    r.arrayName = rows.text(3);
    if (!r.arrayName.empty()) {
        r.start.push_back(rows.i64(4)); r.shape.push_back(rows.i64(6));
        if (rows.i64(7)) { r.start.push_back(rows.i64(5)); r.shape.push_back(rows.i64(7)); }
    }
    for (int i = 0; i < 6; ++i) r.bbox[i] = rows.f64(i + 8);
    return r;
}

// Exact double AABBs in a median-split BVH. Descriptors remain in SQL result
// order; only an array of indices is partitioned. No geometry payload is read.
class ResidentSpatialIndex {
    struct Node {
        std::array<double, 6> box;
        size_t begin, end, left = 0, right = 0;
    };
    std::vector<SpatialRegion> regions_;
    std::vector<size_t> order_;
    std::vector<Node> nodes_;

    size_t build(size_t begin, size_t end) {
        auto box = regions_[order_[begin]].bbox;
        for (size_t i = begin + 1; i < end; ++i)
            for (size_t a = 0; a < 3; ++a) {
                box[a] = std::min(box[a], regions_[order_[i]].bbox[a]);
                box[a+3] = std::max(box[a+3], regions_[order_[i]].bbox[a+3]);
            }
        const auto node = nodes_.size();
        nodes_.push_back({box, begin, end});
        if (end - begin > 8) {
            size_t axis = 0;
            // Half extents avoid overflow for valid, extreme double bounds.
            for (size_t a = 1; a < 3; ++a)
                if (box[a+3]/2 - box[a]/2 > box[axis+3]/2 - box[axis]/2) axis = a;
            const auto middle = begin + (end - begin)/2;
            std::nth_element(order_.begin()+begin, order_.begin()+middle, order_.begin()+end,
                [&](size_t a, size_t b) {
                    const auto& x = regions_[a].bbox;
                    const auto& y = regions_[b].bbox;
                    return std::midpoint(x[axis], x[axis+3]) < std::midpoint(y[axis], y[axis+3]);
                });
            const auto left = build(begin, middle), right = build(middle, end);
            nodes_[node].left = left; nodes_[node].right = right;
        }
        return node;
    }
    static bool overlaps(const std::array<double, 6>& a, const std::array<double, 6>& b) {
        for (size_t axis = 0; axis < 3; ++axis)
            if (a[axis] > b[axis+3] || a[axis+3] < b[axis]) return false;
        return true;
    }
public:
    bool load(sqlite3* db, size_t budget) {
        Statement rows(db, std::string("SELECT ") + regionColumns +
            " FROM spatial_regions b JOIN objects o ON o.id=b.object_id "
            "JOIN object_bounds h ON h.object_id=b.object_id "
            "WHERE b.array_name!='' OR h.revision_uuid IS NULL");
        size_t dynamic = sizeof(*this) + 256;
        while (rows.row()) {
            auto r = readSpatialRegion(rows);
            // Charge actual descriptor capacities, allocation overhead, and an
            // upper bound for the BVH and permutation before retaining the row.
            dynamic += r.objectUuid.capacity() + r.kind.capacity() + r.arrayName.capacity() + 3;
            if (r.revisionUuid) dynamic += r.revisionUuid->capacity() + 1;
            dynamic += (r.start.capacity() + r.shape.capacity()) * sizeof(int64_t) + 192;
            regions_.push_back(std::move(r));
            const auto charge = dynamic + regions_.capacity() * sizeof(SpatialRegion)
                + regions_.size() * (2 * sizeof(Node) + sizeof(size_t));
            if (charge > budget) return false;
        }
        if (dynamic > budget) return false;
        // Sort only after the bounded scan succeeds; SQL ORDER BY could sort
        // an arbitrarily large database before we discover it exceeds budget.
        std::sort(regions_.begin(), regions_.end(), [](const auto& a, const auto& b) {
            return std::tie(a.objectUuid, a.arrayName, a.start) < std::tie(b.objectUuid, b.arrayName, b.start);
        });
        order_.resize(regions_.size());
        std::iota(order_.begin(), order_.end(), size_t(0));
        nodes_.reserve(2 * regions_.size());
        if (!regions_.empty()) build(0, regions_.size());
        return true;
    }

    std::vector<SpatialRegion> query(const ObjectSelection& selection) const {
        std::vector<std::string> uuids;
        for (const auto& uuid : selection.uuids) uuids.push_back(formatUuid(parseUuid(uuid)));
        std::sort(uuids.begin(), uuids.end());
        std::vector<size_t> hits;
        auto visit = [&](auto&& self, size_t n) -> void {
            const auto& node = nodes_[n];
            if (!overlaps(node.box, *selection.bbox)) return;
            if (node.left) { self(self, node.left); self(self, node.right); return; }
            for (size_t i = node.begin; i < node.end; ++i) {
                const auto index = order_[i];
                const auto& r = regions_[index];
                if (selection.kind && r.kind != *selection.kind) continue;
                if (!uuids.empty() && !std::binary_search(uuids.begin(), uuids.end(), r.objectUuid)) continue;
                if (overlaps(r.bbox, *selection.bbox)) hits.push_back(index);
            }
        };
        if (!nodes_.empty() && selection.limit != 0) visit(visit, 0);
        std::sort(hits.begin(), hits.end());
        const size_t start = std::min(uint64_t(selection.offset), uint64_t(hits.size()));
        const size_t count = selection.limit ? std::min(uint64_t(*selection.limit), uint64_t(hits.size()-start)) : hits.size()-start;
        std::vector<SpatialRegion> result;
        result.reserve(count);
        for (size_t i = start; i < start + count; ++i) result.push_back(regions_[hits[i]]);
        return result;
    }
};

class SpatialReadCache {
    size_t budget_;
    int version_ = -1;
    bool attempted_ = false;
    std::unique_ptr<ResidentSpatialIndex> index_;
public:
    explicit SpatialReadCache(size_t budget) : budget_(budget) {}
    void begin(int version) {
        if (version != version_) { index_.reset(); attempted_ = false; version_ = version; }
    }
    std::optional<std::vector<SpatialRegion>> query(sqlite3* db, const ObjectSelection& selection) {
        if (selection.tag || selection.connectedTo || selection.family || selection.minLength ||
            selection.afterUuid || selection.afterLength || selection.orderBy != "uuid") return std::nullopt;
        if (!attempted_) {
            if (pragmaInt(db, "user_version") < 5) return std::nullopt;
            auto index = std::make_unique<ResidentSpatialIndex>();
            if (index->load(db, budget_)) index_ = std::move(index);
            attempted_ = true; // An oversized snapshot is tried only once.
        }
        if (index_) return index_->query(selection);
        return std::nullopt;
    }
};
} // namespace vc::objects::detail
