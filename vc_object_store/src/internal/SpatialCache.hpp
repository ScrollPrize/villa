#pragma once
#include "internal/Database.hpp"
#include <numeric>
#include "spatial/BoundsIndex.hpp"

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
    std::vector<SpatialRegion> regions_;
    BoundsIndex bounds_;
    std::set<std::string> retired_;
public:
    bool load(sqlite3* db, size_t budget) {
        {
            Statement states(db, "SELECT uuid FROM objects JOIN object_lifecycle lc ON lc.object_id=objects.id WHERE lc.retired=1");
            while (states.row()) {
                retired_.insert(formatUuid(states.blob16(0)));
                if (retired_.size() > budget / 128) return false;
            }
        }
        Statement rows(db, std::string("SELECT ") + regionColumns +
            " FROM spatial_regions b JOIN objects o ON o.id=b.object_id "
            "JOIN object_bounds h ON h.object_id=b.object_id "
            "WHERE b.array_name!='' OR h.revision_uuid IS NULL");
        size_t dynamic = sizeof(*this) + 256 + retired_.size() * 128;
        while (rows.row()) {
            auto r = readSpatialRegion(rows);
            // Charge actual descriptor capacities, allocation overhead, and an
            // upper bound for the BVH and permutation before retaining the row.
            dynamic += r.objectUuid.capacity() + r.kind.capacity() + r.arrayName.capacity() + 3;
            if (r.revisionUuid) dynamic += r.revisionUuid->capacity() + 1;
            dynamic += (r.start.capacity() + r.shape.capacity()) * sizeof(int64_t) + 192;
            regions_.push_back(std::move(r));
            const auto charge = dynamic + regions_.capacity() * sizeof(SpatialRegion)
                + regions_.size() * (2 * (sizeof(BoundsIndex::Box) + 4*sizeof(size_t)) + sizeof(BoundsIndex::Box) + sizeof(size_t));
            if (charge > budget) return false;
        }
        if (dynamic > budget) return false;
        // Sort only after the bounded scan succeeds; SQL ORDER BY could sort
        // an arbitrarily large database before we discover it exceeds budget.
        std::sort(regions_.begin(), regions_.end(), [](const auto& a, const auto& b) {
            return std::tie(a.objectUuid, a.arrayName, a.start) < std::tie(b.objectUuid, b.arrayName, b.start);
        });
        std::vector<BoundsIndex::Box> boxes;
        boxes.reserve(regions_.size());
        for(const auto& region:regions_) boxes.push_back(region.bbox);
        bounds_.build(std::move(boxes));
        return true;
    }

    std::vector<SpatialRegion> query(const ObjectSelection& selection) const {
        std::vector<std::string> uuids;
        for (const auto& uuid : selection.uuids) uuids.push_back(formatUuid(parseUuid(uuid)));
        std::sort(uuids.begin(), uuids.end());
        std::vector<size_t> hits;
        if(selection.limit != 0) bounds_.query(*selection.bbox,[&](size_t index){
            const auto& r=regions_[index];
            if (!selection.includeRetired && retired_.contains(r.objectUuid)) return;
            if(selection.kind && r.kind!=*selection.kind)return;
            if(!uuids.empty()&&!std::binary_search(uuids.begin(),uuids.end(),r.objectUuid))return;
            hits.push_back(index);
        });
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
            auto index = std::make_unique<ResidentSpatialIndex>();
            if (index->load(db, budget_)) index_ = std::move(index);
            attempted_ = true; // An oversized snapshot is tried only once.
        }
        if (index_) return index_->query(selection);
        return std::nullopt;
    }
};
} // namespace vc::objects::detail
