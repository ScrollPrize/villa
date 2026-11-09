#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include <memory>

namespace vc::objects {
// One original AFV fiber. XYZ is little-endian float64, never resampled.
struct AfvFiber {
    int64_t id = 0;
    std::string name, annotation;
    FiberDetail detail;
    std::vector<uint8_t> xyz;
};

// A bounded-memory, snapshot-consistent native AFV v1 cursor. Optional ID ranges
// support independent shard readers; IDs need not be dense. Full scans check the
// declared collection totals at EOF, range scans check every included fiber.
// Cursors are single-reader objects; concurrent callers must use separate cursors.
class AfvSource {
public:
    explicit AfvSource(const std::filesystem::path&, int64_t firstId = 0,
                      int64_t pastLastId = INT64_MAX);
    ~AfvSource();
    AfvSource(const AfvSource&) = delete;
    AfvSource& operator=(const AfvSource&) = delete;
    const std::string& metadataJson() const;
    const std::string& uuid() const;
    int64_t fiberCount() const;
    int64_t pointCount() const;
    void setCacheBytes(size_t);
    // Optional readonly SQLite mapping. Returns the platform/build-capped size.
    size_t setMmapBytes(size_t);
    // Balanced contiguous ID intervals for parallel readers, without decoding
    // geometry. A nonzero limit selects an explicit prefix of fibers.
    std::vector<std::array<int64_t,2>> idRanges(size_t partitions, int64_t limit = 0) const;
    // Fine-grained, point/row-budgeted chunks for a dynamic reader pool. A
    // single long fiber remains intact; sparse source IDs are never assumed dense.
    std::vector<std::array<int64_t,2>> workRanges(size_t targetPoints,size_t maxFibers,int64_t limit=0) const;
    // Reuse the same validated snapshot and prepared cursor for another ID
    // interval. Abandons any unfinished scan and resets its accounting. Full
    // collection totals are checked only if [0, INT64_MAX) reaches EOF.
    void selectRange(int64_t firstId, int64_t pastLastId);
    bool next(AfvFiber&);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// AFV-associated extraction data. Each AFC row becomes one ordinary EdgeRecord;
// native anchors hold positions, and unsupported solver quantities are retained
// under meta.afc with explicit original endpoint identities (normalization safe).
// Like AfvSource, each cursor is confined to one reader at a time.
class AfcSource {
public:
    AfcSource(const std::filesystem::path&, const std::string& afvUuid,
              int64_t pastLastFiberId = INT64_MAX, int64_t firstConstraintId = 0,
              int64_t pastLastConstraintId = INT64_MAX);
    ~AfcSource();
    AfcSource(const AfcSource&) = delete;
    AfcSource& operator=(const AfcSource&) = delete;
    const std::string& metadataJson() const;
    int64_t constraintCount() const;
    void setCacheBytes(size_t);
    size_t setMmapBytes(size_t);
    std::vector<std::array<int64_t,2>> idRanges(size_t partitions, int64_t limit = 0) const;
    void selectRange(int64_t firstId,int64_t pastLastId);
    bool next(EdgeRecord&);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
