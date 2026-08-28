#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string_view>

namespace spiral::trackio {

using ProgressCallback = std::function<void(
    std::string_view stage, std::size_t completed, std::size_t total)>;

// Read-only views over the packed track format written by tracks.py.  The
// mappings are kept alive by the value object, so spans and raw pointers stay
// valid across moves and copies of the store.
class PackedTrackStore {
public:
    PackedTrackStore() = default;
    explicit PackedTrackStore(const std::filesystem::path& path) { open(path); }

    void open(const std::filesystem::path& path, const ProgressCallback& progress = {});
    bool empty() const noexcept { return track_count == 0; }
    std::size_t point_begin(std::size_t row) const;
    std::size_t point_end(std::size_t row) const;
    std::span<const std::int32_t> track_coordinates(std::size_t row) const;

    std::filesystem::path path;
    std::int64_t track_count = 0;
    std::int64_t point_count = 0;
    const std::int32_t* coords = nullptr;       // (point_count, 3), z/y/x
    const std::int64_t* offsets = nullptr;      // (track_count + 1)
    const std::uint64_t* source_ids = nullptr;  // stable IDs, row aligned
    const std::int8_t* fams = nullptr;          // 0=H, 1=V
    const std::int8_t* families = nullptr;      // compatibility alias
    const double* arcs = nullptr;
    const double* torts = nullptr;
    const std::int32_t* zbounds = nullptr;      // (track_count, 2)
    const std::int32_t* z_bounds = nullptr;     // compatibility alias

private:
    struct Storage;
    std::shared_ptr<Storage> storage_;
};

class CrossingStore {
public:
    CrossingStore() = default;
    CrossingStore(
        const std::filesystem::path& path,
        const PackedTrackStore& tracks) { open(path, tracks); }

    void open(
        const std::filesystem::path& path,
        const PackedTrackStore& tracks,
        const ProgressCallback& progress = {});
    std::size_t begin(std::size_t row) const;
    std::size_t end(std::size_t row) const;

    std::filesystem::path path;
    const std::int64_t* offsets = nullptr;
    const std::int32_t* partners = nullptr;
    const std::int32_t* self_local = nullptr;
    const std::int32_t* partner_local = nullptr;
    const double* positions = nullptr;
    const double* clearances = nullptr;
    const std::uint64_t* source_ids = nullptr;
    std::size_t track_count = 0;
    std::size_t records = 0;

private:
    struct Storage;
    std::shared_ptr<Storage> storage_;
};

} // namespace spiral::trackio
