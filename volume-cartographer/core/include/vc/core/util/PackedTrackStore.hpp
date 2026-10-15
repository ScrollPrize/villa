#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include "vc/core/util/MemMap.hpp"

namespace vc {
// Read-only Spiral VCTRK01 store. Geometry stays memory mapped; no Python/DBM
// runtime is needed. Coordinates are integer ZYX, not graph-anchor XYZ.
// The source files must remain immutable while this reader is alive.
class PackedTrackStore {
public:
    explicit PackedTrackStore(const std::filesystem::path& directory);
    std::span<const int32_t> coordinates() const { return coordinates_; }
    std::span<const int64_t> offsets() const { return offsets_; }
    std::span<const uint64_t> sourceIds() const { return sourceIds_; }
    std::span<const int8_t> families() const { return families_; }
private:
    memmap::MappedFileRO coordinateMap_, offsetMap_, sourceIdMap_, familyMap_;
    std::span<const int32_t> coordinates_;
    std::span<const int64_t> offsets_;
    std::span<const uint64_t> sourceIds_;
    std::span<const int8_t> families_;
};
} // namespace vc
