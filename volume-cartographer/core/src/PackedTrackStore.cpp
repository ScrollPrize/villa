#include "vc/core/util/PackedTrackStore.hpp"

#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace vc {
namespace {
template <typename T>
std::span<const T> mapArray(memmap::MappedFileRO& map, const std::filesystem::path& path, uint64_t count)
{
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)
        || std::filesystem::file_size(path) != count * sizeof(T))
        throw std::runtime_error("invalid packed track array size: " + path.string());
    if (count == 0) return {};
    map.open(path);
    return {static_cast<const T*>(map.data()), static_cast<size_t>(count)};
}
}
PackedTrackStore::PackedTrackStore(const std::filesystem::path& directory)
{
    if constexpr (std::endian::native != std::endian::little)
        throw std::runtime_error("packed track stores require a little-endian host");
    std::array<char, 64> header{};
    std::ifstream input(directory / "header.bin", std::ios::binary);
    if (!input.read(header.data(), header.size()) || input.peek() != std::char_traits<char>::eof()
        || std::memcmp(header.data(), "VCTRK01\0", 8) != 0)
        throw std::runtime_error("invalid packed track header: " + directory.string());
    uint32_t version, size;
    uint64_t tracks, points;
    std::memcpy(&version, header.data() + 8, 4);
    std::memcpy(&size, header.data() + 12, 4);
    std::memcpy(&tracks, header.data() + 16, 8);
    std::memcpy(&points, header.data() + 24, 8);
    if (version != 1 || size != 64 || tracks >= std::numeric_limits<size_t>::max()
        || points > uint64_t(std::numeric_limits<int64_t>::max())
        || points > std::numeric_limits<size_t>::max() / (3 * sizeof(int32_t)))
        throw std::runtime_error("unsupported packed track header: " + directory.string());
    coordinates_ = mapArray<int32_t>(coordinateMap_, directory / "coordinates.i32", points * 3);
    offsets_ = mapArray<int64_t>(offsetMap_, directory / "offsets.i64", tracks + 1);
    sourceIds_ = mapArray<uint64_t>(sourceIdMap_, directory / "source_ids.u64", tracks);
    families_ = mapArray<int8_t>(familyMap_, directory / "family_codes.i8", tracks);
    if (offsets_.front() != 0 || offsets_.back() != static_cast<int64_t>(points))
        throw std::runtime_error("packed track offsets do not match point count");
    for (size_t i = 0; i < tracks; ++i) {
        if (offsets_[i] < 0 || offsets_[i+1] < offsets_[i]
            || offsets_[i+1] > static_cast<int64_t>(points))
            throw std::runtime_error("invalid packed track offsets");
        if (i && sourceIds_[i] <= sourceIds_[i-1])
            throw std::runtime_error("packed track source IDs must be strictly increasing");
        if (families_[i] < -1 || families_[i] > 1)
            throw std::runtime_error("invalid packed track family code");
    }
}
} // namespace vc
