#include "vc_object_store/PackedTrackStore.hpp"

#include <array>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <optional>
#include <set>
#include <nlohmann/json.hpp>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace vc {
namespace {
template <typename T>
std::span<const T> mapArray(objects::memmap::MappedFileRO& map, const std::filesystem::path& path, uint64_t count)
{
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)
        || std::filesystem::file_size(path) != count * sizeof(T))
        throw std::runtime_error("invalid packed track array size: " + path.string());
    if (count == 0) return {};
    map.open(path);
    return {static_cast<const T*>(map.data()), static_cast<size_t>(count)};
}

// Same rule as spiral-fitting's _db_signatures_match. Name and size match
// exactly. mtime matches exactly, except when either side is a whole second
// (divisible by 1e9, the part HTTP Last-Modified keeps) and the seconds match.
// Two different nanosecond stamps in one second stay stale. JSON integers
// only: floats and other non-integers are malformed and therefore stale.
// Bad metadata is stale rather than a JSON type error. Stamps are after
// 1970 and the Python writer stores integers, so toward-zero division by
// 1000000000 matches that writer's floor division.
constexpr std::int64_t kSignatureSecondNs = 1000000000;

std::optional<std::int64_t> signatureInteger(const nlohmann::json& value)
{
    try {
        if (!value.is_number_integer()) return std::nullopt;
        return value.get<std::int64_t>();
    } catch (const nlohmann::json::exception&) {
        return std::nullopt;
    }
}

bool samePackedDbSignature(const nlohmann::json& stored, const nlohmann::json& expected)
{
    if (!stored.is_array() || stored.size() != expected.size()) return false;
    try {
        for (std::size_t i = 0; i < stored.size(); ++i) {
            const auto& item = stored.at(i);
            const auto& want = expected.at(i);
            if (!item.is_array() || item.size() != 3) return false;
            if (item.at(0) != want.at(0)) return false;
            const auto haveSize = signatureInteger(item.at(1));
            const auto needSize = signatureInteger(want.at(1));
            if (!haveSize || !needSize || *haveSize != *needSize) return false;
            const auto have = signatureInteger(item.at(2));
            const auto need = signatureInteger(want.at(2));
            if (!have || !need) return false;
            if (*have == *need) continue;
            const bool truncated = *have % kSignatureSecondNs == 0 || *need % kSignatureSecondNs == 0;
            if (!truncated || *have / kSignatureSecondNs != *need / kSignatureSecondNs) return false;
        }
    } catch (const nlohmann::json::exception&) {
        return false;
    }
    return true;
}
}
std::filesystem::path resolvePackedTrackStore(const std::filesystem::path& input)
{
    namespace fs = std::filesystem;
    if (fs::is_directory(input)) return fs::canonical(input);
    fs::path logical = fs::absolute(input).lexically_normal();
    if (logical.extension() == ".db" && logical.stem().extension() == ".dbm")
        logical.replace_extension();
    const fs::path packed(logical.string() + ".vctracks");
    if (!fs::is_directory(packed))
        throw std::runtime_error("packed track geometry missing: " + packed.string()
            + "; create it with Spiral's convert_track_store.py for " + logical.string());
    std::ifstream metadataFile(packed / "metadata.json");
    const auto metadata = nlohmann::json::parse(metadataFile);
    nlohmann::json actual = nlohmann::json::array();
    std::set<fs::path> seen;
    for (const auto* suffix : {"", ".db", ".dat", ".dir", ".bak", ".pag"}) {
        const fs::path backing(logical.string() + suffix);
        if (!fs::is_regular_file(backing) || !seen.insert(fs::canonical(backing)).second) continue;
        const auto modified = std::chrono::file_clock::to_sys(fs::last_write_time(backing));
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(modified.time_since_epoch()).count();
        actual.push_back({backing.filename().string(), fs::file_size(backing), ns});
    }
    std::sort(actual.begin(), actual.end());
    if (actual.empty()) throw std::runtime_error("DBM backing files missing: " + logical.string());
    if (metadata.value("version", 0) != 1 || !metadata.contains("source_db_signature")
        || !samePackedDbSignature(metadata.at("source_db_signature"), actual))
        throw std::runtime_error("packed tracks are stale for " + logical.string()
            + "; rebuild with Spiral's convert_track_store.py --force");
    return fs::canonical(packed);
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
