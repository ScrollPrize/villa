#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace vcr {
// Native CT adapter. ZYX buffers; XYZ points. Owns one reader and bounded cache.
class CtReader {
public:
    explicit CtReader(const std::string& location, int level = 0, size_t cacheBytes = 64 * 1024 * 1024);
    ~CtReader();
    CtReader(CtReader&&) noexcept;
    CtReader& operator=(CtReader&&) noexcept;
    std::array<int64_t, 3> shape() const;
    void readRegion(std::array<int64_t, 3> start, std::array<int64_t, 3> shape, std::span<uint8_t> out);
    // Normal XYZ and relative eigengap; invalid support returns all zeros.
    std::vector<std::array<float, 4>> sheetNormals(std::span<const std::array<double, 3>> xyz);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Calls the same implementation as Python's sheet_normals. Centers are local
// XYZ in [48,51), followed by validity; patches are contiguous B*109^3.
void sheetNormalsCpu(std::span<const uint8_t> patches, std::span<const std::array<float, 4>> centers,
                     std::span<std::array<float, 4>> out);
void sheetNormalsCpu(std::span<const float> patches, std::span<const std::array<float, 4>> centers,
                     std::span<std::array<float, 4>> out);

// Shared Zarr v3 -> v2 metadata normalization for native and Python opening.
std::string normalizeZarrMetadata(const std::string& json);
} // namespace vcr
