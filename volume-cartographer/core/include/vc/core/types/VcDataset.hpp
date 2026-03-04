#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vc {

enum class VcDtype { uint8, uint16 };

class VcDataset {
public:
    // Open an existing zarr array directory (must contain .zarray)
    explicit VcDataset(const std::filesystem::path& path);
    ~VcDataset();

    VcDataset(VcDataset&&) noexcept;
    VcDataset& operator=(VcDataset&&) noexcept;

    // --- Metadata (cached from .zarray on construction) ---
    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& defaultChunkShape() const;
    size_t defaultChunkSize() const;  // product of chunk dims
    VcDtype getDtype() const;
    size_t dtypeSize() const;  // 1 for uint8, 2 for uint16
    const std::filesystem::path& path() const;
    const std::string& delimiter() const;

    // --- Decompression ---
    // Decompress raw compressed bytes into output buffer.
    // nElements = expected number of elements (NOT bytes).
    void decompress(std::span<const uint8_t> compressed,
                    void* output, size_t nElements) const;

    // --- Chunk I/O ---
    // Check if a chunk file exists on disk.
    bool chunkExists(size_t iz, size_t iy, size_t ix) const;

    // Read and decompress a chunk, writing decompressed data to output.
    // Returns false if chunk doesn't exist.
    bool readChunk(size_t iz, size_t iy, size_t ix, void* output) const;

    // Write a chunk (input is raw uncompressed data).
    bool writeChunk(size_t iz, size_t iy, size_t ix,
                    const void* input, size_t nbytes);

    // --- Region I/O (replaces z5::multiarray::readSubarray) ---
    bool readRegion(const std::vector<size_t>& offset,
                    const std::vector<size_t>& regionShape,
                    void* output) const;

    bool writeRegion(const std::vector<size_t>& offset,
                     const std::vector<size_t>& regionShape,
                     const void* data);

    struct Impl;
private:
    std::unique_ptr<Impl> impl_;
};

// --- Factory functions ---

// Open zarr root and return one VcDataset per pyramid level.
// Scans subdirectories for .zarray files, sorts numerically.
std::vector<std::unique_ptr<VcDataset>> openZarrLevels(
    const std::filesystem::path& zarrRoot);

// Read .zattrs JSON from a zarr group directory.
nlohmann::json readZarrAttributes(const std::filesystem::path& groupPath);

// Write .zattrs JSON to a zarr group directory.
void writeZarrAttributes(const std::filesystem::path& groupPath,
                          const nlohmann::json& attrs);

// Open or create a zarr group + array for writing (used by Zarr.cpp).
// Returns a VcDataset backed by a newly created zarr array with the given options.
std::unique_ptr<VcDataset> createZarrDataset(
    const std::filesystem::path& parentPath,
    const std::string& name,
    const std::vector<size_t>& shape,
    const std::vector<size_t>& chunks,
    VcDtype dtype,
    const std::string& compressor = "blosc",
    const std::string& dimensionSeparator = ".");

}  // namespace vc
