#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace vc::objects {
namespace detail { struct SurfaceTile; }

// Decoded surface tiles (points and validity) of geometry revisions, kept for
// every read of the stores given it (OpenOptions::surfaceTiles), across their
// connections: geometry regions and nearest-surface queries that find a tile
// here neither read, decompress nor verify its chunks. Revisions never change,
// so a tile never goes stale; past the capacity the least recently used go.
// Thread-safe.
class SurfaceTileCache {
public:
    explicit SurfaceTileCache(std::size_t capacityBytes);
    ~SurfaceTileCache();
    SurfaceTileCache(const SurfaceTileCache&) = delete;
    SurfaceTileCache& operator=(const SurfaceTileCache&) = delete;

    // Evicts down to a smaller capacity at once.
    void setCapacity(std::size_t bytes);
    std::size_t capacity() const;
    // Bytes held.
    std::size_t size() const;

    // For the store's reads: the tile of `revision` at row, column (rows x
    // columns vertices), if held; and keeping one.
    std::shared_ptr<const detail::SurfaceTile> find(const std::string& revision, int64_t row, int64_t column,
                                                    int64_t rows, int64_t columns) const;
    void insert(const std::string& revision, int64_t row, int64_t column, int64_t rows, int64_t columns,
                std::shared_ptr<const detail::SurfaceTile> tile);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
