#pragma once
// Where the slices of a volume go as they are computed, without ever being whole in memory: each slice arrives as the
// block of it that was computed, the rest of the slice being 0 (a masked reconstruction computes a small part of each
// slice). OmeZarrWriter (zarr.hpp) is one.
#include <cstdint>

namespace helirec {

class BlockSink {
  public:
    virtual ~BlockSink() = default;
    // Slice z as its block [x0, x0 + w) x [y0, y0 + h): `rows` (h, w) packed elements, valid during the call; w == 0
    // or h == 0: an empty slice (rows may be null). One call per slice.
    virtual void block(std::int64_t z, int x0, int y0, int w, int h, const void* rows) = 0;
};

}  // namespace helirec
