#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "helirec/calibration/record.hpp"

namespace helirec {
struct ProjectionSubsetSeries {
    std::string entry, role;
    std::int64_t first = 0, count = 0, original_count = 0;
    std::vector<calibration::FramePiece> pieces;
    std::vector<double> angle_deg, height_mm, epoch;
};
struct ProjectionSubset {
    std::string master, label, detector;
    int rows = 0, cols = 0, slices = 0;
    double pixel_um = 0, center_mm = 0, thickness_mm = 0, context_mm = 0, pitch_mm = 0;
    std::vector<ProjectionSubsetSeries> series;
};
// Plan a physical slab from the recorded helix coordinates. Keep all angles,
// whole source-file pieces touching the slab + context, and all dark/flat series.
// Read a local metadata copy of the master to avoid checking every remote file.
ProjectionSubset plan_projection_subset(const std::string& master, double center_mm, int slices,
                                        double context_turns = 1.5);

// A single uncompressed HDF5 frame stack, shared by C++ and Python downloaders.
// Publish the file only after all writes and close() succeed; the caller owns
// atomic rename, metadata, checksums, and restart policy.
class UncompressedFrameWriter {
  public:
    UncompressedFrameWriter(const std::string& path, const std::string& dataset, std::int64_t frames,
                            int rows, int cols, const std::string& dtype);
    ~UncompressedFrameWriter();
    void write(std::int64_t first, std::int64_t count, const std::uint16_t* data);
    void write(std::int64_t first, std::int64_t count, const float* data);
    void close();
    int rows() const;
    int cols() const;
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace helirec
