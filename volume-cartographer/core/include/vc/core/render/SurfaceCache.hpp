#pragma once

#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/types/Sampling.hpp"
#include "vc/core/util/Compositing.hpp"

#include <opencv2/core/mat.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <array>
#include <vector>

class QuadSurface;

namespace vc::render {

// Tiles of resampled *surface* space, as opposed to the axis-aligned volume
// chunks of ChunkCache. A tile is `kTileSize x kTileSize x wCount` samples in
// (u, v, w) where:
//
//   u, v  nominal surface coordinates -- the same space as the flattened
//         viewer's surface pointer, one unit ~ one voxel of arclength along
//         the sheet. Signed.
//   w     signed offset in level-0 voxels along the unit surface normal at
//         (u, v).
//
// The value at (u, v, w) is the volume intensity at S(u,v) + w * N(u,v) for
// the coord/normal QuadSurface::gen() produces, so this class is a
// memoization of the sampling function the flattened renderer already
// evaluates per frame -- not a new one.
//
// Tile (level, tu, tv) samples volume level `level` and steps 2^level nominal
// units per sample in u and v, so a tile always covers the same screen area at
// its matching zoom. `w` steps one level-0 voxel at every level: the band is
// always physically [wMin, wMin + wCount - 1] and is never widened.
class SurfaceGeometryTileCache;

class SurfaceCache {
public:
    static constexpr int kTileSize = 128;

    struct Options {
        std::size_t byteCapacity = 4ULL << 30;
        // 0 -> derive from the backing array's pyramid depth.
        int levels = 0;
        // Stored band; tile depth. Never widened at runtime.
        int wMin = -16;
        int wCount = 32;
        // Bounds this cache's queued-plus-running fills. 0 derives it from the
        // process-wide pool; admission stays within two tiles per worker so
        // panning cannot outrun the fill consumers.
        std::size_t fillWorkers = 0;
        // Upper bound on the tile bytes admitted by one requestView() call.
        // Sized so one screenful at the matching zoom (~96 tiles) is admitted in
        // a single round: each extra round costs a full render round-trip before
        // the next batch is queued, which is what makes a pan fill in visible
        // steps. VC_SURFACE_ADMISSION_PAGE_BYTES overrides.
        std::size_t admissionPageBytes = 64ULL << 20;
        vc::Sampling sampling = vc::Sampling::Trilinear;
        // Share one coords/normals tile cache with a second SurfaceCache over
        // the same surface (the overlay channel) so each tile's gen() runs
        // once and fills both. Created privately when null.
        std::shared_ptr<SurfaceGeometryTileCache> geometry;
    };

    struct Stats {
        std::size_t bytes = 0;
        std::size_t capacity = 0;
        std::size_t tiles = 0;
        std::size_t tilesInFlight = 0;
        std::size_t tilesIncomplete = 0;
    };

    struct SampleStats {
        int coveredPixels = 0;
        // Tiles the view needed at its start level that were not resident.
        int missingTiles = 0;
        // At least one pixel was filled from a coarser cached level.
        bool usedCoarseFallback = false;
    };

    using TileReadyCallbackId = std::uint64_t;

    SurfaceCache(std::shared_ptr<IChunkedArray> volume,
                 std::shared_ptr<QuadSurface> surface,
                 Options options);
    ~SurfaceCache();

    SurfaceCache(const SurfaceCache&) = delete;
    SurfaceCache& operator=(const SurfaceCache&) = delete;

    // Concurrent tile fills the process-wide pool can run. Callers sizing a
    // chunk pool for the filler need this: one round of fills is what has to be
    // resident at once.
    [[nodiscard]] static std::size_t fillWorkerCount();

    [[nodiscard]] int levels() const;
    [[nodiscard]] int wMin() const;
    [[nodiscard]] int wCount() const;
    [[nodiscard]] std::size_t tileBytes() const;
    [[nodiscard]] std::shared_ptr<SurfaceGeometryTileCache> geometryTiles() const;

    // True iff [wLow, wHigh] lies inside the stored band. The caller falls
    // back to direct volume sampling when it does not; the band never grows.
    [[nodiscard]] bool bandCovers(double wLow, double wHigh) const;

    // Inclusive w range a frame needs at normal offset `w`. Exposed so a
    // caller's bandCovers() test and the reduction below share one definition.
    // `composite.enabled` must already account for methods the streaming path
    // cannot serve -- that predicate lives with the renderer.
    static void compositeWRange(const CompositeRenderSettings& composite,
                                double w,
                                double& wLow,
                                double& wHigh);
    static void overlayCompositeWRange(const OverlayCompositeSettings& composite,
                                       double w,
                                       double zStep,
                                       double& wLow,
                                       double& wHigh);

    // --- Render path. Never queues work; reads resident tiles only. ---
    //
    // Fills `out` / `coverage` for a framebuffer whose top-left pixel centre
    // is at (uMin, vMin) with `scale` framebuffer pixels per nominal unit, at
    // normal offset `w`. When `composite.enabled`, the layer stack
    // [w - layersBehind, w + layersFront] (direction per reverseDirection) is
    // reduced per pixel from the same tiles, so compositing costs no extra
    // fetches and no extra tiles. Pixels already marked in `coverage` are left
    // alone. Uncovered pixels fall back to coarser cached levels, and levels
    // beyond the coarsest are served by resampling the coarsest tiles.
    SampleStats sampleView(int startLevel,
                           double uMin,
                           double vMin,
                           double scale,
                           double w,
                           const CompositeRenderSettings& composite,
                           cv::Mat_<uint8_t>& out,
                           cv::Mat_<uint8_t>& coverage) const;

    // Overlay-channel variant: max/mean/min over the overlay layer stack, no
    // iso cutoff. `zStep` keeps "front" pointing the same physical direction
    // as the primary composite.
    SampleStats sampleViewOverlay(int startLevel,
                                  double uMin,
                                  double vMin,
                                  double scale,
                                  double w,
                                  const OverlayCompositeSettings& composite,
                                  double zStep,
                                  cv::Mat_<uint8_t>& out,
                                  cv::Mat_<uint8_t>& coverage) const;

    // --- Scheduler path (UI thread). ---
    //
    // Computes the tile set for this view, admits a bounded page of missing
    // tiles ordered by distance from the view centre, and records the view so
    // queued-but-not-started fills for superseded views can be dropped.
    void requestView(int startLevel,
                     double uMin,
                     double vMin,
                     double scale,
                     int fbW,
                     int fbH,
                     std::uint64_t viewGeneration,
                     ChunkRequestContext request = {});

    TileReadyCallbackId addTileReadyListener(std::function<void()> callback);
    void removeTileReadyListener(TileReadyCallbackId id);

    // Adjustable in place: enforces the new LRU ceiling without rebuilding
    // tiles, so a settings change can be applied live.
    void setByteCapacity(std::size_t bytes);

    void invalidateAll();
    // Drop the tiles overlapping a changed rectangle of the surface's *grid*
    // (row/col) space, e.g. a brush edit.
    void invalidateSurfaceRegion(const cv::Rect& gridCells);

    [[nodiscard]] Stats stats() const;

    // Stop accepting fills and wait for in-flight ones. Called by the
    // destructor; safe to call early and repeatedly.
    void shutdown();

private:
    struct State;
    std::shared_ptr<State> _state;
};

// Coords/normals for one tile of surface space, computed by a single
// QuadSurface::gen() call and shared by the base and overlay SurfaceCache
// instances over that surface.
class SurfaceGeometryTileCache {
public:
    struct Tile {
        // kTileSize x kTileSize, level-0 XYZ and unit normals.
        cv::Mat_<cv::Vec3f> coords;
        cv::Mat_<cv::Vec3f> normals;
        // Coord is a real surface point (sentinel test). Enough for w == 0.
        cv::Mat_<uint8_t> valid;
        // Coord *and* normal are finite, so an offset along the normal is
        // defined. Excludes the one-grid-cell border where gen() has no
        // neighbours to difference.
        cv::Mat_<uint8_t> validOffset;
    };

    explicit SurfaceGeometryTileCache(std::shared_ptr<QuadSurface> surface,
                                      std::size_t maxTiles = 192);
    ~SurfaceGeometryTileCache();

    SurfaceGeometryTileCache(const SurfaceGeometryTileCache&) = delete;
    SurfaceGeometryTileCache& operator=(const SurfaceGeometryTileCache&) = delete;

    // Computes on miss. Concurrent callers for the same key wait for the
    // first one instead of duplicating the gen().
    std::shared_ptr<const Tile> get(int level, int tu, int tv);

    // Resolve sparse framebuffer samples through the same geometry tiles used
    // by SurfaceCache fills. This may generate missing geometry tiles, which
    // makes the following fill reuse them rather than invoking gen() again.
    void sampleView(int level,
                    double uMin,
                    double vMin,
                    double scale,
                    const std::vector<std::array<float, 2>>& viewportPositions,
                    std::vector<cv::Vec3f>& coords,
                    std::vector<cv::Vec3f>* normals = nullptr);

    void invalidateAll();
    void invalidateSurfaceRegion(const cv::Rect& gridCells);
    [[nodiscard]] std::size_t size() const;

    // Nominal (u, v) rectangle covered by a grid-space rectangle of the
    // surface this cache was built for. Shared with SurfaceCache's region
    // invalidation so both use one definition.
    [[nodiscard]] cv::Rect2d gridCellsToNominal(const cv::Rect& gridCells) const;

private:
    struct State;
    std::shared_ptr<State> _state;
};

} // namespace vc::render
