#pragma once

// Ink labels drawn on a segment's vc_render_tifxyz canvas: the label raster,
// the paint operations, and the undo history. GUI-free.
//
// One byte per label pixel holds three bits: ink, supervision (the trainer
// learns from the pixel) and validation. Pixels live in 256 x 256 tiles that
// are either Uniform (one value, no storage) or Dense (a copy-on-write array).
// While no EditTxn is open every reachable tile is immutable, so a snapshot is
// a vector of shared pointers that worker threads can read without locking.

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace vc3d::ink_labels {

inline constexpr uint8_t kInk = 1;
inline constexpr uint8_t kSupervision = 2;
inline constexpr uint8_t kValidation = 4;
inline constexpr uint8_t kValueMask = kInk | kSupervision | kValidation;
inline constexpr int kTileSize = 256;
inline constexpr int kTilePixels = kTileSize * kTileSize;
// Pixels per summary cell edge: a tile's summary is 32 x 32 codes.
inline constexpr int kSummaryCell = 8;
inline constexpr int kSummarySize = kTileSize / kSummaryCell;
inline constexpr std::size_t kHistoryBudgetBytes = std::size_t(256) << 20;
inline constexpr double kMinRadius = 0.5;
inline constexpr double kMaxRadius = 512.0;
// Largest |coordinate| of a shape point, in label px. Shapes are rasterized
// in fixed point with 8 fractional bits, so a point much farther cannot be
// represented in an int; it is also far beyond any render canvas. A shape
// with a point outside this range changes nothing.
inline constexpr double kMaxShapeCoordinate = double(1 << 22);

enum class PaintClass : uint8_t { Ink, Background, Validation };
enum class Actor : uint8_t { User, Agent };

// v' = (v & ~clear) | set. Applying an op twice changes nothing more.
struct PaintOp {
    uint8_t set = 0, clear = 0;
    [[nodiscard]] constexpr uint8_t apply(uint8_t v) const { return uint8_t((v & ~clear) | set); }
    bool operator==(const PaintOp&) const = default;
};

// Ink: set ink and supervision. Background: set supervision and clear ink
// (keep ink when backgroundKeepsInk). Validation: set validation. The eraser
// returns ink and background to unknown (supervision cleared), and clears
// validation for the validation class; it never asserts background.
[[nodiscard]] PaintOp paintOp(PaintClass cls, bool erase, bool backgroundKeepsInk);

// Display codes. Bit 1: any ink. Bit 2: all supervised. Bit 4: any
// validation. Bit 8: any supervised. Pooling ORs the "any" bits and ANDs the
// "all" bit, so the neutral code 2 stands for pixels outside the canvas.
[[nodiscard]] constexpr uint8_t summaryCode(uint8_t v)
{
    return uint8_t((v & 7) | ((v & 2) << 2));
}
[[nodiscard]] constexpr uint8_t poolCodes(uint8_t a, uint8_t b)
{
    return uint8_t(((a | b) & 0b1101) | (a & b & 0b0010));
}
inline constexpr uint8_t kNeutralCode = 2;

// Label pixels by meaning, from per-value counts: ink (I set), background
// (supervised, not ink), validation (V set) and unlabeled (neither I nor S).
struct LabelCounts {
    uint64_t ink = 0, background = 0, validation = 0, unlabeled = 0;
    [[nodiscard]] static LabelCounts from(const std::array<uint64_t, 8>& byValue)
    {
        LabelCounts c;
        for (int v = 0; v < 8; ++v) {
            const uint64_t n = byValue[std::size_t(v)];
            if (v & kInk)
                c.ink += n;
            else if (v & kSupervision)
                c.background += n;
            else
                c.unlabeled += n;
            if (v & kValidation)
                c.validation += n;
        }
        return c;
    }
};

struct Tile {
    std::array<uint8_t, kTilePixels> px{};
    // Level-3 display codes: one per 8 x 8 pixel cell, pooled over in-canvas
    // pixels only.
    std::array<uint8_t, kSummarySize * kSummarySize> summary{};
    // In-canvas pixels per value (v & 7).
    std::array<uint32_t, 8> counts{};
};

struct TileState {
    std::shared_ptr<const Tile> dense;
    uint8_t uniform = 0;
    [[nodiscard]] bool isDense() const { return dense != nullptr; }
    [[nodiscard]] bool sameAs(const TileState& other) const
    {
        return dense == other.dense && (dense || uniform == other.uniform);
    }
};

struct Shape {
    enum class Kind { Disc, Capsule, Polygon } kind = Kind::Disc;
    // Label pixel coordinates, pixel-corner convention (pixel (c, r) covers
    // [c, c+1) x [r, r+1)). Disc: one center. Capsule: two ends. Polygon: at
    // least three vertices, even-odd fill.
    std::vector<cv::Point2d> points;
    double radius = 0;
};

// Where shapes fall on a canvas of `size` label px, judged by their points
// plus or minus the radius the stamps use (vertices only for polygons):
// entirely inside, partly outside, or covering no canvas pixel at all.
enum class CanvasOverlap : uint8_t { Inside, Clipped, Outside };
[[nodiscard]] CanvasOverlap canvasOverlap(const std::vector<Shape>& shapes, cv::Size size);

class TileRaster {
public:
    explicit TileRaster(cv::Size canvas = {});

    [[nodiscard]] cv::Size size() const { return _size; }
    [[nodiscard]] cv::Size tileGrid() const { return _grid; }
    [[nodiscard]] int tileCount() const { return int(_tiles.size()); }
    [[nodiscard]] bool empty() const { return _tiles.empty(); }
    [[nodiscard]] int tileIndex(int tx, int ty) const { return ty * _grid.width + tx; }
    // The tile's rectangle in label pixels, clipped to the canvas.
    [[nodiscard]] cv::Rect tileRect(int index) const;

    [[nodiscard]] uint8_t at(int x, int y) const;
    // `out` becomes rect.size(); pixels outside the canvas read as 0.
    void readRect(const cv::Rect& rect, cv::Mat_<uint8_t>& out) const;
    [[nodiscard]] const TileState& tile(int index) const { return _tiles[std::size_t(index)]; }
    [[nodiscard]] uint32_t revision(int index) const { return _revisions[std::size_t(index)]; }
    // In-canvas pixels per value (v & 7).
    [[nodiscard]] std::array<uint64_t, 8> counts() const;
    // O(tiles) pointer copy. No EditTxn may be open.
    [[nodiscard]] TileRaster snapshot() const { return *this; }
    [[nodiscard]] std::size_t denseTileCount() const;
    // Used by loaders and the history. Bumps the tile's revision.
    void setTile(int index, TileState state);

private:
    friend class EditTxn;
    cv::Size _size;
    cv::Size _grid;
    std::vector<TileState> _tiles;
    std::vector<uint32_t> _revisions;
};

// Recomputes a tile's summary and counts from its pixels. `inCanvas` is the
// tile's in-canvas rectangle in tile-local pixels.
void finalizeTile(Tile& tile, const cv::Rect& inCanvas);
// The state a finalized tile publishes: Uniform when every in-canvas pixel
// holds one value, otherwise Dense.
[[nodiscard]] TileState publishTile(std::shared_ptr<Tile> tile, const cv::Rect& inCanvas);

struct TileChange {
    int index = 0;
    TileState before, after;
};

struct HistoryEntry {
    uint64_t serial = 0;
    std::string label;
    Actor actor = Actor::User;
    std::vector<TileChange> changes;
    uint64_t changedPixels = 0;
    std::size_t bytes = 0;
};

// One operation's edits. Tiles are cloned on first change and written in
// place, so the raster shows the edit live; commit() collapses, finalizes and
// freezes them. Destroying an uncommitted transaction restores every tile.
class EditTxn {
public:
    EditTxn(TileRaster& raster, std::string label, Actor actor);
    ~EditTxn();
    EditTxn(const EditTxn&) = delete;
    EditTxn& operator=(const EditTxn&) = delete;

    // Returns the number of pixels whose value changed.
    uint64_t apply(const Shape& shape, PaintOp op);
    // Applies `op` where mask != 0; mask pixel (0,0) lands on `origin`.
    uint64_t applyMask(cv::Point origin, const cv::Mat_<uint8_t>& mask, PaintOp op);
    // Label pixels changed since the last call, or an empty rect.
    [[nodiscard]] cv::Rect takeDirtyRect();
    [[nodiscard]] bool empty() const { return _working.empty(); }
    [[nodiscard]] uint64_t changedPixels() const { return _changedPixels; }
    [[nodiscard]] Actor actor() const { return _actor; }
    [[nodiscard]] const std::string& label() const { return _label; }
    // Publishes the edited tiles and returns the history entry. The
    // transaction is finished afterwards; further edits are ignored.
    [[nodiscard]] HistoryEntry commit(uint64_t serial);

private:
    struct Working {
        TileState before;
        std::shared_ptr<Tile> tile;
    };
    // A mask in tile-local coordinates: pixel (lx, ly) of the tile reads
    // data[(ly + offset.y) * step + lx + offset.x].
    struct MaskView {
        const uint8_t* data = nullptr;
        std::size_t step = 0;
        cv::Point offset;
    };
    // For every tile meeting `bounds`, draws the shape into a 256 x 256
    // scratch mask with `draw(mask, tile origin)` and applies `op` under it.
    uint64_t applyToTiles(const cv::Rect& bounds,
                          PaintOp op,
                          const std::function<void(cv::Mat_<uint8_t>&, cv::Point)>& draw);
    uint64_t applyMaskedTile(int index, const MaskView& mask, const cv::Rect& local, PaintOp op);
    Tile& workingTile(int index);
    void refreshSummary(Tile& tile, const cv::Rect& local, const cv::Rect& inCanvas);
    void rollback();

    TileRaster& _raster;
    std::string _label;
    Actor _actor;
    std::unordered_map<int, Working> _working;
    std::vector<int> _order;
    uint64_t _changedPixels = 0;
    cv::Rect _dirty;
    bool _finished = false;
};

class History {
public:
    explicit History(std::size_t budgetBytes = kHistoryBudgetBytes) : _budget(budgetBytes) {}

    // Clears redo, then drops the oldest entries while the total exceeds the
    // budget (never the newest).
    void push(HistoryEntry entry);
    bool undo(TileRaster& raster);
    bool redo(TileRaster& raster);
    [[nodiscard]] const HistoryEntry* top() const { return _undo.empty() ? nullptr : &_undo.back(); }
    [[nodiscard]] const std::deque<HistoryEntry>& undoEntries() const { return _undo; }
    [[nodiscard]] const HistoryEntry* nextRedo() const { return _redo.empty() ? nullptr : &_redo.back(); }
    [[nodiscard]] std::size_t undoDepth() const { return _undo.size(); }
    [[nodiscard]] std::size_t redoDepth() const { return _redo.size(); }
    [[nodiscard]] std::size_t bytes() const { return _bytes; }
    // The serial of the current state: the top entry's, or the base serial
    // when nothing can be undone.
    [[nodiscard]] uint64_t stateSerial() const { return _undo.empty() ? _baseSerial : _undo.back().serial; }
    [[nodiscard]] uint64_t nextSerial() { return _nextSerial++; }
    // Called with the serial of the state a save wrote, not the state at
    // completion, so edits made during the save stay dirty.
    void markClean(uint64_t serial) { _cleanSerial = serial; }
    [[nodiscard]] uint64_t cleanSerial() const { return _cleanSerial; }
    [[nodiscard]] bool dirty() const { return stateSerial() != _cleanSerial; }
    // Forget everything and start from `serial` as the clean base.
    void reset(uint64_t serial = 0);

private:
    std::deque<HistoryEntry> _undo, _redo;
    std::size_t _budget;
    std::size_t _bytes = 0;
    uint64_t _baseSerial = 0;
    uint64_t _cleanSerial = 0;
    uint64_t _nextSerial = 1;
};

}  // namespace vc3d::ink_labels
