#pragma once

#include <opencv2/core/mat.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_map>
#include <vector>

namespace vc3d::inklabels {

// One label per pixel of the segment's flattened render. Unlabelled is the
// default: the trainer ignores it, while no-ink is a negative example.
enum class Label : uint8_t { Unlabelled = 0, NoInk = 1, Ink = 2 };

// The full-resolution canvas extent vc_render_tifxyz gives a stored grid
// extent (render scale 1): the reciprocal of the float32 scale in double,
// rounded half away from zero, at least 1. Labels share this frame with the
// renders and the predictions made from them.
int fullResolutionExtent(int stored, float scale);
// scale is a QuadSurface scale: [0] columns, [1] rows.
cv::Size canvasSize(int gridRows, int gridCols, const cv::Vec2f& scale);

// A label image with stroke painting and per-stroke undo. Strokes stamp discs
// (pixel centres within the radius) along the pointer path; undo restores the
// tiles a stroke touched.
class InkLabelLayer
{
public:
    // At least this many strokes stay undoable whatever their size.
    static constexpr std::size_t kMinUndoStrokes = 20;

    explicit InkLabelLayer(cv::Size size = {});

    [[nodiscard]] const cv::Mat_<uint8_t>& labels() const { return _labels; }
    [[nodiscard]] cv::Size size() const { return _labels.size(); }
    [[nodiscard]] bool empty() const { return _labels.empty(); }
    // Replace every label (a load); clears undo history.
    void reset(const cv::Mat_<uint8_t>& labels);

    void beginStroke(Label value, float radius);
    // Extend the stroke to `point` (label pixels, continuous: pixel (x, y)
    // spans [x, x+1) x [y, y+1)). Returns the changed rectangle.
    cv::Rect strokeTo(const cv::Point2f& point);
    // Push the stroke onto the undo history. False when it changed nothing.
    bool endStroke();
    // Revert an unfinished stroke. Returns the changed rectangle.
    cv::Rect cancelStroke();
    [[nodiscard]] bool strokeActive() const { return _stroke.has_value(); }

    [[nodiscard]] bool canUndo() const { return !_undo.empty() && !_stroke; }
    [[nodiscard]] bool canRedo() const { return !_redo.empty() && !_stroke; }
    [[nodiscard]] std::size_t undoDepth() const { return _undo.size(); }
    [[nodiscard]] std::size_t redoDepth() const { return _redo.size(); }
    // Returns the changed rectangle (empty when nothing to do).
    cv::Rect undo();
    cv::Rect redo();

    // Count of pixels per label, indexed by Label.
    [[nodiscard]] std::array<std::size_t, 3> counts() const;
    // Whether labels changed since the last markSaved().
    [[nodiscard]] bool modified() const { return _revision != _savedRevision; }
    void markSaved() { _savedRevision = _revision; }

    // Memory the undo history may use beyond kMinUndoStrokes strokes.
    void setUndoMemoryBudget(std::size_t bytes) { _undoBudget = bytes; }

private:
    static constexpr int kTile = 128;
    struct TileEdit {
        int index{0};
        cv::Mat_<uint8_t> before;
        cv::Mat_<uint8_t> after;
    };
    struct Edit {
        std::vector<TileEdit> tiles;
        std::size_t bytes{0};
    };
    struct Stroke {
        uint8_t value{0};
        float radius{1.0f};
        std::optional<cv::Point2f> last;
        std::unordered_map<int, cv::Mat_<uint8_t>> before;
    };

    cv::Rect tileRect(int index) const;
    cv::Rect stampDisc(const cv::Point2f& centre);
    void captureTiles(const cv::Rect& rect);
    cv::Rect apply(const Edit& edit, bool forward);
    void trimHistory();

    cv::Mat_<uint8_t> _labels;
    int _tilesX{0};
    std::optional<Stroke> _stroke;
    std::deque<Edit> _undo;
    std::deque<Edit> _redo;
    std::size_t _undoBytes{0};
    std::size_t _undoBudget{std::size_t(512) << 20};
    uint64_t _revision{0};
    uint64_t _savedRevision{0};
};

} // namespace vc3d::inklabels
