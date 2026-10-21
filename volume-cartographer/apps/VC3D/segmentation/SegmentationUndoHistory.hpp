#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace segmentation
{

// Delta-based vertex edit for efficient undo
struct VertexDelta
{
    int row{0};
    int col{0};
    cv::Vec3f previousWorld{0.0f, 0.0f, 0.0f};  // Position before the edit
};

// How deep an undo history may grow. Beyond minEntries, the oldest entries are
// evicted while the history is over byteBudget; beyond maxEntries they are
// evicted whatever their size. minEntries keeps the old fixed depth of ten
// steps even when each step is a copy of a large surface.
struct UndoLimits
{
    std::size_t byteBudget{std::size_t{256} << 20};  // 256 MiB
    std::size_t minEntries{10};
    std::size_t maxEntries{100};

    [[nodiscard]] bool exceeded(std::size_t entries, std::size_t bytes) const
    {
        return entries > maxEntries || (bytes > byteBudget && entries > minEntries);
    }
};

// Evicts the oldest entries (from the front) of an undo stack while it exceeds
// limits, keeping at least the newest one. bytesOf(entry) gives an entry's size.
template <typename Stack, typename BytesOf>
void trimOldest(Stack& stack, const UndoLimits& limits, BytesOf bytesOf)
{
    std::size_t bytes = 0;
    for (const auto& entry : stack) {
        bytes += bytesOf(entry);
    }
    while (stack.size() > 1 && limits.exceeded(stack.size(), bytes)) {
        bytes -= bytesOf(stack.front());
        stack.pop_front();
    }
}

// Undo and redo stacks for segmentation edits.
//
// An entry holds the grid state that an edit, an undo or a redo replaced:
// either the changed vertices (deltas) or a full copy of the grid (legacy
// snapshots). The two stacks together are held to UndoLimits, because a
// snapshot costs as much as the whole surface while a delta costs a few bytes
// per vertex. The newest entry is always kept, so the last edit, undo or redo
// can always be reversed.
class UndoHistory
{
public:
    struct Entry
    {
        cv::Mat_<cv::Vec3f> points;        // Full snapshot (legacy, memory intensive)
        std::vector<VertexDelta> deltas;   // Delta-based (memory efficient)
        // Approval mask undo entry painted by the same edit (0 = none).
        std::uint64_t autoApprovalId{0};

        [[nodiscard]] bool isDelta() const { return !deltas.empty(); }
        [[nodiscard]] bool isEmpty() const { return deltas.empty() && points.empty(); }
        [[nodiscard]] std::size_t bytes() const
        {
            return sizeof(Entry) + points.total() * sizeof(cv::Vec3f) +
                   deltas.size() * sizeof(VertexDelta);
        }
    };

    explicit UndoHistory(UndoLimits limits = {})
        : _limits(limits)
    {
    }

    // Capture full surface state (legacy method - memory intensive). Like any
    // new edit, this drops everything that could be redone.
    bool capture(const cv::Mat_<cv::Vec3f>& points)
    {
        if (points.empty()) {
            return false;
        }

        cv::Mat_<cv::Vec3f> clone = points.clone();
        if (clone.empty()) {
            return false;
        }

        Entry entry;
        entry.points = std::move(clone);
        recordNewEdit(std::move(entry));
        return true;
    }

    // Capture only changed vertices (delta-based - memory efficient)
    bool captureDelta(const std::vector<VertexDelta>& deltas)
    {
        if (deltas.empty()) {
            return false;
        }

        Entry entry;
        entry.deltas = deltas;
        recordNewEdit(std::move(entry));
        return true;
    }

    // Drop the last capture because its edit changed nothing. The entries
    // that capture evicted or made unredoable come back, since no edit
    // happened.
    void discardLast()
    {
        if (_undo.empty()) {
            return;
        }
        popBack(_undo);
        while (!_evictedByCapture.empty()) {
            _bytes += _evictedByCapture.back().bytes();
            _undo.push_front(std::move(_evictedByCapture.back()));
            _evictedByCapture.pop_back();
        }
        for (auto& entry : _redoDroppedByCapture) {
            _bytes += entry.bytes();
            _redo.push_back(std::move(entry));
        }
        _redoDroppedByCapture.clear();
    }

    // Link the approval mask entry painted by the edit just captured.
    void setLastAutoApprovalId(std::uint64_t id)
    {
        if (!_undo.empty()) {
            _undo.back().autoApprovalId = id;
        }
    }

    // Remove and return the entry an undo or a redo should apply.
    [[nodiscard]] std::optional<Entry> takeUndo() { return take(_undo); }
    [[nodiscard]] std::optional<Entry> takeRedo() { return take(_redo); }

    // Store the state an undo (redo false) or a redo replaced on the opposite
    // stack, so the step can be reversed. An empty entry (every delta fell
    // outside the grid) leaves nothing to step back to and is dropped.
    void pushReplaced(bool redo, Entry replaced)
    {
        if (!replaced.isEmpty()) {
            push(redo ? _undo : _redo, std::move(replaced));
        }
    }

    // Drop everything that could be redone, as any change made outside the
    // history (growth, for example) must: a redo would write over it.
    void clearRedo()
    {
        forgetCaptureLeftovers();
        while (!_redo.empty()) {
            popBack(_redo);
        }
    }

    // Entries address the vertices of a grid of this size. A different size
    // drops the whole history, since after the grid grows or shrinks no entry
    // addresses the right vertices.
    void setGridSize(cv::Size size)
    {
        if (size != _gridSize) {
            clear();
            _gridSize = size;
        }
    }

    void clear()
    {
        _undo.clear();
        _redo.clear();
        forgetCaptureLeftovers();
        _bytes = 0;
    }

    [[nodiscard]] bool canUndo() const { return !_undo.empty(); }
    [[nodiscard]] bool canRedo() const { return !_redo.empty(); }
    [[nodiscard]] bool empty() const { return _undo.empty(); }
    [[nodiscard]] std::size_t size() const { return _undo.size(); }
    [[nodiscard]] std::size_t redoSize() const { return _redo.size(); }
    [[nodiscard]] std::size_t bytes() const { return _bytes; }
    [[nodiscard]] const UndoLimits& limits() const { return _limits; }

    // The entry that reverses applying entry to grid, read before applying
    // it: the current positions of its vertices inside grid, or a copy of grid
    // for a snapshot. The auto-approval id is not copied; the caller links it
    // once the paired approval step has succeeded.
    [[nodiscard]] static Entry inverseFor(const cv::Mat_<cv::Vec3f>& grid, const Entry& entry)
    {
        Entry inverse;
        if (entry.isDelta()) {
            inverse.deltas.reserve(entry.deltas.size());
            for (const auto& delta : entry.deltas) {
                if (insideGrid(grid, delta)) {
                    inverse.deltas.push_back({delta.row, delta.col, grid(delta.row, delta.col)});
                }
            }
        } else if (!entry.points.empty()) {
            inverse.points = grid.clone();
        }
        return inverse;
    }

    // Writes a delta entry's positions into grid, skipping vertices outside it.
    static void applyDeltas(cv::Mat_<cv::Vec3f>& grid, const Entry& entry)
    {
        for (const auto& delta : entry.deltas) {
            if (insideGrid(grid, delta)) {
                grid(delta.row, delta.col) = delta.previousWorld;
            }
        }
    }

    // One undo (redo false) or redo step on grid, the surface the entries
    // address. Takes the next entry and reads the entry that reverses it from
    // grid (inverseFor) before apply(entry, inverse) writes the entry into
    // grid. apply returns whether it applied; an entry that does not apply is
    // dropped, so it cannot block the steps behind it. When it applied and
    // has a paired auto-approval, stepApproval(id) steps that too and says
    // whether it could; only then is the reverse entry linked to it. The
    // reverse entry goes on the opposite stack, so the step can be reversed.
    // Returns whether a step was applied.
    template <typename Apply, typename StepApproval>
    bool step(bool redo, const cv::Mat_<cv::Vec3f>& grid, Apply&& apply, StepApproval&& stepApproval)
    {
        auto entry = redo ? takeRedo() : takeUndo();
        if (!entry) {
            return false;
        }
        Entry inverse = inverseFor(grid, *entry);
        if (!apply(static_cast<const Entry&>(*entry), static_cast<const Entry&>(inverse))) {
            return false;
        }
        if (entry->autoApprovalId != 0 && stepApproval(entry->autoApprovalId)) {
            inverse.autoApprovalId = entry->autoApprovalId;
        }
        pushReplaced(redo, std::move(inverse));
        return true;
    }

private:
    static bool insideGrid(const cv::Mat_<cv::Vec3f>& grid, const VertexDelta& delta)
    {
        return delta.row >= 0 && delta.row < grid.rows && delta.col >= 0 && delta.col < grid.cols;
    }

    void recordNewEdit(Entry entry)
    {
        forgetCaptureLeftovers();
        // Until the next change, keep what this capture drops, uncounted, in
        // case it is discarded.
        while (!_redo.empty()) {
            _bytes -= _redo.front().bytes();
            _redoDroppedByCapture.push_back(std::move(_redo.front()));
            _redo.pop_front();
        }
        _bytes += entry.bytes();
        _undo.push_back(std::move(entry));
        enforceLimits(_undo, &_evictedByCapture);
    }

    std::optional<Entry> take(std::deque<Entry>& stack)
    {
        forgetCaptureLeftovers();
        if (stack.empty()) {
            return std::nullopt;
        }
        Entry entry = std::move(stack.back());
        stack.pop_back();
        _bytes -= entry.bytes();
        return entry;
    }

    void push(std::deque<Entry>& stack, Entry entry)
    {
        forgetCaptureLeftovers();
        _bytes += entry.bytes();
        stack.push_back(std::move(entry));
        enforceLimits(stack, nullptr);
    }

    void popBack(std::deque<Entry>& stack)
    {
        _bytes -= stack.back().bytes();
        stack.pop_back();
    }

    void forgetCaptureLeftovers()
    {
        _evictedByCapture.clear();
        _redoDroppedByCapture.clear();
    }

    // Evict the oldest undo entries first, then the redo entries farthest
    // from the current state, but never the entry just pushed onto newest.
    // Evicted entries go to keep when it is given.
    void enforceLimits(const std::deque<Entry>& newest, std::deque<Entry>* keep)
    {
        auto evictable = [&newest](const std::deque<Entry>& stack) {
            return stack.size() > (&stack == &newest ? 1u : 0u);
        };
        while (_limits.exceeded(_undo.size() + _redo.size(), _bytes)) {
            std::deque<Entry>* stack = evictable(_undo) ? &_undo : evictable(_redo) ? &_redo : nullptr;
            if (!stack) {
                break;
            }
            _bytes -= stack->front().bytes();
            if (keep) {
                keep->push_back(std::move(stack->front()));
            }
            stack->pop_front();
        }
    }

    UndoLimits _limits;
    cv::Size _gridSize;
    std::size_t _bytes{0};
    std::deque<Entry> _undo;
    std::deque<Entry> _redo;
    // What the last capture evicted (oldest first) or dropped from redo, kept
    // only until the next change so discardLast() can restore it.
    std::deque<Entry> _evictedByCapture;
    std::deque<Entry> _redoDroppedByCapture;
};

} // namespace segmentation
