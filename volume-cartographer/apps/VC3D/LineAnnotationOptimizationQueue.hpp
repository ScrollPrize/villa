#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vc3d::line_annotation {

// The line-annotation solve lifecycle as a pure decision object (the pattern
// FiberMapRebuildQueue set for the Fiber Map), so every transition the
// controller's asynchronous re-optimization depends on is unit-testable
// without a session or a widget.
//
// One solve runs per session at a time. Control-point edits arriving while
// one is in flight do not refuse: the handler applies its cheap geometric
// update to the session immediately, records the union of dirty spans here,
// and the running solve - now describing a session that no longer exists -
// is refused publication through the epoch and superseded by one solve over
// the union when it finishes.
//
// The epoch is the publication token: it advances on every session-geometry
// mutation (noteSessionMutated), and a solve may only publish into the epoch
// it was started in. Dirty spans are identified by their owning control
// point's index; an edit that renumbers control points remaps the pending
// set through its old-to-new index map before adding its own spans.
class OptimizationCoalescingQueue {
public:
    enum class State { Idle, Running, ShuttingDown };

    struct PendingSolve {
        bool requested = false;
        // Dirty tracking was lost or an edit demanded a whole-line pass:
        // re-attempt every span instead of the dirty set.
        bool fullLine = false;
        // Union of dirty span indices (owning control point's index) in the
        // current control-point numbering; sorted and unique. Meaningful
        // only when !fullLine.
        std::vector<std::size_t> dirtySegments;
    };

    [[nodiscard]] State state() const { return _state; }
    [[nodiscard]] std::uint64_t epoch() const { return _epoch; }
    [[nodiscard]] bool hasPending() const { return _pending.requested; }

    // The session's geometry (line points / control points) was mutated. Any
    // in-flight solve no longer describes the session; its publication is
    // refused by the epoch mismatch.
    void noteSessionMutated() { ++_epoch; }

    // Record a solve request. Coalesces into the pending slot regardless of
    // state; the holder decides when to dispatch (debounce while Idle, the
    // finish epilogue while Running). fullLine wins over any dirty set and
    // is never downgraded.
    void addPending(const std::vector<std::size_t>& dirtySegments, bool fullLine)
    {
        if (_state == State::ShuttingDown) {
            return;
        }
        _pending.requested = true;
        if (fullLine || _pending.fullLine) {
            _pending.fullLine = true;
            _pending.dirtySegments.clear();
            return;
        }
        for (const std::size_t segment : dirtySegments) {
            _pending.dirtySegments.push_back(segment);
        }
        normalizePendingDirty();
    }

    // Carry a set of dirty span indices across a control-point renumbering.
    // Span s (owned by old control s, ending at old control s+1) maps onto
    // the new spans owned by controls [oldToNew[s], oldToNew[s+1]) - one
    // span normally, several when a control was inserted inside it, none
    // when its endpoints collapsed into one control (the collapsing edit
    // dirties the replacement's own spans itself). Returns false when a
    // span's right endpoint is not covered by the map - dirty tracking for
    // it is lost and the caller must fall back to a whole-line pass.
    [[nodiscard]] static bool remapDirtySpans(
        std::vector<std::size_t>& dirtySegments,
        const std::vector<std::size_t>& oldToNewControlIndices,
        std::size_t newSpanCount)
    {
        std::vector<std::size_t> remapped;
        remapped.reserve(dirtySegments.size());
        for (const std::size_t segment : dirtySegments) {
            if (segment + 1 >= oldToNewControlIndices.size()) {
                return false;
            }
            const std::size_t newBegin = oldToNewControlIndices[segment];
            const std::size_t newEnd = oldToNewControlIndices[segment + 1];
            for (std::size_t owner = newBegin; owner < newEnd; ++owner) {
                if (owner < newSpanCount) {
                    remapped.push_back(owner);
                }
            }
        }
        std::sort(remapped.begin(), remapped.end());
        remapped.erase(std::unique(remapped.begin(), remapped.end()),
                       remapped.end());
        dirtySegments = std::move(remapped);
        return true;
    }

    // An edit renumbered control points: carry the pending dirty spans into
    // the new numbering (see remapDirtySpans).
    void remapPendingDirty(const std::vector<std::size_t>& oldToNewControlIndices,
                           std::size_t newSpanCount)
    {
        if (!_pending.requested || _pending.fullLine) {
            return;
        }
        if (!remapDirtySpans(_pending.dirtySegments,
                             oldToNewControlIndices,
                             newSpanCount)) {
            _pending.fullLine = true;
            _pending.dirtySegments.clear();
            return;
        }
        if (_pending.dirtySegments.empty()) {
            _pending.requested = false;
        }
    }

    // Take the pending request for dispatch (Idle only; the finish epilogue
    // uses finishSolve instead). Exactly-once: the slot is cleared.
    [[nodiscard]] PendingSolve takePending()
    {
        PendingSolve dispatch = std::move(_pending);
        _pending = PendingSolve{};
        return dispatch;
    }

    // A solve is being launched. Returns false when one is already running
    // or the queue is shutting down; on success hands out the epoch to stamp
    // on the solve.
    [[nodiscard]] bool beginSolve(std::uint64_t& solveEpoch)
    {
        if (_state != State::Idle) {
            return false;
        }
        _state = State::Running;
        solveEpoch = _epoch;
        return true;
    }

    // A solve may publish only into its own epoch, and only while the queue
    // still expects it.
    [[nodiscard]] bool mayPublish(std::uint64_t solveEpoch) const
    {
        return _state == State::Running && solveEpoch == _epoch;
    }

    // The solve finished (published or discarded); return to Idle and
    // surrender the pending request for the holder to dispatch.
    [[nodiscard]] PendingSolve finishSolve()
    {
        if (_state == State::ShuttingDown) {
            _pending = PendingSolve{};
            return PendingSolve{};
        }
        _state = State::Idle;
        return takePending();
    }

    // Teardown: no new solves, pending dropped, in-flight publication
    // refused via the epoch.
    void shutdown()
    {
        _state = State::ShuttingDown;
        _pending = PendingSolve{};
        ++_epoch;
    }

private:
    void normalizePendingDirty()
    {
        std::sort(_pending.dirtySegments.begin(), _pending.dirtySegments.end());
        _pending.dirtySegments.erase(
            std::unique(_pending.dirtySegments.begin(),
                        _pending.dirtySegments.end()),
            _pending.dirtySegments.end());
    }

    State _state = State::Idle;
    PendingSolve _pending;
    std::uint64_t _epoch = 0;
};

}  // namespace vc3d::line_annotation
