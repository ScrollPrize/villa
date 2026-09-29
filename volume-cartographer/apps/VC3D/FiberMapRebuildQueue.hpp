#pragma once

#include <cstdint>

namespace vc3d::fiber_map {

// The Fiber Map's rebuild lifecycle as a pure decision object, so every
// transition the asynchronous rebuild depends on is unit-testable without a
// widget (the pattern FiberMapStaleness.hpp set for the staleness decision).
//
// One build runs at a time. Requests arriving while one is in flight
// coalesce into a single pending request, full-rebuild winning over update
// (a full rebuild subsumes an update; the reverse discards the user's
// explicit escape-hatch intent). The pending request survives every terminal
// outcome of the running build - success, discard, error - and is dispatched
// by exactly one epilogue. Shutdown drops pending work and refuses new
// starts.
//
// The epoch is the publication token: it advances whenever the world a
// running build was started in stops existing (package switch, explicit
// clear, memoization policy change, shutdown), and a build may only publish
// into the epoch it was started in. Dependency-level validation (fiber /
// umbilicus / frame watermarks) is the holder's business on top of this.
class FiberMapRebuildQueue {
public:
    enum class State { Idle, Running, Applying, ShuttingDown };
    enum class Pending { None, Update, Full };
    enum class Request { Start, Coalesced, Refused };

    [[nodiscard]] State state() const { return _state; }
    [[nodiscard]] Pending pending() const { return _pending; }
    // Whether every request folded into the pending slot was automatic
    // (armed by a staleness gate rather than asked for): the holder may
    // drop such a pending request when the world it was armed for has
    // moved on, but never one the user asked for, or one that retries the
    // build they asked for. False when nothing is pending.
    [[nodiscard]] bool pendingAutomatic() const
    {
        return _pending != Pending::None && _pendingAutomatic;
    }
    [[nodiscard]] std::uint64_t epoch() const { return _epoch; }

    // A request to rebuild. Start means the caller must launch the build
    // now (and is handed the epoch to stamp on it); Coalesced means it was
    // folded into the pending slot; Refused means shutdown. `automatic`
    // says who asked (see pendingAutomatic()).
    [[nodiscard]] Request request(bool full, bool automatic = false)
    {
        if (_state == State::ShuttingDown) {
            return Request::Refused;
        }
        if (_state != State::Idle) {
            if (full || _pending == Pending::None) {
                _pending = full ? Pending::Full
                                : (_pending == Pending::Full ? Pending::Full
                                                             : Pending::Update);
            }
            _pendingAutomatic = _pendingAutomatic && automatic;
            return Request::Coalesced;
        }
        _state = State::Running;
        return Request::Start;
    }

    // The world a running build started in stopped existing: package switch,
    // explicit clear, memoization policy change. Any in-flight build's
    // publication is refused by the epoch mismatch.
    void invalidate() { ++_epoch; }

    // A build may publish only into its own epoch, and only while the queue
    // still expects it (a shutdown mid-flight refuses too).
    [[nodiscard]] bool mayPublish(std::uint64_t buildEpoch) const
    {
        return _state == State::Running && buildEpoch == _epoch;
    }

    // The worker finished (any outcome); the holder is about to run the
    // apply/cleanup epilogue.
    void beginApply()
    {
        if (_state == State::Running) {
            _state = State::Applying;
        }
    }

    // The epilogue's final act: return to Idle and surrender the pending
    // request (if any) for the holder to dispatch. Exactly-once semantics:
    // the pending slot is cleared here and nowhere else.
    [[nodiscard]] Pending finishApply()
    {
        if (_state == State::ShuttingDown) {
            clearPending();
            return Pending::None;
        }
        _state = State::Idle;
        const Pending dispatch = _pending;
        clearPending();
        return dispatch;
    }

    // Teardown: no new starts, pending dropped, in-flight publication
    // refused via the epoch.
    void shutdown()
    {
        _state = State::ShuttingDown;
        clearPending();
        ++_epoch;
    }

private:
    void clearPending()
    {
        _pending = Pending::None;
        _pendingAutomatic = true;
    }

    State _state = State::Idle;
    Pending _pending = Pending::None;
    bool _pendingAutomatic = true;
    std::uint64_t _epoch = 0;
};

} // namespace vc3d::fiber_map
