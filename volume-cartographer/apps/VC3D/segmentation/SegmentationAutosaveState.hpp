#pragma once

#include <cstdint>

namespace segmentation {

class AutosaveState
{
public:
    struct Ticket
    {
        std::uint64_t session{0};
        std::uint64_t save{0};
    };

    enum class Completion
    {
        Current,
        Stale
    };

    void beginSession()
    {
        ++_sessionToken;
        clearDeferred();
        _failureNotified = false;
    }

    void endSession()
    {
        ++_sessionToken;
        clearDeferred();
        _failureNotified = false;
    }

    [[nodiscard]] bool pending() const { return _pending; }
    [[nodiscard]] bool saveInProgress() const { return _saveInProgress; }
    [[nodiscard]] bool dirtyAfterSave() const { return _dirtyAfterSave; }
    [[nodiscard]] bool failureNotified() const { return _failureNotified; }

    void markPending()
    {
        _pending = true;
        _failureNotified = false;
    }

    void clearDeferred()
    {
        _pending = false;
        _dirtyAfterSave = false;
    }

    void setFailureNotified(bool notified)
    {
        _failureNotified = notified;
    }

    bool markDirtyIfSaving()
    {
        if (!_saveInProgress) {
            return false;
        }
        _dirtyAfterSave = true;
        return true;
    }

    Ticket startSave()
    {
        _pending = false;
        _saveInProgress = true;
        _dirtyAfterSave = false;
        _activeSaveToken = ++_nextSaveToken;
        return Ticket{_sessionToken, _activeSaveToken};
    }

    Completion completeSuccess(Ticket ticket)
    {
        const Completion completion = complete(ticket);
        if (completion == Completion::Current) {
            _failureNotified = false;
        }
        return completion;
    }

    Completion completeFailure(Ticket ticket, bool retry)
    {
        const Completion completion = complete(ticket);
        if (completion == Completion::Current) {
            _pending = retry;
        }
        return completion;
    }

    bool consumeDirtyAfterSave()
    {
        if (!_dirtyAfterSave) {
            return false;
        }
        _dirtyAfterSave = false;
        _pending = true;
        return true;
    }

private:
    [[nodiscard]] bool isCurrent(Ticket ticket) const
    {
        return ticket.session == _sessionToken && ticket.save == _activeSaveToken;
    }

    Completion complete(Ticket ticket)
    {
        const bool current = isCurrent(ticket);
        if (ticket.save == _activeSaveToken) {
            _saveInProgress = false;
            _activeSaveToken = 0;
        }
        return current ? Completion::Current : Completion::Stale;
    }

    std::uint64_t _sessionToken{0};
    std::uint64_t _nextSaveToken{0};
    std::uint64_t _activeSaveToken{0};
    bool _pending{false};
    bool _saveInProgress{false};
    bool _dirtyAfterSave{false};
    bool _failureNotified{false};
};

} // namespace segmentation
